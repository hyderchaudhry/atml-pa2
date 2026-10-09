from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)

from task2_ppo.ppo import (
    compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # PEFT promotes LoRA weights, but the saved critic head can remain FP16.
    # Keep optimizer parameters/gradients FP32 so GradScaler can unscale both optimizers.
    for model in (policy, value_model):
        for parameter in trainable_parameters(model):
            parameter.data = parameter.data.float()

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    cfg = load_yaml(config_path)
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    result_dir = repo_path(cfg["results_dir"]) / run_name
    if out.exists() or result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite PPO run: {out} or {result_dir}")
    if min(int(cfg[k]) for k in ("updates", "prompts_per_update", "ppo_epochs")) < 1:
        raise ValueError("PPO budgets must be positive")
    bundle = prepare_ppo_continuation(config_path)
    bundle["cfg"] = cfg
    rows = bundle["prompt_rows"]
    if not rows:
        raise ValueError("The released training prompt pool is empty")
    schedule = [
        [((u * int(cfg["prompts_per_update"])) + j) % len(rows)
         for j in range(int(cfg["prompts_per_update"]))]
        for u in range(int(cfg["updates"]))
    ]
    save_json(result_dir / "training_manifest.json", {
        **run_metadata(cfg), "run_name": run_name, "output": str(out),
        "prompt_schedule": [[prompt_identity(rows[i], i) for i in batch] for batch in schedule],
        "prompt_order": "released file order, cycling if necessary",
        "optimizer_state": "fresh AdamW; release supplies policy/value weights only",
        "metric_timing": "rollout metrics before update; losses/clip/gradient norms averaged over PPO epochs",
        "gradient_norm_definition": "policy L2 norm before max_grad_norm clipping; value norm saved separately",
        "runtime_scope": "rollout collection and optimization; excludes model loading, saving and held-out evaluation",
    })
    policy, critic = bundle["policy"], bundle["value_model"]
    # eval disables dropout without disabling gradients: old/new ratios must agree before a step.
    policy.eval()
    critic.eval()
    policy_params, value_params = trainable_parameters(policy), trainable_parameters(critic)
    device = next(policy.parameters()).device
    cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and cfg["dtype"] in {"float16", "fp16"})
    set_seed(int(cfg["seed"]))
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = wall_timer()
    trajectory = []
    for update, indices in enumerate(schedule, 1):
        selected = [rows[i] for i in indices]
        prompts = [prompt_messages(row) for row in selected]
        rollout = generate_rollout(policy, bundle["tokenizer"], prompts, cfg)
        mask = rollout["response_mask"]
        with torch.no_grad():
            old_logp = rollout_logprobs(policy, rollout)
            with reference_mode(policy):
                ref_logp = rollout_logprobs(policy, rollout)
            raw_reward = score_reward_pairs(
                bundle["reward_model"], bundle["reward_tokenizer"], prompts,
                rollout["responses"], max_length=int(cfg["reward_max_length"]),
            ).to(device)
            terminal_reward = raw_reward - float(cfg["missing_eos_penalty"]) * torch.tensor(
                [not eos for eos in rollout["terminated_with_eos"]], device=device,
            )
            values = rollout_values(critic, rollout)
            rewards = shaped_rewards(terminal_reward, old_logp, ref_logp, mask, cfg["kl_beta"])
            advantages, returns = compute_gae(
                rewards, values, mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]),
            )
            advantages = normalize_advantages(advantages, mask)
        epochs = []
        for epoch in range(int(cfg["ppo_epochs"])):
            policy_opt, value_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
            policy_opt.zero_grad(set_to_none=True)
            value_opt.zero_grad(set_to_none=True)
            new_logp = rollout_logprobs(policy, rollout)
            policy_loss, _, clip_fraction = ppo_policy_loss(
                new_logp, old_logp, advantages, mask, eps=float(cfg["clip_epsilon"]),
            )
            scaler.scale(policy_loss).backward()
            scaler.unscale_(policy_opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_params, float(cfg["max_grad_norm"]), error_if_nonfinite=not scaler.is_enabled(),
            )
            scaler.step(policy_opt)
            # Separate backward passes avoid retaining both large model graphs at once.
            value_loss = value_mse_loss(rollout_values(critic, rollout), returns, mask)
            scaler.scale(float(cfg["value_coef"]) * value_loss).backward()
            scaler.unscale_(value_opt)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                value_params, float(cfg["max_grad_norm"]), error_if_nonfinite=not scaler.is_enabled(),
            )
            scaler.step(value_opt)
            scaler.update()
            epochs.append({
                "epoch": epoch + 1, "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()), "clip_fraction": float(clip_fraction),
                "gradient_norm": float(grad_norm) if torch.isfinite(grad_norm) else None,
                "value_gradient_norm": float(value_grad_norm) if torch.isfinite(value_grad_norm) else None,
                "policy_step_skipped": not bool(torch.isfinite(grad_norm)),
                "value_step_skipped": not bool(torch.isfinite(value_grad_norm)),
            })
        record = {
            "update": update, "learned_reward": float(raw_reward.mean()),
            "effective_terminal_reward": float(terminal_reward.mean()),
            "kl": float(sampled_kl(old_logp, ref_logp, mask)),
            "entropy": float(sample_entropy(old_logp, mask)),
            "response_length": mean_response_length(mask),
            "response_length_std": float(mask.sum(-1).float().std(unbiased=False)),
            "response_lengths": rollout["response_lengths"],
            "generated_tokens": int(mask.sum()),
            "prompt_ids": [prompt_identity(row, i) for row, i in zip(selected, indices)],
            "epochs": epochs,
            **{k: mean_finite(e[k] for e in epochs) for k in epochs[0] if k != "epoch"},
        }
        append_jsonl(result_dir / "trajectory.jsonl", record)
        trajectory.append(record)
    if cuda:
        torch.cuda.synchronize(device)
    runtime = elapsed()
    summary = {
        **run_metadata(cfg), "run_name": run_name, "output": str(out),
        "wall_clock_seconds": runtime,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
        "memory_device": str(device), "completed_updates": len(trajectory),
        "generated_tokens": sum(row["generated_tokens"] for row in trajectory),
        "mean_gradient_norm": mean_finite(e["gradient_norm"] for row in trajectory for e in row["epochs"]),
        "max_gradient_norm": max((e["gradient_norm"] for row in trajectory for e in row["epochs"]
                                  if e["gradient_norm"] is not None), default=None),
        "skipped_policy_steps": sum(e["policy_step_skipped"] for row in trajectory for e in row["epochs"]),
        "skipped_value_steps": sum(e["value_step_skipped"] for row in trajectory for e in row["epochs"]),
        "last_update": trajectory[-1],
    }
    policy.save_pretrained(out)
    bundle["tokenizer"].save_pretrained(out)
    save_json(result_dir / "training_summary.json", summary)
    return summary


def mean_finite(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def prompt_identity(row, index):
    return {"row_index": index, **{k: row[k] for k in ("prompt_id", "source_index", "source_split") if k in row}}


def run_metadata(cfg):
    return {
        "seed": int(cfg["seed"]), "config": cfg,
        "policy_checkpoint": cfg["paths"]["ppo_midpoint_policy"],
        "value_checkpoint": cfg["paths"]["ppo_midpoint_value"],
        "reference": {"base_model": cfg["base_model"], "adapter": "disabled"},
        "reward_model": cfg["reward_model"], "updates": int(cfg["updates"]),
        "epsilon": float(cfg["clip_epsilon"]), "beta_kl": float(cfg["kl_beta"]),
        "metric_conventions": {
            "kl": "common.metrics.sampled_kl: valid-token mean of policy minus reference log-probability",
            "entropy": "common.metrics.sample_entropy: valid-token mean negative sampled log-probability",
            "logprobs": "unwarped model log-probabilities, as in common.generation.response_token_logprobs",
            "response_length": "generated tokens including terminal EOS, excluding padding; population std",
        },
    }


def generate_rollout(policy, tokenizer, prompts, cfg, evaluation=False):
    rollout = batch_generate(
        policy, tokenizer, prompts, max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["eval_max_response_length"] if evaluation else cfg["max_response_length"]),
        **cfg["generation"],
    )
    # generate() returns inference tensors; training embeddings need normal tensors to save for backward.
    for key, value in rollout.items():
        if torch.is_tensor(value):
            rollout[key] = value.clone()
    rollout["attention_mask"][:, rollout["prompt_width"]:] = rollout["response_mask"].long()
    return rollout


def rollout_logprobs(policy, rollout):
    logp, _ = response_token_logprobs(
        policy, rollout["sequences"], rollout["attention_mask"],
        rollout["prompt_width"], rollout["response_ids"],
    )
    return logp


def rollout_values(critic, rollout):
    # V(s_t) sees the prefix BEFORE the sampled action, just like the policy logits.
    parameter = next(critic.parameters())
    # Autocast reconciles the low-precision backbone with the FP32 trainable scalar head.
    with torch.autocast(parameter.device.type, dtype=parameter.dtype,
                        enabled=parameter.dtype in (torch.float16, torch.bfloat16)):
        values = token_values(critic, rollout["sequences"], rollout["attention_mask"])
    start = rollout["prompt_width"] - 1
    return values[:, start:start + rollout["response_ids"].shape[1]].float()


def run_forks(config_path, parameter, values, study):
    """Fresh processes guarantee identical checkpoint restoration and release model memory."""
    cfg = load_yaml(config_path)
    # The cached diagnostic may just have run in this process; release its CUDA allocator cache.
    clear_gpu()
    results = []
    for value in values:
        name = f"{study}_{float(value):.2f}"
        output = str(Path(cfg["output"]).parent / name)
        subprocess.run([
            sys.executable, "-m", "task2_ppo.continue_train", "--config", config_path,
            "--output", output, "--run-name", name, "--updates", str(cfg["fork_updates"]),
            parameter, str(value),
        ], cwd=repo_path("."), check=True)
        subprocess.run([
            sys.executable, "-m", "task2_ppo.evaluate", "--config", config_path,
            "--adapter", output, "--name", name,
        ], cwd=repo_path("."), check=True)
        result_dir = repo_path(cfg["results_dir"]) / name
        summary = load_json(result_dir / "training_summary.json")
        results.append({
            "condition": value, "run_name": name, "fork_updates": cfg["fork_updates"],
            "epsilon": summary["epsilon"], "beta_kl": summary["beta_kl"],
            "held_out": load_json(result_dir / "metrics.json"),
            "mean_gradient_norm": summary["mean_gradient_norm"],
            "max_gradient_norm": summary["max_gradient_norm"],
            "skipped_policy_steps": summary["skipped_policy_steps"],
            "skipped_value_steps": summary["skipped_value_steps"],
            "training_manifest": str(result_dir / "training_manifest.json"),
            "trajectory": str(result_dir / "trajectory.jsonl"),
            "examples": str(result_dir / "generations.jsonl"),
        })
        save_json(repo_path(cfg["results_dir"]) / study / "comparison.json", {
            "parameter": parameter, "conditions": results,
            "stability_statistic": "policy L2 gradient norm before clipping, over all optimization epochs",
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
