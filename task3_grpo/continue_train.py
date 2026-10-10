from __future__ import annotations

import argparse
import torch

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import (
    ADVANTAGE_EPS, group_relative_advantages, grpo_policy_loss,
    mask_truncated_sequences, normalization_statistics,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    print(f"Loading tokenizer: {cfg['base_model']}", flush=True)
    tokenizer = load_tokenizer(cfg["base_model"])
    print(f"Loading GRPO midpoint: {cfg['paths']['grpo_midpoint_policy']}", flush=True)
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    print(f"Loading reward model: {cfg['reward_model']}", flush=True)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    print(f"Loading training prompts: {cfg['paths']['rl_prompt_train']}", flush=True)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    for parameter in trainable_parameters(policy):
        parameter.data = parameter.data.float()
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None,
             loss_type: str = "grpo", run_name: str = "standard",
             save_rollouts: bool = False, rollout_source: str | None = None):
    cfg = load_yaml(config_path)
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    result_dir = repo_path(cfg["results_dir"]) / run_name
    if out.exists() or result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite GRPO run: {out} or {result_dir}")
    if loss_type not in {"grpo", "dr_grpo"}:
        raise ValueError("Unknown loss type")
    if min(int(cfg[k]) for k in ("updates", "prompts_per_update", "policy_epochs",
                                "max_completion_length")) < 1 or int(cfg["num_generations"]) < 2:
        raise ValueError("GRPO budgets must be positive and K must be at least two")
    print(f"GRPO run {run_name} | loss_type={loss_type} | updates={cfg['updates']} | "
          f"K={cfg['num_generations']} | cap={cfg['max_completion_length']}", flush=True)
    bundle = prepare_grpo_continuation(config_path)
    rows, policy = bundle["prompt_rows"], bundle["policy"]
    if not rows:
        raise ValueError("The released training prompt pool is empty")
    # Dropout off for consistent old/new likelihoods; eval still permits gradients.
    policy.eval()
    device = next(policy.parameters()).device
    parameters = trainable_parameters(policy)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and
                                 cfg["dtype"] in {"float16", "fp16"})
    schedule = [[(u * int(cfg["prompts_per_update"]) + j) % len(rows)
                 for j in range(int(cfg["prompts_per_update"]))]
                for u in range(int(cfg["updates"]))]
    metadata = run_metadata(cfg, loss_type)
    save_json(result_dir / "training_manifest.json", {
        **metadata, "output": str(out), "run_name": run_name,
        "prompt_schedule": [[prompt_identity(rows[i], i) for i in indices] for indices in schedule],
        "prompt_order": "released file order, cycling if needed",
        "optimizer_state": "fresh AdamW; supplied checkpoint has adapter weights only",
        "rollout_source": rollout_source,
        "sampling": "canonical fork's shared behavior rollouts" if rollout_source else "current policy",
        "saved_rollouts": str(out / "rollouts") if save_rollouts else None,
        "runtime_scope": "rollout collection/replay and optimization; excludes model loading, saving, evaluation",
        "gradient_norm": "parameter L2 norm before clipping",
        "metric_timing": "current-policy KL/entropy before the final optimization epoch; sampled rewards/lengths before update; losses and gradient norms averaged over epochs",
        "length_conditioning": "short <= max_completion_length/2, long > max_completion_length/2; masked responses retained with zero gradient",
    })
    set_seed(int(cfg["seed"]))
    print(f"Beginning continuation on {device}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = wall_timer()
    trajectory = []
    for update, indices in enumerate(schedule, 1):
        selected = [rows[i] for i in indices]
        k = int(cfg["num_generations"])
        prompts = [prompt_messages(row) for row in selected for _ in range(k)]
        identities = [prompt_identity(row, i) for row, i in zip(selected, indices) for _ in range(k)]
        if rollout_source:
            print(f"Update {update}/{cfg['updates']} | loading shared rollout", flush=True)
            cached = torch.load(repo_path(rollout_source) / f"update_{update:03d}.pt",
                                map_location=device, weights_only=True)
            if cached["config"] != cfg or cached["prompt_ids"] != identities:
                raise ValueError("Shared rollout configuration/prompt schedule does not match this fork")
            rollout = cached["rollout"]
            rewards, old_logp, ref_logp = (cached[key] for key in ("rewards", "old_logp", "ref_logp"))
        else:
            print(f"Update {update}/{cfg['updates']} | generating {len(prompts)} completions", flush=True)
            rollout = generate_rollout(policy, bundle["tokenizer"], prompts, cfg)
            print(f"Update {update}/{cfg['updates']} | computing policy/reference log probabilities and rewards", flush=True)
            with torch.no_grad():
                old_logp = rollout_logprobs(policy, rollout)
                with reference_mode(policy):
                    ref_logp = rollout_logprobs(policy, rollout)
                rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"],
                                             prompts, rollout["responses"]).to(device)
            if save_rollouts:
                cache_dir = out / "rollouts"
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"config": cfg, "prompt_ids": identities,
                            "rollout": {key: value.cpu() if torch.is_tensor(value) else value
                                        for key, value in rollout.items()},
                            "rewards": rewards.cpu(), "old_logp": old_logp.cpu(),
                            "ref_logp": ref_logp.cpu()}, cache_dir / f"update_{update:03d}.pt")
        group_ids = torch.arange(len(selected), device=device).repeat_interleave(k)
        advantages = group_relative_advantages(rewards, group_ids).detach()
        raw_mask = rollout["response_mask"]
        mask = mask_truncated_sequences(raw_mask, rollout["truncated"]) if cfg["mask_truncated_completions"] else raw_mask
        reward_stds = rewards.reshape(-1, k).std(-1, unbiased=False)
        epochs = []
        completion_records = []
        for epoch in range(int(cfg["policy_epochs"])):
            print(f"Update {update}/{cfg['updates']} | optimizer epoch {epoch + 1}/{cfg['policy_epochs']}", flush=True)
            bundle["optimizer"].zero_grad(set_to_none=True)
            new_logp = rollout_logprobs(policy, rollout)
            loss, diagnostics = grpo_policy_loss(
                new_logp, old_logp, advantages, mask, ref_logp,
                float(cfg["clip_epsilon"]), float(cfg["kl_beta"]),
                loss_type, int(cfg["max_completion_length"]),
            )
            allocation = normalization_statistics(new_logp, old_logp, advantages, mask,
                                                  float(cfg["clip_epsilon"]), loss_type,
                                                  int(cfg["max_completion_length"]))
            for i in range(len(prompts)):
                completion_records.append({
                    **identities[i], "update": update, "epoch": epoch + 1,
                    "generation_index": i % k, "loss_type": loss_type,
                    "prompt": prompts[i], "response": rollout["responses"][i],
                    "response_token_ids": rollout["response_ids"][i, raw_mask[i].bool()].tolist(),
                    "learned_reward": float(rewards[i]), "advantage": float(advantages[i]),
                    "group_reward_std": float(reward_stds[i // k]),
                    "response_length": rollout["response_lengths"][i],
                    "length_bin": "short" if rollout["response_lengths"][i] <= int(cfg["max_completion_length"]) / 2 else "long",
                    "valid_training_tokens": int(mask[i].sum()),
                    "terminated_with_eos": rollout["terminated_with_eos"][i],
                    "truncated": rollout["truncated"][i],
                    **{key: values[i] for key, values in allocation.items()},
                })
            scaler.scale(loss).backward()
            scaler.unscale_(bundle["optimizer"])
            norm = torch.nn.utils.clip_grad_norm_(parameters, float(cfg["max_grad_norm"]),
                                                 error_if_nonfinite=not scaler.is_enabled())
            # Fully masked batches must not receive AdamW weight decay or momentum updates.
            skipped = not bool(mask.sum()) or not bool(torch.isfinite(norm))
            if bool(mask.sum()):
                scaler.step(bundle["optimizer"])
            scaler.update()
            epochs.append({"epoch": epoch + 1, "loss": float(loss.detach()),
                           **{key: float(value) for key, value in diagnostics.items()},
                           "gradient_norm": float(norm) if torch.isfinite(norm) else None,
                           "step_skipped": skipped})
        for record in completion_records:
            append_jsonl(result_dir / "training_completions.jsonl", record)
        record = {
            "update": update, "learned_reward": float(rewards.mean()),
            "kl": float(sampled_kl(new_logp.detach(), ref_logp, raw_mask)),
            "entropy": float(sample_entropy(new_logp.detach(), raw_mask)),
            "group_reward_std": float(reward_stds.mean()),
            "uninformative_group_fraction": float((reward_stds <= ADVANTAGE_EPS).float().mean()),
            "response_length": float(raw_mask.sum(-1).mean()),
            "response_length_std": float(raw_mask.sum(-1).std(unbiased=False)),
            "generated_tokens": int(raw_mask.sum()), "valid_training_tokens": int(mask.sum()),
            "truncation_rate": sum(rollout["truncated"]) / len(prompts),
            "epochs": epochs,
            **{key: mean_finite(e[key] for e in epochs) for key in epochs[0] if key != "epoch"},
        }
        append_jsonl(result_dir / "trajectory.jsonl", record)
        trajectory.append(record)
        print(f"Update {update}/{cfg['updates']} | reward={record['learned_reward']:.4f} | "
              f"KL={record['kl']:.4f} | loss={record['loss']:.4f} | "
              f"group_std={record['group_reward_std']:.4f} | length={record['response_length']:.1f} | "
              f"elapsed={elapsed():.1f}s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = {
        **metadata, "run_name": run_name, "output": str(out),
        "completed_updates": len(trajectory), "wall_clock_seconds": elapsed(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "memory_device": str(device), "generated_tokens": sum(r["generated_tokens"] for r in trajectory),
        "valid_training_tokens": sum(r["valid_training_tokens"] for r in trajectory),
        "mean_gradient_norm": mean_finite(e["gradient_norm"] for r in trajectory for e in r["epochs"]),
        "skipped_steps": sum(e["step_skipped"] for r in trajectory for e in r["epochs"]),
        "last_update": trajectory[-1],
    }
    print(f"Saving adapter: {out}", flush=True)
    policy.save_pretrained(out)
    bundle["tokenizer"].save_pretrained(out)
    save_json(result_dir / "training_summary.json", summary)
    print(f"Completed {run_name} | saved evidence: {result_dir}", flush=True)
    return summary


def mean_finite(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def prompt_identity(row, index):
    return {"row_index": index, **{key: row[key] for key in ("prompt_id", "source_index", "source_split") if key in row}}


def run_metadata(cfg, loss_type="grpo"):
    return {
        "config": cfg, "seed": int(cfg["seed"]), "loss_type": loss_type,
        "policy_checkpoint": cfg["paths"]["grpo_midpoint_policy"],
        "reference": {"base_model": cfg["base_model"], "adapter": "disabled"},
        "metric_conventions": {
            "kl": "common.metrics.sampled_kl; valid-token mean policy minus reference log probability",
            "entropy": "common.metrics.sample_entropy; valid-token mean negative sampled log probability",
            "response_length": "generated tokens including EOS, excluding padding; population std",
            "group_reward_std": "population standard deviation; informative if > 1e-6",
            "advantage": "within-prompt (reward - mean) / (population std + 1e-6); all K rewards before truncation masking",
            "kl_loss": "released nonnegative sampled KL estimator; unchanged across normalizations",
            "reward_max_length": "released common.generation.score_reward_pairs default: 1024",
            "logp_gradient": "policy surrogate derivative w.r.t. sampled-token log probabilities; KL excluded",
        },
    }


def generate_rollout(policy, tokenizer, prompts, cfg):
    rollout = batch_generate(policy, tokenizer, prompts,
                             max_prompt_length=int(cfg["max_prompt_length"]),
                             max_new_tokens=int(cfg["max_completion_length"]), **cfg["generation"])
    # Normal tensors are needed for training; generation returns inference tensors.
    for key, value in rollout.items():
        if torch.is_tensor(value):
            rollout[key] = value.clone()
    rollout["attention_mask"][:, rollout["prompt_width"]:] = rollout["response_mask"].long()
    return rollout


def rollout_logprobs(policy, rollout):
    logp, _ = response_token_logprobs(policy, rollout["sequences"], rollout["attention_mask"],
                                    rollout["prompt_width"], rollout["response_ids"])
    return logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--save-rollouts", action="store_true", help="Save shared fork samples in the ignored adapter output")
    ap.add_argument("--rollout-source", help="Replay the canonical fork's identical samples for exact token matching")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name,
             args.save_rollouts, args.rollout_source)


if __name__ == "__main__":
    main()
