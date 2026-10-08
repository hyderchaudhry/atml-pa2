from __future__ import annotations

import argparse
import math
import time
from itertools import islice

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (encode_prompt_response, load_yaml, pad_batch,
                         preference_responses, prompt_messages_from_preference,
                         read_jsonl)
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss
from task1_dpo.runtime import (assert_frozen_reference, dataset_metadata, move_batch,
                              output_directory, pair_logprobs, provenance, result_directory)


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            # The supplied encoder keeps the common prompt and masks it to zero.
            # The supplied log-probability helper shifts labels/masks together and
            # SUMS response-token scores (including EOS); no length normalization.
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length,
                                                  overflow_prompt_tokens=max_length // 2))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length,
                                                    overflow_prompt_tokens=max_length // 2))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None,
                    beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        if not 0 < int(max_examples) <= len(rows):
            raise ValueError("max_examples must be positive and no larger than the fixed file")
        rows = rows[:int(max_examples)]
    if not rows:
        raise ValueError("Empty DPO training set")
    run_beta = float(cfg["beta"] if beta is None else beta)
    if run_beta <= 0:
        raise ValueError("beta must be positive")

    tokenizer = load_tokenizer(cfg["base_model"])
    # Every call starts from the public base + a freshly seeded LoRA, never from
    # an earlier adapter. Optimizer and DataLoader RNG are also new for each fork.
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    assert_frozen_reference(model)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    loader = DataLoader(rows, batch_size=int(cfg["batch_size"]), shuffle=True,
                        generator=generator,
                        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])))
    optimizer = AdamW(trainable_parameters(model), lr=float(cfg["learning_rate"]),
                      weight_decay=float(cfg.get("weight_decay", 0.0)))
    return {"cfg": cfg, "rows": rows, "dataset_path": path, "tokenizer": tokenizer,
            "model": model, "loader": loader, "optimizer": optimizer, "beta": run_beta}


def run_training(config_path: str, run_name: str, dataset_path: str | None = None,
                 output_path: str | None = None, beta: float | None = None,
                 max_examples: int | None = None):
    cfg = load_yaml(config_path)
    results = result_directory(cfg, run_name)
    output = output_directory(cfg, run_name, output_path)
    # Refuse to mix logs or overwrite an earlier experiment. Choose a fresh name
    # for intentional reruns; all evaluation files can be regenerated separately.
    if (output.exists() and any(output.iterdir())) or (results / "training_manifest.json").exists():
        raise FileExistsError(f"Existing training run at {output} or {results}; use a fresh run name/output")
    output.mkdir(parents=True, exist_ok=True)
    if run_name == "length_balanced" and dataset_path is None:
        dataset_path = cfg["paths"]["dpo_length_train"]
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    model, loader, optimizer = (bundle[k] for k in ("model", "loader", "optimizer"))
    accum, epochs = int(cfg["grad_accum_steps"]), int(cfg["epochs"])
    if accum < 1 or epochs < 1:
        raise ValueError("epochs and grad_accum_steps must be positive integers")
    budget = {"epochs": epochs, "examples_per_epoch": len(bundle["rows"]),
              "total_examples": epochs * len(bundle["rows"]),
              "microbatches_per_epoch": len(loader),
              "optimizer_steps": epochs * math.ceil(len(loader) / accum),
              "batch_size": int(cfg["batch_size"]), "grad_accum_steps": accum,
              "effective_batch_size": int(cfg["batch_size"]) * accum,
              "short_run": max_examples is not None}
    overflow = []
    for i, row in enumerate(bundle["rows"]):
        length = len(bundle["tokenizer"].apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        if length >= int(cfg["max_sequence_length"]):
            overflow.append({"data_index": i, "prompt_id": row.get("prompt_id"),
                             "original_prompt_tokens": length})
    metadata = {"experiment": run_name, "seed": int(cfg["seed"]), "beta": bundle["beta"],
                "initialization": "original_base_with_fresh_lora", "output": str(output),
                "base_model_revision": getattr(model.config, "_commit_hash", None),
                "optimizer": {"name": "AdamW", "defaults": optimizer.defaults},
                "dataset": dataset_metadata(bundle["dataset_path"], bundle["rows"]),
                "training_budget": budget,
                "encoding": {"overflow_prompt_suffix_tokens": int(cfg["max_sequence_length"]) // 2,
                             "rule": "only if prompt_tokens >= max_sequence_length; otherwise retain full prompt",
                             "overflow_prompts": overflow},
                **provenance(cfg), "status": "running"}
    save_json(results / "training_manifest.json", metadata)
    # PEFT keeps trainable adapters in float32. Scaling protects the fp16 forward
    # pass; unscale before clipping. On CPU/bf16 the scaler is disabled.
    fp16_cuda = torch.cuda.is_available() and next(model.parameters()).dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16_cuda)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step = examples_seen = skipped_steps = 0
    totals = {"dpo_loss": 0.0, "preference_accuracy": 0.0, "preference_margin_mean": 0.0}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        iterator = iter(loader)
        while window := list(islice(iterator, accum)):
            count = sum(c["input_ids"].shape[0] for c, _ in window)
            stats = dict.fromkeys(totals, 0.0)
            for chosen, rejected in window:
                n = chosen["input_ids"].shape[0]
                chosen, rejected = move_batch(chosen, model), move_batch(rejected, model)
                pc, pr, rc, rr = pair_logprobs(model, chosen, rejected)
                loss, diag = dpo_loss(pc, pr, rc, rr, bundle["beta"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite DPO loss; stopping before checkpointing")
                # Weight by examples, including a smaller final accumulation window.
                scaler.scale(loss * n / count).backward()
                stats["dpo_loss"] += float(loss.detach()) * n
                for key in ("preference_accuracy", "preference_margin_mean"):
                    stats[key] += float(diag[key]) * n
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), float(cfg["max_grad_norm"]))
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradients")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            skipped_steps += int(skipped)
            optimizer.zero_grad(set_to_none=True)
            step += 1
            examples_seen += count
            for key in totals:
                totals[key] += stats[key]
            record = {"experiment": run_name, "epoch": epoch + 1, "step": step,
                      "examples_seen": examples_seen, "examples_in_step": count,
                      "beta": bundle["beta"], "optimizer_step_skipped": skipped,
                      "grad_norm": float(norm) if torch.isfinite(norm) else None,
                      "elapsed_seconds": time.perf_counter() - started,
                      **{key: value / count for key, value in stats.items()}}
            append_jsonl(results / "training_log.jsonl", record)
            print(f"{run_name} epoch={epoch+1} step={step} loss={record['dpo_loss']:.5f}", flush=True)

    assert_frozen_reference(model)
    model.save_pretrained(output)
    bundle["tokenizer"].save_pretrained(output)
    summary = {"experiment": run_name, "beta": bundle["beta"], "seed": int(cfg["seed"]),
               "training_budget": budget, "examples_seen": examples_seen,
               "optimizer_steps_attempted": step, "optimizer_steps_skipped": skipped_steps,
               "elapsed_seconds": time.perf_counter() - started,
               "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
               **{key: value / examples_seen for key, value in totals.items()}}
    metadata["status"] = "complete"
    metadata["training_summary"] = summary
    save_json(output / "training_metadata.json", metadata)
    save_json(results / "training_summary.json", summary)
    save_json(results / "training_manifest.json", metadata)
    return str(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
