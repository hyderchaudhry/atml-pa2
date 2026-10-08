from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from common.data import (load_yaml, prompt_messages, prompt_messages_from_preference,
                         read_jsonl, repo_path, write_jsonl)
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed
from common.metrics import parse_word_limit, sampled_kl, word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.runtime import (dataset_metadata, file_digest, length_statistics, move_batch,
                              pair_logprobs, provenance, result_directory, row_identity)
from task1_dpo.train import make_collate


STRATA = ("preferred_longer", "length_matched", "rejected_longer")


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    metadata = load_json(repo_path(adapter) / "training_metadata.json")
    if metadata["status"] != "complete":
        raise ValueError("Evaluation requires a completed training run")
    # Prevent applying new evaluation settings to one condition accidentally.
    if metadata["config"] != cfg:
        raise ValueError("Evaluation config differs from the saved training config")
    set_seed(int(cfg["seed"]))
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.requires_grad_(False)
    policy.eval()
    return {"cfg": cfg, "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
            "tokenizer": load_tokenizer(cfg["base_model"]), "policy": policy,
            "reward": load_reward_model(cfg), "training_metadata": metadata}


@torch.no_grad()
def evaluate_preferences(model, tokenizer, rows, cfg, beta):
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    records = []
    size = int(cfg["batch_size"])
    model.eval()
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        chosen, rejected = collate(chunk)
        chosen, rejected = move_batch(chosen, model), move_batch(rejected, model)
        pc, pr, rc, rr = pair_logprobs(model, chosen, rejected)
        loss, _ = dpo_loss(pc, pr, rc, rr, beta)
        margin = (pc - rc) - (pr - rr)
        losses = -F.logsigmoid(beta * margin)
        # Each pair gets equal weight. In particular, a smaller final batch must
        # not have the same aggregate weight as a full batch.
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite held-out DPO loss")
        for j, row in enumerate(chunk):
            prompt_tokens = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row),
                                                              tokenize=True, add_generation_prompt=True))
            records.append({**row_identity(row, start + j),
                            "original_prompt_tokens": prompt_tokens,
                            "prompt_truncated": prompt_tokens >= int(cfg["max_sequence_length"]),
                            "policy_chosen_logp": float(pc[j]), "policy_rejected_logp": float(pr[j]),
                            "ref_chosen_logp": float(rc[j]), "ref_rejected_logp": float(rr[j]),
                            "preference_margin": float(margin[j]), "correct": bool(margin[j] > 0),
                            "dpo_loss": float(losses[j]),
                            "chosen_response_tokens_scored": int(chosen["response_mask"][j].sum()),
                            "rejected_response_tokens_scored": int(rejected["response_mask"][j].sum())})
    if not records:
        raise ValueError("Empty held-out preference set")
    return summarize_preferences(records), records


def summarize_preferences(records):
    return {"num_pairs": len(records),
            "dpo_loss": sum(r["dpo_loss"] for r in records) / len(records),
            "preference_margin": sum(r["preference_margin"] for r in records) / len(records),
            "preference_accuracy": sum(r["correct"] for r in records) / len(records)}


@torch.no_grad()
def evaluate_generations(model, tokenizer, reward, rows, cfg, path, *, preference_rows=False):
    # Reset for each fixed prompt set, independently of previous evaluations.
    set_seed(int(cfg["seed"]))
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    rm, rm_tokenizer = reward
    # Keep the end of an overflowing chat prompt, including the assistant header.
    tokenizer.truncation_side = "left"
    # Right truncation of a long prompt+completion can remove the entire response
    # before reward scoring. Keep the response and the most recent prompt context.
    rm_tokenizer.truncation_side = "left"
    records = []
    total_kl_numerator = total_tokens = 0.0
    size = int(cfg["batch_size"])
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        prompts = [(prompt_messages_from_preference(r) if preference_rows else prompt_messages(r))
                   for r in chunk]
        generated = batch_generate(model, tokenizer, prompts,
                                   max_prompt_length=int(cfg["max_sequence_length"]),
                                   max_new_tokens=int(cfg["max_generation_tokens"]),
                                   **cfg["generation"])
        args = (generated["sequences"], generated["attention_mask"],
                generated["prompt_width"], generated["response_ids"])
        policy_logp, _ = response_token_logprobs(model, *args)
        with reference_mode(model):
            ref_logp, _ = response_token_logprobs(model, *args)
        mask = generated["response_mask"]
        # The released helper averages over VALID TOKENS. Weight batch estimates
        # by their token counts, rather than averaging batch/sequence means.
        count = float(mask.sum())
        kl = sampled_kl(policy_logp, ref_logp, mask)
        total_kl_numerator += float(kl) * count
        total_tokens += count
        scores = score_reward_pairs(rm, rm_tokenizer, prompts, generated["responses"], max_length=1024)
        if not torch.isfinite(scores).all() or not torch.isfinite(kl):
            raise FloatingPointError("Non-finite generation evaluation metric")
        for j, (row, prompt, response) in enumerate(zip(chunk, prompts, generated["responses"])):
            text = "\n".join(str(m["content"]) for m in prompt if m["role"] == "user")
            numerator = float(((policy_logp[j] - ref_logp[j]) * mask[j]).sum())
            tokens = int(mask[j].sum())
            record = {**row_identity(row, start + j), "messages": prompt, "response": response,
                      "original_prompt_tokens": len(tokenizer.apply_chat_template(
                          prompt, tokenize=True, add_generation_prompt=True)),
                      "response_token_ids": generated["response_ids"][j, :tokens].cpu().tolist(),
                      "response_tokens": tokens, "word_count": word_count(response),
                      "word_limit": parse_word_limit(text),
                      "word_limit_compliance": word_limit_compliance(text, response),
                      "reward_score": float(scores[j]), "kl_log_ratio_sum": numerator,
                      "kl_response_tokens": tokens,
                      "terminated_with_eos": generated["terminated_with_eos"][j],
                      "truncated": generated["truncated"][j]}
            # Persist each completed response batch so later analysis survives exit.
            append_jsonl(path, record)
            records.append(record)
        print(f"generated {min(start+size, len(rows))}/{len(rows)} -> {path.name}", flush=True)
    if not records or total_tokens == 0:
        raise ValueError("Empty generation evaluation")
    limits = [r["word_limit_compliance"] for r in records if r["word_limit_compliance"] is not None]
    summary = {"num_prompts": len(records), "kl_from_reference": total_kl_numerator / total_tokens,
               "kl_log_ratio_sum": total_kl_numerator, "kl_response_tokens": int(total_tokens),
               "reward_model_score": sum(r["reward_score"] for r in records) / len(records),
               "response_length_tokens": length_statistics([r["response_tokens"] for r in records]),
               "word_count": length_statistics([r["word_count"] for r in records]),
               "word_limit_compliance": sum(limits) / len(limits) if limits else None,
               "num_word_limit_prompts": len(limits),
               "truncation_rate": sum(r["truncated"] for r in records) / len(records)}
    return summary


def run_evaluation(config_path, adapter, name):
    cfg = load_yaml(config_path)
    results = result_directory(cfg, name)
    started = time.perf_counter()
    bundle = load_evaluation_bundle(config_path, adapter)
    model, tokenizer = bundle["policy"], bundle["tokenizer"]
    training = bundle["training_metadata"]
    if training["experiment"] != name:
        raise ValueError("Evaluation name must match the saved training experiment")
    beta = float(training["beta"])
    standard = bundle["rows"]
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    words = read_jsonl(cfg["paths"]["word_limit_prompts"])
    if set(r["length_stratum"] for r in stratified) != set(STRATA):
        raise ValueError("Unexpected course length-stratum labels")
    if any(parse_word_limit("\n".join(m["content"] for m in prompt_messages(r))) is None for r in words):
        raise ValueError("A common word-limit prompt could not be parsed")
    manifest = {"experiment": name, "beta": beta, "seed": int(cfg["seed"]),
                "adapter": str(repo_path(adapter)), "training_budget": training["training_budget"],
                "training_metadata_sha256": file_digest(repo_path(adapter) / "training_metadata.json"),
                "model_revisions": {"base": training.get("base_model_revision"),
                                    "reward": getattr(bundle["reward"][0].config, "_commit_hash", None)},
                "adapter_sha256": {p.name: file_digest(p) for p in repo_path(adapter).glob("adapter*") if p.is_file()},
                "datasets": {key: dataset_metadata(cfg["paths"][key], rows) for key, rows in
                             (("dpo_standard_eval", standard), ("dpo_length_eval", stratified),
                              ("word_limit_prompts", words))},
                "settings": {"generation": cfg["generation"], "batch_size": int(cfg["batch_size"]),
                             "max_generation_tokens": int(cfg["max_generation_tokens"]),
                             "max_sequence_length": int(cfg["max_sequence_length"]),
                             "reward_max_length": 1024,
                             "reward_truncation": "left; retain generated response and most recent context",
                             "preference_accuracy": "mean(((pc-rc)-(pr-rr)) > 0); ties incorrect",
                             "sequence_logp": "sum over response tokens, including EOS; prompt/padding masked",
                             "overflow_preference_prompt": "if prompt >= context cap, retain common suffix of cap//2 tokens",
                             "generation_prompt_truncation": "left; retain up to max_sequence_length tokens",
                             "kl": "common.metrics.sampled_kl; global valid-response-token mean",
                             "response_length": "generated tokens after prompt through first EOS inclusive; padding excluded",
                             "length_std": "population (ddof=0)",
                             "word_limit": "common.metrics.word_limit_compliance (word_count <= parsed limit)",
                             "generation_prompts": "all standard held-out pairs in file order; separate common word-limit set"},
                **provenance(cfg), "status": "running"}
    save_json(results / "evaluation_manifest.json", manifest)
    preferences, pairs = evaluate_preferences(model, tokenizer, standard, cfg, beta)
    write_jsonl(results / "preference_pairs.jsonl", pairs)
    length_preferences, length_pairs = evaluate_preferences(model, tokenizer, stratified, cfg, beta)
    write_jsonl(results / "length_preference_pairs.jsonl", length_pairs)
    strata = {key: summarize_preferences([r for r in length_pairs if r["length_stratum"] == key]) for key in STRATA}
    generated = evaluate_generations(model, tokenizer, bundle["reward"], standard, cfg,
                                     results / "generated_responses.jsonl", preference_rows=True)
    common_words = evaluate_generations(model, tokenizer, bundle["reward"], words, cfg,
                                        results / "word_limit_responses.jsonl")
    metrics = {"experiment": name, "seed": int(cfg["seed"]), "beta": beta,
               "training_budget": training["training_budget"], "held_out": preferences,
               "generation": generated, "length_stratified": {"overall": length_preferences, "strata": strata},
               "common_word_limit": common_words, "elapsed_seconds": time.perf_counter() - started}
    save_json(results / "metrics.json", metrics)
    manifest["status"] = "complete"
    save_json(results / "evaluation_manifest.json", manifest)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
