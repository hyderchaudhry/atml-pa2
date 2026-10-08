"""Task 1 batching, provenance, and result conventions (no other task data)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import re
from pathlib import Path

import numpy as np
import torch

from common.data import repo_path
from common.generation import response_sequence_logprobs
from common.models import reference_mode


def result_directory(cfg, name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Run name must contain only letters, digits, '_' or '-'")
    path = repo_path(cfg["results_dir"]) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_directory(cfg, name, override=None):
    if override:
        return repo_path(override)
    if name == "standard":
        return repo_path(cfg["standard_output"])
    if name == "length_balanced":
        return repo_path(cfg["length_output"])
    return repo_path(cfg["standard_output"]).parent / name


def row_identity(row, index):
    return {"data_index": index, **{k: row[k] for k in
            ("prompt_id", "source_index", "source_split", "length_stratum") if k in row}}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_metadata(path, rows):
    path = repo_path(path)
    return {"path": str(path), "sha256": file_digest(path), "num_examples": len(rows),
            "selection": [row_identity(row, i) for i, row in enumerate(rows)]}


def provenance(cfg):
    versions = {}
    for package in ("torch", "transformers", "peft", "tokenizers", "accelerate", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = list(repo_path("task1_dpo").glob("*.py")) + list(repo_path("common").glob("*.py"))
    return {"config": cfg, "versions": versions,
            "code_sha256": {str(p.relative_to(repo_path("."))): file_digest(p) for p in files},
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def move_batch(batch, model):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def assert_frozen_reference(model):
    # reference_mode disables LoRA. The remaining base weights must never be optimized.
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError(f"Expected only trainable LoRA weights; found {trainable[:10]}")


def pair_logprobs(model, chosen, rejected):
    # Reuse the frozen base through PEFT's adapter-disabled context; no second 1.5B
    # copy is needed. Eval mode also disables reference dropout. Policy graphs are
    # built after leaving the context, with the adapter restored.
    with torch.no_grad(), reference_mode(model):
        rc = response_sequence_logprobs(model, chosen)[0]
        rr = response_sequence_logprobs(model, rejected)[0]
    pc = response_sequence_logprobs(model, chosen)[0]
    pr = response_sequence_logprobs(model, rejected)[0]
    return pc, pr, rc, rr


def length_statistics(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty response set")
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)), "min": int(arr.min()), "max": int(arr.max())}


def beta_name(beta):
    return f"beta_{round(float(beta) * 100):03d}"
