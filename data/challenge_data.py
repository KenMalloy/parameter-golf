"""Shared helpers for challenge dataset/tokenizer workflows.

These helpers are intentionally lightweight so they can be reused from tests,
small data tools, and both training entrypoints without pulling in framework
dependencies.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    """Fail fast when a dataset export and tokenizer model do not match.

    The published exports include a manifest that records which tokenizer
    produced each dataset directory. For custom-tokenizer work, accidentally
    pointing the baseline model at the wrong dataset/tokenizer pair is an easy
    way to waste runs while reporting meaningless BPB.
    """

    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None

    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    actual_name = Path(tokenizer_path).name
    if expected_name and actual_name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {actual_name}")

    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def sample_training_doc_indices(
    docs_total: int,
    *,
    num_val_docs: int,
    sample_docs: int,
    seed: int,
) -> list[int]:
    """Return deterministic training-document indices for proxy experiments.

    We reserve the leading ``num_val_docs`` documents to mirror the canonical
    export layout, then sample only from the remaining training region. The
    returned indices are sorted so downstream tools can preserve original
    document order while still using a random sample.
    """

    if docs_total < 0:
        raise ValueError(f"docs_total must be non-negative, got {docs_total}")
    if num_val_docs < 0:
        raise ValueError(f"num_val_docs must be non-negative, got {num_val_docs}")
    if sample_docs < 0:
        raise ValueError(f"sample_docs must be non-negative, got {sample_docs}")
    if num_val_docs > docs_total:
        raise ValueError(
            f"num_val_docs={num_val_docs} exceeds docs_total={docs_total}; proxy sampling would leak validation docs"
        )

    train_indices = list(range(num_val_docs, docs_total))
    if sample_docs >= len(train_indices):
        return train_indices

    rng = random.Random(seed)
    chosen = rng.sample(train_indices, sample_docs)
    chosen.sort()
    return chosen


def count_jsonl_docs(path: str | Path) -> int:
    """Count JSONL records without loading the full file into memory."""

    with Path(path).open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_selected_docs_jsonl(
    source_jsonl: str | Path,
    destination_jsonl: str | Path,
    *,
    doc_indices: list[int],
) -> int:
    """Copy a sorted subset of JSONL records while preserving exact original lines."""

    if doc_indices != sorted(doc_indices):
        raise ValueError("doc_indices must be sorted in ascending order")
    if len(set(doc_indices)) != len(doc_indices):
        raise ValueError("doc_indices must be unique")
    if any(idx < 0 for idx in doc_indices):
        raise ValueError("doc_indices must be non-negative")

    source_path = Path(source_jsonl)
    destination_path = Path(destination_jsonl)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = iter(doc_indices)
    next_idx = next(wanted, None)
    written = 0
    with source_path.open("r", encoding="utf-8") as src, destination_path.open("w", encoding="utf-8") as dst:
        for doc_idx, line in enumerate(src):
            if next_idx is None:
                break
            if doc_idx < next_idx:
                continue
            if doc_idx != next_idx:
                raise ValueError(f"doc index {next_idx} is out of range for {source_path}")
            dst.write(line)
            written += 1
            next_idx = next(wanted, None)

    if next_idx is not None:
        raise ValueError(f"doc index {next_idx} is out of range for {source_path}")
    return written


def build_training_proxy_docs(
    source_jsonl: str | Path,
    destination_jsonl: str | Path,
    *,
    num_val_docs: int,
    sample_docs: int,
    seed: int,
    proxy_val_docs: int = 0,
    docs_total: int | None = None,
) -> dict[str, Any]:
    """Sample a deterministic proxy corpus from the training-doc region only."""

    source_path = Path(source_jsonl)
    docs_total = count_jsonl_docs(source_path) if docs_total is None else int(docs_total)
    doc_indices = sample_training_doc_indices(
        docs_total,
        num_val_docs=num_val_docs,
        sample_docs=sample_docs,
        seed=seed,
    )
    if not (0 <= proxy_val_docs <= len(doc_indices)):
        raise ValueError(f"proxy_val_docs must be in [0, {len(doc_indices)}], got {proxy_val_docs}")

    proxy_docs = write_selected_docs_jsonl(source_path, destination_jsonl, doc_indices=doc_indices)
    return {
        "docs_total": docs_total,
        "source_num_val_docs": num_val_docs,
        "proxy_docs": proxy_docs,
        "proxy_val_docs": proxy_val_docs,
        "sample_docs": sample_docs,
        "seed": seed,
        "selected_doc_indices": doc_indices,
    }
