"""Build a deterministic local tokenizer-comparison proxy from training docs only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.challenge_data import build_training_proxy_docs, count_jsonl_docs
from data.download_hf_docs_and_tokenize import (
    APPEND_EOS,
    DEFAULT_CONFIG,
    NUM_VAL_DOCS,
    SHARD_SIZE,
    VERSION,
    build_tokenizers,
    docs_sidecar_path,
    export_shards,
    load_specs,
    maybe_load_docs_sidecar_meta,
    parse_reuse_sp_models,
    relativize_manifest_paths,
    write_tokenizer_config_export,
)


DOCS_PROXY_FILENAME = "docs_proxy_train.jsonl"
PROXY_MANIFEST_FILENAME = "proxy_manifest.json"


def infer_source_num_val_docs(docs_jsonl: Path, *, override: int | None) -> int:
    if override is not None:
        return int(override)
    docs_sidecar = maybe_load_docs_sidecar_meta(docs_jsonl)
    if docs_sidecar is not None and docs_sidecar.get("docs_val") is not None:
        return int(docs_sidecar["docs_val"])
    return NUM_VAL_DOCS


def infer_proxy_val_docs(*, sample_docs: int, override: int | None) -> int:
    if override is not None:
        return int(override)
    if sample_docs <= 1:
        return 0
    return min(max(1, sample_docs // 8), 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic local proxy dataset from training docs only")
    parser.add_argument(
        "--docs-jsonl",
        default="./data/docs_selected.jsonl",
        help="Local docs_selected.jsonl path. Defaults to ./data/docs_selected.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where the proxy docs, tokenizers, datasets, and manifest are written.",
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=512,
        help="Number of training-region docs to sample into the proxy. Defaults to 512.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260328,
        help="Sampling seed used for the frozen proxy doc list. Defaults to 20260328.",
    )
    parser.add_argument(
        "--num-val-docs",
        type=int,
        default=None,
        help="Number of leading source docs reserved as validation in the original docs cache.",
    )
    parser.add_argument(
        "--proxy-val-docs",
        type=int,
        default=None,
        help="Number of sampled proxy docs to reserve as proxy validation. Defaults to ~12.5%% of the sample.",
    )
    parser.add_argument(
        "--tokenizer-config",
        default=str(DEFAULT_CONFIG),
        help="Local tokenizer config JSON. Defaults to data/tokenizer_specs.json.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=SHARD_SIZE,
        help="Shard size in tokens for the proxy export.",
    )
    parser.add_argument(
        "--tokenizer-train-docs",
        type=int,
        default=None,
        help="Optional cap on how many proxy docs are used to train each tokenizer.",
    )
    parser.add_argument("--skip-byte", action="store_true", help="Skip byte-tokenizer export.")
    parser.add_argument(
        "--reuse-sp-model",
        action="append",
        default=[],
        metavar="VOCAB=MODEL",
        help="Reuse an existing SentencePiece model for the given vocab size instead of retraining it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_tokens <= 0:
        raise ValueError(f"--chunk-tokens must be positive, got {args.chunk_tokens}")
    if args.sample_docs <= 0:
        raise ValueError(f"--sample-docs must be positive, got {args.sample_docs}")

    docs_jsonl = Path(args.docs_jsonl).expanduser().resolve()
    if not docs_jsonl.is_file():
        raise FileNotFoundError(docs_jsonl)

    docs_total = count_jsonl_docs(docs_jsonl)
    source_num_val_docs = infer_source_num_val_docs(docs_jsonl, override=args.num_val_docs)
    if not (0 <= source_num_val_docs <= docs_total):
        raise ValueError(f"--num-val-docs must be in [0, {docs_total}], got {source_num_val_docs}")
    available_train_docs = docs_total - source_num_val_docs
    if available_train_docs <= 0:
        raise ValueError("no source training docs are available after removing the validation prefix")

    effective_sample_docs = min(int(args.sample_docs), available_train_docs)
    proxy_val_docs = infer_proxy_val_docs(sample_docs=effective_sample_docs, override=args.proxy_val_docs)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    proxy_docs_jsonl = output_root / DOCS_PROXY_FILENAME
    proxy_meta = build_training_proxy_docs(
        docs_jsonl,
        proxy_docs_jsonl,
        num_val_docs=source_num_val_docs,
        sample_docs=int(args.sample_docs),
        seed=int(args.seed),
        proxy_val_docs=proxy_val_docs,
        docs_total=docs_total,
    )
    if int(proxy_meta["proxy_docs"]) <= 0:
        raise ValueError("proxy sampling produced zero docs; increase --sample-docs or check --num-val-docs")

    specs = load_specs(Path(args.tokenizer_config).expanduser().resolve())
    reuse_sp_models = parse_reuse_sp_models(args.reuse_sp_model)
    tokenizers, selected_specs = build_tokenizers(
        specs=specs,
        docs_jsonl=proxy_docs_jsonl,
        tokenizers_dir=tokenizers_dir,
        tokenizer_train_docs=args.tokenizer_train_docs,
        skip_byte=args.skip_byte,
        reuse_sp_models=reuse_sp_models,
    )
    write_tokenizer_config_export(output_root, selected_specs)

    source_sidecar = maybe_load_docs_sidecar_meta(docs_jsonl)
    manifest = {
        "version": VERSION,
        "kind": "training_proxy",
        "proxy_docs": proxy_meta["proxy_docs"],
        "proxy_val_docs": proxy_meta["proxy_val_docs"],
        "shard_size": int(args.chunk_tokens),
        "append_eos": APPEND_EOS,
        "docs_jsonl": str(proxy_docs_jsonl),
        "proxy": {
            **proxy_meta,
            "source_docs_jsonl": str(docs_jsonl),
            "source_sidecar_path": str(docs_sidecar_path(docs_jsonl)) if docs_sidecar_path(docs_jsonl).is_file() else None,
            "source_docs_sha256": None if source_sidecar is None else source_sidecar.get("docs_sha256"),
        },
        "tokenizer_specs": selected_specs,
        "tokenizers": [],
        "datasets": [],
    }
    if source_sidecar is not None:
        manifest["proxy"]["source_sidecar"] = source_sidecar

    for tok in tokenizers:
        output_dir = datasets_dir / tok["dataset_name"]
        print(
            f"Exporting proxy dataset: {tok['dataset_name']} "
            f"({proxy_meta['proxy_docs']} docs, proxy_val_docs={proxy_meta['proxy_val_docs']})",
            flush=True,
        )
        stats = export_shards(
            proxy_docs_jsonl,
            tok,
            output_dir,
            num_val_docs=int(proxy_meta["proxy_val_docs"]),
            shard_size=int(args.chunk_tokens),
            docs_total=int(proxy_meta["proxy_docs"]),
        )
        manifest["tokenizers"].append(tok["manifest"])
        manifest["datasets"].append(
            {
                "name": tok["dataset_name"],
                "tokenizer_name": tok["name"],
                "tokenizer_kind": tok["kind"],
                "path": str(output_dir),
                "train_glob": str(output_dir / "fineweb_train_*.bin"),
                "val_glob": str(output_dir / "fineweb_val_*.bin"),
                "vocab_size": tok["vocab_size"],
                "bos_id": tok["bos_id"],
                "eos_id": tok["eos_id"],
                "recommended_bigram_vocab_size": tok["recommended_bigram_vocab_size"],
                "stats": stats,
            }
        )

    manifest = relativize_manifest_paths(manifest, output_root)
    manifest_path = output_root / PROXY_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Proxy manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
