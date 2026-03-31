from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.challenge_data import (
    build_training_proxy_docs,
    count_jsonl_docs,
    sample_training_doc_indices,
    validate_dataset_tokenizer_pair,
    write_selected_docs_jsonl,
)


class ChallengeDataTests(unittest.TestCase):
    def test_validate_dataset_tokenizer_pair_accepts_matching_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "data" / "datasets" / "fineweb10B_sp1024"
            tokenizer_dir = root / "data" / "tokenizers"
            dataset_dir.mkdir(parents=True)
            tokenizer_dir.mkdir(parents=True)
            for idx in range(2):
                (dataset_dir / f"fineweb_train_{idx:06d}.bin").write_bytes(b"")
            tokenizer_path = tokenizer_dir / "fineweb_1024_bpe.model"
            tokenizer_path.write_text("placeholder", encoding="utf-8")
            manifest = {
                "tokenizers": [
                    {
                        "name": "sp_bpe_1024",
                        "model_path": "tokenizers/fineweb_1024_bpe.model",
                    }
                ],
                "datasets": [
                    {
                        "name": "fineweb10B_sp1024",
                        "tokenizer_name": "sp_bpe_1024",
                        "stats": {"files_train": 2},
                    }
                ],
            }
            (root / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            dataset_name, train_files, expected_train_files = validate_dataset_tokenizer_pair(
                str(dataset_dir),
                str(tokenizer_path),
            )

            self.assertEqual(dataset_name, "fineweb10B_sp1024")
            self.assertEqual(train_files, 2)
            self.assertEqual(expected_train_files, 2)

    def test_validate_dataset_tokenizer_pair_rejects_wrong_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "data" / "datasets" / "fineweb10B_sp1024"
            tokenizer_dir = root / "data" / "tokenizers"
            dataset_dir.mkdir(parents=True)
            tokenizer_dir.mkdir(parents=True)
            tokenizer_path = tokenizer_dir / "wrong.model"
            tokenizer_path.write_text("placeholder", encoding="utf-8")
            manifest = {
                "tokenizers": [
                    {
                        "name": "sp_bpe_1024",
                        "model_path": "tokenizers/fineweb_1024_bpe.model",
                    }
                ],
                "datasets": [
                    {
                        "name": "fineweb10B_sp1024",
                        "tokenizer_name": "sp_bpe_1024",
                        "stats": {"files_train": 0},
                    }
                ],
            }
            (root / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expects tokenizer fineweb_1024_bpe.model, got wrong.model"):
                validate_dataset_tokenizer_pair(str(dataset_dir), str(tokenizer_path))

    def test_validate_dataset_tokenizer_pair_rejects_extra_train_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "data" / "datasets" / "fineweb10B_sp1024"
            tokenizer_dir = root / "data" / "tokenizers"
            dataset_dir.mkdir(parents=True)
            tokenizer_dir.mkdir(parents=True)
            for idx in range(3):
                (dataset_dir / f"fineweb_train_{idx:06d}.bin").write_bytes(b"")
            tokenizer_path = tokenizer_dir / "fineweb_1024_bpe.model"
            tokenizer_path.write_text("placeholder", encoding="utf-8")
            manifest = {
                "tokenizers": [
                    {
                        "name": "sp_bpe_1024",
                        "model_path": "tokenizers/fineweb_1024_bpe.model",
                    }
                ],
                "datasets": [
                    {
                        "name": "fineweb10B_sp1024",
                        "tokenizer_name": "sp_bpe_1024",
                        "stats": {"files_train": 2},
                    }
                ],
            }
            (root / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "has more train shards than expected"):
                validate_dataset_tokenizer_pair(str(dataset_dir), str(tokenizer_path))

    def test_sample_training_doc_indices_is_deterministic_and_sorted(self) -> None:
        first = sample_training_doc_indices(100, num_val_docs=10, sample_docs=8, seed=1234)
        second = sample_training_doc_indices(100, num_val_docs=10, sample_docs=8, seed=1234)

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 8)
        self.assertGreaterEqual(min(first), 10)
        self.assertLess(max(first), 100)

    def test_sample_training_doc_indices_never_uses_validation_docs(self) -> None:
        indices = sample_training_doc_indices(12, num_val_docs=5, sample_docs=99, seed=42)
        self.assertEqual(indices, [5, 6, 7, 8, 9, 10, 11])

    def test_sample_training_doc_indices_rejects_invalid_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds docs_total"):
            sample_training_doc_indices(5, num_val_docs=6, sample_docs=1, seed=0)

    def test_write_selected_docs_jsonl_preserves_original_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_jsonl = root / "docs.jsonl"
            docs = [
                {"text": "val zero", "meta": 0},
                {"text": "val one", "meta": 1},
                {"text": "train two", "meta": 2},
                {"text": "train three", "meta": 3},
                {"text": "train four", "meta": 4},
            ]
            docs_jsonl.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
            proxy_jsonl = root / "proxy.jsonl"

            written = write_selected_docs_jsonl(docs_jsonl, proxy_jsonl, doc_indices=[2, 4])

            self.assertEqual(written, 2)
            self.assertEqual(count_jsonl_docs(proxy_jsonl), 2)
            self.assertEqual(
                proxy_jsonl.read_text(encoding="utf-8"),
                "".join(json.dumps(docs[idx]) + "\n" for idx in [2, 4]),
            )

    def test_write_selected_docs_jsonl_rejects_unsorted_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_jsonl = Path(tmpdir) / "docs.jsonl"
            docs_jsonl.write_text('{"text":"a"}\n{"text":"b"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be sorted"):
                write_selected_docs_jsonl(docs_jsonl, Path(tmpdir) / "proxy.jsonl", doc_indices=[1, 0])

    def test_build_training_proxy_docs_samples_only_training_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_jsonl = root / "docs.jsonl"
            docs = [{"text": f"doc {idx}"} for idx in range(8)]
            docs_jsonl.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
            proxy_jsonl = root / "proxy.jsonl"

            meta = build_training_proxy_docs(
                docs_jsonl,
                proxy_jsonl,
                num_val_docs=3,
                sample_docs=3,
                seed=99,
                proxy_val_docs=1,
            )

            self.assertEqual(meta["docs_total"], 8)
            self.assertEqual(meta["source_num_val_docs"], 3)
            self.assertEqual(meta["proxy_docs"], 3)
            self.assertEqual(meta["proxy_val_docs"], 1)
            self.assertEqual(meta["selected_doc_indices"], sorted(meta["selected_doc_indices"]))
            self.assertGreaterEqual(min(meta["selected_doc_indices"]), 3)

            sampled_docs = [json.loads(line)["text"] for line in proxy_jsonl.read_text(encoding="utf-8").splitlines()]
            expected_docs = [docs[idx]["text"] for idx in meta["selected_doc_indices"]]
            self.assertEqual(sampled_docs, expected_docs)

    def test_build_training_proxy_docs_rejects_proxy_val_docs_over_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_jsonl = Path(tmpdir) / "docs.jsonl"
            docs_jsonl.write_text('{"text":"a"}\n{"text":"b"}\n{"text":"c"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "proxy_val_docs must be in"):
                build_training_proxy_docs(
                    docs_jsonl,
                    Path(tmpdir) / "proxy.jsonl",
                    num_val_docs=1,
                    sample_docs=1,
                    seed=0,
                    proxy_val_docs=2,
                )


if __name__ == "__main__":
    unittest.main()
