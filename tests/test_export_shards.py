from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.download_hf_docs_and_tokenize import DATAFILE_MAGIC, DATAFILE_VERSION, default_pure_byte_tokenizer, export_shards


def read_shard_tokens(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256:
        raise ValueError(f"short header for {path}")
    if int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"unexpected shard header for {path}")
    count = int(header[2])
    return np.fromfile(path, dtype="<u2", count=count, offset=256 * np.dtype("<i4").itemsize)


class ExportShardsTests(unittest.TestCase):
    def test_export_shards_preserves_split_order_and_bos_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_jsonl = root / "docs.jsonl"
            docs = [
                {"text": "a"},
                {"text": "bc"},
                {"text": "def"},
                {"text": "ghij"},
            ]
            docs_jsonl.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")

            base_tok = default_pure_byte_tokenizer()
            tok = {
                "vocab_size": base_tok.vocab_size,
                "bos_id": base_tok.bos_id,
                "eos_id": base_tok.eos_id,
                "encode": base_tok.encode,
                "encode_batch": base_tok.encode_batch,
            }
            output_dir = root / "dataset"
            stats = export_shards(
                docs_jsonl,
                tok,
                output_dir,
                num_val_docs=2,
                shard_size=4,
                docs_total=len(docs),
            )

            expected_docs = [
                np.concatenate(([base_tok.bos_id], base_tok.encode(doc["text"]).astype(np.uint16)))
                for doc in docs
            ]
            expected_val = np.concatenate(expected_docs[:2])
            expected_train = np.concatenate(expected_docs[2:])
            actual_val = np.concatenate([read_shard_tokens(path) for path in sorted(output_dir.glob("fineweb_val_*.bin"))])
            actual_train = np.concatenate(
                [read_shard_tokens(path) for path in sorted(output_dir.glob("fineweb_train_*.bin"))]
            )

            self.assertTrue((actual_val == expected_val).all())
            self.assertTrue((actual_train == expected_train).all())
            self.assertEqual(stats["docs_total"], 4)
            self.assertEqual(stats["docs_val"], 2)
            self.assertEqual(stats["docs_train"], 2)
            self.assertEqual(stats["tokens_val"], int(expected_val.size))
            self.assertEqual(stats["tokens_train"], int(expected_train.size))
            self.assertEqual(stats["files_val"], 2)
            self.assertEqual(stats["files_train"], 3)

    def test_export_shards_removes_stale_outputs_before_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_jsonl = root / "docs.jsonl"
            docs_jsonl.write_text('{"text":"abc"}\n{"text":"def"}\n', encoding="utf-8")
            output_dir = root / "dataset"
            output_dir.mkdir(parents=True)
            stale_val = output_dir / "fineweb_val_999999.bin"
            stale_train = output_dir / "fineweb_train_999999.bin"
            stale_val.write_bytes(b"stale")
            stale_train.write_bytes(b"stale")

            base_tok = default_pure_byte_tokenizer()
            tok = {
                "vocab_size": base_tok.vocab_size,
                "bos_id": base_tok.bos_id,
                "eos_id": base_tok.eos_id,
                "encode": base_tok.encode,
                "encode_batch": base_tok.encode_batch,
            }
            export_shards(
                docs_jsonl,
                tok,
                output_dir,
                num_val_docs=1,
                shard_size=64,
                docs_total=2,
            )

            self.assertFalse(stale_val.exists())
            self.assertFalse(stale_train.exists())
            self.assertEqual(len(list(output_dir.glob("fineweb_val_*.bin"))), 1)
            self.assertEqual(len(list(output_dir.glob("fineweb_train_*.bin"))), 1)


if __name__ == "__main__":
    unittest.main()
