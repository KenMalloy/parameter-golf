from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm
import torch

from data.download_hf_docs_and_tokenize import build_sentencepiece_tokenizer
from train_gpt import STRUCT_ROLE_MASK, build_sentencepiece_luts


class SentencePieceLutTests(unittest.TestCase):
    def _build_tokenizer(self, root: Path) -> spm.SentencePieceProcessor:
        docs_jsonl = root / "docs.jsonl"
        docs = [
            {"text": "hello world hello world hello world"},
            {"text": "alpha alphabet alphanumeric beta better best"},
            {"text": "mix of URLs like https://example.com/path and abc123 markers"},
            {"text": "sentence ending. another sentence! newline\nseparated text"},
        ]
        docs_jsonl.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
        tokenizers_dir = root / "tokenizers"
        built = build_sentencepiece_tokenizer(
            spec={
                "name": "sp_bpe_320",
                "dataset_suffix": "sp320",
                "vocab_size": 320,
                "model_prefix": "proxy_test_sp320",
            },
            docs_jsonl=docs_jsonl,
            tokenizers_dir=tokenizers_dir,
        )
        return spm.SentencePieceProcessor(model_file=str(built["manifest"]["model_path"]))

    def test_special_and_padding_slots_default_to_boundary_zero_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = self._build_tokenizer(Path(tmpdir))
            extra_vocab = int(sp.vocab_size()) + 8
            base_bytes, has_leading_space, is_boundary_token, _, _, _ = build_sentencepiece_luts(
                sp,
                vocab_size=extra_vocab,
                device=torch.device("cpu"),
            )

            for token_id in [sp.pad_id(), sp.bos_id(), sp.eos_id(), sp.unk_id(), int(sp.vocab_size()), extra_vocab - 1]:
                self.assertEqual(int(base_bytes[token_id].item()), 0)
                self.assertFalse(bool(has_leading_space[token_id].item()))
                self.assertTrue(bool(is_boundary_token[token_id].item()))

    def test_byte_tokens_are_single_byte_and_use_byte_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = self._build_tokenizer(Path(tmpdir))
            base_bytes, _, is_boundary_token, struct_bits, _, _ = build_sentencepiece_luts(
                sp,
                vocab_size=int(sp.vocab_size()),
                device=torch.device("cpu"),
            )
            byte_tokens = [token_id for token_id in range(int(sp.vocab_size())) if sp.is_byte(token_id)]

            self.assertTrue(byte_tokens, "expected byte_fallback tokens in the SentencePiece vocab")
            probe = byte_tokens[0]
            self.assertEqual(int(base_bytes[probe].item()), 1)
            self.assertFalse(bool(is_boundary_token[probe].item()))
            self.assertEqual(int(struct_bits[probe].item()) & STRUCT_ROLE_MASK, 1)

    def test_leading_space_tokens_strip_dummy_space_from_byte_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = self._build_tokenizer(Path(tmpdir))
            base_bytes, has_leading_space, is_boundary_token, _, _, _ = build_sentencepiece_luts(
                sp,
                vocab_size=int(sp.vocab_size()),
                device=torch.device("cpu"),
            )
            candidates = []
            for token_id in range(int(sp.vocab_size())):
                if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id) or sp.is_byte(token_id):
                    continue
                piece = sp.id_to_piece(token_id)
                if piece.startswith("▁") and len(piece) > 1:
                    candidates.append((token_id, piece))

            self.assertTrue(candidates, "expected at least one non-byte word-start piece")
            token_id, piece = candidates[0]
            stripped = piece[1:]
            self.assertTrue(bool(has_leading_space[token_id].item()))
            self.assertFalse(bool(is_boundary_token[token_id].item()))
            self.assertEqual(int(base_bytes[token_id].item()), len(stripped.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
