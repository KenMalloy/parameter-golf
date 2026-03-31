from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.build_training_proxy import infer_proxy_val_docs, infer_source_num_val_docs


class BuildTrainingProxyTests(unittest.TestCase):
    def test_infer_proxy_val_docs_defaults_to_zero_for_single_doc_proxy(self) -> None:
        self.assertEqual(infer_proxy_val_docs(sample_docs=1, override=None), 0)

    def test_infer_proxy_val_docs_defaults_to_one_eighth_capped_at_sixty_four(self) -> None:
        self.assertEqual(infer_proxy_val_docs(sample_docs=40, override=None), 5)
        self.assertEqual(infer_proxy_val_docs(sample_docs=1000, override=None), 64)

    def test_infer_proxy_val_docs_honors_override(self) -> None:
        self.assertEqual(infer_proxy_val_docs(sample_docs=8, override=3), 3)

    def test_infer_source_num_val_docs_prefers_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_jsonl = Path(tmpdir) / "docs_selected.jsonl"
            docs_jsonl.write_text('{"text":"a"}\n', encoding="utf-8")
            sidecar = docs_jsonl.with_name("docs_selected.source_manifest.json")
            sidecar.write_text(json.dumps({"docs_val": 99}) + "\n", encoding="utf-8")

            self.assertEqual(infer_source_num_val_docs(docs_jsonl, override=7), 7)

    def test_infer_source_num_val_docs_uses_sidecar_before_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_jsonl = Path(tmpdir) / "docs_selected.jsonl"
            docs_jsonl.write_text('{"text":"a"}\n', encoding="utf-8")
            sidecar = docs_jsonl.with_name("docs_selected.source_manifest.json")
            sidecar.write_text(json.dumps({"docs_val": 123}) + "\n", encoding="utf-8")

            self.assertEqual(infer_source_num_val_docs(docs_jsonl, override=None), 123)


if __name__ == "__main__":
    unittest.main()
