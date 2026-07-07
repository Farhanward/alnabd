from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alnabd.batch import evaluate
from alnabd.core import rewrite_text
from alnabd.model import PulseModel, train


class AlNabdTests(unittest.TestCase):
    def test_train_score_and_batch(self):
        with tempfile.TemporaryDirectory(dir="C:/Projects") as tmp:
            data = Path(tmp) / "data.jsonl"
            model = Path(tmp) / "model.json"
            rows = [
                {"text": "new useful fast tool", "success": True},
                {"text": "clear easy save time", "success": True},
                {"text": "bad angry fail problem", "success": False},
                {"text": "hate never works", "success": False},
            ]
            data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            info = train(data, model)
            self.assertEqual(info["rows"], 4)
            self.assertIn("threshold", info)
            loaded = PulseModel.load(model)
            self.assertGreater(loaded.score("new easy tool"), loaded.score("bad fail"))
            result = rewrite_text("bad fail", loaded)
            self.assertIn("rewrite", result)
            summary = evaluate(data, model)
            self.assertEqual(summary["errors"], 0)

    def test_training_rejects_empty_or_single_class_data(self):
        with tempfile.TemporaryDirectory(dir="C:/Projects") as tmp:
            empty = Path(tmp) / "empty.jsonl"
            one_class = Path(tmp) / "one_class.jsonl"
            model = Path(tmp) / "model.json"
            empty.write_text("", encoding="utf-8")
            one_class.write_text('{"text":"good useful","success":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                train(empty, model)
            with self.assertRaises(ValueError):
                train(one_class, model)


if __name__ == "__main__":
    unittest.main()
