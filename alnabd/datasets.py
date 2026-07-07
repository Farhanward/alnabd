from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_SIZE_URL = "https://datasets-server.huggingface.co/size"
DATASET = "cardiffnlp/tweet_eval"
CONFIG = "sentiment"
SPLITS = ("train", "validation", "test")


def _url(endpoint: str, **params: Any) -> str:
    return f"{endpoint}?{urlencode(params)}"


def _get(url: str, retries: int = 6) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "alnabd-local-benchmark/0.1"})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(min(90.0, 5.0 * (attempt + 1)))
    raise RuntimeError("unreachable retry state")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def fetch_tweet_eval(out_path: str | Path, limit: int = 12000, page_size: int = 100, sleep: float = 0.05) -> dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = _get(_url(HF_SIZE_URL, dataset=DATASET, config=CONFIG))
    split_sizes = {s["split"]: int(s["num_rows"]) for s in size.get("size", {}).get("splits", []) if s.get("split") in SPLITS}
    existing = _line_count(out)
    rows = existing
    labels = {"0": 0, "1": 0, "2": 0}
    skipped = 0
    mode = "a" if existing else "w"
    with out.open(mode, encoding="utf-8") as handle:
        for split in SPLITS:
            total = split_sizes.get(split, 0)
            for offset in range(0, total, page_size):
                if rows >= limit:
                    break
                page = _get(_url(HF_ROWS_URL, dataset=DATASET, config=CONFIG, split=split, offset=offset, length=min(page_size, total - offset)))
                for item in page.get("rows", []):
                    if rows >= limit:
                        break
                    if skipped < existing:
                        skipped += 1
                        continue
                    row = dict(item.get("row") or {})
                    label = int(row.get("label", 1))
                    text = str(row.get("text") or "")
                    record = {
                        "dataset": DATASET,
                        "config": CONFIG,
                        "split": split,
                        "row_idx": int(item.get("row_idx", offset)),
                        "text": text,
                        "label": label,
                        "success": label == 2,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    labels[str(label)] = labels.get(str(label), 0) + 1
                    rows += 1
                if sleep:
                    time.sleep(sleep)
            if rows >= limit:
                break
    final_labels = {"0": 0, "1": 0, "2": 0}
    final_rows = 0
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            final_rows += 1
            final_labels[str(rec.get("label", ""))] = final_labels.get(str(rec.get("label", "")), 0) + 1
    return {"dataset": DATASET, "config": CONFIG, "out": str(out.resolve()), "rows": final_rows, "resumed_from": existing, "labels": final_labels}
