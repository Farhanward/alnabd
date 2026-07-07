from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path

from .model import PulseModel


def _p(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round((pct / 100) * (len(values) - 1))))]


def evaluate(input_path: str | Path, model_path: str | Path, repeat: int = 1) -> dict:
    model = PulseModel.load(model_path)
    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    tp = tn = fp = fn = errors = 0
    lat = []
    start = time.perf_counter()
    tracemalloc.start()
    for _ in range(repeat):
        for row in rows:
            expected = bool(row.get("success"))
            t0 = time.perf_counter()
            try:
                predicted = model.score(str(row.get("text") or "")) >= model.threshold
            except Exception:
                errors += 1
                predicted = False
            lat.append((time.perf_counter() - t0) * 1000)
            if expected and predicted:
                tp += 1
            elif expected and not predicted:
                fn += 1
            elif not expected and predicted:
                fp += 1
            else:
                tn += 1
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "input": str(Path(input_path).resolve()),
        "model": str(Path(model_path).resolve()),
        "records": len(rows),
        "repeat": repeat,
        "processed": len(rows) * repeat,
        "errors": errors,
        "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "latency_ms": {"mean": statistics.fmean(lat) if lat else 0.0, "p50": _p(lat, 50), "p95": _p(lat, 95), "p99": _p(lat, 99), "max": max(lat) if lat else 0.0},
        "memory_mb": {"current": current / 1_000_000, "peak": peak / 1_000_000},
        "elapsed_seconds": time.perf_counter() - start,
        "collapse_check": {"passed": errors == 0, "criteria": "errors == 0"},
    }
