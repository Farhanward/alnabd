from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9#@']{2,}|[\u0600-\u06ff]{2,}", re.I)


def tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def _classification_metrics(scored: list[tuple[float, int]], threshold: float) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for score, label in scored:
        predicted = score >= threshold
        expected = bool(label)
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _best_threshold(model: "PulseModel", labeled_texts: list[tuple[str, int]]) -> tuple[float, dict[str, float]]:
    scored = sorted(((model.score(text), label) for text, label in labeled_texts), reverse=True)
    positives = sum(label for _, label in scored)
    negatives = len(scored) - positives
    tp = fp = 0
    fn = positives
    tn = negatives
    best_threshold = 0.5
    best_metrics = _classification_metrics(scored, 0.5)
    best_key = (best_metrics["f1"], best_metrics["accuracy"], -abs(best_threshold - 0.5))
    index = 0
    while index < len(scored):
        threshold = scored[index][0]
        while index < len(scored) and scored[index][0] == threshold:
            _, label = scored[index]
            if label:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
            index += 1
        metrics = _metrics_from_counts(tp, tn, fp, fn)
        key = (metrics["f1"], metrics["accuracy"], -abs(threshold - 0.5))
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


@dataclass
class PulseModel:
    vocab: list[str]
    log_prior_good: float
    log_prior_weak: float
    log_good: dict[str, float]
    log_weak: dict[str, float]
    unknown_good: float
    unknown_weak: float
    threshold: float

    def score(self, text: str) -> float:
        good = self.log_prior_good
        weak = self.log_prior_weak
        for token in tokens(text):
            good += self.log_good.get(token, self.unknown_good)
            weak += self.log_weak.get(token, self.unknown_weak)
        mx = max(good, weak)
        eg = math.exp(good - mx)
        ew = math.exp(weak - mx)
        return eg / (eg + ew)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "PulseModel":
        return cls(list(data["vocab"]), float(data["log_prior_good"]), float(data["log_prior_weak"]), {k: float(v) for k, v in data["log_good"].items()}, {k: float(v) for k, v in data["log_weak"].items()}, float(data["unknown_good"]), float(data["unknown_weak"]), float(data.get("threshold", 0.5)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PulseModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train(input_path: str | Path, out_path: str | Path, max_features: int = 20000, alpha: float = 0.5) -> dict:
    if max_features <= 0:
        raise ValueError("max_features must be greater than zero")
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")
    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training data is empty")
    counts = {0: Counter(), 1: Counter()}
    docs = {0: 0, 1: 0}
    df = Counter()
    labeled_texts: list[tuple[str, int]] = []
    for row in rows:
        label = 1 if row.get("success") else 0
        docs[label] += 1
        text = str(row.get("text") or "")
        labeled_texts.append((text, label))
        ts = tokens(text)
        counts[label].update(ts)
        df.update(ts)
    if docs[0] == 0 or docs[1] == 0:
        raise ValueError("training data must contain both positive and weak examples")
    vocab = [tok for tok, _ in df.most_common(max_features)]
    vocab_set = set(vocab)
    totals = {label: sum(c for t, c in counts[label].items() if t in vocab_set) for label in (0, 1)}
    size = max(1, len(vocab))
    total_docs = docs[0] + docs[1]
    model = PulseModel(
        vocab,
        math.log((docs[1] + alpha) / (total_docs + 2 * alpha)),
        math.log((docs[0] + alpha) / (total_docs + 2 * alpha)),
        {t: math.log((counts[1].get(t, 0) + alpha) / (totals[1] + alpha * size)) for t in vocab},
        {t: math.log((counts[0].get(t, 0) + alpha) / (totals[0] + alpha * size)) for t in vocab},
        math.log(alpha / (totals[1] + alpha * size)),
        math.log(alpha / (totals[0] + alpha * size)),
        0.5,
    )
    model.threshold, threshold_metrics = _best_threshold(model, labeled_texts)
    model.save(out_path)
    return {
        "rows": len(rows),
        "positive": docs[1],
        "weak": docs[0],
        "features": len(vocab),
        "threshold": model.threshold,
        "threshold_metrics": threshold_metrics,
        "out": str(Path(out_path).resolve()),
    }
