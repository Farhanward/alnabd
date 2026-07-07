from __future__ import annotations

from .model import PulseModel


POSITIVE_WORDS = ["clear", "useful", "simple", "today", "free", "new", "save", "easy", "fast", "proven"]
NEGATIVE_WORDS = ["hate", "bad", "never", "can't", "problem", "fail", "angry"]


def score_text(text: str, model: PulseModel) -> dict:
    score = model.score(text)
    reasons = []
    low = text.lower()
    for word in POSITIVE_WORDS:
        if word in low:
            reasons.append(f"إشارة إيجابية: {word}")
    for word in NEGATIVE_WORDS:
        if word in low:
            reasons.append(f"إشارة سلبية: {word}")
    if len(text) < 40:
        reasons.append("النص قصير؛ أضف قيمة أو مثالاً.")
    return {"score": score, "decision": "PROMISING" if score >= model.threshold else "IMPROVE", "reasons": reasons[:6]}


def rewrite_text(text: str, model: PulseModel) -> dict:
    result = score_text(text, model)
    improved = text.strip()
    if result["decision"] == "IMPROVE":
        improved = f"{improved} — جديد، واضح، وسهل التطبيق اليوم."
    return {**result, "rewrite": improved}
