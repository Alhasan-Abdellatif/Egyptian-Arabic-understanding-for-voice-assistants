"""Intent accuracy, slot micro-F1, exact match, JSON validity, and bootstrap CIs."""

from __future__ import annotations

import json
import random
from collections import Counter

from lahja.data.formats import dict_to_pairs
from lahja.data.normalize import normalize
from lahja.data.schema import Example


def parse_prediction(text: str | None) -> dict | None:
    """Extract {"intent", "pairs"} from raw model output; None if it is not valid NLU JSON."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("intent"), str):
        return None
    slots = obj.get("slots", {})
    if slots is None:
        slots = {}
    if not isinstance(slots, dict):
        return None
    try:
        pairs = dict_to_pairs(slots)
    except TypeError:
        return None
    return {"intent": obj["intent"].strip(), "pairs": pairs}


def _slot_counter(pairs: list[tuple[str, str]]) -> Counter:
    return Counter((t.strip(), normalize(v)) for t, v in pairs)


def score_item(gold: Example, raw: str | None) -> dict:
    pred = parse_prediction(raw)
    gc = _slot_counter(gold.slot_pairs())
    n_gold = sum(gc.values())
    if pred is None:
        return {
            "valid": 0,
            "intent_ok": 0,
            "em": 0,
            "tp": 0,
            "fp": 0,
            "fn": n_gold,
            "pred_intent": None,
            "pred_pairs": None,
        }
    pc = _slot_counter(pred["pairs"])
    tp = sum((gc & pc).values())
    intent_ok = int(pred["intent"] == gold.intent)
    return {
        "valid": 1,
        "intent_ok": intent_ok,
        "em": int(intent_ok and gc == pc),
        "tp": tp,
        "fp": sum(pc.values()) - tp,
        "fn": n_gold - tp,
        "pred_intent": pred["intent"],
        "pred_pairs": pred["pairs"],
    }


def bootstrap_ci(flags: list[int], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not flags:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def aggregate(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    tp, fp, fn = (sum(i[k] for i in items) for k in ("tp", "fp", "fn"))
    if tp + fp + fn == 0:
        p = r = f1 = 1.0
    else:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
    em_flags = [i["em"] for i in items]
    lo, hi = bootstrap_ci(em_flags)
    return {
        "n": n,
        "json_valid": sum(i["valid"] for i in items) / n,
        "intent_acc": sum(i["intent_ok"] for i in items) / n,
        "slot_precision": p,
        "slot_recall": r,
        "slot_f1": f1,
        "exact_match": sum(em_flags) / n,
        "exact_match_ci95": [lo, hi],
    }
