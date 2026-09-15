"""Convert examples into model-facing formats: LLM chat (SFT) and word-level BIO (encoder)."""

from __future__ import annotations

import json
import re

from lahja.data.schema import Example


def slots_to_dict(pairs: list[tuple[str, str]]) -> dict[str, str | list[str]]:
    """{type: value}; a type that occurs more than once gets a list of values."""
    out: dict[str, str | list[str]] = {}
    for t, v in pairs:
        if t not in out:
            out[t] = v
        elif isinstance(out[t], list):
            out[t].append(v)
        else:
            out[t] = [out[t], v]
    return out


def dict_to_pairs(slots: dict) -> list[tuple[str, str]]:
    pairs = []
    for t, v in slots.items():
        for x in v if isinstance(v, list) else [v]:
            if isinstance(x, bool) or not isinstance(x, str | int | float):
                raise TypeError(f"Slot {t!r} has non-string value {x!r}")
            pairs.append((str(t), str(x)))
    return pairs


def target_json(intent: str, pairs: list[tuple[str, str]]) -> str:
    return json.dumps({"intent": intent, "slots": slots_to_dict(pairs)}, ensure_ascii=False)


def example_target(ex: Example) -> str:
    return target_json(ex.intent, ex.slot_pairs())


def to_chat(ex: Example, system_prompt: str) -> dict:
    """One SFT record in the `messages` format read by both TRL and mlx-lm."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.utt},
            {"role": "assistant", "content": example_target(ex)},
        ]
    }


def word_bio(ex: Example) -> tuple[list[str], list[str]]:
    """Whitespace tokens with BIO tags; a word touching a slot span gets that slot's tag."""
    tokens, tags = [], []
    for m in re.finditer(r"\S+", ex.utt):
        s, e = m.span()
        tag = "O"
        for slot in ex.slots:
            if s < slot.end and e > slot.start:
                tag = ("B-" if s <= slot.start else "I-") + slot.type
                break
        tokens.append(m.group())
        tags.append(tag)
    return tokens, tags
