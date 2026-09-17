"""Subword BIO tagging for the encoder baseline, driven by tokenizer character offsets.

Offsets of (0, 0) mark special and padding tokens; they get no label and are skipped.
"""

from __future__ import annotations

from lahja.data.schema import Example


def tag_set(slot_types: list[str]) -> list[str]:
    return ["O"] + [f"{p}-{t}" for t in slot_types for p in ("B", "I")]


def bio_labels(ex: Example, offsets: list[tuple[int, int]]) -> list[str | None]:
    labels: list[str | None] = []
    for s, e in offsets:
        if s == e:
            labels.append(None)
            continue
        tag = "O"
        for slot in ex.slots:
            if s < slot.end and e > slot.start:
                tag = ("B-" if s <= slot.start else "I-") + slot.type
                break
        labels.append(tag)
    return labels


def decode_spans(
    utt: str, offsets: list[tuple[int, int]], tags: list[str]
) -> list[tuple[str, str]]:
    """Merge B/I runs into (slot_type, text) pairs. An I- tag without a matching open span starts one."""
    spans: list[list] = []  # [type, start, end]
    current: list | None = None
    for (s, e), tag in zip(offsets, tags, strict=True):
        if s == e:
            continue
        if tag == "O":
            current = None
            continue
        prefix, stype = tag.split("-", 1)
        if prefix == "I" and current is not None and current[0] == stype:
            current[2] = e
        else:
            current = [stype, s, e]
            spans.append(current)
    return [(t, utt[s:e]) for t, s, e in spans]
