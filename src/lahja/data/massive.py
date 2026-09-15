"""Load Amazon MASSIVE and parse its bracket annotation format.

MASSIVE writes slots inline: ``صحيني [date : غدا] [time : السادسة]``. The same notation
is used for the hand-written Egyptian test set, so one parser serves both.
"""

from __future__ import annotations

import re
from pathlib import Path

from lahja.data.schema import Example, Slot, read_jsonl

SLOT_RE = re.compile(r"\[([^\[\]:]+):([^\[\]]*)\]")


def parse_annotated(annot: str) -> tuple[str, list[Slot]]:
    """Return the plain utterance and its slots (with char offsets) from bracket notation."""
    annot = " ".join(annot.split())
    parts: list[str] = []
    slots: list[Slot] = []
    pos = 0
    out_len = 0
    for m in SLOT_RE.finditer(annot):
        before = annot[pos : m.start()]
        parts.append(before)
        out_len += len(before)
        stype = m.group(1).strip()
        value = " ".join(m.group(2).split())
        if not stype or not value:
            raise ValueError(f"Empty slot type or value in: {annot!r}")
        slots.append(Slot(stype, value, out_len, out_len + len(value)))
        parts.append(value)
        out_len += len(value)
        pos = m.end()
    parts.append(annot[pos:])
    utt = "".join(parts)
    if "[" in utt or "]" in utt:
        raise ValueError(f"Unbalanced brackets in: {annot!r}")
    return utt, slots


def to_annotated(ex: Example) -> str:
    """Inverse of parse_annotated: render an example back into bracket notation."""
    out, pos = [], 0
    for s in sorted(ex.slots, key=lambda s: s.start):
        out.append(ex.utt[pos : s.start])
        out.append(f"[{s.type} : {s.value}]")
        pos = s.end
    out.append(ex.utt[pos:])
    return "".join(out)


def find_massive_file(raw_dir: str | Path, locale: str = "ar-SA") -> Path:
    try:
        return next(Path(raw_dir).rglob(f"{locale}.jsonl"))
    except StopIteration:
        raise FileNotFoundError(f"No {locale}.jsonl under {raw_dir}; run `make download`") from None


def load_massive(path: str | Path) -> list[Example]:
    examples = []
    for row in read_jsonl(path):
        utt, slots = parse_annotated(row["annot_utt"])
        meta = {"massive_id": row["id"]}
        if utt != " ".join(row["utt"].split()):
            meta["orig_utt"] = row["utt"]
        examples.append(
            Example(
                id=f"massive-{row['id']}",
                utt=utt,
                intent=row["intent"],
                slots=slots,
                variety="msa",
                source="massive",
                split=row["partition"],
                scenario=row["scenario"],
                meta=meta,
            )
        )
    return examples


def schema_from_examples(examples: list[Example]) -> dict:
    return {
        "intents": sorted({e.intent for e in examples}),
        "slot_types": sorted({s.type for e in examples for s in e.slots}),
    }
