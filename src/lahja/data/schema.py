"""Core data types and JSONL I/O shared by every stage of the pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Slot:
    type: str
    value: str
    start: int  # char offset into Example.utt
    end: int


@dataclass
class Example:
    id: str
    utt: str
    intent: str
    slots: list[Slot]
    variety: str = "msa"  # "msa" | "egy"
    source: str = "massive"  # "massive" | "synthetic" | "human"
    split: str = "train"
    scenario: str = ""
    meta: dict = field(default_factory=dict)

    def slot_pairs(self) -> list[tuple[str, str]]:
        return [(s.type, s.value) for s in self.slots]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Example:
        d = dict(d)
        d["slots"] = [Slot(**s) for s in d.get("slots", [])]
        return cls(**d)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_examples(path: str | Path) -> list[Example]:
    return [Example.from_dict(d) for d in read_jsonl(path)]


def save_examples(path: str | Path, examples: Iterable[Example]) -> int:
    return write_jsonl(path, (e.to_dict() for e in examples))
