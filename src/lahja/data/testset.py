"""Egyptian human test set: CSV templates for hand annotation, then validation into JSONL.

    python -m lahja.data.testset template   # data/annotation/egy_test.csv + egy_challenge.csv
    python -m lahja.data.testset validate   # -> data/processed/egy_test.jsonl, egy_challenge.jsonl

Annotators fill the `egy_annot` column in MASSIVE bracket notation, e.g.
    صحيني [date : بكرة] الساعة [time : ستة الصبح]
See docs/annotation_guide.md.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from lahja import DATA
from lahja.data.massive import parse_annotated, to_annotated
from lahja.data.normalize import normalize
from lahja.data.schema import Example, load_examples, save_examples
from lahja.models.prompts import load_schema

ANNOT_DIR = DATA / "annotation"
MAIN_CSV = ANNOT_DIR / "egy_test.csv"
CHALLENGE_CSV = ANNOT_DIR / "egy_challenge.csv"
MAIN_COLS = ["id", "intent", "scenario", "source_annot", "egy_annot", "notes"]
CHALLENGE_COLS = ["id", "intent", "category", "egy_annot", "notes"]
CATEGORIES = {"code_switch", "arabizi", "digits", "time_expression", "local_entity", "other"}


def sample_round_robin(examples: list[Example], n: int, seed: int = 0) -> list[Example]:
    """Cycle over intents so every intent is covered before any gets a second item."""
    rng = random.Random(seed)
    by_intent: dict[str, list[Example]] = {}
    for ex in sorted(examples, key=lambda e: e.id):
        by_intent.setdefault(ex.intent, []).append(ex)
    for exs in by_intent.values():
        rng.shuffle(exs)
    queues = [by_intent[k] for k in sorted(by_intent)]
    out: list[Example] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def write_templates(n_main: int, n_challenge: int, force: bool) -> None:
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MAIN_CSV, CHALLENGE_CSV):
        if path.exists() and not force:
            sys.exit(f"{path} exists (it may hold your annotations); pass --force to overwrite")

    test = load_examples(DATA / "processed/massive_ar_test.jsonl")
    with open(MAIN_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, MAIN_COLS)
        w.writeheader()
        for ex in sorted(sample_round_robin(test, n_main), key=lambda e: (e.scenario, e.intent)):
            w.writerow(
                {
                    "id": ex.id,
                    "intent": ex.intent,
                    "scenario": ex.scenario,
                    "source_annot": to_annotated(ex),
                    "egy_annot": "",
                    "notes": "",
                }
            )
    with open(CHALLENGE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, CHALLENGE_COLS)
        w.writeheader()
        for i in range(1, n_challenge + 1):
            w.writerow(
                {"id": f"ch-{i:03d}", "intent": "", "category": "", "egy_annot": "", "notes": ""}
            )
    print(f"wrote {MAIN_CSV} ({n_main} rows) and {CHALLENGE_CSV} ({n_challenge} rows)")


def validate_rows(
    rows: list[dict], schema: dict, kind: str
) -> tuple[list[Example], list[str], list[str]]:
    intents, slot_types = set(schema["intents"]), set(schema["slot_types"])
    examples, errors, warnings = [], [], []
    seen: Counter = Counter()
    for line, row in enumerate(rows, start=2):  # row 1 is the header
        annot = (row.get("egy_annot") or "").strip()
        if not annot:
            continue
        where = f"{kind} line {line} ({row['id']})"
        intent = (row.get("intent") or "").strip()
        try:
            utt, slots = parse_annotated(annot)
        except ValueError as e:
            errors.append(f"{where}: {e}")
            continue
        if intent not in intents:
            errors.append(f"{where}: unknown intent {intent!r}")
        bad = sorted({s.type for s in slots} - slot_types)
        if bad:
            errors.append(f"{where}: unknown slot types {bad}")
        if kind == "main":
            _, src_slots = parse_annotated(row["source_annot"])
            if Counter(s.type for s in slots) != Counter(s.type for s in src_slots):
                warnings.append(f"{where}: slot types differ from the source item (ok if intended)")
        elif (row.get("category") or "").strip() not in CATEGORIES:
            warnings.append(f"{where}: category should be one of {sorted(CATEGORIES)}")
        key = normalize(utt)
        seen[key] += 1
        if seen[key] == 2:
            warnings.append(f"{where}: duplicate utterance {utt!r}")
        examples.append(
            Example(
                id=f"egy-{kind}-{row['id']}",
                utt=utt,
                intent=intent,
                slots=slots,
                variety="egy",
                source="human",
                split="test",
                scenario=row.get("scenario", ""),
                meta={k: row[k] for k in ("category", "notes") if row.get(k)},
            )
        )
    return examples, errors, warnings


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate() -> None:
    schema = load_schema()
    all_errors = []
    for kind, path, out in (
        ("main", MAIN_CSV, DATA / "processed/egy_test.jsonl"),
        ("challenge", CHALLENGE_CSV, DATA / "processed/egy_challenge.jsonl"),
    ):
        rows = _read_csv(path)
        examples, errors, warnings = validate_rows(rows, schema, kind)
        for msg in warnings:
            print("WARN ", msg)
        for msg in errors:
            print("ERROR", msg)
        all_errors += errors
        n_blank = sum(not (r.get("egy_annot") or "").strip() for r in rows)
        if not errors:
            save_examples(out, examples)
        print(
            f"{kind}: {len(examples)} annotated, {n_blank} blank, {len(errors)} errors"
            + ("" if errors else f" -> {out}")
        )
    if all_errors:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--n-main", type=int, default=200)
    t.add_argument("--n-challenge", type=int, default=50)
    t.add_argument("--force", action="store_true")
    sub.add_parser("validate")
    args = ap.parse_args()
    if args.cmd == "template":
        write_templates(args.n_main, args.n_challenge, args.force)
    else:
        validate()


if __name__ == "__main__":
    main()
