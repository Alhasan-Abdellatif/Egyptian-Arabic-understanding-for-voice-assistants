"""Parse MASSIVE ar-SA into per-split JSONL and write the label schema used in prompts.

python -m lahja.data.prepare
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from lahja import CONFIGS, DATA
from lahja.data.massive import find_massive_file, load_massive, schema_from_examples
from lahja.data.schema import save_examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=DATA / "raw")
    ap.add_argument("--out-dir", default=DATA / "processed")
    args = ap.parse_args()

    examples = load_massive(find_massive_file(args.raw_dir, "ar-SA"))
    by_split: dict[str, list] = {}
    for ex in examples:
        by_split.setdefault(ex.split, []).append(ex)

    for split, exs in sorted(by_split.items()):
        n = save_examples(f"{args.out_dir}/massive_ar_{split}.jsonl", exs)
        print(f"{split:>5}: {n} examples")

    schema = schema_from_examples(by_split["train"])
    CONFIGS.mkdir(exist_ok=True)
    (CONFIGS / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")

    train = by_split["train"]
    with_slots = sum(bool(e.slots) for e in train)
    changed = sum("orig_utt" in e.meta for e in examples)
    top_slots = Counter(s.type for e in train for s in e.slots).most_common(10)
    print(f"intents={len(schema['intents'])} slot_types={len(schema['slot_types'])}")
    print(f"train with >=1 slot: {with_slots / len(train):.1%}")
    print(f"utterances differing from MASSIVE `utt` after parsing: {changed}")
    print("top slot types:", ", ".join(f"{t}({n})" for t, n in top_slots))


if __name__ == "__main__":
    main()
