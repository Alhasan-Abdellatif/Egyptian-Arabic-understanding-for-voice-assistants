"""Export chat-format SFT data for each training mix (the `messages` format read by mlx-lm and TRL).

    python -m lahja.train.export_sft          # -> data/sft/{C,D}/{train,valid}.jsonl

Mixes:
    C   MASSIVE ar-SA train only (Saudi/MSA)
    D   MASSIVE ar-SA train + filtered Egyptian synthetic train
    Dx  like D, but each seed's MASSIVE original is dropped once it has an Egyptian rewrite,
        so no meaning appears twice (tests content diversity vs. paired MSA/Egyptian data)
C and D are trained for the same number of steps, so "D just saw more examples" is controlled
for by the step budget rather than a separate upsampled mix.
"""

from __future__ import annotations

import argparse
import random

from lahja import DATA
from lahja.data.formats import to_chat
from lahja.data.schema import Example, load_examples, write_jsonl
from lahja.models.prompts import SFT_SYSTEM

PROCESSED = DATA / "processed"
MIXES = {
    "C": {"train": ["massive_ar_train"], "valid": ["massive_ar_dev"]},
    "D": {
        "train": ["massive_ar_train", "egy_synth_train"],
        "valid": ["massive_ar_dev", "egy_synth_dev"],
    },
    "Dx": {
        "train": ["massive_ar_train", "egy_synth_train"],
        "valid": ["massive_ar_dev", "egy_synth_dev"],
        "drop_seeded": True,
    },
}


def _load(names: list[str]) -> list[Example]:
    return [e for name in names for e in load_examples(PROCESSED / f"{name}.jsonl")]


def build_mix(name: str, valid_per_source: int, seed: int = 0) -> dict[str, list[Example]]:
    spec = MIXES[name]
    rng = random.Random(seed)
    train = _load(spec["train"])
    if spec.get("drop_seeded"):
        # Every meaning appears once: keep the Egyptian rewrite, drop the MASSIVE original it came from.
        seeded = {e.meta.get("seed_id") for e in train if e.source == "synthetic"}
        train = [e for e in train if e.source != "massive" or e.id not in seeded]
    rng.shuffle(train)
    valid = []
    for source in spec["valid"]:
        exs = _load([source])
        valid += rng.sample(exs, min(valid_per_source, len(exs)))
    return {"train": train, "valid": valid}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mixes", nargs="+", default=list(MIXES))
    ap.add_argument("--valid-per-source", type=int, default=250)
    ap.add_argument("--out-dir", default=DATA / "sft")
    args = ap.parse_args()

    for name in args.mixes:
        mix = build_mix(name, args.valid_per_source)
        for split, exs in mix.items():
            write_jsonl(
                f"{args.out_dir}/{name}/{split}.jsonl", (to_chat(e, SFT_SYSTEM) for e in exs)
            )
        egy = sum(e.variety == "egy" for e in mix["train"])
        print(f"{name}: train={len(mix['train'])} (egy={egy}) valid={len(mix['valid'])}")


if __name__ == "__main__":
    main()
