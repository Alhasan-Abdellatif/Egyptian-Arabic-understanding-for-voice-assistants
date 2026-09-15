"""Filter generated candidates into the Egyptian synthetic train/dev sets.

python -m lahja.data.build_synth [--candidates data/interim/candidates.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from lahja import DATA
from lahja.data.filters import check_rewrite, dedup, egy_marker_rate
from lahja.data.schema import Example, load_examples, read_jsonl, save_examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=DATA / "interim/candidates.jsonl")
    ap.add_argument("--seeds", default=DATA / "processed/massive_ar_train.jsonl")
    ap.add_argument("--out-dir", default=DATA / "processed")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--copy-threshold", type=float, default=90.0)
    ap.add_argument("--near-dup", type=float, default=95.0)
    args = ap.parse_args()

    seeds = {e.id: e for e in load_examples(args.seeds)}
    candidates = list(read_jsonl(args.candidates))
    reasons: Counter = Counter()
    kept: list[Example] = []
    for c in candidates:
        seed = seeds[c["seed_id"]]
        reason, parsed = check_rewrite(c["annotated"], seed, args.copy_threshold)
        if reason:
            reasons[reason] += 1
            continue
        utt, slots = parsed
        kept.append(
            Example(
                id=f"egy-{c['seed_id']}-{c['variant']}",
                utt=utt,
                intent=seed.intent,
                slots=slots,
                variety="egy",
                source="synthetic",
                scenario=seed.scenario,
                meta={"seed_id": seed.id, "seed_utt": seed.utt, "gen_model": c.get("gen_model")},
            )
        )
    kept, n_exact, n_near = dedup(kept, args.near_dup)
    reasons.update(exact_duplicate=n_exact, near_duplicate=n_near)

    # Split by seed so variants of one seed never straddle train and dev.
    seed_ids = sorted({e.meta["seed_id"] for e in kept})
    random.Random(0).shuffle(seed_ids)
    dev_seeds = set(seed_ids[: round(len(seed_ids) * args.dev_frac)])
    for e in kept:
        e.split = "dev" if e.meta["seed_id"] in dev_seeds else "train"
    train = [e for e in kept if e.split == "train"]
    dev = [e for e in kept if e.split == "dev"]
    save_examples(f"{args.out_dir}/egy_synth_train.jsonl", train)
    save_examples(f"{args.out_dir}/egy_synth_dev.jsonl", dev)

    stats = {
        "candidates": len(candidates),
        "kept": len(kept),
        "train": len(train),
        "dev": len(dev),
        "rejected": dict(reasons.most_common()),
        "egy_marker_rate": round(egy_marker_rate([e.utt for e in kept]), 3),
        "seed_marker_rate": round(egy_marker_rate([s.utt for s in seeds.values()]), 3),
        "intents_covered": len({e.intent for e in kept}),
        "with_latin_script": sum(any("a" <= ch.lower() <= "z" for ch in e.utt) for e in kept),
    }
    (DATA / "processed/egy_synth_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
