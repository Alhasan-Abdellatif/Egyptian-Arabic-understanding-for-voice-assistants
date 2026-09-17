"""Pull real model predictions out of results/ into a markdown showcase.

    python -m lahja.eval.examples            # -> docs/examples.md

Everything here is copied verbatim from `results/<run>/egy_test/scored.jsonl`; nothing is
hand-written or cherry-picked beyond the selection rule stated in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lahja import RESULTS, ROOT
from lahja.data.schema import load_examples, read_jsonl

BEST_LLM = "D-qwen3-17b"
BEST_ENCODER = "E-D-camelbert-da"
FRONTIER = "A_sonnet5_5shot"
COLUMNS = {
    BEST_LLM: "Qwen3-1.7B + LoRA",
    BEST_ENCODER: "CAMeLBERT 110M",
    FRONTIER: "Sonnet 5 (5-shot)",
}


def predictions(run: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(RESULTS / run / "egy_test" / "scored.jsonl")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=ROOT / "docs/examples.md")
    ap.add_argument("--n-correct", type=int, default=6)
    args = ap.parse_args()

    gold = {e.id: e for e in load_examples("data/processed/egy_test.jsonl")}
    runs = {r: predictions(r) for r in COLUMNS}
    llm = runs[BEST_LLM]

    # Deterministic selection: first N (by id) that the fine-tuned LLM gets exactly right and that
    # carry at least one slot, one per intent so the sample is not all alarms.
    correct, seen = [], set()
    for i in sorted(gold):
        if llm[i]["em"] and gold[i].slots and gold[i].intent not in seen:
            correct.append(i)
            seen.add(gold[i].intent)
        if len(correct) == args.n_correct:
            break
    # Instructive disagreements: the LLM is right and the frontier model is not.
    contrast = [i for i in sorted(gold) if llm[i]["em"] and not runs[FRONTIER][i]["em"]][:4]
    # Honest failures: nothing gets these.
    failures = [i for i in sorted(gold) if not any(runs[r][i]["em"] for r in COLUMNS)][:3]

    lines = [
        "# What the model actually does",
        "",
        "Real predictions from `results/`, not illustrations. The task: turn an Egyptian Arabic",
        "voice command into a tool call — one intent plus its slots, as JSON.",
        "",
        "## Correct predictions (Qwen3-1.7B + LoRA)",
        "",
        "| Spoken Egyptian input | Model output |",
        "|---|---|",
    ]
    for i in correct:
        g = gold[i]
        out = json.dumps(  # the model's own output; em == 1 here, so it equals the gold label
            {"intent": llm[i]["pred_intent"], "slots": dict(llm[i]["pred_pairs"] or [])},
            ensure_ascii=False,
        )
        lines.append(f"| {g.utt} | `{out}` |")

    lines += [
        "",
        "## Where fine-tuning beats prompting",
        "",
        "The same sentences through the fine-tuned 1.7B and through Claude Sonnet 5 with five",
        "examples. Sonnet 5 reads the Arabic correctly; what it misses is the dataset's labelling",
        "convention — it takes greedier spans (`قائمة البقالة` where the label is `البقالة`), adds",
        "slots the schema does not mark, and occasionally picks a defensible but different intent.",
        "",
        "| Input | Expected | Qwen3-1.7B + LoRA | Sonnet 5 (5-shot) |",
        "|---|---|---|---|",
    ]
    for i in contrast:
        g = gold[i]
        exp = json.dumps({"intent": g.intent, "slots": dict(g.slot_pairs())}, ensure_ascii=False)
        ours = json.dumps(
            {"intent": llm[i]["pred_intent"], "slots": dict(llm[i]["pred_pairs"] or [])},
            ensure_ascii=False,
        )
        theirs = runs[FRONTIER][i]
        frontier = json.dumps(
            {"intent": theirs["pred_intent"], "slots": dict(theirs["pred_pairs"] or [])},
            ensure_ascii=False,
        )
        lines.append(f"| {g.utt} | `{exp}` | ✅ `{ours}` | ❌ `{frontier}` |")

    lines += [
        "",
        "## Where every model still fails",
        "",
        "| Input | Expected | Best model's output |",
        "|---|---|---|",
    ]
    for i in failures:
        g = gold[i]
        exp = json.dumps({"intent": g.intent, "slots": dict(g.slot_pairs())}, ensure_ascii=False)
        enc = runs[BEST_ENCODER][i]
        got = json.dumps(
            {"intent": enc["pred_intent"], "slots": dict(enc["pred_pairs"] or [])},
            ensure_ascii=False,
        )
        lines.append(f"| {g.utt} | `{exp}` | `{got}` |")

    lines += [
        "",
        "Failures cluster on `music` and `play`: slot values there are song, artist and playlist",
        "names that must be copied character-for-character. See [results.md](results.md) §5.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
