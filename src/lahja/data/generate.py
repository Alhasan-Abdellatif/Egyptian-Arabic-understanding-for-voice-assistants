"""Generate Egyptian rewrites of MASSIVE ar-SA training seeds with Claude.

    python -m lahja.data.generate pilot --n 100     # synchronous; for eyeballing + prompt tuning
    python -m lahja.data.generate submit --n 4000   # Message Batches API (50% cheaper)
    python -m lahja.data.generate status
    python -m lahja.data.generate collect           # -> data/interim/candidates.jsonl

Rewrites are returned in MASSIVE bracket notation, so slot spans are exact by construction.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from lahja import CONFIGS, DATA
from lahja.data.massive import to_annotated
from lahja.data.schema import Example, load_examples, read_jsonl, write_jsonl

MODEL = "claude-sonnet-5" 
INTERIM = DATA / "interim"
FEWSHOT = CONFIGS / "fewshot_egy.jsonl"
TRAIN = DATA / "processed/massive_ar_train.jsonl"

INSTRUCTIONS = """\
You are a native Egyptian Arabic speaker creating training data for a phone voice assistant \
(like Siri) used in Egypt.

You receive one user command from a Saudi Arabic dataset (a mix of Saudi/Gulf colloquial and \
Modern Standard Arabic), with its intent and its slots marked inline as [slot_type : value]. \
Rewrite it as {n} different commands that an Egyptian would really say to their phone, in \
Egyptian colloquial Arabic (عامية مصرية).

Rules:
- Keep the same intent and meaning. Keep exactly the same slot types, each appearing the same \
number of times, marked inline with the same [slot_type : value] notation.
- Slot values should change to natural Egyptian wording where Egyptians would say it \
differently (بكره → بكرة, الحين → دلوقتي, غرفة النوم → أوضة النوم). Keep brackets around \
exactly the words that fill the slot, like the input does.
- Keep names of people, artists, songs, and apps. You may replace a Saudi-specific place, \
currency, or brand with an Egyptian equivalent (الرياض → القاهرة, ريال → جنيه).
- Arabic script, spoken register, no diacritics, no explanations.
- Variant 1: plain everyday Egyptian. Variant 2: noticeably different wording or word order; \
where natural it may mix in an English word Egyptians use (alarm, reminder, playlist, mail) \
or write a number in digits (٦).
- Some inputs are fragments or odd phrasings; rewrite them into what an Egyptian would say \
with the same intent, still keeping the slots.

Return JSON: {{"variants": ["<annotated command>", ...]}}.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {"type": "string"}}},
    "required": ["variants"],
    "additionalProperties": False,
}


def render_system(fewshot: list[dict], n_variants: int) -> str:
    parts = [INSTRUCTIONS.format(n=n_variants), "Examples:"]
    for ex in fewshot:
        parts.append(f"Input:\nintent: {ex['intent']}\ncommand: {ex['seed']}")
        parts.append("Output:\n" + json.dumps({"variants": ex["variants"]}, ensure_ascii=False))
    return "\n\n".join(parts)


def seed_message(ex: Example) -> str:
    return f"intent: {ex.intent}\ncommand: {to_annotated(ex)}"


def request_params(ex: Example, system: str, effort: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 4000,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": seed_message(ex)}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


def select_seeds(
    train: list[Example], n: int, seed: int = 0, min_per_intent: int = 5
) -> list[Example]:
    """Stratified by intent, proportional to intent frequency, at least min_per_intent each."""
    rng = random.Random(seed)
    by_intent: dict[str, list[Example]] = {}
    for ex in sorted(train, key=lambda e: e.id):
        by_intent.setdefault(ex.intent, []).append(ex)
    chosen = []
    for _, exs in sorted(by_intent.items()):
        k = min(len(exs), max(min_per_intent, round(n * len(exs) / len(train))))
        chosen.extend(rng.sample(exs, k))
    rng.shuffle(chosen)
    return chosen


def parse_variants(message) -> list[str]:
    if message.stop_reason != "end_turn":
        return []
    text = "".join(b.text for b in message.content if b.type == "text")
    try:
        variants = json.loads(text)["variants"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    return [v for v in variants if isinstance(v, str)]


def to_candidates(seed: Example, variants: list[str]) -> list[dict]:
    return [
        {
            "seed_id": seed.id,
            "variant": i,
            "intent": seed.intent,
            "annotated": v,
            "gen_model": MODEL,
        }
        for i, v in enumerate(variants)
    ]


def _system(n_variants: int) -> str:
    return render_system(list(read_jsonl(FEWSHOT)), n_variants)


def cmd_pilot(args, client) -> None:
    seeds = select_seeds(load_examples(TRAIN), args.n, seed=1)[: args.n]
    system = _system(args.variants)

    def one(ex):
        msg = client.beta.messages.create(
            **request_params(ex, system, args.effort),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        return ex, msg

    rows, usage = [], Counter()
    with ThreadPoolExecutor(8) as pool:
        for ex, msg in pool.map(one, seeds):
            rows.extend(to_candidates(ex, parse_variants(msg)))
            usage.update(
                input=msg.usage.input_tokens,
                output=msg.usage.output_tokens,
                cache_read=msg.usage.cache_read_input_tokens or 0,
            )
    out = INTERIM / "candidates_pilot.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} candidates from {len(seeds)} seeds -> {out}")
    print(f"tokens: {dict(usage)}")
    by_seed = {ex.id: ex for ex in seeds}
    for r in rows[: 2 * 15]:
        if r["variant"] == 0:
            print(f"\n[{r['intent']}] {to_annotated(by_seed[r['seed_id']])}")
        print(f"   -> {r['annotated']}")


def cmd_submit(args, client) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    seeds = select_seeds(load_examples(TRAIN), args.n, seed=0)
    system = _system(args.variants)
    requests = [
        Request(
            custom_id=ex.id,
            params=MessageCreateParamsNonStreaming(**request_params(ex, system, args.effort)),
        )
        for ex in seeds
    ]
    batch = client.messages.batches.create(requests=requests)
    INTERIM.mkdir(parents=True, exist_ok=True)
    (INTERIM / "batch.json").write_text(
        json.dumps({"batch_id": batch.id, "n_seeds": len(seeds)}) + "\n"
    )
    print(f"submitted batch {batch.id} with {len(seeds)} requests")


def _batch_id() -> str:
    return json.loads((INTERIM / "batch.json").read_text())["batch_id"]


def cmd_status(args, client) -> None:
    batch = client.messages.batches.retrieve(_batch_id())
    print(batch.processing_status, batch.request_counts)


def cmd_collect(args, client) -> None:
    batch_id = _batch_id()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        sys.exit(f"batch still {batch.processing_status}: {batch.request_counts}")
    seeds = {ex.id: ex for ex in load_examples(TRAIN)}
    rows, outcomes = [], Counter()
    for result in client.messages.batches.results(batch_id):
        outcomes[result.result.type] += 1
        if result.result.type == "succeeded":
            variants = parse_variants(result.result.message)
            outcomes["empty_or_unparsed"] += not variants
            rows.extend(to_candidates(seeds[result.custom_id], variants))
    out = INTERIM / "candidates.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} candidates -> {out}; outcomes: {dict(outcomes)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pilot", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=100 if name == "pilot" else 4000)
        p.add_argument("--variants", type=int, default=2)
        p.add_argument("--effort", default="medium")
    sub.add_parser("status")
    sub.add_parser("collect")
    args = ap.parse_args()

    import anthropic

    client = anthropic.Anthropic(max_retries=5)
    {"pilot": cmd_pilot, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[
        args.cmd
    ](args, client)


if __name__ == "__main__":
    main()
