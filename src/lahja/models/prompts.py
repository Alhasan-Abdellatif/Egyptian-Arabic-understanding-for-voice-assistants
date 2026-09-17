"""The NLU prompt shared by every LLM (frontier, few-shot, and fine-tuned)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from lahja import CONFIGS
from lahja.data.formats import example_target
from lahja.data.schema import Example

# Fine-tuned models learn the label set from data, so they get a short prompt instead of the
# full schema (~10x fewer tokens per training example).
SFT_SYSTEM = (
    "You are the language-understanding module of a voice assistant for Arabic speakers. "
    'Reply with JSON only: {"intent": "<intent>", "slots": {"<slot_type>": "<value>"}}.'
)


def load_schema(path: str | Path = CONFIGS / "schema.json") -> dict:
    return json.loads(Path(path).read_text())


def build_system_prompt(schema: dict) -> str:
    return (
        "You are the language-understanding module of a voice assistant for Arabic speakers.\n"
        "Convert the user's request into one intent and its slots. Reply with JSON only, in the "
        'form {"intent": "<intent>", "slots": {"<slot_type>": "<value>"}}.\n'
        "Slot values must be copied exactly from the request. Use {} when there are no slots.\n\n"
        f"Intents: {', '.join(schema['intents'])}\n"
        f"Slot types: {', '.join(schema['slot_types'])}"
    )


def select_shots(pool: list[Example], k: int, seed: int = 0) -> list[Example]:
    """k examples with distinct intents, preferring ones that have slots."""
    rng = random.Random(seed)
    pool = sorted(pool, key=lambda e: e.id)
    rng.shuffle(pool)
    pool.sort(key=lambda e: not e.slots)  # stable: slotted examples first, shuffled within
    shots, intents = [], set()
    for ex in pool:
        if ex.intent not in intents:
            shots.append(ex)
            intents.add(ex.intent)
        if len(shots) == k:
            break
    return shots


def build_messages(utt: str, system: str, shots: list[Example] = ()) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for ex in shots:
        messages.append({"role": "user", "content": ex.utt})
        messages.append({"role": "assistant", "content": example_target(ex)})
    messages.append({"role": "user", "content": utt})
    return messages
