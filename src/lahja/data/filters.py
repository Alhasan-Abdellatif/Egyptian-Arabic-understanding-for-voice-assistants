"""Rule-based quality filters for LLM-generated Egyptian rewrites (no LLM judge).

Rewrites come back in MASSIVE bracket notation, so slot spans are exact by construction;
the filters check that the annotation parses, the slot types match the seed, the rewrite
is not a near-copy of the seed, and that it is not a duplicate.
"""

from __future__ import annotations

from collections import Counter

from rapidfuzz import fuzz

from lahja.data.massive import parse_annotated
from lahja.data.normalize import normalize
from lahja.data.schema import Example, Slot

# Common Egyptian function words. Used only as a reported signal, never as a filter.
EGY_MARKERS = {
    normalize(w)
    for w in (
        "عايز عاوز عايزة عاوزة ازاي ايه إيه فين امتى إمتى دلوقتي دلوقت كده كدا مش ده دي دول "
        "بتاع بتاعي بتاعتي بتاعة اوي أوي خالص بكرة النهاردة إمبارح امبارح لسه برضه بقى "
        "علشان حاجة إزيك ازيك خلي خليني عايزين ليه كام"
    ).split()
}


def check_rewrite(
    annot: str, seed: Example, copy_threshold: float = 90.0
) -> tuple[str | None, tuple[str, list[Slot]] | None]:
    """Return (rejection_reason, None) or (None, (utterance, slots))."""
    try:
        utt, slots = parse_annotated(annot)
    except ValueError:
        return "malformed_annotation", None
    if not utt:
        return "empty", None
    if Counter(s.type for s in slots) != Counter(s.type for s in seed.slots):
        return "slot_types_changed", None
    if fuzz.ratio(normalize(utt), normalize(seed.utt)) >= copy_threshold:
        return "copy_of_seed", None
    return None, (utt, slots)


def dedup(examples: list[Example], near_threshold: float = 95.0) -> tuple[list[Example], int, int]:
    """Drop exact (after normalization) and near duplicates. Near-dup check runs per intent."""
    seen: set[str] = set()
    kept_by_intent: dict[str, list[str]] = {}
    kept: list[Example] = []
    n_exact = n_near = 0
    for ex in examples:
        key = normalize(ex.utt)
        if key in seen:
            n_exact += 1
            continue
        pool = kept_by_intent.setdefault(ex.intent, [])
        if any(fuzz.ratio(key, other) >= near_threshold for other in pool):
            n_near += 1
            continue
        seen.add(key)
        pool.append(key)
        kept.append(ex)
    return kept, n_exact, n_near


def egy_marker_rate(utts: list[str]) -> float:
    if not utts:
        return 0.0
    hits = sum(bool(EGY_MARKERS & set(normalize(u).split())) for u in utts)
    return hits / len(utts)
