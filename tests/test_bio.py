import re

from lahja.data.bio import bio_labels, decode_spans, tag_set
from lahja.data.massive import parse_annotated
from lahja.data.schema import Example

UTT, SLOTS = parse_annotated("صحيني [date : بكرة] الساعة [time : ستة الصبح]")
EX = Example(id="x", utt=UTT, intent="alarm_set", slots=SLOTS)
WORD_OFFSETS = [(0, 0)] + [m.span() for m in re.finditer(r"\S+", UTT)] + [(0, 0)]  # CLS ... SEP


def test_tag_set():
    assert tag_set(["date", "time"]) == ["O", "B-date", "I-date", "B-time", "I-time"]


def test_labels_skip_special_tokens():
    assert bio_labels(EX, WORD_OFFSETS) == [None, "O", "B-date", "O", "B-time", "I-time", None]


def test_decode_roundtrip_from_gold_labels():
    tags = [t or "O" for t in bio_labels(EX, WORD_OFFSETS)]
    assert decode_spans(UTT, WORD_OFFSETS, tags) == [("date", "بكرة"), ("time", "ستة الصبح")]


def test_decode_subwords_and_stray_inside_tag():
    offsets = [(6, 8), (8, 10), (11, 17), (18, 21)]
    tags = ["B-date", "I-date", "O", "I-time"]
    assert decode_spans(UTT, offsets, tags) == [("date", "بكرة"), ("time", "ستة")]


def test_decode_splits_adjacent_spans_of_different_types():
    offsets = [(6, 10), (18, 21)]
    assert decode_spans(UTT, offsets, ["B-date", "I-time"]) == [("date", "بكرة"), ("time", "ستة")]
