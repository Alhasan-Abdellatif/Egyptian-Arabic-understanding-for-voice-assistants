import json

import pytest

from lahja.data.formats import dict_to_pairs, example_target, slots_to_dict, to_chat, word_bio
from lahja.data.massive import parse_annotated
from lahja.data.schema import Example


def make(annot, intent="alarm_set"):
    utt, slots = parse_annotated(annot)
    return Example(id="t", utt=utt, intent=intent, slots=slots)


def test_duplicate_slot_types_roundtrip():
    pairs = [("person", "ماما"), ("date", "بكرة"), ("person", "بابا")]
    d = slots_to_dict(pairs)
    assert d == {"person": ["ماما", "بابا"], "date": "بكرة"}
    assert sorted(dict_to_pairs(d)) == sorted(pairs)


def test_dict_to_pairs_coerces_numbers_rejects_objects():
    assert dict_to_pairs({"time": 6}) == [("time", "6")]
    with pytest.raises(TypeError):
        dict_to_pairs({"time": {"h": 6}})
    with pytest.raises(TypeError):
        dict_to_pairs({"time": True})


def test_target_is_arabic_json_in_span_order():
    ex = make("صحيني [date : بكرة] [time : ٦]")
    assert json.loads(example_target(ex)) == {
        "intent": "alarm_set",
        "slots": {"date": "بكرة", "time": "٦"},
    }
    assert "بكرة" in example_target(ex)  # not \u-escaped


def test_chat_record_shape():
    rec = to_chat(make("صحيني [date : بكرة]"), "SYS")
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][1]["content"] == "صحيني بكرة"


def test_word_bio_including_clitic_word():
    ex = make("شغلي ل[artist_name : عمرو دياب] دلوقتي", intent="play_music")
    tokens, tags = word_bio(ex)
    assert tokens == ["شغلي", "لعمرو", "دياب", "دلوقتي"]
    assert tags == ["O", "B-artist_name", "I-artist_name", "O"]
