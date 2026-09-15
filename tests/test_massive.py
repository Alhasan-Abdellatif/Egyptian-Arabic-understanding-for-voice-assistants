import pytest

from lahja.data.massive import parse_annotated, to_annotated
from lahja.data.schema import Example


def test_parse_offsets_point_at_values():
    utt, slots = parse_annotated("صحيني [date : بكرة] الساعة [time : ٦ الصبح]")
    assert utt == "صحيني بكرة الساعة ٦ الصبح"
    assert [(s.type, s.value) for s in slots] == [("date", "بكرة"), ("time", "٦ الصبح")]
    for s in slots:
        assert utt[s.start : s.end] == s.value


def test_parse_tolerates_messy_whitespace_and_colons_in_values():
    utt, slots = parse_annotated("  اضبط  منبه [ time :  6:30 ]  ")
    assert utt == "اضبط منبه 6:30"
    assert slots[0].type == "time" and slots[0].value == "6:30"
    assert utt[slots[0].start : slots[0].end] == "6:30"


def test_parse_no_slots():
    utt, slots = parse_annotated("هو في ايميلات جديدة")
    assert utt == "هو في ايميلات جديدة" and slots == []


@pytest.mark.parametrize("bad", ["شغل [artist_name : عمرو دياب", "شغل [artist_name : ]", "شغل ] x"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_annotated(bad)


def test_roundtrip():
    annot = "الجو عامل ايه في [place_name : اسكندرية] [date : النهارده]"
    utt, slots = parse_annotated(annot)
    ex = Example(id="x", utt=utt, intent="weather_query", slots=slots)
    assert to_annotated(ex) == annot
