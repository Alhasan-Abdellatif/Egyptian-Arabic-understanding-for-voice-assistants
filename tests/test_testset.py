from lahja.data.schema import Example
from lahja.data.testset import sample_round_robin, validate_rows

SCHEMA = {"intents": ["alarm_set", "play_music"], "slot_types": ["date", "time", "artist_name"]}


def test_round_robin_covers_every_intent_first():
    exs = [Example(id=f"a{i}", utt="x", intent="alarm_set", slots=[]) for i in range(10)]
    exs += [Example(id="m0", utt="y", intent="play_music", slots=[])]
    picked = sample_round_robin(exs, 3)
    assert len(picked) == 3 and {e.intent for e in picked} == {"alarm_set", "play_music"}


def test_validate_rows():
    rows = [
        {
            "id": "massive-1",
            "intent": "alarm_set",
            "scenario": "alarm",
            "source_annot": "حط منبه [date : بكره]",
            "egy_annot": "صحيني [date : بكرة]",
            "notes": "",
        },
        {
            "id": "massive-2",
            "intent": "alarm_set",
            "scenario": "alarm",
            "source_annot": "حط منبه [date : بكره]",
            "egy_annot": "",
            "notes": "",
        },  # blank: skipped
        {
            "id": "massive-3",
            "intent": "alarm_set",
            "scenario": "alarm",
            "source_annot": "حط منبه [date : بكره]",
            "egy_annot": "صحيني [date : بكرة",
            "notes": "",
        },
        {
            "id": "massive-4",
            "intent": "play_music",
            "scenario": "music",
            "source_annot": "شغل [artist_name : عبده]",
            "egy_annot": "شغلي [singer : عبده]",
            "notes": "",
        },
        {
            "id": "massive-5",
            "intent": "alarm_set",
            "scenario": "alarm",
            "source_annot": "حط منبه [date : بكره]",
            "egy_annot": "صحيني [time : ستة]",
            "notes": "",
        },
    ]
    examples, errors, warnings = validate_rows(rows, SCHEMA, "main")
    assert [e.id for e in examples] == [
        "egy-main-massive-1",
        "egy-main-massive-4",
        "egy-main-massive-5",
    ]
    assert examples[0].utt == "صحيني بكرة" and examples[0].source == "human"
    assert len(errors) == 2  # malformed brackets on line 4, unknown slot type on line 5
    assert any("massive-5" in w and "differ" in w for w in warnings)
