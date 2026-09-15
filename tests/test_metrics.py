from lahja.data.massive import parse_annotated
from lahja.data.schema import Example
from lahja.eval.metrics import aggregate, parse_prediction, score_item


def gold(annot, intent="alarm_set"):
    utt, slots = parse_annotated(annot)
    return Example(id="g", utt=utt, intent=intent, slots=slots)


G = gold("صحيني [date : بكرة] الساعة [time : ٦]")


def test_parse_prediction_handles_fences_and_prose():
    raw = 'Sure:\n```json\n{"intent": "alarm_set", "slots": {"date": "بكرة"}}\n```'
    assert parse_prediction(raw) == {"intent": "alarm_set", "pairs": [("date", "بكرة")]}
    assert parse_prediction('{"intent": "alarm_set"}') == {"intent": "alarm_set", "pairs": []}
    for bad in [None, "", "no json", '{"slots": {}}', '{"intent": "x", "slots": []}', "{bad json}"]:
        assert parse_prediction(bad) is None


def test_exact_match_with_normalized_values():
    raw = '{"intent": "alarm_set", "slots": {"time": "6", "date": "بكره"}}'  # digit + ta marbuta
    item = score_item(G, raw)
    assert item["em"] == 1 and (item["tp"], item["fp"], item["fn"]) == (2, 0, 0)


def test_partial_and_invalid():
    wrong_slot = score_item(G, '{"intent": "alarm_set", "slots": {"date": "بكرة", "time": "٧"}}')
    assert (wrong_slot["em"], wrong_slot["tp"], wrong_slot["fp"], wrong_slot["fn"]) == (0, 1, 1, 1)
    wrong_intent = score_item(
        G, '{"intent": "alarm_query", "slots": {"date": "بكرة", "time": "٦"}}'
    )
    assert wrong_intent["intent_ok"] == 0 and wrong_intent["em"] == 0 and wrong_intent["tp"] == 2
    invalid = score_item(G, "oops")
    assert invalid["valid"] == 0 and invalid["fn"] == 2


def test_aggregate():
    items = [
        score_item(G, '{"intent": "alarm_set", "slots": {"date": "بكرة", "time": "٦"}}'),
        score_item(G, "oops"),
    ]
    s = aggregate(items)
    assert s["n"] == 2 and s["json_valid"] == 0.5 and s["exact_match"] == 0.5
    assert s["slot_precision"] == 1.0 and s["slot_recall"] == 0.5
    lo, hi = s["exact_match_ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_no_slots_anywhere_counts_as_perfect_f1():
    g = gold("في ايميلات جديدة", intent="email_query")
    s = aggregate([score_item(g, '{"intent": "email_query", "slots": {}}')])
    assert s["slot_f1"] == 1.0 and s["exact_match"] == 1.0
