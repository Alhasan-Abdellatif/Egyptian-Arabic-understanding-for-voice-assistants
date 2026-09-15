from lahja.data.filters import check_rewrite, dedup, egy_marker_rate
from lahja.data.massive import parse_annotated
from lahja.data.schema import Example

utt, slots = parse_annotated("تكفى حط منبه [date : بكره] على [time : ستة الصبح]")
SEED = Example(id="massive-1", utt=utt, intent="alarm_set", slots=slots)


def test_accepts_good_rewrite_with_exact_spans():
    reason, (new_utt, new_slots) = check_rewrite(
        "صحيني [date : بكرة] الساعة [time : ٦ الصبح]", SEED
    )
    assert reason is None
    assert new_utt == "صحيني بكرة الساعة ٦ الصبح"
    assert all(new_utt[s.start : s.end] == s.value for s in new_slots)


def test_rejections():
    assert check_rewrite("صحيني [date : بكرة] الساعة ستة", SEED)[0] == "slot_types_changed"
    assert check_rewrite("صحيني [date : بكرة الساعة ستة", SEED)[0] == "malformed_annotation"
    copy = "تكفى حط منبه [date : بكرة] على [time : ستة الصبح]"  # only ه→ة changed
    assert check_rewrite(copy, SEED)[0] == "copy_of_seed"
    assert check_rewrite("   ", SEED)[0] == "empty"


def _ex(i, utt, intent="alarm_set"):
    return Example(id=str(i), utt=utt, intent=intent, slots=[])


def test_dedup_exact_and_near():
    exs = [
        _ex(1, "صحيني بكرة الساعة ستة الصبح لو سمحت"),
        _ex(2, "صحّيني بكرة الساعة ستة الصبح لو سمحت"),  # exact after normalization (shadda)
        _ex(3, "صحيني بكرة الساعة ستة الصبح لو سمحتي"),  # near duplicate
        _ex(4, "صحيني بكرة الساعة ستة الصبح لو سمحت", intent="alarm_query"),  # exact, other intent
        _ex(5, "شغلي حاجة لعمرو دياب", intent="play_music"),
    ]
    kept, n_exact, n_near = dedup(exs)
    assert [e.id for e in kept] == ["1", "5"]
    assert (n_exact, n_near) == (2, 1)


def test_marker_rate():
    assert egy_marker_rate(["عايز اسمع اغنية", "أريد أن أسمع أغنية"]) == 0.5
