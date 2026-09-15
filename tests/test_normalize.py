from lahja.data.normalize import normalize


def test_alef_ya_ta_marbuta_and_diacritics():
    assert normalize("أَحْمَد إلى الآن مدرسة") == "احمد الي الان مدرسه"


def test_digits_tatweel_punct_whitespace():
    assert normalize("الساعة ٦:٣٠ ، يـــا  Siri؟") == "الساعه 6 30 يا siri"


def test_keep_punct_when_asked():
    assert normalize("٦:٣٠", strip_punct=False) == "6:30"
