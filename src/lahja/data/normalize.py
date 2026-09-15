"""Light Arabic normalization, applied identically to gold labels and predictions."""

import re
import string

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_CHAR_MAP = str.maketrans(
    {
        "ـ": "",  # tatweel
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        **{chr(0x0660 + i): str(i) for i in range(10)},  # Arabic-Indic digits
        **{chr(0x06F0 + i): str(i) for i in range(10)},  # Extended Arabic-Indic digits
    }
)
_PUNCT = re.compile("[" + re.escape(string.punctuation + "؟،؛«»…“”‘’") + "]")


def normalize(text: str, strip_punct: bool = True) -> str:
    text = _DIACRITICS.sub("", text).translate(_CHAR_MAP).lower()
    if strip_punct:
        text = _PUNCT.sub(" ", text)
    return " ".join(text.split())
