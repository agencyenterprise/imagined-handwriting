from typing import List

DOWNLOAD_DIR = "ImaginedHandwriting/raw"


_ALL_SESSIONS = [
    "t5.2019.05.08",
    "t5.2019.11.25",
    "t5.2019.12.09",
    "t5.2019.12.11",
    "t5.2019.12.18",
    "t5.2019.12.20",
    "t5.2020.01.06",
    "t5.2020.01.08",
    "t5.2020.01.13",
    "t5.2020.01.15",
]


_CHAR_PLAIN_TEXT = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    " ",
    ",",
    "'",
    ".",
    "?",
]

_CHAR_ABBREVIATED = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    ">",
    ",",
    "'",
    "~",
    "?",
]


class SESSIONS:
    all: List[str] = _ALL_SESSIONS
    pretrain: List[str] = _ALL_SESSIONS[:3]
    copy_typing: List[str] = _ALL_SESSIONS[3:8]
    free_typing: List[str] = _ALL_SESSIONS[8:]


class CHARACTERS:
    plain_text: List[str] = _CHAR_PLAIN_TEXT
    abbreviated: List[str] = _CHAR_ABBREVIATED


SAMPLE_RATE = 100  # original data is binned to 100Hz
