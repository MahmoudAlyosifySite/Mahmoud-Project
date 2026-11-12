import random
import re

import pytest

from ba_tts.text import en, fr

EN = [
    ("3 years", "three years"), ("105", "one hundred five"), ("50,000 books", "fifty thousand books"),
    ("in 1998", "in nineteen ninety-eight"), ("in 2005", "in two thousand five"), ("since 2026", "since twenty twenty-six"),
    ("the 1990s", "the nineteen nineties"), ("the 21st century", "the twenty-first century"),
    ("at 7:30 pm", "at seven thirty p m"), ("at 6:05", "at six oh five"), ("at 7:00", "at seven o'clock"),
    ("at 6 am", "at six a m"), ("$5.50", "five dollars and fifty cents"), ("$1", "one dollar"),
    ("€1,200", "one thousand two hundred euros"), ("EGP 300", "three hundred Egyptian pounds"),
    ("$2.5 million", "two point five million dollars"), ("12.5%", "twelve point five percent"),
    ("3.14", "three point one four"), ("2.5 kg", "two point five kilograms"), ("1 kg", "one kilogram"),
    ("120 km/h", "one hundred twenty kilometers per hour"), ("pages 3-5", "pages three to five"),
    ("3/4 cup", "three quarters cup"), ("No. 7", "number seven"), ("Dr. Smith", "Doctor Smith"),
    ("Mr. & Mrs. Jones", "Mister and Missus Jones"), ("March 15, 2026", "March fifteenth, twenty twenty-six"),
    ("2026-03-15", "March fifteenth, twenty twenty-six"), ("15 March", "the fifteenth of March"),
    ("Call +1 555 123 4567.", "Call plus one, five five five, one two three, four five six seven."),
    ("etc.", "et cetera."),
]

FR = [
    ("21", "vingt et un"), ("71", "soixante et onze"), ("80", "quatre-vingts"), ("81", "quatre-vingt-un"),
    ("91", "quatre-vingt-onze"), ("200", "deux cents"), ("201", "deux cent un"), ("1001", "mille un"),
    ("80 000", "quatre-vingt mille"), ("200 000 000", "deux cents millions"), ("1998", "mille neuf cent quatre-vingt-dix-huit"),
    ("à 8h30", "à huit heures trente"), ("à 21h", "à vingt et une heures"), ("1 heure", "une heure"),
    ("21 heures", "vingt et une heures"), ("15,50 €", "quinze euros cinquante"), ("15 euros", "quinze euros"),
    ("20 %", "vingt pour cent"), ("3,5", "trois virgule cinq"), ("le 1er mars", "le premier mars"),
    ("la 2e fois", "la deuxième fois"), ("le 15/03/2026", "le quinze mars deux mille vingt-six"),
    ("au XXe siècle", "au vingtième siècle"), ("5 km", "cinq kilomètres"), ("25 min", "vingt-cinq minutes"),
    ("130 km/h", "cent trente kilomètres heure"), ("pages 3-5", "pages trois à cinq"), ("n° 7", "numéro sept"),
    ("M. Dupont et Mme Martin", "Monsieur Dupont et Madame Martin"),
    ("06 12 34 56 78", "zéro six, douze, trente-quatre, cinquante-six, soixante-dix-huit"),
]


@pytest.mark.parametrize("src,expected", EN)
def test_english(src, expected):
    assert en.normalize(src) == expected


@pytest.mark.parametrize("src,expected", FR)
def test_french(src, expected):
    assert fr.normalize(src) == expected


def test_case_counts():
    assert len(EN) >= 15 and len(FR) >= 15


def _numbers():
    rng = random.Random(1)
    return list(range(0, 2000)) + [rng.randrange(0, 10**9) for _ in range(3000)]


def test_english_cardinal_matches_num2words():
    num2words = pytest.importorskip("num2words").num2words      # test oracle only (LGPL); not a runtime dependency
    for n in _numbers():
        ref = num2words(n, lang="en").replace(" and ", " ").replace(",", "")
        assert en.cardinal(n) == ref, n


def test_french_cardinal_matches_num2words():
    num2words = pytest.importorskip("num2words").num2words
    for n in _numbers():
        assert fr.cardinal(n) == num2words(n, lang="fr"), n


def test_english_ordinals():
    assert [en.ordinal(n) for n in (1, 2, 3, 4, 5, 8, 9, 12, 20, 21, 42, 100, 111)] == [
        "first", "second", "third", "fourth", "fifth", "eighth", "ninth", "twelfth", "twentieth", "twenty-first",
        "forty-second", "one hundredth", "one hundred eleventh"]


def test_eval_texts_have_no_digits_left():
    import json
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "eval", "texts")
    for name, mod in (("en.jsonl", en), ("fr.jsonl", fr)):
        for line in open(os.path.join(base, name), encoding="utf-8"):
            out = mod.normalize(json.loads(line)["text"])
            assert not re.search(r"\d", out), out
