"""English text normalisation: numbers, years, dates, times, currency, percentages,
abbreviations and symbols to words (American reading: "one hundred five", "twenty twenty-six").

Own implementation (no third-party dependency); tests cross-check the number speller
against num2words, which is used in tests only.
"""
from __future__ import annotations

import re

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
             "nine": "ninth", "twelve": "twelfth"}


def _below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    out = []
    if h:
        out.append(f"{ONES[h]} hundred")
    if r:
        if r < 20:
            out.append(ONES[r])
        else:
            t, u = divmod(r, 10)
            out.append(TENS[t] + (f"-{ONES[u]}" if u else ""))
    return " ".join(out)


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    if n == 0:
        return "zero"
    out = []
    for size, name in SCALES:
        if n >= size:
            c, n = divmod(n, size)
            out.append(f"{cardinal(c)} {name}")
    if n:
        out.append(_below_1000(n))
    return " ".join(out)


def ordinal(n: int) -> str:
    words = cardinal(n)
    head, sep, last = words.rpartition(" ")
    pre, dash, tail = last.rpartition("-")
    if tail in ORD_IRREG:
        tail = ORD_IRREG[tail]
    elif tail.endswith("y"):
        tail = tail[:-1] + "ieth"
    else:
        tail += "th"
    return head + sep + pre + dash + tail


def year(n: int) -> str:
    """1998 -> nineteen ninety-eight; 2005 -> two thousand five; 2026 -> twenty twenty-six."""
    if 2000 <= n <= 2009 or n % 1000 == 0 or n < 1000 or n > 9999:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    if lo < 10:
        return f"{cardinal(hi)} oh {ONES[lo]}"
    return f"{cardinal(hi)} {cardinal(lo)}"


def digits_spoken(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def decimal(ip: str, fp: str) -> str:
    return f"{cardinal(int(ip or 0))} point {digits_spoken(fp)}"


def time_words(h: int, m: int, ampm: str | None) -> str:
    hw = cardinal(h % 12 or 12) if ampm else cardinal(h)
    if m == 0:
        w = hw if ampm else f"{hw} o'clock"
    elif m < 10:
        w = f"{hw} oh {ONES[m]}"
    else:
        w = f"{hw} {cardinal(m)}"
    if ampm:
        w += " a m" if ampm.lower().startswith("a") else " p m"
    return w


CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "€": ("euro", "euros", "cent", "cents"),
            "£": ("pound", "pounds", "penny", "pence"),
            "EGP": ("Egyptian pound", "Egyptian pounds", "piaster", "piasters"),
            "LE": ("Egyptian pound", "Egyptian pounds", "piaster", "piasters"),
            "USD": ("dollar", "dollars", "cent", "cents"), "EUR": ("euro", "euros", "cent", "cents")}
SCALE_WORDS = {"k": "thousand", "K": "thousand", "m": "million", "M": "million", "bn": "billion",
               "million": "million", "billion": "billion", "thousand": "thousand"}


def money(ip: str, fp: str | None, cur: str, scale: str | None = None) -> str:
    one, many, sub1, subn = CURRENCY[cur]
    if scale:
        num = decimal(ip, fp) if fp else cardinal(int(ip))
        return f"{num} {SCALE_WORDS[scale]} {many}"
    i = int(ip)
    c = int(fp.ljust(2, "0")[:2]) if fp else 0
    parts = []
    if i or not c:
        parts.append(f"{cardinal(i)} {one if i == 1 else many}")
    if c:
        parts.append(f"{cardinal(c)} {sub1 if c == 1 else subn}")
    return " and ".join(parts)


ABBREV = [(r"\bDr\.(?=\s+[A-Z])", "Doctor"), (r"\bMr\.", "Mister"), (r"\bMrs\.", "Missus"),
          (r"\bMs\.", "Miz"), (r"\bProf\.", "Professor"), (r"\bJr\.", "Junior"), (r"\bSr\.", "Senior"),
          (r"\bSt\.(?=\s+[A-Z])", "Saint"), (r"(?<=[a-z]\s)St\.", "Street"), (r"\bAve\.", "Avenue"),
          (r"\bMt\.", "Mount"), (r"\bvs\.?(?=\s)", "versus"), (r"\betc\.(?=\s*$)", "et cetera."), (r"\betc\.", "et cetera"),
          (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\bapprox\.", "approximately"),
          (r"\bNo\.\s?(?=\d)", "number "), (r"#(?=\d)", "number "), (r"\bdept\.", "department"),
          (r"\bmin\.", "minutes"), (r"\bhrs?\.", "hours")]
ABBREV = [(re.compile(p), r) for p, r in ABBREV]
UNITS = {"km/h": "kilometers per hour", "mph": "miles per hour", "km": "kilometers", "kg": "kilograms",
         "cm": "centimeters", "mm": "millimeters", "m": "meters", "g": "grams", "mg": "milligrams",
         "ml": "milliliters", "l": "liters", "lb": "pounds", "lbs": "pounds", "ft": "feet",
         "in": None, "GB": "gigabytes", "MB": "megabytes", "TB": "terabytes", "°C": "degrees Celsius",
         "°F": "degrees Fahrenheit", "°": "degrees", "kWh": "kilowatt hours", "h": "hours"}
UNIT_SING = {"kilometers": "kilometer", "kilograms": "kilogram", "meters": "meter", "grams": "gram",
             "degrees": "degree", "hours": "hour", "pounds": "pound", "liters": "liter"}


def normalize(text: str) -> str:
    t = text.replace("’", "'").replace("–", "-").replace("—", ", ")
    for rx, r in ABBREV:
        t = rx.sub(r, t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)                  # 50,000 -> 50000

    # ISO and US dates
    def _date(mo, d, y=None):
        mo, d = int(mo), int(d)
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return None
        s = f"{MONTHS[mo - 1]} {ordinal(d)}"
        return s if y is None else f"{s}, {year(int(y))}"
    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: _date(m.group(2), m.group(3), m.group(1)) or m.group(0), t)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lambda m: _date(m.group(1), m.group(2), m.group(3)) or m.group(0), t)
    # "March 15" / "March 15th" -> March fifteenth ; "15 March" -> the fifteenth of March
    mon = "|".join(MONTHS)
    t = re.sub(rf"\b({mon})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!:)", lambda m: f"{m.group(1)} {ordinal(int(m.group(2)))}", t)
    t = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon})\b", lambda m: f"the {ordinal(int(m.group(1)))} of {m.group(2)}", t)
    # years after a month / in / since ... ; decades (1990s)
    t = re.sub(rf"\b((?:{mon})(?:\s+[a-z]+(?:-[a-z]+)?)?,?|in|since|until|till|by|from|year|of|before|after|during|around|circa)\s+(1[1-9]\d\d|20\d\d)\b(?!\s*(?:%|percent))",
               lambda m: f"{m.group(1)} {year(int(m.group(2)))}", t)
    def _decade(m):
        w = year(int(m.group(1)))
        return w[:-1] + "ies" if w.endswith("y") else w + "s"
    t = re.sub(r"\b(1[1-9]\d0|20\d0)s\b", _decade, t)

    # times: 7:30, 7:30 pm, 6 am
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s?[Mm]\.?(?=\W|$)",
               lambda m: time_words(int(m.group(1)), int(m.group(2)), m.group(3) + "m"), t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: time_words(int(m.group(1)), int(m.group(2)), None)
               if int(m.group(1)) < 24 and int(m.group(2)) < 60 else m.group(0), t)
    t = re.sub(r"\b(\d{1,2})\s*([AaPp])\.?\s?[Mm]\.?(?=\W|$)", lambda m: time_words(int(m.group(1)), 0, m.group(2) + "m"), t)

    # phone numbers: (555) 123-4567, +1 555 123 4567, 0100 123 4567
    t = re.sub(r"(?<![\w.])((?:\+\d{1,3}[\s\-.])?\(?\d{2,4}\)?[\s\-.]\d{2,4}[\s\-.]\d{3,4}(?:[\s\-.]\d{2,4})?)(?!\w|\.\d)",
               lambda m: ("plus " if m.group(1).startswith("+") else "")
               + ", ".join(digits_spoken(g) for g in re.split(r"[\s\-.()]+", m.group(1).lstrip("+")) if g), t)

    # currency
    t = re.sub(r"(EGP|USD|EUR|LE|[$€£])\s?(\d+)(?:\.(\d{1,2}))?(?:\s?(k|K|M|bn|million|billion|thousand)\b)?",
               lambda m: money(m.group(2), m.group(3), m.group(1), m.group(4)), t)
    t = re.sub(r"\b(\d+)(?:\.(\d{1,2}))?\s?(EGP|USD|EUR|LE)\b", lambda m: money(m.group(1), m.group(2), m.group(3)), t)
    # percent
    t = re.sub(r"(\d+)(?:\.(\d+))?\s?%", lambda m: (decimal(m.group(1), m.group(2)) if m.group(2) else cardinal(int(m.group(1)))) + " percent", t)
    # units after numbers
    ualt = "|".join(sorted((re.escape(u) for u, v in UNITS.items() if v), key=len, reverse=True))

    def _unit(m):
        num = decimal(m.group(1), m.group(2)) if m.group(2) else cardinal(int(m.group(1)))
        u = UNITS[m.group(3)]
        if not m.group(2) and int(m.group(1)) == 1:
            u = UNIT_SING.get(u.split()[0], u.split()[0]) + (" " + " ".join(u.split()[1:]) if " " in u else "")
        return f"{num} {u}"
    t = re.sub(r"\b(\d+)(?:\.(\d+))?\s?(" + ualt + r")(?![\w/])", _unit, t)
    # ordinals 1st 2nd 3rd 4th
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)
    # ranges 3-5
    t = re.sub(r"\b(\d+)\s?-\s?(\d+)\b", lambda m: f"{cardinal(int(m.group(1)))} to {cardinal(int(m.group(2)))}", t)
    # fractions
    fr = {(1, 2): "one half", (1, 4): "one quarter", (3, 4): "three quarters", (1, 3): "one third", (2, 3): "two thirds"}
    t = re.sub(r"\b(\d)/(\d)\b", lambda m: fr.get((int(m.group(1)), int(m.group(2))), f"{cardinal(int(m.group(1)))} over {cardinal(int(m.group(2)))}"), t)
    # decimals, then integers
    t = re.sub(r"(?<![\d.])(\d*)\.(\d+)(?![\d.])", lambda m: decimal(m.group(1), m.group(2)), t)
    t = re.sub(r"\b\d{7,}\b", lambda m: digits_spoken(m.group(0)), t)
    t = re.sub(r"\d+", lambda m: cardinal(int(m.group(0))), t)
    # symbols
    for p, r in [(r"\s*&\s*", " and "), (r"\s*\+\s*", " plus "), (r"\s*=\s*", " equals "), (r"@", " at "),
                 (r"[“”\"]", ""), (r"…|\.\.\.", ", ")]:
        t = re.sub(p, r, t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(line.strip() for line in t.split("\n")).strip()
