"""French text normalisation: numbers (traditional spelling: vingt et un, quatre-vingts),
ordinals (1er, 2e), dates, times (8h30), currency (15,50 €), percentages, decimals with a
comma, phone numbers read in pairs, abbreviations and symbols.

Own implementation (no third-party dependency); tests cross-check the number speller
against num2words, which is used in tests only.
"""
from __future__ import annotations

import re

UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
         "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"]
TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
          "octobre", "novembre", "décembre"]
FEM_NOUNS = set("""heure heures fois personne personnes minute minutes seconde secondes semaine
semaines année années journée journées page pages place places voiture voitures femme femmes
fille filles chose choses étoile étoiles tonne tonnes livre livres pomme pommes classe classes
chambre chambres nuit nuits""".split())


def _below_100(n: int) -> str:
    if n < 20:
        return UNITS[n]
    t, u = divmod(n, 10)
    if t in (7, 9):            # soixante-dix..., quatre-vingt-dix...
        base = "soixante" if t == 7 else "quatre-vingt"
        r = 10 + u
        if t == 7 and u == 1:
            return "soixante et onze"
        return f"{base}-{UNITS[r]}"
    if t == 8:
        return "quatre-vingts" if u == 0 else f"quatre-vingt-{UNITS[u]}"
    if u == 0:
        return TENS[t]
    if u == 1:
        return f"{TENS[t]} et un"
    return f"{TENS[t]}-{UNITS[u]}"


def _below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _below_100(r)
    if h == 1:
        return "cent" + (f" {_below_100(r)}" if r else "")
    return f"{UNITS[h]} cent" + (f" {_below_100(r)}" if r else "s")


def cardinal(n: int) -> str:
    if n < 0:
        return "moins " + cardinal(-n)
    if n < 1000:
        return _below_1000(n)
    out = []
    for size, sing, plur in ((10**9, "milliard", "milliards"), (10**6, "million", "millions")):
        if n >= size:
            c, n = divmod(n, size)
            out.append(f"{cardinal(c)} {sing if c == 1 else plur}")
    if n >= 1000:
        c, n = divmod(n, 1000)
        if c == 1:
            out.append("mille")
        else:
            w = cardinal(c)
            if w.endswith(("cents", "vingts")):    # quatre-vingt mille, deux cent mille
                w = w[:-1]
            out.append(f"{w} mille")
    if n:
        out.append(_below_1000(n))
    return " ".join(out)


def feminine(words: str) -> str:
    return words[:-2] + "une" if words.endswith("un") else words


def ordinal(n: int, fem: bool = False) -> str:
    if n == 1:
        return "première" if fem else "premier"
    w = cardinal(n)
    if w.endswith(("cents", "vingts")):
        w = w[:-1]
    if w.endswith("cinq"):
        return w + "uième"
    if w.endswith("neuf"):
        return w[:-1] + "vième"
    if w.endswith("e"):
        return w[:-1] + "ième"
    return w + "ième"


def digits_pairs(s: str) -> str:
    """French phone style: 06 12 34 56 78 -> zéro six, douze, trente-quatre, ... (odd length: first digit alone)."""
    ds = re.sub(r"\D", "", s)
    out = []
    i = 0
    if len(ds) % 2:
        out.append(UNITS[int(ds[0])]); i = 1
    while i < len(ds):
        pair = ds[i:i + 2]
        out.append(f"zéro {UNITS[int(pair[1])]}" if pair[0] == "0" else cardinal(int(pair)))
        i += 2
    return ", ".join(out)


def decimal(ip: str, fp: str) -> str:
    frac = " ".join(UNITS[int(c)] for c in fp) if fp.startswith("0") else cardinal(int(fp))
    return f"{cardinal(int(ip or 0))} virgule {frac}"


CURRENCY = {"€": ("euro", "euros", "centime", "centimes"), "EUR": ("euro", "euros", "centime", "centimes"),
            "$": ("dollar", "dollars", "cent", "cents"), "USD": ("dollar", "dollars", "cent", "cents"),
            "£": ("livre", "livres", "penny", "pence"),
            "CHF": ("franc suisse", "francs suisses", "centime", "centimes"),
            "EGP": ("livre égyptienne", "livres égyptiennes", "piastre", "piastres"),
            "euro": ("euro", "euros", "centime", "centimes"), "euros": ("euro", "euros", "centime", "centimes"),
            "dollars": ("dollar", "dollars", "cent", "cents")}


def money(ip: str, fp: str | None, cur: str) -> str:
    one, many, _, _ = CURRENCY[cur]
    i = int(ip)
    c = int(fp.ljust(2, "0")[:2]) if fp else 0
    iw = feminine(cardinal(i)) if one.startswith("livre") else cardinal(i)
    w = f"{iw} {one if i <= 1 else many}"
    if c:
        w += f" {cardinal(c)}"
    return w


ABBREV = [(r"\bM\.(?=\s+[A-ZÉ])", "Monsieur"), (r"\bMM\.(?=\s)", "Messieurs"), (r"\bMme\b\.?", "Madame"),
          (r"\bMmes\b\.?", "Mesdames"), (r"\bMlle\b\.?", "Mademoiselle"), (r"\bDr\b\.?(?=\s)", "Docteur"),
          (r"\bPr\b\.?(?=\s+[A-Z])", "Professeur"), (r"\bSte\b\.?(?=[\s-])", "Sainte"),
          (r"\bSt\b\.?(?=[\s-]+[A-Z])", "Saint"), (r"\betc\.(?=\s*$)", "et cetera."), (r"\betc\.", "et cetera"), (r"\bp\.\s?ex\.", "par exemple"),
          (r"\bc\.-à-d\.", "c'est-à-dire"), (r"\benv\.(?=\s)", "environ"), (r"\bav\.\s?J\.-C\.", "avant Jésus-Christ"),
          (r"\bapr\.\s?J\.-C\.", "après Jésus-Christ"), (r"\bbd\b\.?(?=\s)", "boulevard"),
          (r"\bn[°o]\s?(?=\d)", "numéro "), (r"\bN°\s?(?=\d)", "numéro ")]
ABBREV = [(re.compile(p), r) for p, r in ABBREV]
UNIT = {"km/h": "kilomètres heure", "km": "kilomètres", "kg": "kilos", "cm": "centimètres",
        "mm": "millimètres", "m": "mètres", "g": "grammes", "mg": "milligrammes", "ml": "millilitres",
        "l": "litres", "L": "litres", "°C": "degrés", "°": "degrés", "min": "minutes", "s": "secondes",
        "Go": "gigaoctets", "Mo": "mégaoctets"}
UNIT_SING = {"kilomètres": "kilomètre", "kilos": "kilo", "mètres": "mètre", "degrés": "degré",
             "minutes": "minute", "secondes": "seconde", "litres": "litre", "grammes": "gramme"}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
         "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
         "XIX": 19, "XX": 20, "XXI": 21}


def normalize(text: str) -> str:
    t = text.replace("’", "'").replace(" ", " ").replace(" ", " ").replace("–", "-")
    for rx, r in ABBREV:
        t = rx.sub(r, t)
    # thousands: 50 000 / 50.000 -> 50000 (not decimals: 3,5)
    t = re.sub(r"(?<![\d,])(\d{1,3})(?:[ .](\d{3}))+(?![\d,]|\.\d)", lambda m: re.sub(r"[ .]", "", m.group(0)), t)

    # phone numbers: 06 12 34 56 78, 01.23.45.67.89, +33 6 12 34 56 78
    def _phone(m):
        s = m.group(0)
        if s.startswith("+33"):
            return "plus trente-trois, " + digits_pairs(re.sub(r"\D", "", s)[2:])
        return digits_pairs(s)
    t = re.sub(r"(?<![\w+])(\+33\s?\d|0\d)(?:[\s.]\d{2}){4}(?![\w])", _phone, t)

    # dates 15/03/2026, 2026-03-15
    def _date(d, mo, y=None):
        d, mo = int(d), int(mo)
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return None
        s = f"{'premier' if d == 1 else cardinal(d)} {MONTHS[mo - 1]}"
        return s if y is None else f"{s} {cardinal(int(y))}"
    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: _date(m.group(3), m.group(2), m.group(1)) or m.group(0), t)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lambda m: _date(m.group(1), m.group(2), m.group(3)) or m.group(0), t)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})\b", lambda m: _date(m.group(1), m.group(2)) or m.group(0), t)
    mon = "|".join(MONTHS)
    t = re.sub(rf"\b1(?:er)?\s+({mon})\b", r"premier \1", t)

    # times: 8h30, 21h, 8:30, 12 h 15
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if h > 24 or mi > 59:
            return m.group(0)
        w = f"{feminine(cardinal(h))} heure{'s' if h > 1 else ''}"
        return w + (f" {cardinal(mi)}" if mi else "")
    t = re.sub(r"\b(\d{1,2})\s?h\s?(\d{2})?\b", _time, t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time, t)

    # Roman numerals before siècle: XXe siècle -> vingtième siècle
    t = re.sub(r"\b([IVX]{1,5})(?:e|ème)\s+(siècle|arrondissement)", lambda m: f"{ordinal(ROMAN[m.group(1)])} {m.group(2)}"
               if m.group(1) in ROMAN else m.group(0), t)
    # ordinals: 1er 1re 1ère 2e 2ème 2nd
    t = re.sub(r"\b1(?:er|ers)\b", "premier", t)
    t = re.sub(r"\b1(?:re|ère|res|ères)\b", "première", t)
    t = re.sub(r"\b(\d+)(?:e|ème|eme|ème)s?\b", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(r"\b2(?:nd)\b", "second", t)
    t = re.sub(r"\b2(?:nde)\b", "seconde", t)

    # currency: 15 €, 15,50 €, €15, 15 euros, 15,50 euros
    t = re.sub(r"(\d+)(?:[,.](\d{1,2}))?\s?(€|EUR|\$|USD|£|CHF|EGP)(?!\w)", lambda m: money(m.group(1), m.group(2), m.group(3)), t)
    t = re.sub(r"(€|\$|£)\s?(\d+)(?:[,.](\d{1,2}))?", lambda m: money(m.group(2), m.group(3), m.group(1)), t)
    t = re.sub(r"\b(\d+),(\d{2})\s(euros?|dollars)\b", lambda m: money(m.group(1), m.group(2), m.group(3)), t)
    # percent
    t = re.sub(r"(\d+)(?:,(\d+))?\s?%", lambda m: (decimal(m.group(1), m.group(2)) if m.group(2) else cardinal(int(m.group(1)))) + " pour cent", t)
    # units
    ualt = "|".join(sorted((re.escape(u) for u in UNIT), key=len, reverse=True))

    def _unit(m):
        num = decimal(m.group(1), m.group(2)) if m.group(2) else cardinal(int(m.group(1)))
        u = UNIT[m.group(3)]
        if not m.group(2) and int(m.group(1)) <= 1:
            u = UNIT_SING.get(u, u)
        if u in ("minute", "minutes", "seconde", "secondes"):
            num = feminine(num)
        return f"{num} {u}"
    t = re.sub(r"\b(\d+)(?:,(\d+))?\s?(" + ualt + r")(?![\w/])", _unit, t)
    # ranges 3-5 -> trois à cinq
    t = re.sub(r"\b(\d+)\s?-\s?(\d+)\b", lambda m: f"{cardinal(int(m.group(1)))} à {cardinal(int(m.group(2)))}", t)
    # decimals 3,5
    t = re.sub(r"(?<![\d,])(\d+),(\d+)(?![\d,])", lambda m: decimal(m.group(1), m.group(2)), t)
    t = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", lambda m: decimal(m.group(1), m.group(2)), t)
    # integers; feminine before a feminine noun (21 heures -> vingt et une heures)
    t = re.sub(r"\b(\d+)(?=(\s+(\w+))?)", lambda m: feminine(cardinal(int(m.group(1))))
               if (m.group(3) or "").lower() in FEM_NOUNS else cardinal(int(m.group(1))), t)
    for p, r in [(r"\s*&\s*", " et "), (r"\s*\+\s*", " plus "), (r"\s*=\s*", " égal "), (r"@", " arobase "),
                 (r"[“”\"«»]", ""), (r"…|\.\.\.", ", ")]:
        t = re.sub(p, r, t)
    t = re.sub(r"[ \t]+([,.])", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(line.strip() for line in t.split("\n")).strip()
