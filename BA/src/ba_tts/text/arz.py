"""Egyptian Arabic text normalisation: digits and symbols to the words Egyptians say.

Everything is written in colloquial Cairene forms (خمستاشر, تلاتين, تلتمية, ألفين وستة
وعشرين), never MSA (خمسة عشر, ثلاثين, ثلاثمائة). Covers: Arabic-Indic digits, cardinals
up to the billions, years, dates, times (7:30 -> سبعة ونص), prices and currency (جنيه,
دولار, يورو), percentages, decimals and simple fractions, digit strings read digit by
digit (phone numbers), ordinals (الدور الـ3 -> الدور التالت), counted plurals (4 أيام ->
أربع أيام), ranges, common symbols and abbreviations.

Latin words inside Arabic text (code-switching) are handled by ``latin``:
  "keep"   leave them as written (the base model reads them as English);
  "arabic" write known loanwords in Arabic script as Egyptians say them (meeting -> ميتنج),
           spell short upper-case acronyms letter by letter, leave other words in Latin.
The default is chosen by measurement (BA/eval/results/codeswitch_policy.json).
"""
from __future__ import annotations

import re

from .loanwords_arz import LETTERS, LOANWORDS

# Arabic letters only (no Arabic punctuation ، ؛ ؟, no Arabic-Indic digits, no ٪)
AR = "ء-غـ-ٰٟ-ۓ"
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# ------------------------------------------------------------------ number words
ONES = ["صفر", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة",
        "عشرة", "حداشر", "اتناشر", "تلتاشر", "أربعتاشر", "خمستاشر", "ستاشر", "سبعتاشر",
        "تمنتاشر", "تسعتاشر"]
TENS = {2: "عشرين", 3: "تلاتين", 4: "أربعين", 5: "خمسين", 6: "ستين", 7: "سبعين", 8: "تمانين",
        9: "تسعين"}
HUNDREDS = {1: "مية", 2: "ميتين", 3: "تلتمية", 4: "ربعمية", 5: "خمسمية", 6: "ستمية",
            7: "سبعمية", 8: "تمنمية", 9: "تسعمية"}
# short forms before a counted plural noun and before تلاف (3-10)
SHORT = {3: "تلات", 4: "أربع", 5: "خمس", 6: "ست", 7: "سبع", 8: "تمن", 9: "تسع", 10: "عشر"}
ORD_M = {1: "الأول", 2: "التاني", 3: "التالت", 4: "الرابع", 5: "الخامس", 6: "السادس",
         7: "السابع", 8: "التامن", 9: "التاسع", 10: "العاشر"}
ORD_F = {1: "الأولى", 2: "التانية", 3: "التالتة", 4: "الرابعة", 5: "الخامسة", 6: "السادسة",
         7: "السابعة", 8: "التامنة", 9: "التاسعة", 10: "العاشرة"}
MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر",
          "أكتوبر", "نوفمبر", "ديسمبر"]
DIGIT = ONES[:10]

PLURAL_NOUNS = set("""
أيام ايام ساعات دقايق دقائق ثواني ثوان أسابيع اسابيع شهور أشهر اشهر سنين سنوات مرات
أشخاص اشخاص أفراد افراد ناس عيال ولاد أولاد اولاد بنات رجالة ستات كتب حاجات أوض اوض غرف
أدوار ادوار تذاكر أكياس اكياس جنيهات دولارات أمتار امتار كيلومترات صفحات أسئلة اسئلة رسايل
رسائل مواضيع أطفال اطفال طلاب طلبة موظفين لاعيبة أهداف اهداف شقق عربيات بيوت محلات
فصول حلقات مواسم أجزاء اجزاء نقط نقاط درجات مراحل خطوات أنواع انواع ألوان الوان قطع حبات
""".split())
DUALS = {"يوم": "يومين", "ساعة": "ساعتين", "شهر": "شهرين", "سنة": "سنتين", "أسبوع": "أسبوعين",
         "مرة": "مرتين", "دقيقة": "دقيقتين", "جنيه": "جنيهين"}
ONE_AFTER = {"يوم": "يوم واحد", "شهر": "شهر واحد", "أسبوع": "أسبوع واحد", "ساعة": "ساعة واحدة",
             "سنة": "سنة واحدة", "مرة": "مرة واحدة", "دقيقة": "دقيقة واحدة"}
CURRENCY_WORDS = {"جنيه", "جنيهات", "دولار", "دولارات", "يورو", "ريال", "ريالات", "درهم",
                  "دينار", "قرش", "قروش", "سنت", "استرليني"}
UNIT_WORDS = CURRENCY_WORDS | {"كيلو", "متر", "سانتي", "ملي", "جرام", "لتر", "درجة"}


def _below_100(n: int) -> str:
    if n < 20:
        return ONES[n]
    t, u = divmod(n, 10)
    return TENS[t] if u == 0 else f"{ONES[u]} و{TENS[t]}"


def _below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _below_100(r)
    return HUNDREDS[h] if r == 0 else f"{HUNDREDS[h]} و{_below_100(r)}"


def cardinal(n: int) -> str:
    """Egyptian cardinal, 0 <= n < 10**12: 8500 -> تمن تلاف وخمسمية, 2000000 -> مليونين."""
    if n < 0:
        return "سالب " + cardinal(-n)
    if n < 1000:
        return _below_1000(n)
    parts = []
    for size, one, two, short_word, many in ((10**9, "مليار", "مليارين", None, "مليار"),
                                             (10**6, "مليون", "مليونين", None, "مليون"),
                                             (10**3, "ألف", "ألفين", "تلاف", "ألف")):
        if n >= size:
            c, n = divmod(n, size)
            if c == 1:
                parts.append(one)
            elif c == 2:
                parts.append(two)
            elif short_word and 3 <= c <= 10:
                parts.append(f"{SHORT[c]} {short_word}")
            else:
                parts.append(f"{cardinal(c)} {many}")
    if n:
        parts.append(_below_1000(n))
    return " و".join(parts)


def year(n: int) -> str:
    """1998 -> ألف وتسعمية تمانية وتسعين; 2026 -> ألفين وستة وعشرين."""
    if 1100 <= n <= 1999:
        h, r = divmod(n - 1000, 100)
        s = f"ألف و{HUNDREDS[h]}"
        return s if r == 0 else f"{s} {_below_100(r)}"
    return cardinal(n)


def ordinal(n: int, feminine: bool = False) -> str:
    table = ORD_F if feminine else ORD_M
    return table[n] if n in table else "ال" + cardinal(n)


def digits_spoken(s: str) -> str:
    return " ".join(DIGIT[int(c)] for c in s if c.isdigit())


def hour_word(h: int) -> str:
    h12 = h % 12 or 12
    return "واحدة" if h12 == 1 else cardinal(h12)


def time_words(h: int, m: int) -> str:
    """7:30 -> سبعة ونص; 7:45 -> تمانية إلا ربع; 7:20 -> سبعة وتلت; 7:25 -> سبعة ونص إلا خمسة."""
    cur, nxt = hour_word(h), hour_word(h + 1)
    special = {0: cur, 5: f"{cur} وخمسة", 10: f"{cur} وعشرة", 15: f"{cur} وربع", 20: f"{cur} وتلت",
               25: f"{cur} ونص إلا خمسة", 30: f"{cur} ونص", 35: f"{cur} ونص وخمسة",
               40: f"{nxt} إلا تلت", 45: f"{nxt} إلا ربع", 50: f"{nxt} إلا عشرة",
               55: f"{nxt} إلا خمسة"}
    return special.get(m) or f"{cur} و{cardinal(m)} دقيقة"


def day_period(h24: int) -> str:
    if 5 <= h24 < 12:
        return "الصبح"
    if 12 <= h24 < 15:
        return "الضهر"
    if 15 <= h24 < 18:
        return "العصر"
    return "بالليل"


def decimal_words(int_part: str, frac: str) -> str:
    """2.5 -> اتنين ونص; 0.25 -> ربع; 3.14 -> تلاتة فاصل أربعتاشر."""
    i = int(int_part) if int_part else 0
    f = frac.rstrip("0")
    base = cardinal(i)
    named = {"5": "نص", "25": "ربع", "75": "تلت تربع"}
    if f == "":
        return base
    if f in named:
        return named[f] if i == 0 else f"{base} و{named[f]}"
    fw = digits_spoken(frac) if frac.startswith("0") else cardinal(int(frac))
    return f"{base} فاصل {fw}"


FRACTIONS = {(1, 2): "نص", (1, 4): "ربع", (3, 4): "تلت تربع", (1, 3): "تلت", (2, 3): "تلتين"}

# a separated or attached proclitic before a number: "ب 25", "بـ25", "و 3" (only at word start)
PRE = r"(?:(?<![^\s(])([وبلف])ـ?\s?)?"
NEXT_WORD = rf"(?=(?:[ \t]+([{AR}]+))?)"   # peek at the next word without consuming it
PERIOD = {"ص": "AM", "صباحا": "AM", "صباحاً": "AM", "م": "PM", "مساء": "PM", "مساءً": "PM",
          "مساءا": "PM"}
PERIOD_RE = rf"(?:[ \t]*(صباحاً|صباحا|مساءً|مساءا|مساء|ص|م)(?![{AR}]))?"

UNIT_ABBR = {"ج.م": "جنيه", "ج": "جنيه", "جنيها": "جنيه", "جنيهاً": "جنيه", "EGP": "جنيه",
             "LE": "جنيه", "L.E.": "جنيه", "L.E": "جنيه",
             "كم": "كيلو", "كجم": "كيلو", "كغ": "كيلو", "km": "كيلو", "kg": "كيلو",
             "جم": "جرام", "سم": "سانتي", "cm": "سانتي", "مم": "ملي", "mm": "ملي",
             "مل": "ملي", "ml": "ملي", "USD": "دولار", "EUR": "يورو",
             "جنيه": "جنيه", "دولار": "دولار", "يورو": "يورو", "درجة": "درجة"}
CURRENCY_PREFIX = {"$": "دولار", "€": "يورو", "£": "جنيه استرليني", "E£": "جنيه", "EGP": "جنيه",
                   "USD": "دولار", "EUR": "يورو"}
SUB_UNIT = {"جنيه": "قرش", "دولار": "سنت", "يورو": "سنت", "جنيه استرليني": "بنس"}


def _money(int_part: str, frac: str | None, unit: str) -> str:
    i = int(int_part)
    f = int(frac.ljust(2, "0")[:2]) if frac else 0
    sub = SUB_UNIT.get(unit, "قرش")
    if i == 0:
        return f"{cardinal(f)} {sub}" if f else f"صفر {unit}"
    if unit == "جنيه" and i in (1, 2):
        words = "جنيه واحد" if i == 1 else "جنيهين"
    else:
        words = f"{cardinal(i)} {unit}"
    if not f:
        return words
    tail = {50: "ونص", 25: "وربع"}.get(f) or f"و{cardinal(f)} {sub}"
    return f"{words} {tail}"


def _a(pre: str | None, words: str) -> str:
    return (pre or "") + words


def _date(d, mo, y=None):
    d, mo = int(d), int(mo)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    s = f"{cardinal(d)} {MONTHS[mo - 1]}"
    return s if y is None else f"{s} {year(int(y))}"


def _phone(raw: str) -> str:
    lead = "زائد " if raw.lstrip().startswith("+") else ""
    groups = [g for g in re.split(r"[\s\-.]+", raw.strip().lstrip("+")) if g]
    if len(groups) == 1:
        g = groups[0]
        if len(g) == 11 and g.startswith("01"):   # Egyptian mobile, said 4-3-4
            groups = [g[:4], g[4:7], g[7:]]
        elif len(g) > 4:
            groups = [g[i:i + 3] for i in range(0, len(g), 3)]
    return lead + "، ".join(digits_spoken(g) for g in groups)


def normalize_numbers(text: str) -> str:
    t = text.translate(AR_DIGITS).replace("٫", ".").replace("٪", "%")
    # thousands separators: 8,500 / 1,500,000 / ٨٬٥٠٠
    t = re.sub(r"(?<![\d.,])\d{1,3}(?:[,٬]\d{3})+(?![\d,٬])",
               lambda m: m.group(0).replace(",", "").replace("٬", ""), t)

    # dates: 2026-03-15, 15/3/1998, 15-03-2026, يوم 15/3
    t = re.sub(r"(?<![\d/])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d/])",
               lambda m: _date(m.group(3), m.group(2), m.group(1)) or m.group(0), t)
    t = re.sub(r"(?<![\d/.])(\d{1,2})([/\-.])(\d{1,2})\2(\d{4})(?![\d/])",
               lambda m: _date(m.group(1), m.group(3), m.group(4)) or m.group(0), t)
    t = re.sub(r"(?<![^\s])(يوم|بتاريخ|تاريخ)\s+(\d{1,2})/(\d{1,2})(?![\d/])",
               lambda m: m.group(1) + " " + (_date(m.group(2), m.group(3)) or m.group(0)), t)

    # clock times: 7:30, 07:30 م
    def _time(m):
        pre, h, mi, per = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if h > 23 or mi > 59:
            return m.group(0)
        w = time_words(h, mi)
        if per:
            w += " " + (day_period(h if h >= 12 else h + 12) if PERIOD[per] == "PM" else "الصبح")
        elif h >= 13 or h == 0:
            w += " " + day_period(h)
        return _a(pre, w)
    t = re.sub(PRE + r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])" + PERIOD_RE, _time, t)

    # الساعة 7 -> الساعة سبعة ; الساعة 1 -> الساعة واحدة
    def _hour(m):
        h, per = int(m.group(2)), m.group(3)
        w = f"{m.group(1)} {hour_word(h)}"
        if per:
            w += " " + (day_period(h if h >= 12 else h + 12) if PERIOD[per] == "PM" else "الصبح")
        return w
    t = re.sub(r"(?<![^\s])(الساعة|الساعه)\s+(\d{1,2})(?![\d:.,]\d|\d)" + PERIOD_RE, _hour, t)

    # phone numbers and other long digit strings: read digit by digit
    t = re.sub(r"(?<![\d\w])\+\d[\d \-]{6,}\d(?![\d\w])", lambda m: _phone(m.group(0)), t)
    t = re.sub(r"(?<![\d.\w])0\d(?:[ \-]?\d){5,}(?![\d.\w])", lambda m: _phone(m.group(0)), t)
    t = re.sub(r"(?<![\d.,])\d{9,}(?![\d.,])", lambda m: _phone(m.group(0)), t)

    # percentages: 10%, 2.5 %
    t = re.sub(PRE + r"(?<![\d.])(\d+)(?:\.(\d+))?\s?%",
               lambda m: _a(m.group(1), (decimal_words(m.group(2), m.group(3)) if m.group(3)
                                         else cardinal(int(m.group(2)))) + " في المية"), t)

    # currency sign before the number: $5, € 15.50, EGP 300
    t = re.sub(PRE + r"(E£|EGP|USD|EUR|\$|€|£)\s?(\d+)(?:\.(\d{1,2}))?(?!\d)",
               lambda m: _a(m.group(1), _money(m.group(3), m.group(4), CURRENCY_PREFIX[m.group(2)])), t)

    # number + currency/unit: 350 جنيه, 25.50 ج.م, 5 كم, 30 درجة
    unit_alt = "|".join(sorted((re.escape(k) for k in UNIT_ABBR), key=len, reverse=True))

    def _num_unit(m):
        pre, ip, fp, unit = m.group(1), m.group(2), m.group(3), UNIT_ABBR[m.group(4)]
        if unit in ("جنيه", "دولار", "يورو"):
            return _a(pre, _money(ip, fp, unit))
        return _a(pre, f"{decimal_words(ip, fp) if fp else cardinal(int(ip))} {unit}")
    t = re.sub(PRE + r"(?<![\d.])(\d+)(?:\.(\d+))?\s?(" + unit_alt + rf")(?![{AR}A-Za-z])", _num_unit, t)

    # digits with a written scale word: 5 آلاف -> خمس تلاف, 2 مليون -> مليونين
    def _scale(m):
        mult = {"مليون": 10**6, "مليار": 10**9}.get(m.group(3), 1000)
        return _a(m.group(1), cardinal(int(m.group(2)) * mult))
    t = re.sub(PRE + rf"(?<![\d.])(\d{{1,3}})\s?(آلاف|ألاف|الاف|ألف|الف|مليون|مليار)(?![{AR}])", _scale, t)

    # ranges: 3-5 -> تلاتة لخمسة
    t = re.sub(r"(?<![\d.\-/:])(\d{1,4})\s?[-–]\s?(\d{1,4})(?![\d\-/:])",
               lambda m: f"{cardinal(int(m.group(1)))} ل{cardinal(int(m.group(2)))}", t)

    # fractions: 1/2 -> نص, 3/4 -> تلت تربع
    t = re.sub(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])",
               lambda m: FRACTIONS.get((int(m.group(1)), int(m.group(2))))
               or f"{cardinal(int(m.group(1)))} على {cardinal(int(m.group(2)))}", t)

    # ordinals: الدور الـ3 -> الدور التالت ; المرة الـ3 -> المرة التالتة ; الـ3 أيام -> التلات أيام
    def _ord(m):
        prev, n, nxt = m.group(1) or "", int(m.group(2)), m.group(3)
        if nxt and nxt in PLURAL_NOUNS:
            return f"{prev}ال{SHORT.get(n) or cardinal(n)}"
        fem = prev.rstrip().endswith(("ة", "ه"))
        return f"{prev}{ordinal(n, fem)}"
    t = re.sub(rf"([{AR}]+[ \t]+)?(?<![^\s])(?:الـ[ \t]?|ال)(\d{{1,3}})(?!\d)" + NEXT_WORD, _ord, t)

    # years: سنة 1998, عام 2026, and bare 1100-2099 not followed by a unit word
    t = re.sub(r"(?<![^\s])(سنة|سنه|عام)\s+(\d{3,4})(?!\d)",
               lambda m: f"{m.group(1)} {year(int(m.group(2)))}", t)
    t = re.sub(PRE + r"(?<![\d.])(1[1-9]\d\d|20\d\d)(?![\d.])" + NEXT_WORD,
               lambda m: _a(m.group(1), (cardinal if (m.group(3) in UNIT_WORDS | PLURAL_NOUNS)
                                         else year)(int(m.group(2)))), t)

    # decimals: 2.5 -> اتنين ونص
    t = re.sub(PRE + r"(?<![\d.])(\d+)\.(\d+)(?![\d.])",
               lambda m: _a(m.group(1), decimal_words(m.group(2), m.group(3))), t)

    # counted nouns: 4 أيام -> أربع أيام ; 2 يوم -> يومين ; 1 ساعة -> ساعة واحدة
    counted = "|".join(sorted(PLURAL_NOUNS | set(DUALS) | set(ONE_AFTER), key=len, reverse=True))

    def _counted(m):
        pre, n, noun = m.group(1), int(m.group(2)), m.group(3)
        if noun in PLURAL_NOUNS and 3 <= n <= 10:
            return _a(pre, f"{SHORT[n]} {noun}")
        if n == 1 and noun in ONE_AFTER:
            return _a(pre, ONE_AFTER[noun])
        if n == 2 and noun in DUALS:
            return _a(pre, DUALS[noun])
        return _a(pre, f"{cardinal(n)} {noun}")
    t = re.sub(PRE + r"(?<![\d.])(\d+)[ \t]+(" + counted + rf")(?![{AR}])", _counted, t)

    # remaining integers
    t = re.sub(PRE + r"(?<![\d.])(\d+)(?![\d.])", lambda m: _a(m.group(1), cardinal(int(m.group(2)))), t)
    return t


ABBREV = [(rf"(?<![{AR}])د\.[ \t]?(?=[{AR}])", "دكتور "),
          (rf"(?<![{AR}])أ\.[ \t]?(?=[{AR}])", "أستاذ "),
          (rf"(?<![{AR}])م\.[ \t]?(?=[{AR}])", "مهندس "),
          (rf"(?<![{AR}])ش\.[ \t]?(?=[{AR}])", "شارع "),
          (rf"(?<![{AR}])ص\.ب\.?", "صندوق بريد"),
          (rf"(?<![{AR}])ج\.م\.ع\.?", "جمهورية مصر العربية"),
          (rf"(?<![{AR}])(?:إلخ|الخ)\.?(?![{AR}])", "إلى آخره")]
ABBREV = [(re.compile(p), r) for p, r in ABBREV]

SYMBOLS = [(r"(\d)\s*[–—]\s*(\d)", r"\1-\2"),                 # keep numeric ranges for the range rule
           (r"[ \t]*&[ \t]*", " و"), (r"(?<=\d)\s*\+\s*(?=\d)", " زائد "), (r"[ \t]*=[ \t]*", " يساوي "),
           (r"(?<=\d)\s*[x×]\s*(?=\d)", " في "), (r"@", " آت "), (r"#(?=\w)", "هاشتاج "),
           (r"[“”«»\"]", ""), (r"[ \t]*[–—][ \t]*", "، "), (r"…|\.\.\.", "، "),
           (rf"(?<=\d)[ \t]*°[CcFfمس]?(?![A-Za-z{AR}])", " درجة"), (rf"[ \t]*/[ \t]*س(?![{AR}])", " في الساعة")]
SYMBOLS = [(re.compile(p), r) for p, r in SYMBOLS]


def latin_to_arabic(text: str) -> str:
    """meeting -> ميتنج; الـ link -> اللينك; PDF -> بي دي إف. Unknown lower-case words stay Latin."""
    multi = sorted((k for k in LOANWORDS if " " in k or "-" in k), key=len, reverse=True)
    if multi:
        text = re.sub(r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in multi) + r")(?![A-Za-z])",
                      lambda m: LOANWORDS[m.group(1).lower()], text, flags=re.I)

    def repl(m):
        art, w = m.group(1), m.group(2)
        k = w.lower()
        if k in LOANWORDS:
            a = LOANWORDS[k]
        elif w.isupper() and 1 < len(w) <= 5:
            a = " ".join(LETTERS[c] for c in k)
        else:
            return m.group(0)
        return (art.replace("ـ", "").strip() + a) if art else a
    art = r"((?<![^\s])(?:[وبفك]?ال|لل)ـ[ \t]?|(?<![^\s])(?:[وبفك]?ال|لل)[ \t])?"
    return re.sub(art + r"(?<![A-Za-z])([A-Za-z][A-Za-z']*)(?![A-Za-z'])", repl, text)


def normalize(text: str, latin: str = "keep") -> str:
    if latin not in ("keep", "arabic"):
        raise ValueError(f"latin must be 'keep' or 'arabic', got {latin!r}")
    t = text
    for rx, r in ABBREV:
        t = rx.sub(r, t)
    for rx, r in SYMBOLS:
        t = rx.sub(r, t)
    t = normalize_numbers(t)
    if latin == "arabic":
        t = latin_to_arabic(t)
    t = re.sub(r"[ \t]+([،,.!?؟:;؛])", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(line.strip() for line in t.split("\n")).strip()
