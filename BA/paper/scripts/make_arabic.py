"""Typeset the Arabic words used in the paper as small PDFs (XeLaTeX + Amiri, SIL OFL 1.1), so that main.tex
itself builds with pdfLaTeX. Run: python3 BA/paper/scripts/make_arabic.py [--font-dir DIR]
The font is downloaded from github.com/google/fonts (ofl/amiri) if it is not in --font-dir.
Output: BA/paper/generated/ar/<key>.pdf, included in the text with \\ar{<key>}.
"""
import argparse, os, shutil, subprocess, tempfile, urllib.request

WORDS = {
    "beh": "ب",
    "sab3a-wnoss": "سبعة ونص",
    "arba3-ayyam": "أربع أيام",
    "khamastashar": "خمستاشر",
    "tultumiyya": "تلتمية",
    "khamsata3ashar": "خمسة عشر",
    "thalathumi2a": "ثلاثمائة",
    "meeting": "ميتنج",
}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "generated", "ar")
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf"

ap = argparse.ArgumentParser()
ap.add_argument("--font-dir", default="/tmp/ba/fonts")
a = ap.parse_args()
os.makedirs(a.font_dir, exist_ok=True)
font = os.path.join(a.font_dir, "Amiri-Regular.ttf")
if not os.path.exists(font):
    urllib.request.urlretrieve(FONT_URL, font)
os.makedirs(OUT, exist_ok=True)
for key, word in WORDS.items():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "w.tex"), "w", encoding="utf-8") as f:
            f.write("\\documentclass[border=0.3pt]{standalone}\n\\usepackage{fontspec}\n"
                    f"\\newfontfamily\\arabicfont[Path={a.font_dir}/,Script=Arabic]{{Amiri-Regular.ttf}}\n"
                    "\\TeXXeTstate=1\n\\begin{document}\n"
                    # a strut fixes the height so that every word sits on the same baseline in the paper
                    f"{{\\arabicfont\\strut\\beginR {word}\\endR}}\n\\end{{document}}\n")
        subprocess.run(["xelatex", "-interaction=nonstopmode", "w.tex"], cwd=tmp, check=True, capture_output=True)
        shutil.copy(os.path.join(tmp, "w.pdf"), os.path.join(OUT, f"{key}.pdf"))
print(f"{len(WORDS)} Arabic words -> {OUT}")
