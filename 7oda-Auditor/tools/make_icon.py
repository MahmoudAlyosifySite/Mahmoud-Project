#!/usr/bin/env python3
"""Draws the 7oda Auditor app icon (1024x1024, opaque PNG, no text).

An abstract mark: a rounded phone outline held by a stylised open hand, with
vibration arcs on both sides, on a deep teal gradient. Drawn at 4x and
downsampled for smooth edges.

Usage: python3 tools/make_icon.py [output.png]
Default output: Auditor/Assets.xcassets/AppIcon.appiconset/AppIcon.png
"""
import os
import sys

from PIL import Image, ImageDraw

SIZE = 1024
SCALE = 4
S = SIZE * SCALE

TOP = (8, 46, 61)
BOTTOM = (10, 110, 120)
INK = (236, 248, 246)
ACCENT = (255, 196, 92)


def gradient(size):
    img = Image.new("RGB", (size, size), TOP)
    px = img.load()
    for y in range(size):
        f = y / (size - 1)
        row = tuple(int(TOP[i] + (BOTTOM[i] - TOP[i]) * f) for i in range(3))
        for x in range(size):
            px[x, y] = row
    return img


def draw_icon():
    small = gradient(SIZE)
    img = small.resize((S, S), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    u = S / 1024.0

    def box(x0, y0, x1, y1):
        return [x0 * u, y0 * u, x1 * u, y1 * u]

    # Palm: a soft rounded shape under the phone, with four finger tips on the right
    # and a thumb on the left (abstract, not anatomical).
    d.rounded_rectangle(box(330, 470, 700, 860), radius=150 * u, fill=(20, 140, 140))
    for i, y in enumerate([450, 535, 620, 705]):
        d.rounded_rectangle(box(600, y, 760 - i * 12, y + 70), radius=35 * u, fill=(20, 140, 140))
    d.rounded_rectangle(box(250, 520, 380, 600), radius=40 * u, fill=(20, 140, 140))

    # Phone outline.
    d.rounded_rectangle(box(392, 200, 632, 700), radius=48 * u, outline=INK, width=int(26 * u))
    d.rounded_rectangle(box(470, 228, 554, 246), radius=9 * u, fill=INK)

    # Vibration arcs on both sides of the phone.
    cx, cy = 512, 450
    for k, r in enumerate([190, 255, 320]):
        w = int((22 - 4 * k) * u)
        color = ACCENT if k == 0 else INK
        d.arc(box(cx - r, cy - r, cx + r, cy + r), start=-35, end=35, fill=color, width=w)
        d.arc(box(cx - r, cy - r, cx + r, cy + r), start=145, end=215, fill=color, width=w)

    return img.resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, "..", "Auditor", "Assets.xcassets", "AppIcon.appiconset", "AppIcon.png")
    out = sys.argv[1] if len(sys.argv) > 1 else default
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    draw_icon().save(out, "PNG", optimize=True)
    print(f"wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
