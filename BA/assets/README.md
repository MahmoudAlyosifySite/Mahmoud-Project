# BA assets

Identity rules (palette, contrast, type, clear space, do/don't): [`../docs/brand.md`](../docs/brand.md).

| File | Size (px) | What it is for |
|---|---|---|
| `mark.svg` | 200 × 200 | The mark (ب with a voice wave, turquoise dot) on a rounded sapphire square. Source of truth for the mark. |
| `logo.svg` | 552 × 200 | Full logo: mark, divider, BA wordmark, tagline. For README headers and documents. |
| `banner.svg` / `banner.png` | 2400 × 800 | Header image for the Hugging Face model card (legible at 800 px wide). Lockup, wave rule, مصري · English · Français. |
| `social-card.svg` / `social-card.png` | 1200 × 630 | Link preview (Open Graph / Twitter card) for the model and the Space. |
| `avatar.png` | 512 × 512 | Profile picture for the Hugging Face org/user and the Space. Full-bleed square; the artwork survives a circular crop. |
| `mark-mono-light.svg` | 150 × 98 (scalable) | One-colour mark in ivory `#E8E3D3`, transparent. For dark or photographic backgrounds. |
| `mark-mono-dark.svg` | 150 × 98 (scalable) | One-colour mark in sapphire `#14254D`, transparent. For paper, gold or other light backgrounds. |
| `favicon.svg` | 32 × 32 (scalable) | Browser tab / small icon: heavier line, wave simplified to one cycle. Checked at 32 px and 16 px. |

**Text in SVGs is converted to outlines.** Every word in these SVGs (tagline, language names, the consent line) was shaped with HarfBuzz — the shaping engine Chromium uses — from IBM Plex Sans / IBM Plex Sans Arabic (SIL OFL 1.1) and written as `<path>` outlines, so the files look identical on machines without the fonts and contain no `<text>` elements. Arabic was checked against Chromium's own rendering of the same font (identical joining and order). The PNGs were rendered from the SVGs with headless Chromium and palette-optimised.

The Gradio theme, CSS and header for the Space are in [`../space/`](../space/).
