# Asset generators

These scripts regenerate the outlined SVGs in `BA/assets/` from `mark.svg` and the IBM Plex fonts.

- `brandlib.py`: shapes text with HarfBuzz and turns glyphs into SVG path data with fontTools.
- `gen.py`: writes the banner, the social card and the logo tagline.
- `gen2.py`: writes the avatar, the single-colour marks and the favicon.

Setup:

```bash
pip install uharfbuzz fonttools
# IBM Plex Sans and IBM Plex Sans Arabic static TTFs (OFL 1.1, Google Fonts) in one folder:
export BA_FONTS_DIR=/path/to/fonts   # default /tmp/ba/fonts
python gen.py && python gen2.py
```

The PNGs were rendered from the SVGs with headless Chromium at the sizes listed in `../README.md`.

The banner and the social card list مصري · English · Français. If a language does not pass gate G3 in `BA/docs/execution-plan.md`, remove it here and regenerate both files before publishing.
