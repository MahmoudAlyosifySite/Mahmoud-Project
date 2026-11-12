# BA — brand identity

**Concept.** In ancient Egyptian thought the *ba* is the part of a person that carries their individuality; BA carries one thing of yours, your voice, and only with your consent. The mark is the Arabic letter ب (its name is pronounced *ba*) drawn as one gold line whose bowl is a voice wave, with its dot in turquoise. Egypt is present through lapis and gold, calm proportion and the letter itself, never through pyramids, pharaohs, eyes or hieroglyph patterns.

## Palette

Lapis and gold, used sparingly: the ground is sapphire, gold is the line and the one primary action, turquoise marks what is active. Everything else is ivory text on sapphire.

| Token | Hex | Role |
|---|---|---|
| Sapphire | `#14254D` | Ground. Page background; text on gold and turquoise. |
| Deep | `#0F1D3E` | Input wells, code. |
| Surface | `#1A2F5E` | Panels and blocks on the ground. |
| Line | `#2C477F` | Hairlines, dividers, the wave rule. Decorative only, never the only edge of a control. |
| Rule | `#6380B8` | Borders of inputs and secondary buttons. |
| Gold | `#D9B45A` | The mark's line, the wordmark, the primary button. One gold action per screen. |
| Gold hover | `#E4C578` | Primary button hover. |
| Turquoise | `#3FD0C9` | The mark's dot; focus rings, checked/selected state, links, progress. |
| Ivory | `#E8E3D3` | Body text, headings, tagline. |
| Muted | `#A7B1C8` | Labels, secondary text, the language line. |
| Placeholder | `#7F8BA8` | Placeholder text, inside Deep inputs only. |
| Error | `#F08A7E` | Error text and borders. |
| Paper | `#F4F1E8` | Light background, for print and light pages only. |

**Measured contrast** (WCAG 2.x relative luminance; computed, not estimated):

| Foreground on background | Ratio | Use |
|---|---|---|
| Ivory on Sapphire | 11.67 | body text — AAA |
| Ivory on Surface | 10.15 | text in panels — AAA |
| Ivory on Deep | 12.93 | text in inputs — AAA |
| Muted on Sapphire | 6.97 | secondary text — AA (AAA at large size) |
| Muted on Surface | 6.06 | labels — AA |
| Placeholder on Deep | 4.87 | placeholders — AA |
| Gold on Sapphire | 7.58 | wordmark, gold headings — AAA |
| Gold on Surface | 6.59 | AA |
| Turquoise on Sapphire | 7.90 | links, active text — AAA |
| Turquoise on Surface | 6.87 | AA |
| Sapphire on Gold | 7.58 | primary button label — AAA |
| Sapphire on Gold hover | 8.96 | AAA |
| Sapphire on Turquoise | 7.90 | tick in a checked box, selected chip — AAA |
| Error on Sapphire / Surface | 6.17 / 5.36 | AA |
| Sapphire on Paper | 13.27 | light pages — AAA |
| Rule on Deep / Surface / Sapphire | 4.20 / 3.29 / 3.79 | control borders — meets 3:1 for UI components |

Do not use: Gold on Paper (1.75) or Turquoise on Paper (1.68) for text; Line on anything (1.4–1.7) as the only boundary of a control. On light backgrounds use the single-colour sapphire mark and sapphire text.

## Typography

**IBM Plex Sans** (Latin) and **IBM Plex Sans Arabic** (Arabic). Both SIL Open Font License 1.1, both on Google Fonts. They were drawn as one family, so stroke contrast, x-height/tooth height and weights match when Arabic and English sit on the same line; the tone is engineered and calm rather than decorative, which suits a product that must feel exact and trustworthy. Plex Sans also covers French accents.

| Use | Weight | Size (screen) | Notes |
|---|---|---|---|
| Headline | Plex Sans 500 / Plex Arabic 500 | 32–40 px, line-height 1.2 | Never bold-black; let size do the work. |
| Subhead | 500 | 20–24 px, 1.3 | |
| Body | 400 | 16 px, line-height 1.6 (Arabic 1.75) | Arabic one step larger (17 px) when it stands alone. |
| UI labels, buttons | 500 (labels) / 600 (buttons) | 13–16 px | |
| Tagline / eyebrow | Plex Sans 500, UPPERCASE | 12–15 px, tracking 0.15–0.16 em | Latin only. Never track or uppercase Arabic. |
| Code | IBM Plex Mono 400 | 14 px | |

Fallback stack: `"IBM Plex Sans", "IBM Plex Sans Arabic", ui-sans-serif, system-ui, "Segoe UI", Tahoma, sans-serif` (Arabic-first contexts: put Plex Sans Arabic first). The Latin family has no Arabic glyphs, so one stack serves mixed text.

## The wave

The voice wave in the bowl of ب is the one recurring motif. Use it as:
- a **wave rule**: a hairline in Line `#2C477F` with a short decaying wave burst at its centre (as under the lockup in `banner.svg` and `social-card.svg`), to separate sections;
- a **state**: a live waveform drawn in Gold while playing or recording, Turquoise for the playhead.

Always one line weight, round caps, no fills, no gradients, no glow. At most one wave per view besides the mark. Never repeat it as wallpaper.

## The mark

- **Line weight.** The mark's stroke is 7.5 units in a 200-unit square; the wordmark's is 7 at a 74-unit cap height. Scale both together in a lockup so the line weight stays one weight.
- **Clear space.** Keep free space of at least **25 % of the mark's width** on every side (in the 200-unit square, ≥ 33 units). In a lockup, the divider sits 40 units right of the mark and the wordmark 34 units right of the divider.
- **Minimum size.** Full mark: 48 px wide on screen, 12 mm in print. Below 48 px use `favicon.svg` (heavier line, wave reduced to one cycle), which was tested at 32 and 16 px. Wordmark: 40 px wide minimum; tagline not below 11 px.
- **Colours.** Gold line + turquoise dot on Sapphire (primary). On other backgrounds use one colour: `mark-mono-light.svg` (ivory, for dark/photographic grounds) or `mark-mono-dark.svg` (sapphire, for paper, gold or light grounds).

## Do and don't

| Do | Don't |
|---|---|
| Flat sapphire ground, generous margins | Purple-blue AI gradients, glows, lens flares |
| One gold action per screen | Gold body text, or gold on light backgrounds |
| Turquoise only for focus, active, selected | Turquoise as a decorative fill |
| Real fonts through a text engine; Arabic joined and right-to-left | Hand-drawn or disconnected Arabic letters; letter-spaced Arabic |
| One line weight, round caps | Mixing stroke weights; filled or outlined mark |
| The wave as a single rule or live waveform | Pyramids, pharaoh heads, eye of Horus, ankh, hieroglyph wallpaper |
| Mark proportions as drawn | Stretching, rotating, re-colouring the dot, adding shadows |
| Microphone and audio icons from the UI kit, small | Robot, microphone or brain clip-art as illustration |

## Voice and tone

Calm, plain, exact. Short sentences. Say what happens to the person's voice and what they control; never hype ("revolutionary", "magic"). Consent is stated, not buried.

- **English.** "Your own voice, with your consent." · "Record yourself reading aloud in a quiet room." · "BA does not keep your recordings."
- **Egyptian Arabic** (colloquial, warm, not formal MSA; use ي with dots and Egyptian spellings consistently). «صوتك إنت، بموافقتك.» · «سجّل نفسك وإنت بتقرا بصوت عالي في مكان هادي.» · «BA مش بيحتفظ بتسجيلاتك.»

Copy must match what the product actually does: the Space deletes uploads after each request, so never promise an account, a stored voice or a delete button that does not exist.
- **French** (if shipped). « Votre propre voix, avec votre consentement. »

In mixed-language UI labels, put the native name first (مصري · English · Français); in a field whose first word is Arabic the line runs right-to-left.
