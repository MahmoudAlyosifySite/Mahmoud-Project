"""Text-to-outline helper: shapes with HarfBuzz (same shaper Chromium uses),
draws glyph outlines with fontTools, returns SVG path data."""
import os, re
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FD = os.environ.get('BA_FONTS_DIR', '/tmp/ba/fonts') + '/'
_cache = {}

def _load(name):
    if name not in _cache:
        path = FD + name + '.ttf'
        blob = hb.Blob.from_file_path(path)
        face = hb.Face(blob)
        font = hb.Font(face)
        tt = TTFont(path)
        _cache[name] = (font, tt, face.upem, tt.getGlyphSet())
    return _cache[name]

def text_path(text, fontname, size, x, y, anchor='start', tracking=0.0, rtl=None):
    """Return (d, width). x,y = baseline position of anchor. tracking in em."""
    font, tt, upem, gs = _load(fontname)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    if rtl is not None:
        buf.direction = 'rtl' if rtl else 'ltr'
    hb.shape(font, buf, {'kern': True, 'liga': True})
    s = size / upem
    names = tt.getGlyphOrder()
    infos, poss = buf.glyph_infos, buf.glyph_positions
    track = tracking * upem
    total = sum(p.x_advance for p in poss) + track * max(len(poss) - 1, 0)
    if anchor == 'middle':
        x0 = x - total * s / 2
    elif anchor == 'end':
        x0 = x - total * s
    else:
        x0 = x
    pen = SVGPathPen(gs, ntos=lambda v: ('%.2f' % v).rstrip('0').rstrip('.'))
    cx = 0
    for i, (inf, pos) in enumerate(zip(infos, poss)):
        g = names[inf.codepoint]
        tx = x0 + (cx + pos.x_offset) * s
        ty = y - pos.y_offset * s
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, tx, ty)))
        cx += pos.x_advance + (track if i < len(poss) - 1 else 0)
    return pen.getCommands(), total * s

# ---- mark geometry (from BA/assets/mark.svg, unchanged) ----
_MARK = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mark.svg')).read()
MARK_D = re.search(r'<path d="([^"]+)"', _MARK).group(1)
# in mark.svg the group is translated by (0,-20); stroke 7.5; dot (100,156) r 8.5
MARK_STROKE = 7.5
MARK_DOT = (100, 156, 8.5)
# visual bounds of the mark artwork in its own (untranslated) coords, incl. stroke
MARK_BOX = (32 - 3.75, 82 - 3.75, 163 + 3.75, 156 + 8.5)  # x0,y0,x1,y1

def mark_group(tx, ty, scale, line='#D9B45A', dot='#3FD0C9', stroke=MARK_STROKE):
    """Mark artwork with its top-left visual bound placed at (tx,ty)."""
    x0, y0 = MARK_BOX[0], MARK_BOX[1]
    cx, cy, r = MARK_DOT
    return (f'<g transform="translate({tx:.2f},{ty:.2f}) scale({scale:.4f}) translate({-x0},{-y0})">'
            f'<path d="{MARK_D}" fill="none" stroke="{line}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{dot}"/></g>')

def mark_size(scale):
    return ((MARK_BOX[2] - MARK_BOX[0]) * scale, (MARK_BOX[3] - MARK_BOX[1]) * scale)

# ---- wordmark BA (from BA/assets/logo.svg, unchanged) ----
WORD_D = ['M0 0 L0 74 M0 0 L22 0 C36 0 43 8 43 18 C43 28 36 35 22 35 L0 35 M22 35 L25 35 C40 35 47 44 47 54 C47 66 39 74 25 74 L0 74',
          'M66 74 L92 0 L118 74 M75 50 L109 50']
WORD_STROKE = 7
WORD_W, WORD_H = 118, 74   # path extents (stroke adds 3.5 each side)

def wordmark(tx, ty, scale, color='#D9B45A'):
    """Wordmark with path origin (top-left of B stem centreline) at (tx,ty)."""
    return (f'<g fill="none" stroke="{color}" stroke-width="{WORD_STROKE}" stroke-linecap="round" '
            f'stroke-linejoin="round" transform="translate({tx:.2f},{ty:.2f}) scale({scale:.4f})">'
            + ''.join(f'<path d="{d}"/>' for d in WORD_D) + '</g>')

C = dict(sapphire='#14254D', gold='#D9B45A', turq='#3FD0C9', ivory='#E8E3D3',
         line='#2C477F', muted='#A7B1C8', deep='#0F1D3E', surface='#1A2F5E')
