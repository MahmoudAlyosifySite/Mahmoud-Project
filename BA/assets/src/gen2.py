import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brandlib import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + '/'

def cr(points):
    """Catmull-Rom through points -> cubic bezier path (no M)."""
    out = []
    p = [points[0]] + points + [points[-1]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        out.append(f'C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}')
    return ' '.join(out)

# Favicon: same letter, the wave reduced to one cycle, heavier line.
FAV_PTS = [(162, 104), (146, 113), (131, 118), (116, 101), (99, 141), (84, 118), (64, 116), (48, 111), (38, 104)]
FAV_D = f'M164 80 L162 104 {cr(FAV_PTS)} L33 84'

def favicon():
    vb = '12 44 172 172'
    body = (f'<rect x="12" y="44" width="172" height="172" rx="38" fill="{C["sapphire"]}"/>'
            f'<path d="{FAV_D}" fill="none" stroke="{C["gold"]}" stroke-width="15" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="99" cy="178" r="14" fill="{C["turq"]}"/>')
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="32" height="32">'
         f'<title>BA</title>{body}</svg>\n')
    open(OUT + 'favicon.svg', 'w').write(s)

def mono(color, name, label):
    x0, y0, x1, y1 = MARK_BOX
    pad = 6
    vb = f'{x0-pad} {y0-pad} {x1-x0+2*pad} {y1-y0+2*pad}'
    cx, cy, r = MARK_DOT
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{(x1-x0+2*pad):.0f}" height="{(y1-y0+2*pad):.0f}" role="img" aria-labelledby="t d">\n'
         f'<title id="t">BA</title>\n<desc id="d">BA mark, single colour ({label}). Transparent background.</desc>\n'
         f'<path d="{MARK_D}" fill="none" stroke="{color}" stroke-width="{MARK_STROKE}" stroke-linecap="round" stroke-linejoin="round"/>\n'
         f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>\n</svg>\n')
    open(OUT + name, 'w').write(s)

def avatar():
    # mark.svg artwork, full-bleed square (platforms apply their own crop/rounding),
    # artwork centred optically and scaled so it survives a circular crop.
    mw, mh = mark_size(1)
    S = 200
    sc = 1.02
    w, h = mw * sc, mh * sc
    body = (f'<rect width="{S}" height="{S}" fill="{C["sapphire"]}"/>' + mark_group((S - w) / 2, (S - h) / 2 + 2, sc))
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="512" height="512" role="img" aria-labelledby="t">'
         f'<title id="t">BA</title>{body}</svg>\n')
    open(os.path.join(tempfile.gettempdir(), 'ba-avatar.svg'), 'w').write(s)

favicon()
mono(C['ivory'], 'mark-mono-light.svg', 'light ink #E8E3D3, for dark backgrounds')
mono(C['sapphire'], 'mark-mono-dark.svg', 'dark ink #14254D, for light backgrounds')
avatar()
