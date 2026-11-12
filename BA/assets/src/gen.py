import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brandlib import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + '/'

def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{title}</title>\n<desc id="d">{desc}</desc>\n'
            + body + '\n</svg>\n')

def wave_rule(x0, x1, y, amp, color, sw, centre=None, span=None):
    """Hairline with a short voice-wave burst (decaying sine) around `centre`."""
    centre = (x0 + x1) / 2 if centre is None else centre
    span = span or amp * 12
    pts = []
    step = 2
    x = x0
    while x <= x1:
        t = (x - centre) / (span / 2)
        if abs(t) < 1:
            env = math.cos(t * math.pi / 2) ** 2
            yy = y - amp * env * math.sin(t * math.pi * 3.5)
        else:
            yy = y
        pts.append(f'{x:.1f} {yy:.1f}')
        x += step
    return (f'<path d="M{" L".join(pts)}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')

def languages(x, y, size, anchor, color, sep_color):
    """'مصري · English · Français' as outlines, laid out left-to-right."""
    gap = size * 0.55
    parts = [('مصري', 'IBMPlexSansArabic-Regular', size * 1.04),
             ('·', 'IBMPlexSans-Regular', size), ('English', 'IBMPlexSans-Regular', size),
             ('·', 'IBMPlexSans-Regular', size), ('Français', 'IBMPlexSans-Regular', size)]
    widths = [text_path(t, f, s, 0, 0)[1] for t, f, s in parts]
    total = sum(widths) + gap * (len(parts) - 1)
    cx = x - total / 2 if anchor == 'middle' else (x - total if anchor == 'end' else x)
    out = []
    for (t, f, s), wdt in zip(parts, widths):
        d, _ = text_path(t, f, s, cx, y)
        out.append(f'<path d="{d}" fill="{sep_color if t == "·" else color}"/>')
        cx += wdt + gap
    return ''.join(out), total

def lockup(x, y, u, tag_size, tag_color=C['ivory']):
    """Horizontal lockup, as in logo.svg, at scale u. (x,y) = top-left of mark box.
    Returns svg, width, height."""
    mw, mh = mark_size(u)
    body = [mark_group(x, y, u)]
    # divider: logo.svg puts it 39 units right of mark box, spanning mark box height
    dx = x + mw + 40 * u
    body.append(f'<path d="M{dx:.1f} {y - 2*u:.1f} L{dx:.1f} {y + mh + 8*u:.1f}" stroke="{C["line"]}" '
                f'stroke-width="{2*u:.2f}" stroke-linecap="round"/>')
    wx = dx + 34 * u
    wy = y - 6 * u
    body.append(wordmark(wx, wy, u))
    tb = wy + 74 * u + 40 * u
    d, tw = text_path('YOUR VOICE. YOUR IDENTITY.', 'IBMPlexSans-Medium', tag_size, wx, tb, tracking=0.16)
    body.append(f'<path d="{d}" fill="{tag_color}"/>')
    width = (wx - x) + max(118 * u + 3.5 * u, tw)
    return ''.join(body), width, tb - y, wx, tb

def banner():
    W, H = 2400, 800
    u = 2.25
    tag = 44
    # measure first
    _, lw, lh, _, _ = lockup(0, 0, u, tag)
    mw, mh = mark_size(u)
    x = (W - lw) / 2
    y = 196
    body = [f'<rect width="{W}" height="{H}" fill="{C["sapphire"]}"/>']
    lk, lw, lh, wx, tb = lockup(x, y, u, tag)
    body.append(lk)
    body.append(wave_rule(W/2 - 150, W/2 + 150, 560, 9, C['line'], 3))
    lang, _ = languages(W / 2, 640, 40, 'middle', C['muted'], C['gold'])
    body.append(lang)
    s = svg(W, H, '\n'.join(body), 'BA — Your Voice. Your Identity.',
            'BA banner: the mark (the Arabic letter beh drawn with a voice wave), the BA wordmark, '
            'the tagline, and the languages Egyptian Arabic, English, French.')
    open(OUT + 'banner.svg', 'w').write(s)

def social():
    W, H = 1200, 630
    u = 1.45
    tag = 26
    _, lw, lh, _, _ = lockup(0, 0, u, tag)
    x = (W - lw) / 2
    y = 140
    body = [f'<rect width="{W}" height="{H}" fill="{C["sapphire"]}"/>']
    lk, lw, lh, wx, tb = lockup(x, y, u, tag)
    body.append(lk)
    body.append(wave_rule(W/2 - 90, W/2 + 90, 388, 6, C['line'], 2))
    d, _ = text_path('Your own voice, with your consent.', 'IBMPlexSans-Regular', 28, W / 2, 458, anchor='middle')
    body.append(f'<path d="{d}" fill="{C["ivory"]}"/>')
    lang, _ = languages(W / 2, 512, 24, 'middle', C['muted'], C['gold'])
    body.append(lang)
    s = svg(W, H, '\n'.join(body), 'BA — Your Voice. Your Identity.',
            'Link preview card for BA: mark, wordmark, tagline, a line about consent, and the three languages.')
    open(OUT + 'social-card.svg', 'w').write(s)

if __name__ == '__main__':
    banner(); social()
