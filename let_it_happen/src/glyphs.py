"""An alphabet that does not exist ("speaking in tongues").

Glyphs are built from the video's own vocabulary - discs, half discs, bars -
so they read as writing without being letters. Deterministic per seed.
"""
import numpy as np
from PIL import Image, ImageDraw
from look import MaskBody

GH = 240            # glyph height in mask pixels
GW = 170            # glyph cell width
STROKE = 44

_cache = {}


def _glyph(dr, x0, rng):
    """draw one glyph into the cell starting at x0"""
    y0, y1 = 0, GH
    kind = rng.integers(0, 9)
    s = STROKE
    cx = x0 + GW // 2
    if kind == 0:      # disc on a stem
        dr.rectangle([x0, y1 - s, x0 + GW, y1], fill=255)
        dr.ellipse([cx - 60, y0, cx + 60, y0 + 120], fill=255)
    elif kind == 1:    # half disc up + bar
        dr.pieslice([x0, y0 + 40, x0 + GW, y0 + 40 + GW], 180, 360, fill=255)
        dr.rectangle([x0, y1 - s, x0 + GW, y1], fill=255)
        dr.rectangle([cx - s // 2, y0 + 40 + GW // 2, cx + s // 2, y1], fill=255)
    elif kind == 2:    # ring
        dr.ellipse([x0, y0 + 35, x0 + GW, y0 + 35 + GW], fill=255)
        dr.ellipse([x0 + s, y0 + 35 + s, x0 + GW - s, y0 + 35 + GW - s], fill=0)
    elif kind == 3:    # two bars + dot
        dr.rectangle([x0, y0, x0 + s, y1], fill=255)
        dr.rectangle([x0 + GW - s, y0 + 70, x0 + GW, y1], fill=255)
        dr.ellipse([x0 + GW - s - 8, y0, x0 + GW + 8, y0 + s + 16], fill=255)
    elif kind == 4:    # half disc facing right on a vertical bar
        dr.rectangle([x0, y0, x0 + s, y1], fill=255)
        dr.pieslice([x0 - GW // 2 + s, y0 + 50, x0 + GW // 2 + s + 60, y0 + 50 + GW + 10], 270, 90, fill=255)
    elif kind == 5:    # stacked discs
        dr.ellipse([cx - 55, y0, cx + 55, y0 + 110], fill=255)
        dr.ellipse([cx - 70, y0 + 100, cx + 70, y1], fill=255)
    elif kind == 6:    # arch (inverted U)
        dr.pieslice([x0, y0 + 20, x0 + GW, y0 + 20 + GW], 180, 360, fill=255)
        dr.pieslice([x0 + s, y0 + 20 + s, x0 + GW - s, y0 + 20 + GW - s], 180, 360, fill=0)
        dr.rectangle([x0, y0 + 20 + GW // 2, x0 + s, y1], fill=255)
        dr.rectangle([x0 + GW - s, y0 + 20 + GW // 2, x0 + GW, y1], fill=255)
    elif kind == 7:    # slanted bar + disc
        dr.polygon([(x0, y1), (x0 + s + 10, y1), (x0 + GW, y0), (x0 + GW - s - 10, y0)], fill=255)
        dr.ellipse([x0, y0, x0 + 70, y0 + 70], fill=255)
    else:              # bowl (U)
        dr.pieslice([x0, y1 - GW - 10, x0 + GW, y1 - 10], 0, 180, fill=255)
        dr.pieslice([x0 + s, y1 - GW - 10 + s, x0 + GW - s, y1 - 10 - s], 0, 180, fill=0)
        dr.rectangle([x0, y0 + 30, x0 + s, y1 - GW // 2 - 10], fill=255)
        dr.rectangle([x0 + GW - s, y0 + 30, x0 + GW, y1 - GW // 2 - 10], fill=255)


def glyph_word(seed, n=3, height=1.3):
    key = (seed, n, height)
    if key in _cache:
        return _cache[key]
    rng = np.random.default_rng(seed * 7919 + 17)
    gap = 46
    w = n * GW + (n - 1) * gap
    im = Image.new("L", (w + 8, GH + 8), 0)
    dr = ImageDraw.Draw(im)
    for i in range(n):
        _glyph(dr, 4 + i * (GW + gap), rng)
    m = (np.asarray(im) > 127).astype(np.float32)
    body = MaskBody(m, height=height)
    _cache[key] = body
    return body


def glyph_block(seed, rows=2, n=3, height=2.4):
    """several lines of glyph writing as one body"""
    key = ("block", seed, rows, n, height)
    if key in _cache:
        return _cache[key]
    rng = np.random.default_rng(seed * 7919 + 101)
    gap, lead = 46, 90
    w = n * GW + (n - 1) * gap
    im = Image.new("L", (w + 8, rows * GH + (rows - 1) * lead + 8), 0)
    dr = ImageDraw.Draw(im)
    for r in range(rows):
        m = n - (r % 2)          # ragged lines read more like writing
        for i in range(m):
            _glyph_at(dr, 4 + i * (GW + gap), 4 + r * (GH + lead), rng)
    m = (np.asarray(im) > 127).astype(np.float32)
    body = MaskBody(m, height=height)
    _cache[key] = body
    return body


def _glyph_at(dr, x0, y0, rng):
    """draw a glyph into a cell whose top-left corner is (x0, y0)"""
    tmp = Image.new("L", (GW + 8, GH + 8), 0)
    d2 = ImageDraw.Draw(tmp)
    _glyph(d2, 4, rng)
    dr.bitmap((x0 - 4, y0 - 4), tmp, fill=255)
