"""Edit decision list: every shot of the video, placed on the measured beat grid.

Timeline (video time, the cut at 02:40.263 is 0:00):
  I    TAKE-OFF   bars   0-36   0:00.02  3-bar loop groove
  II   SKIP       bars  36-44   1:09.14  the record is stuck on one beat
  III  OCEAN      bars  44-57   1:24.50  string glides over the stuck beat
  IV   WHIRL      bars  57-76   1:49.46  build, silence 2:25.56-2:25.94
  V    RELEASE    bars  76-88   2:25.94  groove slams back
  VI   TONGUES    bars  88-112  2:48.99  wordless sampler vocals
  VII  CHOP       bars 112-124  3:35.07  gated chords
  VIII READY      bars 124-160  3:58.11  outro, lyrics from 4:20.25
"""
import math
import os
import numpy as np
from look import INK, PAPER, VERM, NAVY, VIOLET, PEACH, GOLD, hexc, TextBody, MaskBody
from glyphs import glyph_word, glyph_block

BEAT = 0.48002
PH = 0.018
PI = math.pi
DUSK = hexc("#3B1E5E")
ROSE = hexc("#E2665A")
DEEP = hexc("#07122E")


def tb(k):
    return PH + k * BEAT


def bar(b):
    return tb(4 * b)


def ease_in(u):
    return u * u * u


def ease_out(u):
    return 1 - (1 - u) ** 3


def ease_io(u):
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


SHOTS = []


def _merge(bases, p):
    out = {}
    for b in bases:
        out.update(b)
    out.update(p)
    return out


def S(k0, k1, name, *bases, **p):
    """shot from beat k0 to beat k1 (beat indices may be fractional);
    positional dicts are merged first, keyword args override them"""
    p = _merge(bases, p)
    if "whip" in name:
        p.setdefault("mblur", 6)      # fast camera drops: temporal supersampling
    elif "carried" in name:
        p.setdefault("mblur", 3)
    SHOTS.append(dict(t0=tb(k0), t1=tb(k1), name=name, p=p))


def St(t0, t1, name, *bases, **p):
    """shot in absolute seconds (for lyric-synced cuts)"""
    SHOTS.append(dict(t0=t0, t1=t1, name=name, p=_merge(bases, p)))


NIGHT = dict(bg=INK, line=PAPER, body=VERM)
# in every silence the current is gone: only the self remains, where it began
SILENCE = dict(bg=INK, body=VERM, lines_on=0.0, wake=0.0, turb=0.0, bodies=((0.0, 0.0, 1.0),),
               cam=(3.2, 0.0, 9.0, 0.0), grain=0.012, kick_pulse=0.0)
DAY = dict(bg=PAPER, line=INK, body=VERM, line_a=0.95)
RED = dict(bg=VERM, line=INK, body=PAPER, line_a=0.9)


def cam_move(a, b, fn=ease_io):
    """camera tween over the shot: a, b = (cx, cy, span, rot)"""
    return lambda c: tuple(lerp(x, y, fn(c.u)) for x, y in zip(a, b))


def whip(cam, dy, frac=0.7):
    """camera drops by dy during the last `frac` of the shot (bass glissando)"""
    def f(c):
        u = max(0.0, (c.u - (1 - frac)) / frac)
        return (cam[0], cam[1] - dy * ease_in(u), cam[2], cam[3] if len(cam) > 3 else 0.0)
    return f


def C(cx, cy, span, rot=0.0):
    return (cx, cy, span, rot)


# ============================================================================
# I  TAKE-OFF  (beats 0-144)
# The self (vermilion disc) holds its position against a fast current.
# Shot rhythm follows the 3-bar loop; density of cuts rises towards the skip.
# ============================================================================
TK = dict(NIGHT, U=1.0, wake=1.0, turb=0.35, spacing=0.11, speed=3.0)
HERO = C(3.2, 0.0, 9.0)
EDGE = C(0.3, 1.2, 2.3)
UNDER = C(0.3, -1.2, 2.3)
WAKE = C(6.5, 0.0, 4.6)
FAR = C(9.0, 0.0, 22.0)
MACRO = C(-1.02, 0.18, 0.75)
V_WIDE = C(2.6, 0.0, 9.5, PI / 2)
V_CLOSE = C(0.5, 0.0, 3.6, PI / 2)
V_WAKE = C(5.0, 0.0, 6.0, PI / 2)
V_FAR = C(7.0, 0.0, 20.0, PI / 2)


def takeoff():
    # P0: establish the world
    S(0, 4, "I hero", TK, cam=cam_move(HERO, C(3.0, 0.0, 8.6)))
    S(4, 8, "I edge", TK, cam=cam_move(EDGE, C(0.5, 1.2, 2.2)))
    S(8, 11, "I wake", TK, cam=cam_move(WAKE, C(6.9, 0, 4.6), ease_out))
    S(11, 12, "I whip", TK, cam=whip(UNDER, 1.6))
    # P1
    S(12, 16, "I far", TK, cam=cam_move(FAR, C(8.0, 0, 20.0)), spacing=0.2, width=1.3)
    S(16, 20, "I macro", TK, cam=cam_move(MACRO, C(-1.02, 0.25, 0.68)), spacing=0.028, width=1.8)
    S(20, 23, "I hero2", TK, cam=cam_move(C(1.6, 0.3, 6.0), C(2.0, 0.3, 6.2)))
    S(23, 24, "I whip2", TK, cam=whip(EDGE, 1.4))
    # P2-P3: rotate the world - the flow runs down the screen: the disc is taking off
    S(24, 28, "I vwide", TK, cam=cam_move(V_WIDE, C(2.0, 0, 8.8, PI / 2)))
    S(28, 32, "I vclose", TK, cam=cam_move(V_CLOSE, C(0.9, 0, 3.4, PI / 2)))
    S(32, 34, "I vwake", TK, cam=V_WAKE, turb=0.5)
    S(34, 35, "I vfar", TK, cam=V_FAR, spacing=0.2, width=1.3)
    S(35, 36, "I vwhip", TK, cam=lambda c: (lerp(0.5, -2.0, ease_in(c.u)), 0.0, 3.6, PI / 2))
    S(36, 40, "I vhero", TK, cam=cam_move(C(1.8, 0.0, 7.0, PI / 2), C(1.2, 0.0, 6.2, PI / 2)))
    S(40, 42, "I vedge", TK, cam=C(0.0, 1.05, 1.6, PI / 2))
    S(42, 44, "I vunder", TK, cam=C(0.0, -1.05, 1.6, PI / 2))
    S(44, 46, "I vfar2", TK, cam=C(6.0, 0.0, 16.0, PI / 2), spacing=0.17, width=1.4)
    S(46, 47, "I vmacro", TK, cam=C(-1.0, 0.0, 0.9, PI / 2), spacing=0.03)
    S(47, 48, "I vwhip2", TK, cam=lambda c: (lerp(-1.0, 1.5, ease_in(c.u)), 0.0, lerp(0.9, 4.0, ease_in(c.u)), PI / 2), spacing=0.05)
    # P4-P5: speed. light pulses ride the current ("all this noise")
    D = dict(TK, dashes=1.0)
    S(48, 50, "I dash hero", D, cam=C(3.0, 0.0, 9.0))
    S(50, 52, "I dash wake", D, cam=C(7.0, 0.4, 5.0))
    S(52, 54, "I dash far", D, cam=C(10.0, 0.0, 24.0), spacing=0.22, width=1.3)
    S(54, 56, "I dash edge", D, cam=C(0.0, 1.2, 2.6))
    S(56, 58, "I dash hero", D, cam=C(2.2, -0.4, 7.0))
    S(58, 59, "I dash under", D, cam=C(0.5, -1.3, 2.4))
    S(59, 60, "I dash whip", D, cam=whip(C(0.5, -1.3, 2.4), -2.0))
    S(60, 62, "I dash v", D, cam=C(2.4, 0.0, 9.0, PI / 2))
    S(62, 64, "I dash vc", D, cam=C(0.6, 0.0, 3.4, PI / 2))
    S(64, 66, "I dash wake2", D, cam=C(5.5, -0.3, 4.2), turb=0.55)
    S(66, 68, "I dash macro", D, cam=C(-1.02, 0.2, 0.8), spacing=0.03)
    S(68, 70, "I dash far2", D, cam=C(12.0, 0.0, 30.0), spacing=0.26, width=1.2)
    S(70, 71, "I dash hero3", D, cam=C(3.2, 0.0, 9.0))
    S(71, 72, "I dash whip2", D, cam=whip(C(3.2, 0.0, 9.0), 4.0))
    # P6-P7: days flick past. night / day alternate on every bar
    for i, k in enumerate(range(72, 96, 4)):
        pal = NIGHT if i % 2 == 0 else DAY
        cams = [HERO, C(0.2, 0.0, 4.0), C(6.0, 0.0, 6.0), C(1.0, 0.0, 12.0),
                C(2.5, 0.0, 8.0, PI / 2), C(0.0, 0.0, 3.2, PI / 2)]
        W = dict(TK, **pal)
        S(k, k + 2, f"I flip {i}a", W, cam=cams[i])
        pal2 = DAY if i % 2 == 0 else NIGHT
        W2 = dict(TK, **pal2)
        if i % 3 == 2:
            S(k + 2, k + 3, f"I flip {i}b", W2, cam=C(cams[i][0] * 0.3, 0.4, cams[i][2] * 0.6, cams[i][3]))
            S(k + 3, k + 4, f"I flip {i}c", W, cam=whip(C(0.4, -1.2, 2.4, cams[i][3]), 1.8))
        else:
            S(k + 2, k + 4, f"I flip {i}b", W2, cam=C(cams[i][0] * 0.3, 0.3, cams[i][2] * 0.55, cams[i][3]))
    # P8-P9: "I can't fight it much longer" - the wake grows, the disc shudders
    def shudder(amp):
        return lambda c: ((amp * math.sin(2 * PI * c.t / (BEAT / 2)) * (0.4 + c.kick), 0.0, 1.0),)
    ST = dict(TK, wake=1.8, turb=0.8, speed=3.8)
    S(96, 98, "I strain hero", ST, cam=C(3.0, 0.0, 9.0), bodies=shudder(0.05))
    S(98, 100, "I strain edge", ST, cam=C(0.3, 1.15, 2.0), bodies=shudder(0.05))
    S(100, 102, "I strain wake", ST, cam=C(5.0, 0.0, 5.0), bodies=shudder(0.06))
    S(102, 104, "I strain v", ST, cam=C(2.0, 0.0, 8.0, PI / 2), bodies=shudder(0.06))
    S(104, 106, "I strain vc", ST, cam=C(0.4, 0.0, 3.0, PI / 2), bodies=shudder(0.07))
    S(106, 107, "I strain far", ST, cam=C(8.0, 0.0, 20.0), spacing=0.2, bodies=shudder(0.1))
    S(107, 108, "I strain whip", ST, cam=whip(C(0.3, 1.15, 2.2), 2.4), bodies=shudder(0.07))
    S(108, 110, "I strain day", dict(ST, **DAY), cam=C(3.0, 0.0, 9.0), bodies=shudder(0.08))
    S(110, 112, "I strain day2", dict(ST, **DAY), cam=C(0.3, -1.15, 2.0), bodies=shudder(0.08))
    S(112, 114, "I strain n", ST, cam=C(2.0, 0.0, 6.0), bodies=shudder(0.09))
    S(114, 116, "I strain macro", ST, cam=C(-1.02, 0.1, 0.9), spacing=0.03, bodies=shudder(0.03))
    S(116, 118, "I strain v2", ST, cam=C(1.5, 0.0, 6.5, PI / 2), bodies=shudder(0.1))
    S(118, 119, "I strain vfar", ST, cam=C(6.0, 0.0, 18.0, PI / 2), spacing=0.2, bodies=shudder(0.12))
    S(119, 120, "I strain vwhip", ST, cam=lambda c: (lerp(0.4, -2.5, ease_in(c.u)), 0.0, 3.0, PI / 2), bodies=shudder(0.08))
    # P10-P11: a cut on every beat. palettes collide.
    seq = [
        (NIGHT, HERO), (DAY, EDGE), (NIGHT, WAKE), (RED, C(0.0, 0.0, 3.4)),
        (NIGHT, V_WIDE), (DAY, V_CLOSE), (NIGHT, FAR), (DAY, MACRO),
        (NIGHT, C(2.0, 0.0, 5.5)), (RED, C(0.2, 1.1, 2.0)), (NIGHT, C(6.0, 0.0, 5.0)), (NIGHT, C(0.3, -1.2, 2.2)),
        (DAY, HERO), (NIGHT, C(0.0, 0.0, 3.0, PI / 2)), (RED, WAKE), (NIGHT, FAR),
        (DAY, C(0.2, 1.1, 2.0)), (NIGHT, V_WIDE), (NIGHT, MACRO), (RED, C(0.0, 0.0, 4.0)),
        (NIGHT, C(3.0, 0.0, 12.0)), (DAY, C(0.5, 0.0, 3.0)), (NIGHT, EDGE),
    ]
    for i, (pal, cam) in enumerate(seq):
        k = 120 + i
        extra = {}
        if cam in (FAR,):
            extra = dict(spacing=0.2, width=1.3)
        if cam == MACRO:
            extra = dict(spacing=0.03)
        S(k, k + 1, f"I beat {i}", dict(ST, **pal), extra, cam=cam, bodies=shudder(0.1 + 0.004 * i))
    # last beat before the skip: pull back hard to the hero frame
    S(143, 144, "I last", ST, cam=lambda c: (lerp(0.5, 2.6, ease_out(c.u)), 0.0, lerp(2.5, 8.0, ease_out(c.u)), 0.0),
      bodies=shudder(0.12))


# ============================================================================
# II  SKIP  (beats 144-176): the record is stuck. So is the picture.
# One beat of video repeats 32 times, sliced into more and more time bands.
# ============================================================================
SKIP_T0 = tb(144)


def skip_frac(c):
    return min(max((c.t - SKIP_T0) / BEAT, 0.0), 1.0)


def skip_body(c):
    # inside the looped beat the disc is shoved downstream, then snaps back
    return ((0.95 * ease_in(skip_frac(c)) - 0.25, 0.0, 1.0),)


def skip_cam(c):
    # a jolt on every restart of the loop
    j = (1 - skip_frac(c)) ** 4
    return (1.1 - 0.15 * j, 0.0, 6.4 * (1 - 0.07 * j), 0.0)


def skip():
    base = dict(NIGHT, U=1.0, wake=1.8, turb=0.8, spacing=0.11, bodies=skip_body,
                cam=skip_cam, kick_pulse=0.9, speed=3.8)
    bands = {36: 1, 37: 1, 38: 2, 39: 3, 40: 4, 41: 6, 42: 9, 43: 14}
    for b, n in bands.items():
        SHOTS.append(dict(t0=bar(b), t1=bar(b + 1), name=f"II skip {n}", p=dict(base),
                          tmap=("loop", SKIP_T0, BEAT), slices=n))


# ============================================================================
# III  OCEAN  (beats 176-228): "an ocean growing inside"
# Strings glide between two chords. The disc swells and fills with waves;
# the swell of the current follows the glide.
# ============================================================================
def ocean_R(c):
    return 1.0 + 1.1 * c.glide


def ocean():
    O = dict(bg=DEEP, line=PAPER, body=VERM, body_style="ocean", U=0.35, wake=0.0, turb=0.0,
             spacing=0.12, line_a=0.75, kick_pulse=0.35)
    inner = lambda amp, sp=0.07: (lambda c: dict(amp=amp * (0.5 + 0.8 * c.glide), k=2.2, speed=0.5, spacing=sp, col=INK))
    swell = lambda base: (lambda c: base * (0.3 + 0.9 * c.glide))
    # glide up (84.5-86.4): the disc swells
    S(176, 186, "III swell", O, cam=cam_move(C(0.0, 0.0, 8.0), C(0.0, 0.0, 6.4)),
      bodies=lambda c: ((0.0, 0.0, ocean_R(c)),), inner=inner(0.18), swell=swell(0.25))
    # glide down: inside the disc - nothing but the ocean
    S(186, 195, "III inside", O, cam=cam_move(C(0.0, 0.0, 2.2), C(0.3, -0.1, 1.8)),
      bodies=((0.0, 0.0, 3.0),), inner=inner(0.16, 0.05), swell=0.0)
    # glide up: the horizon. the disc half under the surface
    S(195, 206, "III horizon", O, cam=cam_move(C(0.0, 0.6, 7.0), C(0.0, 0.9, 6.2)),
      bodies=lambda c: ((0.0, -1.2 + 0.9 * c.glide, 1.9),), inner=inner(0.2), swell=swell(0.35))
    # glide down
    S(206, 211, "III far", O, cam=C(0.0, 0.0, 14.0), spacing=0.16,
      bodies=lambda c: ((0.0, 0.0, 1.2 + 1.3 * c.glide),), inner=inner(0.2, 0.09), swell=swell(0.4))
    # glide up
    S(211, 216, "III edge", O, cam=cam_move(C(-1.6, 1.4, 3.0), C(-1.4, 1.6, 2.6)),
      bodies=lambda c: ((0.0, 0.0, 2.2),), inner=inner(0.2, 0.06), swell=swell(0.3))
    # glide down, long hold
    S(216, 225, "III drift", O, cam=cam_move(C(0.0, 0.0, 9.0), C(0.8, 0.0, 7.5)),
      bodies=lambda c: ((0.0, 0.0, ocean_R(c)),), inner=inner(0.22), swell=swell(0.4))
    # last glide up into the build
    S(225, 228, "III rise", O, cam=cam_move(C(0.0, 0.0, 5.0), C(0.0, 0.0, 2.8), ease_in),
      bodies=lambda c: ((0.0, 0.0, ocean_R(c)),), inner=inner(0.25), swell=swell(0.5))


# ============================================================================
# IV  WHIRL  (beats 228-304): "a whirlwind that's coming 'round,
#     gonna carry off all that isn't bound"
# A vortex grows; the current spirals; cuts accelerate; silence = no current.
# ============================================================================
WH_T0, WH_T1 = tb(228), tb(303.2)
VX, VY = 4.0, -0.5


def whirl_u(t):
    return min(max((t - WH_T0) / (WH_T1 - WH_T0), 0.0), 1.0)


def whirl_body(c):
    u = whirl_u(c.t)
    # late in the build the disc is torn from its place and starts to orbit
    pull = ease_in(max(0.0, (u - 0.45) / 0.55))
    ang = PI * 0.9 + 2 * PI * 0.6 * pull
    rad = lerp(4.0, 1.9, pull)
    x = lerp(0.0, VX + rad * math.cos(ang), pull)
    y = lerp(0.0, VY + rad * math.sin(ang), pull)
    return ((x, y, 1.0),)


def whirl_shot(u0, cam, line=PAPER):
    """parameters for one whirl shot; the arm count is fixed per shot (it must
    be an integer for the spiral to close), everything else is continuous"""
    n = int(round(lerp(10, 64, u0 ** 1.1) / 2)) * 2
    spacing = 0.11
    b = n * spacing / (2 * PI)
    return dict(
        bg=lambda c: tuple(lerp(np.array(DEEP), np.array(INK), whirl_u(c.t))),
        line=line, body=VERM, spacing=spacing, kick_pulse=0.8,
        U=lambda c: lerp(1.0, 0.1, whirl_u(c.t) ** 1.2),
        wake=lambda c: 0.8 * (1 - whirl_u(c.t)), turb=0.2,
        spiral=lambda c: (VX, VY, b * lerp(0.9, 1.6, whirl_u(c.t)), n),
        # the whirl turns, and twists a little further on every kick
        offset=lambda c: b * (0.6 * c.t + 1.2 * c.kick_int),
        eye=lambda c: (VX, VY, lerp(0.05, 0.28, whirl_u(c.t))),
        bodies=whirl_body, cam=cam)


def whirl():
    cams = [
        C(2.5, 0.0, 10.0), C(3.0, -0.3, 8.0), C(1.0, 0.0, 5.0), C(4.0, -0.5, 13.0),
        C(4.0, -0.5, 6.0), C(2.0, 0.3, 9.0, 0.4), C(4.0, -0.5, 11.0, -0.6), C(3.5, -0.5, 6.5, 1.2),
        C(4.0, -0.5, 3.2, 0.0), C(3.0, -1.0, 15.0, 2.0), C(3.0, 0.0, 7.0, -1.0), C(4.0, -0.5, 9.0, 3.1),
    ]
    k = 228
    i = 0
    for step, count in [(8, 4), (4, 6), (2, 4), (1, 11)]:
        for _ in range(count):
            c0 = cams[i % len(cams)]
            if step >= 4:
                cam = cam_move(c0, (c0[0], c0[1], c0[2] * 0.82, c0[3] + 0.25))
            else:
                cam = cam_move(c0, (c0[0], c0[1], c0[2] * 0.9, c0[3] + 0.25 * step))
            line = VERM if (step == 1 and i % 2 == 1) else PAPER
            S(k, k + step, f"IV whirl {i}", whirl_shot(whirl_u(tb(k)), cam, line))
            k += step
            i += 1
    # k == 303: the last beat runs into the silence
    S(303, 303.2, "IV last", whirl_shot(1.0, cam_move(C(4.0, -0.5, 6.0), C(4.0, -0.5, 5.0))))
    # silence: the current is gone. only the self remains.
    St(tb(303.2), tb(304), "IV silence", SILENCE)


# ============================================================================
# V  RELEASE  (beats 304-352): let it happen.
# Palette inverts. The disc now moves WITH the current: the lines stay
# straight - in the frame of the self, the current has disappeared.
# Poster-like frames, cut on kick "1" and "3", swap on snares.
# ============================================================================
def release():
    Rl = dict(DAY, U=1.0, wake=0.0, turb=0.0, kick_pulse=0.4, grain=0.009, obstacle=False)
    comps = [
        dict(cam=C(0.0, 0.0, 9.0), bodies=((-2.4, 1.1, 1.0),)),
        dict(cam=C(0.0, 0.0, 2.6), bodies=((0.0, 0.0, 1.0),), spacing=0.05),
        dict(cam=C(0.0, 0.0, 9.0, PI / 2), bodies=((2.0, 2.8, 0.6),)),
        dict(cam=C(0.0, 0.0, 9.0), bodies=((-5.0, -2.0, 3.2),)),
        dict(cam=C(0.0, 0.0, 9.0), bodies=((0.0, 0.0, 0.25),), spacing=0.22, width=2.4),
        dict(cam=C(0.0, 0.0, 9.0, PI / 4), bodies=((0.0, 0.0, 1.6),)),
        dict(RED, cam=C(0.0, 0.0, 9.0), bodies=((0.0, 0.0, 1.0),)),
        dict(cam=C(0.0, 0.0, 9.0), bodies=((6.2, 3.0, 1.0),)),
        dict(cam=C(0.0, 0.0, 9.0, PI / 2), bodies=((0.0, 0.0, 3.8),), spacing=0.16),
        dict(NIGHT, cam=C(0.0, 0.0, 9.0), bodies=((0.0, 0.0, 1.0),)),
        dict(cam=C(0.0, 0.0, 9.0, -PI / 4), bodies=((-3.0, 1.0, 1.0),)),
        dict(cam=C(0.0, 0.0, 9.0), bodies=((0.0, -4.5, 3.0),)),
    ]

    def pulse(bodies):
        # the disc swells on the double kick (3, 3-and)
        return lambda c: tuple((x, y, R * (1 + 0.06 * c.kick)) for (x, y, R) in bodies)

    def glide(y, R, rot, x0=-10.5, x1=10.5):
        # fixed camera: the disc is carried across the frame, along the lines
        return lambda c: ((lerp(x0, x1, c.u), y, R * (1 + 0.06 * c.kick)),)

    i = 0
    # phrases 1-2: a new poster on every kick "1" and "3"
    for k in range(304, 328, 2):
        comp = dict(Rl)
        comp.update(comps[i % len(comps)])
        comp["bodies"] = pulse(comp["bodies"])
        S(k, k + 2, f"V poster {i}", comp)
        i += 1
    # phrase 3: moving on - the disc travels with the current
    S(328, 332, "V carried 0", Rl, cam=C(0.0, 0.0, 9.0), bodies=glide(0.0, 1.0, 0.0))
    S(332, 336, "V carried 1", Rl, cam=C(0.0, 0.0, 9.0, -PI / 2), bodies=glide(1.5, 0.7, 0.0, -5.4, 5.4),
      spacing=0.14)
    S(336, 340, "V carried 2", dict(Rl, **RED), cam=C(0.0, 0.0, 9.0, -0.5), bodies=glide(-1.0, 1.4, 0.0, -11.5, 11.5))
    # phrase 4: posters again, faster at the end
    for k in [340, 342, 344, 346, 348, 349, 350]:
        comp = dict(Rl)
        comp.update(comps[i % len(comps)])
        comp["bodies"] = pulse(comp["bodies"])
        k1 = k + 2 if k < 348 else min(k + 1, 351.2)
        if k == 350:
            k1 = 351.2
        S(k, k1, f"V poster {i}", comp)
        i += 1
    St(tb(351.2), tb(352), "V silence", SILENCE)


# ============================================================================
# VI  TONGUES  (beats 352-448): wordless sampler vocals.
# Glyphs from an alphabet that does not exist are placed into the current;
# the current reads them the only way it can: by flowing around them.
# ============================================================================
def tongues():
    Tg = dict(NIGHT, U=1.0, wake=0.0, turb=0.0, spacing=0.085, kick_pulse=0.5, bodies=(), offset=0.0425,
              ripple=0.6, speed=2.0)

    def drift(cam, dx=0.9):
        # the writing stays where it is; we travel downstream past it
        return cam_move(cam, (cam[0] + dx, cam[1], cam[2] * 0.93, cam[3]), lambda u: u)

    k = 352
    for i in range(24):          # 24 bars, two shots each (kick "1" and kick "3")
        pal = dict(Tg)
        fill = VERM
        if i % 4 == 3:
            pal.update(DAY)
        if i in (10, 18):
            pal.update(RED)
            fill = PAPER
        a_type = i % 4
        seed = 11 * i + 3
        if a_type == 0:
            # the self speaks: glyphs are shed into its wake
            gw = glyph_word(seed, n=3, height=1.1)
            A = dict(pal, cam=drift(C(0.2, 0.0, 8.0)), bodies=((-3.4, 0.0, 1.0),), text=(gw, 1.9, 0.0, 0.45, fill),
                     wake=0.6)
        elif a_type == 1:
            gw = glyph_word(seed, n=5, height=1.0)
            A = dict(pal, cam=drift(C(0.0, 0.0, 7.5)), text=(gw, 0.0, 0.0, 0.42, fill))
        elif a_type == 2:
            gw = glyph_word(seed, n=1, height=2.6)
            A = dict(pal, cam=drift(C(0.0, 0.0, 5.0), 0.5), text=(gw, 0.0, 0.0, 0.9, fill), spacing=0.07)
        else:
            gw = glyph_word(seed, n=3, height=1.2)
            A = dict(pal, cam=drift(C(0.0, 0.0, 7.0, PI / 2)), text=(gw, 0.0, 0.0, 0.5, fill))
        k1 = 446.2 if i == 23 else k + 4
        if i == 23:
            S(k, k1, f"VI tongue {i}a", A)
            k += 4
            continue
        S(k, k + 2, f"VI tongue {i}a", A)
        # shot b: a reply - a block of writing, or a close look at one sign
        if i % 2 == 0:
            blk = glyph_block(seed + 1, rows=2, n=3, height=2.3)
            B = dict(pal, cam=drift(C(0.0, 0.0, 6.5), 0.6), text=(blk, 0.0, 0.0, 0.5, fill), spacing=0.075)
        else:
            gw2 = glyph_word(seed + 2, n=2, height=2.0)
            B = dict(pal, cam=drift(C(-0.9, 0.5, 3.6), 0.4), text=(gw2, 0.0, 0.0, 0.7, fill), spacing=0.055)
        S(k + 2, k1, f"VI tongue {i}b", B)
        k += 4
    St(tb(446.2), tb(448), "VI silence", SILENCE)


# ============================================================================
# VII  CHOP  (beats 448-496): gated chords.
# The self dissolves into the current: the disc is drawn with the current's
# own lines, their thickness chopped by the gate.
# ============================================================================
def chop():
    Ch = dict(NIGHT, U=1.0, wake=0.0, turb=0.0, spacing=0.11, body_style="stripes", kick_pulse=0.5)
    fill = lambda c: 0.12 + 0.88 * c.gate ** 1.5
    comps = [C(0.0, 0.0, 6.0), C(0.0, 0.0, 3.0), C(0.0, 0.0, 12.0), C(0.0, 0.0, 6.0, PI / 2),
             C(0.6, 0.6, 2.2), C(0.0, 0.0, 8.0, PI / 4)]
    k = 448
    for i in range(12):
        pal = dict(Ch)
        if i % 4 == 3:
            pal.update(DAY)
        R = [1.6, 1.2, 2.6, 1.8, 1.4, 2.0][i % 6]
        S(k, k + 4, f"VII chop {i}", pal, cam=comps[i % len(comps)], bodies=((0.0, 0.0, R),),
          stripe_fill=fill, gate_lines=0.0)
        k += 4


# ============================================================================
# VIII  READY  (beats 496-640): dawn, lyrics, return to the first frame.
# ============================================================================
def dawn_bg(u):
    top = lerp(np.array(DEEP), np.array(DUSK), min(1, u * 1.3))
    bot = lerp(np.array(DUSK), np.array(ROSE), u)
    return tuple(top), tuple(bot)


READY_T0 = tb(496)
LYR_T0 = 260.247
END_T = 307.34


def sun_u(c):
    return min(max((c.t - READY_T0) / (LYR_T0 - READY_T0), 0), 1)


def ready(words):
    Dn = dict(line=PAPER, body=VERM, U=0.8, wake=0.5, turb=0.2, spacing=0.11, line_a=0.85, kick_pulse=0.5)
    bg = lambda c: dawn_bg(sun_u(c))[0]
    bg2 = lambda c: dawn_bg(sun_u(c))[1]
    # "must be morning": the disc rises as a sun through the bars before the vocal
    sunpos = lambda c: ((0.0, lerp(-6.2, -0.4, ease_out(sun_u(c))), 1.5),)
    cams = [C(0.0, 0.0, 9.0), C(0.0, -1.5, 5.0), C(2.8, -0.5, 9.0), C(0.0, 0.0, 14.0), C(-1.2, -1.8, 4.0),
            C(1.5, -1.0, 7.0), C(0.0, 0.0, 11.0, PI / 2)]
    k = 496
    i = 0
    while k < 542:
        step = 4 if k < 520 else 2
        k1 = min(k + step, 542.12)
        c0 = cams[i % len(cams)]
        rot = c0[3]
        S(k, k1 if k1 < 542 else 542.12, f"VIII dawn {i}", Dn, bg=bg, bg2=bg2 if rot == 0 else None, bodies=sunpos,
          cam=cam_move(c0, (c0[0], c0[1] + 0.3, c0[2] * 0.92, rot)))
        k = k1
        i += 1
    lyric_cards(words)


FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Archivo-Black.ttf")
_tb = {}


def word_body(text, height, ghost=False, max_w=8.6):
    """max_w: widest allowed word in world units (frame is 13.2 wide at span 7.4)"""
    key = (text, height, ghost)
    if key not in _tb:
        probe = TextBody(text, FONT, height=1.0, tracking=0.03, res=120)
        h = min(height, max_w / probe.width)
        # ghosts keep their counters: the loops inside B, D, O help to read them
        _tb[key] = TextBody(text, FONT, height=h, tracking=0.03, fill_holes=not ghost)
    return _tb[key]


# card plan: (first word, last word + 1, style)
#   style: B = big key word, s = small connector, R = big on vermilion,
#          G = ghost (no fill: the word exists only as the shape of the current)
CARDS = [
    (0, 1, "B"), (1, 3, "s"), (3, 4, "R"), (4, 5, "B"), (5, 6, "B"),
    (6, 8, "s"), (8, 9, "B"), (9, 11, "s"), (11, 12, "R"), (12, 14, "B"),
    (14, 16, "s"), (16, 17, "R"), (17, 19, "s"), (19, 20, "N"), (20, 22, "s"), (22, 23, "N"),
    (23, 25, "s"), (25, 26, "B"), (26, 28, "s"), (28, 29, "R"), (29, 31, "B"),
    (31, 33, "B"), (33, 35, "s"), (35, 36, "R"), (36, 38, "B"),
    (38, 40, "s"), (40, 41, "G"), (41, 43, "s"), (43, 44, "R"), (44, 46, "G"),
    (46, 48, "G"), (48, 49, "G"), (49, 51, "G"), (51, 52, "G"), (52, 54, "G"), (54, 55, "G"),
    (55, 57, "G"), (57, 58, "G"), (58, 60, "G"), (60, 61, "R"), (61, 63, "E"),
]


def lyric_cards(words):
    base = dict(DAY, U=0.8, wake=0.0, turb=0.0, bodies=(), ripple=1.0, kick_pulse=0.4, grain=0.009)
    for ci, (a, b, style) in enumerate(CARDS):
        t0 = words[a]["t"]
        t1 = words[CARDS[ci + 1][0]]["t"] if ci + 1 < len(CARDS) else END_T
        text = " ".join(w["w"] for w in words[a:b]).upper().replace(",", "").replace("OH ", "OH, ")
        late = t0 > 282.7                       # second pass: the song is fading
        calm = min(max((t0 - 288.0) / 16.0, 0.0), 1.0)
        spacing = lerp(0.085, 0.12, calm)
        drift = lambda c, s0=7.4: (0.0, 0.0, s0 * (1 - 0.06 * c.u), 0.0)
        if style == "s":
            tbody = word_body(text, 0.75)
            p = dict(base, cam=lambda c: (0.0, 0.0, 7.4 * (1 - 0.05 * c.u), 0.0), spacing=spacing,
                     text=(tbody, 0.0, 0.0, 0.5, VERM), offset=spacing / 2)
        elif style in ("B", "N", "G"):
            h = 1.5 if len(text) <= 6 else 1.05
            tbody = word_body(text, h)
            fill = VERM
            p = dict(base, cam=drift, spacing=spacing, text=(tbody, 0.0, 0.0, 0.6 * h, fill), offset=spacing / 2)
            if style == "N":
                p.update(NIGHT)
            if style == "G":
                # ghost: no fill - only the current shows the word
                tbody = word_body(text, h, ghost=True)
                p.update(text=(tbody, 0.0, 0.0, 0.6 * h, None), spacing=lerp(0.07, 0.1, calm))
        elif style == "R":
            h = 1.6
            tbody = word_body(text, h)
            p = dict(base, **RED, cam=drift, spacing=spacing, text=(tbody, 0.0, 0.0, 0.6 * h, PAPER),
                     offset=spacing / 2)
        elif style == "E":
            # ALL ALONG: then the first frame again - in daylight, and the lines no longer bend
            tbody = word_body(text, 1.05)
            p = dict(base, cam=drift, spacing=spacing, text=(tbody, 0.0, 0.0, 0.63, VERM), offset=spacing / 2)
            t_end = words[62]["end"]
            St(t0, t_end, f"VIII card {ci} {text}", p)
            St(t_end, END_T, "VIII end", dict(DAY, U=1.0, wake=0.0, turb=0.0, obstacle=False, grain=0.009,
                                               spacing=0.11, bodies=((0.0, 0.0, 1.0),), cam=C(3.2, 0.0, 9.0),
                                               lines_on=lambda c: 1.0 - 0.6 * ease_io(min(1.0, c.u * 1.6)),
                                               fade=lambda c: 1.0 - ease_in(max(0.0, (c.u - 0.62) / 0.38))))
            continue
        if late:
            p["line_a"] = lerp(0.95, 0.8, calm)
            p["ripple"] = lerp(1.0, 0.3, calm)
        St(t0, t1, f"VIII card {ci} {text}", p)


def build(words=None):
    SHOTS.clear()
    takeoff()
    skip()
    ocean()
    whirl()
    release()
    tongues()
    chop()
    if words is not None:
        ready(words)
    SHOTS.sort(key=lambda s: s["t0"])
    return SHOTS
