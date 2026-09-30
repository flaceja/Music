"""One universal shot renderer. A shot is a dict of parameters; values may be
callables of the shot context (so cameras, bodies etc. can move).

Visual language (whole video):
  lines  = the current (contours of a stream function)
  disc   = the self (vermilion), the only warm element until the dawn
  words  = solid bodies the current has to flow around
"""
import numpy as np
from look import (Cam, canvas, vgradient, over, grain, vignette, smoothstep, to_u8,
                  INK, PAPER, VERM, NAVY, VIOLET, PEACH, GOLD, TextBody)
from field import Field, lines

FPS = 50


class Ctx:
    """Everything a shot can depend on at one output frame."""

    def __init__(self, frame, t, tl, dur, cues, shot):
        self.frame = frame      # output frame index
        self.t = t              # content time (seconds, after remaps)
        self.tl = tl            # time since shot start
        self.dur = dur          # shot duration
        self.cues = cues
        self.shot = shot
        fi = min(max(int(round(t * FPS)), 0), len(cues["kick"]) - 1)
        self.fi = fi
        self.kick = float(cues["kick"][fi])
        self.snare = float(cues["snare"][fi])
        self.loud = float(cues["loud"][fi])
        self.glide = float(cues["glide_s"][fi])
        self.gate = float(cues["gate"][fi])
        self.kick_int = float(cues["kick_int"][fi])

    @property
    def u(self):
        """normalised progress through the shot 0..1"""
        return min(max(self.tl / max(self.dur, 1e-6), 0.0), 1.0)


def val(x, c):
    return x(c) if callable(x) else x


DEFAULTS = dict(
    cam=(0.0, 0.0, 9.0, 0.0),
    bg=INK,
    bg2=None,                 # bottom colour of a vertical gradient
    line=PAPER,
    line_a=0.92,
    width=1.6,                # px at 1080p
    spacing=0.11,
    offset=0.0,
    every=1,
    U=1.0,
    angle=0.0,
    bodies=((0.0, 0.0, 1.0),),
    body=VERM,
    body_style="solid",       # solid | ring | stripes | ocean | none
    wake=1.0,
    wake_t0=0.0,
    turb=0.0,
    turb_scale=1.0,
    spiral=None,              # (x, y, a, arms)
    vortices=(),
    swell=0.0,                # large slow waves over the whole field (ocean)
    swell_k=0.8,
    text=None,                # (TextBody, cx, cy, rho, fill)
    text_fill=VERM,
    dashes=0.0,               # tracer pulses along some lines
    dash_col=None,
    dash_every=5,
    kick_pulse=0.6,           # stroke width boost on kicks
    gate_lines=0.0,           # 1 = lines only visible while the gate is open
    inner=None,               # dict for the ocean inside the body
    lines_on=1.0,
    grain=0.010,
    vig=0.12,
    flash=0.0,
    fade=1.0,                 # global multiplier towards bg (for fades)
    extra=None,               # callable(img, ctx, fields) for special overlays
    obstacle=True,            # False: bodies drift WITH the current (lines stay straight)
    speed=1.0,                # animation speed of the current (wake, turbulence, pulses)
    eye=None,                 # (x, y, r): the calm centre of the whirl
)


def vortex_street(field, bx, by, R, t, U, strength, n=8):
    if strength <= 0:
        return
    a = 4.3 * R
    h = 0.9 * R
    uw = 0.82 * U
    period = a / max(uw, 1e-3)
    k0 = np.floor(t / (period / 2))
    for i in range(n):
        k = k0 - i
        age = t - k * period / 2
        x = bx + 1.9 * R + uw * age
        sgn = 1 if (k % 2 == 0) else -1
        y = by + sgn * h / 2 * (1 - np.exp(-age / (0.35 * period)))
        grow = 1 - np.exp(-age / (0.3 * period))
        decay = np.exp(-age / (5.0 * period))
        g = -sgn * 1.5 * U * R * grow * decay * strength
        core = 0.30 * R + 0.07 * uw * age
        field.vortex(x, y, g, core)


def turbulence(field, t, U, amp, scale, bx, R, seed=7):
    if amp <= 0:
        return
    rng = np.random.default_rng(seed)
    for o in range(3):
        for j in range(4):
            ang = rng.uniform(-1.2, 1.2)
            ph = rng.uniform(0, 2 * np.pi)
            f = (1.0 / scale) * (2.03 ** o)
            kx, ky = np.cos(ang) * f, np.sin(ang) * f
            a = amp * (0.55 ** o) / 4
            field.wave(kx, ky, ph - kx * U * 0.85 * t, a, mask=(bx + 0.3 * R, bx + 3.5 * R, 0.0, 1.9 * R), on_body=True)


def render(shot, ctx, w=1920, h=1080):
    p = dict(DEFAULTS)
    p.update(shot)
    s = h / 1080.0
    t = ctx.t
    cx, cy, span, rot = val(p["cam"], ctx)
    cam = Cam(cx, cy, span, rot, w, h)
    U = val(p["U"], ctx)
    f = Field(U=U, angle=val(p["angle"], ctx))
    bodies = val(p["bodies"], ctx) or ()
    if p["obstacle"]:
        for (bx, by, R) in bodies:
            f.circle(bx, by, R)
    ta = t * val(p["speed"], ctx)     # animation clock of the current
    wake = val(p["wake"], ctx)
    if bodies and wake > 0:
        bx, by, R = bodies[0]
        vortex_street(f, bx, by, R, ta - p["wake_t0"], U, wake)
    turb = val(p["turb"], ctx)
    if bodies and turb > 0:
        bx, by, R = bodies[0]
        turbulence(f, ta, U, turb, p["turb_scale"], bx, R)
    rp = val(p.get("ripple", 0.0), ctx)
    if rp:
        # travelling ripples over the whole current, kicked by the drum
        amp = rp * (0.6 + 0.8 * ctx.kick)
        f.wave(1.3, 0.25, -1.3 * U * ta, 0.05 * amp, on_body=True)
        f.wave(2.1, -0.4, 1.7 - 2.1 * U * 1.1 * ta, 0.03 * amp, on_body=True)
    sw = val(p["swell"], ctx)
    if sw:
        k = p["swell_k"]
        f.wave(k, 0.0, -k * 0.6 * t, sw, on_body=True)
        f.wave(1.7 * k, 0.35 * k, 1.3 - 1.7 * k * 0.8 * t, 0.45 * sw, on_body=True)
    sp = val(p["spiral"], ctx)
    if sp is not None:
        f.spiral(sp[0], sp[1], sp[2], sp[3], val(p["spacing"], ctx))
    for v in val(p["vortices"], ctx):
        f.vortex(*v)
    txt = val(p["text"], ctx)
    if txt is not None:
        tb, tx, ty, rho = txt[:4]
        f.textbody(tb, tx, ty, rho)

    psi, phi, sd = f.eval(cam)
    upp = cam.upp
    if not p["obstacle"] and bodies:
        X, Y = cam.grid()
        sd = np.full(psi.shape, 1e9, np.float32)
        for (bx, by, R) in bodies:
            sd = np.minimum(sd, np.sqrt((X - bx) ** 2 + (Y - by) ** 2) - R)
    outside = np.clip(sd / upp + 0.5, 0.0, 1.0) if bodies or txt is not None else 1.0
    eye = val(p["eye"], ctx)
    if eye is not None:
        X, Y = cam.grid()
        er = np.sqrt((X - eye[0]) ** 2 + (Y - eye[1]) ** 2)
        outside = outside * np.clip((er - max(eye[2], 9 * upp)) / upp + 0.5, 0.0, 1.0)

    # background
    bg = val(p["bg"], ctx)
    bg2 = val(p["bg2"], ctx)
    img = vgradient(bg, bg2, w, h) if bg2 is not None else canvas(bg, w, h)

    # lines
    width = val(p["width"], ctx) * s * (1.0 + p["kick_pulse"] * ctx.kick * 0.5)
    lines_on = val(p["lines_on"], ctx)
    if lines_on > 0:
        spc = val(p["spacing"], ctx)
        a, kidx = lines(psi, spc, width, val(p["offset"], ctx), int(val(p["every"], ctx)))
        a *= outside
        la = val(p["line_a"], ctx) * lines_on
        if p["gate_lines"]:
            g = p["gate_lines"]
            la *= (1 - g) + g * smoothstep(0.15, 0.55, ctx.gate)
        over(img, val(p["line"], ctx), a * la)
        dsh = val(p["dashes"], ctx)
        if dsh > 0:
            # bright pulses riding along every n-th line
            ev = p["dash_every"]
            sel = (np.mod(kidx, ev) == 0)
            hsh = np.mod(kidx * 0.61803, 1.0)
            L = 3.2
            q = np.mod((phi - 2.4 * U * ta) / L + hsh, 1.0)
            pulse = smoothstep(0.0, 0.02, q) * (1 - smoothstep(0.02, 0.16, q))
            dcol = p["dash_col"] if p["dash_col"] is not None else VERM
            a2, _ = lines(psi, spc, width * 2.6, val(p["offset"], ctx), 1)
            over(img, dcol, a2 * outside * pulse * sel * dsh)

    # bodies
    style = val(p["body_style"], ctx)
    if bodies and style != "none":
        cov = np.clip(0.5 - sd / upp, 0.0, 1.0)
        if txt is not None:
            # sd includes the text; only fill the discs here
            X, Y = cam.grid()
            cov = np.zeros_like(sd)
            for (bx, by, R) in bodies:
                cov = np.maximum(cov, np.clip(0.5 - (np.sqrt((X - bx) ** 2 + (Y - by) ** 2) - R) / upp, 0, 1))
        bcol = val(p["body"], ctx)
        if style == "solid":
            over(img, bcol, cov)
        elif style == "ring":
            ring = np.clip(1.5 * s + 0.5 - np.abs(sd) / upp, 0, 1)
            over(img, bcol, ring)
        elif style == "stripes":
            # the body made of the current's own lines: thickness follows the gate
            X, Y = cam.grid()
            sp_ = val(p["spacing"], ctx)
            fill = val(p["stripe_fill"], ctx) if "stripe_fill" in p else 0.5
            q = np.abs(np.mod(Y / sp_, 1.0) - 0.5) * 2         # 0 at line centre .. 1 between
            edge = (fill - q) * sp_ / upp * 0.5
            st = np.clip(edge + 0.5, 0, 1)
            over(img, bcol, cov * st)
        elif style == "ocean":
            over(img, bcol, cov)
            inner = val(p["inner"], ctx) or {}
            fi = Field(U=1.0, angle=0.0)
            amp = inner.get("amp", 0.25)
            k = inner.get("k", 2.0)
            spd = inner.get("speed", 0.7)
            lift = inner.get("lift", 0.0)
            bx, by, R = bodies[0]
            fi.wave(k, 0.0, -k * spd * t, amp)
            fi.wave(1.9 * k, 0.4 * k, 2.0 - 1.9 * k * spd * 1.2 * t, 0.4 * amp)
            fi.wave(0.6 * k, -0.2 * k, 0.7 + 0.6 * k * spd * 0.7 * t, 0.6 * amp)
            icam = Cam(cam.cx - bx, cam.cy - by - lift, cam.span, cam.rot, w, h)
            psi_i, _, _ = fi.eval(icam)
            ai, _ = lines(psi_i, inner.get("spacing", 0.09), width * inner.get("wscale", 1.0), 0.0, 1)
            # clip to the disc, slightly inset
            inset = np.clip(-sd / upp - 1.5 * s, 0, 1)
            over(img, inner.get("col", INK), ai * inset * inner.get("alpha", 0.95))

    if txt is not None and (len(txt) < 5 or txt[4] is not None):
        tfill = txt[4] if len(txt) > 4 else p["text_fill"]
        # text fill = inside of text sd only (bodies excluded above)
        tb, tx, ty = txt[0], txt[1], txt[2]
        X, Y = cam.grid()
        tsd = tb.sample(X, Y, tx, ty)
        over(img, tfill, np.clip(0.5 - tsd / upp, 0, 1))

    if p["extra"] is not None:
        p["extra"](img, ctx, dict(psi=psi, phi=phi, sd=sd, cam=cam))

    fl = val(p["flash"], ctx)
    if fl:
        img += fl
    fade = val(p["fade"], ctx)
    if fade < 1.0:
        img *= fade
    if p["vig"]:
        vignette(img, p["vig"])
    if p["grain"]:
        grain(img, ctx.frame, 0.6 * p["grain"])
    return img
