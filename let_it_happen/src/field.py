"""Fused numba kernels: stream function + contour coverage in one pass per pixel.

A field is described by small parameter arrays that are rebuilt every frame:

  flow    : [U, angle]                        uniform flow
  circles : (n, 3)  bx, by, R                 solid bodies (flow goes around)
  vort    : (n, 8)  x, y, gamma, core, mx0, mx1, my, mw
  spirals : (n, 4)  x, y, a, b                a*log r + b*theta (sink/vortex)
  waves   : (n, 9)  kx, ky, phase, amp, mx0, mx1, my, mw, mode
  text    : signed-distance texture of a word (world space) + transform

Masks: an element is multiplied by smoothstep(mx0, mx1, x) (skipped if
mx0 == mx1) and exp(-((y-my)/mw)^2) (skipped if mw <= 0).
"""
import numpy as np
import numba as nb

EMPTY2 = np.zeros((0, 3), np.float32)


@nb.njit(cache=True, fastmath=True)
def _smooth(a, b, x):
    if a == b:
        return 1.0
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@nb.njit(cache=True, fastmath=True)
def _mask(x, y, mx0, mx1, my, mw):
    m = _smooth(mx0, mx1, x)
    if mw > 0.0:
        d = (y - my) / mw
        m *= np.exp(-d * d)
    return m


@nb.njit(cache=True, fastmath=True)
def _sample_tex(tex, u, v):
    h, w = tex.shape
    if u < 0.0:
        u = 0.0
    if v < 0.0:
        v = 0.0
    if u > w - 1.001:
        u = w - 1.001
    if v > h - 1.001:
        v = h - 1.001
    i = int(v)
    j = int(u)
    fu = u - j
    fv = v - i
    return (tex[i, j] * (1 - fu) * (1 - fv) + tex[i, j + 1] * fu * (1 - fv)
            + tex[i + 1, j] * (1 - fu) * fv + tex[i + 1, j + 1] * fu * fv)


@nb.njit(cache=True, fastmath=True)
def eval_field(w, h, cx, cy, span, rot, flow, circles, vort, spirals, waves,
               tex, tex_on, tx, ty, tscale, trho, out_psi, out_phi, out_sd):
    """Evaluate psi (stream function), phi (along-flow coordinate) and the
    signed distance to the nearest solid body for every pixel."""
    cr = np.cos(rot)
    sr = np.sin(rot)
    U = flow[0]
    ca = np.cos(flow[1])
    sa = np.sin(flow[1])
    th, tw = tex.shape
    for py in range(h):
        vy = -((py + 0.5) - h * 0.5) / h * span
        for px in range(w):
            vx = ((px + 0.5) - w * 0.5) / h * span
            x = cr * vx - sr * vy + cx
            y = sr * vx + cr * vy + cy
            # coordinates in the flow frame
            xf = x * ca + y * sa
            yf = -x * sa + y * ca
            g = 1.0
            phi_b = 0.0
            sd = 1e9
            for i in range(circles.shape[0]):
                dx = x - circles[i, 0]
                dy = y - circles[i, 1]
                R = circles[i, 2]
                r2 = dx * dx + dy * dy
                if r2 < 1e-8:
                    r2 = 1e-8
                r = np.sqrt(r2)
                d = r - R
                if d < sd:
                    sd = d
                if d > 0.0:
                    g *= 1.0 - R * R / r2
                    # along-flow potential of the doublet
                    dxf = dx * ca + dy * sa
                    phi_b += U * R * R * dxf / r2
                else:
                    g = 0.0
            if tex_on:
                u = (x - tx) / tscale + tw * 0.5
                v = -(y - ty) / tscale + th * 0.5
                s = _sample_tex(tex, u, v)
                ex = abs(u - tw * 0.5) - tw * 0.5
                ey = abs(v - th * 0.5) - th * 0.5
                if ex < 0.0:
                    ex = 0.0
                if ey < 0.0:
                    ey = 0.0
                s += np.sqrt(ex * ex + ey * ey) * tscale
                if s < sd:
                    sd = s
                if s > 0.0:
                    q = trho / (trho + s)
                    g *= 1.0 - q * q
                else:
                    g = 0.0
            psi = U * yf * g
            phi = U * xf + phi_b
            for i in range(vort.shape[0]):
                dx = x - vort[i, 0]
                dy = y - vort[i, 1]
                c = vort[i, 3]
                m = _mask(x, y, vort[i, 4], vort[i, 5], vort[i, 6], vort[i, 7])
                psi += m * vort[i, 2] / (4.0 * np.pi) * np.log(dx * dx + dy * dy + c * c)
            for i in range(spirals.shape[0]):
                dx = x - spirals[i, 0]
                dy = y - spirals[i, 1]
                psi += 0.5 * spirals[i, 2] * np.log(dx * dx + dy * dy + 1e-4) + spirals[i, 3] * np.arctan2(dy, dx)
            for i in range(waves.shape[0]):
                m = _mask(xf, yf, waves[i, 4], waves[i, 5], waves[i, 6], waves[i, 7])
                arg = waves[i, 0] * xf + waves[i, 1] * yf + waves[i, 2]
                if waves[i, 8] == 0.0:
                    psi += m * waves[i, 3] * np.sin(arg)
                else:
                    # mode 1: wave amplitude scaled by g (vanishes on bodies)
                    psi += m * waves[i, 3] * np.sin(arg) * g
            out_psi[py, px] = psi
            out_phi[py, px] = phi
            out_sd[py, px] = sd


@nb.njit(cache=True, fastmath=True)
def contour_cov(psi, spacing, offset, width_px, every, fade_px, out_a, out_k):
    """Anti-aliased coverage of contours psi = offset + k*spacing (k % every == 0).
    Contours closer than fade_px pixels fade to their mean coverage."""
    h, w = psi.shape
    inv = 1.0 / spacing
    for py in range(h):
        for px in range(w):
            q = (psi[py, px] - offset) * inv
            # wrapped central differences
            if px == 0:
                a0 = psi[py, px + 1] - psi[py, px]
                a1 = a0
            elif px == w - 1:
                a0 = psi[py, px] - psi[py, px - 1]
                a1 = a0
            else:
                a0 = psi[py, px + 1] - psi[py, px]
                a1 = psi[py, px] - psi[py, px - 1]
            a0 *= inv
            a1 *= inv
            a0 -= np.floor(a0 + 0.5)
            a1 -= np.floor(a1 + 0.5)
            gx = 0.5 * (a0 + a1)
            if py == 0:
                b0 = psi[py + 1, px] - psi[py, px]
                b1 = b0
            elif py == h - 1:
                b0 = psi[py, px] - psi[py - 1, px]
                b1 = b0
            else:
                b0 = psi[py + 1, px] - psi[py, px]
                b1 = psi[py, px] - psi[py - 1, px]
            b0 *= inv
            b1 *= inv
            b0 -= np.floor(b0 + 0.5)
            b1 -= np.floor(b1 + 0.5)
            gy = 0.5 * (b0 + b1)
            g = np.sqrt(gx * gx + gy * gy) + 1e-7
            k = np.floor(q + 0.5)
            dist = abs(q - k) / g
            a = width_px * 0.5 + 0.5 - dist
            if a < 0.0:
                a = 0.0
            elif a > 1.0:
                a = 1.0
            sp = 1.0 / g
            if every > 1:
                sp *= every
                kk = k - every * np.floor(k / every)
                if kk != 0.0:
                    a = 0.0
            mean = width_px / sp
            if mean > 1.0:
                mean = 1.0
            bl = (fade_px - sp) / (0.5 * fade_px)
            if bl < 0.0:
                bl = 0.0
            elif bl > 1.0:
                bl = 1.0
            out_a[py, px] = a * (1.0 - bl) + mean * bl * 0.9
            out_k[py, px] = k


class Field:
    """Convenience wrapper collecting elements for one frame."""

    def __init__(self, U=1.0, angle=0.0):
        self.flow = np.array([U, angle], np.float32)
        self.circles = []
        self.vort = []
        self.spirals = []
        self.waves = []
        self.text = None  # (TextBody, cx, cy, rho)

    def circle(self, bx, by, R):
        self.circles.append((bx, by, R))
        return self

    def vortex(self, x, y, gamma, core=0.15, mask=(0, 0, 0, 0)):
        self.vort.append((x, y, gamma, core) + tuple(mask))
        return self

    def spiral(self, x, y, a, arms, spacing):
        self.spirals.append((x, y, a, arms * spacing / (2 * np.pi)))
        return self

    def wave(self, kx, ky, phase, amp, mask=(0, 0, 0, 0), on_body=False):
        self.waves.append((kx, ky, phase, amp) + tuple(mask) + (1.0 if on_body else 0.0,))
        return self

    def textbody(self, tb, cx, cy, rho):
        self.text = (tb, cx, cy, rho)
        return self

    def eval(self, cam):
        w, h = cam.w, cam.h
        psi = np.empty((h, w), np.float32)
        phi = np.empty((h, w), np.float32)
        sd = np.empty((h, w), np.float32)
        c = np.array(self.circles, np.float32).reshape(-1, 3)
        v = np.array(self.vort, np.float32).reshape(-1, 8)
        s = np.array(self.spirals, np.float32).reshape(-1, 4)
        wv = np.array(self.waves, np.float32).reshape(-1, 9)
        if self.text is not None:
            tb, tx, ty, rho = self.text
            tex, on, sc = tb.sd_tex, True, tb.scale
        else:
            tex, on, tx, ty, sc, rho = np.zeros((2, 2), np.float32), False, 0.0, 0.0, 1.0, 1.0
        eval_field(w, h, cam.cx, cam.cy, cam.span, cam.rot, self.flow, c, v, s, wv,
                   tex, on, tx, ty, sc, rho, psi, phi, sd)
        return psi, phi, sd


def lines(psi, spacing, width_px, offset=0.0, every=1, fade_px=3.2):
    a = np.empty(psi.shape, np.float32)
    k = np.empty(psi.shape, np.float32)
    contour_cov(psi, np.float32(spacing), np.float32(offset), np.float32(width_px), every, np.float32(fade_px), a, k)
    return a, k
