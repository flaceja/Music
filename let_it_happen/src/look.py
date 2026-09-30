"""Rendering primitives: stream-function contour lines, discs, text bodies.

Everything is drawn analytically per pixel, so edges stay clean at any zoom:
a line is the set of pixels whose distance (in pixels) to the nearest contour
of the stream function psi is below half the stroke width.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

W, H = 1920, 1080


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


INK = hexc("#0A0B12")
PAPER = hexc("#EFE9DD")
VERM = hexc("#FF4A1F")
NAVY = hexc("#0B1640")
ULTRA = hexc("#1C2FA0")
VIOLET = hexc("#2A1650")
PEACH = hexc("#FF9E6B")
GOLD = hexc("#FFC37A")


# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------
class Cam:
    """2D camera. span = world units across the frame height."""

    def __init__(self, cx=0.0, cy=0.0, span=8.0, rot=0.0, w=W, h=H):
        self.cx, self.cy, self.span, self.rot, self.w, self.h = cx, cy, span, rot, w, h

    def grid(self):
        w, h = self.w, self.h
        px = (np.arange(w, dtype=np.float32) + 0.5 - w / 2) / h * self.span
        py = -(np.arange(h, dtype=np.float32) + 0.5 - h / 2) / h * self.span
        X, Y = np.meshgrid(px, py)
        if self.rot:
            c, s = np.cos(self.rot), np.sin(self.rot)
            X, Y = c * X - s * Y, s * X + c * Y
        return X + self.cx, Y + self.cy

    @property
    def upp(self):
        """world units per pixel"""
        return self.span / self.h

    def to_px(self, x, y):
        dx, dy = x - self.cx, y - self.cy
        if self.rot:
            c, s = np.cos(-self.rot), np.sin(-self.rot)
            dx, dy = c * dx - s * dy, s * dx + c * dy
        return dx / self.span * self.h + self.w / 2, -dy / self.span * self.h + self.h / 2


# ----------------------------------------------------------------------------
# stream function building blocks (all return arrays shaped like X)
# ----------------------------------------------------------------------------
def psi_uniform(X, Y, U=1.0, ang=0.0):
    # flow along direction ang; psi increases to the left of the flow
    return U * (Y * np.cos(ang) - X * np.sin(ang))


def psi_cylinder(X, Y, bx, by, R, U=1.0):
    """uniform flow along +x past a cylinder (without the uniform part)"""
    dx, dy = X - bx, Y - by
    r2 = dx * dx + dy * dy
    r2 = np.maximum(r2, 1e-6)
    return -U * R * R * dy / r2


def psi_vortex(X, Y, vx, vy, gamma, core=0.15):
    dx, dy = X - vx, Y - vy
    return (gamma / (4 * np.pi)) * np.log(dx * dx + dy * dy + core * core)


def psi_spiral(X, Y, vx, vy, a, arms, spacing):
    """log-spiral field a*log r + b*theta. b chosen so the branch cut jumps by
    an exact multiple of the contour spacing (invisible)."""
    dx, dy = X - vx, Y - vy
    b = arms * spacing / (2 * np.pi)
    return 0.5 * a * np.log(dx * dx + dy * dy + 1e-4) + b * np.arctan2(dy, dx)


def wake_street(X, Y, bx, by, R, t, U=1.0, gamma=1.2, n=10, spacing=None, strength=1.0,
                phase=0.0):
    """Karman vortex street shed from a cylinder at (bx, by)."""
    a = spacing if spacing is not None else 4.3 * R
    h = 0.56 * R * 1.6
    uw = 0.82 * U
    out = np.zeros_like(X)
    # continuous shedding: vortices born at x0, travel downstream
    period = a / uw
    tt = t + phase * period
    k0 = np.floor(tt / (period / 2))
    for i in range(n):
        k = k0 - i
        age = tt - k * period / 2
        x = bx + 1.2 * R + uw * age
        sgn = 1 if (k % 2 == 0) else -1
        y = by + sgn * h / 2 * (1 - np.exp(-age / (0.35 * period)))
        grow = 1 - np.exp(-age / (0.25 * period))
        decay = np.exp(-age / (6.0 * period))
        g = -sgn * gamma * U * R * grow * decay * strength
        core = 0.28 * R + 0.06 * uw * age
        out += psi_vortex(X, Y, x, y, g, core)
    return out


def smooth_noise(X, Y, t, seed=0, scale=1.0, octaves=3, drift=(1.0, 0.0)):
    """cheap band-limited noise: sum of random plane waves (deterministic)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X)
    amp = 1.0
    f = 1.0 / scale
    for o in range(octaves):
        for j in range(4):
            ang = rng.uniform(0, 2 * np.pi)
            ph = rng.uniform(0, 2 * np.pi)
            kx, ky = np.cos(ang) * f, np.sin(ang) * f
            out += amp * np.sin(kx * (X - drift[0] * t) + ky * (Y - drift[1] * t) + ph)
        amp *= 0.5
        f *= 2.03
    return out / 4


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
def contour_alpha(psi, spacing, width_px, upp, offset=0.0, every=1, select=None):
    """Anti-aliased coverage of the contour lines psi = offset + k*spacing.

    width_px : stroke width in pixels (float or array)
    upp      : world units per pixel (for gradient scaling)
    every    : draw only every n-th line (k % every == 0)
    """
    q = (psi - offset) / spacing
    # wrapped finite differences -> robust against branch cuts of multiple of spacing
    gx = np.empty_like(q)
    gy = np.empty_like(q)
    d = np.diff(q, axis=1)
    d -= np.round(d)
    gx[:, 1:-1] = 0.5 * (d[:, 1:] + d[:, :-1])
    gx[:, 0] = d[:, 0]
    gx[:, -1] = d[:, -1]
    d = np.diff(q, axis=0)
    d -= np.round(d)
    gy[1:-1] = 0.5 * (d[1:] + d[:-1])
    gy[0] = d[0]
    gy[-1] = d[-1]
    g = np.sqrt(gx * gx + gy * gy) + 1e-7          # contour index change per pixel
    k = np.round(q)
    dist = np.abs(q - k) / g                        # distance to nearest contour, px
    a = np.clip(width_px * 0.5 + 0.5 - dist, 0.0, 1.0)
    # where contours get denser than ~3 px, fade to their mean coverage (no moire)
    spacing_px = 1.0 / g
    mean_cov = np.clip(width_px / spacing_px, 0, 1)
    blend = np.clip((3.2 - spacing_px) / 1.6, 0.0, 1.0)
    a = a * (1 - blend) + mean_cov * blend * 0.9
    if every > 1:
        keep = (np.mod(k, every) == 0)
        a = np.where(keep, a, 0.0)
    if select is not None:
        a = a * select(k)
    return a.astype(np.float32), k


def disc_alpha(X, Y, bx, by, R, upp, feather=0.0):
    d = (np.sqrt((X - bx) ** 2 + (Y - by) ** 2) - R) / upp
    return np.clip(0.5 - d / (1 + feather), 0.0, 1.0).astype(np.float32)


def ring_alpha(X, Y, bx, by, R, width_px, upp):
    d = np.abs(np.sqrt((X - bx) ** 2 + (Y - by) ** 2) - R) / upp
    return np.clip(width_px * 0.5 + 0.5 - d, 0.0, 1.0).astype(np.float32)


def over(img, color, alpha):
    """composite a flat color with per-pixel alpha onto img (in place)."""
    a = alpha[..., None]
    img *= (1 - a)
    img += a * np.asarray(color, dtype=np.float32)
    return img


def canvas(color, w=W, h=H):
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = np.asarray(color, dtype=np.float32)
    return img


def vgradient(c_top, c_bot, w=W, h=H, curve=1.0):
    s = (np.linspace(0, 1, h, dtype=np.float32) ** curve)[:, None, None]
    img = (1 - s) * np.asarray(c_top, np.float32) + s * np.asarray(c_bot, np.float32)
    return np.repeat(img, w, axis=1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# text bodies: rasterised once in world space -> signed distance texture
# ----------------------------------------------------------------------------
FONT_DIR = None


class TextBody:
    """A word as a solid body in world space (for flow-around-text).

    sd_tex  : signed distance used by the flow (letter counters filled, so the
              current never threads through the inside of an O or an A)
    sd_fill : signed distance of the actual letterforms (for painting)
    """

    def __init__(self, text, font_path, height=1.0, cx=0.0, cy=0.0, tracking=0.02, res=360,
                 fill_holes=True):
        self.text = text
        fnt = ImageFont.truetype(font_path, res)
        widths = [fnt.getlength(ch) for ch in text]
        track_px = tracking * res
        total = sum(widths) + track_px * (len(text) - 1)
        cap = fnt.getbbox("H")
        cap_h = cap[3] - cap[1]
        pad = int(res * 1.2)
        wpx, hpx = int(total + 2 * pad), int(cap_h + 2 * pad)
        im = Image.new("L", (wpx, hpx), 0)
        dr = ImageDraw.Draw(im)
        x = pad
        for ch, wch in zip(text, widths):
            dr.text((x, pad - cap[1]), ch, font=fnt, fill=255)
            x += wch + track_px
        m = np.asarray(im, dtype=np.float32) / 255.0
        self.mask = m
        inside = m > 0.5
        self.scale = height / cap_h                    # world units per texture px
        self.sd_fill = self._sd(inside) * self.scale
        flow_inside = ndimage.binary_fill_holes(inside) if fill_holes else inside
        self.sd = self._sd(flow_inside) * self.scale
        self.sd_tex = np.ascontiguousarray(self.sd, dtype=np.float32)
        self.wpx, self.hpx = wpx, hpx
        self.width = total * self.scale
        self.height = height
        self.cx, self.cy = cx, cy
        self.pad = pad

    @staticmethod
    def _sd(inside):
        d_out = ndimage.distance_transform_edt(~inside)
        d_in = ndimage.distance_transform_edt(inside)
        return (d_out - d_in).astype(np.float32)

    def sample(self, X, Y, cx=None, cy=None, fill=True):
        cx = self.cx if cx is None else cx
        cy = self.cy if cy is None else cy
        src = self.sd_fill if fill else self.sd
        u = (X - cx) / self.scale + self.wpx / 2
        v = -(Y - cy) / self.scale + self.hpx / 2
        uc = np.clip(u, 0, self.wpx - 1)
        vc = np.clip(v, 0, self.hpx - 1)
        sd = ndimage.map_coordinates(src, [vc.ravel(), uc.ravel()], order=1, mode="nearest").reshape(X.shape)
        ex = np.maximum(np.abs(u - self.wpx / 2) - self.wpx / 2, 0) * self.scale
        ey = np.maximum(np.abs(v - self.hpx / 2) - self.hpx / 2, 0) * self.scale
        return sd + np.sqrt(ex * ex + ey * ey)


class MaskBody(TextBody):
    """Arbitrary shape (numpy mask, 1 = solid) as a body in world space."""

    def __init__(self, mask, height=1.0, cx=0.0, cy=0.0, pad=None):
        m = mask.astype(np.float32)
        pad = pad if pad is not None else int(max(m.shape) * 0.6)
        m = np.pad(m, pad)
        self.mask = m
        inside = m > 0.5
        d_out = ndimage.distance_transform_edt(~inside)
        d_in = ndimage.distance_transform_edt(inside)
        sd = (d_out - d_in).astype(np.float32)
        ys = np.flatnonzero(inside.any(axis=1))
        body_h = ys[-1] - ys[0] + 1
        self.scale = height / body_h
        self.sd = sd * self.scale
        self.sd_fill = self.sd
        self.sd_tex = np.ascontiguousarray(self.sd, dtype=np.float32)
        self.hpx, self.wpx = m.shape
        xs = np.flatnonzero(inside.any(axis=0))
        self.width = (xs[-1] - xs[0] + 1) * self.scale
        self.height = height
        self.cx, self.cy = cx, cy
        self.pad = pad


def psi_body_from_sd(Y, cy, sd, rho, U=1.0):
    """Stream function of uniform flow around an arbitrary body given its
    signed distance field. Exact for a circle (rho = radius)."""
    d = np.maximum(sd, 0.0)
    g = 1.0 - (rho / (rho + d)) ** 2
    return U * (Y - cy) * g


# ----------------------------------------------------------------------------
# post
# ----------------------------------------------------------------------------
def grain(img, frame, amount=0.012):
    rng = np.random.default_rng(1000003 + frame)
    n = rng.standard_normal(((img.shape[0] + 1) // 2, (img.shape[1] + 1) // 2), dtype=np.float32)
    n = np.repeat(np.repeat(n, 2, axis=0), 2, axis=1)[: img.shape[0], : img.shape[1]]
    img += n[..., None] * amount
    return img


def vignette(img, strength=0.18):
    h, w = img.shape[:2]
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :] * (w / h) / 1.6
    v = 1 - strength * np.clip(x * x + y * y - 0.25, 0, None)
    img *= v[..., None]
    return img


def bloom(img, amount=0.25, radius=18):
    small = img[::4, ::4]
    b = ndimage.gaussian_filter(small, sigma=(radius / 4, radius / 4, 0))
    b = np.repeat(np.repeat(b, 4, axis=0), 4, axis=1)[: img.shape[0], : img.shape[1]]
    return img + amount * np.clip(b - 0.55, 0, None)


def to_u8(img):
    return (np.clip(img, 0, 1) ** (1 / 1.0) * 255 + 0.5).astype(np.uint8)
