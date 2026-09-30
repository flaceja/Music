"""Render driver.

  python render.py video  OUT.mp4 --audio cut.wav [--preview] [--t0 S --t1 S]
  python render.py stills OUTDIR --times 1.0 2.5 ...
  python render.py sheet  OUT.png            (one frame per shot, labelled)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.signal as ss

sys.path.insert(0, os.path.dirname(__file__))
from look import to_u8  # noqa: E402
import scenes  # noqa: E402
import edl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G = {}


def setup(cues_path, lyrics_path):
    c = dict(np.load(cues_path))
    # smooth the glide control a little more (strings move slowly)
    g = c["glide"]
    c["glide_s"] = ss.savgol_filter(g, 31, 2).clip(0, 1)
    c["kick_int"] = np.cumsum(c["kick"]) / float(c["fps"])
    G["cues"] = c
    words = json.load(open(lyrics_path))["words"] if lyrics_path else None
    G["shots"] = edl.build(words)
    G["cache"] = {}


def shot_at(n, fps):
    shots = G["shots"]
    starts = G.setdefault(("starts", fps), [int(round(s["t0"] * fps)) for s in shots])
    i = np.searchsorted(starts, n, side="right") - 1
    return max(i, 0)


def render_content(si, shot, frame, t_content, tl, w, h, fps):
    key = (si, int(round(t_content * fps)), w)
    if "tmap" in shot and key in G["cache"]:
        return G["cache"][key]
    ctx = scenes.Ctx(frame, t_content, tl, shot["t1"] - shot["t0"], G["cues"], shot)
    img = scenes.render(shot["p"], ctx, w, h)
    if "tmap" in shot:
        if len(G["cache"]) > 80:
            G["cache"].clear()
        G["cache"][key] = img
    return img


def frame_image(n, fps, w, h):
    t = n / fps
    si = shot_at(n, fps)
    shot = G["shots"][si]
    tl = t - shot["t0"]
    if "tmap" not in shot:
        mb = shot["p"].get("mblur", 0)
        if not mb:
            return render_content(si, shot, n, t, tl, w, h, fps)
        # temporal supersampling for fast moves (180 degree shutter)
        acc = None
        for j in range(mb):
            dt = ((j + 0.5) / mb - 0.5) * 0.5 / fps
            img = render_content(si, shot, n, t + dt, tl + dt, w, h, fps)
            acc = img if acc is None else acc + img
        return acc / mb
    _, T0, L = shot["tmap"]
    Lf = int(round(L * fps))
    base_f = int(round(T0 * fps))
    k = n - base_f
    nb = shot.get("slices", 1)
    if nb <= 1:
        tc = (base_f + (k % Lf)) / fps
        return render_content(si, shot, n, tc, tl, w, h, fps).copy()
    out = np.empty((h, w, 3), np.float32)
    edges = np.linspace(0, w, nb + 1).round().astype(int)
    for b in range(nb):
        off = int(round(b * Lf / nb * 0.5))
        tc = (base_f + ((k + off) % Lf)) / fps
        img = render_content(si, shot, n, tc, tl, w, h, fps)
        out[:, edges[b]:edges[b + 1]] = img[:, edges[b]:edges[b + 1]]
    return out


def _chunk(args):
    a, b, fps, w, h, path, cues, lyrics = args
    if "cues" not in G:
        setup(cues, lyrics)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-threads", "1", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for n in range(a, b):
        img = frame_image(n, fps, w, h)
        p.stdin.write(to_u8(img).tobytes())
    p.stdin.close()
    p.wait()
    return a, b, time.time() - t0


def video(out, audio, cues, lyrics, fps, w, h, t0, t1, workers, chunk):
    setup(cues, lyrics)
    n_total = int(round(G["cues"]["dur"] * fps))
    a0 = int(round(t0 * fps)) if t0 is not None else 0
    a1 = min(int(round(t1 * fps)), n_total) if t1 is not None else n_total
    tmp = out + ".parts"
    os.makedirs(tmp, exist_ok=True)
    jobs = []
    for a in range(a0, a1, chunk):
        b = min(a + chunk, a1)
        jobs.append((a, b, fps, w, h, os.path.join(tmp, f"{a:06d}.mp4"), cues, lyrics))
    t_start = time.time()
    done = 0
    with Pool(workers) as pool:
        for a, b, dt in pool.imap_unordered(_chunk, jobs):
            done += b - a
            el = time.time() - t_start
            print(f"chunk {a}-{b} {dt:.0f}s | {done}/{a1 - a0} frames, {el / 60:.1f} min, "
                  f"eta {(a1 - a0 - done) * el / done / 60:.1f} min", flush=True)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for j in jobs:
            f.write(f"file '{os.path.abspath(j[5])}'\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst]
    if audio:
        cmd += ["-ss", f"{a0 / fps:.6f}", "-t", f"{(a1 - a0) / fps:.6f}", "-i", audio,
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "320k"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print("wrote", out)


def stills(outdir, times, cues, lyrics, w, h, fps=50):
    setup(cues, lyrics)
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    for t in times:
        n = int(round(t * fps))
        img = frame_image(n, fps, w, h)
        Image.fromarray(to_u8(img)).save(os.path.join(outdir, f"f_{t:08.3f}.png"))


def sheet(out, cues, lyrics, w=480, h=270, cols=8, t0=None, t1=None, fps=50):
    setup(cues, lyrics)
    from PIL import Image, ImageDraw
    shots = [s for s in G["shots"] if (t0 is None or s["t0"] >= t0) and (t1 is None or s["t0"] < t1)]
    rows = (len(shots) + cols - 1) // cols
    sheet_im = Image.new("RGB", (cols * w, rows * (h + 18)), (40, 40, 40))
    dr = ImageDraw.Draw(sheet_im)
    for i, s in enumerate(shots):
        tm = 0.5 * (s["t0"] + s["t1"])
        n = int(round(tm * fps))
        img = frame_image(n, fps, w, h)
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        sheet_im.paste(Image.fromarray(to_u8(img)), (x, y))
        dr.text((x + 4, y + h + 2), f"{s['t0']:.2f} {s['name']}", fill=(230, 230, 230))
    sheet_im.save(out)
    print(len(shots), "shots")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("out")
    ap.add_argument("--cues", required=True)
    ap.add_argument("--lyrics")
    ap.add_argument("--audio")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--times", type=float, nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    a = ap.parse_args()
    if a.mode == "video":
        fps, w, h = (25, 960, 540) if a.preview else (50, 1920, 1080)
        video(a.out, a.audio, a.cues, a.lyrics, fps, w, h, a.t0, a.t1, a.workers, a.chunk)
    elif a.mode == "stills":
        stills(a.out, a.times, a.cues, a.lyrics, a.w * 4, a.h * 4)
    elif a.mode == "sheet":
        sheet(a.out, a.cues, a.lyrics, a.w, a.h, t0=a.t0, t1=a.t1)
