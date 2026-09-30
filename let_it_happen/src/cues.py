"""Audio analysis -> per-frame cue arrays for the renderer.

Input : cut.wav (song from 02:40.263 on), lyrics .lrc (already shifted)
Output: cues.npz with per-frame envelopes + event lists
"""
import re
import sys
import json
import numpy as np
import soundfile as sf
import scipy.signal as ss
import librosa

FPS = 50
BEAT = 0.48002          # 125.0 BPM, measured by grid fit (resid std 5 ms)
PHASE = 0.018           # first beat after the cut
BAR = 4 * BEAT

# section map in bars (measured: self-similarity, lag analysis, silence gaps)
SECTIONS = [
    ("takeoff", 0, 36),     # 3-bar loop groove
    ("skip", 36, 44),       # song stuck on a single beat
    ("ocean", 44, 57),      # string glides over the stuck beat
    ("whirl", 57, 76),      # build, ends in a 0.38 s silence
    ("release", 76, 88),    # groove returns
    ("tongues", 88, 112),   # wordless sampler vocals
    ("chop", 112, 124),     # gated chords
    ("ready", 124, 160),    # outro, lyrics from 04:20.25
]


# silences before the section slams (seconds, cut time)
GAPS = [(145.56, 145.94), (168.59, 168.99), (214.67, 215.07)]


def beat_t(k):
    return PHASE + k * BEAT


def bar_t(b):
    return PHASE + b * BAR


def band(x, sr, lo, hi):
    if lo <= 0:
        b, a = ss.butter(4, hi / (sr / 2))
    elif hi >= sr / 2:
        b, a = ss.butter(4, lo / (sr / 2), btype="high")
    else:
        b, a = ss.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return ss.filtfilt(b, a, x)


def env_db(x, sr, win):
    h = int(win * sr)
    e = np.sqrt(np.convolve(x ** 2, np.ones(h) / h, mode="same") + 1e-12)
    return 20 * np.log10(e)


def grid_hits(x, sr, lo, hi, thresh, n_slots):
    """Onset strength of a band measured on every 16th-note slot."""
    z = band(x, sr, lo, hi)
    hop = 64
    e = np.sqrt(np.convolve(z ** 2, np.ones(hop * 2) / (hop * 2), mode="same"))[::hop]
    le = np.log(e + 1e-5)
    flux = np.maximum(0, np.diff(le, prepend=le[0]))
    fr = sr / hop
    out = np.zeros(n_slots)
    for k in range(n_slots):
        t = PHASE + k * BEAT / 4
        i0, i1 = int((t - 0.025) * fr), int((t + 0.035) * fr)
        if i1 >= len(flux):
            break
        out[k] = flux[max(0, i0):i1].sum()
    out /= np.percentile(out[out > 0], 97)
    out = np.clip(out, 0, 1)
    out[out < thresh] = 0
    return out


def pulses_to_env(slots, n_frames, tau):
    t = np.arange(n_frames) / FPS
    env = np.zeros(n_frames)
    for k in np.flatnonzero(slots):
        t0 = PHASE + k * BEAT / 4
        i0 = int(np.ceil(t0 * FPS - 1e-6))
        seg = t[i0:i0 + int(6 * tau * FPS)] - t0
        env[i0:i0 + len(seg)] = np.maximum(env[i0:i0 + len(seg)], slots[k] * np.exp(-seg / tau))
    return env


def parse_lrc(path):
    words = []
    lines = []
    line_re = re.compile(r"^\[(\d+):(\d+\.\d+)\](.*)$")
    word_re = re.compile(r"<(\d+):(\d+\.\d+)>")
    for raw in open(path, encoding="utf-8"):
        m = line_re.match(raw.strip())
        if not m:
            continue
        t0 = int(m.group(1)) * 60 + float(m.group(2))
        body = m.group(3)
        if not body.strip():
            lines.append({"t": t0, "text": ""})
            continue
        parts = word_re.split(body)
        # parts: first word, then (min, sec, word) triples
        ws = [(t0, parts[0])]
        for i in range(1, len(parts), 3):
            ws.append((int(parts[i]) * 60 + float(parts[i + 1]), parts[i + 2]))
        ws = [(t, w.strip()) for t, w in ws if w.strip()]
        lines.append({"t": t0, "text": " ".join(w for _, w in ws)})
        for t, w in ws:
            words.append({"t": t, "w": w, "line": len(lines) - 1})
    for i, ln in enumerate(lines):
        ln["end"] = lines[i + 1]["t"] if i + 1 < len(lines) else None
    for i, w in enumerate(words):
        nxt = words[i + 1]["t"] if i + 1 < len(words) and words[i + 1]["line"] == w["line"] else lines[w["line"]]["end"]
        w["end"] = nxt
    return lines, words


def main(wav, lrc, out):
    y, sr = sf.read(wav)
    y = y.mean(axis=1)
    dur = len(y) / sr
    n_frames = int(round(dur * FPS))
    n_slots = int((dur - PHASE) / (BEAT / 4)) + 1

    # percussive part for drum cues
    y22 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=22050)
    H, Pc = librosa.effects.hpss(y22, margin=(1.0, 3.0))
    sr22 = 22050
    kick_m = grid_hits(Pc, sr22, 0, 110, 0.0, n_slots)
    snare_m = grid_hits(Pc, sr22, 1500, 4500, 0.0, n_slots)
    # The mix is dense (bass and synth onsets leak into every band), so hits are
    # placed from the per-section drum pattern (measured as bar-averaged profiles)
    # and only their strength is taken from the audio.
    patterns = {
        "takeoff": ({0: .8, 4: 1., 8: .75, 10: .6}, {4: .55, 12: 1.}),
        "skip": ({0: 1., 4: 1., 8: 1., 12: 1.}, {0: .8, 4: .8, 8: .8, 12: .8}),
        "ocean": ({0: .8, 4: .8, 8: .8, 12: .8}, {0: .5, 4: .5, 8: .5, 12: .5}),
        "whirl": ({0: 1., 8: .9}, {4: .8, 12: .8}),
        "release": ({0: 1., 8: .9, 10: .8}, {4: 1., 12: 1.}),
        "tongues": ({0: 1., 8: .9, 10: .8}, {4: 1., 12: 1.}),
        "chop": ({0: 1., 8: .9, 10: .8}, {4: 1., 12: 1.}),
        "ready": ({0: 1., 8: .9, 10: .8}, {4: 1., 12: 1.}),
    }
    kick = np.zeros(n_slots)
    snare = np.zeros(n_slots)
    for name, b0, b1 in SECTIONS:
        kp, sp = patterns[name]
        for b in range(b0, b1):
            for slot, w in kp.items():
                k = b * 16 + slot
                if k < n_slots:
                    kick[k] = w * np.clip(0.55 + 0.6 * kick_m[k], 0.6, 1.0)
            for slot, w in sp.items():
                k = b * 16 + slot
                if k < n_slots:
                    snare[k] = w * np.clip(0.55 + 0.6 * snare_m[k], 0.6, 1.0)
    for g0, g1 in GAPS:
        for arr in (kick, snare):
            k0, k1 = int(np.ceil((g0 - PHASE) / (BEAT / 4))), int(np.floor((g1 - PHASE) / (BEAT / 4) - 0.01))
            arr[k0:k1 + 1] = 0
    hat = np.zeros(n_slots)

    kick_env = pulses_to_env(kick, n_frames, 0.11)
    snare_env = pulses_to_env(snare, n_frames, 0.09)
    hat_env = pulses_to_env(hat, n_frames, 0.05)

    # loudness (smoothed), 0..1
    t = np.arange(n_frames) / FPS
    db = env_db(y, sr, 0.25)
    loud = np.interp(t, np.arange(len(db)) / sr, db)
    loud = np.clip((loud - (-40)) / 32, 0, 1)

    # glide pitch in the 'ocean' section: peak of CQT residual against the static loop spectrum
    hop = 256
    C = np.abs(librosa.cqt(H, sr=sr22, hop_length=hop, fmin=librosa.note_to_hz("C2"), n_bins=72))  # 1 bin = 1 semitone
    CdB = librosa.amplitude_to_db(C, ref=np.max)
    ct = np.arange(C.shape[1]) * hop / sr22
    stuck = (ct >= bar_t(37)) & (ct < bar_t(44))
    ref = np.median(CdB[:, stuck], axis=1, keepdims=True)
    resid = np.clip(CdB - ref - 3, 0, None)[11:26] ** 2   # C3..D4 register
    cen = (resid * np.arange(11, 26)[:, None]).sum(0) / (resid.sum(0) + 1e-6)
    cen = ss.savgol_filter(ss.medfilt(cen, 31), 41, 2)
    glide = np.interp(t, ct, cen)
    glide = np.clip((glide - 14.5) / (21.0 - 14.5), 0, 1)
    glide[t < bar_t(43.5)] = 0.0
    glide[t > bar_t(57)] = 1.0
    glide_str = np.ones_like(glide)

    # chop gate: mid band envelope (fast), normalised per section
    mid = band(y, sr, 400, 3000)
    gdb = env_db(mid, sr, 0.02)
    gate = np.interp(t, np.arange(len(gdb)) / sr, gdb)
    lo_, hi_ = np.percentile(gate[(t > bar_t(112)) & (t < bar_t(124))], [10, 95])
    gate = np.clip((gate - lo_) / (hi_ - lo_), 0, 1)

    # spectral brightness (for the whirl build)
    S = np.abs(librosa.stft(y22, n_fft=2048, hop_length=512))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr22)[0]
    bright = np.interp(t, np.arange(len(cent)) * 512 / sr22, cent)
    bright = ss.savgol_filter(bright, 51, 2)
    bright = (bright - np.percentile(bright, 5)) / (np.percentile(bright, 95) - np.percentile(bright, 5))

    lines, words = parse_lrc(lrc)

    np.savez_compressed(
        out,
        fps=FPS, beat=BEAT, phase=PHASE, n_frames=n_frames, dur=dur,
        kick_slots=kick, snare_slots=snare, hat_slots=hat,
        kick=kick_env, snare=snare_env, hat=hat_env, loud=loud,
        glide=glide, glide_str=glide_str, gate=gate, bright=np.clip(bright, 0, 1),
    )
    with open(out.replace(".npz", "_lyrics.json"), "w") as f:
        json.dump({"lines": lines, "words": words}, f, indent=1)
    print("frames", n_frames, "dur", dur, "slots", n_slots)


if __name__ == "__main__":
    main(*sys.argv[1:4])
