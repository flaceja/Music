"""Measure how the piano MIDI maps onto the recording (offline, run once).

The MIDI transcription follows the song's form but not its exact timing: the
offset drifts between about +0.4 s and +1.2 s and the tempo map is only
approximate. This script derives a piecewise-linear time warp
(MIDI seconds -> audio seconds) with one anchor per MIDI beat.

Method
  1. Pitch-specific onset salience of the audio: CQT (A1..B7, 1 bin per
     semitone), log magnitude, positive "superflux" difference per bin.
  2. Every MIDI note scores the salience at its fundamental and first two
     overtones (octave, octave+fifth) at its warped onset time.
  3. Viterbi over the audio time of every MIDI beat (states on a 5 ms grid
     inside +-0.3 s around a prior through hand-checked structural anchors,
     see PRIOR), maximising the total note score
     with a penalty on local tempo ratio changes between consecutive beats.
     The penalty is 10x weaker in the free-time (rubato) passages: the solo
     intro and the ritardando into the break.
  4. Report per-note residuals (distance from the warped onset to the nearest
     salience peak) for the warp and for the best constant offset.

Usage
  python align_midi_to_audio.py song.mp3 piano.mid warp.json
"""
import json
import sys
import warnings

import librosa
import numpy as np
import pretty_midi

warnings.filterwarnings("ignore")

SR = 22050
HOP = 256
FPS_A = SR / HOP              # analysis frames per second (86.13)
PITCH_LO = 33                 # CQT starts at A1
N_BINS = 72                   # A1 .. G#7
GRID = 0.005                  # Viterbi state grid (s)
WIN = 0.3                     # search window around the prior (s), < half a beat
# Prior: structural anchors (MIDI s, audio s), checked by eye on a CQT/MIDI
# overlay: the first note, the entrance of the left hand, the start of the
# held F#6, the isolated tutti chords after silence and the last chord.
# Needed because the music is periodic: an unconstrained search happily
# locks on a warp that is off by 2-3 whole beats (score almost identical).
PRIOR = ((0.000, 0.302), (2.250, 2.612), (6.000, 6.745), (11.850, 12.794),
         (14.850, 15.680), (59.585, 60.410), (66.461, 66.621), (119.182, 119.300))
LAM_STEADY = 40.0             # tempo-change penalty in steady sections
LAM_RUBATO = 4.0              # ... in the free-time passages
RUBATO = ((0.0, 14.85), (59.5, 66.46))   # MIDI seconds


def salience(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, n_bins=N_BINS,
                           fmin=librosa.midi_to_hz(PITCH_LO), bins_per_octave=12))
    S = np.log1p(100.0 * C / C.max())
    # two-scale superflux: fast attacks (1-3 frames) and the soft, slow attacks
    # of the lo-fi intro piano (4-8 frames); both compared with a max filter
    # of the past frames, which suppresses vibrato and tremolo
    def flux(lags):
        prev = np.maximum.reduce([np.roll(S, k, axis=1) for k in lags])
        return np.maximum(0.0, S - prev)
    F = flux((1, 2, 3)) + 0.5 * flux((4, 5, 6, 7, 8))
    F[:, :8] = 0
    # small temporal tolerance: +-1 frame max filter
    F = np.maximum.reduce([F, np.roll(F, 1, axis=1), np.roll(F, -1, axis=1)])
    # global normalisation. (A local, per-4-s normalisation was tried: it
    # helps the quiet intro but lets the repetitive chorus slip by 2 beats.
    # The intro is handled by refine_rubato() instead.)
    F /= np.percentile(F[F > 0], 99)
    Sdb = librosa.amplitude_to_db(C, ref=C.max())
    return F, Sdb, len(y) / SR


def note_rows(pitch):
    """CQT rows + weights for a note: fundamental, octave, octave+fifth."""
    rows, w = [], []
    for d, wt in ((0, 1.0), (12, 0.6), (19, 0.35)):
        r = pitch + d - PITCH_LO
        if 0 <= r < N_BINS:
            rows.append(r)
            w.append(wt)
    return rows, w


def note_score_fn(F):
    T = F.shape[1]

    def score(pitches, times):
        """pitches: (n,), times: (..., n) audio seconds -> (...,) summed score"""
        idx = np.clip(np.rint(times * FPS_A).astype(int), 0, T - 1)
        out = np.zeros(times.shape[:-1])
        for j, p in enumerate(pitches):
            rows, w = note_rows(int(p))
            for r, wt in zip(rows, w):
                out += wt * F[r, idx[..., j]]
        return out
    return score


def refine_rubato(anchors, beats, on, pit, Sdb, search=0.35):
    """Snap the beats of the solo intro to the attacks of their notes.

    The intro piano is soft and filtered: its attacks rise over 100-200 ms, so
    the flux peak comes late. Here the onset is the steepest rise of the
    (50 ms smoothed) dB envelope of the note's fundamental + octave, searched
    +-350 ms around the Viterbi anchor. Beats without a note in the MIDI are
    re-interpolated afterwards; the result stays strictly monotonic.
    """
    from scipy.ndimage import uniform_filter1d
    a = anchors.copy()
    fixed = np.zeros(len(a), dtype=bool)
    lo, hi = RUBATO[0]
    for k, b in enumerate(beats):
        if not (lo <= b < hi):
            continue
        hit = np.where(np.abs(on - b) < 1e-3)[0]
        if len(hit) == 0:
            continue
        p = int(pit[hit].max())                       # top voice
        env = sum(Sdb[r] for r in (p - PITCH_LO, p + 12 - PITCH_LO) if 0 <= r < N_BINS)
        env = uniform_filter1d(env, int(0.05 * FPS_A))
        slope = np.gradient(env) * FPS_A              # dB / s
        i0 = int((a[k] - search) * FPS_A); i1 = int((a[k] + search) * FPS_A)
        i0 = max(i0, 1)
        j = i0 + int(np.argmax(slope[i0:i1]))
        if slope[j] > 60.0:                           # a clear attack (> 60 dB/s)
            a[k] = j / FPS_A
            fixed[k] = True
    # re-interpolate the free beats of the intro between the snapped ones
    idx = np.where(fixed | ~((beats >= lo) & (beats < hi)))[0]
    a = np.interp(np.arange(len(a)), idx, a[idx])
    assert np.all(np.diff(a) > 0), "warp must be monotonic"
    return a


def simplify(x, y, eps):
    """Ramer-Douglas-Peucker on a monotonic polyline: drop anchors that lie
    within eps seconds of the straight line through their neighbours."""
    keep = np.zeros(len(x), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(x) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        yy = y[i] + (y[j] - y[i]) * (x[i + 1:j] - x[i]) / (x[j] - x[i])
        d = np.abs(y[i + 1:j] - yy)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return x[keep], y[keep]


def main(audio, midi, out):
    F, Sdb, dur = salience(audio)
    score = note_score_fn(F)
    pm = pretty_midi.PrettyMIDI(midi)
    notes = sorted((n for ins in pm.instruments for n in ins.notes), key=lambda n: n.start)
    on = np.array([n.start for n in notes])
    pit = np.array([n.pitch for n in notes])
    beats = pm.get_beats()
    end = pm.get_end_time()
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    beats = beats[beats <= end + 1e-6] if beats[-1] > end + 1.0 else beats

    # coarse prior: best constant offset
    offs = np.arange(0.0, 2.0, 0.01)
    tot = [score(pit, (on + o)[None, :])[0] for o in offs]
    o0 = offs[int(np.argmax(tot))]
    print(f"best constant offset {o0:.3f} s (reference only, not used)")
    px, py = np.array(PRIOR).T

    K = len(beats)
    cand = np.arange(-WIN, WIN + 1e-9, GRID)
    S = len(cand)
    A = np.interp(beats, px, py, right=None)[:, None] + cand[None, :]  # audio time of each state
    A[beats > px[-1]] = (beats[beats > px[-1]] + (py[-1] - px[-1]))[:, None] + cand[None, :]
    # segment scores between consecutive beats for all (i, j) state pairs
    cost = np.zeros(S)
    back = np.zeros((K, S), dtype=int)
    prev_ratio = None
    ratio_log = np.zeros((K, S))
    for k in range(K - 1):
        b0, b1 = beats[k], beats[k + 1]
        sel = (on >= b0) & (on < b1) if k < K - 2 else (on >= b0)
        u = (on[sel] - b0) / (b1 - b0)
        ai = A[k][:, None, None]
        aj = A[k + 1][None, :, None]
        t = ai + u[None, None, :] * (aj - ai)
        seg = score(pit[sel], t) if sel.any() else np.zeros((S, S))
        dA = A[k + 1][None, :] - A[k][:, None]
        ratio = np.log(np.clip(dA, 1e-3, None) / (b1 - b0))
        valid = (dA > 0.55 * (b1 - b0)) & (dA < 1.8 * (b1 - b0))
        # penalty: change of tempo ratio w.r.t. the best predecessor's ratio
        lam = LAM_RUBATO if any(a <= b0 < b for a, b in RUBATO) else LAM_STEADY
        pen = lam * (ratio - (ratio_log[k][:, None] if k > 0 else 0.0)) ** 2
        tot = cost[:, None] + seg - pen
        tot[~valid] = -1e9
        back[k + 1] = np.argmax(tot, axis=0)
        cost = tot[back[k + 1], np.arange(S)]
        ratio_log[k + 1] = ratio[back[k + 1], np.arange(S)]
    path = np.zeros(K, dtype=int)
    path[-1] = int(np.argmax(cost))
    for k in range(K - 1, 0, -1):
        path[k - 1] = back[k][path[k]]
    anchors_a = A[np.arange(K), path]

    anchors_a = refine_rubato(anchors_a, beats, on, pit, Sdb)

    def warp(t):
        return np.interp(t, beats, anchors_a)

    # residuals: nearest salience peak within +-80 ms for each note
    def residuals(mapped):
        res = []
        for t, p in zip(mapped, pit):
            rows, w = note_rows(int(p))
            i0 = int(round((t - 0.08) * FPS_A)); i1 = int(round((t + 0.08) * FPS_A)) + 1
            if i0 < 0 or i1 >= F.shape[1]:
                continue
            prof = sum(wt * F[r, i0:i1] for r, wt in zip(rows, w))
            if prof.max() < 0.25:
                continue          # no clear onset in the audio (masked by vocals/drums)
            res.append((i0 + int(np.argmax(prof))) / FPS_A - t)
        return np.array(res)

    r_const = residuals(on + o0)
    r_warp = residuals(warp(on))
    for name, r in (("constant", r_const), ("warp", r_warp)):
        print(f"{name:9s}: matched {len(r)}/{len(on)} notes, median |err| "
              f"{np.median(np.abs(r))*1000:.1f} ms, within 1 frame@30fps "
              f"{np.mean(np.abs(r) < 1/30)*100:.1f} %")
    ratios = np.diff(anchors_a) / np.diff(beats)
    print("tempo ratio audio/MIDI per beat: min %.3f max %.3f" % (ratios.min(), ratios.max()))
    sx, sy = simplify(beats, anchors_a, 0.010)
    r_simp = residuals(np.interp(on, sx, sy))
    print(f"simplified warp: {len(sx)} anchors (from {len(beats)}), median |err| "
          f"{np.median(np.abs(r_simp))*1000:.1f} ms")
    print("WARP = [" + ", ".join(f"({a:.3f}, {b:.3f})" for a, b in zip(sx, sy)) + "]")
    json.dump({"warp": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(sx, sy)],
               "midi_beats": [round(float(b), 4) for b in beats],
               "audio_times": [round(float(a), 4) for a in anchors_a],
               "constant_offset": float(o0),
               "audio_duration": dur}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
