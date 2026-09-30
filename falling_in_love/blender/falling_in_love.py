"""
this is what falling in love feels like  --  Claude & the black cat at the piano
===============================================================================

A self-contained Blender script. It builds a cozy, candle-lit room with an
upright piano, the Claude avatar (a coral box with little legs and dynamic
eyes) and a black cat, and animates all of it to JVKE's "this is what falling
in love feels like":

  * every one of the 916 notes of the piano MIDI presses its key, frame-exact
  * the cat walks, sits and stretches on the keyboard; its front paws land on
    real notes of the right hand at the moment they sound
  * Claude "plays" the left hand: it hops onto the bass notes on the beat,
    shifts its weight, spins, and changes its eyes with the music
  * a small camera director cuts on bar lines and keeps both in frame

Tested with Blender 4.5 LTS and 5.0 (EEVEE and Cycles). Blender 4.2+ should work.


SETUP
-----
1. Put the files next to each other (this is the repository layout):
       falling_in_love/blender/falling_in_love.py      <- this script
       falling_in_love/midi/this_is_what_falling_in_love_feels_like_piano.mid
   and, optionally, the song as an .mp3 anywhere in falling_in_love/ (or set
   AUDIO_FILE below). Without audio the scene is built silently.
2. Blender > Scripting workspace > Open this file (or paste it) > Run Script.
   Building takes ~10-30 s. The scene "FallingInLove" is created; re-running
   the script rebuilds it and leaves your other scenes alone.
3. Press Space to play (audio scrubbing is on), or render:
   Render > Render Animation writes an MP4 with audio to OUTPUT_FILE.

Headless:  blender -b --python falling_in_love.py -- --render
           (or --save path.blend to only build and save)

Nothing has to be installed. The MIDI is read with `mido` if Blender's Python
has it (pip install mido into Blender's python), otherwise with the small
built-in Standard MIDI File reader below. Both give identical notes.


DATA CHOICES (why only the piano MIDI + the MP3 are used)
---------------------------------------------------------
* Piano MIDI  -> USED. 2 tracks (right hand 574 notes, left hand 342), in
  B major like the record, 119.4 s long like the record (120.3 s).
* "Violin" MIDI -> NOT USED. It is a string quartet arrangement (2 violins,
  viola, cello) in C major - a semitone away from the recording - 96 s long
  and in a different form. It cannot be synchronised to the song, and on
  screen there is only a piano.
* PDFs -> NOT USED. One is the score of that string quartet, the other a
  different fan arrangement (3/4, E-flat major). Sheet music has no timing.
* MP3 -> used as the soundtrack and, offline, as the timing reference.

AUDIO OFFSET / SYNC
-------------------
The MIDI follows the song's form but not its exact timing (its tempo map is
hand-made): the audio is 0.30 s behind the MIDI at the first note, drifts to
1.0 s during the first half and is 0.16 s behind after the break at 1:05.
tools/align_midi_to_audio.py measures this (pitch-specific onset salience +
Viterbi over all 217 beats) and produced the WARP table below. Every MIDI time
is mapped through it, so the audio strip simply starts at frame 1 with no
offset. Median note error: 16 ms, i.e. less than half a frame at 30 fps.
AUDIO_OFFSET is only a fine-tuning knob (e.g. for a Bluetooth delay).
"""

import bisect
import math
import os
import random
import struct
import sys
from collections import defaultdict, namedtuple

import bpy
import bmesh
from mathutils import Matrix, Vector

# ============================================================================
# 0. CONFIGURATION
# ============================================================================
FPS = 30                        # all timing is in seconds; any FPS works
FRAME_START = 1
RESOLUTION = (1920, 1080)
RESOLUTION_PERCENT = 100
ENGINE = "EEVEE"                # "EEVEE" or "CYCLES"
EEVEE_SAMPLES = 48
CYCLES_SAMPLES = 96
MOTION_BLUR = True
AUDIO_OFFSET = 0.0              # s; > 0 delays the animation against the audio
MIDI_FILE = ""                  # "" = search next to this script (see SETUP)
AUDIO_FILE = ""                 # "" = search for an .mp3/.wav with the song title
OUTPUT_FILE = "//render/falling_in_love.mp4"
KEY_GLOW = 1.2                  # warm glow of pressed keys (0 = off)
SEED = 7                        # blinks, ear twitches, handheld camera
SCENE_NAME = "FallingInLove"
SONG_LENGTH = 120.33            # s, used when no audio file is found

# Optional overrides for headless use, e.g.
#   FIL_AUDIO=song.mp3 FIL_RES=1280x720 blender -b --python falling_in_love.py -- --save out.blend
_env = os.environ.get
MIDI_FILE = _env("FIL_MIDI", MIDI_FILE)
AUDIO_FILE = _env("FIL_AUDIO", AUDIO_FILE)
OUTPUT_FILE = _env("FIL_OUTPUT", OUTPUT_FILE)
ENGINE = _env("FIL_ENGINE", ENGINE)
FPS = int(_env("FIL_FPS", FPS))
EEVEE_SAMPLES = int(_env("FIL_SAMPLES", EEVEE_SAMPLES))
CYCLES_SAMPLES = int(_env("FIL_SAMPLES", CYCLES_SAMPLES))
if _env("FIL_RES"):
    RESOLUTION = tuple(int(v) for v in _env("FIL_RES").lower().split("x"))

# MIDI seconds -> audio seconds (from tools/align_midi_to_audio.py). Linear
# in between, constant offset outside.
WARP = (
    (0.000, 0.302), (0.750, 1.033), (2.250, 2.612), (3.000, 3.111), (3.750, 4.156),
    (4.500, 4.969), (7.500, 8.510), (8.250, 9.218), (9.000, 9.776), (10.500, 10.960),
    (11.100, 11.807), (11.850, 12.794), (14.850, 15.680), (16.004, 16.889),
    (17.158, 18.022), (19.465, 20.355), (21.196, 22.150), (21.773, 22.687),
    (22.350, 23.269), (22.927, 23.876), (23.504, 24.453), (24.081, 25.010),
    (25.812, 26.775), (27.542, 28.511), (29.850, 30.893), (30.427, 31.430),
    (31.004, 32.007), (32.158, 33.116), (33.335, 34.338), (34.272, 35.285),
    (39.897, 40.864), (41.303, 42.230), (41.772, 42.719), (42.710, 43.631),
    (43.647, 44.579), (46.460, 47.356), (47.866, 48.807), (48.335, 49.241),
    (49.272, 50.158), (54.428, 55.284), (56.772, 57.592), (59.585, 60.410),
    (60.585, 61.353), (61.653, 62.333), (62.198, 62.821), (63.430, 63.883),
    (64.661, 65.025), (66.461, 66.621), (67.017, 67.167), (68.128, 68.302),
    (70.350, 70.527), (70.906, 71.067), (72.573, 72.778), (73.128, 73.298),
    (74.795, 74.998), (75.350, 75.528), (75.906, 76.088), (77.573, 77.819),
    (79.239, 79.424), (80.350, 80.559), (80.906, 81.099), (81.461, 81.619),
    (82.017, 82.185), (82.573, 82.725), (84.239, 84.410), (84.795, 84.990),
    (85.350, 85.550), (85.906, 86.070), (90.350, 90.501), (90.906, 91.111),
    (91.461, 91.682), (92.573, 92.772), (93.128, 93.292), (97.017, 97.188),
    (98.128, 98.333), (99.239, 99.408), (102.017, 102.184), (103.128, 103.274),
    (103.684, 103.849), (104.795, 104.954), (107.573, 107.765), (109.239, 109.390),
    (111.423, 111.612), (119.699, 119.802),
)

# Song form, as MIDI bar numbers (0-based, 4/4). Names are used below.
SECTION_BARS = (
    ("intro", 0),        # solo right hand, rubato
    ("run", 2),          # left hand enters, two-octave run up to F#6
    ("hold", 4),         # held F#6 - suspense
    ("stops", 5),        # four tutti chords with silence between (104 bpm)
    ("groove", 9),       # full groove
    ("light", 13),       # 128 bpm, right hand staccato chords only
    ("light_bass", 16),  # ... left hand joins
    ("ritard", 27),      # slows down
    ("break", 29),       # held F#6, silence
    ("stops2", 30),      # tutti chords again (108 bpm)
    ("chorus", 34),      # drums in: the chorus
    ("breakdown", 41),   # one bar without left hand
    ("chorus2", 42),     # final chorus
    ("outro", 50),       # soft, high register (116 bpm)
)

# ============================================================================
# 1. FILES
# ============================================================================
MIDI_NAME = "this_is_what_falling_in_love_feels_like_piano.mid"


def _script_dir():
    """Directory of this script, whether run with --python or from a text block."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    for text in bpy.data.texts:
        if text.filepath:
            return os.path.dirname(bpy.path.abspath(text.filepath))
    return None


def _search_dirs():
    dirs = []
    for d in (_script_dir(), os.path.dirname(bpy.data.filepath) if bpy.data.filepath else None,
              os.getcwd()):
        if d:
            for sub in ("", "..", "../midi", "midi", "../audio", "audio", "../.."):
                p = os.path.normpath(os.path.join(d, sub))
                if os.path.isdir(p) and p not in dirs:
                    dirs.append(p)
    return dirs


def find_midi():
    if MIDI_FILE:
        p = bpy.path.abspath(MIDI_FILE)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"MIDI_FILE not found: {p}")
        return p
    for d in _search_dirs():
        p = os.path.join(d, MIDI_NAME)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not find {MIDI_NAME}. Set MIDI_FILE at the top of the script.")


def find_audio():
    if AUDIO_FILE:
        p = bpy.path.abspath(AUDIO_FILE)
        return p if os.path.isfile(p) else None
    for d in _search_dirs():
        for name in sorted(os.listdir(d)):
            low = name.lower().replace("_", " ")
            if low.endswith((".mp3", ".wav", ".flac", ".ogg")) and "falling in love" in low:
                return os.path.join(d, name)
    return None


# ============================================================================
# 2. MIDI  ->  notes in audio seconds
# ============================================================================
Note = namedtuple("Note", "on off pitch vel hand")     # hand: "R" or "L"


def _read_with_mido(path):
    import mido  # optional
    mf = mido.MidiFile(path)
    tracks = []
    for tr in mf.tracks:
        tick, ev = 0, []
        for msg in tr:
            tick += msg.time
            if msg.type == "set_tempo":
                ev.append((tick, "tempo", msg.tempo))
            elif msg.type == "note_on" and msg.velocity > 0:
                ev.append((tick, "on", msg.channel, msg.note, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                ev.append((tick, "off", msg.channel, msg.note))
        tracks.append(ev)
    return mf.ticks_per_beat, tracks


def _read_builtin(path):
    """Minimal Standard MIDI File reader (format 0/1, PPQ division)."""
    data = open(path, "rb").read()
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file: " + path)
    hlen = struct.unpack(">I", data[4:8])[0]
    _fmt, ntrk, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    pos, tracks = 8 + hlen, []

    def varlen(buf, i):
        v = 0
        while True:
            b = buf[i]
            i += 1
            v = (v << 7) | (b & 0x7F)
            if b < 0x80:
                return v, i

    while pos + 8 <= len(data) and len(tracks) < ntrk:
        cid = data[pos:pos + 4]
        clen = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + clen]
        pos += 8 + clen
        if cid != b"MTrk":
            continue
        ev, i, tick, status = [], 0, 0, 0
        while i < len(body):
            delta, i = varlen(body, i)
            tick += delta
            b = body[i]
            if b == 0xFF:                                  # meta event
                mtype = body[i + 1]
                ln, i = varlen(body, i + 2)
                if mtype == 0x51:
                    ev.append((tick, "tempo", int.from_bytes(body[i:i + ln], "big")))
                i += ln
                if mtype == 0x2F:
                    break
                continue
            if b in (0xF0, 0xF7):                          # sysex
                ln, i = varlen(body, i + 1)
                i += ln
                continue
            if b & 0x80:
                status = b
                i += 1
            hi, ch = status & 0xF0, status & 0x0F
            n = 1 if hi in (0xC0, 0xD0) else 2
            d = body[i:i + n]
            i += n
            if hi == 0x90 and d[1] > 0:
                ev.append((tick, "on", ch, d[0], d[1]))
            elif hi == 0x80 or (hi == 0x90 and d[1] == 0):
                ev.append((tick, "off", ch, d[0]))
        tracks.append(ev)
    return division, tracks


def load_midi(path):
    """-> (notes in MIDI seconds, beat times in MIDI seconds, reader name)"""
    try:
        tpb, tracks = _read_with_mido(path)
        reader = "mido"
    except ImportError:
        tpb, tracks = _read_builtin(path)
        reader = "built-in reader"
    tempos = sorted((e[0], e[2]) for tr in tracks for e in tr if e[1] == "tempo")
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, 500000))
    seg_tick, seg_sec, seg_tempo = [], [], []
    sec, last_tick, last_tempo = 0.0, 0, tempos[0][1]
    for tick, tempo in tempos:
        sec += (tick - last_tick) * last_tempo / 1e6 / tpb
        seg_tick.append(tick)
        seg_sec.append(sec)
        seg_tempo.append(tempo)
        last_tick, last_tempo = tick, tempo

    def t2s(tick):
        k = bisect.bisect_right(seg_tick, tick) - 1
        return seg_sec[k] + (tick - seg_tick[k]) * seg_tempo[k] / 1e6 / tpb

    notes, last = [], 0
    hands = "RL"
    for ti, tr in enumerate(t for t in tracks if any(e[1] == "on" for e in t)):
        open_notes = defaultdict(list)
        for e in tr:
            if e[1] == "on":
                open_notes[(e[2], e[3])].append((e[0], e[4]))
            elif e[1] == "off" and open_notes[(e[2], e[3])]:
                t0, vel = open_notes[(e[2], e[3])].pop(0)
                notes.append(Note(t2s(t0), t2s(e[0]), e[3], vel, hands[min(ti, 1)]))
                last = max(last, e[0])
    notes.sort(key=lambda n: (n.on, n.pitch))
    beats = [t2s(k * tpb) for k in range(last // tpb + 2)]
    return notes, beats, reader


_WX = [w[0] for w in WARP]
_WY = [w[1] for w in WARP]


def to_audio(t_midi):
    """MIDI seconds -> audio seconds (piecewise linear WARP)."""
    if t_midi <= _WX[0]:
        return _WY[0] + (t_midi - _WX[0])
    if t_midi >= _WX[-1]:
        return _WY[-1] + (t_midi - _WX[-1])
    k = bisect.bisect_right(_WX, t_midi) - 1
    u = (t_midi - _WX[k]) / (_WX[k + 1] - _WX[k])
    return _WY[k] + u * (_WY[k + 1] - _WY[k])


def frame_of(t):
    """audio seconds -> (fractional) frame"""
    return FRAME_START + (t + AUDIO_OFFSET) * FPS


def time_of(f):
    """frame -> audio seconds"""
    return (f - FRAME_START) / FPS - AUDIO_OFFSET


class Music:
    """Everything the animation needs to know about the song, in audio seconds."""

    def __init__(self, midi_path, song_length):
        raw, beats_m, self.reader = load_midi(midi_path)
        self.notes = [Note(to_audio(n.on), to_audio(n.off), n.pitch, n.vel, n.hand) for n in raw]
        self.beats = [to_audio(b) for b in beats_m]
        self.downbeats = self.beats[::4]
        self.length = song_length
        self.sections = {}
        for i, (name, bar) in enumerate(SECTION_BARS):
            t0 = self.downbeats[bar]
            t1 = (self.downbeats[SECTION_BARS[i + 1][1]] if i + 1 < len(SECTION_BARS)
                  else song_length)
            self.sections[name] = (t0, t1)
        self.by_pitch = defaultdict(list)
        for n in self.notes:
            self.by_pitch[n.pitch].append(n)
        # right-hand onsets grouped into chords (within 30 ms)
        self.rh_chords = self._group([n for n in self.notes if n.hand == "R"])
        self.lh_chords = self._group([n for n in self.notes if n.hand == "L"])

    @staticmethod
    def _group(notes):
        out = []
        for n in notes:
            if out and n.on - out[-1][0] < 0.03:
                out[-1][1].append(n)
            else:
                out.append((n.on, [n]))
        return out

    def section_at(self, t):
        for name, (t0, t1) in self.sections.items():
            if t0 <= t < t1:
                return name
        return "intro" if t < self.sections["intro"][1] else "outro"

    def bar_time(self, section, bar_offset=0):
        bar = dict(SECTION_BARS)[section] + bar_offset
        return self.downbeats[min(bar, len(self.downbeats) - 1)]

    def beat_phase(self, t):
        """continuous beat count at time t (integer on the beats)"""
        b = self.beats
        if t <= b[0]:
            return (t - b[0]) / (b[1] - b[0])
        if t >= b[-1]:
            return len(b) - 1 + (t - b[-1]) / (b[-1] - b[-2])
        k = bisect.bisect_right(b, t) - 1
        return k + (t - b[k]) / (b[k + 1] - b[k])

    def beat_len(self, t):
        k = min(max(bisect.bisect_right(self.beats, t) - 1, 0), len(self.beats) - 2)
        return self.beats[k + 1] - self.beats[k]

    def beats_in(self, t0, t1):
        return [b for b in self.beats if t0 - 1e-6 <= b < t1 - 1e-6]

    def lowest_lh_near(self, t, tol=0.12):
        best = None
        for t_on, chord in self.lh_chords:
            if abs(t_on - t) <= tol:
                p = min(n.pitch for n in chord)
                best = p if best is None else min(best, p)
        return best


# ============================================================================
# 3. SMALL HELPERS: math, baking, meshes, materials
# ============================================================================
def clamp(x, a, b):
    return a if x < a else b if x > b else x


def lerp(a, b, u):
    return a + (b - a) * u


def smooth(u):
    u = clamp(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def smoother(u):
    u = clamp(u, 0.0, 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def gaussian_smooth(values, sigma_frames):
    """Gaussian filter of a list (edges clamped). Pure python, O(n*k)."""
    if sigma_frames <= 0.01:
        return list(values)
    r = int(3 * sigma_frames) + 1
    ker = [math.exp(-0.5 * (i / sigma_frames) ** 2) for i in range(-r, r + 1)]
    s = sum(ker)
    ker = [k / s for k in ker]
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        acc = 0.0
        for j, k in enumerate(ker):
            acc += k * values[min(max(i + j - r, 0), n - 1)]
        out[i] = acc
    return out


def rate_limit(values, max_step):
    out = list(values)
    for i in range(1, len(out)):
        out[i] = clamp(out[i], out[i - 1] - max_step, out[i - 1] + max_step)
    return out


def wobble(t, seed, freqs=(0.37, 0.61, 1.13)):
    """smooth pseudo-noise in [-1, 1] (sum of incommensurate sines)"""
    return sum(math.sin(2 * math.pi * f * t + seed * (i + 1) * 1.7)
               for i, f in enumerate(freqs)) / len(freqs)


def fcurve(id_data, data_path, index=0):
    """Get/create an F-curve on id_data. Works with the slotted actions of
    Blender 4.4+/5.x and the legacy action API of 4.2/4.3."""
    ad = id_data.animation_data or id_data.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{id_data.name}Action")
    act = ad.action
    if hasattr(act, "fcurve_ensure_for_datablock"):
        return act.fcurve_ensure_for_datablock(id_data, data_path, index=index)
    fc = act.fcurves.find(data_path, index=index)
    return fc or act.fcurves.new(data_path, index=index)


_IPO = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}


def _simplify(frames, values, tol):
    """Ramer-Douglas-Peucker: keep the fewest keys that reproduce the curve
    within tol (linear interpolation)."""
    n = len(frames)
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        fi, fj, vi, vj = frames[i], frames[j], values[i], values[j]
        best, bk = -1.0, -1
        for k in range(i + 1, j):
            v = vi + (vj - vi) * (frames[k] - fi) / (fj - fi)
            d = abs(values[k] - v)
            if d > best:
                best, bk = d, k
        if best > tol:
            keep[bk] = True
            stack += [(i, bk), (bk, j)]
    return [k for k in range(n) if keep[k]]


def bake(id_data, data_path, index, frames, values, tol=1e-4, interp="LINEAR"):
    """Write (frame, value) samples as keyframes, simplified within tol."""
    idx = _simplify(frames, values, tol) if tol else range(len(frames))
    fc = fcurve(id_data, data_path, index)
    kp = fc.keyframe_points
    start = len(kp)
    kp.add(len(idx))
    co = []
    for k in idx:
        co += [frames[k], values[k]]
    if start == 0:
        kp.foreach_set("co", co)
        kp.foreach_set("interpolation", [_IPO[interp]] * len(idx))
    else:
        for j, k in enumerate(idx):
            kp[start + j].co = (frames[k], values[k])
            kp[start + j].interpolation = interp
    fc.update()
    return fc


def bake_vec(obj, data_path, frames, vecs, tol=1e-4, axes=(0, 1, 2)):
    for a in axes:
        bake(obj, data_path, a, frames, [v[a] for v in vecs], tol)


def noise_fcurve(id_data, data_path, index, base, strength, scale, phase):
    """A constant F-curve with a Noise modifier (candle flicker etc.)."""
    fc = fcurve(id_data, data_path, index)
    fc.keyframe_points.add(1)
    fc.keyframe_points[0].co = (FRAME_START, base)
    mod = fc.modifiers.new("NOISE")
    mod.blend_type = "ADD"
    mod.strength = strength
    mod.scale = scale
    mod.phase = phase
    return fc


# ---- meshes ---------------------------------------------------------------
def to_mesh(name, bm, smooth_shade=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth_shade:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def bm_box(sx, sy, sz, center=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * sx + center[0], v.co.y * sy + center[1], v.co.z * sz + center[2]))
    return bm


def bm_sphere(r, scale=(1, 1, 1), center=(0, 0, 0), u=24, v=16):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r)
    for vert in bm.verts:
        vert.co = Vector((vert.co.x * scale[0] + center[0], vert.co.y * scale[1] + center[1],
                          vert.co.z * scale[2] + center[2]))
    return bm


def bm_lathe(length, profile, axis="X", rings=24, segs=20, start=0.0):
    """Surface of revolution along an axis. profile(u) -> (ry, rz, dz) for
    u in [0, 1]. Closed with poles at both ends."""
    bm = bmesh.new()
    ring_verts = []
    for i in range(1, rings):
        u = i / rings
        ry, rz, dz = profile(u)
        ring = []
        for j in range(segs):
            a = 2 * math.pi * j / segs
            p = (start + u * length, math.cos(a) * ry, math.sin(a) * rz + dz)
            ring.append(bm.verts.new(_axis(p, axis)))
        ring_verts.append(ring)
    _, _, dz0 = profile(0.0)
    _, _, dz1 = profile(1.0)
    p0 = bm.verts.new(_axis((start, 0, dz0), axis))
    p1 = bm.verts.new(_axis((start + length, 0, dz1), axis))
    for a, b in zip(ring_verts[:-1], ring_verts[1:]):
        for j in range(segs):
            bm.faces.new((a[j], a[(j + 1) % segs], b[(j + 1) % segs], b[j]))
    for j in range(segs):
        bm.faces.new((p0, ring_verts[0][(j + 1) % segs], ring_verts[0][j]))
        bm.faces.new((p1, ring_verts[-1][j], ring_verts[-1][(j + 1) % segs]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _axis(p, axis):
    x, y, z = p
    if axis == "X":
        return (x, y, z)
    if axis == "Y":
        return (y, x, z)            # length along +Y
    return (y, z, x)                # length along +Z


def pill(r, p=3.0):
    """rounded-cylinder profile for bm_lathe"""
    return lambda u: ((r * (1 - abs(2 * u - 1) ** p) ** (1 / p)),) * 2 + (0.0,)


def bm_cylinder(r, h, segs=24, z0=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r, radius2=r, depth=h)
    for v in bm.verts:
        v.co.z += h / 2 + z0
    return bm


def bm_outline(points, thickness):
    """Flat decal: 2D outline (x, z) in the XZ plane, extruded along -Y."""
    bm = bmesh.new()
    vs = [bm.verts.new((x, 0.0, z)) for x, z in points]
    f = bm.faces.new(vs)
    ext = bmesh.ops.extrude_face_region(bm, geom=[f])
    moved = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(0, -thickness, 0), verts=moved)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def outline_capsule(w, h, n=10):
    pts, r = [], w / 2
    for i in range(n + 1):
        a = math.pi * i / n
        pts.append((r * math.cos(a), h / 2 - r + r * math.sin(a)))
    for i in range(n + 1):
        a = math.pi + math.pi * i / n
        pts.append((r * math.cos(a), -h / 2 + r + r * math.sin(a)))
    return pts


def outline_arc(radius, width, a0, a1, n=14, dz=0.0):
    outer = [((radius + width / 2) * math.cos(a), (radius + width / 2) * math.sin(a) + dz)
             for a in [a0 + (a1 - a0) * i / n for i in range(n + 1)]]
    inner = [((radius - width / 2) * math.cos(a), (radius - width / 2) * math.sin(a) + dz)
             for a in [a1 + (a0 - a1) * i / n for i in range(n + 1)]]
    return outer + inner


def outline_heart(size, n=40):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        z = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((x * size / 34, z * size / 34 + size * 0.05))
    return pts[::-1]


def outline_circle(r, n=24):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]


# ---- objects ----------------------------------------------------------------
class Builder:
    def __init__(self, scene):
        self.scene = scene
        self.colls = {}

    def coll(self, name):
        if name not in self.colls:
            c = bpy.data.collections.new(f"{SCENE_NAME}.{name}")
            self.scene.collection.children.link(c)
            self.colls[name] = c
        return self.colls[name]

    def obj(self, name, data, coll, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1),
            mat=None):
        o = bpy.data.objects.new(name, data)
        self.coll(coll).objects.link(o)
        if parent is not None:
            o.parent = parent
        o.location = loc
        o.rotation_euler = rot
        o.scale = scale
        if mat is not None and data is not None and hasattr(data, "materials"):
            data.materials.append(mat)
        return o

    def empty(self, name, coll, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), size=0.03,
              shape="PLAIN_AXES"):
        o = self.obj(name, None, coll, parent, loc, rot)
        o.empty_display_size = size
        o.empty_display_type = shape
        return o

    def mesh(self, name, bm, coll, mat=None, parent=None, loc=(0, 0, 0), rot=(0, 0, 0),
             scale=(1, 1, 1), smooth_shade=True):
        return self.obj(name, to_mesh(name, bm, smooth_shade), coll, parent, loc, rot, scale, mat)


def add_bevel(o, width, segments=4):
    m = o.modifiers.new("Bevel", "BEVEL")
    m.width = width
    m.segments = segments
    m.limit_method = "NONE"
    return m


# ---- materials ------------------------------------------------------------
def srgb(hexstr):
    """'#D97757' -> linear RGB tuple"""
    h = hexstr.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple((x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def _set(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def new_material(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:          # Blender < 5.0
        m.use_nodes = True
    return m


def principled(name, color, rough=0.5, metallic=0.0, **kw):
    m = new_material(name)
    b = m.node_tree.nodes.get("Principled BSDF")
    _set(b, "Base Color", (*color, 1.0))
    _set(b, "Roughness", rough)
    _set(b, "Metallic", metallic)
    for k, v in kw.items():
        _set(b, k.replace("_", " "), v)
    m.diffuse_color = (*color, 1.0)
    return m


def emission(name, color, strength):
    m = new_material(name)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*color, 1.0)
    return m


def wood(name, dark, light, scale=6.0, coat=0.6, rough=0.35, stretch=(1, 12, 1)):
    """Procedural wood: stretched noise + wave bands between two colours."""
    m = principled(name, dark, rough, Coat_Weight=coat, Coat_Roughness=0.15)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = stretch
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 6.0
    _set(nz, "Distortion", 0.6)
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.inputs["Scale"].default_value = scale * 0.6
    wv.inputs["Distortion"].default_value = 2.5
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].color = (*light, 1)
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY"
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Color"], wv.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], mix.inputs[0])
    nt.links.new(nz.outputs["Fac"], mix.inputs[1])
    nt.links.new(mix.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def key_material(name, color, rough):
    """Piano key: glossy, plus a warm glow driven by the object's "glow"
    custom property (animated per key)."""
    m = principled(name, color, rough, Coat_Weight=0.5, Coat_Roughness=0.08)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "OBJECT"
    at.attribute_name = "glow"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = KEY_GLOW
    nt.links.new(at.outputs["Fac"], mul.inputs[0])
    _set(b, "Emission Color", (1.0, 0.52, 0.22, 1))
    nt.links.new(mul.outputs[0], b.inputs["Emission Strength"])
    return m


def twinkle_material(name, color, strength):
    """Fairy-light bulb: emission that twinkles per bulb (Object Info random +
    a driver on the current frame; a simple expression, no Python needed)."""
    m = emission(name, color, strength)
    nt = m.node_tree
    em = nt.nodes["Emission"]
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    val = nt.nodes.new("ShaderNodeValue")
    drv = val.outputs[0].driver_add("default_value").driver
    drv.type = "SCRIPTED"
    drv.expression = "frame"
    t = nt.nodes.new("ShaderNodeMath")
    t.operation = "MULTIPLY_ADD"
    t.inputs[1].default_value = 0.9 / FPS * 2 * math.pi       # ~0.9 Hz
    r = nt.nodes.new("ShaderNodeMath")
    r.operation = "MULTIPLY"
    r.inputs[1].default_value = 40.0
    s = nt.nodes.new("ShaderNodeMath")
    s.operation = "SINE"
    k = nt.nodes.new("ShaderNodeMath")
    k.operation = "MULTIPLY_ADD"
    k.inputs[1].default_value = 0.35 * strength
    k.inputs[2].default_value = 0.8 * strength
    nt.links.new(oi.outputs["Random"], r.inputs[0])
    nt.links.new(val.outputs[0], t.inputs[0])
    nt.links.new(r.outputs[0], t.inputs[2])
    nt.links.new(t.outputs[0], s.inputs[0])
    nt.links.new(s.outputs[0], k.inputs[0])
    nt.links.new(k.outputs[0], em.inputs["Strength"])
    return m


def paper_material(name):
    """Sheet music: warm paper with faint staves."""
    m = principled(name, (0.24, 0.21, 0.16), 0.8)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.wave_type = "BANDS"
    wv.bands_direction = "Y"
    wv.inputs["Scale"].default_value = 38.0
    wv2 = nt.nodes.new("ShaderNodeTexWave")
    wv2.wave_type = "BANDS"
    wv2.bands_direction = "Y"
    wv2.inputs["Scale"].default_value = 7.0
    lines = nt.nodes.new("ShaderNodeMath")
    lines.operation = "GREATER_THAN"
    lines.inputs[1].default_value = 0.965
    staves = nt.nodes.new("ShaderNodeMath")
    staves.operation = "GREATER_THAN"
    staves.inputs[1].default_value = 0.45
    both = nt.nodes.new("ShaderNodeMath")
    both.operation = "MULTIPLY"
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.24, 0.21, 0.16, 1)
    mix.inputs[7].default_value = (0.22, 0.19, 0.16, 1)
    nt.links.new(tc.outputs["UV"], wv.inputs["Vector"])
    nt.links.new(tc.outputs["UV"], wv2.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], lines.inputs[0])
    nt.links.new(wv2.outputs["Fac"], staves.inputs[0])
    nt.links.new(lines.outputs[0], both.inputs[0])
    nt.links.new(staves.outputs[0], both.inputs[1])
    nt.links.new(both.outputs[0], mix.inputs[0])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    return m


def look_rotation(src, dst, prev=None):
    q = (Vector(dst) - Vector(src)).to_track_quat("-Z", "Y")
    return q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")


# ============================================================================
# 4. THE PIANO
# ============================================================================
K = 1.5                          # piano scale vs. a real one (toy-like proportions)
WHITE_W = 0.0235 * K             # key pitch
WHITE_L = 0.150 * K
WHITE_T = 0.020 * K
BLACK_W = 0.0130 * K
BLACK_L = 0.095 * K
BLACK_H = 0.012 * K              # black key top above white key top
KEY_GAP = 0.0012 * K
KEY_Z = 0.95                     # white key top above the floor
PIVOT_Y = 0.10                   # keys hinge behind the fallboard
PRESS_DEPTH = 0.0105 * K         # front-edge drop of a pressed key
LOWEST, HIGHEST = 21, 108
WHITE_PC = (0, 2, 4, 5, 7, 9, 11)
BLACK_SHIFT = {1: -0.15, 3: 0.15, 6: -0.2, 8: 0.0, 10: 0.2}


def is_black(p):
    return p % 12 not in WHITE_PC


_WIDX = {}
_i = 0
for _p in range(LOWEST, HIGHEST + 2):
    _WIDX[_p] = _i
    if not is_black(_p):
        _i += 1
MID_IDX = _WIDX[60]


def key_x(p):
    """x of a key centre; middle C at x = 0"""
    if not is_black(p):
        return (_WIDX[p] - MID_IDX) * WHITE_W
    return (_WIDX[p] - MID_IDX - 0.5) * WHITE_W + BLACK_SHIFT[p % 12] * BLACK_W


def key_top(p):
    return KEY_Z + (BLACK_H if is_black(p) else 0.0)


def key_theta(p):
    """rotation (rad) of a fully pressed key about its hinge"""
    length = PIVOT_Y + (BLACK_L if is_black(p) else WHITE_L)
    return math.asin((PRESS_DEPTH * (0.85 if is_black(p) else 1.0)) / length)


def key_at(x, y):
    """which key is under the point (x, y)? (y < 0 is on the keys)"""
    if y > -BLACK_L:
        for p in range(LOWEST, HIGHEST + 1):
            if is_black(p) and abs(x - key_x(p)) < BLACK_W / 2:
                return p
    idx = int(math.floor(x / WHITE_W + MID_IDX + 0.5))
    for p in range(LOWEST, HIGHEST + 1):
        if not is_black(p) and _WIDX[p] == idx:
            return p
    return 60


KB_X0 = key_x(LOWEST) - WHITE_W / 2         # left edge of the keyboard
KB_X1 = key_x(HIGHEST) + WHITE_W / 2        # right edge


def build_piano(B):
    mats = {
        "case": wood("Piano.Wood", srgb("#1c0c06"), srgb("#3b1d10"), scale=4, coat=0.8, rough=0.32,
                     stretch=(0.25, 1, 1)),
        "white": key_material("Key.White", srgb("#ede4d3"), 0.28),
        "black": key_material("Key.Black", srgb("#0d0b0b"), 0.22),
        "felt": principled("Felt.Red", srgb("#6e1320"), 0.9),
        "brass": principled("Brass", srgb("#b8863b"), 0.25, metallic=1.0),
        "paper": paper_material("SheetMusic"),
    }
    cheek = 0.075
    x0, x1 = KB_X0 - cheek, KB_X1 + cheek
    cx, w = (x0 + x1) / 2, x1 - x0
    back = 0.55                                  # depth of the upper case
    top = KEY_Z + 0.88
    parts = [
        # name, size, center
        ("Piano.Keybed", (w, WHITE_L + 0.06, 0.05), (cx, -WHITE_L / 2 + 0.01, KEY_Z - WHITE_T - 0.03)),
        ("Piano.KeySlip", (KB_X1 - KB_X0 + 0.004, 0.025, 0.05), (cx, -WHITE_L - 0.0135, KEY_Z - WHITE_T - 0.012)),
        ("Piano.CheekL", (cheek, WHITE_L + 0.05, 0.10), (x0 + cheek / 2, -WHITE_L / 2 - 0.005, KEY_Z - 0.02)),
        ("Piano.CheekR", (cheek, WHITE_L + 0.05, 0.10), (x1 - cheek / 2, -WHITE_L / 2 - 0.005, KEY_Z - 0.02)),
        ("Piano.Fallboard", (KB_X1 - KB_X0, 0.035, 0.11), (cx, 0.0175, KEY_Z + 0.04)),
        ("Piano.Upper", (w, back, top - KEY_Z - 0.09), (cx, back / 2 + 0.035, (top + KEY_Z + 0.09) / 2)),
        ("Piano.Lid", (w + 0.04, back + 0.05, 0.035), (cx, back / 2 + 0.02, top + 0.0175)),
        ("Piano.Lower", (w - 0.04, 0.05, KEY_Z - 0.09), (cx, 0.10, (KEY_Z - 0.09) / 2)),
        ("Piano.Body", (w, back - 0.12, KEY_Z - 0.05), (cx, back / 2 + 0.1, (KEY_Z - 0.05) / 2)),
        ("Piano.LegL", (0.07, 0.07, KEY_Z - 0.08), (x0 + 0.05, -WHITE_L + 0.02, (KEY_Z - 0.08) / 2)),
        ("Piano.LegR", (0.07, 0.07, KEY_Z - 0.08), (x1 - 0.05, -WHITE_L + 0.02, (KEY_Z - 0.08) / 2)),
        ("Piano.Rail", (w - 0.1, 0.04, 0.03), (cx, -0.005, KEY_Z + 0.105)),
    ]
    for name, size, c in parts:
        o = B.mesh(name, bm_box(*size), "Piano", mats["case"], loc=c, smooth_shade=False)
        add_bevel(o, 0.006, 3)
    # felt strip behind the keys
    B.mesh("Piano.Felt", bm_box(KB_X1 - KB_X0, 0.012, 0.012), "Piano", mats["felt"],
           loc=(cx, 0.004, KEY_Z + 0.004), smooth_shade=False)
    # music desk with two pages, tilted back
    desk = B.mesh("Piano.Desk", bm_box(0.62, 0.02, 0.28), "Piano", mats["case"],
                  loc=(0.02, 0.02, KEY_Z + 0.3), rot=(math.radians(-12), 0, 0), smooth_shade=False)
    add_bevel(desk, 0.005, 2)
    for i, dx in enumerate((-0.128, 0.128)):
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
        for v in bm.verts:
            v.co = Vector((v.co.x * 0.25, 0, v.co.y * 0.34))
        me = to_mesh(f"Page{i}", bm, False)
        uv = me.uv_layers.new()
        for li, loop in enumerate(me.loops):
            co = me.vertices[loop.vertex_index].co
            uv.data[li].uv = (co.x / 0.25 + 0.5, co.z / 0.34 + 0.5)
        B.obj(f"Piano.Page{i}", me, "Piano", parent=desk, loc=(dx, -0.0115, 0.0),
                     rot=(0, math.radians(2 if i else -2), 0))
        me.materials.append(mats["paper"])
    # pedals
    for i, dx in enumerate((-0.09, 0.0, 0.09)):
        B.mesh(f"Piano.Pedal{i}", bm_box(0.035, 0.12, 0.015), "Piano", mats["brass"],
               loc=(cx + dx, 0.02, 0.06))
    # keys
    keys = {}
    for p in range(LOWEST, HIGHEST + 1):
        if is_black(p):
            bm = bm_box(BLACK_W, BLACK_L, BLACK_H + 0.012 * K,
                        (0, -BLACK_L / 2 - PIVOT_Y, (BLACK_H - 0.012 * K) / 2))
            # slightly tapered top
            for v in bm.verts:
                if v.co.z > 0:
                    v.co.x *= 0.82
                    if v.co.y < -PIVOT_Y - BLACK_L + 0.001:
                        v.co.y += 0.006
            mat = mats["black"]
        else:
            bm = bm_box(WHITE_W - KEY_GAP, WHITE_L, WHITE_T, (0, -WHITE_L / 2 - PIVOT_Y, -WHITE_T / 2))
            mat = mats["white"]
        o = B.mesh(f"Key.{p:03d}", bm, "Keys", mat, loc=(key_x(p), PIVOT_Y, KEY_Z), smooth_shade=False)
        add_bevel(o, 0.0012 if not is_black(p) else 0.0016, 2)
        o["glow"] = 0.0
        keys[p] = o
    return keys, mats


def build_room(B, mats):
    floor_mat = wood("Floor.Wood", srgb("#3b2416"), srgb("#7a4d2c"), scale=2.5, coat=0.3, rough=0.45,
                     stretch=(1, 8, 1))
    wall = principled("Wall.Plaster", srgb("#3d2a21"), 0.85)
    nt = wall.node_tree
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 18.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    B.mesh("Room.Floor", bm_box(9, 7, 0.02, (0, 0.5, -0.01)), "Room", floor_mat, smooth_shade=False)
    B.mesh("Room.Wall", bm_box(9, 0.05, 3.4, (0, 0.66, 1.7)), "Room", wall, smooth_shade=False)
    B.mesh("Room.WallL", bm_box(0.05, 7, 3.4, (-3.2, -1.5, 1.7)), "Room", wall, smooth_shade=False)
    rug = principled("Rug", srgb("#5c1e1e"), 0.95)
    B.mesh("Room.Rug", bm_cylinder(1.1, 0.01, 64), "Room", rug, loc=(0.1, -0.9, 0.0))
    # window with night sky (cool rim light source in the background)
    sky = emission("Window.Night", srgb("#1b2a4a"), 1.4)
    B.mesh("Room.Window", bm_box(0.9, 0.01, 1.2, (1.95, 0.635, 1.75)), "Room", sky, smooth_shade=False)
    frame = principled("Window.Frame", srgb("#d9cbb4"), 0.6)
    for name, size, c in (("Top", (1.0, 0.04, 0.05), (1.95, 0.62, 2.37)),
                          ("Bottom", (1.05, 0.08, 0.05), (1.95, 0.61, 1.13)),
                          ("L", (0.05, 0.04, 1.25), (1.47, 0.62, 1.75)),
                          ("R", (0.05, 0.04, 1.25), (2.43, 0.62, 1.75)),
                          ("Mid", (0.03, 0.04, 1.2), (1.95, 0.62, 1.75)),
                          ("Cross", (0.9, 0.04, 0.03), (1.95, 0.62, 1.95))):
        B.mesh(f"Room.WindowFrame{name}", bm_box(*size, c), "Room", frame, smooth_shade=False)
    # a picture frame above the piano
    B.mesh("Room.Picture", bm_box(0.5, 0.03, 0.38, (-0.35, 0.625, 2.3)), "Room", mats["case"],
           smooth_shade=False)
    art = principled("Picture.Art", srgb("#c9764f"), 0.7)
    B.mesh("Room.PictureArt", bm_box(0.42, 0.01, 0.30, (-0.35, 0.607, 2.3)), "Room", art,
           smooth_shade=False)


def build_candles(B, mats):
    """Candles on the piano, each with a flickering point light."""
    wax = principled("Candle.Wax", srgb("#efe2c8"), 0.5, Subsurface_Weight=0.6,
                     Subsurface_Radius=(0.08, 0.04, 0.02))
    flame_mat = emission("Candle.Flame", (1.0, 0.55, 0.18), 18.0)
    top = KEY_Z + 0.915
    spots = [  # x, y, z(base), height, radius
        (-0.62, 0.22, top, 0.16, 0.024), (-0.52, 0.30, top, 0.11, 0.022), (-0.43, 0.2, top, 0.07, 0.02),
        (0.78, 0.25, top, 0.13, 0.024), (0.88, 0.2, top, 0.09, 0.021),
        (KB_X0 - 0.037, -0.13, KEY_Z + 0.03, 0.07, 0.018),
        (KB_X1 + 0.037, -0.13, KEY_Z + 0.03, 0.09, 0.018),
    ]
    for i, (x, y, z, h, r) in enumerate(spots):
        hold = B.mesh(f"Candle{i}.Holder", bm_cylinder(r * 1.6, 0.012), "Candles", mats["brass"], loc=(x, y, z))
        add_bevel(hold, 0.003, 2)
        wx = B.mesh(f"Candle{i}.Wax", bm_cylinder(r, h), "Candles", wax, loc=(x, y, z + 0.012))
        add_bevel(wx, 0.004, 3)
        fl = B.mesh(f"Candle{i}.Flame",
                    bm_lathe(0.034, lambda u: (0.0075 * math.sin(math.pi * u) ** 0.7 * (1.25 - u),) * 2 + (0.0,),
                             axis="Z", rings=12, segs=12), "Candles", flame_mat,
                    loc=(x, y, z + 0.012 + h + 0.003))
        noise_fcurve(fl, "scale", 2, 1.0, 0.35, 6.0, i * 13.0)
        noise_fcurve(fl, "scale", 0, 1.0, 0.15, 4.0, i * 7.0)
        ld = bpy.data.lights.new(f"Candle{i}.Light", "POINT")
        ld.energy = 5.0 if z > KEY_Z + 0.5 else 2.5
        ld.color = (1.0, 0.52, 0.2)
        ld.shadow_soft_size = 0.01
        ld.use_shadow = False                    # small, many, and expensive in EEVEE
        B.obj(f"Candle{i}.Light", ld, "Candles", loc=(x, y, z + 0.012 + h + 0.03))
        noise_fcurve(ld, "energy", 0, ld.energy, ld.energy * 0.35, 3.0, i * 21.0)


def build_fairy_lights(B):
    """Two strings of lights: one draped along the key rail right behind the
    keys (soft bokeh behind the characters), one along the top of the case."""
    bulb_mat = twinkle_material("FairyBulb", (1.0, 0.62, 0.3), 14.0)
    wire_mat = principled("FairyWire", (0.02, 0.02, 0.02), 0.5)
    bulb = to_mesh("FairyBulb", bm_sphere(0.0065, (1, 1, 1.3), u=10, v=8))
    bulb.materials.append(bulb_mat)
    strings = (("Low", -0.036, KEY_Z + 0.175, 0.035, 9), ("High", 0.026, KEY_Z + 0.86, 0.12, 6))
    for name, y, z_top, sag_max, swags in strings:
        anchors = [KB_X0 + 0.05 + i * (KB_X1 - KB_X0 - 0.1) / swags for i in range(swags + 1)]
        pts = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            for j in range(10):
                u = j / 10
                pts.append((lerp(a, b, u), y, z_top - sag_max * (1 - (2 * u - 1) ** 2)))
        pts.append((anchors[-1], y, z_top))
        cu = bpy.data.curves.new(f"FairyWire{name}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.0012
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (*c, 1.0)
        B.obj(f"Fairy.Wire{name}", cu, "Fairy", mat=wire_mat)
        for i, c in enumerate(pts[1:-1:2]):
            B.obj(f"Fairy.Bulb{name}{i:02d}", bulb, "Fairy", loc=(c[0], c[1] - 0.004, c[2] - 0.008))
    # the warm glow the low string throws on the characters' backs
    ld = bpy.data.lights.new("Fairy.Glow", "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = KB_X1 - KB_X0, 0.05
    ld.energy = 4.0
    ld.color = (1.0, 0.6, 0.3)
    ld.use_shadow = False
    B.obj("Fairy.Glow", ld, "Fairy", loc=((KB_X0 + KB_X1) / 2, -0.05, KEY_Z + 0.16),
          rot=(math.radians(-80), 0, 0))                  # faces the keys and the camera


def build_lights(B):
    def area(name, loc, target, energy, color, size, shape="DISK", shadow=True):
        ld = bpy.data.lights.new(name, "AREA")
        ld.use_shadow = shadow
        ld.shape = shape
        ld.size = size
        ld.energy = energy
        ld.color = color
        o = B.obj(name, ld, "Lights", loc=loc)
        o.rotation_euler = look_rotation(loc, target)
        return o

    def spot(name, loc, target, energy, color, angle, blend=0.6, radius=0.1, shadow=True):
        ld = bpy.data.lights.new(name, "SPOT")
        ld.use_shadow = shadow
        ld.energy = energy
        ld.color = color
        ld.spot_size = math.radians(angle)
        ld.spot_blend = blend
        ld.shadow_soft_size = radius
        o = B.obj(name, ld, "Lights", loc=loc)
        o.rotation_euler = look_rotation(loc, target)
        return o

    # a warm lamp from the front left (the only big shadow caster), a cool rim
    # from above/behind so the black cat reads, a faint fill and moonlight
    area("Light.Key", (-1.0, -1.35, 1.95), (0.0, -0.15, KEY_Z), 32.0, (1.0, 0.64, 0.38), 0.9)
    area("Light.Fill", (1.6, -1.8, 1.4), (0.1, -0.1, KEY_Z), 3.0, (0.75, 0.8, 1.0), 1.8, shadow=False)
    spot("Light.Rim", (0.25, 0.4, KEY_Z + 1.2), (0.05, -0.17, KEY_Z + 0.05), 55.0,
         (0.6, 0.72, 1.0), 75, 0.8, 0.25)
    spot("Light.Moon", (2.1, 1.2, 2.2), (0.3, -0.1, KEY_Z), 14.0, (0.5, 0.62, 1.0), 45, 0.9, 0.3,
         shadow=False)


def build_world(scene):
    w = bpy.data.worlds.new(f"{SCENE_NAME}.World")
    if w.node_tree is None:
        w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.004, 0.003, 0.003, 1)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = w


# ============================================================================
# 5. CHARACTERS
# ============================================================================
CORAL = srgb("#D97757")            # Claude's colour


class Claude:
    """A coral box with four little legs, side nubs and swappable eyes.

    Hierarchy:  Root (x, hop z, yaw)  ->  Squash (scale, lean; pivot at the feet)
                ->  body, legs, arms, Face (look offset) -> eye sets (one per expression)
    """
    W, H, D = 0.16, 0.12, 0.11
    LEG = 0.028
    EXPRESSIONS = ("open", "happy", "content", "love", "wide")

    def __init__(self, B):
        body_mat = principled("Claude.Body", CORAL, 0.5, Subsurface_Weight=0.12,
                              Subsurface_Radius=(0.05, 0.02, 0.01), Sheen_Weight=0.15)
        leg_mat = principled("Claude.Legs", tuple(c * 0.8 for c in CORAL), 0.55)
        eye_mat = principled("Claude.Eyes", srgb("#1d1714"), 0.18, Coat_Weight=0.8)
        heart_mat = principled("Claude.Hearts", srgb("#e0243c"), 0.25, Emission_Color=(1, 0.1, 0.2, 1),
                               Emission_Strength=0.6, Coat_Weight=0.6)
        blush_mat = principled("Claude.Blush", srgb("#f28c8c"), 0.6)
        shine = emission("Claude.Shine", (1, 1, 1), 3.0)
        W, H, D, L = self.W, self.H, self.D, self.LEG
        self.root = B.empty("Claude.Root", "Claude", size=0.08, shape="ARROWS")
        self.squash = B.empty("Claude.Squash", "Claude", parent=self.root, size=0.05)
        body = B.mesh("Claude.Body", bm_box(W, D, H, (0, 0, L + H / 2)), "Claude", body_mat,
                      parent=self.squash)
        add_bevel(body, 0.022, 5)
        for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
            leg = B.mesh(f"Claude.Leg{i}", bm_box(0.026, 0.026, L + 0.01, (0, 0, (L + 0.01) / 2)), "Claude",
                         leg_mat, parent=self.squash, loc=(sx * 0.052, sy * 0.03, 0))
            add_bevel(leg, 0.008, 3)
        self.arms = []
        for side in (-1, 1):
            pivot = B.empty(f"Claude.Arm{'L' if side < 0 else 'R'}", "Claude", parent=self.squash,
                            loc=(side * W / 2, 0, L + H * 0.48), size=0.02)
            arm = B.mesh(f"Claude.ArmMesh{'L' if side < 0 else 'R'}",
                         bm_box(0.036, 0.034, 0.03, (side * 0.014, 0, 0)), "Claude", body_mat, parent=pivot)
            add_bevel(arm, 0.011, 3)
            self.arms.append(pivot)
        # face: expressions are sets of decals on the front side (-Y)
        self.face = B.empty("Claude.Face", "Claude", parent=self.squash,
                            loc=(0, -D / 2 - 0.0005, L + H * 0.58), size=0.02)
        self.expr = {}
        ex = 0.033                                   # eye spacing (half)
        shapes = {
            "open": (outline_capsule(0.017, 0.034), eye_mat, 0.0),
            "happy": (outline_arc(0.012, 0.0065, math.radians(20), math.radians(160), dz=-0.004), eye_mat, 0.0),
            "content": (outline_arc(0.012, 0.0065, math.radians(200), math.radians(340), dz=0.006), eye_mat, 0.0),
            "love": (outline_heart(0.036), heart_mat, 0.0),
            "wide": (outline_circle(0.0135), eye_mat, 0.0),
        }
        for name, (outline, mat, dz) in shapes.items():
            grp = B.empty(f"Claude.Eyes.{name}", "Claude", parent=self.face, size=0.015)
            for side in (-1, 1):
                eye = B.mesh(f"Claude.Eye.{name}.{side}", bm_outline(outline, 0.003), "Claude", mat,
                             parent=grp, loc=(side * ex, 0, dz), smooth_shade=False)
                if name in ("open", "wide"):
                    B.mesh(f"Claude.Shine.{name}.{side}", bm_sphere(0.0032, (1, 0.4, 1), u=8, v=6), "Claude",
                           shine, parent=eye, loc=(-0.003, -0.0035, 0.006))
            self.expr[name] = grp
        self.blush = B.empty("Claude.Blush", "Claude", parent=self.face, size=0.015)
        for side in (-1, 1):
            B.mesh(f"Claude.BlushMesh{side}", bm_outline(outline_circle(0.011), 0.002), "Claude", blush_mat,
                   parent=self.blush, loc=(side * 0.056, 0.0005, -0.026), scale=(1.3, 1, 0.7),
                   smooth_shade=False)


class Cat:
    """A stylised black cat. Legs are stretchy (Stretch To constraints) so the
    paws can be placed exactly on keys: the animation only moves the body and
    four paw targets.

    Local frame: +X forward, +Y left, +Z up.
    Root (x, y, key top, yaw) -> Torso (height, pitch, roll) -> Head, Tail, legs
    """
    SHOULDER = (0.068, 0.028, -0.02)
    HIP = (-0.072, 0.032, -0.03)
    HEAD_PIVOT = (0.092, 0.0, 0.04)
    TAIL_BASE = (-0.104, 0.0, 0.024)
    TAIL_SEGS = 9
    TAIL_LEN = 0.03
    PAW_R = 0.018
    HEAD_R = 0.058

    def __init__(self, B):
        fur = principled("Cat.Fur", srgb("#0c0b0d"), 0.62, Sheen_Weight=0.9, Sheen_Roughness=0.35,
                         Sheen_Tint=(0.55, 0.62, 0.8, 1), Coat_Weight=0.0)
        pink = principled("Cat.Pink", srgb("#b86d78"), 0.55)
        iris = principled("Cat.Iris", srgb("#c9c23a"), 0.15, Emission_Color=(0.75, 0.8, 0.15, 1),
                          Emission_Strength=0.35, Coat_Weight=1.0)
        pupil = principled("Cat.Pupil", (0.0, 0.0, 0.0), 0.1, Coat_Weight=1.0)
        whisker = principled("Cat.Whisker", (0.55, 0.55, 0.55), 0.4)
        self.root = B.empty("Cat.Root", "Cat", size=0.08, shape="ARROWS")
        self.torso = B.empty("Cat.Torso", "Cat", parent=self.root, size=0.05)

        def torso_profile(u):
            # u: 0 = rump, 1 = chest
            g = [(0.0, 0.0), (0.08, 0.74), (0.22, 0.96), (0.45, 0.86), (0.7, 1.0), (0.88, 0.93), (1.0, 0.0)]
            for (u0, g0), (u1, g1) in zip(g[:-1], g[1:]):
                if u0 <= u <= u1:
                    k = g0 + (g1 - g0) * smooth((u - u0) / (u1 - u0))
                    break
            k = k ** 0.75
            return 0.05 * k, 0.056 * k, 0.004 * math.sin(math.pi * u)
        B.mesh("Cat.Body", bm_lathe(0.22, torso_profile, "X", 28, 22, start=-0.11), "Cat", fur,
               parent=self.torso)
        for side in (-1, 1):
            B.mesh(f"Cat.Haunch{side}", bm_sphere(0.033, (1.25, 0.8, 1.0)), "Cat", fur, parent=self.torso,
                   loc=(-0.058, side * 0.021, -0.014))
        B.mesh("Cat.Chest", bm_sphere(0.04, (1.15, 1.0, 1.05)), "Cat", fur, parent=self.torso,
               loc=(0.07, 0, 0.012))
        # head (big, like a kitten)
        hr = self.HEAD_R
        k = hr / 0.05                                  # proportions were designed at r = 0.05
        self.head = B.empty("Cat.HeadPivot", "Cat", parent=self.torso, loc=self.HEAD_PIVOT, size=0.03)
        hc = Vector((0.046, 0, 0.036))
        B.mesh("Cat.Head", bm_sphere(hr, (1.0, 1.1, 0.92)), "Cat", fur, parent=self.head, loc=hc)
        for side in (-1, 1):
            B.mesh(f"Cat.Muzzle{side}", bm_sphere(0.017 * k, (1.0, 1.0, 0.85)), "Cat", fur, parent=self.head,
                   loc=hc + Vector((0.038, side * 0.012, -0.018)) * k)
        B.mesh("Cat.Chin", bm_sphere(0.013 * k), "Cat", fur, parent=self.head,
               loc=hc + Vector((0.033, 0, -0.03)) * k)
        B.mesh("Cat.Nose", bm_sphere(0.0042 * k, (0.8, 1.3, 0.8)), "Cat", pink, parent=self.head,
               loc=hc + Vector((0.053, 0, -0.006)) * k)
        self.ears = []
        for side in (-1, 1):
            ep = B.empty(f"Cat.Ear{side}", "Cat", parent=self.head,
                         loc=hc + Vector((-0.004, side * 0.028, 0.034)) * k,
                         rot=(side * math.radians(-18), math.radians(-8), 0), size=0.015)
            B.mesh(f"Cat.EarMesh{side}",
                   bm_lathe(0.042 * k, lambda u: (0.023 * k * (1 - u) ** 0.9 + 0.001, 0.012 * k * (1 - u) + 0.001,
                                                  0.0), "Z", 10, 14), "Cat", fur, parent=ep)
            B.mesh(f"Cat.EarInner{side}",
                   bm_lathe(0.03 * k, lambda u: (0.014 * k * (1 - u) + 0.001, 0.004 * (1 - u) + 0.0005, 0.0),
                            "Z", 8, 12), "Cat", pink, parent=ep, loc=(0.0075 * k, 0, 0.003))
            self.ears.append(ep)
        self.eyes, self.pupils = [], []
        for side in (-1, 1):
            eg = B.empty(f"Cat.Eye{side}", "Cat", parent=self.head,
                         loc=hc + Vector((0.038, side * 0.022, 0.008)) * k,
                         rot=(0, 0, side * math.radians(24)), size=0.01)
            B.mesh(f"Cat.Iris{side}", bm_sphere(0.0145 * k, (0.55, 1.0, 1.05)), "Cat", iris, parent=eg)
            pu = B.mesh(f"Cat.Pupil{side}", bm_sphere(0.0145 * k, (0.3, 0.3, 0.96)), "Cat", pupil, parent=eg,
                        loc=(0.0036 * k, 0, 0))
            self.eyes.append(eg)
            self.pupils.append(pu)
            for j, ang in enumerate((-14, 0, 14)):
                # whisker along +Y, fanned up/down, pointing out of the side of the muzzle
                B.mesh(f"Cat.Whisker{side}{j}", bm_lathe(0.06, lambda u: (0.0006, 0.0006, 0.0), "Y", 3, 5),
                       "Cat", whisker, parent=self.head, loc=hc + Vector((0.042, side * 0.022, -0.016)) * k,
                       rot=(math.radians(ang) * side, 0, math.radians(-15) if side > 0 else math.pi + math.radians(15)))
        # tail: chain of overlapping, nearly cylindrical segments along local +X
        self.tail = []
        parent = B.empty("Cat.TailBase", "Cat", parent=self.torso, loc=self.TAIL_BASE, size=0.02)
        self.tail_base = parent
        for i in range(self.TAIL_SEGS):
            r0 = 0.0128 - 0.0045 * i / self.TAIL_SEGS
            seg = B.empty(f"Cat.Tail{i}", "Cat", parent=parent, loc=(0 if i == 0 else self.TAIL_LEN, 0, 0),
                          size=0.012)
            B.mesh(f"Cat.TailMesh{i}", bm_lathe(self.TAIL_LEN + 2 * r0, pill(r0, 6.0), "X", 10, 12,
                                                start=-r0), "Cat", fur, parent=seg)
            self.tail.append(seg)
            parent = seg
        B.mesh("Cat.TailTip", bm_sphere(0.0086, u=12, v=8), "Cat", fur, parent=parent, loc=(self.TAIL_LEN, 0, 0))
        # legs + paw targets
        self.paws = {}
        for name, (lx, ly, lz), r in (("FL", self.SHOULDER, 0.0165), ("FR", self.SHOULDER, 0.0165),
                                      ("HL", self.HIP, 0.019), ("HR", self.HIP, 0.019)):
            side = 1 if name[1] == "L" else -1
            target = B.empty(f"Cat.Paw{name}", "Cat", size=0.02, shape="SPHERE")
            B.mesh(f"Cat.PawMesh{name}", bm_sphere(self.PAW_R, (1.25, 1.0, 0.62)), "Cat", fur, parent=target,
                   loc=(0.004, 0, self.PAW_R * 0.62))
            aim = B.empty(f"Cat.PawAim{name}", "Cat", parent=target, loc=(0, 0, self.PAW_R * 0.7), size=0.01)
            rest = 0.1
            leg = B.mesh(f"Cat.Leg{name}", bm_lathe(rest, pill(r, 3.0), "Y", 10, 12), "Cat", fur,
                         parent=self.torso, loc=(lx, side * ly, lz))
            c = leg.constraints.new("STRETCH_TO")
            c.target = aim
            c.rest_length = rest
            c.volume = "NO_VOLUME"
            try:
                c.keep_axis = "SWING_Y"
            except TypeError:
                c.keep_axis = "PLANE_X"
            self.paws[name] = target


# ============================================================================
# 6. KEY ANIMATION  (exactly the MIDI)
# ============================================================================
MIN_HOLD = 2          # frames a key stays down at least
RELEASE = 2           # frames to come back up


class KeyAnim:
    """Press curves per key, in frames. Also used to put paws on key surfaces."""

    def __init__(self, music):
        self.pts = {}
        for p, notes in music.by_pitch.items():
            self.pts[p] = self._curve(sorted(notes, key=lambda n: n.on))
        self.onsets = {p: [int(round(frame_of(n.on))) for n in ns] for p, ns in music.by_pitch.items()}

    @staticmethod
    def _curve(notes):
        pts = {}
        last_f, last_v = -10**9, 0.0
        for i, n in enumerate(notes):
            f_on = int(round(frame_of(n.on)))
            f_off = max(f_on + MIN_HOLD, int(round(frame_of(n.off))))
            f_next = int(round(frame_of(notes[i + 1].on))) if i + 1 < len(notes) else 10**9
            if last_f < f_on - 1:
                pts[f_on - 1] = last_v
            pts[f_on] = 1.0
            rel0 = max(min(f_off, f_next - 2), f_on)
            rel1 = min(rel0 + RELEASE, f_next - 1)
            last_f, last_v = f_on, 1.0
            if rel0 > f_on:
                pts[rel0] = 1.0
                last_f = rel0
            if rel1 > rel0:
                v = 0.0 if rel1 - rel0 >= RELEASE else 0.45
                pts[rel1] = v
                last_f, last_v = rel1, v
        items = sorted(pts.items())
        return [f for f, _ in items], [v for _, v in items]

    def press(self, p, f):
        """press amount 0..1 of key p at (fractional) frame f"""
        if p not in self.pts:
            return 0.0
        fs, vs = self.pts[p]
        if f <= fs[0]:
            return vs[0]
        if f >= fs[-1]:
            return vs[-1]
        k = bisect.bisect_right(fs, f) - 1
        u = (f - fs[k]) / (fs[k + 1] - fs[k])
        return vs[k] + u * (vs[k + 1] - vs[k])

    def surface_z(self, p, y, f):
        """height of key p's top surface at depth y (y < 0) at frame f"""
        ang = key_theta(p) * self.press(p, f)
        return key_top(p) - (PIVOT_Y - y) * math.sin(ang)

    def apply(self, keys, n_frames):
        for p, o in keys.items():
            if p not in self.pts:
                continue
            fs, vs = self.pts[p]
            th = key_theta(p)
            bake(o, "rotation_euler", 0, fs, [v * th for v in vs], tol=0)
            # glow: flash on the strike, settle while held, fade after release
            glow_f, glow_v = [], []
            onsets = set(self.onsets[p])
            g, since = 0.0, 99
            f0, f1 = max(FRAME_START, fs[0] - 1), min(FRAME_START + n_frames, fs[-1] + 12)
            for f in range(f0, f1 + 1):
                v = self.press(p, f)
                since = 0 if f in onsets else since + 1
                target = v * (0.45 + 0.55 * math.exp(-since / 5.0))
                g = target if target > g else g * 0.72 + target * 0.28
                glow_f.append(f)
                glow_v.append(g)
            bake(o, '["glow"]', 0, glow_f, glow_v, tol=0.01)


# ============================================================================
# 7. CHOREOGRAPHY
# ============================================================================
# Both characters stand on the front part of the keys (y < -0.1425 is white
# keys only), so feet never cut into black keys. A paw reaches back only to
# strike a black key.
Y_CAT = -0.19         # cat's midline (keys span y = -0.225 .. 0)
Y_CLAUDE = -0.19
CAT_SHOULDER_MIN, CAT_SHOULDER_MAX = 0.02, 0.62
CLAUDE_X_MIN = -0.56

CAT_PLAN = (
    # section, mode, facing (+1 right, -1 left, 0 = follow the melody), look,
    #   shoulder x (None = where the section's notes are)
    ("intro", "walk", -1, "keys", None),
    ("run", "walk", +1, "keys", None),
    ("hold", "sit", -1, "claude", None),
    ("stops", "sit_tap", -1, "keys", None),
    ("groove", "walk", 0, "keys", None),
    ("light", "sit_tap", -1, "keys", None),
    ("light_bass", "walk", 0, "keys", None),
    ("ritard", "stretch", -1, "up", 0.24),      # comes to the middle, stretches ...
    ("break", "sit", -1, "claude", 0.24),       # ... and sits face to face with Claude
    ("stops2", "sit_tap", -1, "camera", None),
    ("chorus", "walk", 0, "keys", None),
    ("breakdown", "sit", -1, "claude", None),
    ("chorus2", "walk", 0, "keys", None),
    ("outro", "sit_tap", -1, "claude", None),
)

#            torso height, pitch, front paw x, hind paw x, reach fwd, reach back
CAT_POSES = {
    "walk":    (0.104, 0.00, 0.076, -0.068, 0.080, 0.045),
    "sit":     (0.096, -0.42, 0.062, -0.026, 0.085, 0.050),
    "sit_tap": (0.096, -0.34, 0.070, -0.026, 0.095, 0.055),
    "stretch": (0.078, 0.26, 0.165, -0.080, 0.0, 0.0),
}

CAMERA_GUESS = Vector((0.05, -1.45, KEY_Z + 0.35))


class Timeline:
    """Per-frame sampling grid shared by all animation."""

    def __init__(self, music):
        self.n = int(math.ceil(music.length * FPS))
        self.frames = list(range(FRAME_START, FRAME_START + self.n))
        self.times = [time_of(f) for f in self.frames]

    def idx(self, t):
        return clamp(int(round(frame_of(t))) - FRAME_START, 0, self.n - 1)


class CatAnim:
    def __init__(self, music, keys, tl, rng):
        self.m, self.keys, self.tl, self.rng = music, keys, tl, rng
        self.segments = []
        for sec, mode, facing, look, fixed_x in CAT_PLAN:
            t0, t1 = music.sections[sec]
            self.segments.append((t0, t1, sec, mode, facing, look, fixed_x))
        self.hops = [(music.sections["stops"][0] + 0.03, 0.42, 0.045),       # startled by the 1st chord
                     (music.sections["breakdown"][0] + 0.05, 0.40, 0.05),
                     (music.sections["chorus2"][0] - 0.42, 0.40, 0.035)]
        self._plan_body()

    # -- body -----------------------------------------------------------------
    def seg_at(self, t):
        for s in self.segments:
            if s[0] <= t < s[1]:
                return s
        return self.segments[0] if t < self.segments[0][0] else self.segments[-1]

    def melody_x(self, t):
        """x of the top right-hand note last struck at t"""
        ch = self.m.rh_chords
        if not hasattr(self, "_rh_t"):
            self._rh_t = [c[0] for c in ch]
        k = bisect.bisect_right(self._rh_t, t) - 1
        k = max(k, 0)
        return key_x(max(n.pitch for n in ch[k][1]))

    def _plan_body(self):
        tl, m = self.tl, self.m
        n = tl.n
        # shoulder target
        s = [0.0] * n
        for t0, t1, sec, mode, facing, look, fixed_x in self.segments:
            i0, i1 = tl.idx(t0), tl.idx(t1)
            if fixed_x is not None:
                for i in range(i0, i1 + 1 if i1 == n - 1 else i1):
                    s[i] = fixed_x
            elif mode == "walk":
                for i in range(i0, i1 + 1 if i1 == n - 1 else i1):
                    s[i] = self.melody_x(tl.times[i] + 0.25)
            else:
                xs = sorted(key_x(max(nn.pitch for nn in c[1])) for c in m.rh_chords if t0 <= c[0] < t1)
                val = xs[len(xs) // 2] if xs else None
                for i in range(i0, i1 + 1 if i1 == n - 1 else i1):
                    s[i] = val if val is not None else (s[i0 - 1] if i0 > 0 else 0.35)
        # the outro: sit close to where Claude can join
        s = [clamp(v, CAT_SHOULDER_MIN, CAT_SHOULDER_MAX) for v in s]
        s = gaussian_smooth(s, 0.45 * FPS)
        s = rate_limit(s, 0.24 / FPS)
        self.shoulder = s
        # facing / turns
        turns = []
        facing = self.segments[0][4] or -1
        last_turn = -99.0
        for t0, t1, sec, mode, f_plan, look, _x in self.segments:
            if f_plan and f_plan != facing:
                turns.append((t0 - 0.42, t0 - 0.02, f_plan))
                facing, last_turn = f_plan, t0
            if not f_plan:
                # follow the melody: turn round only for a clear, lasting move
                # backwards, at most every 3 s and not right before the next section
                i = tl.idx(t0 + 0.5)
                while i < tl.idx(t1 - 1.3):
                    t = tl.times[i]
                    j = min(i + int(1.0 * FPS), n - 1)
                    v = s[j] - s[i]
                    if facing * v < -0.07 and t - last_turn > 3.0:
                        turns.append((t, t + 0.4, -facing))
                        facing, last_turn = -facing, t
                    i += 1
        self.turns = turns
        yaw = []
        cur = 0.0 if (self.segments[0][4] or -1) > 0 else -math.pi
        k = 0
        base = cur
        for t in tl.times:
            while k < len(turns) and t >= turns[k][1]:
                base += math.pi if turns[k][2] > 0 else -math.pi
                k += 1
            if k < len(turns) and turns[k][0] <= t < turns[k][1]:
                u = smoother((t - turns[k][0]) / (turns[k][1] - turns[k][0]))
                yaw.append(base + (math.pi if turns[k][2] > 0 else -math.pi) * u)
            else:
                yaw.append(base)
        self.yaw = yaw
        # pose parameters, blended between modes
        mode_at = [self.seg_at(t)[3] for t in tl.times]
        # the stretch: in for the first half of the ritardando, back out before the break
        r0, r1 = m.sections["ritard"]
        for i, t in enumerate(tl.times):
            if mode_at[i] == "stretch" and t > r0 + 0.62 * (r1 - r0):
                mode_at[i] = "sit"
        self.mode_at = mode_at
        params = list(zip(*[CAT_POSES[md] for md in mode_at]))
        sig = 0.16 * FPS
        self.h, self.pitch, self.front_x, self.hind_x, self.reach_f, self.reach_b = \
            [gaussian_smooth(list(p), sig) for p in params]
        # hops and turns lift the body
        self.lift = [0.0] * n
        self.windows = []                                      # paws follow the body here
        for t0, dur, hgt in self.hops:
            self.windows.append((t0, t0 + dur, "rigid"))
            for i in range(tl.idx(t0), tl.idx(t0 + dur) + 1):
                u = (tl.times[i] - t0) / dur
                if 0 <= u <= 1:
                    self.lift[i] = max(self.lift[i], hgt * math.sin(math.pi * u))
        for t0, t1, _ in turns:
            self.windows.append((t0, t1, "rigid"))
            for i in range(tl.idx(t0), tl.idx(t1) + 1):
                u = (tl.times[i] - t0) / (t1 - t0)
                if 0 <= u <= 1:
                    self.lift[i] = max(self.lift[i], 0.03 * math.sin(math.pi * u))
        self.windows.append((r0 - 0.1, r1 + 0.3, "slide"))
        self.windows.sort()
        # body x from the shoulder target and the facing
        self.x = [sh - Cat.SHOULDER[0] * math.cos(y) for sh, y in zip(s, yaw)]
        # little extras: roll with the beat while walking, breathing
        self.roll = [0.0] * n

    def body_matrix(self, i, extra_h=0.0):
        m = (Matrix.Translation((self.x[i], Y_CAT, KEY_Z + self.h[i] + self.lift[i] + extra_h))
             @ Matrix.Rotation(self.yaw[i], 4, "Z")
             @ Matrix.Rotation(self.pitch[i], 4, "Y")
             @ Matrix.Rotation(self.roll[i], 4, "X"))
        return m

    def ideal_paw(self, name, i):
        """where a paw 'wants' to be at frame index i (on the key surface)"""
        side = 1 if name[1] == "L" else -1
        lx = self.front_x[i] if name[0] == "F" else self.hind_x[i]
        ly = (0.028 if name[0] == "F" else 0.03) * side
        yaw = self.yaw[i]
        wx = self.x[i] + math.cos(yaw) * lx - math.sin(yaw) * ly
        wy = Y_CAT + math.sin(yaw) * lx + math.cos(yaw) * ly
        return wx, wy

    # -- paws -----------------------------------------------------------------
    def in_window(self, t):
        for w in self.windows:
            if w[0] <= t <= w[1]:
                return w
        return None

    def plan_paws(self):
        tl, m = self.tl, self.m
        events = {p: [] for p in ("FL", "FR", "HL", "HR")}   # (t_lift, t_land, x, y, key, pressed)
        last_land = {p: -9.0 for p in events}
        last_used = {"FL": -9.0, "FR": -9.0}
        self.contacts = []                                     # (frame, pitch, paw) for checks
        for t_on, chord in m.rh_chords:
            t_on = time_of(round(frame_of(t_on)))           # land on the key's own (rounded) frame
            if any(self.in_window(t_on + d) for d in (-0.21, -0.14, -0.07, 0.0, 0.07, 0.12)):
                continue
            i = tl.idx(t_on)
            facing = math.cos(self.yaw[i])
            options = []
            for paw in ("FL", "FR"):
                ix, iy = self.ideal_paw(paw, i)
                t_free = last_land[paw] + 0.05
                avail = t_on - t_free
                if avail < 0.06:
                    continue
                for nn in chord:
                    dx = (key_x(nn.pitch) - ix) * (1 if facing > 0 else -1)
                    if -self.reach_b[i] <= dx <= self.reach_f[i]:
                        top = nn.pitch == max(c.pitch for c in chord)
                        score = abs(dx - 0.015) - (0.02 if top else 0) + (0.03 if last_used[paw] > last_used[
                            "FR" if paw == "FL" else "FL"] else 0)
                        options.append((score, paw, nn, iy, avail))
            if not options:
                continue
            _, paw, nn, iy, avail = min(options, key=lambda o: o[0])
            swing = clamp(avail, 0.06, 0.19)
            y = clamp(iy, -0.125, -0.03) if is_black(nn.pitch) else min(iy, -0.16)
            events[paw].append((t_on - swing, t_on, key_x(nn.pitch), y, nn.pitch, True))
            last_land[paw] = t_on
            last_used[paw] = t_on
        # maintenance steps: keep all four paws under the body
        for paw in ("FL", "FR"):
            events[paw] = self._maintain(paw, events[paw])
        events["HL"], events["HR"] = self._maintain_hind()
        self.events = events
        self.contacts = sorted((int(round(frame_of(e[1]))), e[4], paw)
                               for paw, evs in events.items() for e in evs if e[5])

    def _maintain(self, paw, evs):
        """Front paw: add plain steps (no key pressed) whenever the body has
        moved away from a planted paw and no note step is coming soon."""
        tl = self.tl
        out = []
        planted = self.ideal_paw(paw, 0)
        k, i = 0, 0
        busy_until = -9.0
        while i < tl.n:
            t = tl.times[i]
            while k < len(evs) and evs[k][1] <= t:
                out.append(evs[k])
                planted = (evs[k][2], evs[k][3])
                busy_until = evs[k][1]
                k += 1
            w = self.in_window(t)
            if w:
                j = tl.idx(w[1])
                planted = self.ideal_paw(paw, j)             # re-planted after a hop/turn/stretch
                out.append((w[1], w[1], planted[0], planted[1], None, False))
                while k < len(evs) and evs[k][0] < w[1] + 0.05:
                    k += 1                                    # drop note steps inside the window
                i = j + 1
                continue
            if k < len(evs) and evs[k][0] <= t:
                i += 1
                continue                                      # swinging to a note
            ahead = min(i + int(0.3 * FPS), tl.n - 1)
            ix, iy = self.ideal_paw(paw, ahead)
            d = math.hypot(planted[0] - ix, planted[1] - iy)
            next_lift = evs[k][0] if k < len(evs) else 9e9
            if d > 0.055 and t > busy_until + 0.05 and next_lift > t + 0.3:
                out.append((t, t + 0.17, ix, iy, None, False))
                planted, busy_until = (ix, iy), t + 0.17
                i = tl.idx(t + 0.17) + 1
                continue
            i += 1
        out.extend(evs[k:])
        out.sort(key=lambda e: e[1])
        return out

    def _maintain_hind(self):
        """Hind paws step alternately to stay under the hips."""
        tl = self.tl
        names = ("HL", "HR")
        out = {p: [] for p in names}
        planted = {p: self.ideal_paw(p, 0) for p in names}
        swing_until = {p: -9.0 for p in names}
        i = 0
        while i < tl.n:
            t = tl.times[i]
            w = self.in_window(t)
            if w:
                j = tl.idx(w[1])
                for p in names:
                    planted[p] = self.ideal_paw(p, j)
                    out[p].append((w[1], w[1], planted[p][0], planted[p][1], None, False))
                    swing_until[p] = w[1]
                i = j + 1
                continue
            ahead = min(i + int(0.3 * FPS), tl.n - 1)
            need = []
            for p in names:
                ix, iy = self.ideal_paw(p, ahead)
                d = math.hypot(planted[p][0] - ix, planted[p][1] - iy)
                if d > 0.05 and t >= swing_until[p] + 0.03:
                    need.append((d, p, ix, iy))
            if need:
                d, p, ix, iy = max(need)
                other = names[1 - names.index(p)]
                if t >= swing_until[other] or d > 0.11:
                    out[p].append((t, t + 0.17, ix, iy, None, False))
                    planted[p] = (ix, iy)
                    swing_until[p] = t + 0.17
            i += 1
        return out["HL"], out["HR"]

    def paw_positions(self, paw):
        """world position per frame for one paw"""
        tl = self.tl
        evs = self.events[paw]
        lands = [e[1] for e in evs]
        out = []
        prev_pos = None
        for i, t in enumerate(tl.times):
            f = tl.frames[i]
            w = self.in_window(t)
            ix, iy = self.ideal_paw(paw, i)
            if w is not None and w[2] == "rigid":
                u = (t - w[0]) / (w[1] - w[0])
                z = (self.keys.surface_z(key_at(ix, iy), iy, f) + self.lift[i]
                     + 0.022 * math.sin(math.pi * clamp(u, 0, 1)))
                pos = Vector((ix, iy, z))
                if prev_pos is not None and u < 0.2:
                    pos = prev_pos.lerp(pos, smooth(u / 0.2))
                out.append(pos)
                prev_pos = pos
                continue
            if w is not None and w[2] == "slide":
                pos = Vector((ix, iy, self.keys.surface_z(key_at(ix, iy), iy, f)))
                if prev_pos is not None and t - w[0] < 0.25:
                    pos = prev_pos.lerp(pos, smooth((t - w[0]) / 0.25))
                out.append(pos)
                prev_pos = pos
                continue
            k = bisect.bisect_right(lands, t) - 1            # last event that has landed
            nxt = k + 1
            if nxt < len(evs) and evs[nxt][0] <= t < evs[nxt][1]:
                e = evs[nxt]
                u = (t - e[0]) / (e[1] - e[0])
                if k >= 0:
                    sx, sy = evs[k][2], evs[k][3]
                else:
                    sx, sy = self.ideal_paw(paw, 0)
                start = Vector((sx, sy, self._surface(evs[k] if k >= 0 else None, sx, sy, self.tl.frames[
                    self.tl.idx(e[0])])))
                end_key = e[4] if e[4] is not None else key_at(e[2], e[3])
                end = Vector((e[2], e[3], self.keys.surface_z(end_key, e[3], frame_of(e[1]))))
                h = 0.032 if e[5] else 0.022
                if e[1] - e[0] > 0.5:
                    h = 0.0
                pos = start.lerp(end, smoother(u))
                pos.z += h * math.sin(math.pi * u) ** 0.8
            else:
                if k >= 0:
                    e = evs[k]
                    pos = Vector((e[2], e[3], self._surface(e, e[2], e[3], f)))
                else:
                    pos = Vector((ix, iy, self.keys.surface_z(key_at(ix, iy), iy, f)))
            out.append(pos)
            prev_pos = pos
        return out

    def _surface(self, ev, x, y, f):
        p = ev[4] if (ev is not None and ev[4] is not None) else key_at(x, y)
        return self.keys.surface_z(p, y, f)

    # -- head, tail, eyes -----------------------------------------------------
    def look_targets(self, claude_head, cam_pos):
        tl = self.tl
        tg = []
        for i, t in enumerate(tl.times):
            seg = self.seg_at(t)
            look = seg[5]
            if look == "claude":
                tg.append(claude_head[i])
            elif look == "camera":
                tg.append(cam_pos[i] if cam_pos else CAMERA_GUESS)
            elif look == "up":
                x = self.x[i] + 0.3 * math.cos(self.yaw[i])
                tg.append(Vector((x, Y_CAT - 0.2, KEY_Z + 0.35)))
            else:
                # the key of the next (or last) front paw strike
                best = None
                for paw in ("FL", "FR"):
                    for e in self.events[paw]:
                        if e[5] and t - 0.3 <= e[1] <= t + 0.5:
                            if best is None or abs(e[1] - t - 0.1) < abs(best[1] - t - 0.1):
                                best = e
                if best:                                    # the key, seen from the front
                    tg.append(Vector((best[2], min(best[3], Y_CAT) - 0.08, KEY_Z)))
                else:
                    x = self.x[i] + 0.25 * math.cos(self.yaw[i])
                    tg.append(Vector((x, Y_CAT - 0.05, KEY_Z)))
        # smooth the target so the head doesn't snap
        xs = gaussian_smooth([v.x for v in tg], 0.12 * FPS)
        ys = gaussian_smooth([v.y for v in tg], 0.12 * FPS)
        zs = gaussian_smooth([v.z for v in tg], 0.12 * FPS)
        return [Vector(v) for v in zip(xs, ys, zs)]

    def bake(self, cat, claude_head, cam_pos):
        tl, m, rng = self.tl, self.m, self.rng
        fr = tl.frames
        n = tl.n
        # root and torso
        breathe = [0.004 * math.sin(2 * math.pi * t / 2.6) for t in tl.times]
        bake(cat.root, "location", 0, fr, self.x)
        bake(cat.root, "location", 1, fr, [Y_CAT] * n)
        bake(cat.root, "location", 2, fr, [KEY_Z] * n)
        bake(cat.root, "rotation_euler", 2, fr, self.yaw, 1e-3)
        # step bob: a small dip when a front paw lands
        bob = [0.0] * n
        for paw in ("FL", "FR"):
            for e in self.events[paw]:
                if e[5]:
                    j = tl.idx(e[1])
                    for d in range(-2, 5):
                        if 0 <= j + d < n:
                            bob[j + d] -= 0.004 * math.exp(-((d - 1) / 1.6) ** 2)
        bake(cat.torso, "location", 2, fr, [h + l + b + br for h, l, b, br in zip(self.h, self.lift, bob, breathe)])
        # sway with the beat while walking
        roll = [(0.05 if md == "walk" else 0.02) * math.sin(math.pi * m.beat_phase(t))
                for md, t in zip(self.mode_at, tl.times)]
        roll = gaussian_smooth(roll, 0.1 * FPS)
        self.roll = roll
        bake(cat.torso, "rotation_euler", 1, fr, self.pitch, 1e-3)
        bake(cat.torso, "rotation_euler", 0, fr, roll, 1e-3)
        # paws
        self.paw_world = {}
        for paw, obj in cat.paws.items():
            pos = self.paw_positions(paw)
            self.paw_world[paw] = pos
            bake_vec(obj, "location", fr, pos)
            bake(obj, "rotation_euler", 2, fr, self.yaw, 1e-3)
        # head look
        targets = self.look_targets(claude_head, cam_pos)
        hy, hp = [], []
        for i in range(n):
            mw = self.body_matrix(i)
            pivot = mw @ Vector(Cat.HEAD_PIVOT)
            d = mw.to_3x3().inverted() @ (targets[i] - pivot)
            yaw = clamp(math.atan2(d.y, d.x), -1.25, 1.25)
            pitch = clamp(-math.atan2(d.z, math.hypot(d.x, d.y)), -0.75, 0.9)
            hy.append(yaw)
            hp.append(pitch)
        hy = gaussian_smooth(hy, 0.08 * FPS)
        hp = gaussian_smooth(hp, 0.08 * FPS)
        # nod on the beat
        nod = [0.07 * math.exp(-((((m.beat_phase(t) + 0.1) % 1.0) - 0.1) / 0.1) ** 2) if md != "stretch" else 0.0
               for md, t in zip(self.mode_at, tl.times)]
        bake(cat.head, "rotation_euler", 2, fr, hy, 1e-3)
        bake(cat.head, "rotation_euler", 1, fr, [a + b for a, b in zip(hp, nod)], 1e-3)
        # tail
        energy = {"intro": 0.4, "run": 0.6, "hold": 0.3, "stops": 0.7, "groove": 0.9, "light": 0.6,
                  "light_bass": 0.8, "ritard": 0.3, "break": 0.2, "stops2": 0.8, "chorus": 1.0,
                  "breakdown": 0.5, "chorus2": 1.0, "outro": 0.4}
        up = {"walk": -0.55, "sit": 0.15, "sit_tap": -0.2, "stretch": -1.1}
        amp = gaussian_smooth([energy[self.seg_at(t)[2]] for t in tl.times], 0.4 * FPS)
        base_p = gaussian_smooth([up[md] for md in self.mode_at], 0.25 * FPS)
        curl = gaussian_smooth([1.0 if md in ("sit", "sit_tap") else 0.0 for md in self.mode_at], 0.3 * FPS)
        bake(cat.tail_base, "rotation_euler", 1, fr, [bp + p for bp, p in zip(base_p, self.pitch)], 1e-3)
        bake(cat.tail_base, "rotation_euler", 2, fr, [math.pi] * n, 0)
        for s, seg in enumerate(cat.tail):
            ys, ps = [], []
            for i, t in enumerate(tl.times):
                ph = math.pi * m.beat_phase(t) / 2.0          # one swish per two beats
                w = amp[i] * 0.22 * (0.4 + 0.6 * s / Cat.TAIL_SEGS) * math.sin(ph - 0.55 * s)
                ys.append(w + curl[i] * 0.33 * (s > 0))
                tip = (0.35 if s >= Cat.TAIL_SEGS - 3 else -0.05) * (1 - curl[i])
                ps.append(tip + curl[i] * (0.05 if s < 3 else -0.02))
            bake(seg, "rotation_euler", 2, fr, ys, 2e-3)
            bake(seg, "rotation_euler", 1, fr, ps, 2e-3)
        # ears: twitch now and then, flat when startled
        for e_i, ear in enumerate(cat.ears):
            vals = [0.0] * n
            t = 1.0 + rng.random() * 3
            while t < m.length:
                j = tl.idx(t)
                for d in range(0, 8):
                    if j + d < n:
                        vals[j + d] += 0.35 * math.sin(math.pi * d / 7)
                t += 2.5 + rng.random() * 5
            for t0, dur, _ in self.hops[:1]:
                j = tl.idx(t0)
                for d in range(0, int(0.8 * FPS)):
                    if j + d < n:
                        vals[j + d] -= 0.6 * math.sin(math.pi * d / (0.8 * FPS))
            side = -1 if e_i == 0 else 1
            bake(ear, "rotation_euler", 0, fr, [side * math.radians(-18) + side * -v for v in vals], 1e-3)
        # eyes: blinks, slow blinks (a cat's "I love you"), closed while stretching
        openness = [1.0] * n
        t = 2.0
        while t < m.length:
            j = tl.idx(t)
            for d in range(0, 5):
                if j + d < n:
                    openness[j + d] = min(openness[j + d], [0.6, 0.1, 0.1, 0.5, 0.9][d])
            t += 2.8 + rng.random() * 3.5
        slow = [m.sections["break"][0] + 0.45, m.sections["outro"][0] + 2.0, m.length - 3.0]
        for t0 in slow:
            for i, t in enumerate(tl.times):
                u = t - t0
                if 0 <= u <= 1.4:
                    v = 1 - smooth(u / 0.45) if u < 0.45 else (0.08 if u < 0.85 else smooth((u - 0.85) / 0.55))
                    openness[i] = min(openness[i], max(0.08, v))
        for i, md in enumerate(self.mode_at):
            if md == "stretch":
                openness[i] = min(openness[i], 0.12)
        openness = gaussian_smooth(openness, 0.04 * FPS)
        for eg in cat.eyes:
            bake(eg, "scale", 2, fr, openness, 1e-3)
        dil = gaussian_smooth([1.6 if self.seg_at(t)[2] in ("break", "outro", "hold") else 1.0
                               for t in tl.times], 0.3 * FPS)
        for pu in cat.pupils:
            bake(pu, "scale", 1, fr, dil, 1e-3)


class ClaudeAnim:
    """Hops on the bass notes, weight shifts on the beat, expressions."""

    def __init__(self, music, tl, cat_anim, rng):
        self.m, self.tl, self.cat, self.rng = music, tl, cat_anim, rng
        self.hops = []
        self._plan()

    def cat_limit(self, t0, t1):
        """rightmost x Claude may use between t0 and t1 (keeps clear of the cat)"""
        i0, i1 = self.tl.idx(t0), self.tl.idx(t1)
        return min(self.cat.x[i] for i in range(i0, i1 + 1)) - 0.34

    def bass_x(self, t, fallback):
        p = self.m.lowest_lh_near(t)
        return key_x(p) if p is not None else fallback

    def add(self, t_land, h, target=None, spin=0.0, air=None, antic=0.12, arms=0.6):
        """Record a hop that lands at t_land. target: x to move towards
        (None = hop on the spot). Travel, timing and spacing are resolved in
        _finalize(), once all hops are known."""
        bl = self.m.beat_len(t_land)
        self.hops.append(dict(t=t_land, air=air or min(0.62 * bl, 0.36), h=h, target=target, spin=spin,
                              antic=antic, arms=arms))

    def _plan(self):
        m = self.m
        S = m.sections

        def beats(sec, every=1, offset=0):
            b = m.beats_in(*S[sec])
            return b[offset::every]

        def downs(sec):
            t0, t1 = S[sec]
            return [d for d in m.downbeats if t0 - 1e-6 <= d < t1 - 1e-6]

        def is_down(t):
            return abs(((m.beat_phase(t) + 0.5) % 4) - 0.5) < 0.1

        # run: two hops onto the left-hand chords
        for t_on, chord in m.lh_chords:
            if S["run"][0] - 0.01 <= t_on < S["run"][1]:
                self.add(t_on, 0.035, self.bass_x(t_on, -0.38) + 0.05)
        # the big leap onto the first stop chord (after a long crouch), then one per chord
        for k, d in enumerate(downs("stops")):
            self.add(d, 0.13 if k == 0 else 0.11, self.bass_x(d, -0.35), spin=(2 * math.pi if k == 2 else 0),
                     air=0.46 if k == 0 else 0.42, antic=0.25, arms=1.0)
        for d in beats("groove"):
            down = is_down(d)
            self.add(d, 0.06 if down else 0.03, self.bass_x(d, None) if down else None, arms=0.8 if down else 0.4)
        for d in beats("light_bass", 2):
            self.add(d, 0.03, self.bass_x(d, None) if is_down(d) else None, arms=0.3)
        # ritardando: hop over to the cat for the love moment in the break
        for d in beats("ritard", 2)[:4]:
            self.add(d, 0.05, 9.0, air=0.38, arms=0.4)
        for k, d in enumerate(downs("stops2")):
            self.add(d, 0.14 if k == 0 else 0.12, self.bass_x(d, -0.35),
                     spin=(2 * math.pi if k in (1, 3) else 0), air=0.44, antic=0.25, arms=1.0)
        for sec, big in (("chorus", 0.10), ("chorus2", 0.12)):
            for d in beats(sec):
                down = is_down(d)
                bar = int(round(m.beat_phase(d) / 4))
                spin = 2 * math.pi if down and bar % 4 == 2 else 0.0
                h = big if spin else (0.065 if down else 0.035)
                self.add(d, h, self.bass_x(d, None) if down else None, spin=spin, arms=1.0 if spin else 0.6)
        # out of the breakdown: the biggest leap, with a spin
        self.add(S["chorus2"][0], 0.15, self.bass_x(S["chorus2"][0], -0.35), spin=2 * math.pi, air=0.5,
                 antic=0.3, arms=1.0)
        # outro: a spin into it, then little hops over to the cat, a last one on the last note
        o0 = S["outro"][0]
        self.hops = [h for h in self.hops if h["t"] < o0 - 0.01]
        self.add(o0, 0.16, None, spin=-2 * math.pi, air=0.5, antic=0.3, arms=1.0)
        for d in beats("outro", 2)[1:6]:
            self.add(d, 0.04, 9.0, air=0.34)
        self.add(m.notes[-1].on, 0.05, None, air=0.3, arms=0.8)
        self._finalize()

    def _finalize(self):
        # chronological, one hop per landing time (the bigger one wins)
        hops = sorted(self.hops, key=lambda h: (h["t"], -h["h"]))
        out = []
        for h in hops:
            if out and abs(h["t"] - out[-1]["t"]) < 0.05:
                continue
            out.append(h)
        x = self.start_x = -0.38
        for k, h in enumerate(out):
            # air time: take off only after the previous landing has settled a bit
            if k:
                h["air"] = min(h["air"], max(0.14, h["t"] - out[k - 1]["t"] - 0.08))
            h["antic"] = min(h["antic"], 0.3 * h["air"] + 0.05)
            # travel: towards the target, at most 15 cm per little hop / 35 cm per big one
            nxt = out[k + 1]["t"] - out[k + 1].get("air", 0.3) if k + 1 < len(out) else self.m.length
            limit = self.cat_limit(h["t"] - h["air"], max(nxt, h["t"] + 0.3))
            target = x if h["target"] is None else min(h["target"], limit)
            cap = 0.35 if h["h"] >= 0.1 else 0.15
            h["x0"] = x
            h["x1"] = clamp(x + clamp(target - x, -cap, cap), CLAUDE_X_MIN, limit)
            x = h["x1"]
        self.hops = out

    def sample(self):
        """per-frame x, z, yaw offset (spin), squash, lean, arms"""
        tl, m = self.tl, self.m
        hops = self.hops
        takeoffs = [h["t"] - h["air"] for h in hops]
        X, Z, SPIN, SQ, ARMS = [], [], [], [], []
        S = m.sections
        crouch = [(S["stops"][0] - 2.7, S["stops"][0] - 0.46, 0.22),
                  (S["stops2"][0] - 1.1, S["stops2"][0] - 0.44, 0.14),
                  (S["chorus2"][0] - 1.9, S["chorus2"][0] - 0.5, 0.15)]
        for t in tl.times:
            k = bisect.bisect_right(takeoffs, t) - 1       # last hop that has taken off
            x, z, spin, sq, arms = self.start_x, 0.0, 0.0, 0.0, 0.0
            if k >= 0:
                h = hops[k]
                if t < h["t"]:                               # in the air
                    u = (t - takeoffs[k]) / h["air"]
                    x = lerp(h["x0"], h["x1"], smooth(u))
                    z = h["h"] * 4 * u * (1 - u)
                    sq = 0.14 * abs(2 * u - 1) ** 2 * min(1.0, h["h"] / 0.05)
                    spin = h["spin"] * smoother(u)
                    arms = h["arms"] * math.sin(math.pi * u)
                else:                                        # landed
                    x = h["x1"]
                    v = (t - h["t"]) / 0.22
                    if v < 1:
                        sq = -min(0.28, 0.9 * h["h"] + 0.05) * math.exp(-3.2 * v) * math.cos(2.4 * math.pi * v)
                        arms = h["arms"] * 0.4 * math.exp(-4 * v)
            if k + 1 < len(hops):                            # anticipation before the next take-off
                h2 = hops[k + 1]
                w = (t - (takeoffs[k + 1] - h2["antic"])) / h2["antic"]
                if 0 <= w < 1:
                    sq -= min(0.22, 1.6 * h2["h"] + 0.04) * math.sin(0.5 * math.pi * w)
            for c0, c1, depth in crouch:
                if c0 <= t < c1:
                    sq -= depth * smooth((t - c0) / (c1 - c0))
            X.append(x)
            Z.append(z)
            SPIN.append(spin)
            SQ.append(sq)
            ARMS.append(arms)
        return X, Z, SPIN, SQ, ARMS

    def bake(self, claude):
        tl, m = self.tl, self.m
        fr, n = tl.frames, tl.n
        X, Z, SPIN, SQ, ARMS = self.sample()
        sec_at = [m.section_at(t) for t in tl.times]
        sway_amp = {"intro": 0.05, "run": 0.06, "hold": 0.0, "stops": 0.05, "groove": 0.08, "light": 0.11,
                    "light_bass": 0.08, "ritard": 0.05, "break": 0.0, "stops2": 0.05, "chorus": 0.1,
                    "breakdown": 0.0, "chorus2": 0.1, "outro": 0.07}
        amp = gaussian_smooth([sway_amp[s] for s in sec_at], 0.3 * FPS)
        lean = [a * math.cos(math.pi * m.beat_phase(t)) for a, t in zip(amp, tl.times)]
        bounce = [a * 0.5 * math.exp(-((m.beat_phase(t) + 0.5) % 1.0 - 0.5) ** 2 / 0.012)
                  for a, t in zip(amp, tl.times)]
        breathe = [0.012 * math.sin(2 * math.pi * t / 3.1) for t in tl.times]
        # look target: the cat's head, or the camera
        look_plan = {"intro": "cat", "run": "cat", "hold": "camera", "stops": "camera", "groove": "cat",
                     "light": "cat", "light_bass": "camera", "ritard": "cat", "break": "cat",
                     "stops2": "camera", "chorus": "cat", "breakdown": "camera", "chorus2": "camera",
                     "outro": "cat"}
        cat_heads = [self.cat.body_matrix(i) @ Vector((0.15, 0, 0.07)) for i in range(n)]
        yaw_look, face_x, face_z = [], [], []
        for i, t in enumerate(tl.times):
            target = cat_heads[i] if look_plan[sec_at[i]] == "cat" else CAMERA_GUESS
            d = target - Vector((X[i], Y_CLAUDE, KEY_Z + 0.1 + Z[i]))
            yaw = math.atan2(d.x, -d.y)                      # 0 = facing the camera (-Y)
            yaw_look.append(clamp(yaw, -0.6, 0.6) * 0.55)
            face_x.append(clamp(yaw - yaw_look[-1], -0.8, 0.8) * 0.012)
            face_z.append(clamp(math.atan2(d.z, math.hypot(d.x, d.y)), -0.6, 0.6) * 0.01)
        yaw_look = gaussian_smooth(yaw_look, 0.2 * FPS)
        face_x = gaussian_smooth(face_x, 0.1 * FPS)
        face_z = gaussian_smooth(face_z, 0.1 * FPS)
        self.world_head = [Vector((X[i], Y_CLAUDE, KEY_Z + 0.1 + Z[i])) for i in range(n)]
        bake(claude.root, "location", 0, fr, X)
        bake(claude.root, "location", 1, fr, [Y_CLAUDE] * n)
        bake(claude.root, "location", 2, fr, [KEY_Z + z for z in Z])
        bake(claude.root, "rotation_euler", 2, fr, [a + b for a, b in zip(yaw_look, SPIN)], 1e-3)
        sz = [max(0.62, 1 + s - b + br) for s, b, br in zip(SQ, bounce, breathe)]
        sxy = [1 / math.sqrt(max(v, 0.3)) for v in sz]
        bake(claude.squash, "scale", 0, fr, sxy, 1e-3)
        bake(claude.squash, "scale", 1, fr, sxy, 1e-3)
        bake(claude.squash, "scale", 2, fr, sz, 1e-3)
        bake(claude.squash, "rotation_euler", 1, fr, lean, 1e-3)
        bake(claude.face, "location", 0, fr, face_x, 1e-4)
        bake(claude.face, "location", 2, fr, [Claude.LEG + Claude.H * 0.58 + v for v in face_z], 1e-4)
        # arms: up on hops, waving in the choruses
        wave = [(0.5 if s in ("chorus", "chorus2") else 0.0) * max(0.0, math.sin(math.pi * m.beat_phase(t)))
                for s, t in zip(sec_at, tl.times)]
        wave = gaussian_smooth(wave, 0.05 * FPS)
        for side, arm in zip((-1, 1), claude.arms):
            raise_ = [-0.25 + 1.1 * a + w for a, w in zip(ARMS, wave)]
            bake(arm, "rotation_euler", 1, fr, [-side * r for r in raise_], 1e-3)
        self._bake_expressions(claude, sec_at)

    def _bake_expressions(self, claude, sec_at):
        tl, m, rng = self.tl, self.m, self.rng
        S = m.sections
        plan = {"intro": "open", "run": "open", "hold": "wide", "stops": "happy", "groove": "happy",
                "light": "content", "light_bass": "happy", "ritard": "open", "break": "open",
                "stops2": "love", "chorus": "happy", "breakdown": "wide", "chorus2": "love",
                "outro": "content"}
        expr = [plan[s] for s in sec_at]
        for i, t in enumerate(tl.times):
            s = sec_at[i]
            if s == "run" and t > S["run"][1] - 0.6:
                expr[i] = "wide"
            if s == "hold" and t > S["stops"][0] - 0.5:
                expr[i] = "happy"
            if s == "break" and t > S["break"][0] + 1.05:
                expr[i] = "love"                            # right after the cat's slow blink
            if s == "chorus":
                bar = int((t - S["chorus"][0]) / (4 * m.beat_len(t)))
                expr[i] = "love" if bar % 4 in (2, 3) else "happy"
            if s == "outro" and t > m.notes[-1].on - 0.2:
                expr[i] = "love"
        blush = gaussian_smooth([1.0 if (e == "love" or s == "outro") else 0.0 for e, s in zip(expr, sec_at)],
                                0.12 * FPS)
        # blinks for open eyes
        blink = [1.0] * tl.n
        t = 1.4
        while t < m.length:
            j = tl.idx(t)
            for d, v in enumerate((0.5, 0.08, 0.08, 0.6, 1.0)):
                if j + d < tl.n:
                    blink[j + d] = v
            t += 2.2 + rng.random() * 3.0
        # heart pulse on the beat
        pulse = [1.0 + 0.16 * math.exp(-((((m.beat_phase(t) + 0.1) % 1.0) - 0.1) / 0.08) ** 2)
                 for t in tl.times]
        for name, grp in claude.expr.items():
            vis = [1.0 if e == name else 0.0 for e in expr]
            # pop in with a little overshoot, out quickly
            val, cur = [], 0.0
            since = 99
            for i, v in enumerate(vis):
                if v > 0 and (i == 0 or vis[i - 1] == 0):
                    since = 0
                since += 1
                if v > 0:
                    cur = min(1.0, since / 3.0)
                    cur *= 1 + 0.18 * math.sin(math.pi * clamp(since / 5.0, 0, 1))
                else:
                    cur = max(0.0, cur - 0.5)
                val.append(cur)
            sx = [v * (pulse[i] if name == "love" else 1.0) for i, v in enumerate(val)]
            sz = [v * (blink[i] if name in ("open", "wide") else 1.0) *
                  (pulse[i] if name == "love" else 1.0) for i, v in enumerate(val)]
            bake(grp, "scale", 0, tl.frames, sx, 1e-3)
            bake(grp, "scale", 1, tl.frames, [1.0] * tl.n, 0)
            bake(grp, "scale", 2, tl.frames, sz, 1e-3)
        bake(claude.blush, "scale", 0, tl.frames, blush, 1e-3)
        bake(claude.blush, "scale", 2, tl.frames, blush, 1e-3)
        self.expr_track = expr


# ============================================================================
# 8. CAMERA DIRECTOR
# ============================================================================
SHOTS = (
    # section, bar offset, subject, azimuth, elevation, distance factor, focal (mm),
    #   drift (deg/s), distance factor at the end of the shot
    ("intro", 0, "cat", 35, 9, 1.0, 55, -0.8, 0.9),
    ("run", 0, "two", 2, 13, 1.05, 40, 0.5, 1.0),
    ("hold", 0, "claude", -12, 6, 1.15, 60, 0.0, 0.8),
    ("stops", 0, "two", 5, 17, 1.12, 35, 0.3, 1.05),
    ("groove", 0, "two", -18, 12, 1.0, 40, 1.0, 0.95),
    ("light", 0, "cat", 28, 11, 1.0, 55, -0.8, 0.9),
    ("light_bass", 0, "two", -8, 15, 1.05, 40, 0.4, 1.0),
    ("light_bass", 4, "claude", -22, 9, 1.0, 55, 0.6, 0.9),
    ("light_bass", 8, "two", 16, 13, 1.0, 40, -0.4, 1.0),
    ("ritard", 0, "cat", 34, 6, 0.95, 55, -0.5, 0.85),
    ("break", 0, "two", 0, 5, 0.95, 50, 0.0, 0.82),
    ("stops2", 0, "two", -6, 18, 1.12, 35, 0.3, 1.05),
    ("chorus", 0, "two", 18, 12, 1.0, 40, -0.6, 0.95),
    ("chorus", 4, "claude", -18, 8, 1.0, 50, 0.5, 0.9),
    ("breakdown", 0, "claude", 0, 6, 0.95, 60, 0.0, 0.85),
    ("chorus2", 0, "two", 0, 16, 1.1, 35, 0.4, 1.0),
    ("chorus2", 4, "cat", -26, 10, 1.0, 50, 0.6, 0.9),
    ("chorus2", 6, "two", 20, 12, 1.0, 40, -0.5, 1.0),
    ("outro", 0, "two", 0, 12, 1.0, 40, 0.0, 1.65),
)
SENSOR = 36.0


class CameraAnim:
    def __init__(self, music, tl, cat, claude, rng):
        self.m, self.tl, self.cat, self.claude, self.rng = music, tl, cat, claude, rng

    def bake(self, cam, focus):
        m, tl = self.m, self.tl
        n = tl.n
        shots = sorted((m.bar_time(sec, bo), sec, bo, subj, az, el, dist, focal, drift, dist_end)
                       for sec, bo, subj, az, el, dist, focal, drift, dist_end in SHOTS)
        shots[0] = (0.0,) + shots[0][1:]
        starts = [s[0] for s in shots]
        claude_ground = [Vector((x, Y_CLAUDE, KEY_Z + 0.075)) for x in self.claude.sample()[0]]
        cat_c = [self.cat.body_matrix(i) @ Vector((0.03, 0, 0.03)) for i in range(n)]
        cat_head = [self.cat.body_matrix(i) @ Vector((0.19, 0, 0.08)) for i in range(n)]
        cat_rump = [self.cat.body_matrix(i) @ Vector((-0.12, 0, 0.0)) for i in range(n)]
        # two-shots frame where both are now *and* over the next 0.6 s
        back, ahead = int(0.3 * FPS), int(0.6 * FPS)
        xs = [min(claude_ground[i].x - 0.09, cat_head[i].x, cat_rump[i].x) for i in range(n)]
        xe = [max(claude_ground[i].x + 0.09, cat_head[i].x, cat_rump[i].x) for i in range(n)]
        span_lo = [min(xs[max(0, i - back):i + ahead]) for i in range(n)]
        span_hi = [max(xe[max(0, i - back):i + ahead]) for i in range(n)]
        raw_c, raw_d, raw_dir, raw_f, raw_focus, shot_id = [], [], [], [], [], []
        for i, t in enumerate(tl.times):
            k = max(bisect.bisect_right(starts, t) - 1, 0)
            t0, sec, bo, subj, az, el, dist, focal, drift, dist_end = shots[k]
            t1 = starts[k + 1] if k + 1 < len(shots) else m.length
            u = clamp((t - t0) / max(t1 - t0, 1e-3), 0, 1)
            hfov = 2 * math.atan(SENSOR / 2 / focal)
            a, c = claude_ground[i], cat_c[i]
            if subj == "two":
                center = (a + c) / 2
                center.x = (span_lo[i] + span_hi[i]) / 2
                half = (span_hi[i] - span_lo[i]) / 2 + 0.13
                focus_p = (a + c) / 2
            elif subj == "cat":
                center = c + Vector((0.04 * math.cos(self.cat.yaw[i]), 0, 0.02))
                half = 0.25
                focus_p = self.cat.body_matrix(i) @ Vector((0.15, 0, 0.08))
            else:
                center = a + Vector((0, 0, 0.01))
                half = 0.15
                focus_p = a + Vector((0, -Claude.D / 2, 0.02))
            d = half / math.tan(hfov / 2) * lerp(dist, dist_end, smooth(u))
            if subj == "cat":                          # film the cat from the side it faces
                side = 1 if math.cos(self.cat.yaw[self.tl.idx(t0 + 0.05)]) > 0 else -1
                az, drift = abs(az) * side, drift * side
            az_r = math.radians(az + drift * (t - t0))
            el_r = math.radians(el)
            direction = Vector((math.sin(az_r) * math.cos(el_r), -math.cos(az_r) * math.cos(el_r), math.sin(el_r)))
            raw_c.append(center)
            raw_d.append(d)
            raw_dir.append(direction)
            raw_f.append(focal)
            raw_focus.append(focus_p)
            shot_id.append(k)
        # smooth inside each shot only (cuts stay cuts)
        cam_pos, look, focus_pos = [None] * n, [None] * n, [None] * n
        i = 0
        while i < n:
            j = i
            while j < n and shot_id[j] == shot_id[i]:
                j += 1
            seg = range(i, j)
            sig = 0.55 * FPS
            cx = gaussian_smooth([raw_c[k].x for k in seg], sig)
            cy = gaussian_smooth([raw_c[k].y for k in seg], sig)
            cz = gaussian_smooth([raw_c[k].z for k in seg], sig * 1.5)
            dd = gaussian_smooth([raw_d[k] for k in seg], sig)
            fx = gaussian_smooth([raw_focus[k].x for k in seg], 0.25 * FPS)
            fy = gaussian_smooth([raw_focus[k].y for k in seg], 0.25 * FPS)
            fz = gaussian_smooth([raw_focus[k].z for k in seg], 0.5 * FPS)
            for q, k in enumerate(seg):
                c = Vector((cx[q], cy[q], cz[q]))
                look[k] = c
                cam_pos[k] = c + raw_dir[k] * dd[q]
                focus_pos[k] = Vector((fx[q], fy[q], fz[q]))
            i = j
        # handheld + punches on the big chords
        accents = [d for sec in ("stops", "stops2") for d in m.downbeats
                   if m.sections[sec][0] - 1e-3 <= d < m.sections[sec][1]]
        accents += [m.sections["chorus"][0], m.sections["chorus2"][0]]
        rots, locs = [], []
        prev = None
        for i, t in enumerate(tl.times):
            shake = 0.0
            for a in accents:
                if 0 <= t - a < 0.5:
                    shake += 0.006 * math.exp(-(t - a) / 0.12) * math.sin(2 * math.pi * 11 * (t - a))
            hand = Vector((wobble(t, 1.0), wobble(t, 2.0) * 0.5, wobble(t, 3.0))) * 0.0035
            p = cam_pos[i] + hand + Vector((0, 0, shake))
            tgt = look[i] + Vector((wobble(t * 0.7, 4.0), 0, wobble(t * 0.7, 5.0))) * 0.003
            e = look_rotation(p, tgt, prev)
            prev = e
            locs.append(p)
            rots.append(e)
        fr = tl.frames
        bake_vec(cam, "location", fr, locs, 1e-4)
        bake_vec(cam, "rotation_euler", fr, rots, 2e-4)
        bake(cam.data, "lens", 0, fr, raw_f, 0)
        for a in (0, 1, 2):
            bake(focus, "location", a, fr, [v[a] for v in focus_pos], 1e-4)
        # set cut keys to CONSTANT so the camera does not glide between shots
        self.positions = locs
        self.cut_frames = [tl.frames[i] for i in range(1, n) if shot_id[i] != shot_id[i - 1]]
        self._harden_cuts(cam)
        self._harden_cuts(cam.data)
        self._harden_cuts(focus)

    def _harden_cuts(self, idb):
        ad = idb.animation_data
        if not ad or not ad.action:
            return
        fcs = _action_fcurves(ad)
        cuts = set(self.cut_frames)
        for fc in fcs:
            pts = fc.keyframe_points
            frames = [kp.co.x for kp in pts]
            # make sure there is a key on the frame before each cut, then hold it
            for c in cuts:
                v_before = fc.evaluate(c - 1)
                v_at = fc.evaluate(c)
                if abs(v_before - v_at) < 1e-9:
                    continue
                if (c - 1) not in frames:
                    pts.insert(c - 1, v_before, options={"FAST"})
                if c not in frames:
                    pts.insert(c, v_at, options={"FAST"})
            fc.update()
            for kp in fc.keyframe_points:
                if int(round(kp.co.x)) + 1 in cuts:
                    kp.interpolation = "CONSTANT"
                elif kp.interpolation == "BEZIER":
                    kp.interpolation = "LINEAR"
            fc.update()


def _action_fcurves(ad):
    act = ad.action
    if hasattr(act, "fcurves") and not hasattr(act, "layers"):
        return list(act.fcurves)
    out = []
    try:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
        if cb:
            out = list(cb.fcurves)
    except Exception:
        pass
    if not out and hasattr(act, "fcurves"):
        out = list(act.fcurves)
    return out


# ============================================================================
# 9. SCENE, RENDER, AUDIO
# ============================================================================
def fresh_scene():
    """(Re)create our scene. Only data that belonged to the old version of this
    scene is removed; other scenes and their data are not touched."""
    old = bpy.data.scenes.get(SCENE_NAME)
    if old is not None:
        owned = set()

        def own(idb):
            if idb is not None:
                owned.add(idb)
                ad = getattr(idb, "animation_data", None)
                if ad and ad.action:
                    owned.add(ad.action)

        for o in old.objects:
            if len(o.users_scene) <= 1:
                own(o)
                own(o.data)
                for slot in getattr(o, "material_slots", []):
                    own(slot.material)
                for m in getattr(o.data, "materials", []) or []:
                    own(m)
        own(old.world)
        own(getattr(old, "compositing_node_group", None))
        if old.animation_data and old.animation_data.action:
            owned.add(old.animation_data.action)
        for o in [i for i in owned if isinstance(i, bpy.types.Object)]:
            bpy.data.objects.remove(o, do_unlink=True)
        for c in list(old.collection.children_recursive):
            bpy.data.collections.remove(c)
        if len(bpy.data.scenes) == 1:
            bpy.data.scenes.new("Scene")
        bpy.data.scenes.remove(old)
        stores = ((bpy.types.Mesh, bpy.data.meshes), (bpy.types.Material, bpy.data.materials),
                  (bpy.types.Light, bpy.data.lights), (bpy.types.Camera, bpy.data.cameras),
                  (bpy.types.Curve, bpy.data.curves), (bpy.types.Action, bpy.data.actions),
                  (bpy.types.World, bpy.data.worlds), (bpy.types.NodeTree, bpy.data.node_groups))
        for idb in owned:
            try:
                if isinstance(idb, bpy.types.Object) or idb.users > 0:
                    continue
                for typ, store in stores:
                    if isinstance(idb, typ):
                        store.remove(idb)
                        break
            except ReferenceError:
                pass
    scene = bpy.data.scenes.new(SCENE_NAME)
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    return scene


def setup_render(scene, n_frames, audio):
    r = scene.render
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_START + n_frames - 1
    r.resolution_x, r.resolution_y = RESOLUTION
    r.resolution_percentage = RESOLUTION_PERCENT
    if ENGINE.upper() == "CYCLES":
        r.engine = "CYCLES"
        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.use_denoising = True
    else:
        for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                r.engine = eng
                break
            except TypeError:
                continue
        ee = scene.eevee
        ee.taa_render_samples = EEVEE_SAMPLES
        for attr, val in (("use_shadows", True), ("shadow_ray_count", 2), ("shadow_step_count", 8),
                          ("use_raytracing", True), ("use_gtao", True), ("use_bloom", True),
                          ("bloom_intensity", 0.04), ("use_soft_shadows", True)):
            if hasattr(ee, attr):
                try:
                    setattr(ee, attr, val)
                except (TypeError, AttributeError):
                    pass
    r.use_motion_blur = MOTION_BLUR
    r.motion_blur_shutter = 0.4
    # shutter opens on the frame: with the default (centred) shutter the frame
    # of every camera cut would be smeared across both shots
    r.motion_blur_position = "START"
    vs = scene.view_settings
    try:
        vs.view_transform = "AgX"
    except TypeError:
        vs.view_transform = "Filmic"
    for look in ("AgX - Medium High Contrast", "AgX - Base Contrast", "Medium High Contrast"):
        try:
            vs.look = look
            break
        except TypeError:
            continue
    vs.exposure = 0.0
    # output: MP4 (H.264 + AAC)
    im = r.image_settings
    if hasattr(im, "media_type"):
        im.media_type = "VIDEO"
    im.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.ffmpeg.ffmpeg_preset = "GOOD"
    r.ffmpeg.audio_codec = "AAC" if audio else "NONE"
    r.ffmpeg.audio_bitrate = 256
    r.ffmpeg.audio_channels = "STEREO"
    r.filepath = OUTPUT_FILE
    _setup_bloom(scene)


def _setup_bloom(scene):
    """Soft bloom on candles and fairy lights (compositor glare)."""
    try:
        if hasattr(scene, "compositing_node_group"):                 # Blender 5.x
            ng = bpy.data.node_groups.new(f"{SCENE_NAME}.Comp", "CompositorNodeTree")
            scene.compositing_node_group = ng
            rl = ng.nodes.new("CompositorNodeRLayers")
            gl = ng.nodes.new("CompositorNodeGlare")
            out = ng.nodes.new("NodeGroupOutput")
            ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            for ident in ("Bloom", "BLOOM"):
                try:
                    gl.inputs["Type"].default_value = ident
                    break
                except (TypeError, ValueError):
                    continue
            gl.inputs["Threshold"].default_value = 1.2
            gl.inputs["Strength"].default_value = 0.35
            gl.inputs["Size"].default_value = 0.5
            ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
            ng.links.new(gl.outputs[0], out.inputs[0])
        else:                                                         # Blender 4.x
            scene.use_nodes = True
            nt = scene.node_tree
            nt.nodes.clear()
            rl = nt.nodes.new("CompositorNodeRLayers")
            gl = nt.nodes.new("CompositorNodeGlare")
            comp = nt.nodes.new("CompositorNodeComposite")
            try:
                gl.glare_type = "BLOOM"
            except TypeError:
                gl.glare_type = "FOG_GLOW"
            if "Threshold" in gl.inputs:                              # 4.5: settings are sockets
                gl.inputs["Threshold"].default_value = 1.2
                gl.inputs["Strength"].default_value = 0.35
                gl.inputs["Size"].default_value = 0.5
            else:                                                     # 4.2 - 4.4
                gl.threshold = 1.2
                gl.mix = -0.6
                gl.size = 7
            nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
            nt.links.new(gl.outputs["Image"], comp.inputs["Image"])
        scene.render.use_compositing = True
    except Exception as exc:                                          # bloom is optional
        print("[FallingInLove] bloom skipped:", exc)


def add_audio(scene, path):
    se = scene.sequence_editor or scene.sequence_editor_create()
    strips = se.strips if hasattr(se, "strips") else se.sequences
    snd = strips.new_sound("Song", path, channel=1, frame_start=FRAME_START)
    scene.sync_mode = "AUDIO_SYNC"
    scene.use_audio_scrub = True
    return snd


def fade(scene, n_frames):
    vs = scene.view_settings
    f0, f1 = FRAME_START, FRAME_START + n_frames - 1
    pts = [(f0, -9.0), (f0 + int(0.6 * FPS), 0.0), (f1 - int(1.6 * FPS), 0.0), (f1, -9.0)]
    for f, v in pts:
        vs.exposure = v
        vs.keyframe_insert("exposure", frame=f)
    vs.exposure = 0.0


# ============================================================================
# 10. MAIN
# ============================================================================
def build():
    rng = random.Random(SEED)
    midi_path = find_midi()
    audio_path = find_audio()
    scene = fresh_scene()
    song_len = SONG_LENGTH
    if audio_path:
        snd = add_audio(scene, audio_path)
        song_len = max(snd.frame_final_duration / FPS, 1.0) if hasattr(snd, "frame_final_duration") else song_len
    music = Music(midi_path, song_len)
    tl = Timeline(music)
    print(f"[FallingInLove] MIDI: {len(music.notes)} notes via {music.reader}; "
          f"audio: {audio_path or 'none'}; {tl.n} frames @ {FPS} fps")
    scene.unit_settings.system = "METRIC"
    B = Builder(scene)
    build_world(scene)
    keys, mats = build_piano(B)
    build_room(B, mats)
    build_candles(B, mats)
    build_fairy_lights(B)
    build_lights(B)
    claude = Claude(B)
    cat = Cat(B)
    # camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_width = SENSOR
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 2.4
    cam_data.clip_start = 0.02
    cam = B.obj("Camera", cam_data, "Camera")
    focus = B.empty("Camera.Focus", "Camera", size=0.03)
    cam_data.dof.focus_object = focus
    scene.camera = cam

    # animation, in dependency order
    key_anim = KeyAnim(music)
    key_anim.apply(keys, tl.n)
    cat_anim = CatAnim(music, key_anim, tl, rng)
    claude_anim = ClaudeAnim(music, tl, cat_anim, rng)
    cat_anim.plan_paws()
    claude_anim.bake(claude)
    cam_anim = CameraAnim(music, tl, cat_anim, claude_anim, rng)
    cam_anim.bake(cam, focus)
    cat_anim.bake(cat, claude_anim.world_head, cam_anim.positions)
    setup_render(scene, tl.n, audio_path)
    fade(scene, tl.n)
    scene.frame_set(FRAME_START + int(16.0 * FPS))
    # expose for checks / tools
    scene["fil_notes"] = len(music.notes)
    return dict(scene=scene, music=music, tl=tl, keys=keys, key_anim=key_anim, cat=cat, cat_anim=cat_anim,
                claude=claude, claude_anim=claude_anim, camera=cam, cam_anim=cam_anim, audio=audio_path)


def _cli():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return argv


if __name__ == "__main__":
    RESULT = build()
    args = _cli()
    if "--save" in args:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args[args.index("--save") + 1]))
    if "--render" in args:
        bpy.ops.render.render(animation=True, scene=RESULT["scene"].name)
    print("[FallingInLove] done.")
