"""Build the scene headless and verify the animation against the MIDI.

    blender -b --python tools/check_scene.py            (or: python check_scene.py with the bpy module)

Checks
  1. keys:     a key is only ever down while a character is on it (a cat
               paw at its surface, or one of Claude's feet)
  2. paws:     every cat paw contact lies on the struck key (inside its x/y
               bounds, within 3 mm of its top surface) while the key is down
  3. surfaces: paws never sink more than 4 mm into a key, never float away
               from the keyboard (except during hops/turns)
  4. spacing:  Claude and the cat never overlap
  5. framing:  the subjects of every shot stay inside the picture
  6. sanity:   no NaN/inf in any F-curve, cat legs never stretch absurdly
Exit code 1 if anything fails.
"""
import math
import os
import runpy
import sys

import bpy  # noqa: F401  (must come before bpy_extras)
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "blender", "falling_in_love.py")

mod = runpy.run_path(SCRIPT, run_name="falling_in_love")
R = mod["build"]()
scene, music, tl = R["scene"], R["music"], R["tl"]
keys, cat, claude, cat_anim = R["keys"], R["cat"], R["claude"], R["cat_anim"]
frame_of, key_x, key_top, key_theta = mod["frame_of"], mod["key_x"], mod["key_top"], mod["key_theta"]
is_black, key_at = mod["is_black"], mod["key_at"]
WHITE_W, BLACK_W, WHITE_L, BLACK_L = mod["WHITE_W"], mod["BLACK_W"], mod["WHITE_L"], mod["BLACK_L"]
KEY_Z = mod["KEY_Z"]
fails = []


def fail(msg):
    fails.append(msg)


def fc_value(obj, path, index, f):
    ad = obj.animation_data
    for fc in mod["_action_fcurves"](ad):
        if fc.data_path == path and fc.array_index == index:
            return fc.evaluate(f)
    return getattr(obj, path.strip('[]"')) if not path.startswith("[") else obj[path[2:-2]]


# 1. keys ---------------------------------------------------------------------
# every frame a key is down, a character must be touching that key: a cat paw
# on it (within its bounds, at its surface) or one of Claude's feet standing on it
ka = R["key_anim"]
X, Z = R["claude_anim"].sample()[:2]
CFX = mod["CLAUDE_FOOT_X"]
pressed_frames = untouched = 0
by_char = {"cat": 0, "claude": 0}
for p, (fs, vs) in ka.pts.items():
    half = (BLACK_W if is_black(p) else WHITE_W) / 2
    for f in range(max(fs[0], tl.frames[0]), min(fs[-1], tl.frames[-1]) + 1):
        if ka.press(p, f) < 0.05:
            continue
        pressed_frames += 1
        i = f - tl.frames[0]
        who = None
        for paw in cat.paws:
            q = cat_anim.paw_world[paw][i]
            if (abs(q.x - key_x(p)) <= half + 0.004 and -(BLACK_L if is_black(p) else WHITE_L) <= q.y <= 0
                    and -0.006 < q.z - ka.surface_z(p, q.y, f) < 0.015):   # on it, or mid-strike
                who = "cat"
                break
        if who is None and Z[i] < 1e-3:
            for sd in (-1, 1):
                if key_at(X[i] + sd * CFX, mod["Y_CLAUDE"]) == p:
                    who = "claude"
        if who is None:
            untouched += 1
            if untouched <= 8:
                q = {pw: tuple(round(v, 4) for v in cat_anim.paw_world[pw][i]) for pw in cat.paws}
                print("   ", p, f, round(ka.press(p, f), 2), round(key_x(p), 4), round(ka.surface_z(p, -0.1, f), 4), q,
                      "claude", round(X[i], 3), round(Z[i], 3))
                fail(f"key {p} is down at frame {f} ({mod['time_of'](f):.2f}s) but nobody is on it")
        else:
            by_char[who] += 1
n_down = sum(len(v) for v in ka.onsets.values())
print(f"[check] keys: {n_down} key strikes, {pressed_frames} key-down frames, "
      f"{untouched} without a character on the key (cat {by_char['cat']}, Claude {by_char['claude']})")

# 2 + 3. paws -----------------------------------------------------------------
contacts = cat_anim.contacts
bad = 0
for f, p, paw in contacts:
    obj = cat.paws[paw]
    pos = Vector([fc_value(obj, "location", a, f) for a in range(3)])
    half = (BLACK_W if is_black(p) else WHITE_W) / 2
    y_front = -(BLACK_L if is_black(p) else WHITE_L)
    surf = R["key_anim"].surface_z(p, pos.y, f)
    press = R["key_anim"].press(p, f)
    ok = (abs(pos.x - key_x(p)) <= half + 1e-4 and y_front <= pos.y <= 0 and abs(pos.z - surf) < 0.003
          and press > 0.99)
    if not ok:
        bad += 1
        if bad <= 10:
            fail(f"paw {paw} at frame {f} misses key {p}: pos {tuple(round(v, 4) for v in pos)}, "
                 f"key x {key_x(p):.4f}, surface {surf:.4f}, press {press:.2f}")
print(f"[check] paws: {len(contacts) - bad}/{len(contacts)} contacts land on their key while it sounds "
      f"({len({c[1] for c in contacts})} different keys)")
if len(contacts) < 150:
    fail(f"only {len(contacts)} paw contacts - the cat hardly plays")

sink, flo = 0.0, 0
for paw, obj in cat.paws.items():
    pos = cat_anim.paw_world[paw]
    for i, f in enumerate(tl.frames):
        t = tl.times[i]
        if cat_anim.in_window(t):
            continue
        p = key_at(pos[i].x, pos[i].y)
        surf = R["key_anim"].surface_z(p, pos[i].y, f)
        sink = max(sink, surf - pos[i].z)
        if pos[i].z - surf > 0.06:
            flo += 1
if sink > 0.004:
    fail(f"a paw sinks {sink * 1000:.1f} mm into a key")
print(f"[check] paws: max penetration {sink * 1000:.1f} mm, frames with a paw > 6 cm up: {flo}")

# 4. spacing ------------------------------------------------------------------
min_gap, worst = 9.0, None
for i, f in enumerate(tl.frames):
    scene.frame_set(f) if i % 3 == 0 else None
    cx = fc_value(claude.root, "location", 0, f)
    m = cat_anim.body_matrix(i)
    pts = [m @ Vector(v) for v in ((0.2, 0, 0.07), (-0.13, 0, 0), (0.1, 0, 0), (0, 0, 0))]
    gap = min(p.x for p in pts) - (cx + 0.09)
    if gap < min_gap:
        min_gap, worst = gap, tl.times[i]
if min_gap < 0.0:
    fail(f"Claude and the cat overlap by {-min_gap * 100:.1f} cm at {worst:.2f}s")
print(f"[check] spacing: min gap Claude -> cat {min_gap * 100:.1f} cm (at {worst:.2f}s)")

# 5. framing ------------------------------------------------------------------
cam = R["camera"]
out = 0
worst_f = []
# intended: Claude leaps out of the top of the close-up; the cut to the wide
# shot comes with the landing on the first stop chord
allowed = [(music.sections["stops"][0] - 0.6, music.sections["stops"][0]),
           (music.sections["chorus2"][0] - 0.6, music.sections["chorus2"][0])]
for i in range(0, tl.n, 2):
    f = tl.frames[i]
    scene.frame_set(f)
    t = tl.times[i]
    if any(a <= t < b for a, b in allowed):
        continue
    sec = music.section_at(t)
    subjects = []
    ch = claude.root.matrix_world.translation + Vector((0, 0, 0.08))
    ct = cat.torso.matrix_world @ Vector((0.05, 0, 0.03))
    subjects = [("claude", ch), ("cat", ct)]
    # which subject(s) must be visible? from the shot list
    shots = sorted((music.bar_time(s[0], s[1]), s[2]) for s in mod["SHOTS"])
    kind = [k for t0, k in shots if t0 <= t + 1e-6][-1] if t >= shots[0][0] else shots[0][1]
    need = {"two": ("claude", "cat"), "cat": ("cat",), "claude": ("claude",)}[kind]
    for name, p in subjects:
        if name not in need:
            continue
        v = world_to_camera_view(scene, cam, p)
        if not (0.03 < v.x < 0.97 and 0.03 < v.y < 0.97 and v.z > 0):
            out += 1
            if len(worst_f) < 8:
                worst_f.append(f"{name} out of frame at {t:.2f}s ({sec}, {kind} shot): ({v.x:.2f}, {v.y:.2f})")
if out > 0.01 * tl.n / 2:
    fail(f"{out} sampled frames with a subject out of frame")
for w in worst_f:
    print("   ", w)
print(f"[check] framing: {out} of {tl.n // 2} sampled frames have a subject outside the picture")

# 6. sanity -------------------------------------------------------------------
nan = 0
for o in scene.objects:
    for idb in (o, getattr(o, "data", None)):
        ad = getattr(idb, "animation_data", None) if idb is not None else None
        if not ad or not ad.action:
            continue
        for fc in mod["_action_fcurves"](ad):
            for kp in fc.keyframe_points:
                if not all(math.isfinite(c) for c in kp.co):
                    nan += 1
if nan:
    fail(f"{nan} non-finite keyframes")
stretch_max = 0.0
for i in range(0, tl.n, 5):
    m = cat_anim.body_matrix(i)
    for paw in cat.paws:
        sh = mod["Cat"].SHOULDER if paw[0] == "F" else mod["Cat"].HIP
        side = 1 if paw[1] == "L" else -1
        hip = m @ Vector((sh[0], side * sh[1], sh[2]))
        L = (cat_anim.paw_world[paw][i] + Vector((0, 0, 0.011)) - hip).length
        stretch_max = max(stretch_max, L / 0.1)
if stretch_max > 2.0:
    fail(f"a cat leg stretches to {stretch_max:.2f}x its rest length")
print(f"[check] sanity: {nan} bad keys, max leg stretch {stretch_max:.2f}x")
n_keys = sum(len(fc.keyframe_points) for o in scene.objects for idb in (o, getattr(o, 'data', None))
             if idb is not None and getattr(idb, "animation_data", None) and idb.animation_data.action
             for fc in mod["_action_fcurves"](idb.animation_data))
print(f"[check] {len(scene.objects)} objects, {n_keys} keyframes, {tl.n} frames")

if fails:
    print("\nFAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("\nall checks passed")
