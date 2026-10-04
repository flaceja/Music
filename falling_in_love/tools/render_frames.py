"""Render frames of a saved build headless (used for the MP4 and for stills).

    python render_frames.py scene.blend out_dir  START END [STEP] [WIDTHxHEIGHT] [SAMPLES]
    python render_frames.py scene.blend out_dir  --times 3.0,16.5,65.2 [WIDTHxHEIGHT] [SAMPLES]

Frames are written as PNG (out_dir/frame_00001.png, ...) so an interrupted
render can resume: existing frames are skipped. tools/make_video.sh turns
them into the MP4 with the song.
"""
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
blend, out = argv[0], argv[1]
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend))
scene = bpy.data.scenes["FallingInLove"]
bpy.context.window_manager  # noqa
if argv[2] == "--times":
    fps = scene.render.fps
    frames = [scene.frame_start + int(round(float(t) * fps)) for t in argv[3].split(",")]
    rest = argv[4:]
else:
    start, end = int(argv[2]), int(argv[3])
    step = int(argv[4]) if len(argv) > 4 else 1
    frames = list(range(start, end + 1, step))
    rest = argv[5:]
if rest:
    w, h = (int(v) for v in rest[0].lower().split("x"))
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
if len(rest) > 1:
    scene.eevee.taa_render_samples = int(rest[1])
    scene.cycles.samples = int(rest[1])
if os.environ.get("FIL_CPU_DRAFT"):
    # cheaper EEVEE for CPU-only machines (software OpenGL): no screen-space ray
    # tracing, fewer shadow rays/steps, half-resolution shadow maps
    ee = scene.eevee
    ee.use_raytracing = False
    ee.shadow_ray_count, ee.shadow_step_count = 1, 4
    ee.shadow_resolution_scale = 0.5
im = scene.render.image_settings
if hasattr(im, "media_type"):
    im.media_type = "IMAGE"
im.file_format = "PNG"
im.color_mode = "RGB"
os.makedirs(out, exist_ok=True)
for f in frames:
    path = os.path.join(out, f"frame_{f:05d}.png")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    t0 = time.time()
    scene.frame_set(f)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=scene.name)
    print(f"[render] frame {f} ({(f - scene.frame_start) / scene.render.fps:.2f}s) {time.time() - t0:.1f}s",
          flush=True)
