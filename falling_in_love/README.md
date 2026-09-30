# this is what falling in love feels like — Claude & the cat at the piano

A Blender scene built and animated entirely by one Python script: a candle-lit
room, an upright piano with string lights, the **Claude avatar** (a coral box
with little legs and dynamic eyes) and a **black cat**, playing JVKE's
*this is what falling in love feels like* together.

* all **916 notes** of the piano MIDI press their key on the exact frame
  (with a warm glow while the key is down)
* the **cat plays the right hand**: it walks, sits and stretches on the
  keyboard, and its front paws land on real melody notes at the moment they
  sound (185 paw strikes, each verified to hit its key)
* **Claude plays the left hand**: it hops onto the bass notes on the beat,
  shifts its weight, spins on the big chords and changes its eyes
  (open, happy ^ ^, content, wide, hearts that pulse on the beat)
* a small **camera director** cuts on bar lines, keeps both in frame and pulls
  focus

## Files

| Path | What |
|------|------|
| `blender/falling_in_love.py` | **The deliverable.** Self-contained script for Blender's Scripting workspace |
| `midi/this_is_what_falling_in_love_feels_like_piano.mid` | The piano MIDI the animation is driven by |
| `tools/align_midi_to_audio.py` | Offline measurement of the MIDI → audio time warp (produced the `WARP` table in the script) |
| `tools/check_scene.py` | Builds the scene headless and verifies keys, paws, spacing, framing |
| `tools/render_frames.py` | Resumable headless frame renderer |
| `tools/make_video.sh` | Muxes the frames with the song into the MP4 |

The song itself is not part of this repository.

## Which input files are used, and why

| File | Used? | Reason |
|------|:-----:|--------|
| Piano MIDI (`…JVKE_For_Piano.mid`) | **yes** | 2 tracks (right hand 574 notes, left hand 342), B major like the record, 119.4 s long like the record (120.3 s). Drives every key, the cat's paws and Claude's hops. |
| "Violin" MIDI (`…JVKE.mid`) | no | Actually a string quartet (2 violins, viola, cello) in **C major**, a semitone away from the recording (chroma correlation: best at 11 semitones up), 96 s long, different form. It cannot be synchronised to the song and nothing on screen plays strings. |
| PDF "…JVKE" | no | The score of that string quartet (Gustavo Monteiro). |
| PDF "…fanmade extended version" | no | A different fan arrangement (3/4, E-flat major, 96/126 bpm). |
| MP3 | yes | Soundtrack, and the timing reference for the sync measurement. |

Sheet music carries no timing, so neither PDF adds information the MIDI does
not already have.

## Sync

The MIDI follows the song's form but not its exact timing; its tempo map is
hand-made. A constant offset does not work: the audio is **0.30 s** behind the
MIDI at the first note, drifts to **1.0 s** during the first half, and after
the break at 1:05 it is **0.16 s** behind (the MIDI's break is ~0.7 s too long).

`tools/align_midi_to_audio.py` measures a piecewise-linear warp with one
anchor per MIDI beat (217 beats):

1. pitch-specific onset salience of the audio: CQT (1 bin per semitone),
   log magnitude, two-scale "superflux" (fast and slow attacks)
2. each MIDI note scores the salience at its fundamental, octave and
   octave + fifth at its warped onset
3. Viterbi over the audio time of every beat (5 ms grid, ±0.3 s around a prior
   through hand-checked structural anchors: first note, left-hand entrance,
   isolated tutti chords after silence), with a penalty on tempo changes that is
   10× weaker in the rubato intro and the ritardando
4. the rubato intro is refined per note on the steepest rise of the note's
   dB envelope (its lo-fi piano attacks are slow)

The prior is needed because the music is periodic: an unconstrained search
locks onto warps that are 2–3 whole beats off with almost the same score.
Result, simplified to 84 anchors (≤ 10 ms deviation): **median note error
16 ms**, i.e. less than half a frame at 30 fps (constant offset: 37 ms, and
wrong by up to a second). The table is embedded in the script as `WARP`, so the
audio strip simply starts at frame 1 with no offset. `AUDIO_OFFSET` only
exists for fine-tuning (e.g. a Bluetooth delay).

## Choreography

| Section | Audio time | Cat (right hand) | Claude (left hand) | Camera |
|---|---|---|---|---|
| intro | 0:00.3 – 0:06.7 | walks left, stepping on the solo melody | out of frame | cat, 3/4 front |
| run | 0:06.7 – 0:12.8 | turns, walks up the two-octave run | hops onto the two left-hand chords | two-shot |
| hold | 0:12.8 – 0:15.7 | sits at the top, watches Claude | long crouch, wide eyes, leaps out of frame | Claude close-up, push-in |
| stops | 0:15.7 – 0:25.0 | startled hop, then bats the pickups | big hop onto every tutti chord, one spin | wide, shake on the chords |
| groove | 0:25.0 – 0:34.3 | walks with the melody | hops every beat, bigger on the downbeat | two-shot |
| light | 0:34.3 – 0:39.9 | sits and bats the staccato chords | sways, content eyes | cat close-up |
| light_bass | 0:39.9 – 1:00.4 | walks, turns with the melody | hops every 2 beats along the bass | two-shot / Claude / two-shot |
| ritard | 1:00.4 – 1:04.5 | comes to the middle and **stretches** | hops over to the cat | cat close-up, low |
| break | 1:04.5 – 1:06.6 | sits face to face, **slow blink** | eyes turn into **hearts**, blush | close two-shot |
| stops2 | 1:06.6 – 1:15.5 | bats the chords, looks at camera | big hops, spins, hearts pulse | wide |
| chorus | 1:15.5 – 1:31.1 | walks with the melody | hops every beat, spin every 4 bars | cuts on bar lines |
| breakdown | 1:31.1 – 1:33.3 | playful hop | freezes, then the biggest leap | Claude close-up |
| chorus2 | 1:33.3 – 1:51.1 | walks with the melody | hearts, hops, spins | cuts on bar lines |
| outro | 1:51.1 – 2:00.3 | sits and plays the soft chords | spins in, hops over to the cat, content eyes, hearts on the last note | slow pull-back, fade out |

## How it works

* **MIDI reader**: `mido` if Blender's Python has it, otherwise a 60-line
  built-in Standard MIDI File reader. Both give the same 916 notes (and match
  `pretty_midi` to 1 µs).
* **Keys** pivot on a hinge behind the fallboard. A key is at rest one frame
  before its onset, fully down on the onset frame, held for the note (at least
  2 frames), then released over 2 frames; repeated notes get a partial lift.
  A per-key custom property `glow` drives the emission of a shared material.
* **Cat rig**: body, head, tail chain and four paw targets. The legs are
  *Stretch To* constraints from shoulder/hip to the paw target, so the
  animation only places paws; they always connect. A planner moves the body
  after the (smoothed) melody, decides when to turn around, chooses which paw
  strikes which note within reach, and adds plain steps whenever a paw falls
  behind the body. Hind paws alternate.
* **Claude**: squash-and-stretch hops (anticipation, stretch in the air,
  damped squash on landing), a weight shift that peaks on every beat, arms that
  fly up on big hops, eyes that look at the cat or the camera, and expression
  sets that pop in with a little overshoot.
* **Camera**: a shot list on bar lines (subject, angle, distance, focal
  length, drift, push-in). Two-shots frame where both characters are now and in
  the next 0.6 s. Smoothing happens inside each shot only, so cuts stay cuts.
* Everything is baked to F-curves (simplified with Ramer–Douglas–Peucker), so
  the file plays back in real time and needs no script at runtime.
* Compatible with Blender 4.2+ (slotted actions of 4.4+/5.x, legacy actions,
  both compositor APIs). Tested with **4.5.4 LTS** and **5.0.1**.

## Run it

**In Blender:** open `blender/falling_in_love.py` in the Scripting workspace,
put the MP3 somewhere in `falling_in_love/` (any name containing
"falling in love") or set `AUDIO_FILE` at the top, press *Run Script*. The scene
`FallingInLove` is created (re-running rebuilds it and leaves other scenes
alone). Space plays it with sound; *Render → Render Animation* writes
`//render/falling_in_love.mp4` (H.264 + AAC) next to the .blend.

**Headless:**

```sh
# build and save
FIL_AUDIO=song.mp3 blender -b --python blender/falling_in_love.py -- --save scene.blend
# or build and render the MP4 directly
FIL_AUDIO=song.mp3 blender -b --python blender/falling_in_love.py -- --render
# verify the animation against the MIDI
FIL_AUDIO=song.mp3 blender -b --python tools/check_scene.py
```

Environment overrides: `FIL_MIDI`, `FIL_AUDIO`, `FIL_OUTPUT`, `FIL_ENGINE`
(`EEVEE`/`CYCLES`), `FIL_FPS`, `FIL_RES` (e.g. `1280x720`), `FIL_SAMPLES`.

Settings in the script: 30 fps, 1920×1080, EEVEE 48 samples, motion blur,
depth of field, AgX. Only the key light and the rim light cast shadows (the
candles and string lights would cost a lot in EEVEE for little visible gain).

## Verification

`tools/check_scene.py` (Blender 5.0.1 and 4.5.4 LTS):

```
[check] keys: 916/916 notes strike on their frame
[check] paws: 185/185 contacts land on their key while it sounds (18 different keys)
[check] paws: max penetration 0.0 mm, frames with a paw > 6 cm up: 0
[check] spacing: min gap Claude -> cat 3.9 cm
[check] framing: 0 of 1805 sampled frames have a subject outside the picture
[check] sanity: 0 bad keys, max leg stretch 1.36x
all checks passed
```

## The rendered MP4

Rendered without a GPU (EEVEE on Mesa llvmpipe, 4 CPU cores), so with reduced
settings (`FIL_CPU_DRAFT=1`): 1280×720, 30 fps, 4 EEVEE samples, no
screen-space ray tracing, half-resolution shadow maps (~6 s per frame,
3,611 frames), then muxed with the MP3. On this machine one Blender process
was faster than two in parallel.

```sh
FIL_AUDIO=song.mp3 FIL_RES=1280x720 blender -b --python blender/falling_in_love.py -- --save final.blend
FIL_CPU_DRAFT=1 blender -b --python tools/render_frames.py -- final.blend frames 1 3611 1 1280x720 4
tools/make_video.sh frames song.mp3 falling_in_love.mp4
```

On a machine with a GPU the script's defaults (1080p, 48 samples) render in a
fraction of that time.

## Rebuild the sync table

```sh
pip install numpy scipy librosa pretty_midi
python tools/align_midi_to_audio.py song.mp3 midi/this_is_what_falling_in_love_feels_like_piano.mid warp.json
```

It prints the `WARP = (...)` table for the script and the error statistics.
