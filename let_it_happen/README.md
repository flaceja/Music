# Let It Happen — music video (from 02:40.263)

Procedural music video for Tame Impala, *Let It Happen*, starting at song
position **02:40.263** (= video 00:00.000) and running to the end of the song
(05:07.34 of video, 160 bars at 125.0 BPM).

Every frame is computed from code. Nothing is AI-generated, nothing is stock
footage. All cuts are placed on a beat grid measured from the audio, the lyric
cards are placed on the (shifted) LRC word timestamps.

## Concept

One image carries the whole video: **a vermilion disc in a current of lines.**

* the lines are the contours of a fluid stream function: the *current*,
  "all this noise", everything that is happening
* the disc is the *self*. When it resists, the current has to bend around it,
  pile up and break into a turbulent wake
* the title is the instruction: *let it happen*. A body that moves **with** the
  current does not disturb it; seen from the body, the lines become straight
  and still

The song section after the cut is exactly this transformation, so the video
follows its structure chapter by chapter:

| # | Chapter | Video time | Bars | What the music does | What the picture does |
|---|---------|-----------:|------|---------------------|-----------------------|
| I | Take-off | 0:00.02 | 0–36 | 3-bar loop groove, right after *"if my take-off fails … tell my mother I'm sorry"* | The disc holds against a fast current. Shot rhythm repeats with the 3-bar loop; every bass glissando at the end of a loop is a camera drop. The world rotates 90°: the same image becomes a lift-off. Night/day alternate on every bar, then a cut on every beat as the disc starts to shudder (*"I can't fight it much longer"*). |
| II | Skip | 1:09.14 | 36–44 | The record gets stuck on a single beat | The picture gets stuck too: one beat of video (24 frames) repeats 32 times, then gets sliced into 2 → 14 vertical time bands, like a skipping disc. |
| III | Ocean | 1:24.50 | 44–57 | Strings glide between chords over the stuck beat | *"An ocean growing inside"*: the disc fills with waves and swells with the measured pitch glide; the current heaves with it. Cuts sit just before each glide. |
| IV | Whirl | 1:49.46 | 57–76 | Long build, ends in a 0.38 s silence | *"A whirlwind that's coming 'round … gonna carry off all that isn't bound"*: a vortex grows from a kink to a full spiral, twisting further on every kick; the disc is torn from its place into orbit. Cuts accelerate 8 → 4 → 2 → 1 beats. In the silence the current is gone: only the disc remains. |
| V | Release | 2:25.94 | 76–88 | The groove slams back | The palette inverts. The disc now travels with the current: the lines stay perfectly straight. Poster-like frames cut on the kicks, the disc is carried across the frame. |
| VI | Tongues | 2:48.99 | 88–112 | Wordless sampler vocals ("speaking in tongues") | Glyphs of an alphabet that does not exist, built from the disc's own geometry: the disc sheds them into its wake, they stand in lines and blocks of writing. The current reads them the only way it can: by flowing around them. Two shots per bar, cut on kick 1 and 3. |
| VII | Chop | 3:35.07 | 112–124 | Gated chords | The self dissolves into the current: the disc is drawn with the current's own lines. While a chord sounds the lines fatten into a solid disc, between the stabs it falls apart into thin stripes (driven by the measured gate). |
| VIII | Ready | 3:58.11 | 124–160 | Outro, vocals from 4:20.25 | *"Must be morning"*: the disc rises as a sun. Then every sung word becomes a body in the current, cut on its LRC timestamp. In the repeat the words turn into ghosts that exist only as the shape of the current. On *"all along"* the video returns to its very first frame — in daylight, and with straight lines. |

Palette: ink `#0A0B12`, paper `#EFE9DD`, vermilion `#FF4A1F`, plus deep navy
for the ocean/whirl and a dusk gradient for the sunrise. Typeface: Archivo
Black (SIL Open Font License).

## Sync

* Beat grid: 125.0 BPM, beat *k* at `0.018 + 0.48002·k` s (linear fit over 627
  tracked beats, residual σ = 5 ms). At 50 fps one beat is exactly 24 frames.
* Section boundaries from self-similarity, lag analysis (3-bar loop vs. 1-beat
  loop) and the measured silences at 145.56, 168.59 and 214.67 s.
* Drum pulses (line thickness, disc swell) use per-section drum patterns with
  per-hit strength measured from the percussive signal.
* String glide = pitch centroid of the C3–D4 register after removing the
  stuck-loop spectrum. Gate = 400–3000 Hz envelope.
* Lyric cards cut exactly on the shifted LRC word times.

## Files

* `lyrics/Let_It_Happen_from_02m40.263s.lrc` — LRC shifted by −160.263 s
  (line and word timestamps). Lines that ended before the cut are removed.
  No line was being sung at the cut point (the last line before it ends at
  02:39.76), so no line had to be moved to 00:00.00.
* `src/shift_lrc.py` — the LRC shift
* `src/cues.py` — audio analysis → per-frame cue arrays
* `src/field.py` — numba kernels: stream function + anti-aliased contour lines
* `src/look.py` — camera, palette, text and glyph bodies (signed distance fields)
* `src/scenes.py` — the one shot renderer every shot uses
* `src/edl.py` — the edit decision list (all 269 shots)
* `src/glyphs.py` — the invented alphabet
* `src/render.py` — parallel renderer + ffmpeg encode

## Rebuild

```sh
pip install numpy scipy librosa soundfile pillow numba
ffmpeg -i song.mp3 -map 0:a -c:a pcm_f32le full.wav
python -c "import soundfile as sf; y,sr=sf.read('full.wav'); sf.write('cut.wav', y[round(160.263*sr):], sr, subtype='FLOAT')"
python src/shift_lrc.py original.lrc lyrics/Let_It_Happen_from_02m40.263s.lrc 307.34
python src/cues.py cut.wav lyrics/Let_It_Happen_from_02m40.263s.lrc cues.npz
python src/render.py video let_it_happen.mp4 --cues cues.npz --lyrics cues_lyrics.json --audio cut.wav
```

The audio itself is not part of this repository.
