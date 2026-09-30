#!/bin/sh
# Mux rendered PNG frames with the song into an MP4 (H.264 + AAC).
#   make_video.sh frames_dir song.mp3 out.mp4 [fps]
# Frame 1 of the render is t = 0 of the song, so no offset is needed.
set -e
FRAMES=$1; AUDIO=$2; OUT=$3; FPS=${4:-30}
ffmpeg -y -framerate "$FPS" -start_number 1 -i "$FRAMES/frame_%05d.png" -i "$AUDIO" \
  -map 0:v -map 1:a -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 256k -shortest "$OUT"
