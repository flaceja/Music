"""Shift an .lrc file so that a given song position becomes 00:00.00.

Rules:
- every line and word timestamp is reduced by OFFSET seconds
- lines that end before the cut are dropped
- a line still being sung at the cut point is moved to 00:00.00
- [length:] is updated to the remaining duration
"""
import re
import sys

OFFSET = 160.263
LINE_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
WORD_RE = re.compile(r"<(\d+):(\d+(?:\.\d+)?)>")


def to_sec(m, s):
    return int(m) * 60 + float(s)


def fmt(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 6000:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def shift(src, dst, remaining_s):
    header, lines = [], []
    for raw in open(src, encoding="utf-8").read().splitlines():
        m = LINE_RE.match(raw)
        if m:
            lines.append((to_sec(m.group(1), m.group(2)), m.group(3)))
        elif raw.strip():
            header.append(raw)
    out = []
    for i, (t, text) in enumerate(lines):
        t_next = lines[i + 1][0] if i + 1 < len(lines) else float("inf")
        if t_next <= OFFSET:
            continue  # finished before the cut
        if t < OFFSET:
            if not text.strip():
                continue  # instrumental gap marker, nothing is being sung
            new_t = 0.0  # still being sung at the cut point
        else:
            new_t = t - OFFSET
        text = WORD_RE.sub(lambda w: "<" + fmt(to_sec(w.group(1), w.group(2)) - OFFSET) + ">", text)
        out.append(f"[{fmt(new_t)}]{text}")
    mins, secs = divmod(int(round(remaining_s)), 60)
    header = [f"[length: {mins:02d}:{secs:02d}]" if h.startswith("[length") else h for h in header]
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n\n" + "\n".join(out) + "\n")


if __name__ == "__main__":
    shift(sys.argv[1], sys.argv[2], float(sys.argv[3]))
