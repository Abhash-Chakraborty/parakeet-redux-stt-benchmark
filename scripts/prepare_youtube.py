"""Build subset A: 6 YouTube clips (~3.5 min each) with human-made English captions as the reference.

For each video: download audio + the manual caption track (never auto-captions), keep the caption cues
that fall fully inside the target window, trim the audio to exactly [first cue start, last cue end],
and clean the cue text (tags, [Music]/(Applause), speaker markers). Raw downloads go to data/raw/youtube/.
"""
import csv
import html
import re
import subprocess
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/youtube"
OUT = ROOT / "data/audio/A"
WINDOW_S = 210  # target clip length

VIDEOS = [  # id, group, manual caption track, window start (s)
    ("HANo__Z8K6s", "ted_clean", "en", 60),
    ("xAt1xcC6qfM", "conversation_podcast", "en-IN", 300),
    ("xAcTmDO6NTI", "technical_lecture", "en-j3PyPqV-e1s", 600),
    ("yP9wGEo3NqI", "news_fast", "en", 60),
    ("kcW4ABcY3zI", "indian_english", "en", 120),
    ("OlyYE6USuO0", "noisy_outdoor", "en-US", 30),
]


def ts(s):
    h, m, rest = ("0:" + s if s.count(":") == 1 else s).split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))


def parse_vtt(path):
    cues, lines = [], path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"([\d:.,]+)\s+-->\s+([\d:.,]+)", lines[i])
        if m:
            start, end, text = ts(m.group(1)), ts(m.group(2)), []
            i += 1
            while i < len(lines) and lines[i].strip():
                text.append(lines[i].strip())
                i += 1
            cues.append((start, end, " ".join(text)))
        i += 1
    return cues


def clean(text):
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪|♪", " ", text)       # [Music], (Applause), lyrics
    text = re.sub(r"(^|\s)(>>|-)\s*", " ", text)                       # speaker-change markers
    text = re.sub(r"(^|\s)[A-Z][A-Z .'-]{1,30}:\s", " ", text)         # "JUDY WOODRUFF: "
    return re.sub(r"\s+", " ", text).strip()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid, group, lang, w0 in VIDEOS:
        opts = {"format": "bestaudio/best", "outtmpl": str(RAW / f"{vid}.%(ext)s"), "writesubtitles": True,
                "writeautomaticsub": False, "subtitleslangs": [lang], "subtitlesformat": "vtt", "quiet": True,
                "noprogress": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
        audio = next(p for p in RAW.glob(f"{vid}.*") if p.suffix not in (".vtt",))
        vtt = RAW / f"{vid}.{lang}.vtt"
        cues = [c for c in parse_vtt(vtt) if c[0] >= w0]
        keep = []
        for c in cues:
            if c[1] > w0 + WINDOW_S and keep:
                break
            keep.append(c)
        start, end = keep[0][0], keep[-1][1]
        reference = clean(" ".join(c[2] for c in keep))
        uid = f"A_{group}"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                        "-i", str(audio), "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(OUT / f"{uid}.wav")],
                       check=True)
        rows.append(dict(id=uid, subset="A", group=group, duration_s=round(end - start, 3),
                         source=f"https://www.youtube.com/watch?v={vid} [{start:.2f}-{end:.2f}s] captions={lang}",
                         licence=f"{info.get('license') or 'YouTube standard licence'}; local research use only, not redistributed",
                         audio=f"data/audio/A/{uid}.wav", reference=reference))
        print(f"{uid}: {end - start:.0f}s, {len(reference.split())} ref words, channel={info.get('channel')}", flush=True)
    with open(ROOT / "data/manifest_youtube.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
