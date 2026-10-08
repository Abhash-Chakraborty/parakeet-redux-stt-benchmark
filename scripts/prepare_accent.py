"""Build subset B2 from the Speech Accent Archive (Kaggle: rtatman/speech-accent-archive).

Every speaker reads the same paragraph, so the reference is exact. One speaker per first language below,
chosen at random (seed 42) among speakers whose recording exists; the English control is restricted to
speakers born in the USA.
"""
import csv
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/accent"
OUT = ROOT / "data/audio/B2"
L1S = ["hindi", "bengali", "tamil", "mandarin", "spanish", "arabic", "russian", "japanese", "french", "german",
       "portuguese", "english"]


def main():
    rng = np.random.default_rng(42)
    ref = " ".join((RAW / "reading-passage.txt").read_text(encoding="utf-8").split())
    meta = pd.read_csv(RAW / "speakers_all.csv")
    rec_dir = RAW / "recordings/recordings"
    meta = meta[meta["filename"].map(lambda f: (rec_dir / f"{f}.mp3").exists())]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for l1 in L1S:
        pool = meta[meta["native_language"] == l1]
        if l1 == "english":
            pool = pool[pool["country"] == "usa"]
        if pool.empty:
            print("no speaker for", l1)
            continue
        r = pool.iloc[rng.integers(len(pool))]
        uid = f"B2_{l1}"
        wav = OUT / f"{uid}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(rec_dir / f"{r['filename']}.mp3"),
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(wav)], check=True)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(wav)], capture_output=True, text=True).stdout)
        rows.append(dict(id=uid, subset="B2", group=l1, duration_s=round(dur, 3),
                         source=f"kaggle:rtatman/speech-accent-archive/{r['filename']}.mp3 (born {r['birthplace']}, "
                                f"{r['sex']}, age {r['age']}, English onset {r['age_onset']})",
                         licence="CC BY-NC-SA 2.0 (Speech Accent Archive)", audio=f"data/audio/B2/{uid}.wav",
                         reference=ref))
        print(f"{uid}: {r['filename']} {dur:.1f}s")
    with open(ROOT / "data/manifest_accent.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
