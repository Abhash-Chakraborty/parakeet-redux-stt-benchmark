"""Build data/units.csv: the exact audio units every system receives (PROTOCOL.md section 4, Chunking).

Clips <= 30 s are one unit each. Longer clips are cut once into <= 30 s chunks, each cut 10-28 s after the
previous one, at the middle of the longest Silero-VAD silence in that window (fallback: the quietest 200 ms).
No cut ever lands at a fixed time, so words are not split.
"""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps

ROOT = Path(__file__).resolve().parents[1]
SR, MAX_S, MIN_CUT_S, MAX_CUT_S = 16000, 30.0, 10.0, 28.0
MANIFESTS = ["manifest_youtube.csv", "manifest_bench.csv", "manifest_accent.csv", "manifest_own.csv"]


def cut_points(pcm, vad):
    """Split into <= MAX_S chunks. Each cut lies 10-28 s after the previous one: at the middle of the longest
    VAD silence in that window, or, if VAD finds none, at the quietest 200 ms (smoothed RMS) of the window."""
    speech = get_speech_timestamps(pcm, vad, sampling_rate=SR, min_silence_duration_ms=100)
    gaps = [((a["end"] + b["start"]) // 2, b["start"] - a["end"]) for a, b in zip(speech, speech[1:])]
    hop = SR // 100  # 10 ms frames
    rms = np.sqrt(np.convolve(pcm ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    smooth = np.convolve(rms, np.ones(20) / 20, mode="same")  # 200 ms
    cuts, s = [0], 0
    while (len(pcm) - s) / SR > MAX_S:
        lo, hi = s + int(MIN_CUT_S * SR), s + int(MAX_CUT_S * SR)
        inside = [g for g in gaps if lo <= g[0] <= hi]
        if inside:
            cut = max(inside, key=lambda g: g[1])[0]
        else:
            f_lo, f_hi = lo // hop, hi // hop
            cut = (f_lo + int(np.argmin(smooth[f_lo:f_hi]))) * hop
        cuts.append(cut)
        s = cut
    cuts.append(len(pcm))
    return list(zip(cuts, cuts[1:]))


def main():
    vad = load_silero_vad()
    units = []
    for m in MANIFESTS:
        path = ROOT / "data" / m
        if not path.exists():
            print("skip (not built yet):", m)
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            pcm, sr = sf.read(ROOT / r["audio"], dtype="float32")
            if len(pcm) / SR <= MAX_S:
                units.append(dict(unit_id=r["id"], parent_id=r["id"], subset=r["subset"], group=r["group"],
                                  path=r["audio"], duration_s=round(len(pcm) / SR, 3)))
                continue
            out_dir = ROOT / "data/audio" / f"{r['subset']}_chunks"
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, (s, e) in enumerate(cut_points(pcm, vad)):
                uid = f"{r['id']}__c{k:02d}"
                sf.write(out_dir / f"{uid}.wav", pcm[s:e], SR, subtype="FLOAT")
                units.append(dict(unit_id=uid, parent_id=r["id"], subset=r["subset"], group=r["group"],
                                  path=f"data/audio/{r['subset']}_chunks/{uid}.wav", duration_s=round((e - s) / SR, 3)))
    with open(ROOT / "data/units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(units[0].keys()))
        w.writeheader()
        w.writerows(units)
    d = np.array([u["duration_s"] for u in units])
    print(f"{len(units)} units, {d.sum() / 60:.1f} min, max {d.max():.1f}s, by subset:",
          {s: sum(1 for u in units if u["subset"] == s) for s in sorted({u["subset"] for u in units})})


if __name__ == "__main__":
    main()
