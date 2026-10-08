"""Build data/manifest_own.csv (subset D) from data/audio/D/D_own_NN.wav and data/own_voice/script.txt."""
import csv
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]

sentences = [s.strip() for s in (ROOT / "data/own_voice/script.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
rows = []
for i, ref in enumerate(sentences):
    wav = ROOT / f"data/audio/D/D_own_{i:02d}.wav"
    if not wav.exists():
        print("missing", wav.name)
        continue
    rows.append(dict(id=f"D_own_{i:02d}", subset="D", group="own_voice", duration_s=round(sf.info(wav).duration, 3),
                     source="user recording, default microphone (Realtek Microphone Array), local only",
                     licence="private: never published; cloud only with consent", audio=f"data/audio/D/{wav.name}",
                     reference=ref))
with open(ROOT / "data/manifest_own.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)}/{len(sentences)} sentences, {sum(r['duration_s'] for r in rows):.0f}s")
