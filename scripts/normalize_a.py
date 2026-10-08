"""Deviation D2: scale each subset-A clip (and its chunks) by ONE linear gain so the peak is 0.99.

Chunk boundaries are kept as they are; only the sample values are scaled. Files already within [-1, 1]
are left unchanged.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
units = pd.read_csv(ROOT / "data/units.csv")
for pid, g in units[units.subset == "A"].groupby("parent_id"):
    peak = max(float(np.abs(sf.read(ROOT / p, dtype="float32")[0]).max()) for p in g.path)
    if peak <= 1.0:
        print(f"{pid}: peak {peak:.3f}, unchanged")
        continue
    gain = 0.99 / peak
    for p in list(g.path) + [f"data/audio/A/{pid}.wav"]:
        x, sr = sf.read(ROOT / p, dtype="float32")
        sf.write(ROOT / p, (x * gain).astype(np.float32), sr, subtype="FLOAT")
    print(f"{pid}: peak {peak:.3f} -> gain {gain:.4f} ({20 * np.log10(gain):+.2f} dB) on {len(g)} chunks + parent")
