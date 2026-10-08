r"""Record subset D: one WAV per short script (one paragraph per line of data/own_voice/script.txt),
from the default microphone (the same one Handy uses).

Run it yourself in a terminal:  envs\cloud\Scripts\python.exe scripts\record_own_voice.py
Press Enter to start, read the paragraph the way you dictate into Handy, press Enter to stop.
Type r + Enter to redo the previous script.
"""
import textwrap
import queue
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/audio/D"
REC_SR = 48000

sentences = [s.strip() for s in (ROOT / "data/own_voice/script.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
OUT.mkdir(parents=True, exist_ok=True)
dev = sd.query_devices(kind="input")
print(f"Microphone: {dev['name']}\n{len(sentences)} sentences. Speak the way you normally dictate into Handy.\n")


def record():
    q = queue.Queue()
    with sd.InputStream(samplerate=REC_SR, channels=1, dtype="float32", callback=lambda d, *_: q.put(d.copy())):
        input("   recording... press Enter to stop ")
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


i = 0
while i < len(sentences):
    print(f"--- Script {i + 1} of {len(sentences)} " + "-" * 50)
    print(textwrap.fill(sentences[i], width=78, initial_indent="  ", subsequent_indent="  "))
    print("-" * 70)
    cmd = input("   Enter = start, r = redo previous: ").strip().lower()
    if cmd == "r" and i > 0:
        i -= 1
        continue
    audio = record()
    if len(audio) < REC_SR * 3:
        print("   too short, try again\n")
        continue
    pcm16k = librosa.resample(audio, orig_sr=REC_SR, target_sr=16000).astype(np.float32)
    sf.write(OUT / f"D_own_{i:02d}.wav", pcm16k, 16000, subtype="FLOAT")
    peak = float(np.abs(audio).max())
    print(f"   saved {len(pcm16k) / 16000:.1f}s (peak {peak:.2f}{'  <- quiet, move closer?' if peak < 0.05 else ''})\n")
    i += 1
print("All done. The recording is finished.")
