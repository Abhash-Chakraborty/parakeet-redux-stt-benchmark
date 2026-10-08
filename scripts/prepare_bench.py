"""Build subsets B (benchmark utterances) and C (noise robustness) from the Open ASR Leaderboard test sets.

B: 10 utterances each from LibriSpeech test-clean, test-other, Earnings-22, AMI and VoxPopuli.
   The source repo is sorted longest-first and sharded, so one shard per subset is read and
   utterances are restricted to 4-25 s (dictation-length) before sampling with seed 42.
C: the 10 test-clean utterances mixed with pink noise at 5 dB SNR (seed 42).

Writes 16 kHz mono float WAVs to data/audio/{B,C}/ and rows to data/manifest_bench.csv.
"""
import io
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import librosa
from huggingface_hub import hf_hub_download

REPO = "hf-audio/esb-datasets-test-only-sorted"
SUBSETS = {  # name -> shards to try in order (repo is sorted longest-first; first shard with enough candidates wins)
    "librispeech_clean": ["librispeech/test.clean-00000-of-00001.parquet"],
    "librispeech_other": ["librispeech/test.other-00000-of-00001.parquet"],
    "earnings22": ["earnings22/test-00002-of-00005.parquet", "earnings22/test-00001-of-00005.parquet"],
    "ami": [f"ami/test-{i:05d}-of-00015.parquet" for i in (4, 3, 2, 1)],
    "voxpopuli": [f"voxpopuli/test-{i:05d}-of-00004.parquet" for i in (2, 1, 0)],
}
N_PER_SUBSET, MIN_S, MAX_S, SEED, SNR_DB = 10, 4.0, 25.0, 42, 5.0
ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def decode(audio_cell):
    data, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data.astype(np.float32)


def pink_noise(n, rng):
    # Voss-McCartney approximation via 1/f filtering in the frequency domain.
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec))
    f[0] = 1
    pink = np.fft.irfft(spec / np.sqrt(f), n)
    return (pink / np.std(pink)).astype(np.float32)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for name, shards in SUBSETS.items():
        for shard in shards:
            path = hf_hub_download(REPO, shard, repo_type="dataset")
            n_ok = sum(MIN_S <= sf.info(io.BytesIO(a["bytes"])).duration <= MAX_S
                       for a in pq.read_table(path, columns=["audio"]).column("audio").to_pylist())
            if n_ok >= 3 * N_PER_SUBSET:
                break
        table = pq.read_table(path)
        cols = table.column_names
        text_col = "text" if "text" in cols else next(c for c in cols if "text" in c or "transcript" in c)
        id_col = "id" if "id" in cols else None
        audios = table.column("audio").to_pylist()
        texts = table.column(text_col).to_pylist()
        ids = table.column(id_col).to_pylist() if id_col else [f"{name}_{i}" for i in range(len(texts))]
        candidates = []
        for i, a in enumerate(audios):
            pcm = decode(a)
            dur = len(pcm) / SR
            if MIN_S <= dur <= MAX_S and texts[i] and texts[i].strip():
                candidates.append((i, pcm, dur))
        pick = rng.choice(len(candidates), size=N_PER_SUBSET, replace=False)
        out_dir = ROOT / "data/audio/B"
        out_dir.mkdir(parents=True, exist_ok=True)
        for j in sorted(pick):
            i, pcm, dur = candidates[j]
            uid = f"B_{name}_{len([r for r in rows if r['group'] == name]):02d}"
            sf.write(out_dir / f"{uid}.wav", pcm, SR, subtype="FLOAT")
            rows.append(dict(id=uid, subset="B", group=name, duration_s=round(dur, 3),
                             source=f"{REPO}/{shard}#{ids[i]}", licence="per source dataset (research use)",
                             audio=f"data/audio/B/{uid}.wav", reference=texts[i].strip()))
            if name == "librispeech_clean":  # subset C: same utterance + pink noise at SNR_DB
                noise = pink_noise(len(pcm), rng)
                p_sig, p_noise = np.mean(pcm ** 2), np.mean(noise ** 2)
                noisy = pcm + noise * np.sqrt(p_sig / (p_noise * 10 ** (SNR_DB / 10)))
                noisy = (noisy / max(1.0, np.abs(noisy).max())).astype(np.float32)
                cid = uid.replace("B_librispeech_clean", "C_noisy5db")
                (ROOT / "data/audio/C").mkdir(parents=True, exist_ok=True)
                sf.write(ROOT / f"data/audio/C/{cid}.wav", noisy, SR, subtype="FLOAT")
                rows.append(dict(id=cid, subset="C", group="pink_noise_5db", duration_s=round(dur, 3),
                                 source=f"{uid} + pink noise {SNR_DB} dB SNR seed {SEED}",
                                 licence="derived", audio=f"data/audio/C/{cid}.wav", reference=texts[i].strip()))
        print(f"{name}: {shard} -> {len(candidates)} candidates in range, picked {N_PER_SUBSET}", flush=True)
    with open(ROOT / "data/manifest_bench.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["duration_s"] for r in rows)
    print(f"wrote {len(rows)} rows, {total / 60:.1f} min audio")


if __name__ == "__main__":
    main()
