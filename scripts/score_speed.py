"""Aggregate speed and resource runs (results/raw/*/*_speed_*.jsonl) into results/metrics_speed.csv.

RTFx per pass = total audio s / total wall s; reported value = median over the 3 timed passes.
Warm latency = per-clip wall time (median over the 3 passes per clip), then p50/p95 across clips.
Cold start = interpreter start -> model loaded -> first transcript of the speed subset's first clip.
RAM/CPU from the worker's self-monitor (works inside Docker too); VRAM from the parent's NVML sampling.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = os.environ.get("STT_RESULTS", "results")  # results root; Kaggle outputs use kaggle/_out


def main():
    rows = []
    for f in sorted((ROOT / RES / "raw").glob("*/*_speed_*.jsonl")):
        if f.stem.endswith("_clips"):
            continue
        recs = [json.loads(line) for line in open(f, encoding="utf-8")]
        load = next(r for r in recs if r["kind"] == "load")
        clips = pd.DataFrame([r for r in recs if r["kind"] == "clip"])
        selfres = next((r for r in recs if r["kind"] == "self_resources"), {})
        res_path = f.with_name(f.stem + "_resources.json")
        parent = json.loads(res_path.read_text()) if res_path.exists() else {}
        per_pass = clips.groupby("pass").apply(lambda g: g.audio_s.sum() / g.wall_s.sum(), include_groups=False)
        per_clip = clips.groupby("id").wall_s.median()
        audio = clips[clips["pass"] == 0].set_index("id").audio_s
        infer = selfres.get("infer", {})
        rows.append(dict(
            system=f.parent.name, config=f.stem, device=load["device"], threads=load["threads"],
            docker="docker" in f.stem,
            rtfx=float(np.median(per_pass)), rtfx_min=float(per_pass.min()), rtfx_max=float(per_pass.max()),
            latency_p50_s=float(per_clip.quantile(0.5)), latency_p95_s=float(per_clip.quantile(0.95)),
            latency_per_audio_s=float((per_clip / audio).median()),
            import_s=load["import_s"], load_s=load["load_s"], first_clip_audio_s=load["first_clip_audio_s"],
            first_clip_s=load["first_clip_s"], cold_start_s=load["import_s"] + load["load_s"] + load["first_clip_s"],
            peak_rss_mb=infer.get("peak_rss_mb"), mean_cpu_pct=infer.get("mean_cpu_pct"),
            peak_vram_delta_mb=(parent.get("infer") or {}).get("peak_vram_delta_mb", infer.get("peak_vram_delta_mb")),
            subset_audio_s=float(audio.sum()), n_clips=len(audio)))
    df = pd.DataFrame(rows).sort_values(["device", "rtfx"], ascending=[True, False])
    df.to_csv(ROOT / RES / "metrics_speed.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[["system", "config", "rtfx", "latency_p50_s", "latency_p95_s", "cold_start_s", "peak_rss_mb",
                  "mean_cpu_pct", "peak_vram_delta_mb"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
