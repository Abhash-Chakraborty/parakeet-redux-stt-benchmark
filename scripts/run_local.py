"""Run one local system/device configuration in a fresh worker process while sampling its resources.

    python run_local.py --system tdt_q8 --device gpu --set accuracy
    python run_local.py --system tdt_q8 --device cpu --set speed --threads 6

--set accuracy : every unit in data/units.csv, 1 pass
--set speed    : the speed subset (first 4 clips of each B group), 1 warm-up pass + 3 timed passes
--set determinism : 10 B clips, 1 pass (compared against the accuracy run)

Writes results/raw/<system>/<device>_<set>[_t<threads>].jsonl (worker output) and a matching
*_resources.json with idle baseline, peak RSS, CPU%, and peak VRAM, split into load and inference phases.
Runs with the interpreter of envs/tcpp, or inside Docker for --system redux (deviation D1).
"""
import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_S = 0.05

try:
    import pynvml
    pynvml.nvmlInit()
    GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # no NVML: VRAM is not recorded
    GPU = None


def vram_used_mb():
    return pynvml.nvmlDeviceGetMemoryInfo(GPU).used / 2**20 if GPU else float("nan")


def units_for(which, subsets=None):
    rows = list(csv.DictReader(open(ROOT / "data/units.csv", encoding="utf-8")))
    if which == "accuracy":
        return [r for r in rows if not subsets or r["subset"] in subsets]
    b = [r for r in rows if r["subset"] == "B"]
    groups = sorted({r["group"] for r in b})
    if which == "speed":
        return [r for g in groups for r in [x for x in b if x["group"] == g][:4]]
    if which == "determinism":
        return [r for g in groups for r in [x for x in b if x["group"] == g][:2]]
    raise ValueError(which)


_HANDLES = {}


def tree(proc):
    """Worker process tree, reusing one psutil handle per pid so cpu_percent() accumulates between samples
    (on Windows the venv python.exe is a launcher whose child is the real interpreter)."""
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    out = []
    for p in procs:
        if p.pid not in _HANDLES:
            _HANDLES[p.pid] = p
            try:
                p.cpu_percent(None)
            except psutil.NoSuchProcess:
                continue
        out.append(_HANDLES[p.pid])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--set", choices=["accuracy", "speed", "determinism"], required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--docker", action="store_true", help="run the worker in the stt-photon Linux container")
    ap.add_argument("--subsets", default="", help="accuracy set only: comma list, e.g. D (default: all)")
    a = ap.parse_args()

    subsets = [x for x in a.subsets.split(",") if x]
    units = units_for(a.set, subsets)
    tag = f"{a.device}_{a.set}" + (f"_{''.join(subsets)}" if subsets else "") + (f"_t{a.threads}" if a.set == "speed" else "") + ("_docker" if a.docker else "")
    out_dir = ROOT / "results/raw" / a.system
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_file = out_dir / f"{tag}_clips.txt"
    passes, warmup = (3, len(units) - 1) if a.set == "speed" else (1, 0)

    if a.docker or a.system == "redux":
        clips_file.write_text("".join(f"{u['unit_id']}\t/work/{u['path']}\n" for u in units))
        rel = lambda p: "/work/" + p.relative_to(ROOT).as_posix()
        cmd = ["docker", "run", "--rm", "--name", "stt-worker", f"--cpus={a.threads}",
               "-v", f"{ROOT}:/work", "-v", f"{Path.home() / '.cache/huggingface'}:/root/.cache/huggingface",
               "stt-photon:1", "python", "-u", "/work/scripts/worker.py"]
        worker_paths = (rel(clips_file), rel(out_dir / f"{tag}.jsonl"))
    else:
        clips_file.write_text("".join(f"{u['unit_id']}\t{ROOT / u['path']}\n" for u in units))
        cmd = [str(ROOT / "envs/tcpp/Scripts/python.exe"), "-u", str(ROOT / "scripts/worker.py")]
        worker_paths = (str(clips_file), str(out_dir / f"{tag}.jsonl"))
    cmd += ["--system", a.system, "--device", a.device, "--threads", str(a.threads), "--clips", worker_paths[0],
            "--out", worker_paths[1], "--passes", str(passes), "--warmup", str(warmup)]

    # idle baseline (5 s) before the worker starts
    base_vram = [vram_used_mb() for _ in range(10) if not time.sleep(0.5)]
    base_cpu = psutil.cpu_percent(interval=2.0)

    samples, phase = [], {"name": "load"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    ps = psutil.Process(proc.pid)
    log_lines = []

    def reader():
        for line in proc.stdout:
            log_lines.append(line)
            if line.startswith("READY"):
                phase["name"] = "infer"
            elif line.startswith("DONE"):
                phase["name"] = "done"

    threading.Thread(target=reader, daemon=True).start()
    for p in tree(ps):
        try:
            p.cpu_percent(None)
        except psutil.NoSuchProcess:
            pass
    t_start = time.perf_counter()
    while proc.poll() is None:
        rss, cpu = 0, 0.0
        for p in tree(ps):
            try:
                rss += p.memory_info().rss
                cpu += p.cpu_percent(None)
            except psutil.NoSuchProcess:
                pass
        samples.append((time.perf_counter() - t_start, phase["name"], rss / 2**20, cpu, vram_used_mb(),
                        psutil.cpu_percent(None)))
        time.sleep(SAMPLE_S)
    wall = time.perf_counter() - t_start
    (out_dir / f"{tag}_worker.log").write_text("".join(log_lines), encoding="utf-8")
    if proc.returncode != 0:
        print("".join(log_lines[-30:]))
        raise SystemExit(f"worker failed with code {proc.returncode}")

    def summarize(name):
        s = [x for x in samples if x[1] == name]
        if not s:
            return None
        return {"n_samples": len(s), "peak_rss_mb": max(x[2] for x in s),
                "mean_proc_cpu_pct": sum(x[3] for x in s) / len(s), "peak_proc_cpu_pct": max(x[3] for x in s),
                "mean_system_cpu_pct": sum(x[5] for x in s) / len(s),
                "peak_vram_used_mb": max(x[4] for x in s)}

    vb = min(base_vram)
    res = {"system": a.system, "device": a.device, "set": a.set, "threads": a.threads, "docker": a.docker or
           a.system == "redux", "wall_s": wall, "baseline_vram_used_mb": vb, "baseline_system_cpu_pct": base_cpu,
           "logical_cpus": psutil.cpu_count(), "load": summarize("load"), "infer": summarize("infer")}
    for ph in ("load", "infer"):
        if res[ph]:
            res[ph]["peak_vram_delta_mb"] = res[ph]["peak_vram_used_mb"] - vb
    (out_dir / f"{tag}_resources.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
