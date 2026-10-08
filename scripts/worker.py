"""Worker: load ONE system on ONE device, transcribe a list of clips, write JSONL. Run in a fresh process.

    python worker.py --system tdt_q8 --device gpu --threads 6 --clips clips.txt --out out.jsonl \
                     --passes 1 [--warmup N]

Each output line is one (pass, clip) with text, wall time and audio duration. A first line of kind
"load" records the cold-start timings. Stdout prints "READY" after loading and "DONE" at the end, which
lets the monitor separate load-phase from inference-phase resource use.
"""
import argparse
import json
import sys
import time
from pathlib import Path

T_PROCESS_START = time.perf_counter()

import threading

import numpy as np
import psutil
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))


class SelfMonitor:
    """Samples this process's RSS and CPU% every 50 ms, per phase (works natively and inside Docker)."""

    def __init__(self):
        self.p, self.phase, self.stats = psutil.Process(), "load", {}
        self.p.cpu_percent(None)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            rss, cpu = self.p.memory_info().rss / 2**20, self.p.cpu_percent(None)
            s = self.stats.setdefault(self.phase, {"n": 0, "peak_rss_mb": 0.0, "cpu_sum": 0.0, "peak_cpu_pct": 0.0})
            s["n"] += 1
            s["peak_rss_mb"] = max(s["peak_rss_mb"], rss)
            s["cpu_sum"] += cpu
            s["peak_cpu_pct"] = max(s["peak_cpu_pct"], cpu)
            time.sleep(0.05)

    def report(self):
        return {ph: {"peak_rss_mb": s["peak_rss_mb"], "mean_cpu_pct": s["cpu_sum"] / max(1, s["n"]),
                     "peak_cpu_pct": s["peak_cpu_pct"], "n": s["n"]} for ph, s in self.stats.items()}


def load_pcm(path):
    pcm, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and pcm.ndim == 1, path
    return pcm


class TcppSystem:
    def __init__(self, system, device, threads):
        import transcribe_cpp as tc
        from models import TCPP, gguf

        devs = {d.device_type: d for d in tc.backends()}
        self.model = tc.Model(gguf(*TCPP[system]), device=devs["gpu" if device == "gpu" else "cpu"])
        self.session = self.model.session(n_threads=threads)
        self.whisper = system.startswith("w")

    def transcribe(self, path):
        r = self.session.run(load_pcm(path), language="en") if self.whisper else self.session.run(load_pcm(path))
        return r.text


class PhotonSystem:
    def __init__(self, system, device, threads):
        import moondream as md

        # cpu_threads sizes both torch and Photon's native kernel pool. Without it Photon uses
        # min(physical cores, 4) for ternary models, so threads=4 equals its out-of-the-box setting.
        self._ctx = md.photon("moondream/parakeet-redux", device=device, cpu_threads=threads)
        self.sp = self._ctx.__enter__()

    def transcribe(self, path):
        return self.sp.transcribe(audio=str(path))["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--clips", required=True, help="text file: one 'id<TAB>path' per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0, help="clips transcribed (and discarded) before timing")
    a = ap.parse_args()

    clips = [line.split("\t") for line in Path(a.clips).read_text().splitlines() if line.strip()]
    durations = {cid: sf.info(p).duration for cid, p in clips}

    mon = SelfMonitor()
    t0 = time.perf_counter()
    cls = PhotonSystem if a.system == "redux" else TcppSystem
    sysobj = cls(a.system, a.device, a.threads)
    t_loaded = time.perf_counter()
    cid, p = clips[0]
    first_text = sysobj.transcribe(p)
    t_first = time.perf_counter()
    print("READY", flush=True)
    mon.phase = "infer"

    with open(a.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"kind": "load", "system": a.system, "device": a.device, "threads": a.threads,
                              "import_s": t0 - T_PROCESS_START, "load_s": t_loaded - t0,
                              "first_clip": cid, "first_clip_audio_s": durations[cid],
                              "first_clip_s": t_first - t_loaded, "first_text": first_text}) + "\n")
        for cid, p in clips[1:1 + a.warmup]:
            sysobj.transcribe(p)
        for k in range(a.passes):
            for cid, p in clips:
                s = time.perf_counter()
                text = sysobj.transcribe(p)
                e = time.perf_counter() - s
                out.write(json.dumps({"kind": "clip", "pass": k, "id": cid, "audio_s": durations[cid],
                                      "wall_s": e, "text": text}, ensure_ascii=False) + "\n")
                out.flush()
        out.write(json.dumps({"kind": "self_resources", **mon.report()}) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
