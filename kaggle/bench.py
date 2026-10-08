"""Kaggle worker: one system, one device, one set, in a fresh process. Same JSONL format as scripts/worker.py.

    python bench.py --backend tcpp --system tdt_q8 --device gpu --set speed --threads 4 --data DIR --out FILE

Backends:
  tcpp    transcribe-cpp 0.3.1 GGUF (same engine and files as the laptop run); gpu = Vulkan if available
  photon  Parakeet Redux via moondream Photon; device cpu or cuda
  fw      faster-whisper (CTranslate2), Whisper large-v3 / large-v3-turbo, CUDA float16
  nemo    NVIDIA NeMo, Parakeet TDT 0.6B v3 / Parakeet Unified EN 0.6B, original weights on CUDA
  hf      Hugging Face transformers, Moonshine base on CUDA float16

Decoding is greedy everywhere, English where the API allows, no prompts.
"""
import argparse
import csv
import json
import threading
import time
from pathlib import Path

T_PROCESS_START = time.perf_counter()

import numpy as np
import psutil
import soundfile as sf

GGUF = {
    "tdt_f16": ("handy-computer/parakeet-tdt-0.6b-v3-gguf", "parakeet-tdt-0.6b-v3-F16.gguf"),
    "tdt_q8": ("handy-computer/parakeet-tdt-0.6b-v3-gguf", "parakeet-tdt-0.6b-v3-Q8_0.gguf"),
    "unified_q8": ("handy-computer/parakeet-unified-en-0.6b-gguf", "parakeet-unified-en-0.6b-Q8_0.gguf"),
    "wlv3_q8": ("handy-computer/whisper-large-v3-gguf", "whisper-large-v3-Q8_0.gguf"),
    "wturbo_q8": ("handy-computer/whisper-large-v3-turbo-gguf", "whisper-large-v3-turbo-Q8_0.gguf"),
    "moon_f16": ("handy-computer/moonshine-base-gguf", "moonshine-base-F16.gguf"),
}
NEMO = {"tdt_nemo": "nvidia/parakeet-tdt-0.6b-v3", "unified_nemo": "nvidia/parakeet-unified-en-0.6b"}
FW = {"wlv3_fw": "large-v3", "wturbo_fw": "large-v3-turbo"}


def load_pcm(path):
    pcm, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and pcm.ndim == 1, path
    return pcm


class Monitor:
    """Per-phase peak RSS / mean CPU% of this process, plus peak GPU memory above the idle baseline (NVML)."""

    def __init__(self):
        self.p, self.phase, self.stats = psutil.Process(), "load", {}
        self.p.cpu_percent(None)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nv = pynvml
            self.base = self.gpu_used()
        except Exception:
            self.h, self.base = None, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def gpu_used(self):
        return self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**20 if self.h else 0.0

    def _run(self):
        while True:
            s = self.stats.setdefault(self.phase, {"n": 0, "peak_rss_mb": 0.0, "cpu_sum": 0.0, "peak_gpu_mb": 0.0})
            s["n"] += 1
            s["peak_rss_mb"] = max(s["peak_rss_mb"], self.p.memory_info().rss / 2**20)
            s["cpu_sum"] += self.p.cpu_percent(None)
            s["peak_gpu_mb"] = max(s["peak_gpu_mb"], self.gpu_used() - self.base)
            time.sleep(0.05)

    def report(self):
        return {ph: {"peak_rss_mb": s["peak_rss_mb"], "mean_cpu_pct": s["cpu_sum"] / max(1, s["n"]),
                     "peak_vram_delta_mb": s["peak_gpu_mb"], "n": s["n"]} for ph, s in self.stats.items()}


class Tcpp:
    def __init__(self, system, device, threads):
        import transcribe_cpp as tc
        from huggingface_hub import hf_hub_download

        devs = {d.device_type: d for d in tc.backends()}
        if device == "gpu" and "gpu" not in devs:
            raise RuntimeError(f"no Vulkan GPU device; backends: {[str(d) for d in tc.backends()]}")
        self.model = tc.Model(hf_hub_download(*GGUF[system]), device=devs["gpu" if device == "gpu" else "cpu"])
        self.session = self.model.session(n_threads=threads)
        self.whisper = system.startswith("w")

    def transcribe(self, path):
        pcm = load_pcm(path)
        return (self.session.run(pcm, language="en") if self.whisper else self.session.run(pcm)).text


class Photon:
    def __init__(self, system, device, threads):
        import moondream as md

        kw = {"cpu_threads": threads} if device == "cpu" else {}
        self.sp = md.photon("moondream/parakeet-redux", device="cuda" if device == "gpu" else "cpu", **kw).__enter__()

    def transcribe(self, path):
        return self.sp.transcribe(audio=str(path))["text"]


class FasterWhisper:
    def __init__(self, system, device, threads):
        from faster_whisper import WhisperModel

        self.m = WhisperModel(FW[system], device="cuda", compute_type="float16")

    def transcribe(self, path):
        # Pass decoded PCM: the PyAV build on Kaggle rejects faster-whisper's open(..., metadata_errors=...).
        segs, _ = self.m.transcribe(load_pcm(path), language="en", beam_size=1, vad_filter=False,
                                    condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs)


class Nemo:
    def __init__(self, system, device, threads):
        import logging
        import nemo.collections.asr as nemo_asr

        logging.getLogger("nemo_logger").setLevel(logging.ERROR)
        self.m = nemo_asr.models.ASRModel.from_pretrained(NEMO[system]).to("cuda").eval()
        if self.m.cfg.get("validation_ds") is None:  # parakeet-unified ships without it; transcribe() reads it
            from omegaconf import OmegaConf, open_dict
            with open_dict(self.m.cfg):
                self.m.cfg.validation_ds = OmegaConf.create({})

    def transcribe(self, path):
        out = self.m.transcribe([str(path)], batch_size=1, verbose=False)
        o = out[0] if isinstance(out, list) else out
        o = o[0] if isinstance(o, list) else o
        return o.text if hasattr(o, "text") else str(o)


class HfMoonshine:
    def __init__(self, system, device, threads):
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
        self.m = MoonshineForConditionalGeneration.from_pretrained(
            "UsefulSensors/moonshine-base", torch_dtype=torch.float16).to("cuda").eval()

    def transcribe(self, path):
        pcm = load_pcm(path)
        inp = self.proc(pcm, sampling_rate=16000, return_tensors="pt").to("cuda")
        inp["input_values"] = inp["input_values"].to(self.torch.float16)
        limit = int(len(pcm) / 16000 * 6.5) + 10  # model card's token budget: ~6.5 tokens per second of audio
        with self.torch.inference_mode():
            ids = self.m.generate(**inp, max_length=limit)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        return self.proc.decode(ids[0], skip_special_tokens=True)


BACKENDS = {"tcpp": Tcpp, "photon": Photon, "fw": FasterWhisper, "nemo": Nemo, "hf": HfMoonshine}


def units_for(data, which):
    rows = list(csv.DictReader(open(Path(data) / "units.csv", encoding="utf-8")))
    if which == "accuracy":
        return rows
    b = [r for r in rows if r["subset"] == "B"]
    return [r for g in sorted({r["group"] for r in b}) for r in [x for x in b if x["group"] == g][:4]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--set", choices=["accuracy", "speed"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    units = units_for(a.data, a.set)
    clips = [(u["unit_id"], str(Path(a.data) / u["path"]), float(u["duration_s"])) for u in units]
    passes, warmup = (3, len(clips) - 1) if a.set == "speed" else (1, 0)

    mon = Monitor()
    t0 = time.perf_counter()
    sysobj = BACKENDS[a.backend](a.system, a.device, a.threads)
    t_loaded = time.perf_counter()
    cid, p, dur = clips[0]
    first_text = sysobj.transcribe(p)
    t_first = time.perf_counter()
    mon.phase = "infer"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"kind": "load", "backend": a.backend, "system": a.system, "device": a.device,
                              "threads": a.threads, "import_s": t0 - T_PROCESS_START, "load_s": t_loaded - t0,
                              "first_clip": cid, "first_clip_audio_s": dur, "first_clip_s": t_first - t_loaded,
                              "first_text": first_text}) + "\n")
        for cid, p, dur in clips[1:1 + warmup]:
            sysobj.transcribe(p)
        for k in range(passes):
            for cid, p, dur in clips:
                s = time.perf_counter()
                text = sysobj.transcribe(p)
                out.write(json.dumps({"kind": "clip", "pass": k, "id": cid, "audio_s": dur,
                                      "wall_s": time.perf_counter() - s, "text": text}, ensure_ascii=False) + "\n")
                out.flush()
        out.write(json.dumps({"kind": "self_resources", **mon.report()}) + "\n")
    print("DONE", a.backend, a.system, a.device, a.set, flush=True)


if __name__ == "__main__":
    main()
