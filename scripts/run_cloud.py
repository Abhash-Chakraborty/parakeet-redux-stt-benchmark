"""Transcribe units with proprietary cloud APIs (PROTOCOL.md section 3).

    python run_cloud.py --system oai --set accuracy --subsets A,B,B2,C
    python run_cloud.py --system el --smoke          # JFK sample only

Every system receives the same bytes: each unit converted to 16-bit PCM WAV, 16 kHz mono, in memory.
Wall time per request covers upload + processing + response (the user-visible latency).
Keys come from .env (OPENAI_KEY, ELEVENLABS_KEY, VERCEL_KEY) and are never printed or logged.
"""
import argparse
import base64
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
import soundfile as sf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
VERCEL_MODEL = "google/gemini-3.5-transcribe"


def wav16(path):
    pcm, sr = sf.read(path, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def call_oai(data):
    r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_KEY']}"},
                      files={"file": ("audio.wav", data, "audio/wav")},
                      data={"model": "gpt-transcribe", "response_format": "json", "language": "en"}, timeout=120)
    r.raise_for_status()
    return r.json()["text"]


def call_el(data):
    r = requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                      headers={"xi-api-key": os.environ["ELEVENLABS_KEY"]},
                      files={"file": ("audio.wav", data, "audio/wav")},
                      data={"model_id": "scribe_v2", "language_code": "en", "tag_audio_events": "false",
                            "diarize": "false", "timestamps_granularity": "none"}, timeout=120)
    r.raise_for_status()
    return r.json()["text"]


def call_vercel(data):
    r = requests.post("https://ai-gateway.vercel.sh/v4/ai/transcription-model",
                      headers={"Authorization": f"Bearer {os.environ['VERCEL_KEY']}",
                               "ai-gateway-protocol-version": "0.0.1",
                               "ai-transcription-model-specification-version": "4",
                               "ai-model-id": VERCEL_MODEL, "Content-Type": "application/json"},
                      json={"audio": base64.b64encode(data).decode(), "mediaType": "audio/wav"}, timeout=180)
    r.raise_for_status()
    return r.json()["text"]


CALLS = {"oai": call_oai, "el": call_el, "vercel": call_vercel}


def transcribe(system, data, retries=4):
    for k in range(retries):
        try:
            t0 = time.perf_counter()
            text = CALLS[system](data)
            return text, time.perf_counter() - t0
        except requests.HTTPError as e:
            code = e.response.status_code
            body = e.response.text[:300]
            if code in (429, 500, 502, 503, 504) and k < retries - 1:
                time.sleep(2 ** (k + 1))
                continue
            raise RuntimeError(f"HTTP {code}: {body}") from None
        except requests.RequestException as e:
            if k < retries - 1:
                time.sleep(2 ** (k + 1))
                continue
            raise RuntimeError(f"{type(e).__name__}") from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(CALLS), required=True)
    ap.add_argument("--set", default="accuracy")
    ap.add_argument("--subsets", default="A,B,B2,C")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="seconds between request starts (Vercel's Gemini STT allows 5 requests/min per team)")
    a = ap.parse_args()

    if a.smoke:
        text, wall = transcribe(a.system, wav16(ROOT / "data/smoke/jfk.wav"))
        print(f"{a.system}: {wall:.2f}s -> {text!r}")
        return
    subsets = a.subsets.split(",")
    units = [u for u in csv.DictReader(open(ROOT / "data/units.csv", encoding="utf-8")) if u["subset"] in subsets]
    out_dir = ROOT / "results/raw" / a.system
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cloud_accuracy_{''.join(subsets)}.jsonl"
    done = set()
    if out_path.exists():  # resume
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8") if '"clip"' in l}
    last_start = -1e9
    with open(out_path, "a", encoding="utf-8") as out:
        for i, u in enumerate(units):
            if u["unit_id"] in done:
                continue
            data = wav16(ROOT / u["path"])
            wait = a.min_interval - (time.perf_counter() - last_start)
            if wait > 0:
                time.sleep(wait)
            last_start = time.perf_counter()
            try:
                text, wall = transcribe(a.system, data)
            except RuntimeError as e:
                print(f"FAIL {u['unit_id']}: {e}", file=sys.stderr, flush=True)
                continue
            out.write(json.dumps({"kind": "clip", "pass": 0, "id": u["unit_id"], "audio_s": float(u["duration_s"]),
                                  "upload_bytes": len(data), "wall_s": wall, "text": text}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"{a.system}: {i + 1}/{len(units)}", flush=True)
    print(f"{a.system}: done -> {out_path.name}")


if __name__ == "__main__":
    main()
