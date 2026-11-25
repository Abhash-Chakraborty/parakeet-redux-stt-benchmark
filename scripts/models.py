"""Registry of the local systems under test (PROTOCOL.md section 3)."""
import glob
import os

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def gguf(repo, filename):
    repo_slug = repo.replace("/", "--") if "/" in repo else f"handy-computer--{repo}"
    hits = glob.glob(f"{HUB}/models--{repo_slug}/snapshots/*/{filename}")
    if not hits:
        raise FileNotFoundError(f"{repo}/{filename} not in HF cache")
    return hits[0]


TCPP = {  # system id -> (repo, file)
    "tdt_f16": ("parakeet-tdt-0.6b-v3-gguf", "parakeet-tdt-0.6b-v3-F16.gguf"),
    "tdt_q8": ("parakeet-tdt-0.6b-v3-gguf", "parakeet-tdt-0.6b-v3-Q8_0.gguf"),
    "unified_q8": ("parakeet-unified-en-0.6b-gguf", "parakeet-unified-en-0.6b-Q8_0.gguf"),
    "wlv3_q8": ("whisper-large-v3-gguf", "whisper-large-v3-Q8_0.gguf"),
    "wturbo_q8": ("whisper-large-v3-turbo-gguf", "whisper-large-v3-turbo-Q8_0.gguf"),
    "moon_f16": ("moonshine-base-gguf", "moonshine-base-F16.gguf"),
    "fourier_q8": ("oruk/oruk-fourier-2b", "Fourier-2B-Q8_0.gguf"),
}

LABELS = {
    "redux": "Parakeet Redux (ternary)",
    "tdt_f16": "Parakeet TDT 0.6B v3 (original, F16)",
    "tdt_q8": "Parakeet TDT 0.6B v3 (Q8_0)",
    "unified_q8": "Parakeet Unified EN 0.6B (Handy, Q8_0)",
    "wlv3_q8": "Whisper large-v3 (Q8_0)",
    "wturbo_q8": "Whisper large-v3-turbo (Q8_0)",
    "moon_f16": "Moonshine base (F16)",
    "fourier_q8": "Oruk Fourier 2B (Q8_0, English-only, CC BY-NC 4.0)",
    "oai": "OpenAI gpt-transcribe",
    "el": "ElevenLabs Scribe v2",
    "vercel": "Vercel AI Gateway STT",
}
