# Protocol (pre-registered)

Written 2026-10-05, before any benchmark result was collected. Changes after this point are recorded as deviations in `LOG.md`.

## 1. Questions

1. **RQ1:** Accuracy cost of Parakeet Redux (ternary, 178 MB) vs. its original `parakeet-tdt-0.6b-v3` (F16, 1.26 GB) and a standard 8-bit quantization of it (Q8_0, 740 MB).
2. **RQ2:** Redux vs. the user's current Handy model, `parakeet-unified-en-0.6b` Q8_0.
3. **RQ3:** All local models vs. Whisper large-v3 / large-v3-turbo, Moonshine base, and proprietary cloud models.
4. **RQ4:** On this machine: CPU/GPU compatibility, throughput, latency, RAM, VRAM and CPU load.
5. **RQ5:** Cost and privacy of cloud vs. local.

## 2. Machine

Ryzen 5 5600H (6 cores / 12 threads, AVX2, no AVX-512/AVX-VNNI), 15.3 GB RAM, GTX 1650 4 GB (Vulkan; NVIDIA driver 610.88), Windows 11 Home 10.0.26300. "GPU" always means the GTX 1650 (the Radeon iGPU is excluded).

## 3. Systems under test

| ID | Model | Weights | Runtime | Devices |
|---|---|---|---|---|
| redux | moondream/parakeet-redux | ternary, 178 MB | Photon (moondream 2.6.1) in Linux container (deviation D1) | CPU |
| tdt_f16 | nvidia parakeet-tdt-0.6b-v3 | F16 GGUF | transcribe-cpp 0.3.1 | CPU, GPU |
| tdt_q8 | nvidia parakeet-tdt-0.6b-v3 | Q8_0 GGUF | transcribe-cpp 0.3.1 | CPU, GPU |
| unified_q8 | nvidia parakeet-unified-en-0.6b (Handy's file) | Q8_0 GGUF | transcribe-cpp 0.3.1 | CPU, GPU |
| wlv3_q8 | openai whisper-large-v3 | Q8_0 GGUF | transcribe-cpp 0.3.1 | CPU, GPU |
| wturbo_q8 | openai whisper-large-v3-turbo | Q8_0 GGUF | transcribe-cpp 0.3.1 | CPU, GPU |
| moon_f16 | usefulsensors moonshine-base | F16 GGUF | transcribe-cpp 0.3.1 | CPU, GPU |
| oai | OpenAI `gpt-transcribe` | — | API | cloud |
| el | ElevenLabs `scribe_v2` | — | API | cloud |
| vercel | one STT model via Vercel AI Gateway, if the account has STT beta access | — | API | cloud |

Decoding: each runtime's default greedy decoding; language fixed to English where the API allows; default punctuation and capitalization on; no prompts or vocabulary hints.

## 4. Data

All audio is 16 kHz mono float32 WAV.

| Subset | Content | Size | Reference |
|---|---|---|---|
| A | 6 YouTube clips (3 to 4 min each): TED, conversation, technical lecture, news, Indian-English speaker, noisy setting | ~22 min | Human-made captions in the trimmed window |
| B | 10 utterances each from LibriSpeech test-clean, test-other, Earnings-22, AMI, VoxPopuli (4 to 25 s, seed 42) | 50 clips | Dataset transcripts |
| B2 | Speech Accent Archive (Kaggle): 12 speakers, different first languages, same paragraph | 12 clips | The fixed paragraph |
| C | The 10 test-clean utterances + pink noise at 5 dB SNR | 10 clips | Same as B |
| D | The user's own dictation: 4 short scripts (one paragraph each, ~25 s), Handy's microphone | 4 clips | The script |

**Chunking.** Clips longer than 30 s (subset A) are split once into ≤30 s chunks at silences with Silero VAD. Every system receives the identical chunk files, and the chunk hypotheses are joined per clip before scoring.

**Cloud consent.** Subset D is sent to cloud systems only after explicit user approval.

## 5. Metrics

**Accuracy (primary): normalized WER.** Reference and hypothesis both pass through the Whisper `EnglishTextNormalizer` (the Open ASR Leaderboard convention), then WER is computed with `jiwer`.
- **Corpus WER** is reported per system and per subset, as total errors / total reference words.
- **95% CI** comes from a bootstrap over clips (10,000 resamples, seed 42).
- **RQ1 test:** paired bootstrap of the WER difference redux − tdt_f16 over the same clips. "No meaningful loss" means the upper CI bound is < +0.5 WER points.

**Accuracy (secondary):**
- **Formatted WER:** lowercase only, punctuation kept as separate tokens. This reflects dictation quality.
- **CER**, and the substitution/deletion/insertion breakdown.

**Speed:**
- **RTFx** = audio seconds / processing seconds (higher is faster), measured on the *speed subset*: the first 4 clips of each B group (20 clips, ~4 min). 1 warm-up pass, then 3 timed passes; median reported. (Reduced from all of B before any timed run, after the smoke test showed Whisper large-v3 below real time on CPU.)
- **Cold start:** process start → model loaded → first transcript of a 5 s clip.
- **Warm latency:** per-clip p50 and p95 over the speed subset, in the dictation scenario (one short clip at a time, model already loaded).
- **Cloud:** wall clock from request start to response, including upload.

**Resources:**
- **CPU and RAM:** peak RSS and mean/peak CPU% of the worker process tree, sampled every 50 ms with psutil, minus a 5 s idle baseline.
- **VRAM:** peak dedicated GPU memory via NVML, minus baseline.
- **Disk:** model file size.

**Cost:** list price per audio hour (cloud). **Privacy:** rubric of where audio goes, retention, training use, zero-retention availability, offline capability and licence, with vendor sources cited.

## 6. Controls

1. AC power, Windows power mode "Best performance".
2. Handy, browsers and other heavy apps closed during timed runs. Docker runs only for the redux system and its overhead calibration.
3. Each (system, device) runs in a fresh process.
4. CPU runs use **6 threads** for every system. Redux additionally runs at 4 and 12 threads (the tuning probe).
5. Systems run in randomized order (seed 42) with a 60 s cool-down between configurations.
6. Accuracy comes from one deterministic pass per system, run on GPU where supported. Determinism check: 10 clips rerun on CPU must give identical text; any difference is logged.
7. All package versions, model file SHA-256 and the driver version are recorded in `results/env.json`.

## 7. Validity checks

1. JFK smoke test passes on every system/device before full runs.
2. WER(ref, ref) = 0. Three alignments are inspected by hand.
3. Plausibility: on LibriSpeech test-clean the Parakeet models should land near their published ~1.5 to 2.5% WER.
4. **VM overhead (D1):** tdt_q8 CPU run both natively and inside the Docker container. Redux speed is reported raw and with the measured overhead ratio.
