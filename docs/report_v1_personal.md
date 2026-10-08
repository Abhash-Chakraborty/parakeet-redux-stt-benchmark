# Speech-to-Text Benchmark: Parakeet Redux vs Local & Cloud Models

Oct 5, 2026 · Abhash Chakraborty

## Executive summary

Keep Handy on Parakeet Unified EN 0.6B on the GPU. On this laptop it matches the best cloud models on accuracy, transcribes a sentence in about 0.2 s, and never sends audio off the machine. Parakeet Redux is not an upgrade here: it is the fastest accurate model on a CPU, but it makes about 17% more errors than its original, more on accented speech, and does not run natively on this Windows PC.

| Question | Answer |
| --- | --- |
| Most accurate overall | Cloud and Handy's model are tied: ElevenLabs 4.03%, OpenAI 4.09%, Parakeet Unified 4.16% WER (78 public clips; differences within noise) |
| On the user's own voice | Parakeet Unified 2.7%, the cloud models 2.2 to 3.1%, Redux 4.9% |
| Fastest on this GPU | Parakeet models at 37 to 39x real time, about 0.2 s per clip; Moonshine 66x but much less accurate |
| Fastest accurate on CPU | Parakeet Redux at 15x real time (Linux container), vs 9x for 8-bit Parakeet |
| Redux vs its original | +0.8 WER points (95% CI -0.1 to +1.5), 7x smaller, 2x faster on CPU, no GPU support |
| Whisper large-v3 | No more accurate than Parakeet here, 8x slower on the GPU, slower than real time on the CPU |
| Cost and privacy | Local: free, audio stays on the PC. Cloud: $0.22 to $0.37 per audio hour; ElevenLabs keeps audio by default |

What to use, by situation:

- **A Windows PC with any recent GPU (this laptop):** Parakeet Unified or Parakeet TDT v3 (8-bit) through Handy. There is no accuracy or speed reason to change.
- **A CPU-only machine:** 8-bit Parakeet (9x real time, 0.8 s per sentence) is accurate and fast enough for dictation. Redux is the choice only if speed matters more than about 1 point of accuracy, and it needs Linux or macOS on CPUs like this one (its Windows build had no AVX2 kernel; newer CPUs with AVX-VNNI were not tested).
- **Cloud APIs:** worth it only for features the local models lack, such as many languages, speaker labels or the best punctuation (OpenAI). They are not more accurate for English dictation on this test.
- **Skip:** Whisper large-v3 on this hardware, and Moonshine for anything beyond quick notes.

## Objective and research questions

The experiment asks which speech-to-text model to use today on a mid-range Windows laptop, and what [Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) (178 MB, ternary weights) gives up versus the model it was compressed from.

1. How accurate is Parakeet Redux compared with its original, `parakeet-tdt-0.6b-v3` (about 1.2 GB)?
2. How does it compare with the model currently used in Handy, `parakeet-unified-en-0.6b` (Q8\_0)?
3. How do these compare with Whisper large-v3, other open models and premium cloud models (OpenAI, ElevenLabs)?
4. On this machine, which models run on CPU, which on GPU, and how fast, with what RAM, VRAM and CPU load?
5. What do cloud models cost, and what privacy trade-off do they carry versus fully local models?

Fairness rule: every open-weights model runs locally on the same machine with the same audio and scoring. Only proprietary models are called through their APIs.

## Test machine and controls

All local models ran on one mid-range Windows laptop, under the same conditions.

| Component | Spec |
| --- | --- |
| CPU | AMD Ryzen 5 5600H, 6 cores / 12 threads, AVX2 (no AVX-512, no AVX-VNNI) |
| RAM | 15.3 GB |
| GPU | NVIDIA GTX 1650, 4 GB, Turing; used through Vulkan (driver 610.88) |
| OS | Windows 11 Home 10.0.26300 |
| Runtimes | transcribe-cpp 0.3.1 (same engine as Handy); Photon / moondream 2.6.1 in a Linux container for Redux |

Controls:

- AC power, Windows "Best performance" mode; Handy, browsers and other heavy apps closed for timed runs.
- Each model and device runs in a fresh process; model order randomized (seed 42); 60 s cool-down between configurations.
- Every CPU run is pinned to 6 threads. Redux is also run at 4 and 12 threads.
- Speed is measured as 1 warm-up pass, then 3 timed passes, and the median is reported. Accuracy comes from one deterministic pass.
- Package versions, model checksums and the driver version are recorded in `results/env.json`.

## Models under test

Ten systems were tested: seven open-weights configurations run locally, and up to three proprietary cloud APIs.

| Model | Type | Weights on disk | Runtime | Licence |
| --- | --- | --- | --- | --- |
| [Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) | open | 178 MB (ternary) | Photon, Linux container, CPU | CC-BY-4.0 |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (original) | open | 1,256 MB (F16) | transcribe-cpp | CC-BY-4.0 |
| Parakeet TDT 0.6B v3, 8-bit | open | 740 MB (Q8\_0) | transcribe-cpp | CC-BY-4.0 |
| [Parakeet Unified EN 0.6B](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) (Handy's model) | open | 731 MB (Q8\_0) | transcribe-cpp | NVIDIA Open Model Licence |
| Whisper large-v3 | open | 1,669 MB (Q8\_0) | transcribe-cpp | MIT |
| Whisper large-v3-turbo | open | 886 MB (Q8\_0) | transcribe-cpp | MIT |
| Moonshine base | open | 132 MB (F16) | transcribe-cpp | MIT |
| OpenAI `gpt-transcribe` | proprietary | n/a | API | commercial |
| ElevenLabs `scribe_v2` | proprietary | n/a | API | commercial |
| Vercel AI Gateway STT model | proprietary | n/a | API (only if the account has STT beta access) | commercial |

All GGUF files are the [handy-computer](https://huggingface.co/handy-computer) conversions, so every open model except Redux shares the engine Handy uses. Redux exists only for Moondream's Photon engine. The two Parakeet TDT rows are its parent at original precision and at standard 8-bit, which isolates what the ternary compression changes.

## Test data

About 36 minutes of English audio in five subsets, each with a human-written reference transcript.

| Subset | Content | Clips | Audio | Reference |
| --- | --- | --- | --- | --- |
| A: real-world | 6 YouTube clips: TED (CRISPR), Bill Gates podcast, MIT lecture, PBS NewsHour, Shashi Tharoor TEDx, street interviews | 6 | 20.7 min | Human-made caption track (auto-captions excluded) |
| B: benchmark | 10 utterances each from LibriSpeech test-clean, test-other, Earnings-22, AMI, VoxPopuli; 4 to 25 s; seed 42 | 50 | 6.3 min | Dataset transcripts ([Open ASR Leaderboard sets](https://huggingface.co/datasets/hf-audio/esb-datasets-test-only-sorted)) |
| B2: accents | Speech Accent Archive (Kaggle): 12 first languages incl. Hindi, Bengali, Tamil, plus a US English control | 12 | 4.9 min | The fixed reading passage |
| C: noise | The 10 test-clean clips with pink noise at 5 dB SNR | 10 | 1.2 min | Same as B |
| D: own voice | The user dictating 4 short paragraph scripts (email, tech note, numbers and places, casual) into Handy's microphone | 4 | \~2 min | The script |

Clips over 30 s are cut once into 7 to 28 s chunks at speech pauses. Every model receives the identical chunk files, and chunk outputs are joined per clip before scoring. Subset D is sent to cloud models only with the user's consent.

## Metrics and how they were measured

Word error rate (WER) is the primary accuracy metric. Real-time factor (RTFx) is the primary speed metric.

| Metric | Definition | How it was measured |
| --- | --- | --- |
| WER (normalized) | (substitutions + deletions + insertions) / reference words | Both texts pass the Whisper `EnglishTextNormalizer` (the Open ASR Leaderboard convention), then `jiwer`. Corpus-level, with a 95% bootstrap CI over clips (10,000 resamples) |
| Formatted WER | WER with punctuation kept as tokens, lowercase | Only on subsets with punctuated references (A, B2, D); reflects how clean dictated text looks |
| CER | Character error rate on normalized text | `jiwer` |
| Redux vs original | Difference in WER on identical clips | Paired bootstrap; "no meaningful loss" = upper 95% bound below +0.5 WER points |
| RTFx | Audio seconds / processing seconds (higher is faster) | Speed subset: 20 clips (\~4 min); 1 warm-up pass, median of 3 timed passes |
| Cold start | Process start to first transcript of a short clip | Includes model load and GPU shader compilation |
| Warm latency | Per-clip time, p50 and p95 | Model already loaded; the dictation scenario |
| RAM and CPU | Peak resident memory, mean CPU% of the worker | psutil every 50 ms, inside the worker and from the parent (they agree within 2%) |
| VRAM | Peak GPU memory above idle baseline | NVML every 50 ms |
| Cost and privacy | Price per audio hour; where audio goes and how long it is kept | Vendor pricing and data-policy pages |

## Compatibility on this PC

Every open model runs on both CPU and GPU here except Parakeet Redux, which runs only on CPU and only under Linux (a Docker container on this Windows machine).

| Model | CPU (native Windows) | GPU (GTX 1650) | Notes |
| --- | --- | --- | --- |
| Parakeet Redux | No (runs in Linux container) | No | Photon's Windows build ships no AVX2 kernel; it needs AVX-512-VNNI or AVX-VNNI, which this Ryzen lacks. In Linux the AVX2 path works. Photon's CUDA path needs an Ampere-or-newer GPU. |
| Parakeet TDT v3 (F16, 8-bit) | Yes | Yes (Vulkan) |  |
| Parakeet Unified (Handy) | Yes | Yes (Vulkan) | Same setup Handy uses |
| Whisper large-v3 / turbo | Yes, but slower than real time | Yes (Vulkan) |  |
| Moonshine base | Yes | Yes (Vulkan) |  |
| Cloud models | n/a | n/a | Need internet; each recording leaves the PC |

CPU and GPU produce word-for-word identical transcripts for every local model, so the choice of device affects speed only, never accuracy.

## Accuracy results

The Parakeet Unified model Handy already uses (4.16% WER) is statistically tied with the best cloud models (4.03 to 4.38%). Parakeet Redux (5.33%) is the least accurate model except Moonshine.

![Word error rate by system with 95% confidence intervals](results/figures/wer_overall.png)

*Source: results/metrics_accuracy.csv · subsets A, B, B2, C · 10 systems, identical audio units*

The confidence intervals overlap heavily: on 78 clips, differences under about 1 point are not reliable on their own. The paired tests in the Redux section compare systems clip by clip, which is more sensitive.

| System | YouTube (A) | Benchmarks (B) | Accents (B2) | Noise 5 dB (C) | Formatted WER |
| --- | --- | --- | --- | --- | --- |
| ElevenLabs Scribe v2 | 4.84% | 3.99% | 1.81% | 0.73% | 16.0% |
| OpenAI gpt-transcribe | 4.35% | 5.12% | 2.42% | 2.18% | 12.2% |
| Parakeet Unified (Handy) | 5.01% | 3.99% | 1.81% | 1.09% | 14.1% |
| Whisper large-v3-turbo | 4.72% | 6.24% | 1.33% | 1.82% | 12.3% |
| Gemini 3.5 Transcribe | 4.96% | 5.53% | 1.93% | 0.36% | 13.0% |
| Parakeet TDT v3, 8-bit | 5.48% | 4.20% | 2.17% | 0.73% | 13.3% |
| Parakeet TDT v3, original | 5.51% | 4.20% | 2.17% | 0.73% | 13.3% |
| Whisper large-v3 | 6.46% | 4.50% | 0.85% | 1.82% | 13.9% |
| Parakeet Redux | 6.29% | 4.40% | 3.62% | 1.82% | 14.5% |
| Moonshine base | 6.95% | 9.01% | 5.31% | 5.82% | 16.1% |

- **Formatted WER** keeps punctuation and is scored on A and B2 only, the subsets with punctuated references. OpenAI and Whisper-turbo produce the cleanest punctuation (about 12%); ElevenLabs and Moonshine the least (16%).
- **Accents (B2)** is one passage read by 12 speakers, so it is small. Redux has the highest error rate there among the stronger models (3.62% vs 2.17% for its original).
- **Whisper large-v3** writes clean, non-verbatim text: it drops repeated fillers such as "you know", which the verbatim podcast captions count as deletions. Its A score is pessimistic for dictation use.
- **CPU and GPU give identical text.** A 10-clip rerun on CPU matched the GPU output word for word for every local model, so these accuracy figures hold for both devices.

### On the user's own voice

On four dictated paragraphs (131 s, 225 words, Handy's microphone), Handy's current model scores 2.7% WER against what was actually said. That ties it with ElevenLabs, Gemini and OpenAI; Redux scores 4.9%.

| System | WER vs what was said | WER vs the script | Formatted WER |
| --- | --- | --- | --- |
| Whisper large-v3 | 1.3% | 7.9% | 7.8% |
| ElevenLabs Scribe v2 | 2.2% | 8.8% | 8.6% |
| Parakeet Unified (Handy) | 2.7% | 9.3% | 9.4% |
| Gemini 3.5 Transcribe | 2.7% | 8.4% | 8.6% |
| Parakeet TDT v3, 8-bit | 2.7% | 9.3% | 10.2% |
| OpenAI gpt-transcribe | 3.1% | 8.8% | 6.7% |
| Parakeet TDT v3, original | 3.1% | 9.7% | 10.6% |
| Whisper large-v3-turbo | 3.1% | 9.7% | 10.2% |
| Parakeet Redux | 4.9% | 11.5% | 12.2% |
| Moonshine base | 5.8% | 12.3% | 13.3% |

- **Two references.** The speaker departed from the script in places ("looks good" for "looks great", "I'll be joining", "Thank you"). Where at least 5 of the 6 strong local models agreed on the same everyday-word change, it was taken as what was said. Names and rare words (Ananya, Parakeet) keep the script spelling, because recognizing them is part of the test. Time formats (3.30 vs 3:30) and "11 labs" vs "ElevenLabs" count as equal.
- **Redux's errors here are accent-sensitive:** "flight lines" for "lands", "revenue grow" for "grew". The same pattern shows on the accent set.
- **Every model misheard "Ananya" as "Aranya"**, and most heard "Parakeet" as "Paratic". Rare names are the main remaining weakness for all of them.
- **Small sample:** 225 words, so 1 error = 0.44 points. Treat differences under about 1.5 points as ties.

## Speed, latency and resources

On this laptop the fastest accurate setup is Parakeet on the GPU: 37 to 39 times faster than real time, about 0.2 s per dictation-length clip. Redux is the fastest accurate model on the CPU at 15x, and Whisper is unusable on the CPU.

![Real-time factor by model and device](results/figures/rtfx_by_device.png)

*Source: results/metrics_speed.csv · speed subset: 20 clips, 4.1 min · 1 warm-up + 3 timed passes, run-to-run spread under 1%*

| Configuration | Latency per clip, p50 / p95 | Cold start | Peak RAM | Extra GPU memory |
| --- | --- | --- | --- | --- |
| Parakeet Unified (Handy), GPU | 0.20 / 0.38 s | 2.1 s | 225 MB | 772 MB |
| Parakeet TDT original, GPU | 0.21 / 0.34 s | 3.0 s | 228 MB | 1,271 MB |
| Moonshine base, GPU | 0.10 / 0.24 s | 1.5 s | 154 MB | 181 MB |
| Whisper large-v3-turbo, GPU | 1.45 / 1.52 s | 3.6 s | 275 MB | 956 MB |
| Whisper large-v3, GPU | 1.81 / 2.11 s | 5.1 s | 258 MB | 1,973 MB |
| Parakeet Redux, CPU | 0.48 / 1.03 s | 9.4 s | 982 MB | none |
| Parakeet Unified (Handy), CPU | 0.81 / 1.73 s | 2.7 s | 1,188 MB | none |
| Parakeet TDT original, CPU | 1.00 / 2.12 s | 3.7 s | 1,724 MB | none |
| Moonshine base, CPU | 0.30 / 0.90 s | 1.4 s | 266 MB | none |
| Whisper large-v3, CPU | 13.75 / 15.30 s | 16.6 s | 2,062 MB | none |

Latency is per clip of 4 to 25 s, the dictation case, with the model already loaded. Cold start runs from process launch to the first transcript.

- **Whisper's latency does not shrink for short clips.** It always processes a 30-second window, so on the CPU a 5-second sentence takes 11 to 14 s.
- **Redux's small file does not mean small memory.** It needs 950 to 980 MB of RAM, only about 20% less than 8-bit Parakeet, because its PyTorch-based runtime is heavy. It also has the slowest cold start (8 to 9 s).
- **Redux thread tuning:** 4 threads (its default) gives 14.6x, 6 threads 15.1x, 12 threads 13.3x. Using the CPU's hyper-threads makes it slower.
- **The container does not flatter Redux.** The same 8-bit Parakeet model ran 16% faster in the Linux container (10.7x) than natively (9.3x). Like for like, Redux is 1.4x faster than 8-bit Parakeet on the CPU.
- **First GPU run ever:** the very first Vulkan call on a fresh install compiles shaders (4 to 8 s, seen in the smoke test). After that the shader cache makes GPU cold starts 2 to 5 s.

## Parakeet Redux vs its original model

Redux's compression is 7x smaller on disk, makes the model 2x faster on a CPU, and costs about 17% more word errors (+0.8 points). An ordinary 8-bit version of the same original gives up no accuracy.

|  | Parakeet Redux | Original, 8-bit | Original, F16 |
| --- | --- | --- | --- |
| Weights on disk | 178 MB | 740 MB | 1,256 MB |
| WER, 78 public clips | 5.33% | 4.52% | 4.54% |
| WER, accent set (12 speakers) | 3.62% | 2.17% | 2.17% |
| WER, user's own voice (verbatim) | 4.89% | 2.67% | 3.11% |
| CPU speed, 6 threads | 15.1x (Linux container) | 9.3x native, 10.7x in container | 7.6x native |
| Peak RAM on CPU | 982 MB | 1,232 MB | 1,724 MB |
| Runs on this PC's GPU | No | Yes, 36.7x | Yes, 38.7x |
| Runs natively on Windows (this CPU) | No | Yes | Yes |

Paired bootstrap over the same 78 clips (10,000 resamples), difference in WER points:

| Comparison | Difference | 95% CI | Share of resamples where the first is worse |
| --- | --- | --- | --- |
| Redux minus original F16 | +0.80 | -0.10 to +1.54 | 96% |
| Redux minus original 8-bit | +0.81 | -0.14 to +1.60 | 96% |
| Original 8-bit minus F16 | -0.02 | -0.07 to +0.05 | 19% |
| Redux minus Parakeet Unified (Handy) | +1.18 | +0.23 to +2.08 | 99% |

- **Pre-registered test:** "no meaningful loss" required the upper bound to stay below +0.5 points. It reaches +1.54, so that claim is **not supported**. The loss is not quite significant at 95% on 78 clips either, but 96% of resamples point the same way.
- **Where it loses:** accented speech. On the accent set and on the user's own voice, Redux has 1.5 to 2 points more errors than the original. On clean benchmark audio the gap nearly disappears (4.40% vs 4.20%).
- **What standard quantization shows:** the 8-bit version is 41% smaller than F16 with identical accuracy, and the same speed class on the GPU. Most of the size saving is available without Redux's accuracy cost; Redux's real advantage is CPU speed.
- **The vendor's claim:** the [model card](https://huggingface.co/moondream/parakeet-redux) reports 6.55% average WER on 7 Open ASR Leaderboard sets and 9.04% under background noise, with no accent breakdown. Our clean-audio result agrees the loss is small there; the accent results show it is not uniform.

## Cost and privacy

Local models cost nothing per hour and keep audio on the machine. Cloud models cost $0.22 to about $0.37 per audio hour and send every recording to a third party, under retention terms that differ by vendor.

| System | Price per audio hour | Where the audio goes | Kept by default? | Zero retention available? |
| --- | --- | --- | --- | --- |
| All 7 local models | $0 | Stays on this PC; works offline | No | Not needed |
| [OpenAI `gpt-transcribe`](https://developers.openai.com/api/docs/models/gpt-transcribe) | $0.27 ($0.0045/min) | OpenAI (US or EU region) | [Not stored for abuse monitoring on `/audio/transcriptions`; not used for training](https://developers.openai.com/api/docs/guides/your-data) | Yes, for approved customers |
| [ElevenLabs Scribe v2](https://elevenlabs.io/pricing/api) | $0.22 | ElevenLabs | [Yes: retained per privacy policy, history on by default](https://elevenlabs.io/docs/eleven-api/resources/zero-retention-mode) | Enterprise only (`enable_logging=false`) |
| [Gemini 3.5 Transcribe via Vercel](https://vercel.com/ai-gateway/models/gemini-3.5-transcribe) | \~$0.37 (estimate\*) | Vercel AI Gateway, then Google | [Gateway keeps nothing; upstream depends on routing](https://vercel.com/docs/ai-gateway/capabilities/zdr) | Opt-in, Pro and Enterprise plans |

\*Gemini is billed per token ($2 per 1M input, $12 per 1M output). The estimate assumes 32 audio tokens per second and about 12,000 output tokens per hour of speech.

All cloud runs here used default account terms, with no zero-retention option requested, which is what a typical user gets. For subset D (the user's own voice), cloud submission required separate consent.

## Threats to validity and deviations

Two deviations from the plan affected how systems were run; neither changes a conclusion. The main limit is sample size: 78 public clips and 4 dictated paragraphs.

| Deviation | Why | Effect |
| --- | --- | --- |
| D1: Redux run in a Linux container | Photon's Windows build has no AVX2 kernel for this CPU | Measured: the container ran 8-bit Parakeet 16% faster than Windows did, so Redux's CPU speed is, if anything, slightly favored |
| D2: YouTube clips peak-normalized, all systems rerun | 31 of 66 chunks had samples above full scale; Photon rejected them and cloud uploads clipped them | One linear gain per clip (-0.6 to -5.5 dB); all 10 systems rerun on identical audio |
| WSL raised to 12 CPUs during runs | Docker was capped at 4 CPUs, below the 6-thread rule | Restored afterwards |
| Speed subset reduced to 20 clips | Whisper runs below real time on CPU | Decided before any timed run |
| Own-voice set: 4 paragraphs instead of 15 sentences | User request, before recording | Same content |
| Gemini rate limit (5 requests/min) | Vercel team limit | Paced requests; accuracy unaffected |

Other limits:

- **Reference style.** YouTube captions are verbatim in places and edited in others. Whisper writes clean text (it drops repeated fillers), which verbatim captions count as errors.
- **Single machine.** Speeds are for a Ryzen 5 5600H and GTX 1650. A CPU with AVX-VNNI or AVX-512 would run Redux faster; Photon's fast paths were not available here.
- **Small subsets.** Accent (12 clips), noise (10) and own voice (225 words) results are directional; their confidence intervals are wide.
- **Model versions.** Cloud models change without notice. Results are as of 2026-10-05: `gpt-transcribe`, `scribe_v2`, `google/gemini-3.5-transcribe`.
- **Runtime choice.** Open models other than Redux ran in transcribe-cpp (the engine Handy uses). Other runtimes (NeMo, faster-whisper, CUDA) could change the speed figures and, slightly, the output text. The Parakeet Unified GGUF's publisher reports validating it against NVIDIA's NeMo reference; the other conversions were not checked against their originals here.

## Reproducibility

Everything needed to rerun or audit the experiment is in `D:\Development\Experiment`. The method was written down before any result was collected, and every later change is timestamped in the lab notebook.

| File | Contents |
| --- | --- |
| `README.md` | Overview and the exact commands, in order |
| `docs/PROTOCOL.md` | Pre-registered method: questions, systems, data, metrics, controls |
| `docs/LOG.md` | Lab notebook: every decision, failure and deviation |
| `data/units.csv` | The exact 147 audio units every system received |
| `data/manifest_*.csv` | Source, licence and reference text for every clip |
| `data/own_voice/` | The dictation scripts and the verbatim reference |
| `results/raw/<system>/` | Every transcript, timing and resource log |
| `results/metrics_accuracy.csv`, `metrics_speed.csv`, `paired.csv`, `per_clip.csv` | Scored results |
| `results/env.json` | Package versions, GPU driver, SHA-256 of every model file |
| `docker/Dockerfile.photon` | The Linux runtime used for Redux |

Key versions: transcribe-cpp 0.3.1, moondream 2.6.1 (kestrel-kernels 0.7.4), jiwer 4.0.0, whisper-normalizer 0.1.15, NVIDIA driver 610.88. Audio files are not redistributed (licences; the user's private recording). The scripts rebuild them from the sources listed in the manifests.
