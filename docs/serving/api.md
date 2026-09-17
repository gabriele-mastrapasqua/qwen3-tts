# HTTP Server — the API

_[Serving index](README.md) · **api** · [CPU server](cpu.md) · [CUDA server](gpu-cuda.md) · [boxes](boxes.md)_


> **This page is the API.** For how to run the process in production — pre-forked workers,
> finding the `W x K` topology for a given box before quoting anything from it, the deployment
> profile, the benchmark suite and what its numbers mean — see
> [`cpu-operations.md`](cpu-operations.md).
>
> **The endpoints and the streaming contract on this page are backend-independent.** Everything
> operational below assumes the CPU backend, which is the qualified one. The same server also runs
> on `--backend cuda`, with different sizing, different flags and a different maturity level
> (implemented and runtime-verified, not performance-qualified) — see
> [`gpu-cuda.md`](gpu-cuda.md). Do not carry a number
> between the two.


The built-in HTTP server loads the model once at startup and keeps weights in memory
across requests. The tokenizer is cached after the first call, so subsequent requests
skip all loading overhead and go straight to inference.

> **Serving many users at once?** Add `--batch-size N` to step concurrent requests
> **together** through the model (vLLM-style request batching: weights read once,
> continuous scheduling, per-request streaming). See
> [cpu-batching.md](cpu-batching.md). This page covers the single-request server.
>
> **Running it in production?** `--prefork W --prefork-threads K` (Linux) forks *W* workers
> after the weights are loaded and pins each to its own slice of cores, sharing the weights
> copy-on-write. Note that `--batch-size` is then also the **per-worker in-flight cap**, and
> that it defaults to 1: with `--prefork 12` and no `--batch-size`, twelve requests run at
> once and the rest wait in the listen backlog. How to choose *W x K* on your box, and how to
> measure it: [cpu-operations.md](cpu-operations.md).

## Starting the Server

```bash
# Basic — preset voices
./qwen_tts -d qwen3-tts-0.6b --serve 8080

# With a custom voice preloaded — DEFAULT: 8 KB x-vector .bin (clean, no room reverb)
./qwen_tts -d qwen3-tts-0.6b --load-voice voices/mario.bin --xvector-only --serve 8080

# ALTERNATIVE: ICL .qvoice graft at startup, for max timbre mimicry
./qwen_tts -d qwen3-tts-0.6b --load-voice voices/mario.qvoice --icl-only --serve 8080

# With INT8 quantization (1.7B)
./qwen_tts -d qwen3-tts-1.7b --int8 --serve 8080
```

Cloning is set **at server start** via `--load-voice` (per-request bodies are preset-only —
there is no per-request clone field). The recommended default is the 8 KB **x-vector `.bin`**
with `--xvector-only`: it carries identity without the reference recording's room reverb, so it
stays clean across requests. Make it with `python3 tests/qvoice_to_xvec.py voices/X.qvoice`. The
ICL `.qvoice` with `--icl-only` also works at startup (preloading WDELTA weight deltas if present)
for maximum timbre mimicry. The voice language is preserved from the voice metadata across all
requests — clients don't need to specify language or speaker.

## Endpoints

### `POST /v1/tts` — Generate full WAV

Returns a complete WAV file (24 kHz, 16-bit PCM, mono).

```bash
# Minimal — defaults to speaker=ryan, language=English
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"Hello, how are you today?"}' -o output.wav

# With explicit options
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"Ciao, come stai?","speaker":"vivian","language":"Italian"}' \
  -o ciao.wav

# With style control (1.7B model only)
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"I cannot believe it!","instruct":"Speak angrily"}' \
  -o angry.wav

# With emotion — same ear-validated recipe as the CLI --emotion
# (joy/happy/excited/proud/news/dramatic/calm/sad/gloomy/annoyed/stern/angry).
# Sets the Code-Predictor steering vector for (emotion, language) + volume/tempo.
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"What a wonderful day!","speaker":"ryan","language":"English","emotion":"joy"}' \
  -o joy.wav

# Per-sentence DYNAMIC emotion — inline [mood] tags switch emotion mid-text within ONE request.
# The server auto-detects markup, synthesizes each span with its own emotion and concatenates.
# Same tag set as the CLI: [joy] [sad] [excited] [proud] [calm] [angry] … + [neutral], [pause:400ms],
# and paralinguistics [laugh]/[sigh]. Works on /v1/tts (full) and /v1/tts/stream (span-by-span).
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"[joy] What wonderful news! [sad] But I have to go now. [pause:400ms] [calm] Take care.","speaker":"ryan","language":"English"}' \
  -o dynamic.wav
```

### `POST /v1/tts/stream` — Streaming PCM

Returns chunked raw PCM (s16le, 24 kHz, mono) as it generates.
First audio arrives within ~1 second.

```bash
# macOS — real-time playback via ffplay
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you today?"}' | \
  ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -

# macOS — real-time playback via sox
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you today?"}' | \
  play -t raw -r 24000 -e signed -b 16 -c 1 -

# Linux — real-time playback via aplay
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you today?"}' | \
  aplay -f S16_LE -r 24000 -c 1

# Save raw PCM to file, then convert
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello"}' -o output.raw
ffmpeg -f s16le -ar 24000 -ac 1 -i output.raw output.wav
```

### `POST /v1/audio/speech` — OpenAI-Compatible

Drop-in replacement for the OpenAI TTS API. Maps `input` to text, `voice` to speaker.

```bash
curl -s http://localhost:8080/v1/audio/speech \
  -d '{"input":"Hello world","voice":"ryan"}' -o output.wav
```

### `GET /v1/speakers` — List Available Speakers

```bash
curl -s http://localhost:8080/v1/speakers | python3 -m json.tool
```

### `GET /v1/health` — Health Check

```bash
curl -s http://localhost:8080/v1/health
```

`200` with `"status":"ok"` means the server can take work; `503` with
`"status":"unavailable"` means it cannot, and only a **batched** server can say so — its
scheduler thread is the part that can die under it. `mode` says which server answered:

| mode | `scheduler` | when |
|---|---|---|
| `single` | `none` | the plain server (`--serve` without `--batch-size`). There is no scheduler to report, and its absence is not a fault |
| `batched` | `running` / `down` | `--batch-size N`. `down` before the scheduler has come up, and after it has failed — the one case where health answers `503` |

The counters (`admitted`, `done`, `rejected_queue_full`, `rejected_queue_timeout`, `timed_out`)
are cumulative for the life of the process, and with `--prefork` each worker keeps its own.

## Request Body

```json
{
  "text": "Hello world",
  "speaker": "ryan",
  "language": "English",
  "instruct": "Speak cheerfully",
  "emotion": "joy",
  "volume": 1.0,
  "rate": 1.0,
  "seed": 42,
  "temperature": 0.5,
  "top_k": 50,
  "top_p": 1.0,
  "rep_penalty": 1.05
}
```

All fields except `text` are optional. Defaults: speaker=ryan, language=English,
temperature=0.5, top_k=50, top_p=1.0, rep_penalty=1.05, seed=random.

**Expressivity fields:**

| Field | Meaning |
|---|---|
| `instruct` | Free-form style prompt (**1.7B only**), e.g. `"Speak angrily"`. |
| `emotion` | Named mood — same recipe as the CLI `--emotion`: `joy`, `happy`, `excited`, `proud`, `news`, `dramatic`, `calm`, `sad`, `gloomy`, `annoyed`, `stern`, `angry`. Sets the Code-Predictor steering vector for the `(emotion, language)` pair (applied during generation, so it works on **both** `/v1/tts` and `/v1/tts/stream`) and applies the recipe's volume/tempo. Best on 1.7B. |
| `volume` | Linear output gain (`1.0` = unchanged). Overrides the emotion recipe's volume; applied on both full and streaming paths. |
| `rate` | Pitch-preserving tempo (`>1` faster). Overrides the emotion recipe's rate. Applied on `/v1/tts`; **not** on `/v1/tts/stream` (needs the full buffer). |
| inline `[mood]` markup (in `text`) | **Per-sentence dynamic emotion.** If the `text` carries inline tags — `[joy]`/`[sad]`/`[excited]`/… to switch mood mid-text, `[neutral]` to reset, `[pause:400ms]`/`[break:1s]` for gaps, `[laugh]`/`[sigh]` paralinguistics — the server splits the text into spans, synthesizes each with its own emotion, and concatenates them. Auto-detected on **both** endpoints; `/v1/tts/stream` flushes span-by-span (low time-to-first-audio). This is the same composer as the CLI's `--compose` / auto-detected `--text`. The top-level `emotion` field sets a single mood for the whole request; inline tags let one request span several. Same tag set as [docs/markup.md](../markup.md). |

Each request resets its **sampling parameters** to defaults (speaker, language, temperature,
top-k/p, rep-penalty, seed) **and clears any prior emotion steering**, so nothing leaks between requests.

> **Reproducibility (fixed 2026-06-03):** identical consecutive requests now produce **bit-identical**
> output, and a cold server request matches the CLI. The earlier cross-request divergence was a stale
> `ctx->dec_x` left over on a full-prefix match; the fix forces a fresh prefill in that case (the
> partial-match delta-prefill optimization is preserved). Regression-guarded by `make test-serve-repro`
> (3 identical requests, bit-identical) and `make test-serve-concurrent` (per-worker clones, corr=1.0).

## Limits, validation and errors

Every POST goes through the same two functions — one precheck on the HTTP envelope, one parse of
the JSON body — so `/v1/tts`, `/v1/tts/stream` and `/v1/audio/speech` answer identically, and so
do the plain server and the batched one (`--batch-size`, `--prefork`). A client can be written
against one table.

| status | when |
|---|---|
| `400` | body is not a JSON object, malformed JSON, nesting deeper than 16, a number longer than 40 characters, an **unknown field**, `text` missing or empty, `text` over the length limit, `speed` outside 0.25–4.0, an unknown speaker, `voice_design` on a model that has none |
| `405` | right path, wrong method — `GET /v1/tts` |
| `413` | request body larger than the read buffer, refused from the `Content-Length` before the body is read |
| `415` | a `Content-Type` other than `application/json`: no form data, no multipart, no file upload |
| `503` | queue full at admission, or a queued request that waited past `--queue-timeout-ms` |

Errors carry an OpenAI-shaped body, and the message says which bound was hit rather than that one
was:

```json
{"error":{"message":"unknown field 'voise' - this server implements: text, speaker, language, seed, temperature, top_k, top_p, rep_penalty, instruct, emotion, volume, rate","type":"invalid_request_error","param":null,"code":null}}
```

An unknown field is **rejected, not ignored**: a typo'd `"voise"` that is silently dropped
produces a perfectly successful request in the wrong voice, and nothing in the response says so.
Parameters that have a sensible range are clamped instead of refused — `temperature` to 0–2,
`top_p` to 0–1, `top_k` to the codec vocabulary, `rep_penalty` to 0.5–2 — because a value out of
range there has an obvious intent, while an unknown key does not.

### The input-length limit is derived, not a constant

`8192` characters is the compiled ceiling, not the answer. The effective limit is the smaller of
what a batch slot's prompt budget holds (`QWEN_BATCH_MAX_PROMPT × 3.5` characters, so 1792 at the
default 512) and what the server can finish inside its per-request cap
(`--max-request-seconds × 30` characters per second, so 1800 at the default 60 s), floored at 200
so a tight cap cannot make the server refuse ordinary sentences. A request that could not have
finished is refused at the door, with a message naming the bound it hit, instead of being killed
at minute two with a slot already spent:

```json
{"error":{"message":"text too long: 2000 characters, maximum 1792 - a longer prompt does not fit a batch slot's 512-token budget (QWEN_BATCH_MAX_PROMPT)","type":"invalid_request_error","param":null,"code":null}}
```

Both ends are configurable — `--max-request-seconds N` / `QWEN_MAX_REQUEST_S` (default 60, `0`
disables the cap) and `--max-text-chars N` / `QWEN_MAX_TEXT_CHARS` — and the server prints the
effective pair at startup, so a log can be audited after the fact:

```
[serve] per-request generation cap: 60 s -> text limit 1792 characters, frame cap 750 = 60.0 s of audio (from --max-request-seconds); a request that reaches the frame cap is TRUNCATED and logged (--max-request-seconds N / --max-text-chars N; 0 disables the text cap)
```

The seconds bound the generation as well as the text: the batched engine stops a request at
`--max-request-seconds × 12.5` codec frames (`QWEN_BATCH_MAX_FRAMES` overrides it, and a
too-large value is clamped to the RoPE cache). Reaching that cap is **not** an end-of-speech:
the audio ends where the model was cut. The server says so rather than letting the stream look
complete — one `[serve] WARNING: request TRUNCATED after N frames (S s of audio)` line per
request on stderr, and `truncated=1` in the `[REQ]` trace when `QWEN_REQ_TRACE` is on. Until
2026-09-08 this cap was a silent 600 frames (48 s) regardless of `--max-request-seconds`, so a
long prompt admitted under the 60 s text limit came back truncated with no trace at all.

`GET /v1/health` reports the same numbers live, which is how a client discovers the limits it is
subject to instead of hard-coding them:

```json
{"status":"ok","mode":"batched","scheduler":"running","num_requests_running":0,
 "num_requests_waiting":0,"queue_max":1,"queue_timeout_ms":0,"max_request_ms":60000,
 "max_text_chars":1792,"admitted":12,"done":12,"rejected_queue_full":0,
 "rejected_queue_timeout":0,"timed_out":0}
```

A deployment profile records the same two knobs, so what a machine was qualified with travels
with it: see [`configs/perf/README.md`](../../configs/perf/README.md) and
[`cpu-operations.md`](cpu-operations.md).

## Performance

> These are **single-request** figures on the development machine — what one caller experiences,
> not what a server holds. They say nothing about concurrency, and they predate the current serving
> architecture. For serving capacity, and which hosts it was qualified on, see
> [`boxes.md`](boxes.md).

Benchmarked on Apple M1 8-core, 16 GB RAM, 4 threads, same text + seed (`--seed 42`). bf16 below;
**with `--int8` the 0.6B server is faster than real-time warm — RTF ~0.88** (and ~0.93 with a cloned
`.qvoice`). See [performance.md](../performance.md) for the full int8 sweet-spot table.

| 0.6B, bf16 | Short text (~8s audio) | Long text (~16s audio) |
|---|---|---|
| **First call** (cold) | 12.2s → RTF 1.50 | 20.0s → RTF 1.28 |
| **Warm call** | 11.3s → RTF 1.39 | 19.7s → **RTF 1.26** |
| **Warm call, `--int8`** | **RTF ~0.88** ⚡ | even lower |

The first request pays a one-time cost for tokenizer parsing (~200ms) and warming the
OS page cache for mmap'd weights. Warm calls benefit from:

- **Cached tokenizer** — parsed once, reused across requests
- **Resident weight pages** — mmap'd BF16 weights stay in RAM
- **Pre-allocated buffers** — zero malloc in decode loop
- **LRU text embedding cache** — ~8MB for 2048 tokens, skip 2 matvec per cached token
- **Decoder thread overlap** — speech decoder runs in background during generation

### Custom Voice Server Performance

RTF with `.qvoice` loaded (Apple M1, 4 threads, Italian, seed 42):

| Mode | 0.6B RTF | 1.7B RTF |
|------|----------|----------|
| CLI | 1.48 | — |
| Server (warm) | **1.44** | 3.32 |
| Server stream | **1.48** | 3.18 |
| Server (cold) | 2.01 | 3.57 |

Custom voices have **no meaningful RTF penalty** compared to preset voices.

## Testing

```bash
make test-serve          # Health, speakers, TTS integration test
make test-serve-bench    # 2 runs, same seed, verify bit-identical output
make test-serve-openai   # OpenAI-compatible /v1/audio/speech endpoint
make test-serve-parallel # 2 concurrent requests, verify both complete
make test-serve-all      # Run all server tests
make serve               # Start server on port 8080
```
