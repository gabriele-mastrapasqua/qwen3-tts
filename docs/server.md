# HTTP Server

The built-in HTTP server loads the model once at startup and keeps weights in memory
across requests. The tokenizer is cached after the first call, so subsequent requests
skip all loading overhead and go straight to inference.

## Starting the Server

```bash
# Basic — preset voices
./qwen_tts -d qwen3-tts-0.6b --serve 8080

# With a custom voice preloaded
./qwen_tts -d qwen3-tts-0.6b --load-voice voices/mario.qvoice --serve 8080

# With INT8 quantization (1.7B)
./qwen_tts -d qwen3-tts-1.7b --int8 --serve 8080
```

When started with `--load-voice`, the server preloads the `.qvoice` at startup
(including WDELTA weight deltas if present). The voice language is preserved from
the `.qvoice` metadata across all requests — clients don't need to specify language
or speaker.

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

## Request Body

```json
{
  "text": "Hello world",
  "speaker": "ryan",
  "language": "English",
  "instruct": "Speak cheerfully",
  "seed": 42,
  "temperature": 0.5,
  "top_k": 50,
  "top_p": 1.0,
  "rep_penalty": 1.05
}
```

All fields except `text` are optional. Defaults: speaker=ryan, language=English,
temperature=0.5, top_k=50, top_p=1.0, rep_penalty=1.05, seed=random.

Each request starts from clean defaults — parameters do not leak between requests.

## Performance

Benchmarked on Apple M1 8-core, 16 GB RAM, 4 threads. Same text, same seed (`--seed 42`),
identical output (bit-for-bit):

| | Short text (~8s audio) | Long text (~16s audio) |
|---|---|---|
| **First call** | 12.2s → RTF 1.50 | 20.0s → RTF 1.28 |
| **Warm call** | 11.3s → RTF 1.39 | 19.7s → **RTF 1.26** |

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
