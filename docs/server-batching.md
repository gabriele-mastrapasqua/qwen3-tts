# Server request-batching (vLLM-style)

> **PRODUCT 2 of the batching arc.** Serve **N different users' different requests** concurrently
> with maximum efficiency, by stepping them **together** through Talker + Code Predictor
> (weight-stationary — each weight read from DRAM once, reused across all in-flight requests).
> Distinct from `--batch` (PRODUCT 1), which splits **one long text** for **one user** and
> concatenates. Same engine (~90% shared); different use case. See `docs/batching.md`.

Built on `feat/server-batching`. Opt-in; default server behaviour is unchanged.

## TL;DR

```bash
# Continuous request-batching server, up to 4 concurrent users stepped together:
./qwen_tts -d qwen3-tts-0.6b --serve 8000 --batch-size 4

# Each user hits the normal endpoints; the server batches them transparently:
curl -s localhost:8000/v1/tts -d '{"text":"...","speaker":"ryan","language":"Italian"}' -o a.wav
curl -s localhost:8000/v1/tts/stream -d '{"text":"...","speaker":"vivian"}' -o b.pcm   # batched AND streamed
```

`--batch-size 1` (default) = the classic single/`--workers` server, untouched.

## Architecture

Three roles, one process:

```
            accept()            cq            readers              jq (batch)        continuous scheduler (owns ctx)
 client ───────────────▶ [conn queue] ──▶ parse HTTP ──┬─ batch ─▶ [job queue] ──▶ ┌─────────────────────────────┐
                                          (read-only on │                          │ persistent frame loop:       │
                                           ctx)         │                          │  • admit queued → free slots │
                                          └─ single ──▶ [jq_single] ─▶ clone ctx    │  • batched ragged Talker+CP  │
                                            (instruct/                  worker      │  • per-slot sample / EOS     │
                                             voice_design)                          │  • per-slot stream OR decode │
                                                                                    └─────────────────────────────┘
```

- **One scheduler thread owns `ctx` and is the SOLE batch synthesizer.** It runs a persistent
  frame-stepping loop (`qwen_tts_serve_continuous`): every frame it (1) admits queued requests into
  free slots, (2) steps all active slots together through the batched ragged Talker + Code Predictor,
  (3) samples per-slot with that request's own params + RNG, (4) on EOS finalizes the request and
  frees its slot.
- **Continuous batching** (not static): a slot freed by an EOS'd request is **immediately refilled**
  with the next queued request (prefill → inject its KV into the slot → generate in-flight). No
  waiting for the slowest request in a group — maximum utilization on ragged EOS.
- **Reader pool** parses HTTP into jobs and is **read-only on `ctx`** (never synthesizes). Hands the
  connection fd to the scheduler.
- **Single-job worker** runs `instruct` / `voice_design` requests on a **cloned ctx** (shares weights)
  so they never stall the batch.
- **Opportunistic, zero added latency**: a batch synth takes seconds, so concurrent requests pile up
  in the queue meanwhile; the scheduler drains them into the next frames. No artificial linger window.

### Per-request independence

Each slot carries its **own** text, preset speaker, language, sampling params (temp/top_k/top_p/
rep_penalty), seed and greedy-warmup. Per-slot **RNG state** is swapped in/out per slot per frame, so
each request reproduces its single-stream output **bit-for-bit** (validated mel-corr 1.0). Per-seq KV
isolates the prompts. Zero cross-talk between users.

**Scope:** per-request **preset** voices only. A loaded custom `.qvoice` / quant mode is per-SERVER
(shared weights), set at startup — not switchable per slot.

### Streaming composes (the vLLM property)

Because the loop steps **one frame at a time**, after each batched step every active request has a new
frame. A streaming request (`/v1/tts/stream`, preset voice) keeps its **own** streaming decoder state
and emits that frame as chunked PCM immediately — while the Talker + CP compute stays **batched**. So
you get **batched throughput + per-request parallel streaming at the same time**, exactly how vLLM
streams tokens to N concurrent users. (`instruct`/`voice_design` streams still use the single clone
worker.)

## Endpoints

Same as the normal server (see `docs/server.md`):

| Endpoint | Method | Batched? |
|---|---|---|
| `/v1/tts` | POST | ✅ continuous batch → WAV |
| `/v1/audio/speech` | POST | ✅ continuous batch → WAV (OpenAI-compatible) |
| `/v1/tts/stream` | POST | ✅ batched **and** streamed (preset voice); chunked s16le PCM |
| `/v1/speakers`, `/v1/health` | GET | answered inline by readers |

Request JSON: `text`/`input`, `speaker`/`voice`, `language`, `temperature`, `top_k`, `top_p`,
`rep_penalty`, `seed`, (`instruct`, `voice_design` → single clone worker).

## Testing (M1, correctness)

| Make target | What it asserts |
|---|---|
| `make test-serve-batch` | 3 concurrent **different** users (ryan/IT, vivian/EN, serena/IT): each response mel-corr **1.0** vs its own single-stream (`QWEN_BATCH_FORCE_MATVEC=1`), cross-talk 0.21 (zero leak), `[BATCH]` shows real batching; + production matmat smoke + stream fallback. |
| `make test-serve-continuous` | **Continuous admission**: N=6 requests at max_batch=2 → peak `admitted` climbs 2→6 (> batch width) as slots free; 6/6 complete; c_0 mel-corr 1.0. |
| `make test-serve-stream-batch` | **Streaming × batching**: 2 concurrent `/v1/tts/stream` batched **and** streamed; streamed PCM corr **1.0** + exact sample count vs single-stream. |

Scripts: `tests/serve_batch.sh`, `tests/serve_continuous_stress.sh`, `tests/serve_stream_batch.sh`.
All wired into `make test-serve-all`.

**`QWEN_BATCH_FORCE_MATVEC=1`** makes the batched matmul do B matvecs (bit-exact to single-stream) —
the rigorous correctness gate. The default matmul path forks the greedy trajectory benignly at temp>0
(fp-order, like int8) → compare by ear / mel-corr, not md5.

> Server-test hygiene (see CLAUDE.md): always `timeout` curls and kill the server **by name**
> (`pkill -9 -f "qwen_tts.*--serve"`); `wait` only on curl PIDs — never on the never-exiting server.

## Performance — measured on rented silicon (2026-07-11)

The throughput win (read each weight once, reuse across B) shows up on **bandwidth-bound** boxes.
On M1 (bf16, the dev box) it is **correctness-validated** and the aggregate RTF for a small batch
already dips ~0.95. Measured on real hardware:

| Backend / box | Batch result |
|---|---|
| **CUDA A100** (1.7B quant-mixed, B=8) | 8 concurrent users in 30 s for 63.5 s of audio → **aggregate RTF 0.47, ~2.1× throughput** (per-request 1.27 — the known latency/throughput trade) |
| **ARM Graviton3** (i8mm, 0.6B int8, B=4) | 4 concurrent, aggregate RTF 0.84; the **SMMLA GEMM twins cut wall −19%** vs the pre-MMLA batch (int8 batch matmat 0.34×→**2.1×**, bf16 →1.5×, int4 →1.6×) |
| **ARM Graviton3** (0.6B **int4**, B=4) | aggregate RTF **0.94** — int4 batched serving is viable on ARM now (the old scalar q4 batch was a 3.4× loss) |
| **x86 EPYC Turin** (B=4) | stall-free (8-9 s/req; the historical 262 s scheduler bug is fixed + re-validated); inactive slots are compacted (skip per-slot vector work) |

int8/int4 inherit from the batched matmat twins (batching pays most at low precision — it amortizes
the unpack); per-box details in `docs/hardware-testing.md`.

## Status / next

- ✅ S1 batch-multi engine + dynamic scheduler · ✅ S2 continuous batching · ✅ S3 per-request streaming
  compose — all built and correctness-validated on M1.
- ⏭ Validate **throughput** on x86 EPYC. Per-request `.qvoice`/quant switching (needs per-slot weight
  deltas). Tune admission (small linger under light load) + back-pressure. int8/int4 batched-path RTF
  on the rented boxes.
