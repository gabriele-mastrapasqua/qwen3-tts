# Runtime map — 1.7B int8 serving on AWS c8a.4xlarge, production C4 (2x8)

Date: 2026-09-04. Build: `b152946` + thread names (binary `2845f934`), profile
`aws-c8a-16c-vnni-ttfa`, model `qwen3-tts-1.7b-base --int8`, server
`--batch-size 8 --prefork 2 --prefork-threads 8`, masks `0-7 | 8-15`.

Every number is tagged **M** (measured on the box), **D** (derived arithmetically from
measurements) or **E** (estimated). Artifacts are under `profiles/` on the box; the ones
this document rests on are `runtime_map_20260904_175808` (M1 cost map + census at C1,
M2 malloc census, M3 thread tables, M4 `perf stat` A/B/C at C4, M5 `perf record` C),
`census_c4_20260904_153212` (per-thread ticks, state A), `census2_c4_20260904_164104`
(state B), `admit_probe_20260904_165102` (admission timeline), `sgemm_census_20260904_163329`
(BLAS shapes), `budget_final_20260904_163522/soak` and `rung2b_20260904_134115/soak`
(canonical 5-minute soaks).

Three states are compared throughout:

| state | binary | what it is |
|---|---|---|
| A | `qwen_tts.base` (`0968c2bf`) | original canonical baseline (three worker teams per worker) |
| B | `a449f60` (= current with `QWEN_CP_REGION=0`) | engine-owned budget: decoder tiles on the engine pool, OpenBLAS held at 1 thread, decoder SGEMM partitioned on the pool |
| C | `b152946` (current default) | B + code-predictor step as one persistent parallel region |

---

## 1. Serving call graph (production C4, one prefork worker)

One worker = one process pinned to one CCX (8 cpus). The batched loop runs on the
server's scheduler thread (`srv-sched`, `qwen_tts_server.c:1680 scheduler_main` →
`qwen_tts.c` batched serve loop). Every stage below runs on that thread; parallel work
is dispatched to the engine pool (`qwen_tts_thread.c: qwen_parallel`, 7 workers
`qwen-pool-0..6` + the calling thread).

```
srv-read-N (8)  accept/parse HTTP, enqueue request        qwen_tts_server.c:1388
      |
srv-sched  ──► batched loop iteration (qwen_tts.c ~2980-3300)
      |
      ├─ ADMISSION  (free slot & queued request)           qwen_tts.c:2988 ADMIT_PREFILL
      │     qwen_tts_generate(ctx, text, prefill_only=1)   qwen_tts.c:1360-1465
      │       tokenizer + prompt/ICL/speaker embeds → input_embeds (malloc per request)
      │       prefix cache lookup (pfx_find)  → delta_start / pos0
      │       qwen_talker_prefill(ctx, embeds, seq_len)    qwen_tts_talker.c:1297
      │         28 layers × { rms_norm | QKV | q/k norm+rope | KV store bf16 |
      │                       causal_attention_prefill | o_proj | norm | gate_up |
      │                       swiglu | down }
      │         projections: prefill_proj_matmat_qkv / prefill_proj_matmat
      │                      (bf16 weights, 16-token chunks → qwen_matmat_bf16)
      │     ADMIT_INSTALL: memcpy KV (28 layers) into the batch slot   qwen_tts.c:2996
      │
      ├─ TALKER STEP (all active slots, B = n_active)      qwen_tts.c:3268
      │     qwen_batch_talker_step_ragged → batch_talker_step_impl   qwen_tts_talker.c:2012
      │       28 layers × { per-slot rms_norm (serial) |
      │                     qwen_batch_proj_qkv  → qwen_matmat_int8_qkv (VNNI, 1 dispatch)
      │                     per-slot q/k norm, rope, KV store, attention (serial, loop thread)
      │                     batch_proj_q o_proj  → qwen_matmat_int8 (1 dispatch)
      │                     per-slot rms_norm_residual (serial)
      │                     batch_proj_q gate_up → qwen_matmat_int8 (1 dispatch)
      │                     per-slot swiglu (serial)
      │                     batch_proj_q down    → qwen_matmat_int8 (1 dispatch)
      │                     per-slot residual/norm (serial) }
      │       final norm → last_hidden
      │     codec head: qwen_batch_proj(logits, codec_head_bf16, 3072×2048)  qwen_tts.c:3047
      │                 (bf16 matmat, B = n_active, 1 dispatch, 12.6 MB read)
      │     SAMPLE_SLOT per slot (serial: clamp, EOS rules, top-p)         qwen_tts.c:3055
      │
      ├─ CODE PREDICTOR (all active slots)                  qwen_tts.c:3062
      │     qwen_batch_cp_predict                           qwen_tts_code_predictor.c:1090
      │       n_active == 1 → solo path qwen_cp_predict (per-op matvec, B=1)
      │       n_active ≥ 2 → 16 transformer steps (pos 0..15):
      │         per slot cp_mtp_project (matvec int8 1024×2048, 1 dispatch/slot)
      │         batch_cp_transformer_step:
      │           state C: ONE region cp_region_task (5 layers, 8 spin barriers/layer)
      │           state A/B: batch_cp_layer ×5 (4 dispatches/layer + serial per-slot)
      │         per slot cp_lm_argmax (argmax matvec int8 1024×2048, 1 dispatch/slot)
      │     RECORD_FRAME_AND_EMBED per slot (serial, 15 bf16 accumulations)
      │
      ├─ DECODER (inline, per slot whose pending frames ≥ target)   qwen_tts.c:3102
      │     qwen_speech_decoder_decode_streaming_st → sd_stream_st_body   speech_decoder.c:1931
      │       vq embed + rvq out proj (SGEMM) → pre_conv (int8 conv) → in_proj (SGEMM)
      │       → 4-layer decoder transformer (SGEMM ×9 per chunk, attention over KV)
      │       → latent out (SGEMM) → conv stack: convnext blocks (SGEMM pw1/pw2 1024↔4096),
      │         causal_conv1d int8 tiles (sd_conv1d_worker, sd_tile_2x4 VNNI),
      │         causal_conv_transpose1d_blas (SGEMM per kernel tap, 4 upsampling stages),
      │         snake activations (snake_row), final conv
      │       parallel work: sd_pool_run → qwen_parallel (state B/C) or sd-pool (state A)
      │       SGEMM: every cblas_sgemm in the decoder → qwen_sd_sgemm (qwen_tts_sd_gemm.c)
      │              state B/C: OpenBLAS at 1 thread, N or M partitioned on the engine pool
      │              state A: OpenBLAS's own 7-thread team
      │
      └─ OUTPUT  sink->on_chunk → sink_on_chunk → send_pcm_chunk    qwen_tts_server.c:1508
              malloc(int16 n) + f32→s16 + write(fd) per chunk (chunked HTTP)
```

Pool dispatch entry: `qwen_parallel(nt, fn, ctx)` — one job slot, `submit_mtx` held by the
caller for the whole region, workers spin `QWEN_POOL_SPIN=4096` then sleep on a condvar;
completion = atomic counter + caller spin then condvar. In-region barrier (state C):
`qwen_barrier_wait` (sense-reversing spin).

---

## 2. Call census

Source: M1 = canonical `tools/costmap_parity.sh --conc 1 --level 2` on the current binary
(7 requests of the short bank, **171 frames**, 24.4 frames/request, 1199 ms/request wall
at C1, B=1 → solo CP path). All numbers **M** unless noted.

### 2a. Per request and per frame at C1 (B = 1)

| item | calls/request | calls/frame | note |
|---|---|---|---|
| Talker prefill layers | 28 | — | 1 prefill/request; 7 requests → 196 QKV calls |
| Talker prefill bf16 matmat calls | 4 shapes × chunks | — | B = 8..16 token chunks (`matmat_bf16` rows in census) |
| Talker decode int8 GEMV (`matvec_int8`, 4 shapes) | 676 | **110.7** | 4 × 27.67/frame = 28 layers × 4 projections (VNNI) |
| Codec head bf16 (3072×2048) | 24 | 1 | `qwen_batch_proj` B=1 |
| CP decode int8 GEMV (`matvec_int8` ×3 + `_qkv`) | 6760 | **276.7** | 4 × 69.18/frame = 5 layers × 14 groups |
| CP lm_head `argmax_matvec_int8` | 362 | 14.8 | one per group |
| CP mtp projection `matvec_int8` 2048→1024 | 362 | 14.8 | one per group |
| CP prefill2 int8 matmat (B=2) | 121 | 4.9 | per-frame 2-token CP prefill |
| Decoder kernel calls (conv int8 + SGEMM) | 480 | 19.6 | 84 % VNNI conv, 16 % BLAS by GMAC |
| Decoder chunks (`decoder total`) | 6.3 | — | 44 chunks / 7 requests (1,2,4,8-frame chunks) |
| **Pool dispatches** (`runtime.pool_dispatch`) | **10 722** | **439** | 75 057 / 171 frames |
| Pool wait for completion | 116 ms/req | — | 9.7 % of request wall (caller waiting for workers) |
| Kernel calls total: talker / cp / decoder | 2 956 / 8 233 / 480 | 121 / 337 / 19.6 | GMAC/request: 52.6 / 32.0 / 60.6 |

### 2b. Per frame-pair at production C4 (B = 2 per worker) — **D** from the code paths + 2a

| item | state A/B | state C | how derived |
|---|---|---|---|
| Talker dispatches | 113 | 113 | 28 × 4 int8 matmat + codec head |
| CP dispatches | 16 × (20 + 2 + 2) = **384** | 16 × (1 + 2 + 2) = **80** | 5 layers × 4 per step; mtp + lm_head stay per-slot matvec |
| CP spin barriers | 0 | 16 × 5 × 8 = **640** | `cp_region_task` |
| Decoder dispatches | ≈ 2 slots × 35 / 8 frames ≈ 9 | ≈ 9 | conv tiles + partitioned SGEMM per chunk |
| Total pool dispatches / frame-pair | **≈ 506** | **≈ 202** | 12.5 frame-pairs/s → 6.3k vs 2.5k dispatches/s per worker |
| Voluntary context switches / s per worker | 11.2k (A, perf stat 224k/20 s, both workers) | **4.5k** (M4c, worker 0, 99 % voluntary, 39/s involuntary) | csw ≈ pool wake-ups that found workers asleep |

### 2c. Allocator traffic (M2: `perf stat` on libc uprobes, single 1x8 worker, C1, 3 requests, 8.48 s audio = 106 frames each) — **M**

| counter | non-streaming `/v1/tts` (per request) | streaming `/v1/tts/stream` (per request) | per frame (stream) |
|---|---|---|---|
| `malloc` | 555 | 612 | 5.8 |
| `calloc` | 1 | 1 | — |
| `realloc` | 49 | 66 | 0.6 |
| `posix_memalign` (= `aligned_malloc`) | 676 | **11 899** | **112** |
| `free` | 1 236 | **12 516** | 118 |
| `mmap` | 28 | **78** | 0.7 (≈ 6 per 8-frame chunk) |
| `munmap` | 34 | **154** | 1.5 |
| `brk` | 3 | 1 | — |
| `futex` syscalls | 5 114 | 30 309 | 286 |
| context switches | 2 025 (357/s) | 10 030 (1 750/s) | 95 |

Idle server (5 s): 0 on every counter (**M**). The streaming path therefore adds ≈ 11.2k
aligned allocations and ≈ 50 mmap/munmap pairs per request over the non-streaming path:
the per-chunk decoder scratch (section 5).

---

## 3. Thread ownership table (per prefork worker, 40 threads) — **M** (`/proc/<tid>/comm`, `wchan`, ticks; M3c idle, M4c under C4 state C)

| group | count | created at | affinity | idle state | C4 state | %CPU/thread at C4 | computes model? | destroyed |
|---|---|---|---|---|---|---|---|---|
| `srv-sched` (batched loop = "loop thread") | 1 | `qwen_tts_server.c:1823` | worker mask | futex wait | **runs the loop** | **77 %** | yes (serial sections + its chunk of every dispatch) | process exit |
| `qwen-pool-0..6` (engine pool) | 7 | `qwen_tts_thread.c:454` at `qwen_set_threads(8)` | worker mask | futex wait after spin | **compute** | **75 %** each (5.2 cores) | yes: every Talker/CP/decoder tile, partitioned SGEMM | `qwen_threadpool_stop` |
| main thread (unnamed `qwen_tts`) | 1 | process | worker mask | `unix_stream_data_wait` (parent socket) | idle | 0 | no | — |
| OpenBLAS pthreads (unnamed `qwen_tts`) | **15** | libopenblas init (16 cpus − 1) | worker mask | futex wait | **parked** (0 ticks, vol csw 2/s) | 0 | state A: yes (decoder SGEMM, 8 % each); B/C: never | library exit |
| `sd-pool-0..6` (decoder private team) | 7 | `qwen_tts_kernels.c:8488` **during pre-warm**, before the server selects the engine-pool mode | worker mask | futex wait | **parked** (0 ticks) | 0 | state A: yes (6 % each); B/C: never | never joined |
| `srv-read-0..7` | 8 | `qwen_tts_server.c:1819` | worker mask | futex wait | wake per request | 0 | no | process exit |
| `srv-single` | 1 | `qwen_tts_server.c:1827` | worker mask | futex wait | idle (batched mode) | 0 | no | process exit |
| `pf-helper`, `dec-thr`, `dec-ovlp` | 0 | opt-in only (`QWEN_PREFILL_HELPER`, `QWEN_DECODER_THREAD`, CLI overlap) | — | — | — | — | — | — |

Parent process: 4 threads (main + `qwen-pool-0..2` from `qwen_init_threads`), all idle,
mask 0-15. State A additionally had OpenBLAS ×7 at 8 % and `sd-pool` ×7 at 6 % under C4
(`census_c4_20260904_153212`), i.e. 21 compute-active threads on 8 cpus; state B/C: **8**.

Note for the runtime model: the "loop thread" of a worker is the server scheduler thread.
Its 77 % includes the serial per-slot sections (norms, attention, sampling, embedding,
decoder serial parts, HTTP writes) plus its own share of every parallel region; the 23 %
idle is where it waits for the pool.

---

## 4. Scheduler / OS census at canonical C4 (`perf stat -p worker0,worker1`, 20 s inside a 6-wave C4 run; counters only, no sampling) — **M**

| metric (20 s, both workers) | A original | B budget (`a449f60`) | C + CP region (`b152946`) |
|---|---|---|---|
| task-clock | 153.8 s | 154.8 s | 158.8 s |
| context switches | **224 389** (11.2k/s) | 113 103 (5.7k/s) | 117 067 (5.9k/s) |
| cpu migrations | 965 | 424 | 460 |
| cycles | 666.5 G | 679.4 G | 670.1 G |
| instructions | **679.4 G** | 545.5 G | 541.1 G |
| IPC | 1.02 | 0.80 | 0.81 |
| cache-misses (event `cache-misses`) | 13.01 G | 13.01 G | 13.04 G |
| wave C4 in the same run: STREAM p50/p95 | 1.006 / 1.041 | 0.951 / 0.982 | 0.977 / 1.005 (noise band ±0.03) |
| wave C4: host cores busy | 13.04 | 13.90 | 13.93 |

Readings (**D**): A executes 25 % more instructions for the same cycles: those are spin
instructions of three competing teams (`worker_main` 10 % of samples in A, plus OpenBLAS
and sd-pool wake-ups), which is why IPC *drops* when they are removed. `cache-misses` is
identical across states (13.0 G × 64 B / 20 s ≈ 41.6 GB/s summed over the two workers,
≈ 21 GB/s per CCX if the event counts last-level misses; DRAM traffic structure unchanged).
Voluntary vs involuntary (state C, worker 0, M4c): pool 4 389 vol / 39 invol per s; loop
thread 152 / 7; everything else ≈ 0 → context switches are pool sleep/wake, not preemption.

Per-thread CPU at C4 (worker 0, 12.3 s window, state C): pool 7 × 74.7 % = 5.23 cores,
loop 0.77 cores, all 32 other threads 0.00 → **6.0 of 8 cores useful+spin, 2.0 idle**.
Where the busy time goes (M5 `perf record -F 499`, state C, both workers, 20 s):

| symbol | share | meaning |
|---|---|---|
| `int8_matmat_vnni_tile_m4n4` | 49.7 % | Talker + CP batched int8 GEMM (B=2) |
| `int8_matvec_vnni_rowsum` | 10.3 % | B=1 int8 GEMV: CP mtp + lm_head per slot, single-slot steps |
| `worker_main` | 9.9 % | pool spin between dispatches (useful: 0) |
| `sd_tile_2x4` + `sd_conv1d_worker` | 7.4 % | decoder int8 conv tiles |
| `bf16_matmat_avx512_m2/m1/m4` | 6.8 % | codec head (B=2) + admission prefill (bf16) |
| `qwen_barrier_wait` | 4.5 % | CP region barriers (spin; replaces dispatch wake-ups) |
| OpenBLAS `sgemm_*` (kernel + pack) | 4.0 % | decoder SGEMM, serial per pool task |
| `qwen_parallel` | 0.9 % | caller-side dispatch/wait |
| other (quant, swiglu, attention, snake, kernel) | ≈ 6 % | |

So of the 6.0 busy cores per worker: ≈ 4.9 useful (kernels), ≈ 0.6 spin in `worker_main`,
≈ 0.27 spin in barriers (**D**). Idle 2.0 cores = the loop thread's serial sections and
the pool's sleep after the spin budget.

---

## 5. Allocations and transient work on the steady streaming path (code audit + M2 counts)

| site | what | frequency class | evidence |
|---|---|---|---|
| `qwen_tts_generate` (admission) | `input_embeds` malloc/free, prefill buffers (`pp_xT/pp_yT` TLS, grow-only), KV memcpy into slot (28 layers × pl × kvd × 2) | REQUEST ONLY | 555 malloc + 676 memalign per non-stream request |
| `mm_scratch_qx/pack/packb` | int8/bf16 activation scratch | INIT ONLY (TLS, grow-only) | `posix_memalign` only when capacity grows |
| batched Talker step | none (all buffers in `qwen_batch_t`) | — | |
| CP region (state C) | `qx`, `swtmp` static grow-once | INIT ONLY | |
| CP solo path (`cp_transformer_step`) | TLS scratch `S` realloc grow-once | INIT ONLY | |
| `sd_stream_st_body` | ≈ 28 `aligned_malloc`/`aligned_calloc` per chunk (vq_cf, pre_conv_out, hidden, pre_conv_rm, q/new_k/new_v/x_norm/attn_out, ffn_gate/up, …), `scores` malloc when n_keys > 512 | **FRAME HOT PATH** (per 8-frame chunk) | part of the 112 memalign/frame |
| `causal_conv1d_blas` (`col` im2col) and `causal_conv_transpose1d_blas` (`rk`, out_ch × in_len floats) | per call, hundreds of KB → **served by `mmap`/`munmap`** (glibc threshold) → page faults + zero fill every chunk | **FRAME HOT PATH** | 78 mmap + 154 munmap per stream request ≈ 6 pairs per chunk |
| `snake_row` (`temp = malloc(n)`) | one malloc **per channel row per snake call** | **FRAME HOT PATH**, highest count | dominant share of the 11.2k extra allocations per stream request |
| `convnext_mlp` (`pw1_out` 4096 × len floats) | per call | FRAME HOT PATH (mmap-sized) | |
| `send_pcm_chunk` | `malloc(int16 × n)` per output chunk + f32→s16 conversion + `write` | FRAME HOT PATH (1 per chunk) | small |
| acc_aud realloc (non-stream) | growth-doubling | REQUEST ONLY | |
| `qwen_sd_stream_init/free` | stream state (KV, latent cache) | REQUEST ONLY | |
| activation quantisation (`quantize_act_int8_col`), bf16 packing (`Xb`) | per projection call | LAYER HOT PATH (compute, no allocation) | 0.6 % of samples |
| `batch_gather/scatter` transposes | per projection | LAYER HOT PATH (memcpy-class) | inside dispatch time |

Everything at FRAME/LAYER/CP-GROUP frequency is inside the decoder chunk: **≈ 900 allocations
and ≈ 6 mmap/munmap pairs per 8-frame chunk**, none in the Talker/CP steady state.

---

## 6. Consolidated cost map (production C4, per worker; request = one stream)

Wall shares from the C1 cost map (**M**, 1199 ms/request, 24.4 frames); C4 values **D** from
the per-frame model (B=2 batched: Talker/CP weight pass shared by the two slots).

| component | calls/request (C1) | calls/frame | wall/request C1 (M) | % request C1 | dispatches/frame C4 (D) | allocs | weight bytes read / frame-pair (D) | backend | thread owner | ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| admission + setup (tokenizer, embeds, prefix cache, KV install) | 1 | — | ≈ 5 ms (E) | < 1 % | 0 | REQUEST | 0 | scalar | loop | — |
| Talker prefill | 1 (28 layers) | — | **61 ms** (M, ≤16 tokens); 108-110 ms for 20-30 tokens (M, admit probe) | 5.1 % | 112 (once per request) | REQUEST | 2.7 GB per 16-token pass (bf16) | bf16 AVX-512 matmat | loop + pool | stalls the other slot for its full duration: ~1 admission/s/worker × 0.11 s ≈ 10 % of wall at C4 soak |
| Talker decode | 28 layers × 4 | 112 GEMV (C1) / 112 GEMM B=2 (C4) | **533 ms** (M) | 44.5 % | 113 | none | 1.41 GB int8 (shared by 2 slots) + 12.6 MB codec head | int8 VNNI matmat/matvec | pool | at 89 % of the worker-local read roof: bandwidth-bound |
| Code predictor (16 steps × 5 layers) | 8 233 kernel calls | 337 (C1) | **375 ms** decode + 34 ms prefill2 (M) | 34.1 % | 80 (C) / 384 (A,B) | INIT | ≈ 1.1 GB int8 (shared) + 64 MB lm_head/mtp per slot | int8 VNNI | pool + loop (per-slot serial) | serial per-slot sections + 640 barriers/frame-pair; mtp/lm_head still per-slot GEMV (2× weight reads) |
| speech decoder (per 8-frame chunk) | 6.3 chunks, 480 kernel calls | 19.6 | **223 ms** (M) | 18.6 % | ≈ 9 | **≈ 900 allocs + 6 mmap pairs per chunk** | conv int8 VNNI 84 % + SGEMM 16 % (by GMAC) | pool (tiles) + loop (serial) | conv stack 206 ms/req (M) = 92 % of decoder; allocation/page-fault overhead not yet separated |
| OpenBLAS SGEMM (decoder) | ≈ 15 calls/chunk, 77 shapes / ≈ 8 families | ≈ 2 | 187 ms per ≈ 10 s audio (M) ≈ 2 % of wall | 2 % (direct arithmetic) | inside decoder | packing buffers inside OpenBLAS | — | fp32 OpenBLAS, serial per pool task (B/C) | direct arithmetic small; removal is hygiene + 15 parked threads |
| engine pool dispatch/wait | 10 722 dispatches/request | 439 (C1) / ≈ 202 (C, C4) | 116 ms waiting for workers (M, C1) | 9.7 % | — | none | — | `qwen_parallel` | loop | spin `worker_main` 9.9 % + barriers 4.5 % of samples at C4 |
| output/streaming | 6.3 chunks | — | < 1 ms/chunk (E) | < 1 % | 0 | 1 malloc/chunk | — | write(2) | loop | — |

Sum of measured stages at C1 = 61 + 533 + 375 + 34 + 223 = 1 226 ms ≈ 1 199 ms request wall
(regions are inclusive; prefill and admission overlap the first frames).

Production C4 consequence (**M**): canonical 5-minute soak, state B defaults
(`budget_final_20260904_163522`): 614 requests, 0 errors, STREAM p50 0.993-1.006, p95
1.057-1.069, TTFA 154-168 / 193-205 ms. Wave C4 (short bank, no admissions mid-stream)
sits at p50 0.92-0.98 / p95 0.93-1.00 for states B/C. The wave→soak gap (≈ +0.08 p95) is
the admission prefill stalling the active slot (probe: chunk gap 440-565 ms → 673-779 ms
on each admission).

---

## 7. Ranked interventions (evidence-backed)

| # | callsite / function | measured cost | structural problem | expected upper bound on C4 soak STREAM | scope | quality risk | validation |
|---|---|---|---|---|---|---|---|
| 1 | **admission prefill on the loop thread** `qwen_tts.c:2988 ADMIT_PREFILL → qwen_talker_prefill` (bf16, 16-token chunks) | 108-110 ms per admission (M), ≈ 1 admission/s/worker at C4 → ≈ 10 % of wall; wave→soak gap +0.06-0.08 p95 (M) | the prefill traverses 2.7 GB of bf16 weights per 16-token chunk (2 passes for 20-30 tokens) while both slots' frames stop | −5 to −8 % p95 if the prefill cost halves; scheduling alone is zero-sum (measured: helper/priority −3 % STREAM for +100/+250 ms TTFA) | (a) one bf16 pass for ≤ 32 tokens: needs a proper wide-B kernel (the naive block sweep measured 2-8× slower, bit-identical); (b) piggy-back the prefill rows on the decode step's int8 weight pass: changes prefill numerics → **separate opt-in path only** (`QWEN_PREFILL_INT8MM=1` exists: prefill 60 ms, soak 0.96/1.01, mel-corr vs bf16 0.39-0.60 = different rendition, not qualified) | (a) none, bit-identical; (b) quality path, needs corpus + listening | md5 parity for (a); admit probe prefill_ms; 2-min soak; canonical 5-min soak |
| 2 | **decoder per-chunk allocations** `sd_stream_st_body` (≈ 28 aligned allocs), `causal_conv1d_blas` `col`, `causal_conv_transpose1d_blas` `rk`, `convnext_mlp` `pw1_out`, `snake_row` `temp` | ≈ 900 `posix_memalign`+`free` and ≈ 6 `mmap`/`munmap` pairs per 8-frame chunk (M); page-fault + zero-fill cost not yet timed | scratch re-created and released at frame frequency; large ones cross the mmap threshold every chunk | −1 to −3 % (E): decoder is 18.6 % of wall and its serial/allocation overhead is inside the 15.6 % self time of `conv_stack` | keep per-stream persistent scratch in `qwen_sd_stream_state_t`, grow-once; `snake_row` temp from a per-thread buffer | none (bit-identical) | md5 parity, M2 malloc census (target < 20 allocs/chunk, 0 mmap), C4 wave, 2-min soak |
| 3 | **CP per-slot GEMVs inside the batched step** `cp_mtp_project` (2048→1024) and `cp_lm_argmax` (1024→2048) per slot per group, `qwen_tts_code_predictor.c:1119-1140` | 4 dispatches/step = 64/frame-pair (D); 2 slots × 2 × 2 MB × 16 = **128 MB of weights re-read per frame-pair** (D) vs 64 MB batched | at B=2 the same weight is read once per slot; the region already holds the team | −1 to −2 % (E) from bytes (≈ 2 ms of 80 ms) + 64 fewer dispatches | batch the two slots into one int8 matmat (B=2) inside `cp_region_task`, argmax per column | matvec vs matmat quantisation differ per element? verify md5; if not identical, keep the region's per-slot GEMV | md5 parity, C4 wave, 2-min soak |
| 4 | **Talker batched step as one region** `batch_talker_step_impl` (28 layers × 4 dispatches + per-slot attention on the loop thread) | 113 dispatches/frame-pair (D); per-slot attention over the full KV (pos ≈ 100-500) serial on the loop thread | same structure the CP region removed: 4 wake/complete cycles per layer, serial per-slot sections | −1 to −2 % (E) (the CP region gave −2 % p50 / −2.5 % p95 with 5× more dispatches removed) | reuse `qwen_i8mm_*` + `qwen_barrier_wait`, one region per step, attention one slot per thread | none (bit-identical, same kernels/partition) | md5 parity at B=2, C4 wave, 2-min soak |
| 5 | **decoder SGEMM families → project-owned kernels** (`qwen_sd_sgemm`, 8 families cover 90 % of BLAS time: conv-transpose T·N K=1536/768/384/192, convnext N·N 1024↔4096, transformer N·T M=3..100 × 512/1024) | direct SGEMM arithmetic ≈ 2 % of wall (M); packing = 2/3 of that time (perf) | general BLAS packs both operands per call for tiny M; 15 parked OpenBLAS threads per worker | −1 % (E) + removal of the dependency and the parked team | fixed-shape bf16/f32 kernels with prepacked static weights, OpenBLAS as fallback | numerics change from fp32 to bf16 weights if converted: keep fp32 first | md5 parity with fp32 kernels; census of remaining BLAS calls = 0 on the hot path |
| 6 | **pool sleep/wake policy** `qwen_parallel` / `worker_main` (spin 4096 then condvar) | 4.4k voluntary csw/s per worker (M); `worker_main` 9.9 % of samples = 0.6 core spinning (M); pool asleep ≈ 25 % of wall (M) | dispatches arrive in bursts separated by serial sections longer than the spin budget → sleep, then futex wake for every worker on the next dispatch | ≤ −2 % (E); spin 65536 measured neutral in wave and **+11 % p95 in soak** → not a knob, needs the serial sections shortened (items 3, 4) | no change alone | — | — |
| 7 | Talker/CP int8 GEMM kernels | 60 % of samples (M), Talker at 89 % of the worker-local read roof (M) | bandwidth-bound | ≈ 0 without fewer bytes per frame | out of scope | — | — |

Ceiling arithmetic (**D**): the soak needs p95 1.06 → ≤ 0.95 (−10 %). Items 1(a)+2+3+4
are bit-identical and sum to an estimated −8 to −15 %; item 1(b) is the largest single
lever but is a quality path and stays opt-in until qualified.

---

## Runs that must not be repeated

* `QWEN_DECODER_THREAD=1` and budgeted overlap variants: +20-50 % STREAM (extra team).
* `QWEN_POOL_SPIN=65536`: neutral in wave, +11 % p95 in soak.
* Internal cost map / census at sustained C4: distorts (use C1 for structure, external counters at C4).
* Any script that backgrounds the server and then calls bare `wait`.
