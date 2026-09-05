# Cross-backend audit of the 2026-09-04 runtime work, and the CPU kernel parity map

Two questions, both answered from the code as it is in the tree today (commits up to `620279c`
plus the uncommitted opt-in prefill experiments), not from the docs and not from memory:

1. The runtime fixes of 2026-09-04 were measured on one box (AWS c8a, AVX-512 VNNI, 2x8 prefork,
   1.7B int8). Which of them are valid on the other backends we build — x86 AVX2, AVX-512F,
   AVX-512 VNNI/BF16, AMX, Arm NEON+DOTPROD (Apple), Arm i8mm+BF16 (Linux, KleidiAI), CUDA,
   Metal — and which are x86-only implementations of a common design?
2. Where does each CPU ISA stand on GEMV and GEMM per weight type, fused QKV, prefill and the
   decoder, measured against the most complete backend we have (Arm Linux with KleidiAI)?

Method: read every gate (`#if`, `qwen_mm_use`, `qwen_i8mm_usable`, the `*_available()` /
`*_enabled()` predicates), every dispatcher in `qwen_tts_kernels.c` and every caller in the
Talker, Code Predictor, speech decoder and server; diff the `getenv()` census against
`g_qwen_reported_flags[]` and `docs/feature-flags.md`; compile the tree on the Apple M1
(`make blas`, clang, GCD pool) as the one non-x86 build available without renting a box.
Nothing here is a measurement on Arm, AMX or a GPU. Where a fix is "not applied" on a backend
the equivalent inefficiency is still there and was not measured there.

Terminology for the classification, as the plan defined it:

- **A** common, runtime-level: one implementation, every backend takes it automatically.
- **B** common design, backend-specific implementation: applies where the implementation
  exists, the other backends keep the old behaviour.
- **C** x86-specific: the design itself is tied to one ISA family.

## 1. The 2026-09-04 fixes, backend by backend

| Fix (commit) | Class | Applied automatically on | Not applied / equivalent cost still present | Needs |
|---|---|---|---|---|
| Decoder tiles on the engine pool, `QWEN_SD_POOL=qwen` (a449f60) | A | every CPU backend, every OS — but **only in the batched server** (`qwen_tts_serve_continuous`, i.e. `--batch-size >= 2`, including prefork with batching) | plain `--serve` without `--batch-size` (single or prefork), and the CLI, still create the private `sd-pool` team next to the engine pool | decide whether the plain server should inherit the same default (one-line change in `qwen_tts_serve_ex`), then a short wave on a box |
| OpenBLAS held at one thread + decoder SGEMMs partitioned on the pool, `QWEN_BLAS_OWN=1` (a449f60, `qwen_tts_sd_gemm.c`) | B | Linux + OpenBLAS (any ISA). CUDA builds: GPU SGEMM first, the CPU fallback is partitioned | **macOS/Accelerate**: `qwen_blas_set_threads` is a no-op (only `openblas_set_num_threads` is known), yet `qwen_blas_own(1)` still turns the partition on, so Accelerate SGEMM slices run on the GCD pool while Accelerate keeps its own internal threading. Unmeasured; may be neutral or oversubscribe. Windows: no weak OpenBLAS symbols, nothing changes | macOS: either gate the partition on "the BLAS thread count is really 1" (`qwen_blas_threads_now() == 1`) or cap Accelerate through `VECLIB_MAXIMUM_THREADS` at start; measure once on Apple silicon |
| Nested-dispatch guard `qwen_parallel_active()` | B | pthread pool (Linux, and macOS with `QWEN_FORCE_PTHREAD`) | GCD and Windows stubs return 0. Safe on GCD (`dispatch_apply` is reentrant) and unreachable on Windows because the only in-region callers are the VNNI regions | nothing |
| Prefill helper at LOW pool priority, `QWEN_PREFILL_LOW_MS` (d528dc5) | B | pthread pool | GCD/Windows: `qwen_parallel_set_low_until` is a stub, the helper is first-come-first-served as before | nothing (opt-in knob, not a fix) |
| CP transformer step as one region, `QWEN_CP_REGION` (b152946) | **C** | x86 AVX-512 VNNI, int8 weights, pthread pool, B 2..16. On an **AMX** host the region runs only while `qwen_mm_use(INT8_AMX, B)` is false, i.e. B < 4 with the default `QWEN_AMX_MIN_B=4`; at B >= 4 every step falls back to the dispatched AMX path, so the mode flips with the number of active slots | Arm (SMMLA / KleidiAI), AVX2, AVX-512F, AMX at B >= 4, int4 and bf16 weights, GCD pool: 20 dispatches per step and the per-slot sections on the loop thread | per-backend row-block runners exposed like `qwen_i8mm_run*` (SMMLA, KleidiAI i8 GEMM, AMX tiles); the region body is ISA-neutral |
| Talker step as one region, `QWEN_TK_REGION` (faa496c) | **C** | same gate as the CP region | same as above: 112 dispatches per step elsewhere | same as above |
| CP batched heads, `QWEN_CP_BATCH_HEAD` (110ff48) | C (gate) / A (body) | x86 VNNI int8 heads, B 2..16 (not when AMX would take the shape) | Arm, AVX2, AMX at B >= 4, int4/bf16 heads: MTP projection and every lm_head run once per slot as GEMVs, the 2 MB weight streamed once per slot per group | the body only calls `qwen_matmat_int8`; the gate could admit any ISA with an int8 matmat. Acceptance must then be **code parity** (argmax), not bit parity: KleidiAI quantises the LHS per row, not with the per-column quantiser the VNNI path shares with the GEMV |
| Streaming decoder scratch arena, stream-state recycling (b0570a6) | A | every CPU backend and OS, and CUDA/Metal servers whenever the conv decoder runs on the CPU | non-streaming decode (`conv_decoder_forward`, the CLI without `--stream`) still allocates its buffers per request (once per call, not per chunk — by design) | nothing |
| Grow-once TLS scratch in the int8 conv workers (b0570a6) | B | the int8 conv kernels: AVX-512 VNNI and Arm DOTPROD | AVX2 / AVX-512F have no int8 conv kernel at all (fp32 im2col + SGEMM, buffers now in the arena) | see §2, decoder row |
| Per-thread PCM buffer in the server (b0570a6) | A | every backend | — | nothing |
| Thread names (63af9dc) | A | Linux (`prctl`), macOS (`pthread_setname_np`) | Windows: unnamed | nothing |
| Uncommitted: A1 row-major bf16 prefill, A2a shared QKV pack, `QWEN_PREFILL_INT8MM`, `QWEN_PREFILL_CHUNK` | C | AVX-512 BF16/VNNI, all opt-in and off | — | keep off; INT8MM stays a separate, non-qualified numerical path |

Compile check on the M1 (`make blas`, clang, `-march=native`, GCD pool, Accelerate): clean build,
seven warnings, none from the new code paths being wrong on Arm: unused `kt_t0`/`kt_B` in the
non-x86 branches of `qwen_matmat_int8_qkv` and `qwen_matmat_bf16_qkv`, unused `batch_proj` on
Arm, and a pointer-to-array truth test in the server. Not a runtime validation.

### Allocations that are still on a hot path (found by the same census, other backends)

The allocator work removed the per-chunk traffic of the streaming decoder on every backend.
These per-call `malloc`/`free` pairs remain, each one on a path that a backend other than the
c8a runs on every step or every prefill chunk:

| Site | Backend that pays it | Per |
|---|---|---|
| `qwen_matmat_bf16`, AMX branch: `Xb` (`qwen_tts_kernels.c`, the AMX bf16 matmat) | AMX hosts, bf16 weights or the bf16 prefill (which is AMX on those hosts) | projection call |
| `qwen_matmat_bf16`, BFMMLA branch: `Xb` | Arm Linux with KleidiAI bf16 disabled (`QWEN_NO_KAI_BF16`) | projection call |
| `qwen_matmat_bf16_qkv`: `Xb` | x86 batched bf16 QKV (AVX-512 BF16 and AMX) | call |
| `qwen_matmat_q4_0`, DOTPROD-only fallback (`B x matvec`): `xcol`/`ycol` | Apple silicon with int4 weights at B >= 2 (no i8mm, so no KleidiAI, no SMMLA) | call |
| `qwen_q8r_matmul` at B > 1: `tmp` | Arm with GGUF q8_0 weights, batched | call |
| Apple snake activation: `temp` (vDSP path) | macOS decoder | channel row per chunk |

The VNNI, AVX-512 BF16 and AMX int8 paths use the grow-once `mm_scratch_*` TLS buffers; the
KleidiAI wrappers use their own TLS scratch (`kai_scratch_*`); none of those allocate per call.

## 2. CPU kernel parity map

Read from the dispatchers and the gate table (`g_mm_gate[]`, `qwen_mmk_compiled()`, the
`--caps` / `--dispatch-map` candidate lists). "GEMV" is B = 1 (one stream, one token); "GEMM"
is the batched matmat, B 2..16 (server with concurrency >= 2, and the Talker prefill in chunks
of 16). "twin" is not a GEMM: it is the matvec run over B columns, two at a time.

Columns: **AVX2** (Zen 2/3, older Intel, `SIMD=portable`) · **AVX-512F** (Skylake-X, no VNNI) ·
**VNNI** (AVX-512 VNNI, with BF16 where the host has it: Zen 4/5, Ice Lake+) · **AMX**
(Sapphire/Emerald/Granite Rapids) · **Apple** (NEON + DOTPROD, no i8mm on M1, GCD pool,
Accelerate) · **Arm Linux** (NEON + DOTPROD + i8mm + BF16: Neoverse N2/V2, Graviton 3/4,
KleidiAI built in).

| Op | AVX2 | AVX-512F | VNNI (+BF16) | AMX | Apple | Arm Linux |
|---|---|---|---|---|---|---|
| bf16 GEMV | 2-row fused FMA | 2-row fused, AVX-512 | VDPBF16PS dot (`QWEN_NO_BF16DOT`) | = VNNI (AMX has no GEMV) | NEON 2-row fused; q8_0 repack GEMV for GGUF | **KleidiAI bf16 GEMV** (1x36 dot), BFDOT opt-in, NEON fused |
| bf16 GEMM | twin | twin | dpbf16 m1/m2/m4 kernels, B <= 16; row-major entry for the prefill | AMX bf16 tiles (B >= 4, rows/cols >= 32), dpbf16 below | twin (BFMMLA is `apple_off`, and M1 has no bf16 unit) | **KleidiAI bf16 GEMM** (8x12 mmla), in-house BFMMLA fallback (B <= 64) |
| int8 GEMV | **none**: f32-accumulate fused (dequant on the fly) | none, same | vpdpbusd + cached row sums, u8 activation trick, optional prepack | = VNNI | SDOT `vdotq` | **KleidiAI i8 GEMV** (dotprod 1x4 / 1x8), SDOT fallback |
| int8 GEMM | maddubs (B 2..16) | maddubs (the AVX2 kernel) | vpdpbusd GEMM, in-region row blocks | AMX int8 tiles (B >= 4, rows >= 32, cols >= 64), VNNI below; `QWEN_AMX_B32` experimental | SDOT "loop over B" (matvec class; SMMLA needs i8mm) | **KleidiAI i8 GEMM** (i8mm 4x8 16x4), SMMLA fallback |
| q4_0 GEMV | **none**: generic f32 dequant | none, same | q4 VNNI rows (v3 default, v4 opt-in) | = VNNI | SDOT q4 | **KleidiAI q4 GEMV** (dotprod 1x8), SDOT fallback |
| q4_0 GEMM | maddubs q4 | maddubs q4 | q4 VNNI GEMM (measured 0.80x of int8; known follow-up) | q4 AMX tiles via an int8 unpack stage (B >= 4) | **B x matvec fallback**, with a malloc per call | **KleidiAI q4 GEMM** (i8mm 16x4), SMMLA fallback |
| fused QKV, GEMV | 3 matvecs | 3 matvecs | fused VNNI (shared activation quant + row sums), fused q4 VNNI | = VNNI | fused SDOT int8 and q4 | **KleidiAI fused i8 QKV** (native, default on) |
| fused QKV, GEMM | none (`qwen_matmat_*_qkv` return 0 without VNNI/BF16) | none | fused int8 VNNI matmat, fused bf16 dpbf16 matmat, in-region variant | fused int8 AMX (own `QWEN_AMX_INT8_QKV_MIN_B`), fused bf16 AMX | none: three separate matmats | **KleidiAI fused i8 QKV GEMM**, called directly by the Talker and CP when the batch is contiguous |
| Talker prefill (seq tokens, bf16 weights) | **f32 convert + OpenBLAS SGEMM** (3x the traffic; the predicate says so) | same | dpbf16 rows kernel, chunks of 16 | AMX bf16 tiles, chunks of 16 | Accelerate SGEMM on f32 weights | **KleidiAI bf16 GEMM over the whole sequence** (`QWEN_KAI_NCHUNK` 384) |
| CP prefill pair path `QWEN_CP_PREFILL2` | opt-in | opt-in | default on | default on | opt-in | opt-in |
| decoder int8 conv | none (fp32 im2col + SGEMM) | none | VNNI tiles, **default on** | = VNNI | DOTPROD tiles, **opt-in** (`QWEN_SD_INT8=1`) | DOTPROD tiles, opt-in |
| decoder SGEMM | OpenBLAS, partitioned | OpenBLAS, partitioned | OpenBLAS, partitioned | OpenBLAS, partitioned | Accelerate (see §1) | OpenBLAS, partitioned |
| snake | AVX2 polynomial | AVX2 polynomial | AVX2 polynomial | AVX2 polynomial | vDSP/vvsinf (malloc per row) | NEON polynomial |
| attention over bf16 KV | AVX2 | AVX2 (no AVX-512 variant) | AVX2 | AVX2 | NEON | NEON |
| rms norm | AVX2 | AVX-512 | AVX-512 | AVX-512 | NEON | NEON |

Batched regions and batched heads: VNNI only (§1). Persistent-config and weight prepack:
AMX has both (`QWEN_AMX_PERSIST_CFG`, `QWEN_AMX_PREPACK`, `QWEN_AMX_PREPACK_KINDS`); VNNI has
prepack and a row-sum cache (`QWEN_VNNI_PREPACK`); Arm has KleidiAI's packed RHS for every
registered family (q4, i8, bf16) and the q8_0 repack for GGUF.

### What the map says, per backend

- **Arm Linux with KleidiAI is the reference**: every weight type has a packed GEMV and a
  packed GEMM, fused QKV exists in both classes, the prefill is one GEMM over the whole
  sequence. What it lacks is exactly the x86-only runtime work of §1 (regions, batched heads)
  and the int8 decoder conv default.
- **AVX-512 VNNI (the c8a) is complete on int8/q4/bf16 GEMV+GEMM and fused QKV, and is the
  only backend with the regions.** Its q4 GEMM is the one kernel measured slower than the
  int8 equivalent.
- **AMX is a GEMM-only accelerator on top of the VNNI backend, and that is where the slices
  are missing** (the user's suspicion is right):
  1. no in-region runner: the CP/Talker regions switch themselves off whenever AMX would take
     the projection (B >= 4), so at C4 with B = 4 an AMX worker runs the old 20/112-dispatch
     path;
  2. the batched CP heads are off for the same reason at B >= 4;
  3. the bf16 AMX matmat mallocs its activation block per call (the int8 AMX path does not);
  4. AMX has no GEMV by nature, so B = 1..3 is VNNI — the gate (`min_b` 4, min rows 32, min
     cols 64 for int8) means a 2x8 prefork at C4 sees B = 2 per worker and never touches AMX
     at all in steady state; AMX pays for the prefill (bf16 tiles) and for B >= 4 batches;
  5. `QWEN_AMX_B32` accumulation is a prototype behind an env, not a selected path.
  Everything else (fused QKV int8/bf16 tiles, q4 tiles, prepack, persistent tile config,
  self-test with and without AMX) is there.
- **AVX2 (and AVX-512F without VNNI) has no int8 or q4 GEMV** — a single stream with
  `--int8`/`--int4` on such a host dequantises on the fly in f32, and the prefill converts
  the bf16 weights to f32 and calls SGEMM. Both GEMMs (maddubs) exist. This is the widest
  gap on the x86 side for one-stream users on older hardware; on servers those hosts are
  rare.
- **Apple silicon (M1 class)** has SDOT GEMV for int8/q4 and fused SDOT QKV, but no matrix
  unit reachable: KleidiAI is not built (the Makefile gates it on `__ARM_FEATURE_MATMUL_INT8`,
  which M1 lacks), so int8 GEMM is the SDOT loop over B, q4 GEMM is B matvecs with a malloc
  per call, bf16 GEMM is the twin. The GCD pool has no persistent team, so no region can run
  there even if the kernels existed. The fused GPU Metal path is the answer for batching on
  Apple, not the CPU. M2 and later have i8mm and bf16 and would build KleidiAI, untested.
- **CUDA and Metal** take the batched Talker and CP steps before any CPU region is consulted
  (the GPU checks precede `cp_region_ok`/`tk_region_run`), and use the CPU speech decoder
  unless the CUDA conv decoder is on; the arena and the execution budget apply to that CPU
  decoder unchanged.

## 3. Flags: what the code has, what the docs and the registry have

Census of `getenv("QWEN_*")` across every `.c`, `.m`, `.cu`:

| Set | Count |
|---|---|
| distinct `QWEN_*` names read by `getenv` | 152 |
| names in `g_qwen_reported_flags[]` (the census registry) | 172, plus the 7 added today |
| names documented in `docs/feature-flags.md` | 91 |

- **Registry**: seven names were read by the GPU backends but not declared —
  `QWEN_CUDA_DP4A`, `QWEN_DEC_NAIVE7`, `QWEN_DEC_NAIVET`, `QWEN_METAL_BATCH_MMA`,
  `QWEN_METAL_CP_NOSYNC`, `QWEN_METAL_PROFILE`, `QWEN_METAL_Q4_VEC`. Added.
- **Docs**: 79 names read by the code are not in `feature-flags.md`. Most are diagnostics
  (`*_DEBUG`, `*_JSON`, `QWEN_DUMP_*`, `QWEN_TUNE_*`, `QWEN_VNNI_PHASE_TIMING`) or the
  experimental batch/prefill knobs. The runtime-relevant ones that were missing —
  `QWEN_TK_REGION`, `QWEN_CP_BATCH_HEAD`, `QWEN_SD_SCRATCH_STATS`, `QWEN_SD_THREADS` — are
  documented now. Still undocumented and relevant to an operator on those hosts:
  `QWEN_AMX_PREPACK`, `QWEN_AMX_PREPACK_KINDS`, `QWEN_AMX_PERSIST_CFG`, `QWEN_AMX_B32`,
  `QWEN_NO_VNNI_QKV`, `QWEN_NO_X86_QKV`, `QWEN_Q4_VNNI_V3`/`V4`, `QWEN_CUDA_*`,
  `QWEN_METAL_*`.
- **The 19 names in the docs that no `getenv` reads** (`QWEN_AMX_MIN_B`, `QWEN_VNNI_MIN_B`,
  `QWEN_KLEIDI_MIN_B`, `QWEN_*_NCHUNK`, `QWEN_NO_AMX_*`, `QWEN_NO_SMMLA`, ...) are real: they
  are read through the gate table (`qwen_mm_env_int`), not by a literal `getenv`.

The KleidiAI knob family and its x86 counterparts, from the code:

| Arm / KleidiAI | Purpose | x86 VNNI / BF16 equivalent | AMX equivalent |
|---|---|---|---|
| `QWEN_NO_KLEIDI`, `QWEN_NO_KAI_I8`, `QWEN_NO_KAI_BF16` | kill switches per family | `QWEN_NO_VNNI`, `QWEN_NO_BF16DOT`, `QWEN_NO_BF16_MATMUL`, `QWEN_NO_AVX2MM` | `QWEN_NO_AMX`, `QWEN_NO_AMX_INT8`, `QWEN_NO_AMX_BF16`, `QWEN_NO_AMX_Q4` |
| `QWEN_KAI_OPS` (which ops go to KleidiAI, incl. the prefill) | per-op routing | `QWEN_PREFILL_MATMAT`, `QWEN_NO_VNNI_QKV`, `QWEN_NO_X86_QKV` | `QWEN_AMX_PREPACK_KINDS` (which weight kinds are packed) |
| `QWEN_KAI_NCHUNK` | rows per pool task in the bf16 GEMM | `QWEN_VNNI_NCHUNK`, `QWEN_AVX512_NCHUNK`, `QWEN_X86_NCHUNK` | `QWEN_AMX_NCHUNK` |
| `QWEN_KAI_LHS=sym` | LHS quantisation mode | `QWEN_NO_VNNI_ACT_QUANT` (u8 activation trick) | same code path as VNNI |
| `QWEN_KAI_QKV_FUSED` | fused QKV on/off | `QWEN_NO_VNNI_QKV` | `QWEN_AMX_INT8_QKV_MIN_B` |
| `QWEN_KLEIDI_MIN_B`, `QWEN_SMMLA_MIN_B`, `QWEN_BFMMLA_MIN_B` | crossover B | `QWEN_VNNI_MIN_B`, `QWEN_BF16_MATMUL_MIN_B`, `QWEN_AVX2MM_MIN_B` | `QWEN_AMX_MIN_B`, `QWEN_AMX_INT8_MIN_B`, `QWEN_AMX_BF16_MIN_B`, `QWEN_AMX_MIN_ROWS`, `QWEN_AMX_*_MIN_COLS` |
| packed RHS at registration (`qwen_kleidi_register_*`), `QWEN_NO_Q8REPACK` | weight prepack | `QWEN_VNNI_PREPACK` (+ row-sum cache, `QWEN_NO_VNNI_ROWSUM`) | `QWEN_AMX_PREPACK`, `QWEN_AMX_PERSIST_CFG` |
| — | regions / batched heads | `QWEN_CP_REGION`, `QWEN_TK_REGION`, `QWEN_CP_BATCH_HEAD` | off at B >= 4 (see §1) |

`--caps`, `--self-test` and `--dispatch-map` are the cross-ISA gates. The self-test covers
bf16/int8/q4 GEMV, the three GEMMs against B matvecs, argmax and the int8 conv; it does not
cover the fused QKV matmats, the AMX QKV path, the in-region runners against the dispatched
path, or the partitioned SGEMM wrapper — those were verified by WAV/md5 parity on the c8a only.

### Profile values per machine: why the Axion values do not port verbatim

The active values below are read from the named JSON profiles. They are deployment choices,
not universal defaults. `absent` means the profile's forbidden-env check requires that the
variable not be inherited from the shell.

| control | Axion 16c | x86 C4 VNNI/BF16 starting profile | x86 AMX profile | what it really controls |
|---|---:|---:|---:|---|
| `OPENBLAS_THREAD_TIMEOUT` | `1` | `1` | `1` | park idle BLAS workers; prevents a second team from spinning against the engine pool |
| `OPENBLAS_NUM_THREADS` | absent | absent | absent | the engine sizes BLAS from the worker phase budget; an inherited value invalidates the split |
| `QWEN_PREFIX_CACHE` | `1` | `1` | `1` | reuse the request-independent prompt head; backend-neutral |
| `QWEN_PREFILL_MATMAT` | `1` / KleidiAI BFMMLA | `1` / AVX-512 `VDPBF16PS` | `1` / AMX BF16 tiles, VNNI below the shape gate | selects the BF16 prefill family; the name is common, the kernel is not |
| `QWEN_KAI_NCHUNK` | `384` | absent | absent | Arm/KleidiAI GEMM n-subtiling; it is not an x86 chunk-size equivalent |
| `QWEN_CP_PREFILL2` | absent | `1` | `1` | two-position int8 CP prefill; default/available on AVX-512 VNNI |
| `QWEN_POOL_SPIN` | `65536` | `4096` | `4096` | spin iterations before a pool worker parks; it is not scheduler priority |
| `QWEN_DECODER_BATCH` | `1` | `0` provisional | `0` on the measured 8c host | one decoder pass for several active slots; depends on effective per-worker B |
| `QWEN_STREAM_DECODE_CHUNK` / `_BUSY` | absent / absent | `8` / `0` | `8` / `0` | streaming chunk policy, independent of the BF16 prefill fix |
| `QWEN_VNNI_GEMV_MR` | absent | `2` provisional | absent | x86 VNNI GEMV row microkernel; no effect on AMX/BF16 prefill |
| `QWEN_AMX_MIN_B` | absent | absent | `2` on the measured AMX profile | AMX crossover batch; only meaningful in an AMX build |

The important one is `QWEN_POOL_SPIN`: `65536` helped the 16-core Axion because the measured
idle-window/context-switch trade-off was different. On the 8-core x86 host it was worse than
`4096`, and `0` also hurt because workers immediately parked while work was arriving. The C4
profile therefore starts at `4096`, but its corrected BF16 build still needs a short paired
`0/4096/65536` measurement. A high value is not a general way to reduce scheduler work: it
burns a worker while it spins and can steal a physical core from useful work.

`QWEN_DECODER_BATCH=0` is equally not an x86 law. It won on the narrow 8-core workers because
the decoder gang averaged about 1.4 slots and often had nothing to amortise. A 12-physical-core
C4 worker may cross that point, especially with SMT on, so the profile pins `0` only to keep the
first corrected-build comparison isolated. It must be remeasured before changing it.

All x86 `QWEN_NO_*` kill switches, `QWEN_VNNI_PREPACK`, the VNNI/AVX-512 output chunk knobs,
and `QWEN_PREFILL_QUANT` are intentionally absent in the C4 profile: they either disable the
path being tested, change quality, or are experimental and unqualified. The new C4 JSON also
marks the profile **unqualified**; no old C4 number is silently promoted after the BF16-build
correction.

## 4. Ranked follow-ups (none executed here)

1. **AMX**: in-region runners for AMX tiles (or, cheaper, let the region keep VNNI row blocks
   at B >= 4 and measure which wins on an AMX box); admit AMX to the batched CP heads; move
   the bf16 AMX activation block to `mm_scratch_packb`. Validate on c8i (the AMX reference
   page) with the canonical wave and soak.
2. **Arm Linux**: region runners on KleidiAI i8 GEMM row blocks and the batched heads with
   argmax-parity acceptance; then decide the int8 decoder conv default from a measurement
   (it is opt-in there because the first frame measured slower).
3. **Plain server mode**: give `qwen_tts_serve_ex` the same execution budget as the batched
   server, or state in `server.md` that the budget is a batched-server property.
4. **macOS**: make the SGEMM partition conditional on a serial BLAS, measure once.
5. **AVX2**: int8 and q4 GEMV kernels (maddubs) so `--int8`/`--int4` single-stream users
   on older x86 stop paying the f32 dequant; a bf16-to-f32 prefill is the second cost there.
6. Docs: the AMX and x86 QKV knobs listed in §3.
