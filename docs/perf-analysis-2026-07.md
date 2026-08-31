# Performance analysis — MTP truth, bottlenecks, and the CPU optimization map (2026-07-02)

*Analysis only (no code changed). Companion docs: `docs/hardware-testing.md` (ISA workflow + rented-box
matrix), `docs/gpu-accel-analysis.md` (optional GPU backends), `docs/audit_engine-health-2026-07.md`
(correctness/leak audit).*

---

## 1. The MTP question, answered

**The Code Predictor is NOT speculative decoding and Qwen3-TTS does not use MTP to accept multiple
Talker tokens.** It is purely the **multi-codebook RVQ predictor**: per audio frame it predicts the 15
residual codebooks conditioned on the Talker hidden state + codebook-0, *sequentially*.

Loop structure (`qwen_tts_code_predictor.c:779-883`, `qwen_cp_predict`):
- `:789` CP KV reset per frame (cache is only 64 slots).
- `:797-812` pos 0 = projected talker hidden (+ optional steer) → 5-layer transformer step.
- `:814-827` pos 1 = embed code0 via the **Talker's** codec embedding → step.
- `:829-838` codebook 1 via fused `qwen_argmax_matvec_*` (greedy per-codebook lm_head).
- `:841-872` 14 more autoregressive steps (embed previous code via `cp_codec_emb_bf16[g-1]`, one step,
  fused argmax vs `cp_lm_head[g]`).

⇒ **16 full passes over the 5-layer CP transformer per frame** (2 "prefill" positions + 14 decode steps).

Talker loop (`qwen_tts.c:1510-1677`): one token (= one frame's code0) per iteration; next-step embedding =
codec_embed(code0) + Σ 15 CP codebook embeds + tts_pad (`:1630-1640`) → `qwen_talker_step` (`:1661`).

### Why Talker speculative decoding is blocked (and low-ceiling anyway)
1. **The next Talker input depends on ALL 16 codebooks of the previous frame** (`qwen_tts.c:1630-1640`).
   Drafting Talker token N+1 forces drafting the CP's 15 codes for frame N → speculative CP, which is
   already abandoned for cause: the 16 CP steps are sequentially conditioned (codebook feedback loop),
   and quant-ladder measured int8-CP agreeing with gold only 78% (int4 collapses on late codebooks) — a
   cheap draft would rarely be accepted.
2. Even perfect Talker speculation only touches the Talker share: ~26% (0.6B) / ~35% (1.7B-int8). CP
   dominates and cannot overlap the Talker (hard sequential dependency).
3. The verify primitive exists if anyone revisits (self-speculation: same Talker at int4/q2 or
   layer-skip draft, verify K frames in one `qwen_matmat_bf16` pass — annotated as "the
   spec-decode-verify primitive" at `qwen_tts_kernels.c:802-822`) — but blocker (1) stands.

### What actually reduces CP cost
(a) int8 CP — shipped; (b) **cross-request batching** — shipped (`qwen_batch_cp_predict`,
code_predictor.c:965-1016: B frames in lockstep, weights read once per step for all B — the legitimate
"batch the 16 passes" answer; within one stream they cannot be batched); (c) contextual sparsity in the
CP FFN — probe exists (`QWEN_FFN_SPARSITY`, :704-710), skip logic not built; (d) hybrid per-tensor
precision — env knobs exist (`QWEN_CP_LMHEAD_PREC`/`QWEN_CP_LAYER_PREC`, :586-604).

---

## 2. Cost split + bandwidth math (verified against code)

- 0.6B bf16 (M1, 4T): **CP 86.6 ms/f (74%) · Talker 30.6 ms/f (26%)** · sampling 0.35 ms · prefill 1.65 s.
- CP internal: **90.7% matvec** (FFN gate_up 34% + down 19%; QKV 24.5% + O-proj 13.3%; lm_head 4.3%);
  attention 0.7%.
- Bytes: per CP layer bf16 ≈ 31 MB → ×5 layers ≈ **155 MB re-read on each of the 16 passes** ≈ 2–2.5
  GB/frame → ~25–29 GB/s at 86.6 ms/f = **at M1's practical DRAM ceiling**.
- Hence: int8 CP (half the bytes) ⇒ −24% RTF on 0.6B; int4-on-CP **slower on M1** (nibble-unpack >
  bandwidth saved) but the win on x86 (Ryzen 2.81→2.02).
- 1.7B `--int8`: Talker 65.8→49.7 ms/f, CP ~64→59.2 ⇒ split ≈45/55. **CP is the bottleneck on 0.6B,
  co-equal on 1.7B-int8.**

**Governing law**: the hot path is DRAM-bandwidth-bound. The levers that move RTF are **byte-count**
(quant, hybrid precision, sparsity) and **byte-reuse** (cross-request batching; true matmul units on
newer ISAs), not more single-stream SIMD.

---

## 3. Hot-loop inventory (what exists / what's missing)

| Kernel | Paths today | Missing / opportunity |
|---|---|---|
| `qwen_matvec_bf16` (+fused) `kernels.c:558-800` | AVX-512 2-row, NEON 2-row 8-acc + prefetch, AVX2, scalar; threaded rows≥256 | **No BFDOT/BFMMLA** (M2+/Graviton3/Grace — flagged AVAILABLE-UNUSED :121); no AVX-512-BF16 VDPBF16PS |
| `qwen_matvec_int8` `:1719-1768` | NEON **SDOT** w/ dynamic act-quant; AVX-512 **VNNI** (written, unvalidated on HW); AVX2 widen-FMA; scalar | **No i8mm SMMLA** twin; `_qkv` variant re-quantizes the activation 3× (:1770-1781) |
| `qwen_matvec_q4_0` `:1883-1999` | NEON/AVX2 nibble-unpack→f32 FMA | **No SDOT-q4 path** (unpack block→int8 once, then `vdotq`, llama.cpp-style) — the kernel that would make int4 viable on ARM |
| `qwen_matmat_*` `:802-1233` | fixed-B (1-8,16) register-blocked; weight-stationary (1.74× on M1 bf16) | right primitive for SMMLA/VNNI/AMX true GEMM (TODO at :816-822); SDOT-matmat opt-in measured slower on M1 |
| attention `:2107-2499` | NEON/AVX2 online-softmax, inline bf16→f32 K/V; single-threaded per token | fine (0.7% of CP); per-head parallelism only if kv_len ≫ 1k |
| rms_norm `:237-448` | NEON+AVX2, fused residual | done |
| swiglu `:2510-2531` | vvexpf (Apple); **scalar expf elsewhere** | NEON/AVX exp approx = easy off-Mac win |
| Talker prefill `talker.c:627-877` | BLAS sgemm on f32, **bf16→f32 conversion of every layer per prefill call** (:698-703) | dominates 1.65 s prefill = TTFA floor; route through `qwen_matmat_bf16` instead |
| speech decoder | f32 + BLAS; naive scalar depthwise k=7 + per-t LayerNorm (:1237-1265); snake w/ scalar sinf off-Mac | hidden by pipeline overlap, but competes for DRAM with Talker/CP |

Threading: `qwen_parallel` = GCD on macOS (reentrant) / persistent pthread pool on Linux (single job slot,
NOT reentrant → server serializes) / Win32 pool. Thread count hardcoded `min(ncpus,4)` (`kernels.c:73-80`)
— right for M1 matvec, wrong for 16-core servers on compute-bound batched matmat.

---

## 4. Memory layout / struct / cache findings (analysis 2c)

- **Talker KV**: bf16 `[layers][kv_max][kv_dim]` (qwen_tts.h:476-479), grow-by-doubling per-layer memcpy.
  int8 KV **not worth it** (attention ≈1–3% of frame). CP KV = 640 KB, L2-resident, non-issue.
- **Weights**: mmapped bf16 row-major, streamed with 2-row blocking + prefetch — already the right GEMV
  layout. The missing re-layouts are (a) q4→SDOT-friendly int8-interleave and (b) blocked/tiled formats
  for future SMMLA/AMX.
- **`q4_0_block_t`** = f32 scale + 16B nibbles (20 B/32w). **fp16 scale ⇒ −10% q4 bytes** — direct gain on
  bandwidth-bound q4 paths (x86).
- **Batched activations**: kept `[B][dim]` but matmat wants `[dim][B]` → explicit gather/scatter transpose
  per projection (`talker.c:888-951`) ≈ 336 transposes per batched Talker step. Keep `[dim][B]` native.
- **Alignment**: work buffers 64B-aligned via aligned_malloc — OK. False sharing: matvec workers write
  disjoint contiguous row ranges — negligible. AoS/SoA: everything hot is already flat SoA; the ctx struct
  is pointers-only, no meaningful padding waste. **No struct-packing low-hanging fruit** — the wins here
  are the re-layouts above, not field reordering.
- **Streaming decoder state**: pre-transformer KV is f32 and grows unboundedly (speech_decoder.c:1554-1562)
  though the attention window is 72 → ring-buffer it; latent_cache keeps everything but needs only
  `conv_rf+chunk` frames (:1776-1806).

---

## 5. Server + continuous batching — where the throughput goes

Architecture: `--serve --workers N` = acceptor + N workers on cloned ctxs (mmapped weights shared; truly
parallel on macOS/GCD, bandwidth-multiplied); `--batch-size N` = reader pool → one scheduler thread owns
ctx → ragged batched Talker step + batched CP via weight-stationary `qwen_matmat_*` (per-slot KV/RNG/
sampling; single-stream bit-reproducibility preserved; streaming per-frame per-slot).

Concrete limiters found:
1. **Admission prefill blocks the whole batch** (qwen_tts.c:2209-2225): a new request's prefill runs
   single-stream inside the frame loop → all slots stall ~1–2 s per admission.
2. **Per-slot codec head = B separate matvecs** (qwen_tts.c:2245; also :1880, :2061) — should be one
   `qwen_matmat_bf16` (3072×h × B) like everything else.
3. **Inactive slots still processed** by batched matvecs (compaction TODO, talker.c:1024).
4. **Per-slot streaming decode runs synchronously in the scheduler loop** (qwen_tts.c:2284-2292) — one
   streaming client stalls generation for all slots.
5. Batched driver is **bf16-only** (returns -2 otherwise, qwen_tts.c:2133) though precision-aware
   `qwen_batch_proj_q` kernels exist — stale gate.
6. Batched int8 loses on M1 because the batched int8 twin dequants to f32 and loses SDOT
   (kernels.c:1091-1099) — fixed properly by an SMMLA/VNNI matmat, not by tweaking the current twin.

## 6. Streaming / TTFA

Decoder-thread overlap is always on; stream mode ramps the first chunk to 2 frames.
**TTFA = prefill (~1.65 s, 0.6B bf16 M1) + 2 frames gen (~0.24 s) + 2-frame decode** → prefill dominates.
Levers: (a) delta-prefill (server, shipped), (b) prefill without the bf16→f32 conversion (route through
`qwen_matmat_bf16`; est. −30–50% prefill), (c) int8.
Streaming decoder re-runs the conv stack over `conv_rf(=20)+new` frames and discards the context audio
(speech_decoder.c:1809-1848): 3× redundant conv work at chunk=10, 11× at the 2-frame ramp. A true stateful
streaming conv (per-layer FIFO tails) removes it — but since the decoder is overlapped, the visible gain
is bandwidth-contention relief + smaller chunks at no RTF cost, not headline RTF (M/L effort).

---

## 7. Ranked optimization list

| # | What | Where | Est. gain | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Non-blocking admission prefill in continuous batching | qwen_tts.c:2199-2235 | p95 −1–2 s/admission; +10–30% throughput under churn | M | Med |
| 2 | Batch the codec head (B matvecs → 1 matmat) + per-slot logit loops | qwen_tts.c:2245,1880,2061 | 3–6% batched frame | S | Low |
| 3 | **SDOT-native q4_0 matvec** (unpack→int8 once per block, vdotq) | new twin near kernels.c:1883 | int4 ≈int8-speed at half bytes on M1; −20–30% CP on ambient ARM | M | Med |
| 4 | CP FFN contextual sparsity (probe exists → skip logic) | code_predictor.c:704-710 | 5–10% CP if sparsity 30–50% | M | Med |
| 5 | BFDOT/BFMMLA bf16 matvec+matmat (M2+/Graviton3/Grace) | kernels.c TODO :816-822 | up to ~2× bf16 where compute-bound; biggest for matmat | M-L | Low (HW gate) |
| 6 | i8mm SMMLA int8 matmat (M2+/Graviton3) | same | batched int8 stops losing to bf16; +30–50% server int8 | M-L | Low-Med |
| 7 | Fuse act-quant in `matvec_int8_qkv` (quantize once) | kernels.c:1770-1781 | <1% | S | None |
| 8 | Scratch buffer for `argmax_matvec_q4_0` (8 KB malloc ×15/frame) | kernels.c:1830-1840 | noise | S | None |
| 9 | **Prefill via `qwen_matmat_bf16`** (kill bf16→f32 convert + sgemm) | talker.c:694-842 | prefill −30–50% ⇒ TTFA | M | Med (golden-gate) |
| 10 | Keep batched activations `[dim][B]` native (kill 336 transposes/step) | talker.c:888-951 | 2–5% batched step | M | Low |
| 11 | Compact active slots before batched matvecs | talker.c:1024-1105 | up to (B−active)/B | M | Low |
| 12 | Per-slot streaming decode off the scheduler thread | qwen_tts.c:2284-2292 | removes all-slot stalls | M | Med |
| 13 | x86: validate VNNI path on HW + add AVX-512-BF16 VDPBF16PS | kernels.c:1616-1694 | int8 decode −20–40% on Zen4/SPR | S/M | Low |
| 14 | AMX int8 tile GEMM for server batching (SPR+) | new; hooks via qwen_batch_proj_q | server throughput | L | Med |
| 15 | fp16 scale in q4_0_block_t (20→18 B) | kernels.h + quantize | ~5% on q4 paths | S | Low |
| 16 | Ring-buffer streaming pre-transformer KV (window 72) + trim latent_cache | speech_decoder.c:1554,1776 | memory cap (long streams / many slots) | S | Low |
| 17 | True stateful streaming conv decoder (FIFO tails) | speech_decoder.c:1809-1848 | decoder work −60% @chunk10; enables tiny chunks → TTFA | L | Med |
| 18 | NEON/AVX exp approx for swiglu+softmax off-Mac | kernels.c:2510 | few % on Linux | S | Low |
| 19 | Hybrid CP precision flag (int8 attn + q4/q2 FFN; knobs exist) | code_predictor.c:332-425 | CP −10–20% on bandwidth-bound boxes | S | Med (quality-gate) |
| 20 | Adaptive thread count per kernel class (matvec 4 vs matmat N) | kernels.c:73-80 | up to 1.5× batched on ≥8-core | S-M | Low |

**Already tried — do not re-litigate** (PLAN history): Metal GPU single-stream (1.3× slower — see
gpu-accel-analysis for the *right* framing), 4-row fused matvec (register spill), pthread-on-Mac,
speculative CP, CP↔Talker overlap, softmax/SiLU micro-SIMD, batch text-embedding sgemm, SDOT-matmat
default-on (slower on M1).
