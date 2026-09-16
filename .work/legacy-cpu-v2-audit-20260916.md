# Legacy CPU and v2 serving architecture audit

Task · Audit the current v2 CPU streaming/server design for graceful performance on older/common x86 and ARM CPUs, and produce an evidence-based implementation plan without changing execution code.

Question · Does the current dispatch, quantization, packing, GEMV/GEMM selection and v2 scheduler degrade usefully from AMX/VNNI/KleidiAI to AVX2, AVX-512 without VNNI, ARM dotprod and plain NEON?

Known facts · The repository contains separate native integer GEMV and batched paths for VNNI, ARM dotprod and KleidiAI; AVX2 has an integer-emulation-style batched path but its GEMV path is widen/dequant plus FMA. The current v2 server chooses GEMV for one active slot and matmat for multiple active slots, but the crossover is a gate table plus environment overrides, not a measured per-backend cost model.

Unknowns · No Ryzen 7 6800H or AVX-512-without-VNNI host was available in this audit, so all claims about their absolute speed, frequency effects and bandwidth are code-derived hypotheses until measured. No end-to-end legacy-ISA baseline is promoted by this document.

Files/functions inspected · `ENGINEERING.md`, `docs/BENCHMARKING.md`, `docs/backend-matrix.md`, `docs/batching.md`, `Makefile`; `qwen_tts_dispatch.c`; `qwen_tts_kernels.c`; `qwen_tts_q8repack.c`; `qwen_tts_kleidi.c/.h`; `qwen_tts_gguf.c`; `qwen_tts_talker.c`; `qwen_tts_code_predictor.c`; `qwen_tts_speech_decoder.c`; `qwen_tts_sd_gemm.c`; `qwen_tts.c`; `qwen_tts_server.c`; `qwen_tts_thread.c/.h`; `tests/roof_matvec_int8.c`; `tests/membw.c`; `tests/prefill_bench.c`; `tests/kernel_census_bench.c`; `tests/matmat_parity.c`; `tests/decode_quantum_bench.c`; `tests/batching_bench.c`; `tests/x86_b1_gemv_bench.c`; `tests/x86_qkv_bench.c`; `tests/onednn_w8a8_bench.c`.

Evidence · Audit performed at HEAD `15a58501aab8a2a02795bd6deb420672fab7f5dd` (`feature/x86-amx-vnni-oss`), clean tree. `make blas` rebuilt the current source on the local Apple ARM host; `./qwen_tts --caps` and `./qwen_tts --dispatch-map` reported `build/src=15a5850:clean`; `./qwen_tts --self-test` passed all 0 failure cases. A pre-existing binary printed `b7b916b-dirty`; it was discarded as evidence and is not used below.

Conclusion · The portability gap is not “all old CPUs fall to scalar”. The important cliffs are more specific: AVX2/AVX-512F without VNNI lack native integer GEMV and decoder INT8; AVX2 batched INT8/Q4 is a useful PMADDUBSW/PMADDWD-style path but has a signedness/saturation proof obligation; dotprod-only ARM has strong GEMV but no default SDOT matmat; and the v2 scheduler has no backend performance profile that feeds its batch, thread or prefill policy. The first implementation work should therefore be dispatch/profile correctness and measurement, followed by one native legacy GEMV at a time. Do not start by changing the server scheduler.

Next action · Establish clean M1, Ryzen AVX2, AVX-512-no-VNNI, VNNI/AMX and modern ARM manifests; then execute the work items in §11 in order, keeping each kernel and server change independently benchmarkable.

## 1. Current execution architecture

### Model load and representations

1. `qwen_tts_load_ex()` in `qwen_tts.c` loads configuration and safetensors, then invokes the Talker, Code Predictor and speech-decoder loaders. `qwen_talker_load()` (`qwen_tts_talker.c:480`) loads BF16 source weights, f32 norms, and constructs a fused BF16 gate/up matrix per Talker layer (`:534-543`).
2. Talker and CP quantization is load-time work. `qwen_talker_load()` selects INT8, Q4 or mixed precision from the context and `QWEN_TALKER_PREC`, then creates per-row INT8 scales or Q4_0 block weights. The original BF16 tensors remain available.
3. `qwen_tts_gguf.c` additionally registers Q8_0 tensors with `qwen_q8r_register()` when the Arm Q8 repacker is supported. This is a persistent layout cache keyed by the BF16/weight pointer; it is not a general x86 Q8 backend.
4. The speech decoder loader (`qwen_speech_decoder_load()`, `qwen_tts_speech_decoder.c:1143`) loads f32 decoder tensors, repacks ConvTranspose weights once (`:1312-1347`), and builds optional persistent AMX decoder packs. Its normal control path remains f32/BLAS or the explicitly enabled INT8 path.

### Capability and dispatch

`isa_class()` in `qwen_tts_dispatch.c:81-112` identifies x86 as AMX, AVX-512 BF16, AVX-512 VNNI, AVX2 or portable; ARM is Apple i8mm+BF16, Apple M1-class, Linux i8mm+BF16, dotprod or portable. This is a coarse reporting class, not a performance model.

The build selects ISA flags in `Makefile:6-44`. Relevant profiles are:

- `SIMD=portable`: `-mavx2 -mfma` on x86. Despite the name, this is the AVX2 baseline, not scalar.
- `SIMD=avx512`: AVX-512F/BW/VL plus AVX2/FMA, but deliberately no VNNI/DQ/BF16.
- `SIMD=avx512vnni`: AVX-512F/BW/VL/DQ/VNNI plus AVX2/FMA.
- `SIMD=avx512bf16`: VNNI plus AVX-512 BF16.
- `SIMD=amx`: AMX tile/INT8/BF16 plus the AVX-512 and VNNI flags.
- ARM KleidiAI sources are added only when the compiler defines `__ARM_FEATURE_MATMUL_INT8` (`Makefile:91-120`).

The actual gate table is `g_mm_gate[]` in `qwen_tts_kernels.c:2318-2342`; `qwen_mm_use()` (`:2401-2448`) combines compiled capability, runtime support, environment policy, B limits and shape floors. Gate state is cached on first use. Changing an environment variable after a kernel family has been resolved is therefore not a valid A/B in the same process.

Candidate priority in `qwen_matmat_int8()` (`qwen_tts_kernels.c:4793-4970`) is KleidiAI, AMX, VNNI, AVX2, ARM SMMLA/SDOT, then fixed-B/generic fallback. `qwen_matvec_int8()` (`:6802-6875`) is KleidiAI, VNNI, ARM SDOT, then `int8_matvec_fused()`.

The rebuilt local M1 map is an important current proof point:

- native INT8/Q4 GEMV is SDOT;
- B>1 INT8 matmat is the f32-accum twin because `QWEN_INT8_SDOT_MM` is opt-in and unset;
- Q4 B>1 is B×matvec;
- no KAI/i8mm or BF16 matrix unit is compiled;
- Talker prefill resolves to BF16→f32 conversion plus Accelerate SGEMM;
- decoder INT8, CP prefill2 and regions are selectable or available only behind their policy gates, not default-on on this host.

### Talker, CP and decoder execution

- `qwen_talker_step()` (`qwen_tts_talker.c:678-850`) is the B=1 decode path. Every layer performs norm, QKV, attention/KV update, output projection, gate/up, SwiGLU and down projection through the dtype-specific GEMV functions.
- `qwen_talker_prefill()` (`qwen_tts_talker.c:1408`) resolves native BF16 matmat once per thread. Without a native BF16 unit in a BLAS build, it converts weights/activations for SGEMM. Reusable `pp_xT`/`pp_yT` scratch grows on demand; it is not allocated once per projection.
- `qwen_talker_prefill_range()` (`qwen_tts_talker.c:1961`) is the range/resumable prefill entry used by the v2 admission path. The existing prefill-slice experiments do not change the ISA kernel choice.
- `qwen_batch_proj()` (`qwen_tts_talker.c:2237`) uses per-slot GEMV for B=1 and gathers `[cols][B]`, calls matmat, then scatters for B>1. `qwen_batch_proj_q()` (`:2271`) adds the contiguous KAI fast path; otherwise it has the same gather/matmat/scatter contract. `qwen_batch_proj_qkv()` (`:2326`) can share one activation pack for Q/K/V on supported fused paths.
- `qwen_batch_talker_step_ragged()` (`:2798`) falls back to a single-slot Talker step when one stream is runnable and uses the batched path only when active slots can be aggregated. Thus server concurrency is not the same thing as kernel B.
- `qwen_cp_predict()` (`qwen_tts_code_predictor.c:838`) emits the first code and then sequentially predicts codebooks 1..14 (`:912-946`): embeddings/projection, a CP transformer step, norm and an LM-head argmax per codebook. `qwen_batch_cp_predict()` (`:1533`) uses a single-slot fallback when only one slot is active, then batches MTP/LM-head work where shapes and weights permit.
- `qwen_speech_decoder.c` uses f32 im2col/BLAS by default on non-VNNI/non-dotprod builds. INT8 decoder availability is explicitly limited by `qwen_sd_int8_available()` (`qwen_tts_kernels.c:8636-8649`) to ARM dotprod or AVX-512 VNNI. This is a serving-level cliff, not just a missing micro-optimization.

### v2 server and worker allocation

`qwen_tts_serve_continuous()` (`qwen_tts.c:2959`) owns the active-slot loop. It maintains active masks, playback lead masks, TTFA priority masks, optional width masks, Talker/CP state, decoder state and admission/prefill state. The active-mask logic (`:3875-3922`) can reduce effective B before a kernel is called.

The decoder may run inline or on a private lane (`:3076-3147`). The lane has one mailbox per slot, optional multislot cohorts, nonblocking modes, and a private clone. The engine pool and decoder lane are separate scheduling domains, but the overall CPU budget is still shared unless the deployment topology isolates them.

Decoder enqueue/cohort and completion handling are in the same continuous loop (`qwen_tts.c:4040-4210`). This is relevant to legacy CPUs: a slower GEMV/GEMM backend changes the time spent in each turn and can make an otherwise reasonable cohort or admission policy unsafe.

The Linux prefork path (`qwen_tts_server.c:2758-2945`) plans on the inherited CPU mask, detects physical cores/SMT, creates worker masks, sets threads after fork and can split decoder lanes. The plain batched server (`:2414-2480`) also warns when requested batch exceeds the resolved INT8 batch ceiling. The policy is operationally explicit, but kernel characteristics are not represented as a first-class backend profile consumed by the scheduler.

## 2. ISA/kernel census

Legend: **optimized** means an ISA-specific path exists for the relevant operation; **generic SIMD** means vectorized but not an integer/matrix-unit kernel; **twin** means the fixed-B software fallback; **missing** means no dedicated path was found; **policy-gated** means present but not default or not reachable for the target.

| Operation / serving role | AVX2 + FMA | AVX-512 without VNNI | AVX-512 VNNI | AMX | plain NEON | NEON + dotprod | i8mm / KleidiAI |
|---|---|---|---|---|---|---|---|
| INT8 GEMV | **generic SIMD** `int8_matvec_fused()` (`qwen_tts_kernels.c:5892`, AVX2 widen/load + FMA); no integer dot | same AVX2 body; **no AVX-512-BW dot GEMV** | **optimized** native VNNI path in `qwen_matvec_int8()` | VNNI/native GEMV or generic fallback; AMX is not a B=1 GEMV path | **generic SIMD** NEON widen + FMA | **optimized** SDOT `int8_matvec_sdot()` | KAI native GEMV when compiled/supported; otherwise SMMLA or fallback |
| INT8 GEMM / B>1 matmat | **optimized integer emulation** `int8_matmat_avx2_slice()` (`:4628-4674`), gated B2..16 | falls to the AVX2 gate/body; no dedicated AVX-512-BW family | optimized VNNI row/tile families, B2..16 | AMX tile family for B>=3 and rows/cols floors; VNNI below shape/work gates | fixed-B/twin; no integer dot | SDOT loop over B exists but `QWEN_INT8_SDOT_MM` is opt-in | KAI persistent RHS/LHS pack and dotprod/i8mm GEMM; KAI is first candidate |
| Q4 GEMV | generic AVX2 unpack/dequant + FMA (`:7044-7130`) | same AVX2/generic body | VNNI activation-quantized path | VNNI or generic for B1; AMX Q4 is matmat | generic NEON unpack/dequant + FMA | SDOT Q4 path | KAI Q4 native GEMV/GEMM when available |
| Q4 GEMM | AVX2 `maddubs`/`madd` path (`:5375-5412`) with correction terms | falls to AVX2 path | VNNI Q4 path | AMX Q4 for gated shapes | fixed-B generic twin | B×matvec/SDOT-oriented path; no default matrix-unit Q4 | KAI Q4 persistent packed RHS |
| BF16 GEMV | AVX2 BF16-to-f32 loads plus FMA (`:1510-1560`) | AVX-512F BF16 loads plus FMA, not BF16 dot | same unless AVX-512 BF16 is compiled | AMX is not the normal B1 path | NEON BF16-to-f32 + FMA | BFDOT only if BF16 extension, not dotprod alone | KAI BF16 if BF16+i8mm build/CPU |
| BF16 GEMM | fixed-B software twin; no BF16 dot | fixed-B software twin | fixed-B twin unless AVX-512 BF16 | AMX BF16 for B>=4 and shape floors | fixed-B twin | fixed-B twin unless BF16/BFMMLA | KAI BF16 BFMMLA family when supported |
| Q8_0 repack | missing as a first-class x86 family | missing | missing | missing | not applicable | **optimized/policy-gated** 4-row Q8 repack + SDOT (`qwen_tts_q8repack.c:246-335`) | Q8 repack uses SDOT for GEMV and i8mm for B>1 |
| Talker prefill | f32 conversion + BLAS SGEMM by default; optional INT8 prefill is separately gated | same f32/SGEMM fallback; AVX-512F alone is not BF16 matmul | f32/SGEMM unless AVX-512 BF16/AMX BF16 path is compiled and resolved | native BF16/AMX path when shape/gate allows | f32 conversion + platform BLAS | same, with optional SDOT/int8 policies | KAI prepared BF16/int8 path where weight/shape/flags allow |
| Talker decode | B1 GEMV generic SIMD; B>1 AVX2 matmat when active batch exists | B1 generic SIMD; B>1 AVX2 matmat | native VNNI GEMV/matmat; regions available | VNNI at small B, AMX at sufficiently large work; region/packed paths | NEON generic GEMV/twin | SDOT GEMV; B>1 matmat needs opt-in SDOT or twin | KAI native GEMV/GEMM/fused QKV/regions |
| Code Predictor | same dtype dispatch; CP prefill2/regions not VNNI | same legacy fallback; no special AVX512-no-VNNI CP path | VNNI CP batched projections/heads/regions when gates hold | AMX only for adequately large row-work/B; otherwise VNNI | NEON generic paths | SDOT B1; B>1 depends on SDOT matmat gate | KAI region and packed paths when all CP weight families qualify |
| Speech decoder | f32 im2col + SGEMM control path; no decoder INT8 | same; no decoder INT8 availability | decoder INT8 convolution is available and default policy can enable it | optional AMX Design-D/BF16 paths, with VNNI/f32 fallbacks | f32/BLAS control path | decoder INT8 available but policy-gated | KAI BF16/int8 prepared paths where current shape contract allows |

### Census qualifications

1. The AVX2 INT8 matmat path is not scalar. It quantizes activation columns, treats absolute INT8 weights as unsigned, sign-adjusts activations, and uses `_mm256_maddubs_epi16` followed by `_mm256_madd_epi16` and horizontal reduction. This is the expected class of AVX2 dot-product emulation, but `maddubs` has saturating 16-bit intermediate semantics. Any future modification must prove that the selected packing/ranges cannot saturate or replace it with a non-saturating widening sequence. The existing self-test checks output parity, not a proof over all accumulation ranges.
2. AVX-512F/BW compilation does provide AVX-512 vector helpers for quantization and BF16 conversion, but the INT8/Q4 matrix family is selected only under `__AVX2__`, `__AVX512VNNI__`, AMX or ARM macros. Therefore AVX-512 without VNNI currently inherits AVX2 INT8/Q4 matmat rather than using a 512-bit BW-specific dot implementation.
3. KAI source contains both dotprod and i8mm ukernels, but `QWEN_KLEIDI_BUILD` in `qwen_tts_kleidi.c:23-47` requires both `__ARM_FEATURE_MATMUL_INT8` and `__ARM_FEATURE_DOTPROD`; the runtime check also requires dotprod and i8mm (`:65-90`). A dotprod-only ARM box therefore uses the in-house SDOT path and cannot use the KAI dotprod GEMV family today.
4. `qwen_q8r_matmul()` is reached by BF16-shaped call sites and GGUF registration, but support is Arm NEON + dotprod/i8mm only. It is not evidence of a general Q8 path on x86.
5. The architecture-specific kernel translation units `qwen_tts_kernels_generic.c`, `qwen_tts_kernels_neon.c` and `qwen_tts_kernels_avx.c` are one-line stubs; the implementation is concentrated in the 13k-line `qwen_tts_kernels.c`. This does not by itself cause a performance bug, but it makes ISA ownership and regression review harder.

## 3. Packing, layout and quantization audit

### Weight lifetime

- Safetensors source is BF16 for Talker/CP; INT8 is per-output-row with one f32 scale, and Q4_0 uses 32-value blocks with an f16 scale. `qwen_talker_load()` retains source BF16 and creates quantized copies.
- KAI registers persistent RHS packs keyed by source pointer (`qwen_tts_kleidi.c:108-125` and `:389-410`). x86 VNNI/AMX packs are optional caches in `qwen_tts_kernels.c` and are deliberately gated because the extra resident bytes and pack cost can outweigh a low-B serving win.
- Q8 repack (`qwen_tts_q8repack.c:81-115`) rearranges four rows while preserving Q8_0 data. It is a layout change, not quantization.

### Activation and output layout

The normal ragged batch is slot-major FP32. For B>1, `qwen_batch_proj_q()` gathers it to `[cols][B]`, dynamically quantizes activation columns for INT8/Q4 paths, runs the selected kernel, then scatters to slot-major output. For non-contiguous active indices the gather/scatter is unavoidable in the current API. The contiguous KAI path can bypass this staging for registered weights, but it is only a KAI case.

QKV has a valuable reuse point: `qwen_batch_proj_qkv()` shares one gathered activation/quantized panel across Q, K and V when the fused family is selected. The generic path can still perform three projection calls and the batched region path has its own preparation contract. A future legacy backend should preserve this sharing rather than adding three independent GEMV repacks.

BF16 conversion is vectorized for NEON and AVX2 in `qwen_bf16_to_f32_vec()` (`qwen_tts_kernels.c:8615-8634`) and related load helpers, with scalar tails. No F16C-specific path was found; the model’s common low-precision storage is BF16, not FP16 arithmetic.

### Repacking risks

The likely legacy cost is not a per-request malloc on the steady Talker/CP projection path: batch scratch and KAI scratch are grow-once thread-local buffers. The more important repeated work is data movement:

1. slot-major → `[cols][B]` gather;
2. FP32 activation quantization and scale calculation;
3. optional Q8/AMX/KAI LHS pack;
4. output scaling and `[rows][B]` → slot-major scatter.

This work is shape- and backend-dependent. A backend profile must measure it separately from the dot kernel. A “GEMM faster than GEMV” claim that excludes these phases is not sufficient for v2 serving.

## 4. GEMV versus GEMM behavior

The current crossover is structural:

- B=1 or `QWEN_BATCH_FORCE_MATVEC`/`QWEN_BATCH_NOMATMUL` → one GEMV per slot.
- B>1 → gather and matmat, subject to the kernel gate.
- Most x86 INT8/Q4 gates accept B2..16; AMX INT8 starts at B3 with rows/cols and rows-per-thread floors; AMX BF16/Q4 start at B4 with 32-sized shape floors; KAI has a wider B range but still has shape predicates.
- Active masks, playback lead and `force_matvec` can make an advertised server C level execute at lower effective B. The server warning reports only the configured batch ceiling, not the per-stage effective B distribution.

There is no measured cost model that asks “for this backend, shape, thread count and B, does gather + quantize + matmat beat B GEMVs?” The current gate table is a safe capability/shape filter, not a crossover model. This is most visible on the local M1: SDOT is native GEMV, but B>1 INT8 matmat is an f32-accum twin by default. Existing `docs/batching.md` records historical M1 measurements where the INT8 batched twin was 0.70–1.18x relative to sequential SDOT for representative shapes, while BF16/int4 behavior differed. That is direct evidence that a global “B>1 means matmat” rule is not enough.

Likely architecture-specific consequences, to validate rather than assume:

| backend | likely first crossover question |
|---|---|
| AVX2 | Does PMADDUBSW batched matmat amortize one weight read before gather/activation quantization dominates? Is GEMV’s f32 dequant/FMA cheaper at B2? |
| AVX-512 no VNNI | Does wider BF16/FP32 conversion help enough to offset the same AVX2 INT8 matrix fallback? Is AVX2 frequency retention better? |
| VNNI | At which B do row sums, activation quantization and VNNI tile shapes amortize? B1 native GEMV remains a first-class path. |
| AMX | Does B and rows-per-thread clear tile setup and activation-pack costs? Small B should stay VNNI/GEMV. |
| dotprod | Does SDOT B1 remain better until a larger B because the current SDOT matmat path is only a loop over B? |
| KAI/i8mm | When does persistent RHS + LHS pack beat one-slot KAI GEMV and what shapes need the i8mm 4x8 family? |

Acceptance for any crossover change must be paired kernel timing plus end-to-end TTFA/STREAM/stall data. A microbench-only crossover is insufficient.

## 5. v2 server integration audit

The v2 design has the right conceptual separation—admission, Talker/CP execution, decoder lane and playback lead—but the backend contract is still mostly a set of flags and fixed gates:

- worker count and thread masks come from deployment topology and prefork parameters;
- `qwen_batch_talker_step_ragged()` chooses single versus batched execution based on runnable active slots;
- decoder lane settings are separate from the engine pool but consume the same machine budget;
- prefill policy is a global path decision (`qwen_prefill_matmat_resolved()`), not a per-backend measured service budget;
- queue/lead/TTFA policies do not receive a predicted per-frame cost from the selected GEMV/GEMM family;
- the scheduler has no stable object equivalent to “backend capabilities → cost characteristics → worker/batch policy”.

On an AVX2 or dotprod host, blindly using the current VNNI/Arm-v2 all-on deployment profile can be wrong in either direction: too much batching may spend more time staging than computing, while forcing GEMV may reread weights for every active slot. The correct integration point is a small resolved backend performance profile, not ISA conditionals scattered through the scheduler.

Candidate profile fields, to be added only after measurement:

```text
backend_class
native_int8_gemv / native_q4_gemv
int8_matmat_family / q4_matmat_family / bf16_matmat_family
measured crossover B by representative shape family
max efficient B and minimum rows/cols
activation-pack and gather/scatter cost bands
preferred worker threads for GEMV, matmat and decoder
prefill mode and bounded quantum recommendation
SMT preference and physical-core requirement
```

The scheduler should consume the profile for batch width, worker/thread defaults and admission/prefill budget. It should not inspect `__AVX2__`, `__ARM_FEATURE_DOTPROD` or KAI flags directly.

## 6. Code Predictor audit

The CP path deserves separate treatment. `qwen_cp_predict()` performs 15 sequential codebook passes; each pass contains embedding/projection, a CP transformer step, norm and an LM-head argmax. That makes CP sensitive to B1 GEMV latency, thread-pool rendezvous and LM-head memory traffic even if Talker matmat throughput is excellent.

Current behavior:

- CP weights follow Talker precision selection, but `qwen_cp_prefill2_requested()` defaults the two-token CP prefill optimization only on AVX-512 VNNI; other architectures are opt-in.
- `qwen_batch_cp_predict()` can batch MTP/LM-head operations, but a single active slot deliberately falls back to `qwen_cp_predict()`.
- Whole-frame and CP regions require an integer runner and compatible INT8 weights. VNNI/AMX and KAI can satisfy this more often than AVX2, plain NEON or dotprod-only ARM.
- CP LM heads use `qwen_argmax_matvec_*`; the argmax/output path is not a matrix-unit substitute and must be timed separately.

The first CP baseline must report per-pass stage time, B distribution, GEMV/GEMM family and thread count. Optimizing Talker alone risks leaving CP as the end-to-end limiter on legacy CPUs.

## 7. Memory and arithmetic intensity

For GEMV, each active slot streams approximately one weight matrix once per projection, plus per-row scales and output. With INT8 weights the nominal lower bound is `rows*cols` bytes; with Q4_0 it is approximately `(rows*cols/32)*sizeof(q4_0_block_t)`. `qwen_matmat_stats_report()` exposes this as a **weight traffic lower bound**, not measured DRAM bytes.

Actual useful traffic additionally includes:

- source FP32 activation reads;
- gather writes/reads for `[cols][B]`;
- absmax/quantization passes and q8 writes;
- optional LHS packing (KAI/AMX/Q8);
- scales, row sums and correction terms;
- output scale and scatter writes;
- KV/attention and CP embedding traffic.

GEMM increases arithmetic intensity only if the same weight traversal serves several active slots and the staging cost is not larger than the saved weight reads. On a bandwidth-bound AVX2 or ARM dotprod GEMV, a native dot instruction may not be the main win; on a batched workload, a good packed integer matrix path can matter substantially. The audit therefore separates compute capability from memory/layout cost and requires bandwidth plus phase measurements for each work item.

## 8. Performance cliffs and discrepancies

### Confirmed from current code

1. **AVX2/AVX-512F INT8 GEMV cliff.** `qwen_int8_gemv_native()` returns true only for KAI, VNNI or SDOT (`qwen_tts_kernels.c:8806-8820`). AVX2 and AVX-512F execute `int8_matvec_fused()` with widen/dequant/FMA.
2. **AVX-512-no-VNNI has no integer matrix family of its own.** `g_mm_gate[]` has AVX2 INT8/Q4 rows and VNNI rows, but no AVX-512-BW-no-VNNI row. The AVX512 build still compiles the AVX2 branch.
3. **Decoder INT8 cliff.** `qwen_sd_int8_available()` is false for AVX2 and AVX-512F without VNNI, so the decoder’s standard control path remains f32/im2col/BLAS. This can dominate first audio and stream wall even if Talker kernels are improved.
4. **Dotprod-only matmat cliff.** ARM SDOT native GEMV exists; SDOT matmat is gated by `QWEN_INT8_SDOT_MM` and otherwise falls to an f32-accum twin. KAI’s dotprod implementation is not compiled without i8mm.
5. **Prefill fallback.** `qwen_prefill_matmat_resolved()` explicitly resolves to BF16→f32 conversion plus SGEMM when no native BF16 unit exists. This is the expected path on AVX2, AVX-512F and current M1 builds.
6. **Batched layout overhead.** B>1 ragged projections pay gather/scatter unless the contiguous KAI fast path is available. B=1 avoids this, so server C and kernel B must be reported separately.

### Documentation/runtime discrepancies to reconcile

- `docs/backend-matrix.md` contains historical rows that call Arm regions “hardware-blocked”, while current `qwen_tts_kleidi.c`, `qwen_tts_talker.c`, `qwen_tts_code_predictor.c` and `qwen_tts_speech_decoder.c` contain KAI region interfaces and implementations. The current `--dispatch-map` is authoritative for a built target; the matrix should be regenerated from current manifests before implementation work.
- Existing backend docs correctly record missing AVX2 native GEMV, but they should distinguish “AVX2 optimized batched emulation” from “generic/scalar fallback” so the AVX2 gap is not overstated.
- The local pre-audit `--caps` binary was dirty/stale. The current rebuilt output is now clean and should be the reference for future local documentation.

## 9. Benchmark and evidence plan

Every run must record build/source fingerprint, CPU flags, topology, physical-core/SMT placement, thread count, environment, selected dispatch map and actual in-process census. Use the existing tools; do not compare a requested path with the path that actually ran.

### Microbench baseline

1. `make cpu-check`, `make membw`, `make roofs` per execution mask.
2. `./qwen_tts --caps`, `./qwen_tts --dispatch-map`, `./qwen_tts --self-test`; run fallback self-test with `QWEN_NO_SDOT=1 QWEN_NO_VNNI=1` where supported.
3. `make matmat-bench`, `make prefill-bench`, `make kernel-census`; collect B=1,2,4,8,16 for representative Talker and CP shapes.
4. `tests/roof_matvec_int8.c` for INT8 weight-streaming and `tests/membw.c` for host bandwidth. Keep measured per-mask roofs separate; never divide a host roof to infer a worker roof.
5. `tests/matmat_parity.c` plus the existing `make check-matmat-parity-x86` under Rosetta where useful. For a new AVX2 dot kernel add adversarial signedness/range cases specifically targeting PMADDUBSW saturation.

### Required hardware matrix

| target | build / isolation | mandatory comparisons |
|---|---|---|
| Ryzen 7 6800H | `SIMD=portable`; physical cores only, then SMT; verify no AVX-VNNI/AVX-512 | INT8/Q4 GEMV and matmat B1/2/4/8; activation pack; CP/Talker; short v2 C1/2/4/8 |
| AVX-512 no VNNI | `SIMD=avx512`; verify VNNI absent; compare AVX2 flags if possible | AVX2 vs AVX-512 conversion/FMA, INT8/Q4 fallback, frequency and bandwidth, B crossover |
| VNNI reference | `SIMD=avx512vnni` | native GEMV versus matmat and existing VNNI prepack/region policies |
| AMX reference | `SIMD=amx` | B/rows-per-thread AMX gate, VNNI fallback, tile setup and activation pack |
| Apple M1 | current native M1 build | SDOT GEMV versus SDOT-matmat opt-in versus f32 twin; CP 15-pass timing; physical cores |
| ARM dotprod server | ARM build without i8mm | SDOT GEMV, SDOT B-loop, KAI unavailable cost, decoder INT8 policy |
| ARM i8mm/KleidiAI | `-march`/native KAI build | KAI GEMV/GEMM, persistent RHS/LHS pack, regions and B crossover |
| plain NEON | `-march=armv8-a` or equivalent | build/self-test, f32-FMA quantized GEMV, no-dotprod performance cliff |

### End-to-end metrics

For WAVE, SOAK and POISSON where appropriate, record TTFA and TTFB separately, STREAM/TOTAL RTF p50/p95, required prebuffer, safe-play-start, stall@250/@500, completions/throughput, rejects/errors/timeouts, CPU utilization, RSS, threads, FDs, scratch growth, context switches and memory bandwidth. A short run is a screen only; the existing qualification rules remain the gate.

## 10. Design recommendation

Keep the serving control plane common. Add one backend-performance profile produced after dispatch resolution and benchmark calibration:

```text
resolved ISA/backend
    -> kernel families actually selectable
    -> measured shape/B crossover and staging cost
    -> worker/thread/SMT profile
    -> batch, prefill and scheduler policy
```

The profile should be immutable for the process and printed into every benchmark manifest. It can start with conservative measured bands rather than an online autotuner. The scheduler should use it to choose default B windows, thread counts and prefill budgets; explicit flags can still override it for A/B runs.

Do not make the profile an ISA-name switch. `arm_dotprod` and `x86_avx2` are capability labels; the profile must distinguish native GEMV, batched family, staging cost and decoder availability.

## 11. Prioritized implementation plan

### LEGACY-CPU-1 — Dispatch/capability truth and backend profile

- Motivation: make “compiled”, “supported”, “resolved”, “effective” and “measured crossover” unambiguous before tuning.
- Files/functions: `qwen_tts_dispatch.c`, `qwen_tts_kernels.c`, `Makefile`, `tools/backend_matrix.py`, `tools/dispatch_expect.json`, benchmark manifest writers.
- Expected benefit: prevents running a legacy benchmark on an unintended fallback and gives v2 one stable profile input.
- Correctness risk: low; reporting only, but stale cached gates must not be misreported.
- Benchmark: current `--caps`, `--dispatch-map`, `--self-test`, `make cpu-check` on every target class.
- Acceptance: exact kernel family and fallback reason are printed for B1/2/4/8/16, prefill, decoder, CP and Talker; generated backend matrix matches current source.
- Dependencies: none. Do first.

### LEGACY-X86-1 — AVX2 INT8 GEMV

- Motivation: B1 decode currently performs per-weight widen/dequant/FMA despite the existing AVX2 batched integer-emulation path.
- Files/functions: `qwen_tts_kernels.c` `int8_matvec_fused()`, `qwen_matvec_int8()`, quantize/correction helpers; `tests/x86_b1_gemv_bench.c`, `tests/matmat_parity.c`.
- Expected benefit: lower Talker/CP B1 latency and improve physical-core scaling on Ryzen-class servers.
- Correctness risk: high for signedness, activation scale, row scales, tail dimensions and PMADDUBSW saturation.
- Benchmark: B1 representative Talker/CP rows, AVX2 widen+FMA versus signed-safe dot emulation; physical cores versus SMT; end-to-end CP/Talker.
- Acceptance: bit/relative parity within existing INT8 tolerance on adversarial ranges, no audio/golden regression, and a measured end-to-end win at B1 on Ryzen. If not faster, reject the path rather than keeping it for ISA symmetry.
- Dependencies: LEGACY-CPU-1.

### LEGACY-X86-2 — AVX2 Q4 GEMV

- Motivation: Q4 B1 also uses unpack/dequant/FMA while Q4 B>1 already has an integer-emulation style path.
- Files/functions: `qwen_tts_kernels.c` `q4_0_matvec_inner()`, `qwen_matvec_q4_0()`, Q4 activation quantization/correction helpers.
- Expected benefit: lower weight traffic with less scalar nibble conversion in B1 decode.
- Correctness risk: Q4 nibble sign/zero correction, block scale and tail handling.
- Benchmark: Q4 B1/B2/B4 exact shapes, cold/warm pack, CP and Talker; compare to current path and INT8.
- Acceptance: parity and quality gates pass; retain only if it wins complete-call latency, not just dot time.
- Dependencies: LEGACY-CPU-1; preferably after INT8 GEMV.

### LEGACY-X86-3 — AVX-512 without VNNI specialization

- Motivation: current AVX-512F/BW build inherits AVX2 integer matmat and generic GEMV. The benefit of wider vectors may be left unused, while downclock may erase it.
- Files/functions: `qwen_tts_kernels.c` AVX2 integer matmat/GEMV and AVX512 quantization helpers; `qwen_tts_dispatch.c` gate table.
- Expected benefit: possible better activation quantization and/or safer wider integer emulation; not assumed.
- Correctness risk: high if a new 512-bit signed dot emulation changes accumulation semantics; frequency and thermal behavior can reverse a microbench win.
- Benchmark: AVX2 versus AVX512-no-VNNI at B1/2/4/8, frequency, power if available, bandwidth and end-to-end TTFA/RTF.
- Acceptance: implement only if full-call performance improves on a real AVX512-no-VNNI host without unacceptable downclock; otherwise document AVX2 as the preferred profile.
- Dependencies: LEGACY-X86-1 and measured hardware.

### LEGACY-ARM-1 — Dotprod-only matmat and KAI split

- Motivation: dotprod-only ARM has SDOT GEMV but does not compile KAI’s dotprod family because KAI currently requires i8mm too; B>1 INT8 matmat is opt-in SDOT or f32 twin.
- Files/functions: `Makefile:91-120`, `qwen_tts_kleidi.c:23-90`, `qwen_tts_kernels.c` SDOT matmat family, `qwen_tts_dispatch.c`.
- Expected benefit: better B2/B4 CP/Talker batching on M1-class and older ARM servers; possibly lower packing overhead by reusing a dotprod-compatible KAI path.
- Correctness risk: KAI ABI/packing contracts and dotprod versus i8mm family selection; must not enable an i8mm kernel on dotprod-only hardware.
- Benchmark: M1 B1/2/4/8 exact shapes, SDOT loop, fixed-B twin, KAI dotprod-only build if feasible; CP 15-pass timing.
- Acceptance: compiled/runtime map distinguishes dotprod-only from i8mm; best path is selected only after parity and complete-call benchmark win.
- Dependencies: LEGACY-CPU-1; no scheduler changes.

### LEGACY-ARM-2 — Plain NEON fallback audit

- Motivation: plain NEON currently has functional vectorized widen+FMA paths but no integer dot. The size of this cliff must be measured before deciding whether to optimize it.
- Files/functions: `qwen_tts_kernels.c` NEON branches of `int8_matvec_fused()`, Q4 matvec, BF16 conversion; `qwen_tts_q8repack.c` capability guard.
- Expected benefit: either a small NEON GEMV improvement or a reliable conservative serving profile.
- Correctness risk: low to moderate; preserve quantization contracts and scalar tails.
- Benchmark: armv8-a no-dotprod build, B1/B2, CP/Talker/decoder, physical-core scaling.
- Acceptance: no scalar surprise, build/self-test clean, profile records the measured degradation and sets conservative worker/B defaults.
- Dependencies: LEGACY-CPU-1.

### LEGACY-PREFILL-1 — Legacy prefill dataflow

- Motivation: AVX2, AVX512-no-VNNI and M1 use BF16→f32 plus SGEMM for Talker prefill; repeated conversion and staging may dominate TTFA.
- Files/functions: `qwen_tts_talker.c:qwen_prefill_matmat_resolved/qwen_talker_prefill`, `qwen_tts_kernels.c` BF16 conversion/matmat helpers, `tests/prefill_bench.c`.
- Expected benefit: lower first-frame latency without changing serving semantics, potentially through persistent f32/prepacked weights or a measured backend-specific chunk.
- Correctness risk: high TTFA/audio risk if rounding, layout or cache lifetime changes.
- Benchmark: cold/warm conversion, pack and SGEMM separately; TTFA p50/p95 and audio parity.
- Acceptance: no hidden repeated allocation/repack in steady state, no quality regression, and end-to-end TTFA win on at least one legacy class.
- Dependencies: dispatch profile; do not mix with scheduler changes.

### V2-BACKEND-1 — Profile-driven worker/batch policy

- Motivation: v2 currently has fixed B gates and deployment knobs but no backend cost profile; weaker SIMD must not merely run the VNNI profile slower.
- Files/functions: new small profile API near `qwen_tts_dispatch.c`; consumers `qwen_tts.c`, `qwen_tts_server.c`, `qwen_tts_thread.c`; profile/config manifest code.
- Expected benefit: backend-appropriate B cap, threads/worker, SMT policy, prefill quantum and decoder budget.
- Correctness risk: medium; policy can affect streaming continuity and admission fairness.
- Benchmark: baseline versus profile-selected WAVE and short SOAK at C1/2/4/8 as hardware allows; retain resolved dispatch and effective-B census.
- Acceptance: policy changes only when the profile says it should; C1 quality unchanged; legacy target does not regress against its own best fixed baseline.
- Dependencies: LEGACY-CPU-1, at least one measured crossover from x86 and ARM.

### CP-1 — Code Predictor-specific profile and kernels

- Motivation: CP is sequential, runs 15 codebook passes and may dominate end-to-end latency even after Talker optimization.
- Files/functions: `qwen_tts_code_predictor.c:qwen_cp_predict/qwen_batch_cp_predict/cp_batch_*`, `qwen_tts_kernels.c` LM-head GEMV/matmat, CP dispatch rows.
- Expected benefit: reduced TTFA/RTF on legacy CPUs without assuming Talker batching transfers to CP.
- Correctness risk: medium; codebook argmax parity and code sequence stability are mandatory.
- Benchmark: per-pass CP timing, B1/B2/B4, LM-head memory behavior, context-switch/thread overhead and full TTFA.
- Acceptance: code sequence/golden parity and a measured CP wall reduction with no Talker/decoder continuity regression.
- Dependencies: LEGACY-CPU-1; profile before scheduler coupling.

### VALID-1 — Cross-ISA parity and qualification gates

- Motivation: every new low-bit kernel or layout can change numerical accumulation and audio even when tensor error looks small.
- Files/functions: `tests/matmat_parity.c`, `tests/decoder_batch_parity.c`, `make test-golden`, `docs/BENCHMARKING.md`, manifest/report tools.
- Expected benefit: reproducible promotion decisions and no “requested path” versus “actual path” confusion.
- Correctness risk: test coverage cost only; gate must be strict enough to catch silent saturation/layout errors.
- Benchmark: self-test, fallback self-test, tensor parity, code sequence parity, paired audio, WAVE and then canonical SOAK.
- Acceptance: no work item is promoted without exact dispatch proof, numerical parity, complete-call metrics and a clean manifest.
- Dependencies: all implementation items.

## 12. Audit verdict

The current v2 architecture is portable functionally and has useful optimized families beyond the newest targets, but it is not yet performance-portable. The highest-confidence gaps are AVX2/AVX512-no-VNNI INT8/Q4 GEMV, decoder INT8 availability on non-VNNI x86, dotprod-only ARM matmat/KAI reachability, and the absence of a backend performance profile feeding v2 policy. The highest-risk mistake would be to “fix” all of these with one global GEMM or scheduler rule.

The safe order is: make dispatch truth measurable, benchmark complete calls on the named ISA classes, add one legacy GEMV family with parity, then add backend-aware crossover/profile policy. Until those runs exist, no AVX2, AVX512-no-VNNI, M1 or plain-NEON concurrency claim should be inferred from VNNI/AMX/KleidiAI results.

## 13. Implementation status — first legacy candidate

### LEGACY-X86-1 — AVX2 INT8 GEMV

The first implementation is now isolated in the dedicated legacy worktree. It adds an
experimental B=1 path that reuses the existing activation quantization contract and computes
the signed INT8 dot by widening both operands to signed 16-bit lanes before `PMADDWD`.
It intentionally does not reuse the existing AVX2 B>1 `PMADDUBSW` sequence: that instruction
has saturating 16-bit pairwise intermediates, so the candidate avoids making an unproved
range assumption. The implementation is bounded to the existing 8192-element activation
scratch contract and returns to the FMA GEMV outside that bound.

The path is selected only with `QWEN_AVX2_INT8_GEMV=1`; the FMA widen/dequant GEMV remains the
default. `--caps`/`--dispatch-map` expose compiled, runtime-supported and policy-enabled
state, and shape census gets the leaf name `avx2-int8-emulated-dot-gemv`.

Status:

| property | status | evidence / limitation |
|---|---|---|
| IMPLEMENTED | YES | candidate, opt-in dispatch, precise report row and appended census leaf |
| PARITY VERIFIED | PENDING AVX2 HOST | adversarial signed-extreme, saturation-sensitive and tail tests are compiled into `--self-test`; the current M1 host correctly skips them; x86 cross-compilation passed |
| PERFORMANCE VERIFIED | NO | no Ryzen/AVX2 execution host available in this session |
| DEFAULT/PROMOTED | NO | explicit opt-in only |

The current ARM build was rebuilt with `make blas`; `--caps`, `--dispatch-map`, and
`--self-test` passed with zero failures. The AVX2 translation units for `qwen_tts_kernels.c`
and `qwen_tts_dispatch.c` compiled for `x86_64-apple-darwin` with `-mavx2 -mfma`. The existing
generated flag-scope header had unrelated pre-existing CUDA flag omissions; only the new
candidate flag was added, and the full generated header was not imported to avoid unrelated
CUDA scope churn.

Next action: on a real AVX2 host run the self-test with the candidate enabled and disabled,
then run the complete-call B1 Talker/CP GEMV benchmark before deciding whether to keep the
candidate. Only after that parity/performance decision should LEGACY-X86-2 Q4 GEMV start.
