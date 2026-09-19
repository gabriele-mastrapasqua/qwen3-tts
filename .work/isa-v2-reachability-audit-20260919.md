# Released-main ISA and v2 reachability audit (2026-09-19)

Task · ISA-AUDIT-1 — audit the released `main` tree before any new SIMD implementation.

Question · Which old/common CPU capabilities and SIMD kernels are compiled, selectable, and actually reached by the v2 continuous/ragged server for B=1 and B>1? Which existing paths are missing, gated too narrowly, or bypassed by the server?

Known facts · The audit baseline is released `main` at `e391ec5467b0218eeb175f4888ad65b259d1e7c7` (`v0.23.0`). The working tree contains unrelated uncommitted integration work, so all source conclusions below were made from a clean archive of that commit. The current server uses a live active width (`B_eff`), not the configured server capacity, and several optimized gates are limited to B<=16.

Unknowns · No Linux x86 or Linux Arm host was available in this pass; the native execution check is an Apple M1-class build. We have not measured complete-call or streaming performance for the candidate fallbacks, AVX-512 frequency effects, or any cloud VM. The cloud table is a qualification plan, not a claim that a family always exposes a particular feature set.

Files/functions inspected · `Makefile`; `qwen_tts_kernels.c/.h`; `qwen_tts_kleidi.c`; `qwen_tts_q8repack.c`; `qwen_tts_gguf.c`; `qwen_tts_talker.c`; `qwen_tts_code_predictor.c`; `qwen_tts.c`; `qwen_tts_speech_decoder.c`; `qwen_tts_server.c`; `qwen_tts_dispatch.c`; `third_party/ingot/src/{cpu,kernels}.c`; `PLAN.md`; `ENGINEERING.md`.

Evidence · Source guards, runtime predicates, call sites, and fallback branches were read in the clean archive. The archive was built with `make blas`; `--caps`, `--dispatch-map`, and `--self-test` were run from the resulting binary. The Apple M1-class self-test passed all cases. The exact command output is summarized in §L; raw build output is not tracked.

Conclusion · The released tree is usable across the scalar/NEON/AVX2/AVX-512/AMX families, but support is uneven by operation and batch width. AVX-512 without VNNI is a real wide FP/SIMD build that reuses AVX2 integer matmat; it is not an AVX-512 integer backend. The largest baseline reachability defects are (1) ragged decoder batching is disabled for ordinary VNNI/SDOT INT8 paths and (2) live B_eff above 16 silently misses most native matmat gates. The former remains an intentional control pending an equivalent multislot panel implementation; the latter now has an opt-in chunking control. The dotprod-only KleidiAI gate has since been split for a default-off B=1 candidate, while full KAI GEMM/regions remain i8mm-only. B=1 INT8/Q4 on AVX2 and AVX-512-no-VNNI remains a fallback/candidate problem, not proof that AVX2 is complete.

Next action · Track P0 wiring/observability and P1 gate fixes first; qualify them on native Linux hosts; only then benchmark or implement P2 kernels.

## Executive findings

1. **The server batch is ragged.** `qwen_batch_pack_active()` computes `B_eff` from runnable slots (`qwen_tts_talker.c:2406-2438`). A C4 or C8 server can therefore execute B=1, 2, 3, … on successive steps. Solo mode routes a single runnable slot directly to the B=1 path (`qwen_tts_talker.c:2798-2868`). Configured `--batch-size` is not kernel B.
2. **There is an unhandled B=16 boundary.** INT8/Q4/SMMLA/VNNI/AMX gates have `max_b=16` (`qwen_tts_kernels.c:2323-2345`). The server passes live `B_eff` without chunking (`qwen_tts_talker.c:2236-2395`). A capacity or active cohort above 16 therefore falls to generic/f32 logic, or to per-slot GEMV in region/forced paths, without a clear server census. This needs an explicit cap, chunking, or visible fallback.
3. **AVX2/FMA is active for B>1 matmat, but B=1 integer GEMV is not native by default.** `qwen_matmat_int8()` and `qwen_matmat_q4_0()` can select AVX2 for B>1. `qwen_matvec_int8()` and `qwen_matvec_q4_0()` select the signed-widening AVX2 candidates only when their environment flags are set; otherwise they use the fused FP32/dequant fallback. The known negative candidate must remain opt-in.
4. **AVX-512 without VNNI is not a separate integer tier.** The `avx512` profile compiles AVX512F/BW/VL plus AVX2/FMA, but no VNNI or BF16. FP32, attention, RMS, conversion, and helper code can use AVX-512; INT8/Q4 B>1 reuses AVX2-width kernels; B=1 uses the same fallback/candidate policy as AVX2.
5. **VNNI and AMX are shape-gated, not host-wide labels.** VNNI B=1 GEMV and B>1 matmat are available when shapes and environment permit. AMX INT8 begins at B>=3 and rows>=32/cols>=64; AMX BF16 begins at B>=4 and rows/cols>=32. B=1 on an AMX host still uses GEMV/VNNI/FP32 paths.
6. **Baseline KAI support was coupled to i8mm.** The full Makefile/backend still requires `__ARM_FEATURE_MATMUL_INT8` and runtime dotprod+i8mm for GEMM/regions, but the branch now builds a separate dotprod-only B=1 Q4/INT8 candidate from `__ARM_FEATURE_DOTPROD`, with its own packing and `QWEN_KAI_DOTPROD_GEMV=1` opt-in. Linux complete-call qualification remains open.
7. **Decoder reachability differs from Talker/CP.** Per-item INT8 decoder convolution is compiled only for AVX512VNNI or Arm dotprod. The v2 decoder batch entry returns to per-item decode unless AMX decoder support or the explicit multislot-v2 lane is enabled (`qwen_tts_speech_decoder.c:4512-4547`). A VNNI or SDOT host therefore does not automatically get a B>1 decoder kernel.
8. **Observability is not yet enough to prove production leaf reachability.** `--caps` and `--dispatch-map` report representative B values and family predicates, but no per-stage `B_eff`, runnable/parked reason, contiguity, or leaf histogram. A small default-off census extension is needed before declaring C4/C8 coverage.
9. **The released dispatch class hides the AVX-512-no-VNNI tier.** `qwen_tts_dispatch.c:81-113` checks VNNI and then AVX2, so a no-VNNI AVX-512 build is reported as `x86_avx2` even though its FP helpers were compiled at AVX-512 width. The server and qualification reports cannot currently distinguish “AVX2 host” from “AVX-512F/BW/VL host reusing AVX2 integer kernels.”

## A. Scope and method

### Baseline

The audit uses only the clean released `main` tree at `e391ec5`. The current checkout was not reset and its uncommitted changes were excluded with a temporary clean archive. No kernel, dispatcher, server, model, or production default was changed by this audit.

### Evidence levels

* **Compiled** means a source guard and Makefile profile include the code.
* **Supported** means the runtime predicate accepts the host.
* **Selected** means the dispatcher branch can return that leaf for a shape and environment.
* **Reached** means a v2 caller passes that operation and width to the dispatcher.
* **Useful** remains an empirical question: the complete call, packing, ragged width, and frequency behavior must be measured.

The census labels are `ACTIVE`, `OPT-IN`, `FALLBACK`, `COMPILED BUT UNREACHABLE`, `DEAD`, and `UNCLEAR`.

## B. ISA inventory from source

### x86

| Tier/capability | Compile guard/profile | Runtime guard | Current engine scope | Status |
|---|---|---|---|---|
| Scalar | no ISA macro; generic C | none | Every operation has scalar/generic fallback | ACTIVE fallback |
| SSE/SSE2/SSE4.x | No engine-specific SSE/SSE2/SSE4 guards or leaf; compiler baseline may emit scalar-build instructions | No per-feature dispatch | Not an independently selected backend | No dedicated backend |
| AVX | No standalone AVX engine branch | `qwen_check_runtime_isa` only rejects a binary whose required profile is unavailable | Covered indirectly by AVX2/FMA profiles | Not a tier |
| AVX2 + FMA/FMA3 | `-mavx2 -mfma`; `__AVX2__`, `__FMA__` | `__builtin_cpu_supports("avx2")`/startup check | FP32 matvec/attention/RMS/activation/conversion; INT8/Q4 B>1 matmat; opt-in B=1 candidates | ACTIVE, with B=1 fallback |
| F16C | No qwen engine guard; only vendored Ingot code contains F16C code/capability checks | Ingot runtime only | No production v2 call into Ingot F16/Q4_K paths | COMPILED BUT UNREACHABLE from v2 |
| BMI1/BMI2 | No project guard or runtime predicate | None | No identified production kernel contract | Not implemented |
| AVX-512F/BW/VL, no VNNI | `SIMD=avx512`: `-mavx512f -mavx512bw -mavx512vl -mavx2 -mfma`; no `-mavx512vnni`, BF16, or DQ | CPU feature check for AVX512F/BW where used; AVX2 check remains required | Wide FP32/attention/RMS/conversion helpers; AVX2 integer matmat | ACTIVE as FP/wide + AVX2 integer backend |
| AVX-512DQ | No explicit project flag in the `avx512` profile and no engine guard | Not separately reported/required | No independent DQ kernel found | Not implemented |
| AVX512-VNNI | `SIMD=avx512vnni` and higher profiles; `__AVX512VNNI__` | `__builtin_cpu_supports("avx512vnni")`, `QWEN_NO_VNNI`, B/shape gate | INT8/Q4 GEMV and B>1 matmat; fused x86 INT8 QKV; decoder INT8 conv | ACTIVE where shape/gates pass |
| AVX512-BF16 | `SIMD=avx512bf16`/AMX profiles; `__AVX512BF16__` | `avx512bf16`, `QWEN_NO_BF16_MATMUL`, B/shape gate | BF16 matmat, row-pack/pre-fill, DPBF16 matvec | ACTIVE where compiled; otherwise BLAS/generic |
| AVX512-FP16 | No `__AVX512FP16__` engine guard or profile | Not reported | No FP16 engine kernel found | Not implemented |
| AMX-INT8 | AMX profile: `-mamx-tile -mamx-int8` plus AVX512/VNNI; compile guards | CPUID plus Linux `arch_prctl` tile permission; `QWEN_NO_AMX_INT8`; B>=3, rows>=32, cols>=64 | INT8/Q4 matmat and selected Talker/CP regions; B=1 does not use tiles | ACTIVE, shape-gated |
| AMX-BF16 | AMX profile: `-mamx-bf16`; `__AMX_BF16__` | CPUID/permission; `QWEN_NO_AMX_BF16`; B>=4 and rows/cols>=32 | BF16 matmat and decoder BF16 options | ACTIVE, shape/opt-in gated |
| Apple Silicon | Darwin uses `-march=native`; no x86 guards | sysctl feature queries | Separate Arm NEON/SDOT/BFMMLA/SMMLA behavior | See ARM/Apple below |

The Makefile's Linux auto selection is ordered AMX → AVX512-BF16 → AVX512-VNNI → AVX512 → AVX2 → scalar (`Makefile:8-48`). A profile name therefore does not prove every instruction in the family is available to a runtime process; the runtime predicates and shape gates remain authoritative.

### Arm / Apple

| Tier/capability | Compile/runtime evidence | Current engine scope | Status |
|---|---|---|---|
| Plain NEON | AArch64 NEON guards; Linux HWCAP/Apple sysctl | 2-row FP32/BF16 conversion, RMS, attention, activations, generic matvec support | ACTIVE fallback/backend |
| FP16 | No separate engine FP16 dispatch; BF16/FP16 data is converted through existing paths | No independent FP16 GEMM/GEMV contract | No dedicated tier |
| Dotprod SDOT/UDOT | `__ARM_FEATURE_DOTPROD`; HWCAP ASIMDDP/sysctl | INT8/Q4 B=1 GEMV; INT8 B>1 SDOT matmat is opt-in; Q4 B>1 is B×GEMV; q8 repack B=1 | ACTIVE/OPT-IN |
| i8mm SMMLA/UMMLA | `__ARM_FEATURE_MATMUL_INT8`; HWCAP2 I8MM/sysctl | Native INT8/Q4 B>1 matmat; q8 repack B>1; KAI packed GEMM/regions | ACTIVE when dotprod+i8mm compiled and present |
| BF16/BFDOT/BFMMLA | `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC` in kernels; BFMMLA runtime; Apple BFMMLA default off; `QWEN_ARM_BFDOT` opt-in | BF16 matmat BFMMLA; BF16 GEMV usually NEON 2-row; BFDOT is opt-in | ACTIVE/OPT-IN |
| SVE/SVE2 | No qwen engine SVE/SVE2 guards or dispatch; only vendor helper declarations | No SVE/SVE2 GEMV/GEMM | Not implemented |
| SME | No qwen engine SME kernels/dispatch; vendor declarations only | No SME path | Not implemented |
| KleidiAI dotprod | Separate `qwen_tts_kleidi_dotprod.c` and dotprod vendor pack/kernel sources; runtime HWCAP ASIMDDP/sysctl | Opt-in B=1 Q4/INT8 GEMV with dotprod-specific RHS packing; no B>1 KAI region | IMPLEMENTED OPT-IN; local pack/kernel parity, Linux complete-call qualification pending |
| KleidiAI i8mm | `QWEN_KLEIDI_BUILD` requires AArch64 + DOTPROD + MATMUL_INT8; runtime checks both | Q4/INT8/BF16 GEMV/GEMM, prepared regions, selected Talker/CP | ACTIVE on i8mm hosts; GEMM/regions remain i8mm-only |
| Apple Silicon M1 | Darwin sysctl reports NEON+dotprod; BFMMLA/SMMLA default off; full KAI unavailable without i8mm; dotprod candidate is opt-in | SDOT B=1; B>1 generic/B×GEMV; optional KAI dotprod B=1; BF16 prefill via Accelerate | ACTIVE fallback profile; candidate locally parity-checked |
| Apple Silicon M2+ class | Runtime can expose dotprod+i8mm/BF16; Apple MMLA still explicitly opt-in | Potential SMMLA/BFMMLA/KAI paths after env and shape gates | OPT-IN/needs native qualification |

The full KAI Makefile/source block remains conditional on `__ARM_FEATURE_MATMUL_INT8` and `qwen_kleidi_supported()` still requires runtime dotprod+i8mm. The branch additionally builds `qwen_tts_kleidi_dotprod.c` and the vendor dotprod packers/1x kernels when `__ARM_FEATURE_DOTPROD` is present. Its runtime predicate checks HWCAP ASIMDDP/sysctl and its separate registry is only used with `QWEN_KAI_DOTPROD_GEMV=1`; this is a candidate, not a claim that full KAI is available on dotprod-only CPUs.

## C. Kernel census and caller reachability

This table records the production qwen dispatcher. Vendor Ingot kernels are listed separately because the source is compiled into the binary but the production call graph does not call them.

| Operation / leaf | Source and guards | Runtime/env gate | v2 callers and actual reachability | State |
|---|---|---|---|---|
| INT8 GEMV, VNNI | `qwen_matvec_int8()`; `__AVX512VNNI__` | VNNI present, `QWEN_NO_VNNI` unset, columns<=8192 | Talker/CP B=1 projections and codec heads on x86 VNNI | ACTIVE |
| INT8 GEMV, SDOT | `qwen_matvec_int8()`; DOTPROD | `QWEN_NO_SDOT` unset, columns<=8192 | Talker/CP B=1 on Arm dotprod | ACTIVE |
| INT8 GEMV, KAI | full `qwen_tts_kleidi.c` on dotprod+i8mm; separate `qwen_tts_kleidi_dotprod.c` on dotprod | `QWEN_NO_KLEIDI` for full KAI; `QWEN_KAI_DOTPROD_GEMV=1` for candidate | Full KAI prepared/GEMV paths when model regions and state qualify; dotprod-only candidate reaches B=1 after opt-in/prepack | ACTIVE full KAI; OPT-IN candidate |
| INT8 GEMV, AVX2 signed widening | `qwen_matvec_int8()`; `__AVX2__` | `QWEN_AVX2_INT8_GEMV=1` and related minimum/shape flags | B=1 only; v2 can reach it if explicitly enabled | OPT-IN; known negative candidate |
| INT8 GEMV, generic | FP32 fused twin/dequant fallback | Always | All unsupported B=1 cases including AVX2/no-VNNI | FALLBACK |
| INT8 B>1 matmat, AVX2 | `qwen_matmat_int8()`; `__AVX2__` | `QWEN_NO_AVX2MM`, min B=2, max B=16, shape | Talker/CP batch projections when B_eff>1; AVX512-no-VNNI reuses this | ACTIVE |
| INT8 B>1 matmat, VNNI | `qwen_matmat_int8()`; `__AVX512VNNI__` | VNNI gate, B 2..16 | Talker/CP; QKV fused path can use it | ACTIVE |
| INT8 B>1 matmat, SMMLA | `qwen_matmat_int8()`; DOTPROD+MATMUL_INT8 | `QWEN_NO_SMMLA`, B 2..16; Apple opt-in | Arm i8mm Talker/CP when direct dispatcher path is used | ACTIVE/OPT-IN |
| INT8 B>1 matmat, SDOT loop | `qwen_matmat_int8()`; DOTPROD | `QWEN_INT8_SDOT_MM=1`, B 2..16 | Dotprod-only Arm can reach it only with explicit env; default uses f32 twin/B×GEMV | OPT-IN |
| INT8 B>1 matmat, AMX | `qwen_matmat_int8()`; AMX INT8 | permission, B>=3, rows>=32, cols>=64 | Talker/CP shapes that meet tile gate; not B=1 | ACTIVE/shape-gated |
| INT8 fused QKV | `qwen_matmat_int8_qkv()` | x86 AMX or VNNI; B 2..16 | `qwen_batch_proj_qkv()` only for contiguous all-INT8 groups; other ISAs perform three projections | ACTIVE but narrow |
| Q4 GEMV, VNNI | `qwen_matvec_q4_0()`; VNNI | VNNI gate and shape | Talker/CP B=1 on VNNI | ACTIVE |
| Q4 GEMV, SDOT | `qwen_matvec_q4_0()`; DOTPROD | dotprod and shape | Talker/CP B=1 on Arm dotprod | ACTIVE |
| Q4 GEMV, AVX2 candidate | `qwen_matvec_q4_0()`; AVX2 | `QWEN_AVX2_Q4_GEMV=1` | B=1 only, opt-in | OPT-IN |
| Q4 GEMV, KAI | KAI Q4 dotprod/i8mm | KAI enabled | B=1 KAI path when model/prepared state qualifies | ACTIVE on KAI |
| Q4 GEMV, fused dequant | qwen FP32 fused fallback | Always | AVX2/no-VNNI and unsupported shapes | FALLBACK |
| Q4 B>1 matmat, AVX2 | `qwen_matmat_q4_0()`; AVX2 | B 2..16, `QWEN_NO_AVX2MM` | Talker/CP B_eff>1 on AVX2 and AVX512-no-VNNI | ACTIVE |
| Q4 B>1 matmat, VNNI | same; VNNI | VNNI, B 2..16 | x86 VNNI | ACTIVE |
| Q4 B>1 matmat, SMMLA | same; i8mm | SMMLA gate, B 2..16 | Arm i8mm | ACTIVE/OPT-IN |
| Q4 B>1 dotprod | Explicit B-column loop over `qwen_matvec_q4_0()` | dotprod; no matrix unit | Dotprod-only Arm; rereads weights and is census-labeled B×GEMV | ACTIVE fallback |
| Q4 B>1 AMX | same; AMX INT8 | AMX Q4 gate, B>=4, shape | AMX shapes only | ACTIVE/shape-gated |
| BF16 GEMV | `qwen_matvec_bf16()` and NEON/DPBF16 helpers | AVX512 BF16/ARM BFDOT gates where applicable | Talker/CP B=1; generic NEON on plain/dotprod Arm | ACTIVE/fallback |
| BF16 B>1 AVX512 | `qwen_matmat_bf16()`; `__AVX512BF16__` | BF16 gate, B<=16, columns>=32 | Talker/CP matmat and row-pack prefill | ACTIVE |
| BF16 B>1 BFMMLA | Arm BFMMLA | B>=2, Apple off by default | Arm BF16-capable hosts after opt-in/gate | OPT-IN |
| BF16 B>1 AMX | AMX BF16 | B>=4, rows/cols>=32 | Talker/CP; decoder BF16 options | ACTIVE/shape-gated |
| BF16 B>1 KAI | KAI BF16 sources, Makefile BF16 gate | KAI + BF16 compiler/runtime | Prepared KAI regions; not dotprod-only | ACTIVE/conditional |
| BF16 generic fixed-B / f32 twin | `qwen_matmat_bf16()` | Always | AVX2, AVX512-no-VNNI, plain NEON and unsupported shapes | FALLBACK |
| Q8/repacked GEMV | `qwen_tts_q8repack.c` | NEON + dotprod; env `QWEN_NO_Q8REPACK` | GGUF registration; B=1 q8 repack can use SDOT | ACTIVE |
| Q8/repacked B>1 | same | Runtime i8mm required for GEMM; rows%4/cols%32 | `qwen_matmat_bf16()` tries q8r first for registered weights | ACTIVE only on i8mm; B×GEMV/fallback otherwise |
| FP32 GEMV/GEMM | common C/NEON/AVX2/AVX512 helpers; BLAS for prefill | shape/thread/env | All stages as fallback; prefill often external SGEMM | ACTIVE fallback |
| Attention | `qwen_rms_norm`, AVX2/AVX512/NEON attention and BF16 KV helpers | ISA-specific compile guards | Talker/CP step independent of integer matmul selection | ACTIVE |
| Activation/quantization | NEON/AVX2/AVX512 RMS, SiLU/SwiGLU, `qwen_quantize_rows` | compile guards | Talker/CP and prefill activation quantization | ACTIVE |
| Packing/repacking | row-pack, q8 repack, KAI prepared states | backend and shape gates | Prefill and batched projections; packing can determine later leaf | ACTIVE with gaps |
| Decoder INT8 conv | `qwen_conv1d_int8`; only DOTPROD or AVX512VNNI body | `QWEN_SD_INT8`, shape `in==out<=768` | Per-item decoder; AVX2/no-VNNI has no integer conv | ACTIVE on VNNI/SDOT, missing on AVX2/no-VNNI |
| Decoder INT8 conv v2/multi | `qwen_conv1d_int8_v2{,_multi}` | `QWEN_SD_RES1_V2`, `QWEN_SD_MULTISLOT`, lane shape | Only selected exact-stream/multislot decoder modes; ordinary VNNI/SDOT batch entry may bypass | OPT-IN / narrow |
| Decoder BF16 pre-up / GEMM | `sd_bf16_preup_matmat`, KAI/AVX512 BF16 or SGEMM | decoder env flags and backend | Batch decoder only for qualified modes; otherwise BLAS/f32 | OPT-IN/fallback |
| Decoder transposed conv | `convt_stack`/per-tap GEMM | decoder flags | Batch exact path or per-item control | ACTIVE/fallback |
| Q6/Q2 and other quantized matvec | generic/AVX2/NEON matvec helpers | no independent matmat family | Model-dependent fallback; no v2 B>1 native census row | ACTIVE fallback / UNCLEAR for production model coverage |

### Vendor code that is not a v2 backend

`third_party/ingot/src/kernels.c` includes AVX2/AVX512/NEON Q4_K, Q8, BF16, F16 and related CPU capability code (`third_party/ingot/src/cpu.c`). A source search of production qwen callers finds no calls to the Ingot `matvec/matmat/q4_k/q8_0/bf16_mat/f16_mat` entry points; only the dedicated Ingot benchmark/test code calls them. These are therefore `COMPILED BUT UNREACHABLE` from the v2 server, not evidence of qwen AVX512-FP16/F16C support.

The files `qwen_tts_kernels_avx.c`, `qwen_tts_kernels_neon.c`, and `qwen_tts_kernels_generic.c` are one-line translation-unit stubs. The implementation is in `qwen_tts_kernels.c`; the stub filenames do not represent independent backend leaves.

## D. v2 execution map

```text
HTTP request
  -> qwen_tts_server.c admission/worker
  -> qwen_tts_serve_continuous(ctx, max_batch, sink)
  -> qwen_batch_t active/step_active/lead/priority masks
  -> qwen_batch_pack_active() -> B_eff and active indices
  -> Talker prefill or qwen_batch_talker_step_ragged()
       -> qwen_batch_proj_qkv()/qwen_batch_proj_q()
       -> qwen_matvec_* (B_eff=1) or qwen_matmat_* (B_eff>1)
       -> Talker attention/activation/codec head
  -> qwen_batch_cp_predict()
       -> CP projection/QKV/frame heads and optional CP region
       -> sampling/code predictor
  -> qwen_speech_decoder_decode_streaming_batch()
       -> exact ragged decoder batch only for its explicit gates
       -> otherwise per-slot qwen_speech_decoder_decode_streaming_st()
  -> streaming PCM sink
```

### Admission, active width, and prefill

* `qwen_tts_server.c` invokes `qwen_tts_serve_continuous()` with the configured capacity.
* `qwen_tts.c:2959-2995` allocates the slot batch. `qwen_tts_talker.c:2406-2438` packs only runnable slots and records `B_eff`.
* Lead feedback, TTFA priority, maximum Talker width, decoder QoS, and paused slots can reduce `step_active`. The same C can produce different B_eff values on adjacent steps.
* `qwen_batch_proj_q()` uses per-slot GEMV when B_eff=1, `QWEN_BATCH_NO_MATMUL` is set, or a force-matvec policy is active; otherwise it gathers active columns and calls a matmat dispatcher.
* `qwen_batch_proj_qkv()` can use fused x86 INT8/BF16 QKV only for contiguous all-INT8/all-BF16 groups and only for AMX/VNNI or AMX/AVX512-BF16. Q4 and unsupported Arm paths call three projections.
* Talker prefill (`qwen_tts_talker.c:835-957`) chunks INT8/BF16 token batches, but `QWEN_PREFILL_INT8MM=1` is opt-in. BF16 native prefill is selected only for KAI/AMX/BFMMLA/AVX512-BF16; otherwise it converts to FP32 and uses SGEMM (`qwen_tts_talker.c:1338-1405`).

### Talker and CP regions

* A single active slot defaults to the solo `qwen_talker_step()`/`qwen_cp_predict()` paths (`qwen_tts_talker.c:2798-2868`, `qwen_tts_code_predictor.c:1533-1595`). This is why C4/C8 does not imply B4/B8.
* The Talker and CP region runners require all-INT8 weights, a holdable team, and B in 2..16. x86 requires VNNI/AMX; Arm requires prepared KAI state (`qwen_tts_talker.c:2661-2705`, `qwen_tts_code_predictor.c:1230-1256`). Q4 skips these regions.

### Decoder

`qwen_speech_decoder_decode_streaming_batch()` filters empty frames, then falls back to per-item decode when `nb==1`, exact streaming is off, or INT8 is requested without decoder AMX and the v2 multislot gate (`qwen_tts_speech_decoder.c:4512-4547`). The ordinary VNNI/SDOT INT8 conv leaf is therefore reachable per item, but not automatically as a ragged B>1 decoder kernel. The exact batch path uses BF16/KAI/AVX512-BF16 or BLAS matmuls and its own ragged convolution conditions.

## E. Critical ISA × stage × B reachability matrix

Cells describe what the **released v2 server actually reaches**, with `B=1` meaning the active kernel width and `B>1` meaning a live packed cohort that passes the stage's gates. `C` is server concurrency; it is not B.

| Stage / weight type | B=1 path | B>1 path | AVX2 | AVX-512 no-VNNI | VNNI | AMX | NEON | SDOT / dotprod | i8mm / KAI |
|---|---|---|---|---|---|---|---|---|---|
| Talker step INT8 | GEMV | matmat or generic | B1 FP32 fused; B>1 native AVX2 if 2..16 | B1 FP32 fused; B>1 AVX2-width native | native VNNI GEMV/matmat | B1 VNNI/GEMV; B>=3 native AMX when rows/cols pass | generic/f32 | native SDOT GEMV; B>1 SDOT only opt-in, else f32 twin/B×GEMV | full KAI GEMV/GEMM or SMMLA when prepared; dotprod-only KAI GEMV is opt-in B=1 |
| Talker step Q4 | GEMV | matmat/B×GEMV | B1 fused dequant or opt-in candidate; B>1 AVX2 | B1 fused dequant; B>1 AVX2 | native VNNI | B>=4 AMX Q4; B1 VNNI | generic/fused | SDOT GEMV; B>1 explicit B×GEMV | dotprod-only KAI 1x GEMV opt-in; i8mm KAI GEMM on full profile |
| Talker step BF16 | NEON/FP32 or native BF16 | fixed-B/generic or native | generic fixed-B/f32 twin | generic fixed-B/f32 twin; no BF16 instruction | generic fixed-B/f32 unless BF16 profile | B>=4 AMX BF16 with shape | 2-row NEON/f32 | no SDOT BF16 GEMM | KAI BF16 only BF16+i8mm build |
| Talker prefill INT8 | usually non-INT8/BLAS unless opt-in | chunks <=16 then INT8 dispatcher | opt-in AVX2 matmat; otherwise fallback | same AVX2-width fallback | VNNI only when opt-in and B gate passes | AMX if chunk/shape gate passes | generic/BLAS | SDOT opt-in; otherwise generic | KAI/prepared if model and build qualify |
| Talker prefill BF16 | convert + SGEMM unless native backend | chunks default 16 (env 16..64) | FP32 conversion + BLAS | FP32 conversion + BLAS | FP32 conversion + BLAS unless BF16 profile | native AMX BF16 if chunk/shape | FP32/Accelerate or generic | not a BF16 GEMM path | KAI BF16 if independently compiled |
| CP step INT8/QKV | GEMV | matmat or CP region | B1 fused FP32; B>1 AVX2 if gate | AVX2-width B>1; no AVX512 integer | native VNNI; CP region if enabled | AMX/QKV/region for shape | generic/f32 | SDOT B1; B>1 opt-in or generic | full KAI region/GEMM if prepared; dotprod-only candidate is B=1 GEMV |
| CP prefill/frame heads | GEMV/three projections | QKV/heads or region | no fused QKV; AVX2 projections | no fused QKV; AVX2 projections | fused QKV and CP region when contiguous | fused QKV/region for shape | three generic projections | three SDOT/B×GEMV paths | KAI prepared heads/region |
| Decoder per item INT8 conv | native conv only on supported ISA | not a batch operation | **missing** integer conv; f32/BLAS | **missing** integer conv; f32/BLAS | native VNNI conv when `QWEN_SD_INT8` and square shape | AMX decoder D/BF16 options, not the ordinary VNNI leaf | NEON/f32 unless dotprod | native SDOT conv only with decoder INT8 opt-in | KAI decoder BF16/MLP only in qualified prepared modes |
| Decoder ragged batch | per-item fallback for one frame/slot | exact batch only with explicit gates | no AVX2 int8 batch; BLAS/f32 | no VNNI decoder batch; BLAS/f32 | **not reached by default**; batch gate falls back to per item | native/AMX decoder batch if exact+AMX | per-item/generic | **not reached by default**; per item if enabled | KAI/prepare path only where decoder mode explicitly enables it |

### C1/C2/C4/C8 implications

| Server concurrency C | Typical live B_eff | What is proven by current code |
|---:|---|---|
| C1 | always 1 | Solo Talker/CP and per-item decoder. No B>1 kernel is reachable. |
| C2 | 1 or 2 | B=2 only when both slots are runnable and contiguous; any lead/priority/QoS pause returns to B=1. |
| C4 | 1..4 | The scheduler can produce every width in that range; a configured C4 does not prove native B4. |
| C8 | 1..8 | Same ragged behavior; B>1 paths are exercised only during overlapping runnable windows. |

The selection report's B=1/2/4/8/16 probes (`qwen_kernel_selection_report`) are useful dispatcher probes but are not server evidence: they do not report active masks, contiguity, or the leaf selected for each live step.

## F. Existing code that v2 does not exploit or exploits only narrowly

| Finding | Evidence | Classification |
|---|---|---|
| KAI dotprod 1x implementation was gated with the i8mm backend | `qwen_tts_kleidi_dotprod.c` now has a separate dotprod pack/registry; full KAI still owns i8mm GEMM/regions | **FIXED OPT-IN**; native Linux qualification remains |
| AVX2/no-VNNI INT8 and Q4 B1 candidates are opt-in and not default server coverage | `qwen_matvec_int8/q4_0()` check `QWEN_AVX2_*_GEMV`; default falls to FP32/dequant | P2 candidate, not a wiring bug; preserve negative benchmark |
| AVX-512 no-VNNI has no AVX512 integer GEMV/GEMM leaf | `qwen_matmat_int8/q4_0()` selects AVX2; no VNNI/AVX512 integer guard | P2 only after complete-call/downclock evidence |
| SDOT INT8 B>1 matmat is opt-in; Q4 B>1 is B×GEMV | `QWEN_INT8_SDOT_MM`; explicit Q4 B-column loop | P1 policy/observability, P2 new kernel |
| Fused x86 QKV does not cover AVX2/no-VNNI, Q4, or ordinary Arm | `qwen_matmat_int8_qkv/qwen_matmat_bf16_qkv` return 0 outside AMX/VNNI/BF16 x86 gates | P2 only if complete-call shapes justify |
| VNNI/SDOT decoder kernels are bypassed by ordinary ragged batch | `qwen_speech_decoder_decode_streaming_batch()` fallback condition | P0/P1 reachability fix or explicit design decision |
| B_eff>16 misses native matmat gates | `g_mm_gate.max_b=16`; server passes live B_eff with no chunking | P0 wiring/gating |
| Batch-ceiling diagnostic does not match the batched call graph | The warning in `qwen_tts_server.c:3147-3154` describes one GEMV per slot, but `qwen_batch_proj_q()` calls `qwen_matmat_int8()`/`qwen_matmat_q4_0()`; declined gates reach the fixed-B/generic f32 matmat fallback (`qwen_tts_talker.c:2319-2323`, `qwen_tts_kernels.c:4937-4943`) | P1 observability/diagnostic defect |
| Q8 repack B1 is wired; B>1 requires i8mm | GGUF registration and `qwen_q8r_matmul()` | Active B1; P1 gap/qualification for dotprod-only B>1 |
| Ingot AVX2/AVX512/NEON Q4_K/Q8/BF16/F16 code has no production qwen caller | source call graph | Compiled but unreachable/dead from v2 |
| `qwen_tts_kernels_{avx,neon,generic}.c` are stubs | one-line files; implementation is monolithic | Not a missing backend, only misleading file split |

## G. Capability and observability defects

1. **KAI gate was stricter than the GEMV contract (P1; fixed opt-in).** Dotprod 1x build/runtime support is now separate from i8mm GEMM/region support. The candidate uses its own packing and remains opt-in pending native complete-call qualification.
2. **BF16 macro contract (fixed; compile verified).** The Makefile and KAI source guard now use `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`, matching the vendor BF16 translation units. Native Arm runtime qualification remains pending.
3. **First-use environment caching is order-sensitive (P1).** `qwen_mm_use`, `qwen_kleidi_enabled`, and decoder flags cache environment decisions atomically. Running `--caps`, `--dispatch-map`, self-test, or a library probe before server initialization can freeze a value before the server applies its environment. The server should initialize flags before any report or explicitly reject late changes.
4. **The reports are too coarse to prove v2 reachability (P1).** `qwen_tts_dispatch.c` exposes compiled/supported/resolved rows and representative B probes, but not stage, active mask, B_eff, contiguity, region use, or fallback reason per server step. Add a default-off low-cost histogram/event row to the existing census.
5. **The capability report omits some audited features (P2 observability).** `--caps` reports AVX/AVX2/FMA/AVX512F/BW/VNNI/BF16 and AMX-int8, but not F16C, BMI1/2, AVX512VL/DQ, AVX512-FP16, or an explicit AMX-BF16 runtime line. Their absence is a reporting gap, not evidence of support.
6. **Prefill policy is not symmetric with serving.** BF16 prefill can be forced to BLAS conversion even when a native matmat backend exists unless the native gate and chunk shape pass; INT8 prefill is opt-in. This must be visible in the per-stage census before changing defaults.
7. **The ISA class must be split before qualification (P1).** Add an explicit released-main class for AVX-512F/BW/VL without VNNI (and retain the AVX2 requirement for its integer fallback). Otherwise `--dispatch-map`, startup reports, and profile manifests collapse two physically different tiers and make a no-VNNI comparison hard to reproduce.

## H. Missing optimization opportunities, ranked

### P0 — existing path is not reliably wired or proven

* Add per-stage v2 census fields: server capacity C, runnable count, `B_eff`, active contiguity, operation/weight format, selected path, selected leaf, and fallback reason. Aggregate B=1..16 and `>16` without logging every token by default.
* Resolve B_eff>16 explicitly: either enforce a documented server/kernel ceiling, chunk the active cohort into supported widths, or add a deliberate generic path with a visible census state. Do not let the current max-B gate silently decide.
* Decide and prove decoder batching for ordinary VNNI and SDOT. If the exact ragged decoder batch is intentionally AMX-only, report that as a deliberate capability; otherwise wire the existing per-item integer kernels into the batch scheduler and qualify parity.

### P1 — current fallback/gate is clearly narrower than existing code

* Keep the new KAI dotprod GEMV opt-in and make q8-repack B1/B>1 behavior explicit on a native Arm host.
* Make SDOT B>1 policy shape-aware and visible. Retain the Apple M1 evidence where B×SDOT GEMV beat the SDOT matmat candidate; no global promotion follows from that result.
* Initialize dispatch flags before caps/dispatch-map/self-test or remove first-use caching for server-controlled flags.
* Add parity/census coverage for CP and Talker so the same B_eff reaches the same family policy; currently CP regions and QKV have narrower conditions.

### P2 — new kernels or wider formulations require measurement

* AVX2/no-VNNI INT8 and Q4 B1 native GEMV, including complete quantization/correction, tails, packing, and full streaming effect. The existing signed-widening INT8 candidate is known to lose in at least one comparison and must not be promoted by existence alone.
* AVX512-no-VNNI B>1 INT8/Q4 512-bit formulations. First compare complete-call cost against the existing AVX2 path, memory behavior, and AVX-512 frequency/downclock on a Skylake-SP-class host.
* Fused AVX2/Arm QKV only if the three-projection fallback dominates measured serving traces and packing can be reused.
* Decoder AVX2/no-VNNI integer convolution only after decoder batch/per-item reachability is measured; the current source has no such production body.
* Promote the dotprod-only KAI GEMV candidate only if native complete-call measurements beat the existing SDOT/B×GEMV paths; the implementation and parity hook are now present.

### P3 — theoretical or low-priority

* SVE/SVE2/SME kernels, AVX512-FP16/BF16 refinements, BMI1/BMI2, F16C integration, and AMX FP16. None is a first-pass fix for the current v2 reachability gaps.

## I. Hardware qualification matrix

This is a compatibility/qualification matrix, not a performance claim. Every target must record `lscpu`/`getauxval`, compiler profile, binary hash, server C, observed B_eff histogram, and resolved census.

| Tier to prove | Representative physical target | ISA question | Required proof |
|---|---|---|---|
| Scalar/SSE baseline | older x86-64 server without AVX2 | Does startup choose scalar safely and preserve all fallbacks? | scalar build plus runtime negative test |
| X1 AVX2/FMA | Haswell/Broadwell; AMD Zen/Zen2/Zen3 (Rome/Milan) | B1 INT8/Q4 fallback/candidate, B>1 AVX2 matmat, BF16 prefill and decoder | native Linux C1/C2/C4/C8, B histogram, complete-call benchmark |
| X2 AVX512 no-VNNI | Skylake-SP with F/BW/VL and no VNNI/AMX | Does the wide build use AVX512 FP helpers while integer paths remain AVX2? | `lscpu` feature check, dispatch map, frequency and streaming A/B |
| X3 VNNI | Cascade Lake, Ice Lake, Sapphire Rapids | Which Talker/CP/QKV/decoder leaves use VNNI at B1/B>1? | per-stage census and decoder batch test |
| X4 AMX | Sapphire Rapids/Granite Rapids with AMX permissions | Are AMX gates reached only at B/shape thresholds and permission succeeds? | AMX permission check, B1 vs B3/B4/B8, decoder AMX mode |
| A1 plain NEON | Neoverse N1 / Ampere Altra / Graviton2 | End-to-end generic/f32 paths, no accidental dotprod assumption | Linux HWCAP and C1/C4 stream |
| A2 dotprod-only | Neoverse N1/Graviton2/Altra or Apple M1-like dotprod without i8mm | SDOT B1, SDOT opt-in B>1, Q4 B×GEMV, optional KAI dotprod B1 | verify no i8mm, candidate census leaf/reason and complete-call A/B |
| A3 dotprod+i8mm | Graviton3/V1-class with i8mm or equivalent | SMMLA/KAI GEMM and q8-repack B>1 | separate native GEMV/GEMM and prepared-region tests |
| A4 BF16 Arm | Neoverse V1/V2 or Graviton3/4 exposing BF16 | BFMMLA/BFDOT/KAI BF16 guards and defaults | compiler macros, HWCAP, B2/B4 prefill/step |
| Apple separately | M1, then M2/M3/M4 class | Darwin sysctl, MMLA defaults, Accelerate prefill | no Linux HWCAP assumptions; explicit env/default census |

## J. Cloud candidate matrix (availability is secondary)

Provider pages identify the processor family but do not replace a guest feature check. A VM can expose a masked CPU, and a family can contain more than one processor generation. Before any qualification use `lscpu`, `/proc/cpuinfo`, and for Arm `getauxval(AT_HWCAP/AT_HWCAP2)` from the actual guest.

| Provider | Candidate | Documented processor class | Intended audit tier | Qualification note |
|---|---|---|---|---|
| AWS | C5/C5n, C5d | Intel Xeon Platinum 8124M (Skylake-SP) and 8275CL (Cascade Lake) | X2 no-VNNI candidate on 8124M; X3 candidate on 8275CL | The current AWS table lists both processors under C5; verify exact size/host flags. |
| AWS | C5a, C6a | AMD EPYC 7R32 (Rome/Zen2), 7R13 (Milan/Zen3) | X1 AVX2/FMA | Verify SMT/topology and flags; no VNNI assumption. |
| AWS | C6i | Intel Xeon Ice Lake | X3 VNNI/BF16 qualification | Verify AVX512-BF16 exposure separately from AVX512/VNNI. |
| AWS | C7i | Intel Xeon Sapphire Rapids | X3/X4 VNNI/AMX | AMX permission and VM exposure must be tested. |
| AWS | C7a | AMD EPYC 9R14 (Genoa) | X1 AVX2/FMA, modern control | Do not infer VNNI/AMX from the family label. |
| AWS | C8a | AMD EPYC 9R45 (Turin) | future control, not old-CPU proof | Use as a modern comparison only. |
| AWS | C6g | Graviton2 | A1 plain NEON/N1-like | `lscpu`/HWCAP required; no i8mm assumption. |
| AWS | C7g/C7gn | Graviton3/3E | A3 dotprod+i8mm candidate | Confirm dotprod, i8mm, BF16 flags in the guest. |
| AWS | C8g | Graviton4 | A4 modern Arm control | Verify exposed BF16/SVE2 features; engine currently has no SVE2 backend. |
| GCP | C2/C2D | Cascade Lake / AMD EPYC Milan | X3 or X1 | Current C2 documentation names Cascade Lake; use a separately exposed Skylake platform for a strict no-VNNI proof. |
| GCP | C3/C3D | 4th-gen Intel Xeon / AMD EPYC Genoa | X3 or X1 modern controls | Read the guest CPU platform; do not infer BF16 from “C3”. |
| GCP | C4/C4D | Granite/Emerald Rapids / AMD EPYC Turin | X4/modern control | AMX and AVX512 feature exposure must be checked. |
| GCP | T2A | Ampere Altra, Neoverse N1 | A1 plain NEON | Official docs state one vCPU per core and no SMT; useful for clean A1 tests. |
| GCP | C4A | Google Axion, Neoverse V2 | A3/A4 modern Arm | Verify i8mm/BF16/SVE2; current engine will use only implemented NEON/dotprod/i8mm/BF16 paths. |

Official references consulted on 2026-09-19: [AWS compute-optimized instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html), [AWS general-purpose instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html), [Google Compute Engine CPU platforms](https://docs.cloud.google.com/compute/docs/cpu-platforms), [Google general-purpose machine families](https://docs.cloud.google.com/compute/docs/general-purpose-machines), and [Google Arm VMs](https://docs.cloud.google.com/compute/docs/instances/arm-on-compute).

## K. Minimum instrumentation proposal

No instrumentation change was made in this audit. The minimum useful addition is a default-off extension of the existing census, emitted at server shutdown or an explicit diagnostic endpoint:

```text
stage=talker|cp|decoder|prefill
capacity_C=<server capacity>
runnable=<active runnable slots>
B_eff=<packed kernel width>
contiguous=0|1
weight=int8|q4|bf16|fp32
operation=gemv|matmat|qkv|conv|sgemm
path=<path enum>
leaf=<leaf enum>
fallback_reason=<solo|gate|max_b|noncontiguous|env|unsupported|...>
count=<calls>
```

This must use the already resolved census/path enums, remain low overhead when off, and distinguish server C from B_eff. It is sufficient to prove C1/C2/C4/C8 and to show whether a candidate is reached by Talker, CP, prefill, or decoder.

## L. Validation performed

The clean `main` archive was built with:

```text
make blas
```

The build completed successfully on Darwin arm64 (Apple M1-class). The resulting binary ran:

* `./qwen_tts --caps`: reported NEON + SDOT, no BF16 matmul, and B>1 INT8/Q4 generic/B×GEMV behavior.
* `./qwen_tts --dispatch-map`: reported `apple_m1`, native SDOT GEMV, opt-in SDOT matmat, Q8 repack, no KAI, decoder batch off, and per-item FP32 decoder mode.
* `./qwen_tts --self-test`: **SELF-TEST PASSED (0 cases failed)**, including matmat parity, INT8/Q4 paths, decoder conv/v2/multislot correctness, quantization, and attention.

This validates build/dispatch/parity on the available Arm host. It does not qualify Linux x86/Arm feature exposure, performance, frequency behavior, cloud topology, or production stream capacity.

## M. Recommended implementation sequence

1. Add the default-off per-stage B_eff/leaf census and make fallback reasons visible.
2. Fix or explicitly cap/chunk B_eff>16; add tests for C1/C2/C4/C8 with ragged masks.
3. Decide and wire the decoder VNNI/SDOT batch policy; preserve a per-item control path and parity tests.
4. Split KAI dotprod GEMV from i8mm GEMM/region gates; resolve the BF16 macro contract.
5. Run native-host dispatch and complete-call microbenchmarks, then v2 continuous streaming tests with the same source/feature manifest.
6. Only after those gates, evaluate P2 AVX2 B1, AVX512-no-VNNI width, QKV, or decoder kernels. Promote nothing from an intrinsic or benchmark-only caller without production reachability evidence.

## N. Implementation update on `feature/old-cpus-simd` (2026-09-19)

The released-main findings above remain the historical baseline. The first implementation
milestone is now present in the working branch (preserved work is committed separately as
`76ddee9c`). No production SIMD default was changed.

| Original finding | Current state | Evidence / remaining proof |
|---|---|---|
| P0 per-stage v2 reachability was not observable | **INSTRUMENTED — NEEDS NATIVE QUALIFICATION** | `qwen_tts_v2_census.c/.h` adds an off-by-default aggregate census with stage, capacity C, runnable count, B_eff, contiguity, weight, operation, path, leaf, reason, count and MACs. Talker, CP, prefill, decoder, and the Talker/CP region runners are wrapped. `QWEN_V2_CENSUS=1` emits deterministic CSV; `QWEN_V2_CENSUS_JSON` writes a machine-readable artifact. Local unit coverage is in `make test-v2-census`. |
| B_eff > 16 silently misses optimized gates | **FIXED AS AN EXPLICIT CONTROL; DEFAULT POLICY UNCHANGED** | `QWEN_BATCH_CHUNK_MAX_B=2..16` enables ordered `16+tail` chunking in the existing Talker projection helpers. The default remains unset, so native qualification can compare the prior generic B>16 behavior with the chunked control. Partition tests cover B=1..32; numerical serving parity still needs a model-backed run at B=17/20/24/31/32. |
| Ordinary VNNI/SDOT decoder batch bypass | **INSTRUMENTED — POLICY STILL OPEN** | The batch entry now records the exact decoder policy and actual AMX-BF16/AMX-INT8 path hint before either the ragged body or the per-item fallback. The released semantic gate is unchanged: plain VNNI/SDOT INT8 remains the control until an equivalent multislot implementation is proven. Native VNNI/SDOT parity and performance are still required before wiring it. |
| KAI dotprod-only support gap | **IMPLEMENTED — OPT-IN; NEEDS LINUX NATIVE QUALIFICATION** | `qwen_tts_kleidi_dotprod.c` adds a separate dotprod-only registry and distinct Q4/int8 RHS packing for B=1 GEMV. `Makefile` now builds the vendor dotprod packers/kernels when `__ARM_FEATURE_DOTPROD` is present and adds i8mm objects only for `__ARM_FEATURE_MATMUL_INT8`. Existing i8mm GEMM/region APIs and gates remain unchanged. `QWEN_KAI_DOTPROD_GEMV=1` is default-off; `make test-kai-dotprod` proves pack/kernel parity on the local Apple dotprod host. Linux HWCAP, model reachability, and performance remain required. |
| BF16 macro contract | **FIXED — COMPILE VERIFIED; NATIVE QUALIFICATION PENDING** | The Makefile and KAI source guard now use `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`, matching every vendor BF16 translation unit. `make check-isa` covers the Arm BF16 profile; runtime BF16/KAI behavior still needs a Linux Arm host. |
| AVX-512 no-VNNI classification | **FIXED IN PRESERVATION COMMIT** | The dispatch/reporting change from the previous work distinguishes `x86_avx512_no_vnni`; the new v2 census keeps the actual AVX2 integer leaf separate from wide FP/helper classification. |
| First-use environment cache ordering | **OPEN — NEEDS POLICY CONTRACT TEST** | The census does not change cache semantics. Standalone `--caps`/`--dispatch-map` exit before server startup, while server defaults are applied before continuous serving. A dedicated initialization contract test is still needed before changing the existing cached policies. |

### New control and qualification facts

* Chunking is deliberately opt-in and clamps at 16, matching the current optimized gate. It
  preserves active-slot ordering for both indexed/ragged and contiguous batches; the helper
  unit test proves partition coverage, while model-backed `--batch-test` remains the numerical
  oracle.
* Census hooks run even when the older shape census is disabled. Existing dispatcher path and
  leaf enums are reused, so a row reports the production leaf rather than a benchmark-only
  label. Region runners receive an envelope row because their worker kernels do not enter the
  ordinary dispatcher hooks.
* The current Apple host proves compile and kernel parity only. It cannot execute the x86
  VNNI/AVX2/AVX-512/AMX or Linux Arm KAI branches. Rows marked native qualification therefore
  remain unresolved until the prepared host suite is run.
* `tools/cpu_qualify.py` and `make cpu-qualify` now package the paid-host procedure. The
  model-free mode captures hardware/HWCAP/compiler/provenance, `--caps`, `--dispatch-map`,
  compile checks, native/fallback self-tests, matmat parity, and ISA-candidate A/B logs. With
  `--model`, it runs the Talker/CP batch oracle at the requested B values; `--serve` adds the
  C1/C2/C4/C8 wave and fails if the v2 census JSON is not produced. Missing model/server work
  is labelled `NATIVE RUNTIME REQUIRED` rather than silently treated as passed.
* The first smoke run exposed a real integration omission: the standalone matmat parity target
  did not link the new census module. `PARITY_SRC` now includes `qwen_tts_v2_census.c`; native
  and Rosetta parity both pass again.
* The KAI split was implemented without changing a production default. On a dotprod-only build,
  model registration delegates B=1 Q4/int8 weights to the separate candidate registry only when
  `QWEN_KAI_DOTPROD_GEMV=1`; B>1 continues through the existing SDOT/generic policy. Dispatch
  reports now show `kleidi.dotprod_gemv` independently from `kleidi.enabled` (full i8mm KAI).
* `make test-kai-dotprod` passed locally for adversarial synthetic Q4/int8 rows. This is vendor
  ABI/packing and arithmetic parity evidence, not a claim that the candidate wins against native
  SDOT or that Linux HWCAP exposure is correct.
* The qualification harness runs the dotprod parity target and includes a `kai_dotprod_gemv`
  opt-in arm in model-free A/B manifests. Complete-call A/B still requires a model and a Linux
  dotprod-only host; the candidate remains default-off until that measurement.
