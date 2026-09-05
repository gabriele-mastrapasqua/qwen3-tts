# Current Plan (local, untracked — ENGINEERING.md §1)

Goal: backend parity of the runtime work, and a proven (not inferred) explanation of the
Arm/x86 serving gap. Addenda in `.work/`; the old long plans (`plan_profile_cpu.md`,
`plan_x86_parity.md`) are history, read only when a task points at them.

## P0 — correctness of our performance evidence

- [x] P0.0 GCP c4 fallback: explicit `QWEN_PREFILL_MATMAT=1` on a non-BF16 build ran the
      generic twin; resolver + gate fixed (6ce84e4). Old c4 numbers marked unqualified.
- [ ] P0.1 Audit AWS c8a canonical runs: proven actual path per stage from the box artifacts
      (perf symbols already show bf16_matmat_avx512_m* and int8_vnni; leaf census never run)
- [ ] P0.2 In-process preflight: server prints the resolved table after env; the census
      compares ACTUAL leaves with an expected/allowed/forbidden manifest per operation and
      profile (native-BF16 prefill: f32/generic forbidden; decoder serial OpenBLAS SGEMM may
      stay allowed; auto: report resolved, never fail for differing from another backend)
- [ ] P0.3 Explicit request vs `auto` distinguished in the gate (auto reports, never fails)
- [ ] P0.4 `run_manifest.json` in every benchmark directory, produced from the serving
      configuration after env: commit, dirty, binary hash/build id, CPU/features, compile
      SIMD, profile, exact env, topology/masks, requested + resolved dispatch per operation.
      A pre-env dispatch file is never authoritative
- [ ] P0.5 Canonical benchmark runbook audited against current code/scripts/profiles
      (`docs/BENCHMARKING.md`, written 2026-09-05 from the tree). Open: `bench-matrix`/`bench.sh`
      still called "the per-box report" in hardware-testing.md; `bench-server` microbench vs
      WAVE; no tracked `perf` wrapper; `run_manifest.json` not produced yet (P0.4)

## P1 — backend maturity matrix

- [x] P1.0 HEAD compiled and ran only as the working tree: conv scratch freed, prctl include
      missing — fixed in fb3d32d; rule ENGINEERING.md §13
- [x] P1.1 Static cross-backend audit — `docs/cross-backend-audit-2026-09-05.md` (fb3d32d)
- [ ] P1.2 Verify Arm KleidiAI GEMV/GEMM shape coverage on a box
- [ ] P1.3 Verify VNNI GEMV/GEMM shape coverage (q4 GEMM 0.80x is the known weak one)
- [x] P1.4 Verify AMX B>=4 region/head paths — detail: `.work/p1-4-amx-runtime.md`
      Runtime verified on GCP AMX host: production-like C4 decode remains VNNI at
      observed B1/B2; AMX BF16 prefill executes; AMX INT8 observed only at C8 B4/B5.
      `suspicious=1` was a census reporter false positive (B>=2 vs real AMX INT8 B>=4).
      P2/P3 implementation follow-ups are now unblocked.
- [ ] P1.5 Verify AVX2 / AVX-512F non-VNNI fallbacks (no int8/q4 GEMV, f32 SGEMM prefill)
- [ ] P1.6 Verify Apple/GCD/Accelerate runtime (SGEMM partition with Accelerate threads)

## P2 — runtime parity (evidence for the whole block — detail: `.work/p2-cross-backend-runtime.md`)

- [x] P2.1 CP region ported to the AMX in-region runner. `qwen_i8mm_usable/qkv_usable` no
      longer return 0 for AMX shapes; `qwen_i8mm_run/run_qkv` pack the activation per thread
      and call `int8_amx_task`/`int8_qkv_amx_task`. Box-proven: AMX build B=4 runs
      "AMX int8 tiles" in-region, bit-identical to the dispatched path (md5 1c580479)
- [x] P2.2 Talker region inherits the same runners and gates (validated in the same run)
- [x] P2.3 Batched CP heads decoupled from the VNNI-only gate: available on AMX at B>=4,
      where they previously fell back to one GEMV per slot. Bit parity, not just argmax
- [x] P2.4 Hot-path allocations removed (bf16 AMX/BFMMLA Xb, bf16 QKV Xb, q4 B-x-matvec,
      q8 repack B>1, Apple snake) -> grow-once TLS scratch. M1 self-test + golden 4/4
- [ ] P2.5 Execution-budget ownership outside the batched server (plain `--serve`, CLI)
- [x] P2.6 Pool interface made truthful: GCD and Windows report `qwen_parallel_active()`
      from a real TLS depth (nested callers run inline; on Windows this also removes a
      latent single-job-slot deadlock), `qwen_parallel_team()` returns 0 = "no holdable
      team" instead of a fake 1. Decoder SGEMM partition now gated on
      `qwen_blas_own_effective()` (real BLAS thread control), not on the claim

## P3 — feature/knob parity

- [x] P3.1 Authoritative getenv inventory from code (audit §3; 7 GPU names registered)
- [x] P3.2 KAI_* to VNNI/AMX/AVX2 equivalents (audit §3 table)
- [ ] P3.3a Static audit of Arm/x86 pool spin/wake defaults: getenv wiring, compile/runtime
      gates, callsites — what each backend does today, before any cloud time
      detail: `.work/p3-runtime-knob-parity.md` (knob tables done; thread.c submit/wait path still to read)
- [ ] P3.3b Cloud A/B only for the differences P3.3a leaves unresolved
- [ ] P3.4 Decoder int8-conv default Arm vs VNNI (measure the first-frame cost)
- [x] P3.5 AMX/x86-QKV/decoder knobs documented in docs/feature-flags.md; region rows
      updated for AMX; `QWEN_CP_FRAME_REGION` added. check-flag-registry 180/180

- [ ] P2.7 ARM persistent region BLOCKED, not skipped: the SMMLA row-block runner exists
      and shares `quantize_act_int8_col`, but `kai_i8_try` takes the shape first on every
      build that has i8mm, so an in-region SMMLA runner would change the numbers vs today's
      KleidiAI output. The output-preserving fix is a KleidiAI pack/run split
      (`qwen_kleidi_i8_prep` + per-thread n-block run). Needs an Arm i8mm box; M1 is
      dotprod-only and cannot compile or run it

## P4 — architecture cleanup (no implementation before P0-P3 are understood)

- [ ] P4.1 Draft module boundaries: common runtime / cpu dispatch+caps / cpu arm / cpu x86 / gpu
- [ ] P4.2 Promote `g_mm_gate[]` + compiled/supported predicates to the one capability table,
      extended to non-matmat capabilities (regions, heads, conv int8, prefill, budget)
- [ ] P4.3 Make fallback selection observable and testable from that table

## Later

- [ ] Rename legacy CLI `--batch/--batch-words/--batch-dry` (deprecation alias)
- [ ] Custom decoder fp32 GEMM evaluation (non-bitwise acceptance first)
- [ ] INT8 prefill quality qualification (`QWEN_PREFILL_INT8MM`, separate numerical path)
- [ ] Decide whether to publish the pending GCP C3D/8c VNNI reference note (not in this
      committed tree) and the stray `main-5b5256e9.o.tmp`
- [ ] P5.0 [low] Set `QWEN_POOL_SPIN=65536` as the x86 server default and update related JSON profiles.
- [ ] P5.1 [low] Compare AutoRound/LLM Compressor W4A16 and Intel ARK packed kernels with runtime INT8: https://vllm.ai/blog/2025-12-09-intel-autoround-llmc https://github.com/intel/auto-round/tree/main/auto_round_extension/ark
- [ ] P5.2 [low] Run isolated Xeon AMX/VNNI GEMV/GEMM oracle probes with oneDNN benchdnn and OpenVINO CPU: https://github.com/uxlfoundation/oneDNN/tree/main/tests/benchdnn https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.rst
- [ ] P5.3 [low] Audit vLLM CPU, oneDNN and IPEX prepacking/fusion/cache behavior against CP, Talker and INT8 conv: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/A-Practical-Guide-to-CPU-Optimized-LLM-Deployment-on-Intel-Xeon/post/1737233
- [ ] P5.4 [low] Run GCP oracle probes only from an isolated copied folder and only at 0% CPU with no competing workload.

### X86 dataflow follow-ups

- [x] X86-1 Direct source-row INT8 quantization for CP/Talker regions — landed in `d847d9c`; details: `docs/x86-int8-dataflow-2026-09-05.md`
- [ ] X86-2 Low-B x86 AMX/VNNI crossover — validate B1/B2 on real engine shapes, not only oneDNN.
- [ ] X86-3 Persistent packed-RHS consumption audit — prove hot B1/B2 projections consume the packed representation.
- [ ] X86-4 Activation preparation/fusion follow-up — remove remaining generic gather, q8-pack, and scatter passes.
- [ ] X86-5 AMX activation-pack reuse — check reuse across workers inside a held region.
- [ ] X86-6 Real 0.6B server qualification — repeat shape-oracle conclusions with the actual checkpoint and C1/C2/C4 screens.
- [ ] X86-7 0.6B vs 1.7B serving profile — compare Talker, CP, and decoder time shifts after the Talker width change.
