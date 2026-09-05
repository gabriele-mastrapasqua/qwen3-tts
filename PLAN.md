# Current Plan (ENGINEERING.md §1)

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
- [x] P2.5 One execution budget for every entry mode (e78e861): `qwen_exec_budget_engine_owned()`
      called by CLI, plain `--serve` (incl. prefork children), batched server. No-op at one
      thread, explicit QWEN_SD_POOL/QWEN_BLAS_OWN still win, private decoder team never created.
      macOS reports "claimed but no thread control"; the BLAS half is Linux+OpenBLAS only
- [x] P2.6 Pool interface made truthful: GCD and Windows report `qwen_parallel_active()`
      from a real TLS depth (nested callers run inline; on Windows this also removes a
      latent single-job-slot deadlock), `qwen_parallel_team()` returns 0 = "no holdable
      team" instead of a fake 1. Decoder SGEMM partition now gated on
      `qwen_blas_own_effective()` (real BLAS thread control), not on the claim

## P3 — feature/knob parity

- [x] P3.1 Authoritative getenv inventory from code (audit §3; 7 GPU names registered)
- [x] P3.2 KAI_* to VNNI/AMX/AVX2 equivalents (audit §3 table)
- [x] P3.3a Pool capability parity (47ede94): `qwen_parallel_is_reentrant()` stood for three
      different questions and on pthread returned the QWEN_PREFILL_HELPER opt-in, so a feature
      flag drove decoder-team and server-serialisation policy. Replaced by
      `qwen_pool_nested_dispatch_ok()` / `qwen_pool_concurrent_submit_ok()`, both reported in
      the dispatch map. NOTE: on Linux the legacy threaded `--serve N` path therefore stops
      serialising synthesis by default (per-worker contexts, submit_mtx makes it safe);
      prefork is unaffected. detail: `.work/p3-runtime-knob-parity.md`
- [ ] P3.3b Cloud A/B only for the differences P3.3a leaves unresolved
- [x] P3.4a Decoder int8-conv capability split from policy (47ede94): `qwen_sd_int8_available`
      (kernels compiled) + `qwen_sd_int8_usable` (shapes the kernels cover, moved off the
      decoder call site) vs a named per-backend default with its reason. AVX2/AVX-512F have no
      int8 conv kernel at all, so there is nothing to wire there — not made symmetric on purpose
- [ ] P3.4b Measure the ARM dotprod first-frame cost and decide whether it may default ON
- [x] P3.7 Path selection observable (8aa95db): `matmat.{int8,q4,bf16}.family` name the family
      that actually serves each dtype (dispatcher order, resolved through `qwen_mm_use`), so a
      build where every gate is off no longer stays silent. Audited AVX2/AVX-512F for removable
      waste and found none: the bf16 generic is fixed-B twins converting each weight once, the
      AVX2 int8 matmat is a real maddubs kernel with no row-sum correction to cache
- [x] P3.8 `QWEN_PREFILL_LOW_MS` was a no-op on GCD/Windows and said nothing (8aa95db):
      `qwen_pool_priority_ok()` + `pool.submit_priority`, and the prefill helper reports the
      knob as ignored instead of pretending
- [x] P3.9 Renamed `qwen_i8mm_*` -> `qwen_region_i8_*` (8aa95db): the name became false when the
      AMX tiles became a valid in-region runner
- [x] P3.10 x86 SIMD activation-panel quantiser (9933948). Reference is the SCALAR expression
      (round-half-away-from-zero + truncate, clamp [-127,127]) — NOT lrintf — reproduced with
      copysign+cvttps; amax is an order-independent max reduction. Permanent `--self-test`
      parity gate (8 input classes, byte- and bit-exact) that reports n/a instead of a green
      tautology where the x86 path is absent. Box: all cases equal, identical WAV md5 with the
      path on/off, function 428->278 ns (96x7 column) to 83.1->26.8 us (96x672 panel), one shape
      +4% where gcc already auto-vectorises, end-to-end -0.8%. im2col is pure memcpy, no asymmetry
- [ ] P3.11 NEON `qwen_int8_quant_rows` rounds half-to-EVEN (`vcvtnq_s32_f32`) while the scalar
      tail of the same function rounds half-AWAY-from-zero, so within one row the vector body and
      the tail disagree on .5 boundaries. Predates all of this; ARM-only; decide which rounding is
      the contract before touching it, and gate it with the same parity test
- [ ] P3.6 AVX2 / AVX-512F have int8+q4 GEMM but NO integer GEMV, so every B=1 dequantises to
      the f32 fused twin. Now visible (`matvec.int8.native` / `matvec.q4.native`, 47ede94); the
      fix is a kernel, not a gate — no wasted conversion exists to remove and reusing the GEMM
      at B=1 would change the arithmetic
- [x] P3.5 AMX/x86-QKV/decoder knobs documented in docs/feature-flags.md; region rows
      updated for AMX; `QWEN_CP_FRAME_REGION` added. check-flag-registry 180/180

- [ ] P2.7 ARM persistent region: interface COMPLETE, wiring blocked on hardware (8aa95db adds
      the fused Q/K/V phases beside the plain ones, so every shape the region needs is exposed).
      Remaining: give the region body a row-major gather shape; needs an ARM i8mm box.
      Original analysis: blocker NARROWED, no longer numerical (e78e861).
      `qwen_kleidi_i8_region_usable/_prep/_run` expose the pack and n-block phases the
      KleidiAI int8 path already had, so a region can reuse the exact dispatched kernel and
      output. What remains: the CP/Talker region body gathers k-major `[cols][B]` for the
      shared per-column quantiser while KleidiAI wants row-major activations it quantises
      itself, so the region needs a second gather shape. Needs an Arm i8mm box to validate.

## P4 — architecture cleanup (no implementation before P0-P3 are understood)

- [ ] P4.1 Draft module boundaries: common runtime / cpu dispatch+caps / cpu arm / cpu x86 / gpu
- [ ] P4.2 Promote `g_mm_gate[]` + compiled/supported predicates to the one capability table,
      extended to non-matmat capabilities (regions, heads, conv int8, prefill, budget)
- [ ] P4.3 Make fallback selection observable and testable from that table

## Later

- [ ] Rename legacy CLI `--batch/--batch-words/--batch-dry` (deprecation alias)
- [ ] Custom decoder fp32 GEMM evaluation (non-bitwise acceptance first)
- [ ] INT8 prefill quality qualification (`QWEN_PREFILL_INT8MM`, separate numerical path)
- [ ] Decide `docs/reference-gcp-c3d-8c-vnni.md` (untracked, unreviewed) and the stray
      `main-5b5256e9.o.tmp`
- [ ] P5.10 [low, DEFERRED — not a current task] BLAS removal / replacement audit. Inventory the
      remaining hot-path OpenBLAS/SGEMM usage; measure the COMPLETE per-operation cost, not the
      GEMM arithmetic alone (layout and conversion work, thread-runtime overhead, the partition
      itself); compare against a fixed-shape custom kernel, oneDNN, and the native paths we
      already have; remove only where the measured win justifies the numerical and maintenance
      risk. Sequenced last on purpose: it is optimization work and waits until structural
      parity is substantially closed.
- [ ] P5.0 [low] Set `QWEN_POOL_SPIN=65536` as the x86 server default and update related JSON profiles.
- [ ] P5.1 [low] Compare AutoRound/LLM Compressor W4A16 and Intel ARK packed kernels with runtime INT8: https://vllm.ai/blog/2025-12-09-intel-autoround-llmc https://github.com/intel/auto-round/tree/main/auto_round_extension/ark
- [ ] P5.2 [low] Run isolated Xeon AMX/VNNI GEMV/GEMM oracle probes with oneDNN benchdnn and OpenVINO CPU: https://github.com/uxlfoundation/oneDNN/tree/main/tests/benchdnn https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.rst
- [ ] P5.3 [low] Audit vLLM CPU, oneDNN and IPEX prepacking/fusion/cache behavior against CP, Talker and INT8 conv: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/A-Practical-Guide-to-CPU-Optimized-LLM-Deployment-on-Intel-Xeon/post/1737233
- [ ] P5.4 [low] Run GCP oracle probes only from an isolated copied folder and only at 0% CPU with no competing workload.
      (P5.5-P5.9, the earlier private x86-dataflow probes, are folded into the X86-* section
      below: same questions, now against a tracked document instead of a `.work/` note.)

### X86 dataflow follow-ups — detail: `docs/x86-int8-dataflow-2026-09-05.md`

- [x] X86-1 Direct source-row INT8 quantization for CP/Talker regions — landed on this branch as
      `40c156f` (`d847d9c` in the worktree it was written in). Subsumes the old P5.6 staging half.
- [ ] X86-2 Low-B x86 AMX/VNNI crossover — validate B1/B2 on real engine shapes, not only oneDNN.
      Subsumes old P5.5 and P5.8 (oneDNN/benchdnn on real CP/Talker shapes, small-B VNNI vs oracle).
- [ ] X86-3 Persistent packed-RHS consumption audit — prove hot B1/B2 projections consume the
      packed representation. Was P5.9.
- [ ] X86-4 Activation preparation/fusion follow-up — remove remaining generic gather, q8-pack and
      scatter passes. Rest of old P5.6; the decoder half of old P5.7 is partly done by 9933948
      (x86 SIMD `qwen_int8_quant_rows`), the im2col fusion is not.
- [ ] X86-5 AMX activation-pack reuse — check reuse across workers inside a held region.
- [ ] X86-6 Real 0.6B server qualification — repeat the shape-oracle conclusions with the actual
      checkpoint and C1/C2/C4 screens.
- [ ] X86-7 0.6B vs 1.7B serving profile — compare Talker, CP and decoder time shifts after the
      Talker width change.
- [ ] RESOLVED cause of the `git add` anomaly (2026-09-05): not a git alias, hook or wrapper.
      A second agent stages files in the SAME working tree concurrently, so the shared index
      carries its work as well as ours. Mitigation used: build a commit through a private
      `GIT_INDEX_FILE` so the shared index is never disturbed. Open question: agree a staging
      protocol before two agents share a tree again.
- [x] `make test-serve-concurrent` http=415 fixed (ad8c1ef): the test sent its body with
      `curl -d` and no header, so it arrived form-encoded and the endpoint rejected it correctly.
      It now reaches the engine and fails on P6.2 instead, deterministically — same non-batched
      path, so it is not a production gate either.
- [x] P6.1 Pre-warm ran the wrong configuration (ccf8146): it generated with whatever the CLI
      left in the context (language_id -1, which no request uses) and so primed per-request
      state for a path nothing takes. The first request came out a different LENGTH from all
      the rest. `reset_request_state()` before the warm-up, not only after. One of two causes.
- [ ] P6.2 [NON-BLOCKING for production — screened on Linux x86, ad8c1ef] Sequential-history
      dependency: an identical request returns a different trajectory depending on the LENGTH of
      the request before it. Screen on the box, 1.7B --int8, sequence A A SHORT A LONG A LONG A A:
      batched + prefork 1 worker on `/v1/tts/stream` -> all five A identical (96000 samples, one
      md5), NO dependency; plain `--serve` on the same endpoint -> A alternates 96000/99840, two
      md5s. So it lives in the NON-BATCHED single-request server path (Linux and macOS alike);
      the batched/prefork production path is clean. Ruled out earlier: prefix cache,
      repetition-penalty history, streaming chunk size; a pre-warm generating the same text makes
      it vanish, so the suspect is a grown-once/recycled buffer whose valid length is implicit.
      Do NOT block CPU backend parity on it, and do NOT treat `make test-serve-repro` or
      `make test-serve-concurrent` as production gates: both exercise that non-batched path.
      CLI-vs-server hash divergence is expected and is NOT part of this item.
- [x] Renamed `qwen_i8mm_*` -> `qwen_region_i8_*` (8aa95db) — superseded by P3.9.
