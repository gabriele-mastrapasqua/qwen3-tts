# Current Plan (ENGINEERING.md §1)

Goal: backend parity of the runtime work, and a proven (not inferred) explanation of the
Arm/x86 serving gap. Addenda in `.work/`; the old long plans (`plan_profile_cpu.md`,
`plan_x86_parity.md`) are history, read only when a task points at them.

## PRIORITY ORDER (set 2026-09-05, after the decoder finding)

This replaces "pick the next interesting component". The decoder fix exposed a process
failure larger than any single defect: a component worth ~31% of the request was running its
most expensive kernel on 2 of 6 workers, and it took a manual dissection to see it. Weeks went
into individual kernels and backends while that sat in plain sight. So:

1. close ONLY PARITY-1/2/3 below — nothing else from the parity backlog;
2. then P0-PROFILER, which OUTRANKS all further x86 kernel work;
3. then P1-CONFIG, the server config control plane;
4. only then X86-4/X86-5/X86-6/X86-7/X86-8 and P5.10, and their queue order comes from what
   the profiler's FAST run measures, not from what looks interesting.

PARITY IS CLOSED ONLY WHEN ALL SIX HOLD. Not "ARM has this, x86 has something equivalent":

  1. every common semantic feature has a BACKEND MATRIX generated from the code;
  2. every runtime flag has requested / effective / default semantics PER BACKEND;
  3. every ARM performance feature has an explicit x86 status:
     equivalent / deliberately different / missing;
  4. no flag is silently inert anywhere;
  5. the server can dump its COMPLETE EFFECTIVE configuration at runtime — this does NOT wait
     for the profiler, it is config correctness, and without it the profiler could report 50 ms
     in AMX while an env everyone believed active had been ignored;
  6. the current GCP and AWS server profiles are reproducible FROM DECLARED CONFIG.

- [ ] PARITY-1 Common/shared code path parity across the supported CPU backends, expressed as a
      generated matrix with THREE distinct states per cell — implemented / selectable /
      ACTUALLY EFFECTIVE — because that is exactly where `QWEN_NO_SIMD_QUANT` hid: it existed,
      it looked like a feature, and on ARM it did nothing. Columns: common · ARM/KleidiAI ·
      AVX2 · VNNI · AMX · Apple · CUDA/Metal, plus flag, default and effective. Rows are
      semantic features, not functions: fused QKV · persistent region · activation reuse ·
      packed RHS · native GEMV · GEMM-used-as-GEMV at small B · snake SIMD · decoder int8 conv ·
      pool ownership · pool spin · BF16 prefill · fused activation quant · direct source-row
      quant · prepack lifetime parent->fork · batch-aware dispatch · row/block scheduling ·
      activation scratch reuse · full-sequence prefill GEMM.
      Open pieces: P2.7 (ARM region wiring, hardware-blocked), P4.1/P4.2.
- [ ] PARITY-2 Feature-flag / runtime-knob parity, and the EFFECTIVE-CONFIG DUMP that proves it.
      178 getenv calls cannot be the operating contract of a production server: env should mean
      debug, experiment, forced dispatch, kill switch and profiling — not configuration. Every
      artifact must therefore open with the effective server configuration AFTER parsing,
      defaults and capability gating, not with what happened to be in the environment:
          profile · prefork · threads/worker · SMT · per-worker affinity · pool_spin ·
          blas_owner · openblas_runtime_threads · prefill mode and chunk · prefill_helper ·
          amx_int8_min_b · amx_min_rows_per_thread · amx_prepack · decoder_int8 · ...
      and where a request cannot be honoured it must say so in that dump:
          foo.requested = 1 · foo.effective = 0 · foo.reason = unsupported_on_avx512vnni
      Builds on P3.1/P3.2; P3.3b is the open remainder.
- [ ] PARITY-3 Backend/dispatch functional parity, driven by ARM AS THE ORACLE. Do not ask the
      generic question "what is x86 missing"; take the mature ARM/KleidiAI backend and, feature
      by feature, ask where the x86 equivalent is and answer all of: same semantics · same
      default · same flag · same runtime observability · same numerical contract · same batch
      range · same persistent lifetime. A "no" anywhere is a parity gap even when x86 works.
      Also lands the BLAS ownership decision, which is architectural and not a 2% question: two
      compute schedulers in one process is the defect. The end state is either "BLAS is always
      forced single-threaded and cannot escape engine ownership" or "BLAS removed from the hot
      paths" — never "usually one thread if we remembered the right variable".
      Open: P1.2, P1.3, P3.4b, P4.3.
      Deliberately NOT started here, but the immediate consequence of closing this: a
      declarative versioned server profile (a per-host-class JSON beside `configs/perf/`) the server
      verifies at start, printing EXPECTED vs ACTUAL and refusing to benchmark on a mismatch —
      `FATAL: production benchmark profile mismatch`, not a warning it measures through anyway.

- [ ] P0-PROFILER — ENGINE RUNTIME PROFILER. [Do NOT start before PARITY-1/2/3 are closed.
      Outranks every further x86 kernel optimization once they are.]

      NOT another pile of ad-hoc timing logs and NOT another `PROFILE=1` that prints 800 lines.
      One coherent profiler for the real engine/server flow: from ONE short real-model run
      (~2 s of generated audio) it produces a complete, readable X-ray of the request, and the
      next optimization queue is built from its measured total-request impact.

      TWO LEVELS AT ONCE, because neither alone is enough. `perf` can say `sd_gemm_panel` is
      X% but not that the work belongs to Decoder -> conv_up -> res1 -> conv1d -> block0, and
      above all not that there were 2 work items for a pool of 6. Internal timers alone cannot
      see LLC misses, context switches, migrations, IPC or page faults. The combination is the
      point.

      POOL OCCUPANCY IS A FIRST-CLASS METRIC, not a footnote next to wall_ms. Every parallel
      region must answer `nt_requested / nt_active / tasks / utilization`. The defect just
      fixed would have been one red line on the first profiling run:
          conv_up/res1  wall=... request_share=31%
            pool_threads=6 work_items=2 active_threads=2 occupancy=33%
            WARNING: parallel work decomposition underfills the pool

      MUST REPORT, without anyone reading the source:
      · HOST/BUILD — CPU model, physical cores, SMT on/off, logical CPUs, NUMA, ISA caps, build
        flags and SIMD backend, model and quantization mode, worker topology, threads per
        worker, actual affinity masks, pinned or not, effective values of the runtime flags.
      · REQUEST/SERVER — concurrency, observed batch size over time and per worker, TTFA,
        STREAM_RTF, prebuffer, frames and audio duration, scheduler/admission behaviour.
      · FULL EXECUTION TREE — per phase/component: parent->child, inclusive and self time,
        % of request wall, call count, time/call, frames or tokens per call, backend AND kernel
        actually selected, and the FALLBACK REASON when the preferred path was not taken.
      · THREADING — per expensive component: requested vs actually participating threads, idle
        workers, work items/panels/tiles, distribution, pool dispatches, barriers, waits/spins,
        nested dispatch, BLAS thread participation, effective parallel efficiency.
      · MEMORY — malloc/calloc/realloc/free counts and bytes, aligned allocations, mmap/munmap,
        hot-path allocations, scratch growth, temporary copies, bytes copied/gathered/scattered/
        packed/quantized where observable, persistent packed-weight footprint, per-request
        temporary footprint, repeated packing or conversion.
      · HARDWARE COUNTERS (Linux, `perf_event_open`, low overhead): cycles, instructions/IPC,
        context switches, CPU migrations, page faults, cache references/misses, LLC misses,
        branch misses, stalled cycles where supported. Uncore/bandwidth counters when the host
        exposes them — absence or missing permissions must never break the profiler.
      · DATAFLOW MAP — per major projection/conv: input -> gather/copy -> conversion/quant ->
        activation pack -> weight representation -> kernel -> output conversion/scatter, with
        duplicated transformations flagged, so avoidable data movement is visible at a glance.

      AUTOMATIC RED FLAGS: component >5% wall with poor occupancy · scalar fallback on a
      SIMD-capable host · expected backend not selected · repeated weight conversion/packing ·
      hot-path heap allocation · excessive memcpy/gather/scatter · excessive barriers or spin ·
      BLAS unexpectedly spawning threads · SMT sibling contention · CPU migrations despite
      pinning · batch ceiling forcing a fallback · significant unattributed time · large
      cache-miss/memory-traffic component · large inclusive time with low useful-arithmetic
      occupancy.

      OUTPUT: one command produces `profile-summary.md` (hierarchical, scannable: the request
      tree, then a sorted HOTSPOTS table, then RED FLAGS / FALLBACKS / UNDERUTILIZATION) and
      `profile.json`.

      MODES: FAST (~one short request, ~2 s audio, enough for architectural diagnosis, the
      development default) and DEEP (optional, more counters and tracing, allowed to perturb
      the runtime, attribution only — never a production benchmark claim). Production
      qualification stays instrumentation-light.

      DESIGN RULE: instrument SEMANTIC engine components and dispatch decisions, not function
      addresses. The wanted line is "Talker layer 7 down projection, INT8, VNNI, B=2, activation
      quant 4.1 us, kernel 38 us, scatter 2 us, pool 6/6", not "qwen_matmat_int8() 12.7%".

      It becomes a permanent engineering and regression tool for ARM, x86 and future backends:
      it does not optimise the TTS directly, it makes every later optimisation much faster.

- [ ] P1-CONFIG SERVER CONFIG CONTROL PLANE — detail: `.work/config-control-plane.md`.
      [After P0-PROFILER. Not "tidy the JSONs": remove the normal path that can ignore them.]
      The proof it is an abstraction problem and not a documentation one: the correct GCP
      topology was ALREADY in `configs/perf/gcp-c4-standard-24-vnni-ttfa.json` (2 workers x 6
      threads, affinity 0-5 / 6-11, with a warning about the 24 logical CPUs) and three
      campaigns still ran at 2x8 from memory. The JSON was right and the experiment was wrong.
      Measured duplication today: 22 of 28 env keys appear in more than one profile,
      `QWEN_POOL_SPIN` in seven (4096 on six x86, 65536 on Arm), so P5.0 — "make 65536 the x86
      default" — means editing six files and hoping none is missed. A common engine default has
      no owner. One 200-line object also mixes runtime config, experimental candidates,
      benchmark history and open TODOs in one namespace.
      Layering, one owner per value: common engine -> Linux CPU server -> backend family
      (ARM/KAI | x86 VNNI | x86 AMX) -> host topology -> experiment override (only the variable
      under test). Done when: no duplicated semantic default across host profiles · one
      inheritance/resolution mechanism · ONE canonical launcher every suite goes through ·
      qualification cannot bypass resolution · effective config emitted per run · real host
      topology verified against the resolved profile, mismatch FATAL · one owner and default per
      flag · overrides contain only what was intentionally changed · resolved-config + binary +
      model hashes stored with results · changing one common x86 default propagates everywhere.

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
- [ ] P1.2 [BLOCKED: no ARM i8mm host in reach] Verify Arm KleidiAI GEMV/GEMM shape coverage on
      a box. M1 does NOT qualify and must not be used as a stand-in: `kleidi.enabled` reads
      "not compiled (needs an i8mm target)" there, so an M1 run measures the NEON fallback and
      proves nothing about the KleidiAI kernels. Needs: an aarch64 host with i8mm (Graviton 3/4,
      Neoverse V1/V2/N2, Oracle A1 is N1 = no i8mm), `make blas` picking up KleidiAI, then
      `--dispatch-map` + `--self-test` + a golden run
- [ ] P1.3 Verify VNNI GEMV/GEMM shape coverage (q4 GEMM 0.80x is the known weak one).
      Partly answered while closing P1.5: prefill chunks itself to B<=16 (`prefill_proj_matmat`)
      so it never crosses `max_b`, but the batched DECODE passes the live slot count straight
      through, and every batched int8 gate stops at B=16 while `--batch-size` is not clamped to
      anything. New row `matmat.int8.batch_ceiling` + a one-line server warning make that
      visible; chunking it silently is NOT the fix, since a remainder column would move onto the
      B=1 dequant twin and change its arithmetic. What remains is the per-shape q4 GEMM measurement
- [x] P1.4 Verify AMX B>=4 region/head paths — detail: `.work/p1-4-amx-runtime.md`
      Runtime verified on GCP AMX host: production-like C4 decode remains VNNI at
      observed B1/B2; AMX BF16 prefill executes; AMX INT8 observed only at C8 B4/B5.
      `suspicious=1` was a census reporter false positive (B>=2 vs real AMX INT8 B>=4).
      P2/P3 implementation follow-ups are now unblocked.
- [x] P1.5 AVX2 / AVX-512F non-VNNI fallbacks verified on the GCP x86 box, and one real defect
      found: `SIMD=avx512` (F/BW/VL, no DQ, no VNNI) did NOT COMPILE — three |x| reductions used
      `_mm512_andnot_ps`, which is AVX512DQ, inside `__AVX512F__` code. Replaced with the integer
      abs-mask (same bits). All three profiles now build, `--self-test` PASSES on each, and a
      real generation runs: avx512 and portable 69120 samples, VNNI 67200, each bit-repeatable
      across two runs (different integer kernels -> different trajectory, both valid audio).
      Resolved rows match the static contract: `matvec.int8.native`/`matvec.q4.native` = no on
      both non-VNNI builds (B=1 dequantises to the f32 fused twin, P3.6), `matmat.{int8,q4}.family`
      = AVX2 maddubs even on the AVX-512F build, `matmat.bf16.family` = the fixed-B twin on all
      three, `decoder.int8` OFF (no kernel on those ISAs, P3.4a). Also fixed the `--caps` "lever"
      line, which asked the CPU instead of the build and advertised VNNI on a binary without it
- [x] P1.6 Apple/GCD/Accelerate runtime verified on M1 at the capability level (no macOS perf
      campaign, deliberately). Resolved map is consistent with the code: `blas.owned_effective`
      OFF -> the decoder SGEMM partition is skipped and Accelerate keeps its own team, which is
      the right answer where BLAS thread count cannot be set; `decoder.pool` private;
      `pool.nested_dispatch` ON, `pool.concurrent_submit` ON, `pool.submit_priority` OFF with
      QWEN_PREFILL_LOW_MS reported as ignored (GCD has no submit priority); prefill resolves to
      the f32 convert + SGEMM twin, no bf16 matmat unit. Known Apple asymmetry, left as is: a
      private decoder team and Accelerate's own threads can oversubscribe. That is a performance
      question, not a capability lie

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
      the dispatch map. CORRECTION (2026-09-05): 47ede94 also let the new predicate decide
      whether the legacy threaded `--serve N` path serialises synthesis, which was the same
      mistake one level down -- "may two threads submit to the pool" is not "may two syntheses
      overlap in one process", and the engine's process-wide state has never been audited for
      that. The long-standing serialised default is restored; prefork was never affected.
      detail: `.work/p3-runtime-knob-parity.md`
- [ ] P3.3b Cloud A/B only for the differences P3.3a leaves unresolved. The one that matters is
      now named: whether two syntheses may overlap inside ONE process (the threaded `--serve N`
      path). Answering it is an engine-state audit plus a concurrent A/B, not a pool question,
      and it is not on the production path (prefork isolates by process)
- [x] P3.4a Decoder int8-conv capability split from policy (47ede94): `qwen_sd_int8_available`
      (kernels compiled) + `qwen_sd_int8_usable` (shapes the kernels cover, moved off the
      decoder call site) vs a named per-backend default with its reason. AVX2/AVX-512F have no
      int8 conv kernel at all, so there is nothing to wire there — not made symmetric on purpose
- [ ] P3.4b [BLOCKED: needs an ARM box and a measurement campaign] Measure the ARM dotprod
      first-frame cost and decide whether it may default ON
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
- [x] P3.11 The NEON body of `qwen_int8_quant_rows` rounds half-to-EVEN (`vcvtnq_s32_f32`) while
      the scalar tail rounded half-AWAY, so one value quantised differently depending on whether
      its index landed in the vector body or the remainder. Fixed as an INTERNAL consistency
      defect, not by normalising the platforms: `quant_round_i32()` gives each platform ONE rule
      everywhere (ARM half-to-even, matching its body and its qualified audio; x86 half-away,
      matching its body and 9933948). The x86/ARM difference is left standing on purpose -- it is
      a policy question, and neither side's audio was qualified against the other's rounding.
      `QWEN_NO_SIMD_QUANT` now also gates the NEON path, which it never did, so the `--self-test`
      parity gate is real on ARM instead of comparing NEON with itself. Proven discriminating:
      restoring the old tail makes 3 of the 8 cases FAIL. M1 golden 4/4 unchanged
      (1.00000/1.00000/1.00000/0.99995), so no qualified ARM audio moved
- [x] P3.5 AMX/x86-QKV/decoder knobs documented in docs/feature-flags.md; region rows
      updated for AMX; `QWEN_CP_FRAME_REGION` added. check-flag-registry 180/180

- [ ] P2.7 [BLOCKED: same ARM i8mm host requirement as P1.2] ARM persistent region: interface COMPLETE, wiring blocked on hardware (8aa95db adds
      the fused Q/K/V phases beside the plain ones, so every shape the region needs is exposed).
      Remaining: give the region body a row-major gather shape; needs an ARM i8mm box.
      Original analysis: blocker NARROWED, no longer numerical (e78e861).
      `qwen_kleidi_i8_region_usable/_prep/_run` expose the pack and n-block phases the
      KleidiAI int8 path already had, so a region can reuse the exact dispatched kernel and
      output. What remains: the CP/Talker region body gathers k-major `[cols][B]` for the
      shared per-column quantiser while KleidiAI wants row-major activations it quantises
      itself, so the region needs a second gather shape. Needs an Arm i8mm box to validate.

- [x] P3.12 `make check-matmat-parity` did not link: `qwen_tts_thread.c` calls
      `qwen_region_begin_/end_` (cost-map instrumentation) but `PARITY_SRC` never listed
      `qwen_tts_costmap.c`, so the batched-twin arithmetic gate had been failing at the linker
      instead of running. Fixed; it now PASSES on M1 native (int8 twin within 2.2e-2 rel of the
      integer reference, q4 exact) and on the Rosetta x86-64-v3 build (both exact).
      `prefill-bench` had the identical gap in its own hand-written source list and was also
      dead at the linker (confirmed by rebuilding it from the unpatched Makefile); both fixed
      and both re-verified on the LINUX/gcc box, where the int8 twin is exact (0.000e+00,
      it is the real VNNI integer kernel there) against 2.2e-2 on the M1 f32 twin.
      `make test-golden` now guards MODEL_SMALL the way it already guarded MODEL_LARGE, and
      refuses to report PASS when nothing ran: a rented box carries one checkpoint, and a
      missing model was being reported as a numerical regression. The x86 box can now run the
      golden net (numpy + librosa installed there)

## P4 — architecture cleanup (no implementation before P0-P3 are understood)

- [ ] P4.1 Draft module boundaries: common runtime / cpu dispatch+caps / cpu arm / cpu x86 / gpu
- [ ] P4.2 Promote `g_mm_gate[]` + compiled/supported predicates to the one capability table,
      extended to non-matmat capabilities (regions, heads, conv int8, prefill, budget)
- [ ] P4.3 Make fallback selection observable and testable from that table. Partly delivered
      ahead of P4.2 where it cost nothing: `--dispatch-map` now carries the persistent regions
      (`region.int8_runner` names AMX tiles / VNNI row blocks / none, `region.team`,
      and the four region knobs) and `matmat.int8.batch_ceiling`. Before this the only place
      that said whether a region runs was a one-shot stderr line the engine prints at the first
      batched step, i.e. after traffic. dispatch_gate coverage 42 -> 46 flags resolved

## Later

- [ ] Rename legacy CLI `--batch/--batch-words/--batch-dry` (deprecation alias)
- [ ] Custom decoder fp32 GEMM evaluation (non-bitwise acceptance first)
- [ ] INT8 prefill quality qualification (`QWEN_PREFILL_INT8MM`, separate numerical path)
- [ ] Decide the two unreviewed GCP reference notes (c3d 8c VNNI, c4-standard-24 VNNI
      corrected) and the stray object file left in the primary checkout. They exist only as
      untracked files there, so they are named, not linked: a plan reference must resolve in
      any clone (ENGINEERING.md §2b).
- [ ] P5.10 [low, DEFERRED — not a current task] BLAS removal / replacement audit. Inventory the
      remaining hot-path OpenBLAS/SGEMM usage; measure the COMPLETE per-operation cost, not the
      GEMM arithmetic alone (layout and conversion work, thread-runtime overhead, the partition
      itself); compare against a fixed-shape custom kernel, oneDNN, and the native paths we
      already have; remove only where the measured win justifies the numerical and maintenance
      risk. Sequenced last on purpose: it is optimization work and waits until structural
      parity is substantially closed.
- [ ] P3.6 [moved here 2026-09-05: this is kernel work, not a parity gap] AVX2 / AVX-512F have
      int8+q4 GEMM but NO integer GEMV, so every B=1 dequantises to the f32 fused twin. Visible
      since 47ede94 (`matvec.int8.native` / `matvec.q4.native`) and confirmed at runtime on the
      box in P1.5. The fix is a kernel: there is no wasted conversion to remove, and reusing the
      GEMM at B=1 would change the arithmetic.
- [ ] P0.6 [PARKED 2026-09-05 by explicit decision: not a priority, do NOT work on it] The golden references DO NOT hold
      on x86, and nobody knew because that box had neither numpy nor librosa, so `make test-golden`
      printed "SKIP: librosa not installed" and exited 0. Installed both and measured the 1.7B
      reference on the GCP host: `SIMD=amx` mel-corr 0.60122 with duration 5.20s vs 4.56s (+14%),
      `avx512vnni` 0.74555 (dur 1.8%), `portable` 0.79514 (dur 1.8%) — all far under the 0.98 gate,
      and the three profiles differ from each other as well. NOT a regression: the same run against
      commit 85f2678 gives byte-identical WAVs (md5 48176f02 amx, 85e95b89 vnni on both trees), so
      this predates everything on this branch. What it kills is the belief that mel-corr ≥0.98 is
      our cross-ISA check: the references are M1-generated and only hold on ARM. Decide which:
      per-ISA reference sets, a looser cross-ISA threshold justified by listening, or state plainly
      that golden is an ARM-only regression net and give x86 its own. The +14% duration on AMX
      deserves an ear check before anything else — it is a different-length utterance, not noise.
- [ ] P5.0 [low, BLOCKED-BY P1-CONFIG] Set `QWEN_POOL_SPIN=65536` as the x86 server default.
      Deliberately not done by hand: the value is duplicated in seven profiles today, so this is
      the worked example of why a common default needs one owner. Do it once at the x86 level
      after the control plane exists, not as six edits.
- [ ] P5.1 [low] Compare AutoRound/LLM Compressor W4A16 and Intel ARK packed kernels with runtime INT8: https://vllm.ai/blog/2025-12-09-intel-autoround-llmc https://github.com/intel/auto-round/tree/main/auto_round_extension/ark
- [ ] P5.2 [low] Run isolated Xeon AMX/VNNI GEMV/GEMM oracle probes with oneDNN benchdnn and OpenVINO CPU: https://github.com/uxlfoundation/oneDNN/tree/main/tests/benchdnn https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.rst
- [ ] P5.3 [low] Audit vLLM CPU, oneDNN and IPEX prepacking/fusion/cache behavior against CP, Talker and INT8 conv: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/A-Practical-Guide-to-CPU-Optimized-LLM-Deployment-on-Intel-Xeon/post/1737233
- [ ] P5.4 [low] Run GCP oracle probes only from an isolated copied folder and only at 0% CPU with no competing workload.
      (P5.5-P5.9, the earlier private x86-dataflow probes, are folded into the X86-* section
      below: same questions, now against a tracked document instead of a `.work/` note.)

### X86 dataflow follow-ups — detail: `docs/x86-int8-dataflow-2026-09-05.md`

- [x] X86-1 Direct source-row INT8 quantization for CP/Talker regions — landed on this branch as
      `40c156f` (`d847d9c` in the worktree it was written in). Subsumes the old P5.6 staging half.
- [ ] X86-2 Low-B x86 AMX/VNNI crossover — MEASURED on the AMX box, `tests/region_lowb_bench.c`
      (paired arms interleaved in one process via the new `qwen_mm_force()` hook; unpaired
      cross-process arms swung 14% on an unchanged config, larger than the effect).
      Findings: (a) the discriminator is ROWS PER THREAD, not B and not the rows/cols ratio — the
      same projection flips sign with the thread count (CP Down -17.3% at 4 threads, +1.3% at 8),
      and a rows>=cols rule I first shipped was wrong (TK Down is AMX -5.2% at 8 threads despite
      being 3x deeper than tall). `rows/thread >= 256`, fused QKV judged on q+2kv, agrees with 23
      of 24 measured cells. (b) The BATCHED SERVER measures B 0.9-3.8 per prefork worker across
      C=1..8, so the B>=4 gate left AMX essentially unused in production; INT8 AMX now starts at
      B=3, where the rule keeps only the winners (Gate/Up -12..-14%, TK Down -4.4%, TK WO -3.0%).
      A first server A/B of the shape rule alone was a null result for exactly that reason.
      Subsumes old P5.5 and P5.8 (oneDNN/benchdnn on real CP/Talker shapes, small-B VNNI vs oracle).
- [x] X86-3 Persistent packed-RHS consumption audit — ANSWERED and MEASURED; stays opt-in.
      `QWEN_AMX_PREPACK` defaults to 0, so `qwen_amx_pack_weights()` returns NULL and every AMX
      tile load runs STRIDED off the row-major weights (`_tile_loadd(2, w0, pW ? 64 : cols)`).
      The persistent representation already existed and is built in the parent before fork
      (`qwen_amx_prepack_model`, inherited by prefork workers through copy-on-write).
      Microbench (paired, 6 threads, host-native): the packed RHS helps AMX exactly where it was
      losing — CP WO goes from +24.3% to -11.0% at B=3, CP Down from +8.4% to +0.6% — confirming
      the strided tile load was the reason short/deep projections lost.
      Server A/B (2x6, SMT off, C=2/4/6): STREAM_RTF p95 1.036 -> 1.015 at C=4 and 1.440 -> 1.341
      at C=6, but TTFA p95 838 -> 929 ms at C=6 and PSS 11.0 -> 15.3 GB. 4.2 GB of packed copies
      for <=7% sustained RTF and a latency regression is not a default: the copies evict the
      working set, which is the same L3 pressure the microbench showed. KEPT OPT-IN.
      Landed anyway: the INT8 half of the prepack now packs only what the gate can select, judged
      with the SERVING worker's thread count rather than the packing process's (the parent
      prepacks before it forks and runs a different -j). VNNI has no packed RHS in production
      either (`QWEN_VNNI_PREPACK` rejected on Zen5), only a cached row-sums array, by design.
      Open follow-up if this is ever enabled: the bf16 half is still packed unconditionally and
      is the bulk of the 4.2 GB. Was P5.9.
- [x] X86-4b [FIXED, promoted by measurement 2026-09-05] The speech decoder CONV STACK is the
      single largest block of a request, not the CP/Talker integer path. Level-2 cost map on the
      real server (GCP host-native 2x6, SMT off, 1.7B --int8, C=2, `serve` role, blocks are
      sequential on that thread and sum to request.total 5595 ms/req):
        decoder.conv_stack  1923 ms/req  34%   (95.3% of all decoder time)
        talker.decode       2379 ms/req  43%
        cp.decode           1050 ms/req  19%   (gate_up 337, qkv 209, down 166, out_proj 116,
                                                lm_head 50, attention 40; only 3.1% unattributed)
        prefill both        219 ms/req    4%
      Pool synchronisation costs 7.5% of the request (518k dispatches, 3644 ms waiting for worker
      completion = 7.6% of dispatch, 572 ms on the submit lock = 1.2%).
      This is the "unless profiling shows the cost distribution has materially changed" case for
      the decoder: 9933948 measured only 0.8% end to end because it optimised the ACTIVATION
      QUANTISER, not the convolutions themselves.
      CAUSE FOUND, and it was pool fill, not arithmetic. `conv_up` is 90.3% of decoder time and
      `res1` is 57% of that. The INT8 decoder conv parallelises over OUTPUT COLUMNS only, one
      fixed 128-column panel per work item, never over out_ch. At the real shapes the most
      expensive layer per column (blk0, M=768 K=5376, 1057 MMAC = a quarter of all conv1 work)
      has 256 columns = TWO panels, so four of six workers idled on it, and at one frame per
      chunk it was a single panel running single-threaded. `sd_conv_nc()` now sizes the panel
      from the layer length and the pool (floor 24, rounded to a multiple of 4 for the tile
      path); `QWEN_SD_CONV_NC=128` restores the old fixed panel and is the A/B arm.
      Bit-preserving, PROVEN not asserted: at one thread, widths 128/124/44/24 all give WAV md5
      fe66a43d. Two earlier md5 differences were multi-thread float ordering, and the first
      explanation offered for them (tile-tail grouping) was wrong.
      Measured, host-native 2x6 SMT off, same binary both arms:
        decoder conv_up  frames=8 126.2 -> 116.2 ms, frames=4 82.3 -> 66.2 ms (-19.5%)
        decoder res1     frames=8  72.4 ->  64.7 ms, frames=4 50.7 -> 34.4 ms (-32.2%)
        server C=2  STREAM_RTF p50 0.660 -> 0.634  p95 0.692 -> 0.661
        server C=4  STREAM_RTF p50 0.937 -> 0.896  TTFA p95 577 -> 674 (worse)
        server C=6  STREAM_RTF p50 1.140 -> 1.021  p95 1.311 -> 1.178
      Open: the TTFA regression at C=4 is unexplained and worth one look; out_ch is still never
      a parallel axis, which is the remaining fix if a layer is short AND narrow.
- [ ] X86-4 Activation preparation/fusion follow-up — remove remaining generic gather, q8-pack and
      scatter passes. Rest of old P5.6; the decoder half of old P5.7 is partly done by 9933948
      (x86 SIMD `qwen_int8_quant_rows`), the im2col fusion is not.
- [ ] X86-5 AMX activation-pack reuse — REAL but currently worthless on this serving profile;
      do not spend on it until per-worker B rises. The redundancy is confirmed by reading:
      `qwen_region_i8_run` calls `amx_pack_act_int8` on EVERY thread, each packing the same
      B x cols activation into its own scratch, so the pack is duplicated nt times. (The fused
      QKV already shares one pack across Q, K and V, which is also why its gate is judged on
      q+2kv.) Sizing from the B-sweep: the AMX cost is nearly flat in B and the pack slope is
      about 1.4 us per unit of B on TK Down (cols 6144), i.e. ~10% of that projection at B=4, so
      packing once instead of nt times is worth roughly 5-10% of AMX projection time.
      Why it does not pay HERE: the batched server measures B 1.2-2.9 per prefork worker at
      C=2..6, and INT8 AMX needs B>=3 plus rows/thread >= 256, so AMX barely executes at this
      concurrency. Revisit when a host or profile actually sustains B>=3 per worker; the fix
      needs a barrier inside the runner or the pack hoisted into the region body, which has one.
      NOTE (2026-09-05, box): the B=32 two-accumulator prototype in `tests/prefill_bench.c` is
      not usable as it stands. On the AMX build it dies with SIGILL before its first line while
      the engine's AMX path is live in the same process; `QWEN_NO_AMX=1` makes the SAME binary
      run to completion, and a dedicated probe on that host showed the engine's own AMX matmat
      is healthy with the permission requested either before or after the pool exists, so the
      fault is the prototype's tile handling, not the runtime. It is also numerically wrong
      (worst relative 2.18 vs the shipped path, its own check says so). Now behind
      `QWEN_PB_AMX_PROTO=1` so `make prefill-bench` measures the per-call fixed cost instead of
      crashing on exactly the machines it exists for
- [ ] X86-6 Real 0.6B server qualification — repeat the shape-oracle conclusions with the actual
      checkpoint and C1/C2/C4 screens.
- [ ] X86-7 0.6B vs 1.7B serving profile — compare Talker, CP and decoder time shifts after the
      Talker width change.
- [ ] X86-8 SERVER OPERATING PROFILES — latency-first vs streaming-safe. [LATER: do not start
      before X86-2..X86-5 close. NOT a note: this is a serving OBJECTIVE and outranks further
      TTFA work once the structural x86 work is done.]

      WHY, from the real server screen (GCP AMX box, 1.7B --int8, batched + prefork 2x8,
      `serve_parallel_wave`, C=1..8, 2026-09-05): STREAM_RTF p50 is already 0.958 at C=4 with
      p95 1.008, and C=6/C=8 are in sustained underrun (18/18 and 24/24 requests starved,
      prebuffer needed 1.2-2.7 s). Ultra-low TTFA on its own is therefore NOT an acceptable
      production objective — the stream does not survive at the concurrency it advertises.

      PRINCIPLE to record explicitly: for sustained human streaming, TTFA 200 ms with
      STREAM_RTF 1.15 is NOT preferable to TTFA 500 ms with STREAM_RTF 0.85.

      Deliberately search for TWO operating points from ONE engine and one codebase — scheduler,
      prebuffer, admission and batching policy only, never two implementations:

        1. latency-first — minimise TTFA, for highly interactive workloads; must still declare
           a SAFE supported concurrency rather than a best-case one.
        2. streaming-safe / balanced — allow a higher TTFA where needed (~400-600 ms), require
           STREAM_RTF p95 < 1 with real headroom, target roughly 0.8-0.9 where achievable, and
           optimise for uninterrupted playback and stable concurrency.

      Decision priority for the balanced profile, in order:
        1. zero underruns, errors and rejects;
        2. STREAM_RTF p95 < 1 with margin;
        3. TTFA inside an acceptable interactive envelope;
        4. throughput and concurrency.
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
