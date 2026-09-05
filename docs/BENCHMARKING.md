# Benchmarking runbook — CPU server and box qualification

The single source of truth for HOW a CPU box or server build is validated and measured.
Normative through `ENGINEERING.md`. Derived from the tree as of 2026-09-05 (Makefile,
`tools/`, `tests/`, `configs/perf/`), not from memory. When a canonical tool, profile or
procedure changes, the same commit updates this file (§10). No results live here.

## 1. Canonical tools — CURRENT ONLY

| need | current tool | when | status |
|---|---|---|---|
| CPU / ISA identity, topology (sockets, NUMA, SMT, physical cores), caches, measured bandwidth | `make cpu-check` → `tools/cpu_check.sh` (writes `hardware.json`/`.txt` via `tools/box_info.sh`, `topology.json`, `roofs.txt`, `caps.txt`, `dispatch.json`, `dispatch_gate.txt`, `env.txt`, `profile_env.txt`, `manifest.json`, `gate.txt`) | first thing on every box, and after every rebuild | **authoritative preflight** |
| hardware inventory alone | `make bench-fingerprint` → `tools/box_info.sh [--out hw.json]` | quick identity without a model | authoritative for identity, subset of cpu-check |
| memory bandwidth per execution domain | `make membw` (`tests/membw.c` → `$(MEMBW_BIN)`), `make roofs ROOF_MASKS=0-7,8-15` → `tools/roofs.py`, `make topology-report` | before choosing `W x K`; a roof belongs to one cpu mask | authoritative; never divide a host roof |
| compiled / runtime kernel capability | `./qwen_tts --caps`, `make test-caps`, `make check-isa` (compile-check of the ISA paths this host cannot run) | after build; before renting | authoritative |
| build | `make blas` (`SIMD=auto`), or explicit `make blas SIMD=avx512bf16` / `amx` / `avx512vnni` / `portable`; `make info` | every box; explicit `SIMD=` for a qualification build | authoritative; the chosen target is printed by `[simd]` and `--caps` |
| kernel correctness | `make test-selftest` (`--self-test`, native and forced-fallback), `make test-golden` (mel-corr + duration, quiet box), `make check-matmat-parity` | after build, before any number | authoritative gates |
| dispatch / capability inspection | `./qwen_tts --dispatch-map` (`QWEN_DISPATCH_JSON=path` for JSON), `tools/dispatch_gate.py dispatch.json --expect tools/dispatch_expect.json`, `make profile-cpu-check`, `make check-flag-registry` | in the profile's env, from cpu-check; again whenever env, binary or source change | authoritative for resolved gates; not a substitute for the in-process census (§5) |
| performance profile (server argv + env) | `configs/perf/*.json`, `tools/perf_profile.py validate | show | command | server-env | forbidden-env | check-flags`, `python3 tests/test_perf_profile.py` | every server launch in a measurement | authoritative; the profile's status (qualified / provisional / unqualified) is in `configs/perf/README.md` |
| server launch for measurement | `tools/perf_profile.py command <profile> --model DIR --port N` (prints the exact `./qwen_tts … --prefork W --prefork-threads K --cpu-mask … --batch-size B` line); the harnesses below launch it themselves from `--profile` | never hand-typed from memory | authoritative |
| WAVE (finite screening) | `tests/serve_parallel_wave.py --profile P --topo WxK --conc 1,4 --waves N …`; wrappers `make bench-topo`, `make bench-suite` (rungs realistic / fast / short-diverse / long-diverse, identity gate `tests/serve_identity_gate.py`) | topology and knob screening, A/B arms | canonical WAVE; screening only |
| SOAK (sustained, fixed concurrency) | `make bench-soak SOAK_PROFILE=P SOAK_CONCURRENCY=C SOAK_MINUTES=M …` → `tests/serve_soak.py` (+ `tests/soak_client.py`), analysed by `tests/soak_drift.py` → `soak_summary.json`; `--strict-kpi` for the production gate | the production qualification | **canonical SOAK; the only production gate** |
| POISSON / overload | `tests/load_test.py --arrival poisson --rate R` (also `uniform`, `all-at-once`) | only when open-arrival behaviour is the question | available, on request; not part of `bench-suite` |
| lightweight profiling | `make profile-cpu` (shape census, `tools/census_report.py`), `make cost-map` (C1 only; `tools/costmap_parity.sh`), external `perf stat` / `perf record -F 499 -p <pids>` by hand | DIAGNOSTIC runs only | never a qualification number; no tracked wrapper for `perf` |
| result summarisation | `tests/soak_drift.py` (`soak_summary.json`), `tools/envelope_report.py` (`make envelope WAVE_JSON=…`), `tools/topology_report.py`, `tools/census_report.py`, `tools/costmap_report.py` | after the run, from its artifacts | authoritative readers of the canonical artifacts |
| source identity to a box | `tools/box_sync.sh user@host` (ships `.source_fingerprint`), `tools/source_fingerprint.sh` | before building on a rented box | authoritative |

Models: `MODEL_SMALL=qwen3-tts-0.6b`, `MODEL_LARGE=qwen3-tts-1.7b` (`BENCH_MODEL`), profiling
default `qwen3-tts-1.7b-base`. Text bank: `tests/load_texts_en.txt`.

## 2. Legacy / forbidden-for-qualification paths

Exist in the tree, must not be used to prove streaming-server behaviour. Do not delete here.

- **CLI `--batch`, `--batch-words N`, `--batch-dry`**: long-text paragraph splitting for one
  generation (audiobook-like). Not server batching, not concurrency. Never evidence for
  server B>=2 or C>=2. Rename pending (PLAN, Later).
- **CLI `--batch-test`, `--batch-multi-test B`, `--batch-bench`**: engine batched-step
  self-test and microbench (`qwen_batch_self_test`, `qwen_batch_bench`). Kernel-level parity
  only; not the server scheduler, not the streaming path.
- **CLI `--matmat-bench`, `--matmat-tune`, `make matmat-bench`, `make kernel-tune`,
  `make tune-archive`**: kernel gate thresholds; no model, no server.
- **`make bench`, `make bench-full` (`bench.sh`), `make bench-matrix[-full]`
  (`tests/bench_matrix.sh`), `tests/x86_bench.sh`, `tests/avx512_parity_bench.sh`**: CLI
  wall/RTF matrices (single, `--batch`, `--stream`). `docs/hardware-testing.md` still calls
  `bench-matrix` "the per-box report"; for a server it is not. Unresolved: PLAN P0.5.
- **`make bench-server` / `make server-batch-microbench[-full]` (`tests/serve_batch_bench.sh`)**:
  request-batching throughput microbench on the plain server; not the WAVE/SOAK harness.
- **`make server-hw-check` / `make box-report`**: `bench_matrix.sh --silicon-only`; superseded
  by `make cpu-check`.
- **Deprecated shims** (print a warning and exit): `tests/kleidi_parallel_capacity.py`,
  `kleidi_server_matrix.py`, `kleidi_thread_curve.py`, `kleidi_topology.py`,
  `kleidi_topology_curve.py`, `kai_procstats.py`, `kai_mem.py`.
- **Ad-hoc probes, not referenced by any target or doc**: `tests/serve_shard_vs_batch.sh`,
  `serve_twin_vs_matvec.sh`, `serve_slot_drift.sh`, `serve_concurrency_matrix.py`,
  `serve_topology_bench.py`, `serve_topology_probe.py`, `serve_thread_curve.py`,
  `serve_memory_probe.py`, `cadence_dump.py`, `playback_sim.py`. Exploration only; a number
  from them is DIAGNOSTIC.
- **Correctness, not performance**: `make test-serve-*`, `tests/test_parallel.sh`,
  `tests/serve_continuous_stress.sh`, `tests/batch_determinism.py`.

## 3. Box qualification order

A. CPU identity and ISA (`make cpu-check` / `make bench-fingerprint`) ·
B. sockets / NUMA / physical cores / SMT (`hardware.json`, `topology.json`) ·
C. cache topology (same; on a cloud slice `lscpu` L3 is the socket's, trust the measured roof) ·
D. bandwidth and scaling per mask (`make roofs`, `make topology-report`) ·
E. build target and compiled capabilities (`make blas SIMD=…`, `--caps`, `check-isa`) ·
F. runtime dispatch in the profile's env (`cpu-check CPU_PROFILE=…`, `dispatch_gate`) ·
G. single-request sanity (`make test-selftest`, `make test-golden`, one `curl` against the
   profile's server) ·
H. finite WAVE screening (`bench-topo` / `bench-suite`, C=1 and the target C) ·
I. sustained canonical SOAK (`bench-soak --strict-kpi`, 5 minutes minimum for a gate) ·
J. POISSON only when specifically required.
Never start at I on an unknown box.

## 4. Mandatory run identity

Established BEFORE measurement and kept with the artifacts: date and time (UTC and local);
machine label (hostname only if public-safe); CPU model; physical cores and SMT; NUMA; CPU
masks; git commit; dirty state; binary SHA256; build id (`--caps` build line /
`.source_fingerprint`); compiler; compile SIMD/ISA flags; model; quantization; server mode;
prefork topology (e.g. 2x8); threads per worker; `--batch-size`; prefill configuration;
scheduler/pool settings; every non-default `QWEN_*` performance flag; exact command or
profile. `make cpu-check` produces most of it (`manifest.json`, `hardware.json`, `env.txt`,
`profile_env.txt`); the soak writes its own `manifest.json` and `server_command.txt`.
Canonical SOAKs come from a clean committed tree (ENGINEERING.md §13); a dirty binary is
NON-QUALIFYING and the report says so.

## 5. Requested path != actual path

Before measurement, obtain the dispatch from the serving process after its environment is
applied: the `[FLAGS]` line the server prints (checked by `perf_profile.py check-flags` and
by both harnesses), `--dispatch-map` run under `profile_env.txt` (cpu-check does this), and
for DIAGNOSTIC runs the shape census (`make profile-cpu`, `tools/census_report.py`) which
counts the leaf that actually executed. Record requested AND resolved for: Talker prefill,
Talker GEMV, Talker GEMM/matmat, CP GEMV, CP GEMM/matmat, CP heads, codec head, speech
decoder GEMM, speech decoder convolution, packing/repacking. Record fallbacks by name
(generic twin, f32/SGEMM, scalar). Never infer execution from a profile name, an env
variable alone, a binary name, the compile target, or a dispatch dump generated before the
env. Explicit request resolving to an incompatible or fallback path: ABORT the
qualification. `auto` resolving to a supported path: report it, do not fail.

## 6. Machine-readable run manifest

Target: every benchmark artifact directory contains `run_manifest.json` with §4 plus
requested/resolved dispatch per operation, generated from the actual run configuration.
Today (2026-09-05) this file does not exist yet: the soak writes `manifest.json`
(configuration, `[FLAGS]` identity), cpu-check writes `manifest.json` + `dispatch.json`,
the wave writes `topology_*.json` / `parallel_*.json`. PLAN P0.4 creates `run_manifest.json`;
until then a report cites those files and never reconstructs configuration afterwards.

## 7. Benchmark terminology

WAVE: finite screening run. SOAK: sustained fixed-concurrency production qualification.
POISSON: open-arrival queue/overload characterisation. DIAGNOSTIC: profiling or
instrumented run; its numbers never substitute qualification numbers. The CLI paragraph
splitter is never called "batching" in a server report.

## 8. Canonical server metrics

Always: requests started, completed, failed, rejected, timeout, outstanding/killed;
STREAM_RTF p50/p95; TOTAL_RTF p50/p95 when relevant; TTFA p50/p95; throughput (req/s);
wall duration; artifact path; survivors after teardown. Never a bare "RTF".
Gate for C=N realtime: errors=rejects=0, no starvation or drift (`soak_drift`),
TTFA p50 < 500 ms and p95 < 700 ms, STREAM_RTF p50 and p95 < 1 (prefer p95 <= 0.97).

## 9. Run lifecycle

ENGINEERING.md §10 applies: explicit server PID, explicit client PIDs, never a bare `wait`,
bounded timeouts, CPU load and request counters verified shortly after launch, profiler
and sampler stopped explicitly, server stopped explicitly, survivors=0, artifact path and
phase printed at every step. The harnesses own the server; do not start a second one by hand.

## 10. Updating the benchmark system

A change to any canonical benchmark script, profile, server invocation, dispatch gate,
capability checker, topology tool, bandwidth tool or result parser updates this runbook in
the same task and commit, or states that the canonical procedure did not change. No tool
becomes canonical because an agent created it: this file names the current canonical tool
and the one it supersedes. Competing tools without a proven winner are a PLAN task, not a
choice made here. Before a tooling commit: `python3 tools/check_repo_integrity.py` (no
tracked script may depend on an untracked file) and `python3 tools/check_plan.py`.
