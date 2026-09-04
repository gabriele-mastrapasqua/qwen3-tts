# CPU profiling gate — `make cpu-check`, `make profile-cpu-check`, `make cost-map`

> A fast kernel does not make an efficient engine. The two largest x86 wins of 2026-09
> were both *around* the kernels — a dispatch predicate that silently chose the f32/SGEMM
> prefill on an AVX-512-BF16 host (~400 ms of TTFA), and a transpose that undid itself —
> and neither was visible in any existing report. This gate exists so that the tooling,
> not the engineer's memory, proves what the engine is forcing the CPU to do **before**
> anyone optimises it.

Three levels share one artifact directory and one gate language (`[PASS] / [FAIL] /
[WARN] / [SKIP]`, exit non-zero on any FAIL):

| target | budget | what it proves | status |
|---|---|---|---|
| `make cpu-check` | 10–20 s, no model | provenance, hardware, **resolved** dispatch map vs the expectation for this ISA class, self-test native + fallback, process-clean | shipped |
| `make profile-cpu-check` | 1 s | the last `cpu-check` still describes THIS binary, source, host, env, dispatch map and model | shipped |
| `make profile-cpu` | ~10–15 min, model | clean + instrumented C1/C4/Poisson, shape census, region cost, `perf stat/record`, measurement overhead, `summary.md` | planned (V0 spec in the working plan) |
| `make qualify-cpu` | 1–2 h | topology sweep + `bench-suite` + profile + roofline + p95 decomposition | planned |

## `make cpu-check`

```bash
make cpu-check                                # the preflight
make cpu-check CPU_PROFILE=aws-c8a-16c-vnni-ttfa   # + resolve a serving profile, forbidden env absent
make cpu-check CPU_MODEL=qwen3-tts-1.7b-base       # + fingerprint the model (config + file sizes)
CPU_CHECK_STRICT=1 make cpu-check             # SMT/governor/cgroup gates FAIL instead of WARN
```

Every run writes `profiles/<date>_<host>_<binary-sha8>/` (gitignored) and repoints
`profiles/LATEST`:

```
manifest.json      provenance: binary sha256 + build tag, commit, dirty flag, dirty-source hash,
                   compiler, host, QWEN_*/OPENBLAS_*/OMP_* env, model fingerprint, isa_class,
                   every resolved dispatch value, the hardware block
hardware.json/txt  tools/box_info.sh (+ measured Copy/Triad bandwidth and knee via tests/membw.c)
build.txt          make info + compiler version
caps.txt           ./qwen_tts --caps
selftest_native.txt / selftest_fallback.txt
dispatch.txt/json  ./qwen_tts --dispatch-map
dispatch_gate.txt  tools/dispatch_gate.py: expected-vs-observed for this isa_class
env.txt · gate.txt
```

What the gates say and why each exists:

- **process-clean** — `surviving=0` for `qwen_tts`, `serve_parallel_wave`, `load_test.py`,
  `serve_soak.py`. A harness that outlived its server once ran concurrently with the next
  campaign and voided a day of numbers.
- **binary matches tree** — the build tag the binary reports (`GIT_REV`, `-dirty`) against
  `git rev-parse` now. A stale binary is the cheapest way to measure yesterday's code.
- **hardware fingerprint** + the three invalidating conditions from `box_info.sh` (SMT on,
  governor not `performance`, a cgroup CPU quota). WARN by default because cloud slices
  rarely let you fix them; `CPU_CHECK_STRICT=1` on hardware you own.
- **flag registry** — every `QWEN_*` the sources read is declared in `[FLAGS]`.
- **self-test native / fallback** — the dispatched kernels and the scalar path are both
  numerically correct on this ISA.
- **resolved dispatch map** and **expected-vs-observed** — see below. This is the gate that
  would have caught the prefill bug on day one.
- **profile resolves / forbidden env absent** — only with `CPU_PROFILE=`; the same checks
  `bench-suite` makes before a rung.

## `--dispatch-map`: resolved, not typed

`[FLAGS]` prints the environment as the operator typed it. That cannot show a compiled
default, a clamped value, or a predicate that lives outside the gate table. The map prints,
per logical feature, `compiled · supported · env · resolved · reason`, where **resolved is
the runtime predicate's own answer** (the Talker prefill's `use_matmat`, the CP `prefill2`
request, the decoder int8 gate, prefix cache, prepacks, pool spin, KleidiAI…), and a second
table with every `g_mm_gate[]` row as `qwen_mm_use()` answers it at `B = min_b`:

```
[DISPATCH] v=1 pid=… isa_class=x86_avx512bf16 build=1496938 simd=avx512bf16
  feature                          compiled supported env                        resolved reason
  talker.prefill.matmat_bf16       yes      yes       QWEN_PREFILL_MATMAT=unset  ON       avx512_bf16_matmat_available (VDPBF16PS)
  talker.prefill.f32_blas_fallback yes      yes       -                          OFF      not taken: bf16 matmat selected
  cp.prefill2                      yes      yes       QWEN_CP_PREFILL2=unset     ON*      default ON with AVX-512 VNNI; * needs every CP layer int8/int4
  decoder.int8                     yes      yes       QWEN_SD_INT8=unset         ON       default ON with AVX-512 VNNI …
  prepack.vnni                     yes      yes       QWEN_VNNI_PREPACK=unset    OFF      opt-in; int8 only; REJECTED 2026-09-03 on Zen5 …
[DISPATCH-GATE] v=1 rows=13 (resolved by qwen_mm_use at B=min_b; min_b/rows/cols = resolved(compiled))
  gate.int8.vnni     int8 VNNI vpdpbusd   yes  yes  ON   2(2)  0(0)  0(0)  QWEN_NO_VNNI   default ON
  gate.bf16.avx512   bf16 AVX-512 dpbf16  yes  yes  ON   1(1)  …
```

The same binary on the Arm reference host (16-core Axion, KleidiAI compiled in) shows the
Arm side of the same decisions — and the class of asymmetry the map is for: a feature that
is automatic on one ISA and opt-in on the other is visible in one glance at the `reason`
column:

```
[DISPATCH] v=1 pid=… isa_class=arm_i8mm_bf16 build=… simd=native
  feature                          compiled supported env                        resolved reason
  talker.prefill.matmat_bf16       yes      yes       QWEN_PREFILL_MATMAT=unset  ON       arm_bf16_matmat_available (BFMMLA)
  matvec.int8.sdot                 yes      yes       QWEN_NO_SDOT=unset         ON       default ON (vdotq_s32)
  matvec.bf16.bfdot                yes      yes       QWEN_ARM_BFDOT=unset       OFF      opt-in; default is the NEON 2-row fused bf16 matvec
  q8repack.neon                    yes      yes       QWEN_NO_Q8REPACK=unset     ON       q8_0 4-row repack, SDOT (SMMLA where i8mm exists); default ON
  kleidi.enabled                   yes      yes       QWEN_NO_KLEIDI=unset       ON       default ON when supported
  kleidi.int8 / kleidi.bf16 / kleidi.prefill / kleidi.qkv_fused                  ON       …
  kleidi.lhs_sym                   yes      yes       QWEN_KAI_LHS=unset         OFF      symmetric LHS quantisation (opt-in, QWEN_KAI_LHS=sym)
  kleidi.nchunk                    -        -         QWEN_KAI_NCHUNK=unset      384      compiled default 384 rows per bf16 GEMM chunk
  pool.spin                        -        -         QWEN_POOL_SPIN=unset       65536    compiled default 65536 (Linux/aarch64)
[DISPATCH-GATE] …
  gate.bf16.bfmmla   bf16 BFMMLA (arm)     yes  yes  ON   2(2) …  QWEN_NO_BFMMLA          default ON
  gate.int8.smmla    int8 SMMLA (i8mm)     yes  yes  ON   2(2) …  QWEN_NO_SMMLA           default ON
  gate.int8.sdot_mm  int8 SDOT loop over B yes  yes  OFF  2(2) …  QWEN_INT8_SDOT_MM (opt-in) opt-in, env unset
  gate.q4.kleidi     q4   KleidiAI (arm)   yes  yes  ON   1(1) …  QWEN_NO_KLEIDI          default ON
```

On an Apple M1 (`apple_m1`) the same table says `kleidi.enabled OFF — not compiled (needs an
i8mm target)`, `talker.prefill.matmat_bf16 OFF — no bf16 matmat unit -> f32 convert + SGEMM`,
and `pool.spin n/a — GCD dispatch on macOS`; on an M2+ (`apple_i8mm_bf16`) the BFMMLA/SMMLA
gate rows show `default OFF on Apple (QWEN_APPLE_MMLA=1)` while `gate.q4.smmla` is ON. Those
are the *expected* states for those classes, and `tools/dispatch_expect.json` says so; a
Linux Arm host with bf16 whose prefill row said OFF would be SUSPICIOUS exactly like the x86
case.

Rules the map obeys:

- a feature's `resolved` is produced by *calling* the same function the runtime calls
  (`qwen_prefill_matmat_resolved()`, `qwen_cp_prefill2_requested()`,
  `qwen_sd_int8_enabled()`, `qwen_mm_gate_describe()` → `qwen_mm_use()`, …). If a predicate
  is `static` in another file, that file exports a thin wrapper; the report never re-derives
  it. Adding a dispatch decision means adding a row, in `qwen_tts_dispatch.c`.
- `QWEN_DISPATCH_JSON=path` writes the same rows as JSON.
- `QWEN_DISPATCH_MAP=1`, `QWEN_SERVE_PROFILE`, or `QWEN_SHAPE_CENSUS` make the server print
  the map in its banner, so the engagement proof sits **inside the timed run's log**.

`tools/dispatch_gate.py dispatch.json` compares the map with `tools/dispatch_expect.json`
for the host's `isa_class` (`x86_amx`, `x86_avx512bf16`, `x86_avx512vnni`, `x86_avx2`,
`arm_i8mm_bf16`, `arm_dotprod`, `apple_m1`, `apple_i8mm_bf16`) and prints:

```
DISPATCH GATE  isa_class=x86_avx512bf16
  [SUSPICIOUS] talker.prefill.matmat_bf16: expected ON for x86_avx512bf16; compiled=yes supported=yes resolved=OFF; …
```

`SUSPICIOUS` = expected ON, compiled, supported, resolved OFF without an explicit env
switch — a fallback nobody asked for. `MISMATCH` = any other disagreement (including an
explicit env that contradicts the expectation: a choice, but a recorded one). `NOTE` =
the expectation names a feature this binary does not carry (wrong `SIMD=`?). Two generic
rules run on every class: an opt-out gate row that is compiled+supported and OFF without
its switch, and the f32/SGEMM prefill while a bf16 unit is compiled and supported.
`tools/dispatch_gate.py --selftest` replays the 2026-09-03 case and must print `SELFTEST PASS`.

## `make profile-cpu-check`

```
profile: …/profiles/2026-09-03_212303_box_323c2755  (2026-09-03T19:23:06Z)
ERROR: binary SHA256 differs (rebuilt since the profile): profile=… current=…
ERROR: env differs: QWEN_POOL_SPIN: profile=<unset> current=1
ERROR: resolved dispatch differs: decoder.int8: profile=OFF current=ON
ERROR: last profile was produced before qwen_tts_talker.c changed (2026-09-04 09:12)
PROFILE VALID: NO  (4 mismatches) -> run `make cpu-check` again …
```

The fingerprint is computed by one implementation (`tools/profile_check.py`) in both
directions, so "what was recorded" and "what is compared" cannot drift. It is the first
command of an optimisation session, and the one that protects a long session from
compaction: the profile you are reasoning from must describe the binary you are about to
change.

## `make cost-map`: where the wall time goes around the kernels

The shape census answers *which* kernel ran, with which shape, on which ISA path. It does
not answer *where the time went*, and in particular it says nothing about the work that is
not a kernel: transposes, RMS norms, KV writes, the swap between layouts, the wait for the
thread pool. `QWEN_COST_MAP` adds a second, coarse layer for that.

**The semantics are fixed before any number is collected**, because a table that mixes
inclusive and exclusive timing is worse than no table:

- every region is **inclusive** — it contains everything entered inside it;
- **self time is derived**, `ns - child_ns`, never measured separately;
- **nesting is declared** in the region table and **verified at runtime**: a `begin` whose
  dynamic parent is not the declared one bumps `nest_mismatch` instead of being accepted.
  `QWEN_RGN_MULTI` marks the regions that legitimately have several parents;
- accumulation is **thread-local**, merged only at dump time, so the hot path has no
  atomic read-modify-write and no lock;
- one `clock_gettime(CLOCK_MONOTONIC)` per begin and per end, **never inside an inner
  kernel loop**. When the map is off, a marker is one load of an `int` and a branch.

Two levels, so the per-layer sites cannot tax the default:

| `QWEN_COST_MAP` | what it adds |
|---|---|
| unset / `0` | off |
| `1` | component totals, Talker prefill stages, decoder stages, pool and server |
| `2` | plus the per-layer CP decode stages and the Talker prefill layout transposes |

```
make cost-map           PROFILE_MODEL=… PROFILE_PROFILE=… COSTMAP_CONC=1 COSTMAP_LEVEL=1
make cost-map-overhead  …same…
```

`cost-map` runs the workload **twice with the same binary** — census only, then census plus
cost map — and diffs the two census reports. The cost map ships with the proof that turning
it on did not move a single executed path id, coverage number, UNKNOWN or fallback count;
without that proof the numbers would be unfalsifiable. `cost-map-overhead` measures the
price **A/B/B/A interleaved**, never one clean run followed by one instrumented run: on this
class of box the drift between two identical runs can be larger than the effect, and the
report says so explicitly when the clean arms spread more than the measured delta.

### `sync_wait` is not one bucket

The one thing a wait accounting must not become is a drawer for everything unexplained.
What this pool can and cannot separate:

| category | status |
|---|---|
| waiting for worker completion | **measured** — `runtime.pool_wait_completion`, the window between the caller finishing its own chunks and `completed == need`, one clock pair per dispatch |
| waiting to submit a job | **measured** — `runtime.pool_submit_wait`, the `submit_mtx` acquisition |
| scheduler / admission idle | **derived** — `runtime.admission`, from the job timestamps (`enq_ms` → `t_admit`) |
| barrier synchronisation | **UNRESOLVED** — this pool has no distinct barrier primitive; completion is a counter plus a condvar, so barrier time is not separable from "waiting for worker completion" without changing the pool |

On Apple's GCD backend `dispatch_apply` both dispatches and joins, so the wait is inside the
call and is *not* recorded rather than guessed; only `runtime.pool_dispatch` appears there.

### Regions that are derived, not bracketed

The server request lifecycle crosses threads: the connection thread receives, the scheduler
admits, the batch loop serves. No thread-local stack can bracket that, so
`runtime.request.total` and `runtime.admission` are added as durations from the job
timestamps and carry `"mode": "derived"` in the JSON. Everything else is `"mode": "stack"`.
The report prints the column, so the two kinds are never read as the same measurement.

### The prefork parent's work is not counted N times

A prefork server loads the model and pre-warms in the parent, then forks. Without care
every worker would inherit those counters and report them again, so merging N workers
would count one pre-warm N times. `qwen_costmap_after_fork()` clears the map in the child
next to `qwen_threadpool_after_fork()`: a worker's cost map describes only what that
worker actually served, and start-up work is out of scope by construction.

### Reading the table

Shares are given **within a thread role**. The speech decoder runs on its own thread
concurrently with the Talker and the code predictor, so the roles do **not** sum to 100% of
the request, and any report that presented them as one flat breakdown would be lying.

The integrity block is a gate: `nest_mismatch`, region stack overflows and unbalanced ends
must all be zero, otherwise the tree is not trustworthy and the numbers are refused.
`leaked` counts regions closed by the unwind that every component entry point performs on
the way out — early `return`s inside a decoder body cannot leave a region open forever.

## Execution domain: topology identity and per-mask roofs

A bandwidth roof belongs to **one CPU mask**, and a worker roof can never be obtained by
dividing a host roof — on any architecture. Measured on the AWS c8a.4xlarge (2 CCX x 8
cores, SMT off):

| mask | read | copy | triad | read t90 / t95 | triad t90 / t95 |
|---|---|---|---|---|---|
| 0-15 (host) | 165.0 | 110.0 | 111.6 | 16T / 16T | 4T / 4T |
| 0-7 (one CCX) | 66.9 | 53.4 | 55.2 | 4T / 4T | 2T / 2T |
| 8-15 (other CCX) | ~67 | ~53 | ~54 | — | — |

The prefork server pins each worker to one CCX, so a single request can only reach ~67
GB/s of read bandwidth. Judging it against the host's 165 turned "at roof" into "57% of
roof" and invented ~230 ms/request of headroom that does not exist. That is why the
comparison is now a contract in code rather than a habit.

### Topology identity

`2x8` and `1x8` are not identities. These are four different machines:

| id | workers | threads | mask | domain |
|---|---|---|---|---|
| `1W8T_m0-7` | 1 | 8 | 0-7 | one CCX |
| `1W8T_m0-15` | 1 | 8 | 0-15 | 8 threads spanning both |
| `1W16T_m0-15` | 1 | 16 | 0-15 | 16 threads spanning both |
| `2W8T_m0-7\|8-15` | 2 | 8 each | 0-7 + 8-15 | one CCX each |

**Single-worker affinity differs from prefork affinity.** `--prefork > 1` splits the host
and calls `sched_setaffinity` per worker. `--prefork 1` never reaches that path, so
without help "1x8" means *8 threads free to roam every cpu of the host*. `--cpu-mask 0-7`
supplies the missing control; it is a **benchmark/qualification** flag, changes nothing by
default, is refused together with `--prefork > 1`, and is applied before the thread pool
exists because pthreads inherit the creating thread's affinity. Both serving paths now
print `[TOPOLOGY] v=1 worker=N pid=P configured_mask=M actual_mask=A threads=K mode=X` —
both masks, because "asked for" and "granted by the kernel" are different claims.

### Tools, inputs, outputs

| tool | consumes | produces | when |
|---|---|---|---|
| `tests/membw.c` | `--cpus MASK`, `--threads`, `--l3-mb` | per-mask copy/triad/**read** roof, full thread sweep, t90/t95/t99, working set, residency | offline |
| `tools/topology.py` | server log (`[TOPOLOGY]`, legacy `prefork:`), `/proc/<pid>/status`, sysfs LLC/NUMA, `hardware.json` | `topology.json` | offline |
| `tools/roofs.py` | `membw` + masks | `profiles/roofs/roofs_<hwfp>.json`; also the `select_roof` contract | offline |
| `tools/topology_report.py` | `topology.json` + roof store (+ numerators) | the doctor block | offline |
| `tools/envelope_report.py` | canonical `parallel_*.json`, census, cost map, topology, roofs | `envelope.json` + cliffs | offline |
| `tests/roof_matvec_int8.c` | the four Talker decode shapes | `PRIMITIVE` roof, DRAM-resident | offline |

Nothing above runs on the serving path. The only engine-side additions are `--cpu-mask`
(one `sched_setaffinity` at startup) and one `[TOPOLOGY]` line per process.

Artifacts land in the run's `profiles/<date>_<host>_<sha8>/` (`topology.json`,
`roofs.txt`, `topology_report.txt`) and in `profiles/roofs/` for the roof cache.

### Roof scopes

* `HOST` — every cpu the process may use.
* `WORKER[i]` — measured under **that worker's** mask.
* `PRIMITIVE[op]` — same mask, thread count and access pattern as the operation judged.

A roof is a property of *(hardware, mask, benchmark)* and **not** of the model, so the
store is keyed by a hardware fingerprint and survives model switches; it is re-measured
when the hardware changes or when `tests/membw.c` itself changes.

### ROOF UNKNOWN / NOT COMPARABLE

`roofs.select_roof()` proves hardware fingerprint, mask, benchmark kind and residency
before any division:

* no roof of that kind measured at all → **ROOF UNKNOWN**;
* roofs exist but describe another mask → **NOT COMPARABLE**;
* mask matches but residency differs → **NOT COMPARABLE**.

There is **no fallback from WORKER to HOST** anywhere in the path. A percentage is never
printed unless the scopes were checked.

### Reproducing a topology qualification

```
make membw
make roofs ROOF_MASKS=0-15,0-7,8-15            # per-mask roofs, cached
make cpu-check CPU_MODEL=… CPU_PROFILE=…       # topology.json + doctor block
python3 tests/serve_parallel_wave.py --topo '1x8@0-7,1x8@0-15,1x16@0-15,2x8' \
    --conc 1,2,4,5,6 --waves 3 --model … --profile … --out OUT
make envelope WAVE_JSON='OUT/parallel_*.json' TOPO_JSON=OUT/topology_*.json
```

`--topo WxK@MASK` is the canonical way to express a single-worker cell; use `+` inside a
mask (`0+2+4`) because `--topo` is comma-separated. One server per topology, held while
concurrency rises. Every result row carries `topology_id`, `worker_masks` and
`mask_confidence`.

## Counters that now have a delimiter

Every SIGUSR1 dump of the server (`POOLSTATS`, `[shape-census]`, `[batch-audit]`,
`[kernel-timing]`) is bracketed by

```
[DUMP] v=1 pid=… seq=N ts=… clock=CLOCK_MONOTONIC begin
…
[DUMP] v=1 pid=… seq=N end
```

so a harness that signals before and after a cell can subtract the two.

## Related

- [ENGINEERING-METHOD.md](ENGINEERING-METHOD.md) — the rules these gates encode (§8 harness
  validity, §9 runtime over source, §17 the ledger).
- [serving-operations.md](serving-operations.md) §2 — the box break-in, now starting with
  `make cpu-check`.
- [feature-flags.md](feature-flags.md) §7 — the diagnostics, and what `[FLAGS]` cannot show.
- `make kernel-tune` / `make tune-archive BOX=` — measured dispatcher thresholds, archived
  under `docs/boxes/` for `tests/mm_tune_compare.py`.
