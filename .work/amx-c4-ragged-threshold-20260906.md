# AMX C4 ragged dispatch threshold A/B — 2026-09-06

## Decision

Lowering the ragged decoder pool-dispatch threshold is a real but incomplete
serving improvement. The best short mixed-bank arm was
`QWEN_SD_RAG_MIN_PANELS=2`, with pooled `STREAM_RTF p50/p95 = 0.9004/0.9538`
and three window p95 values `0.9513, 0.9508, 0.9538`. It had zero errors,
queue rejects and request timeouts. This is materially better than the threshold
8 control, but it does not meet the useful-margin target (`<=0.90`) and the
zero-buffer prebuffer diagnostic remains about `2.37 s` p95. Do not call C4
qualified and do not start C5/C6.

The runtime knob is kept as an experimental control with default `8`; no
production default changed. The next work must explain the remaining steady-state
tail or create a measured larger compatible matrix workset, not try more threshold
values.

## Identity and controls

- canonical branch: `feature/x86-amx-vnni-oss`;
- source base: `7236218` plus the uncommitted threshold candidate, fingerprint
  `7236218-dirty:433bc7d29976`;
- remote binary SHA-256:
  `343ad3471911cb837e476122d78460c39b8827d3764d9f858de85dcd42f3cf40`;
- host: GCP `c4-standard-24`, Xeon Platinum 8581C, one socket/NUMA, CPUs
  `0-11` online, SMT off;
- topology: two prefork workers × six physical cores, server batch cap `2`;
- model: Qwen3-TTS 1.7B, `tests/load_texts_en.txt`, stratified seed `42`;
- fixed controls: chunk `32`, Design D INT8, decoder batching on, engine pool,
  `QWEN_SD_AMX_BF16=0`, no profiler/census, 1.5 minute run, 15 second warm-up,
  three 30 second windows;
- each arm used a separate port and completed before the next arm started.

The remote AMX build was clean-built with `make clean` and `make -B blas
SIMD=amx -j12`; `--self-test` passed. `--caps` reported `SIMD=amx`, active AMX
INT8 and BF16. Every server arm logged `QWEN_SD_AMX_D=1`,
`QWEN_SD_POOL=engine` resolved to `engine`, and decoder batching enabled. The
screen did not enable intrusive per-kernel census, so these logs prove the
selected AMX configuration and build capability, not a new wall/MAC share.

## Phase 1: serial ragged work

The existing diagnostic had 252 ragged decoder calls. A call uses the engine
pool only when `n_panels >= 8`; otherwise the same worker body runs on the
caller. The `<8` bucket was:

| `n_panels` | calls | columns | logical MACs | Kp-normalized work | diagnostic wall sum | AMX timer sum |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 3,840 | 15.854 G | 15.854 G | 123.4 ms | 85.5 ms |
| 2 | 24 | 6,144 | 25.367 G | 25.367 G | 189.1 ms | 137.2 ms |
| 3–4 | 15 | 5,376 | 10.305 G | 10.494 G | 82.3 ms | 53.5 ms |
| 5–7 | 24 | 15,360 | 15.854 G | 16.609 G | 127.6 ms | 76.1 ms |
| **1–7 total** | **99** | **30,720** | **67.381 G (31.9%)** | **68.325 G (29.9%)** | **522.3 ms (19.4%)** | **352.2 ms (24.9%)** |

The denominator is the diagnostic sum: `211.063 G` logical MACs,
`228.757 G` Kp-normalized work, `2,692.5 ms` timer wall and `1,413.7 ms`
AMX timer. These percentages are diagnostic sums, not a qualification wall-time
claim. Shapes explain the policy: `M=96/192` commonly create many N panels,
while `M=768` often creates only one or two and is therefore serial at the old
threshold.

## Phase 2: server A/B

All arms used the same short mixed-bank workload. `prebuffer` and `stall_max`
are zero-buffer client diagnostics, not direct server underrun counters.

| minimum panels | completed / KPI / probes | TTFA p50/p95 ms | STREAM p50/p95 | pooled prebuffer p95 | pooled stall p95 | windows STREAM p95 | resource | errors / rejects / timeouts |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 8 control | 38 / 34 / 4 | 195.2 / 505.5 | 0.9072 / **1.0035** | 2.550 s | 2.038 s | 0.948, 0.973, 1.004 | FAIL (PSS drift) | 0 / 0 / 0 |
| 6 | 39 / 35 / 4 | 225.0 / 523.7 | 0.9157 / **0.9974** | 2.654 s | 2.024 s | 0.961, 0.997, 0.946 | PASS | 0 / 0 / 0 |
| 4 | 40 / 36 / 4 | 206.7 / 727.9 | 0.9076 / **0.9927** | 2.587 s | 2.167 s | 0.937, 1.036, 0.974 | PASS | 0 / 0 / 0 |
| **2** | **38 / 34 / 4** | **195.2 / 564.9** | **0.9004 / 0.9538** | **2.370 s** | **1.993 s** | **0.951, 0.951, 0.954** | **PASS** | **0 / 0 / 0** |

Relative to the control, threshold 2 improves pooled p95 by `0.0497` RTF
(`~4.95%` relative) and adds about `59 ms` TTFA p95. The three threshold-2
windows are more consistent than the control's last window, but the short run
is not long enough to establish production stability. The threshold-4 arm also
contains a window with TTFA p95 `770.9 ms`, so lower dispatch overhead is not
monotonically free.

The server logs report batch cap `2` and decoder batching, but this experiment
did not add a new effective-batch/panel census. Existing non-intrusive evidence
puts the global effective batch near `2.3` and the two workers near `1.1/1.3`.
The threshold change therefore makes more existing ragged jobs enter the pool;
it does not create cross-request matrix aggregation.

## Validation

- local `make -B blas -j4`: pass;
- local `./qwen_tts --self-test`: pass;
- local `make test-sd-pool-config check-flag-registry test-selftest`: pass;
- `python3 tools/check_plan.py`: pass before this addendum;
- `git diff --check`: pass before this addendum;
- remote clean AMX build and self-test: pass;
- all four server arms: zero errors, queue rejects and request timeouts;
- `make test-golden`: environment failure in local `librosa`/Numba cache loading
  (`cannot cache function '__o_fold'`), not an audio comparison result.

After the screen, the committed tree was synchronized as `b604a28:clean` and
rebuilt cleanly on the same host. The reproducibility binary SHA-256 was
`f6492aa690a364611b897a6cb94088c1d6ab4514f0e47b0bdaffe2ecc4022939`;
`--caps` again reported active AMX INT8/BF16 and `--self-test` passed with
zero failures. The host remained SMT off with CPUs `0-11` online.

## Next action

Do not run a full five-minute qualification for threshold 2 yet: it is below one
in this screen but not near the requested `<=0.90` margin. Keep default 8 and use
threshold 2 only as a reproducible control for the next attribution/aggregation
experiment. The highest-ROI question is now whether the remaining tail is caused
by low useful per-worker work and missing compatible cross-request aggregation,
or by preparation/pool cadence that threshold lowering cannot remove.
