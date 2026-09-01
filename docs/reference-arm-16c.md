# A 16-core Arm serving box, measured end to end

One machine, one build, one profile, every rung of the suite. This page exists so that
[`serving-operations.md`](serving-operations.md) can describe a *procedure* without carrying a
table for every cell, and so that a second box has something to be compared against.

**Nothing here transfers by itself.** It is one hardware configuration at one precision with
one text bank; the shapes (what rises with concurrency, what rises with input length, where
realtime is lost) are the part worth carrying to another machine — the milliseconds are not.

## The setup

| | |
|---|---|
| CPU | 16 cores, Arm Neoverse-V2 class, **SMT absent** (`Thread(s) per core: 1`), 1 NUMA node |
| cache / bandwidth | 80 MiB LLC (5.0 MiB per core) · measured Triad **336 GB/s**, knee at 16 threads |
| build | `make blas`, OpenBLAS pthread, KleidiAI with SMMLA and BFMMLA both live (`--caps`), `--self-test` PASSED |
| model | open weights, 1.7B and 0.6B, `--int8` |
| profile | [`configs/perf/axion-16c-ttfa.json`](../configs/perf/axion-16c-ttfa.json), applied and verified with `check-flags` |
| topology | `2x8` — two pre-forked workers, 8 threads each, `--batch-size 8` |
| bank | `tests/load_texts_en.txt`, classes `short` / `medium` / `long` / `conversational` / `italian` |
| KPIs | TTFA = send → first audio chunk · stream RTF = per request, after the first chunk · errors |

Every table below comes from `make bench-suite`, whose manifest carries the binary sha256, the
source commit, the resolved environment and the exact commands.

## 1. Choosing the topology

`make bench-topo BENCH_TOPO=1x16,2x8,4x4 BENCH_CONC=1,4` — 1.7B, three waves:

| topology | C | TTFA p50 | TTFA p95 | stream RTF p50 | measured batch | errors |
|---|---:|---:|---:|---:|---:|---:|
| `1x16` | 1 | **46 ms** | 54 ms | 0.35 | — | 0 |
| `1x16` | 4 | 141 ms | 142 ms | 0.91 | — | 0 |
| `2x8` | 1 | 54 ms | 72 ms | 0.43 | 0.64 | 0 |
| `2x8` | 4 | **124 ms** | 138 ms | 0.72 | 2.73 | 0 |
| `4x4` | 1 | 87 ms | 110 ms | 0.72 | 0.73 | 0 |
| `4x4` | 4 | 136 ms | 160 ms | 0.80 | 2.77 | 0 |

Every core on one request wins at C=1 and loses at C=4; four narrow workers lose both. `2x8` is
the shape that holds both, and it is what the rest of this page uses.

## 2. The qualification curve (`realistic` rung, waves of C at t=0)

Four waves per level, whole bank, both languages, audio p50 ≈ 4.5 s (1.7B) and 6.5 s (0.6B):

**1.7B**

| C | TTFA p50 | TTFA p95 | stream RTF p50 | stream RTF p95 | req/s | measured batch | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 76 ms | 165 ms | 0.44 | 0.45 | 0.31 | 0.90 | 0 |
| 2 | 105 ms | 213 ms | 0.47 | 0.48 | 0.39 | 1.30 | 0 |
| 4 | 233 ms | 335 ms | 0.74 | 0.86 | 0.42 | 1.79 | 0 |
| 6 | 278 ms | 466 ms | 0.92 | 1.04 | 0.50 | 2.55 | 0 |
| 8 | 384 ms | 472 ms | **1.14** | 1.28 | 0.59 | 3.40 | 0 |

**0.6B**

| C | TTFA p50 | TTFA p95 | stream RTF p50 | stream RTF p95 | req/s | measured batch | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45 ms | 85 ms | 0.35 | 0.38 | 0.25 | 0.92 | 0 |
| 2 | 65 ms | 119 ms | 0.37 | 0.39 | 0.30 | 1.31 | 0 |
| 4 | 121 ms | 197 ms | 0.62 | 0.66 | 0.33 | 1.93 | 0 |
| 6 | 150 ms | 240 ms | 0.78 | 0.86 | 0.41 | 2.81 | 0 |
| 8 | 207 ms | 271 ms | **1.00** | 1.09 | 0.46 | 3.80 | 0 |

Read three things from those two tables:

- **realtime is lost between C=6 and C=8 on the 1.7B** (0.92 → 1.14 at p50) and exactly *at*
  C=8 on the 0.6B. A player that starts at the first chunk stalls above 1.0, so the honest
  claim for this box is a band up to C=6, which is what the profile records;
- **the measured batch is never C.** At client concurrency 8 the engine's own counter says 3.4,
  because two workers spread eight requests and each one's batch is what it holds *at that
  step*. This is the field to quote, not the client's C;
- **errors stayed at zero everywhere**, including the levels above realtime — the box degrades
  by getting slower, not by dropping requests.

## 3. Input length is prefill, and it dominates first audio

Same machine, same topology, same profile — only the class of text changes:

| rung | audio p50 | model | C | TTFA p50 | stream RTF p50 |
|---|---:|---|---:|---:|---:|
| `short-diverse` | 1.9 s | 1.7B | 4 | 117 ms | 0.73 |
| `long-diverse` | 18.4 s | 1.7B | 4 | **375 ms** | 0.79 |
| `short-diverse` | 2.5 s | 0.6B | 4 | 77 ms | 0.63 |
| `long-diverse` | 24.6 s | 0.6B | 4 | **191 ms** | 0.70 |

First audio triples on the 1.7B for a ten-fold longer text while the stream realtime factor
barely moves, because prefill grows with the prompt and everything after the first chunk does
not. A first-audio figure quoted without the text length behind it is not a figure.

## 4. The engineering inner loop (`fast` rung)

Short texts, five waves, English only — the number to iterate against, never to publish:

| model | C | TTFA p50 | TTFA p95 | stream RTF p50 | audio p50 |
|---|---:|---:|---:|---:|---:|
| 1.7B | 1 | 53 ms | 73 ms | 0.44 | 1.84 s |
| 1.7B | 4 | 110 ms | 138 ms | 0.72 | 1.84 s |
| 0.6B | 1 | 37 ms | 43 ms | 0.34 | 2.48 s |
| 0.6B | 4 | 74 ms | 88 ms | 0.62 | 2.48 s |

## 5. What the profile's environment is worth here

Same binary, same bank, `2x8`, four waves, the arms differing only in whether the profile
environment was applied ([`feature-flags.md`](feature-flags.md) §0 explains each variable):

| arm | C | TTFA p50 | stream RTF p50 | context switches/s |
|---|---:|---:|---:|---:|
| profile | 1 | 57 ms | 0.44 | 11,910 |
| compiled defaults | 1 | 51 ms | 0.45 | 7,880 |
| profile | 4 | **109 ms** | **0.72** | 30,153 |
| compiled defaults | 4 | 133 ms | 0.81 | 68,984 |

## 6. Correctness and audio, not only the clock

- `--self-test`: PASSED, 0 failing cases (the cross-ISA kernel oracle).
- **identity gate: PASS on all four rungs** of the 1.7B — a traced pass at C=1 and C=4
  produces the same output, so nothing above batch 2 changed the content.
- `tools/wav_qc.py` over a `--save-audio` run: 4 files, 0 FAIL, 1 WARN (a step discontinuity
  flagged for listening). A clean row means the waveform is intact and nothing more; the ear is
  the verdict.

---

## Reproducing this page

```bash
make blas GIT_REV=$(git rev-parse --short HEAD)
make bench-fingerprint
make bench-topo   BENCH_MODEL=<1.7B dir> BENCH_PROFILE=axion-16c-ttfa BENCH_TOPO=1x16,2x8,4x4
make bench-suite  BENCH_MODEL=<1.7B dir> BENCH_PROFILE=axion-16c-ttfa BENCH_TOPO=2x8 \
                  BENCH_ARGS="--corpus tests/load_texts_en.txt --identity-gate"
make bench-suite  BENCH_MODEL=<0.6B dir> BENCH_PROFILE=axion-16c-ttfa BENCH_TOPO=2x8 \
                  BENCH_ARGS="--corpus tests/load_texts_en.txt"
```

Let the box settle between runs: the suite refuses to start at `loadavg >= 2.0`, and that
includes the decay from the run before it.
