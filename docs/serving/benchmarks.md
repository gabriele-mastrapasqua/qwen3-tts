# Measuring a serving box — which tool answers which question

[← Serving index](README.md)

There are six instruments in this repository and they are not interchangeable. Picking the
wrong one is how a number that cannot support a claim ends up in a report, so this page is
organised by the **question** rather than by the tool.

| the question | the tool | how long |
|---|---|---|
| what is this machine capable of, before any model runs? | `make doctor` | < 1 min |
| does this build do what its name says? | `--caps`, `--self-test`, `make test-golden` | seconds |
| how does one request behave? | `./qwen_tts --stream`, `tests/load_test.py -c 1` | seconds |
| how many simultaneous streams survive a synchronised burst? | `tests/serve_parallel_wave.py` | minutes |
| **is this change a win?** | **`tests/load_test.py` as a screen** (see below) | 5–10 min |
| at what concurrency does playback break? | `make soak-fast` | 10–30 min |
| **what number goes in the profile?** | **`make bench-soak` (30 min closed loop)** | 30 min+ |
| is a running server healthy right now? | `--metrics-port` + [`metrics.md`](metrics.md) | continuous |

Two of these produce numbers you may publish. The rest produce numbers you may act on.

## The rule that everything else follows

**A screen detects a change. A qualification measures a number.**

They are different instruments and their outputs are not interchangeable in either direction.
A screen's figure never goes into `configs/perf/*.json`, into a release note, or into a doc; a
qualification is too slow to run on every idea. The discipline is the same one `make doctor`
applies with its `[MEASURED]` / `[PREDICTED]` labels: the instrument decides what you are
entitled to claim.

## Why a two-minute run cannot answer a precision question

A percentile is estimated from the samples in its tail, and there are `N × (1−q)` of those. At
C16 a 32-core box serves roughly 1.8 requests/second, so:

| soak | requests | p50 | p95 | p99 |
|---|---|---|---|---|
| 2 min | 216 | ±6% | **±9%** | **±15%** |
| 10 min | 1,080 | ±2% | ±5% | ±7% |
| **30 min** | **3,240** | **±1%** | **±2%** | **±4%** |
| 60 min | 6,480 | ±1% | ±2% | ±3% |

*(Central 95% of the estimate over 300 simulated runs of a lognormal fitted to real TTFA:
p50 116 ms, p95 199 ms. A real distribution has a heavier tail than a lognormal, so these are
**optimistic**, especially for p99.)*

Three consequences, and the first one is the good news:

- **The 30-minute soaks behind the registered profiles are sound.** p95 to ±2% is a number you
  can quote.
- **A p99 from a short run is two samples.** At N=216 the count above the p99 is
  `Binomial(N, 0.01)`: mean 2.15, standard deviation 1.5. It can move 50% between identical
  runs. Never quote it, in either direction.
- **Precision improves as `sqrt(N)`**, so halving the error bar costs four times the samples.
  That is why the screen changes the *corpus* rather than only the clock.

`tests/load_test.py` now prints a bootstrap confidence interval and the supporting sample count
beside every percentile, and marks any that rests on fewer than ten samples. Note the interval
is itself optimistic at the extreme tail — resampling cannot invent a value worse than the worst
one observed — so out there trust the count, not the interval.

## The fast screen

A screen wants **samples per minute**, not minutes. Short requests give more of both, at the
price of representativeness:

| at equal 2 minutes | req/s | N | p95 |
|---|---|---|---|
| long-text mini-soak | ~1.8 | 216 | ±9% |
| `medium`-class screen | ~6 | 720 | **±5%** |

So the screen runs the `medium` class — ~4–5 s of audio per request. Not `short`: at ~1.5 s a
request produces too few chunks for the cadence metrics to see anything, and the decoder is
barely exercised.

Two things a screen cannot do, which is why it does not replace the soak:

- **Its absolute numbers will not match the qualification.** A different corpus is a different
  latency distribution. Validate a screen by checking that a *known* regression shows up in it,
  never by checking that it agrees with the 30-minute figure — it will not, and it should not.
- **It is blind to anything that only appears on long requests**: KV growth, long-sequence
  decoder behaviour, resource drift. Those need the real soak.

Define a screen by **target sample count, not duration**. The sample count is what controls
precision; the duration is a proxy that changes with the box, the model size and the text mix.

## The corpus, and why it is versioned

`tests/load_texts_en_v2.txt` — 277 texts across `short`, `medium`, `conversational`, `long` and
`italian`, each class keeping the character band it had in v1.

The v1 bank held **21 texts in total**, three of which carried the whole `long` class: in a
30-minute stratified soak each of those was spoken about 216 times. That repetition does not
trip an engine cache — the prefix cache is keyed on speaker and language, not on text — but it
does something quieter and worse: it **removes text variability from the measurement**, so the
spread comes out narrower than reality and a regression that only shows on unusual lengths is
invisible.

**A bank change is an era boundary.** Numbers measured on v1 and on v2 are not comparable, in
exactly the sense that pre-v2 serving numbers are not comparable with v2 ones. So:

- every artifact records `text_bank`, `text_bank_version` and `text_bank_sha256`;
- v1 stays at `tests/load_texts_en.txt` and reports itself as `unversioned`, so older runs
  remain identifiable;
- classes were **extended inside their bands, never redefined** — making `short` mean 70
  characters instead of 25 would have broken comparability silently.

## What never goes into a profile

- A screen result, at any length.
- A p99 from anything under thirty minutes.
- A number from a bank whose version the artifact does not record.
- Anything from the metrics endpoint: that is an operations signal, and
  [`metrics.md`](metrics.md) explains why a bucketed quantile is not an acceptance figure.

## See also

- [`cpu-operations.md`](cpu-operations.md) — running the suite, the profile gate, the soak
- [`metrics.md`](metrics.md) — what a live server publishes about itself
- [`../ENGINEERING-METHOD.md`](../ENGINEERING-METHOD.md) — why the measurement rules are shaped
  the way they are
