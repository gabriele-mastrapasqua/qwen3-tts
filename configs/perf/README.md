# Server performance profiles

**One question: given this hardware and this objective, how should the engine be run?**

```
configs/perf/schema.json          the format, architecture-neutral
configs/perf/axion-16c-ttfa.json  recommended low-TTFA config for a 16-core Google Axion host
configs/perf/recommended.json     stable entry point; POINTS at the current recommendation
```

```bash
tools/perf_profile.py validate                                     # all profiles
tools/perf_profile.py command recommended --model DIR --port 8000  # the exact invocation
tools/perf_profile.py server-env recommended                       # comma form for the harness
tools/perf_profile.py check-flags recommended --log server.log     # what the engine declared
python3 tests/test_perf_profile.py
```

## The separation that makes this worth having

| | contains | example |
|---|---|---|
| **performance profile** | how the server should be configured | `2 workers x 8 threads, --int8, OPENBLAS_THREAD_TIMEOUT=1` |
| **run manifest** | what exactly was measured that once | commit, binary sha, corpus sha, seed, timestamp, host |

A profile carries **no** measurement provenance, so it survives new binaries and new
campaigns. A benchmark result names the profile it used:

```json
{ "performance_profile": "axion-16c-ttfa" }
```

Later this becomes `axion-16c-throughput`, `x86-32c-ttfa`, … under the same schema, each with
its own qualification. Nobody will have to remember *why* Axion was 2x8.

## Rules the loader enforces

- a profile is **complete** or an **alias**, never half of each — an alias that repeats a
  value is a copy that will drift;
- `workers x threads` may not exceed the declared physical cores;
- an environment value may not contain whitespace: `--server-env` splits on **commas**, and a
  space folds every later variable into the previous one's *value*. That defect produced an
  A/B whose two arms both ran with the flag unset;
- a `QWEN_*` variable the engine does not declare is refused: it would be set and never
  verifiable;
- a setting fixed both on the command line and in the profile body is refused.

`objective` states the regime, because `2x8` is a measured property of this machine and not a
universal one. `precision` is per component, because "int8" names completely different paths
in the talker, the code predictor and the speech decoder.

**These are deployment recommendations, not engine defaults.** Nothing here is compiled in;
`recommended.json` does not claim 2x8 is right on every CPU.

## The profile is a gate, not a document (2026-08-31)

A profile that has to be remembered is a profile that will be forgotten, and the run that
forgets it still prints a table. Measured on the qualification host, same binary, same bank,
same uptime, interleaved arms:

| round | without the profile | with it |
|---|---:|---:|
| 1 | 108 ms | **66 ms** |
| 2 | 66 ms | **66 ms** |
| 3 | 99 ms | **66 ms** |

C=1 on the FAST bank. The bare arm is bimodal — it lands on either value and looks equally
definitive — and the mechanism is visible beside it: 42 500 context switches per second
against 12 000, and 7.9 cores busy against 7.2, because OpenBLAS idles by spinning and
contends with the engine's own pool. Nothing in the output said which configuration had
produced the number.

So three things now enforce it, and each exits non-zero:

- **`tests/serve_parallel_wave.py` refuses to start** without `--profile <name>` or an
  explicit `--no-profile '<reason>'`. Values passed with `--server-env` are merged on top and
  every override is printed as `profile_override=`.
- **`tools/perf_profile.py forbidden-env`** lists the variables whose profile value is `null`.
  They must be **absent from the environment**, not merely unset by us: the engine sizes
  OpenBLAS to the thread budget at startup with `openblas_set_num_threads()`, and
  `qwen_blas_set_threads()` returns immediately when
  `OPENBLAS_NUM_THREADS` is present, so somebody else's export silently replaces the
  qualified topology.
- **`make bench-suite`** is the single entry point for numbers that leave this repo. It owns
  the profile, checks the environment, stamps the binary, runs every rung, runs the identity
  gates and writes a manifest containing the exact commands.

Individual rungs remain available for investigation. The suite is the gate for a report.

## Which profiles ship, and why there are not more

| profile | status | what it is |
|---|---|---|
| `axion-16c-ttfa` | **qualified** | a 16-core Arm host, measured end to end: topology, thread split, batch width, runtime environment and the concurrency band the claim covers |
| `generic-16c-starting-point` | **unqualified** | a place to START on a 16-core Arm server. Nothing in it was measured on your machine, and it says so in its own `qualification.notes` |
| `recommended` | alias | resolves to the qualified one; carries no values of its own |

There is deliberately no profile per machine type we have ever touched. A profile claims
that a configuration was **measured** on that hardware, and a file that looks like a
qualification while carrying guesses is worse than no file: it is a wrong default that
nobody re-measures, which is the same failure the runtime-environment gate above exists to
prevent.

The way to add one is the break-in sweep in
[`../../docs/serving-operations.md`](../../docs/serving-operations.md) §2, then a profile
with `status: qualified` and the numbers that earned it.

### The one rule that generalises, and the one that does not

**Generalises:** a single request saturates at about eight threads and regresses at sixteen.
That is a property of the work, not of a particular chip.

**Does not:** which `W x K` topology wins. It depends on the concurrency you expect, because
a worker only sees a batch at `C >= 2W` and the engine takes a different kernel above batch
2 — so four workers can win at high concurrency and lose at the concurrency you actually
care about. That one is measured per box, every time.

## Adding a machine

More of these are expected: a profile per deployment that has actually been measured, not per
machine type anyone has touched.

```bash
tools/perf_profile.py new <id> [--like axion-16c-ttfa]   # a skeleton, marked unqualified
$EDITOR configs/perf/<id>.json
tools/perf_profile.py validate
```

`new` starts from an existing profile's STRUCTURE and blanks everything that was measured:
the cpu family, the hardware notes and the whole qualification block. Copying a profile by
hand is how a value from another machine becomes a claim about this one — every field is
filled in, every field looks deliberate, and nothing says which of them anybody measured.
A skeleton starts at `unqualified` and makes filling a field a decision.

Then run the break-in sweep in
[`../../docs/serving-operations.md`](../../docs/serving-operations.md) §2, put what it
measured into the file, and set `qualification.status` to `qualified` with the numbers that
earned it.

### Naming

`<platform>-<cores>c-<objective>` — `axion-16c-ttfa`, `graviton4-32c-throughput`. The
platform is what a reader would recognise, the core count is the shape the topology was
qualified at, and the objective is what the profile optimises, because the same machine has
a different answer for first-audio latency than for throughput.

### The `recommended` alias

`recommended.json` carries a pointer and **no values of its own** — a second copy is a copy
that drifts. It resolves to whichever profile is currently the default recommendation, and
changing that is editing one field:

```json
{ "profile": { "id": "recommended", "alias_of": "<the qualified profile>" } }
```

It must point at a profile whose status is `qualified`. Pointing it at a starting point would
make an unmeasured configuration the default answer to "what should I run", which is the
failure this whole directory exists to prevent.
