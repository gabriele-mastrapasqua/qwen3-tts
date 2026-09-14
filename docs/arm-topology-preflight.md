# Arm topology preflight

Before running a serving wave or a sustained soak on a 32-core Arm box, run the
doctor first:

```bash
make doctor
```

On Arm Linux, the doctor runs a short `roof_matvec_int8` discriminator near the
top of its report. It measures the same cold Talker INT8 workload as the normal
roof, with fixed disjoint CPU masks, in three shapes:

| shape | execution |
|---|---|
| `1x8` | one 8-thread worker, isolated reference |
| `2x8` | two 8-thread workers started simultaneously |
| `4x8` | four 8-thread workers started simultaneously |

The report shows worker latency, per-worker and aggregate GB/s, aggregate scale
versus `1x8`, and the 4×8 per-worker slowdown. The verdict is specifically about
the tested **4×8 serving topology**, not about whether the entire CPU instance is
usable:

- `PASS`: aggregate scale is at least about 2× and per-worker slowdown is at most
  about 2×; 4×8 is a reasonable starting shape.
- `WARNING`: topology-sensitive; measure an alternate shape before qualification.
- `STRONG WARNING`: 4×8 is not recommended; try `2x16`, then `1x32`.
- `FAIL`: 4×8 aggregate throughput is no better than isolated `1x8`; do not use
  4×8 as the serving baseline on that box.

The raw parsed values and each worker's output are stored below the doctor
artifact directory as `arm_gemv_scaling.json` and `arm_gemv_*.txt`. The JSON
contains isolated GB/s, 2×8 and 4×8 aggregate GB/s, the scale ratio, per-worker
slowdown, masks, and verdict. A G5-like result is an explicit reason to test
`2x16`/`1x32` before spending time on engine or soak qualification; it is not a
claim that the host itself is unusable.

The preflight is intentionally a few short kernel runs, not a production
benchmark. It is skipped on non-Arm hosts, hosts without `taskset`, or boxes
with fewer than 32 online CPUs. Use `--no-arm-gemv-preflight` only when a
deliberate doctor run must avoid the discriminator; keep the output artifact and
record why it was skipped.

After the preflight, continue with the normal gates:

```bash
make cpu-check
./qwen_tts --caps
./qwen_tts --self-test
make bench-topo BENCH_MODEL=<dir> BENCH_TOPO=<candidate-shapes> BENCH_CONC=1,4
```

The doctor result is a topology screen, not a serving qualification. A shape
must still pass the resolved-dispatch checks, true-wave screening, playback
metrics, and the canonical closed-loop soak.
