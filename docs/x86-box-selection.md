# Choosing an x86 box: what a probe decides, and what it cannot

Renting a machine to serve this engine is not a question of picking the biggest number on the
instance page. This note records the method used to screen AWS candidates, the measured
candidate table, and — the part that keeps being the expensive lesson — the boundary between
what a one-hour silicon probe settles and what only a serving campaign settles.

The per-box pages are [`reference-x86-8c-amx.md`](reference-x86-8c-amx.md) and
[`reference-scaleway-16c-vnni.md`](reference-scaleway-16c-vnni.md) (both qualified),
[`reference-aws-c8i-8c-amx.md`](reference-aws-c8i-8c-amx.md) (screened, not qualified) and
[`reference-aws-c8i-flex-8c.md`](reference-aws-c8i-flex-8c.md) and
[`reference-aws-m8i-flex-8c.md`](reference-aws-m8i-flex-8c.md) and
[`reference-aws-r8a-16c.md`](reference-aws-r8a-16c.md) (bandwidth screens only). The general
rental playbook is [`hardware-testing.md`](hardware-testing.md).

## The two axes that actually decide it

This workload is not one thing. It has a memory-bound half — the Code Predictor re-reads its
weights sixteen times per frame — and a compute-bound half that a matrix unit can shorten. A box
is therefore chosen on two independent axes, and a candidate that wins one can lose the decision:

| axis | what it prices | the number to compare |
|---|---|---|
| **bandwidth per stream** | how many realtime streams the box can hold | measured Triad at the thread count one request will actually get, **and the thread count at which the curve flattens** |
| **matrix unit** | time to first audio, and throughput once requests batch | does AMX exist, does the dispatcher reach it, at which `B` |

The peak Triad number alone answers neither. Two boxes with the same roof behave completely
differently if one reaches it with a quarter of its cores and the other needs all of them.

## The candidate table

Everything below was measured with the same two tools — `tools/box_info.sh` and
`tests/membw.c`, three 512 MiB buffers, best of five — except the `c8a` row, whose provenance is
weaker; see the caveat under the table.

| | **`c8a.4xlarge`** | `c8i.4xlarge` | `c8i-flex.4xlarge` | `m8i-flex.4xlarge` | `r8a.4xlarge` | Scaleway 16c | Arm 16c ref |
|---|---|---|---|---|---|---|---|
| CPU | **EPYC 9R45 (Zen 5)** | Xeon 6975P-C (GNR) | Xeon 6975P-C | Xeon 6975P-C | EPYC 9R45 | EPYC 9555P | Neoverse-V2 |
| **physical cores** | **16** | 8 | 8 | 8 | 16 | 16 | 16 |
| SMT | **not supported** | 2/core → off | 2/core → off | 2/core → off | not supported | off | absent |
| RAM | 30.8 GiB | 30.8 GiB | 30.8 GiB | 61 GiB | 123 GiB | 16 GiB | — |
| **AMX** | **❌ VNNI only** | ✅ | ✅ | ✅ | ❌ | ❌ | n/a (SMMLA/BFMMLA) |
| LLC per core | 4.0 MiB *(real)* | 60 MiB *(socket, false)* | idem | idem | 4.0 MiB *(real)* | *(socket)* | 5.0 MiB |
| Triad @1 thread | **44.1** | 20.1 | 18.2 | 18.5 | 44.4 | — | — |
| Triad @4 | **100.6** | 69.9 | 66.3 | 64.7 | 103.5 | — | — |
| Triad peak | 103.6 | 104.8 | 99.2 | 94.6 | 106.3 | ~88 | **336** |
| **knee** | **4 threads** | 8 | 8 | 8 | 4 | — | 16 |
| best topology | **`2x8`** | `2x4` | — | — | — | `2x8` | `2x8` |
| **realtime streams** | **4** (RTF 0.952) | 2 (0.832) | — | — | — | 2 | 4 (0.72) |
| TTFA p50 @C=4 | **186 ms** | 235 ms | — | — | — | 185 ms | 124 ms |
| req/s @C=8 | **3.15** | 1.92 | — | — | — | 1.55¹ | — |
| single-stream | **72 ms · 0.523** | 90 ms · 0.571 | — | — | — | 70 ms · 0.51 | 46 ms · 0.35 |
| $/hr `us-east-1` | $0.862 | $0.7497 | $0.712 | $0.8044 | $1.278 | n/a | n/a |
| $/hr `eu-west-1` | $0.925 | ~$0.81–0.84 | $0.764 | $0.8966 | $1.430 | n/a | n/a |
| $ per core·hr | **$0.0539** | $0.0937 | $0.0890 | $0.1006 | $0.0799 | — | — |
| $ per GB/s·hr | $0.0083 | $0.0072 | $0.0072 | $0.0085 | $0.0120 | — | — |
| **$ per realtime stream·hr** | **$0.216** | $0.375 | — | — | — | — | — |
| campaign | **screened, full** | screened, full | bandwidth only | bandwidth only | bandwidth only | qualified | qualified |

¹ Scaleway's C=8 row is stream RTF 1.55; its req/s was not recorded in the same form.

**The verdict, on measured numbers: `c8a.4xlarge`.** It wins every serving cell against the AMX
box on the same binary, holds twice the realtime streams, costs 42% less per stream and 42% less
per core, and has exactly one dispatcher to optimise. Its structural twin is the Arm reference —
same cores, same absent SMT, same `2x8` — so an Axion-shaped workflow ports without new
concepts, at roughly 30% worse latency and RTF because Arm has 3.2× the bandwidth.

**Caveat on the prices.** On-demand, Linux, shared tenancy, list price, captured **2026-09-03**;
no Savings Plan, Reserved or Spot discount, and no data-transfer or storage cost. The
`c8i.4xlarge` `eu-west-1` figure is *derived*: it was not fetched, but `c8a` and `c8i-flex` each
carry the same 7.3% EU-over-US uplift, so $0.7497 was scaled by it. Prices move — re-check
before anyone spends money on this table.

**Caveat on the `c8a` row — largely resolved.** Those figures came from an earlier probe that
recorded neither affinity, nor idle/runqueue state, nor the toolchain. They are now corroborated:
[`r8a.4xlarge`](reference-aws-r8a-16c.md), the same EPYC 9R45 with the same 16 cores, measured
44.4 / 86.5 / 103.5 / 105.1 / 106.3 with full provenance — within 1% of the `c8a` row at every
thread count, same peak, same knee at 4. The bandwidth column can be trusted; what `c8a` still
lacks is any serving measurement at all.

**Caveat on comparing bandwidth tables at all.** `box_info.sh` sizes the membw buffers from the
*reported* L3, so it used 512 MiB on the Intel boxes (480 MiB reported) and 256 MiB on `r8a`
(64 MiB reported). Smaller buffers read about 3% high. Every cross-machine number in this note
is the 512 MiB one; a per-box page may show both.

## What the probe settled

- **`c8i.4xlarge` is an 8-core machine.** It ships 16 vCPU as 8 cores × 2 threads. Comparing it
  to a 16-vCPU AMD instance on vCPU count compares 8 cores against 16. The hyperthreads can be
  taken offline at runtime (`echo off > /sys/devices/system/cpu/smt/control`) without recreating
  the instance, because the sibling map is the clean `N ↔ N+8` interleave.
- **AMX is present and the engine reaches it.** `SIMD=auto` selects `amx`, `--caps` reports both
  tile paths ACTIVE, the dispatcher switches to tiles at exactly `B=4`, and `--self-test` passes
  natively and in fallback with the batched twins bit-exact.
- **The two Zen 5 candidates and the two Intel candidates are different machines, not different
  sizes of one.** `c8a` reaches its bandwidth roof with 4 of its 16 cores and has more than
  double the per-core bandwidth; `c8i` needs all 8 of its cores to reach a roof of the same
  height, and pays for that with a matrix unit the AMD parts do not have.
- **`c8i` is the existing AMX reference box plus about 28% of bandwidth**, at the same core
  count, same NUMA shape, same matrix unit — so the already-measured profile and result table
  applied directly, and the comparison below is a controlled experiment rather than an analogy.

## What the `c8i` campaign then settled

The full tables are in [`reference-aws-c8i-8c-amx.md`](reference-aws-c8i-8c-amx.md); the
decision-relevant part is short.

- **+28% of bandwidth bought 9–20% of stream RTF and 7–20% of throughput, and 2–7% of TTFA.**
  Same 8 cores, same AMX, same profile, same bank: the only substantial variable was the memory
  roof, and it moved the sustained half of the workload and left first audio alone. First audio
  is prefill and prefill is the AMX BF16 path, which both machines already had.
- **It crossed a threshold, not just a percentage.** At `2x4`, C=2 is stream RTF **0.832**
  against the reference's 1.027: this class of box now holds **two** concurrent realtime
  streams. TTFA p95 stays under 500 ms out to C=8 (454 ms), where the reference had to stop its
  claim at C=6.
- **AMX INT8 was worth nothing here, and AMX BF16 was worth more.** Removing BF16 costs 29% of
  C=1 TTFA and 68% of C=4 p95; removing INT8 is indistinguishable from the control on every
  column. The batch per worker is about 1.5 at C=4, so most decode calls are B=1 GEMV and never
  reach a tile — which also makes the profile's `QWEN_AMX_MIN_B=2` pin worthless on this host.
  **A lever is worth what the dispatcher lets it be worth**, and that is host-specific.

## What the `c8i-flex` screen settled

Same silicon, same 8 cores, same AMX, same knee. **The flex discount is the same size as what it
takes away**: −5.0% on price against a −5.3% Triad roof and −9% per-core bandwidth. On a
memory-bound workload that is a wash before risk, and the risk is real but unquantified — four
back-to-back sustained runs produced 96.1, 98.6, 97.8 and **84.0** GB/s, which flags burst
behaviour without proving it. Flex is for a low duty cycle; a request here keeps all eight cores
busy for its whole duration. Details in
[`reference-aws-c8i-flex-8c.md`](reference-aws-c8i-flex-8c.md).

## Memory-optimized families: settled on both vendors, and the answer is no

| question | measurement | verdict |
|---|---|---|
| Intel: does `m8i` provision more bandwidth than `c8i`? | 94.6 vs 99.2 GB/s, +13% price | **no** — −4.6% |
| AMD: does `r8a` provision more bandwidth than `c8a`? | 106.3 vs 106.5 GB/s, +48% price | **no** — identical within 1% |

More GiB per vCPU is a **capacity** product. It does not come with more memory channels for the
slice, and this workload's footprint already fits in the compute-optimized variant. `r8a` at
$0.0120 per GB/s·hr and `m8i-flex` at $0.0085 are the two worst values in the set, against
$0.0072 for both `c8i` variants. Neither is a candidate unless the deployment genuinely needs
the RAM — several resident models, a large prefix cache, heavy per-request buffers.

## What the `m8i-flex` screen settled: C vs M provisioning

`m8i` gives 4 GiB per vCPU where `c8i` gives 2. On identical silicon (same Xeon 6975P-C, same 8
cores, same AMX), `m8i-flex.4xlarge` measured **94.6 GB/s** against `c8i-flex`'s 99.2 and
`c8i`'s 104.8 — **−4.6% and −9.7%**, same curve shape, same knee at 8 threads. It also costs 13%
more than `c8i-flex`.

**The M family buys capacity, not memory channels.** For a workload whose footprint already fits
in 30 GiB, that is the most expensive per-GB/s option in this set ($0.0085 against $0.0072) and
the extra 30 GiB are dead weight. M would only become right if the deployment genuinely needed
the memory — several resident models in one process, a large prefix cache, heavy per-request
buffers. Details in [`reference-aws-m8i-flex-8c.md`](reference-aws-m8i-flex-8c.md).

## What is still not settled, and must not be assumed

- **How `c8a` serves.** No model was ever placed on it: no TTFA, no stream RTF, no topology
  choice, no census. Its case rests entirely on 16 cores and a knee at 4 threads, which is an
  argument, not a measurement. A kernel cell is not a request — the Scaleway census attributes
  only about a third of the counted work to batched INT8 matrix-matrix, with about 36% in GEMV
  and 19% single-slot — and the `c8i` campaign is the proof: its int8 tile is 1.3–1.6× faster in
  the kernel and exactly 0% faster in the request.
- **Whether `c8i` survives a real workload.** The campaign is three waves per cell on one short
  bank: no mixed-length bank, no open-loop arrivals, no soak, no per-class percentiles. The
  C=2 realtime result deserves a soak before anything is deployed on it.
- **Whether the working set fits the LLC — on Intel.** `lscpu` reports 480 MiB of L3 on `c8i`,
  256 MiB on the Scaleway host, 260 MiB on the Emerald Rapids box. Those are **socket** figures
  visible to a guest that owns a fraction of the socket, and `box_info.sh` turns them into a
  `llc_fits_*` verdict that is unverified on a slice.
  **On AMD the same field is trustworthy**, because L3 is exposed per CCD: `r8a` reports 32 MiB
  × 2 with the shared-CPU lists to match, and reaches the opposite and credible conclusion —
  the 1.7B CP working set does **not** fit. So the fix is not "distrust the field" but "know
  which vendor's report is a slice and which is a socket". The general remedy is still an
  array-size sweep to find the effective LLC, which `tests/membw.c` does not yet do.
- **Price per served stream, for every candidate but one.** `c8i` can be priced — 2 realtime
  streams at $0.7497/hr is about **$0.375 per realtime stream per hour** — because it has both a
  price and a streams-below-RTF-1.0 figure. Nothing else in the table has both. `$ per GB/s·hr`
  is in the table as a crude stand-in, and it is genuinely crude: it prices bandwidth as if that
  were the whole machine, which the AMX attribution in the `c8i` page shows it is not.
  **The reference box's instance type and price were never recorded, so its cost per stream is
  unrecoverable** — record instance type and on-demand price on every future box page.

## What to do next, in order

1. ~~Run the existing AMX profile on `c8i`.~~ **Done** — §4–§6 of the `c8i` page.
2. ~~Put a model on `c8a` and run the same `bench-topo`.~~ **Done** — and it overturned the
   prediction. The arithmetic said C=4 would land at ~1.04, above the realtime line; it measured
   **0.952**. That is the whole reason the rule against transferring results between hosts
   exists.
3. **Write a `c8a`-specific profile.** The sweep borrowed `scaleway-16c-vnni-ttfa` to get
   comparable numbers. Its pins were measured on a different machine and must be re-derived
   here: `QWEN_POOL_SPIN`, `QWEN_DECODER_BATCH`, `QWEN_STREAM_DECODE_CHUNK`, and the CCD
   affinity question below.
4. **Soak the C=4 result.** It is 0.952 with a worst case of 0.999 — a threshold result with no
   margin. `make bench-suite BENCH_TOPO=2x8` is what tells you whether it survives a realistic
   bank, open-loop arrivals and ten minutes of running.
5. **Test the CCD alignment.** The two L3 domains are exactly `cpu 0-7` and `cpu 8-15`, and
   `2x8` won. Whether the alignment is the *cause* needs an explicit affinity A/B — pinned to
   CCD boundaries against deliberately straddling them.
6. **Fix batched q4 on VNNI.** `--matmat-bench` measures 0.96–0.97× — batching q4 is a net loss
   on this box, reproducing the standing follow-up on a third machine. The AMX box reaches 4.2×
   on the same cells, so the deficit is in `q4_matmat_vnni_slice`, not in the idea.

## Rules this note exists to enforce

- **vCPU is not a core.** Check `Thread(s) per core` before comparing two instances, and disable
  SMT before measuring anything, or declare that you did not.
- **Compare the knee, not just the peak.** Two boxes with equal roofs and different knees offer
  completely different amounts of compute slack for a given number of streams.
- **A number without provenance is a direction, not a datum.** Commit, diff hash, binary hash,
  toolchain, idle state and thread count, or the row is soft — and label it soft in the table
  rather than letting it age into a fact.
- **Never carry a result across hosts.** A Scaleway measurement is not an AWS A/B; a kernel cell
  is not a serving result; a bit-exact microbenchmark win is not a request that got faster. The
  `c8i` campaign cost nothing to run and overturned two inherited numbers — `QWEN_AMX_MIN_B=2`
  and the value of AMX INT8 — on a machine that differed from the reference only in bandwidth.
- **A profile is qualified for one host.** Borrowing one to get comparable numbers is right;
  shipping on a borrowed one is not. Re-derive its measured pins on the new box.
