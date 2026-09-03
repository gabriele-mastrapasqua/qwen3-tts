# An AWS `r8a.4xlarge` — the AMD side, and what a trustworthy L3 looks like

A light screen: identity, cache, bandwidth, price. No engine build, no ISA gates, no kernel
cells, no model, no serving numbers. The instance was released after this page.

It was rented to ask the AMD version of the question the
[`m8i-flex` page](reference-aws-m8i-flex-8c.md) asked of Intel: **does the memory-optimized
family provision more bandwidth than the compute-optimized one?**

**No. `r8a.4xlarge` is bandwidth-identical to `c8a.4xlarge` — within 1% at every thread count,
same peak, same knee — and costs 48% more. You are paying for the RAM and nothing else.** It
also turned out to settle something more useful about cache reporting.

## Identity

| | measured value |
|---|---|
| instance | AWS **`r8a.4xlarge`**, `us-east-1a` |
| CPU | **AMD EPYC 9R45** (Zen 5), 1 socket |
| cores | **16 physical**, SMT **not supported** — nothing to disable |
| memory | 123 GiB (8 GiB per vCPU) |
| NUMA | 1 node |
| ISA | AVX-512F, `avx512_vnni`, `avx512_bf16` — **no AMX** |
| L1d / L2 | 48 KiB · 1.0 MiB per core |
| **L3** | **32 MiB × 2 instances = 64 MiB**, 4.0 MiB per core |
| gates | SMT **PASS** · no cgroup quota **PASS** · frequency stable **PASS** |

## The L3 is real here, and it changes the verdict

Every Intel box in this set reported **480 MiB of L3 in one instance** — the whole Granite
Rapids socket, seen by a guest that owns eight of its cores. `tools/box_info.sh` propagated that
into "60 MiB per core → the 1.7B Code Predictor working set FITS", which is almost certainly
false and is flagged as untrustworthy on those pages.

AMD exposes L3 **per CCD**, so this guest sees its two CCDs:

```text
usable LLC   64 MiB  (L3, sum of 2 instances), 4.0 MiB per physical core
             32768K shared by cpu 0-7
             32768K shared by cpu 8-15
```

That is a number the instance plausibly owns, and the shared-CPU lists confirm the shape: two
independent 32 MiB domains, eight cores each. With it, `box_info.sh` reaches the **opposite**
conclusion from the Intel boxes, and this time it is credible:

| model | CP working set, int8 | verdict on this box |
|---|---:|---|
| 0.6B | ~60 MB | FITS, barely |
| 1.7B | ~120 MB | **does NOT fit** |

So the `llc_fits_*` field is usable on an AMD slice and not on an Intel one — a reporting
difference, not a hardware one. On Intel the honest answer remains "unknown until an array-size
sweep measures it".

Note also the topology consequence: the two L3 domains are `cpu 0-7` and `cpu 8-15`. A 2×8
server topology maps one worker per CCD exactly. That is a hypothesis worth an A/B, not a
result — nothing was served on this box.

## Bandwidth

Measured with the same recipe as every other page: `box_info.sh --membw`, best of five, idle
box. **Both array sizes are shown because they are not interchangeable.**

| threads | 256 MiB buffers (`box_info` default here, 4× the real 64 MiB L3) | **512 MiB buffers (the Intel protocol)** |
|---:|---:|---:|
| 1 | 44.8 | **44.4** |
| 2 | 88.3 | **86.5** |
| 4 | 105.7 | **103.5** |
| 8 | 109.0 | **105.1** |
| 16 | 109.5 | **106.3** |
| peak / knee | 109.5 @16, knee 4 | **106.3 @16, knee 4** |

The smaller buffers read about 3% high. `box_info.sh` sizes them from the *reported* L3, so on
this box it chose 256 MiB and on the Intel boxes 512 MiB — the reports are internally correct
but **not comparable to each other**. The right-hand column is the one to use in any
cross-machine table.

### Against the other AMD instance

The `c8a.4xlarge` figures from the earlier probe were 44.5 / 85.4 / 102.4 / 106.5 / 106.1, peak
106.55, knee 4. This box, on the same protocol: 44.4 / 86.5 / 103.5 / 105.1 / 106.3, peak 106.3,
knee 4. **Within 1% at every thread count.**

Two things follow. The R family gives the same bandwidth as the C family on identical silicon —
**8 GiB per vCPU is capacity, not memory channels**, exactly as `m8i-flex` showed on Intel. And
the previously soft `c8a` row is now corroborated by an independently measured box with full
provenance, which is most of what was wrong with it.

### Against Intel

| | `r8a` / `c8a` (AMD, 16c) | `c8i` (Intel, 8c) |
|---|---:|---:|
| Triad @1 thread | **44.4** | 20.1 |
| Triad @4 | **103.5** | 69.9 |
| peak | 106.3 | 104.8 |
| **knee** | **4 threads of 16** | 8 threads of 8 |
| AMX | ❌ | ✅ |

Same roof, completely different machine. AMD has **2.2× the per-core bandwidth** and reaches the
roof with a quarter of its cores, leaving twelve for compute; Intel needs every core it has and
compensates with a matrix unit. Which one wins is a serving question, and it is still open —
nothing has ever been served on an AMD box in this set.

### Sustained

Four back-to-back 8-thread runs: **108.9 / 108.6 / 108.5 / 107.4** GB/s — a 1.4% spread, the
tightest of any box measured here. Two 16-thread runs: 109.3 / 108.4.

This is the non-flex control the `c8i-flex` page wanted. It shows that a 17% spread is not
normal for a cloud instance under this measurement, which makes that box's outlier a little more
interesting again — but it is a different vendor and a different family, so it constrains the
question without answering it.

## What it costs

On-demand, Linux, shared tenancy, list price, captured **2026-09-03**.

| | `us-east-1` | `eu-west-1` | peak Triad | $ per GB/s·hr |
|---|---:|---:|---:|---:|
| `c8i-flex.4xlarge` | $0.712 | $0.764 | 99.2 | **$0.0072** |
| `c8i.4xlarge` | $0.7497 | ~$0.81–0.84 *(derived)* | 104.8 | **$0.0072** |
| `m8i-flex.4xlarge` | $0.8044 | $0.8966 | 94.6 | $0.0085 |
| `c8a.4xlarge` | $0.862 | $0.925 | 106.5 | $0.0081 |
| **`r8a.4xlarge`** | **$1.278** | **$1.430** | 106.3 | **$0.0120** |

**`r8a` costs 48% more than `c8a` for bandwidth identical within 1%**, and is the most expensive
option per GB/s in the whole set by a wide margin. The 96 extra GiB are real and, for a model
that fits in 32, irrelevant.

The pattern now holds on both vendors: **the memory-optimized family is a capacity product, and
this workload does not buy capacity.** `m8i-flex` was +13% for −4.6% bandwidth on Intel; `r8a` is
+48% for +0% on AMD. Neither is a rental candidate unless something in the deployment genuinely
needs the RAM.

## What was not run

No `--caps`, no `--self-test`, no kernel cells, no model, no serving numbers. If AMD is ever
taken seriously as a serving candidate, the missing experiment is a topology sweep on `c8a` —
the cheaper of the two — following the *Reproducing* section of
[`reference-aws-c8i-8c-amx.md`](reference-aws-c8i-8c-amx.md), with `2x8` as the starting
topology and the CCD boundary at `cpu 0-7` / `cpu 8-15` as the obvious affinity split to test.
