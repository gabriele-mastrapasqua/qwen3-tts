# An AWS `m8i-flex.4xlarge` — does the M family provision more bandwidth than C?

One question, one measurement, one answer.

The `m8i` family gives 4 GiB per vCPU where `c8i` gives 2. The question worth asking before
paying for it is whether that extra memory arrives as **more memory channels for the slice** —
which this workload would care about, being memory-bound — or merely as **more capacity**, which
it would not, since the 1.7B model at int8 fits comfortably in 30 GiB.

**The answer is capacity. `m8i-flex` measured 4.6% *less* bandwidth than `c8i-flex` and costs 13%
more.**

Sibling pages: [`reference-aws-c8i-8c-amx.md`](reference-aws-c8i-8c-amx.md) (the full campaign)
and [`reference-aws-c8i-flex-8c.md`](reference-aws-c8i-flex-8c.md). This page is a bandwidth and
identity screen only — no `--caps`, no `--self-test`, no kernel cells, no model, no serving
numbers. The instance was released after it.

## Identity

Every silicon field is identical to both `c8i` instances. Memory is the only hardware difference.

| | measured value |
|---|---|
| instance | AWS **`m8i-flex.4xlarge`**, `us-east-1a` |
| CPU | Intel Xeon 6975P-C (Granite Rapids), 1 socket |
| cores | **8 physical**, 16 vCPU as delivered; SMT disabled at runtime for the measurement |
| **memory** | **61 GiB** (vs 30.8 GiB on both `c8i` boxes) — the only difference |
| NUMA | 1 node |
| ISA | AVX-512 VNNI / BF16 / FP16, **`amx_tile` + `amx_int8` + `amx_bf16`** |
| L3 as reported | 480 MiB, one instance — a socket figure, not this slice |
| gates | SMT off **PASS** · no cgroup quota **PASS** · frequency stable **PASS** |

Measured with `tools/box_info.sh --membw`, three 512 MiB `double` buffers, best of five, idle
box, `snapd` and `unattended-upgrades` stopped — the same recipe as both sibling pages.

## The answer

| threads | `c8i.4xlarge` | `c8i-flex.4xlarge` | **`m8i-flex.4xlarge`** |
|---:|---:|---:|---:|
| 1 | 20.1 | 18.2 | **18.5** |
| 2 | 37.9 | 35.6 | **34.5** |
| 4 | 69.9 | 66.3 | **64.7** |
| 8 | **104.8** | 99.2 | **94.6** |
| peak / knee | 104.8 @8 | 99.2 @8 | **94.6 GB/s @8, knee 8** |

`m8i-flex` is **−4.6% against `c8i-flex`** and **−9.7% against `c8i`**. The curve shape and the
knee are identical: eight threads, which is every core the instance has.

So the M provisioning does not buy memory channels. Doubling GiB per vCPU is a capacity
decision, and on a workload whose footprint already fits, capacity is dead weight.

## What it costs

On-demand, Linux, shared tenancy, list price, captured **2026-09-03**.

| | `us-east-1` | `eu-west-1` | peak Triad | $ per GB/s·hr |
|---|---:|---:|---:|---:|
| `c8i-flex.4xlarge` | $0.712 | $0.764 | 99.2 | **$0.0072** |
| `c8i.4xlarge` | $0.7497 | ~$0.805 *(derived)* | 104.8 | **$0.0072** |
| `c8a.4xlarge` | $0.862 | $0.925 | 106.5 | $0.0081 |
| **`m8i-flex.4xlarge`** | **$0.8044** | **$0.8966** | 94.6 | **$0.0085** |

**`m8i-flex` is the worst value in the set on the axis this workload cares about**: 13% more
expensive than `c8i-flex` for 4.6% less bandwidth, and the most expensive per GB/s of any
candidate measured. The 30 extra GiB are real, and irrelevant here.

It would become the right choice only if something in the deployment actually needed the memory
— many resident models in one process, a large prefix cache, or heavy per-request buffers. None
of those is the current shape.

## A side effect: the flex throttling flag got weaker

The `c8i-flex` page raised, without resolving, whether flex burst credits make its roof
unsustainable: four back-to-back 8-thread runs there produced 96.1, 98.6, 97.8 and **84.0**
GB/s. The same series on this second flex instance:

| run | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Triad GB/s | 92.4 | 93.4 | 92.9 | 90.0 |

A 3.8% spread against the other box's 17%. **The drop did not reproduce on a second flex
machine**, which makes the earlier 84.0 look like a one-off rather than a family behaviour. That
is evidence against the hypothesis, not proof: two boxes, four samples each, a few minutes of
observation, and no long soak on either. It remains the right thing to check on any flex
instance before committing to it, but it is no longer a reason to avoid the family.

## What was not run

No engine build, no ISA gates, no kernel cells, no model, no serving numbers. If this family is
ever revisited, the *Reproducing* section of
[`reference-aws-c8i-8c-amx.md`](reference-aws-c8i-8c-amx.md) is the script.
