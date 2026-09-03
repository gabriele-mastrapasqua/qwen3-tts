# An AWS `c8i-flex.4xlarge` — the same silicon, sold differently

A short page for a short experiment. This instance was rented to answer one question: **does the
`flex` variant of a machine we have already measured behave like the machine we measured?**

The non-flex sibling is [`reference-aws-c8i-8c-amx.md`](reference-aws-c8i-8c-amx.md), which has
the full campaign — ISA gates, kernel cells, a fifteen-cell topology sweep and an AMX
attribution A/B. **This page has none of that.** It is a bandwidth and identity screen, taken in
a few minutes before the instance was released. Nothing here is a serving result and nothing
here is a qualification.

The answer, in one line: **same silicon, about 5% less bandwidth, and one unresolved question
about whether that 5% is even sustained.**

## Identity

Every hardware field is identical to the non-flex instance. The family is the only difference.

| | measured value |
|---|---|
| instance | AWS **`c8i-flex.4xlarge`**, `us-east-1a` |
| CPU | Intel Xeon 6975P-C (Granite Rapids, family 6 model 173 stepping 1), 1 socket |
| cores | **8 physical**, 16 vCPU as delivered; SMT disabled at runtime for the measurement |
| NUMA / memory | 1 node, 30.8 GiB, no swap |
| ISA | AVX-512F/BW/VL/DQ/CD, `avx512_vnni`, `avx512_bf16`, `avx512_fp16`, **`amx_tile` + `amx_int8` + `amx_bf16`** |
| L1d / L2 | 48 KiB · 2.0 MiB per core |
| L3 as reported | 480 MiB, one instance — a socket figure, not this slice; see the sibling page |
| governor | not exposed; the hypervisor owns the frequency |
| gates | SMT off **PASS** · no cgroup quota **PASS** · frequency stable **PASS** |

```text
kernel=7.0.0-1006-aws   microcode=0x1000434   hypervisor=amazon
idle before measuring: loadavg 0.18, 1 running, snapd and unattended-upgrades stopped
sibling map: cpu N <-> cpu N+8 over core_id 0..7  ->  `echo off > .../cpu/smt/control` is enough
probe=tools/box_info.sh --membw <tests/membw.c>   membw: 3 x 512 MiB double, best of 5
```

## Bandwidth, like for like

Both columns are `box_info.sh`-driven runs: three 512 MiB `double` buffers, best of five, SMT
off, idle box.

| threads | `c8i.4xlarge` | **`c8i-flex.4xlarge`** | delta |
|---:|---:|---:|---:|
| 1 | 20.1 | **18.2** | −9% |
| 2 | 37.9 | **35.6** | −6% |
| 4 | 69.9 | **66.3** | −5% |
| 8 | 104.8 | **99.2** | −5% |
| peak / knee | 104.8 GB/s @8 | **99.2 GB/s @8** | −5% |

Same curve shape, same knee at eight threads — which is every core the instance has, so there is
no configuration where some cores saturate DRAM while others compute. The roof is 5% lower and
per-core bandwidth is 9% lower.

Given what the sibling page measured — that +28% of bandwidth bought 9–20% of stream RTF and
2–7% of TTFA — a −5% roof should be expected to cost a small single-digit percentage of
sustained RTF and almost nothing of first audio. **That is an extrapolation from one controlled
experiment, not a measurement on this box**, and it is exactly the kind of transfer this project
does not promote to a claim.

## The measurement trap, stated once

Running `./membw` without `--l3-mb` makes it assume a 32 MiB L3 and allocate 128 MiB buffers
instead of 512 MiB. Part of the working set then stays resident and the number inflates:

| run | 8 threads | 16 threads (SMT on) |
|---|---:|---:|
| 128 MiB buffers (the bare default) | 106.2 GB/s | 135.4 GB/s |
| 512 MiB buffers (what `box_info.sh` passes) | 99.2 GB/s | — |

**135 GB/s is not this machine's memory bandwidth.** Only numbers taken through
`tools/box_info.sh`, which passes the reported L3 size, are comparable across the boxes in this
set. The SMT-on cell at 512 MiB was not measured here.

## The open question: is 99 GB/s sustained?

`flex` instances are sold with baseline CPU performance and the ability to burst, so a short
best-of-five run is precisely the shape of measurement that could sit inside a burst window.
Four back-to-back 8-thread runs at 512 MiB:

| run | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Triad GB/s | 96.1 | 98.6 | 97.8 | **84.0** |

The fourth is 14% below the first three. That is **a flag, not a verdict**: one outlier in four
samples is not distinguishable from noise, and the equivalent repeated series was never run on
the non-flex instance, so there is no control. The honest statement is that this box's
sustainability was not established either way.

The discriminator, if it ever matters, is cheap and symmetric: run the same repeated series on
both families for several minutes and compare the distributions, not the best-of-N. A best-of-5
is designed to report a ceiling, which is the wrong statistic for a machine whose selling point
is that its ceiling is conditional.

**Update, same day.** The identical series on a second flex instance
([`m8i-flex.4xlarge`](reference-aws-m8i-flex-8c.md)) produced 92.4 / 93.4 / 92.9 / 90.0 GB/s — a
3.8% spread against the 17% seen here. **The drop did not reproduce**, which makes the 84.0
above look like a one-off rather than flex behaviour. Evidence against the hypothesis, not proof
either way: still four samples per box and no soak. Worth checking on any flex instance; no
longer a reason to avoid the family.

## What it costs, and why that settles it

On-demand, Linux, shared tenancy, captured **2026-09-03**. List prices, no Savings Plan,
Reserved or Spot discount.

| region | `c8i-flex.4xlarge` | `c8i.4xlarge` (non-flex) | flex discount |
|---|---:|---:|---:|
| `us-east-1` | **$0.712 /hr** | $0.7497 /hr | **−5.0%** |
| `eu-west-1` | **$0.764 /hr** | ~$0.805 /hr *(derived)* | −5.1% |
| `eu-central-1` | not captured | not captured | — |

The `c8i.4xlarge` EU figure was not fetched directly; it is $0.7497 scaled by the 7.3% EU uplift
that both other instance types in this set carry. An inference, not a quote.

Now put the two deltas side by side:

| | flex vs non-flex |
|---|---:|
| price | **−5.0%** |
| measured Triad roof | **−5.3%** |
| per-core bandwidth | −9% |

**The discount is the same size as what it takes away — very slightly smaller.** On a workload
this memory-bound, `flex` is a wash before any risk is priced, and the risk is not zero: the
sustained-bandwidth question above is unresolved, and per-core bandwidth (which prices a single
stream) is down 9% against a 5% discount.

There is one shape where flex still wins: a low duty cycle. If the box is idle most of the time
and serves bursts, burst credits are exactly what is being bought and the baseline never binds.
That is not this workload — a request keeps all eight cores busy for its whole duration — but it
may be the shape of a dev or CI box.

## Verdict

Same silicon, same AMX, same core count, same topology, 5% less roof, and an unquantified
sustainability risk on a workload that keeps all eight cores busy for the length of every
request. `flex` is worth taking only if its price advantage clearly exceeds that 5% **and** the
duty cycle is low enough that burst credits are not the thing being bought. For sustained
serving, prefer the non-flex instance, which is also the one that has a measured campaign behind
it.

## What was not run

No `--caps`, no `--self-test`, no kernel cells, no model, no serving numbers. The engine was
never built on this host. If this instance is revisited, the sibling page's *Reproducing*
section is the script to follow, and the first thing worth adding is the sustained-bandwidth
series above, run on both families.
