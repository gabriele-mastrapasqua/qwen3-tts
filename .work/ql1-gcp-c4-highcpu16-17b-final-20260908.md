# 8-core AMX final 1.7B capacity decision — 2026-09-08

## Scope and identity

This is the final known-text 1.7B decision on the GCP `c4-highcpu-16` AMX
host. It does not reopen topology, quantum, kernel, batching or LS-4 work.

| item | value |
|---|---|
| host | GCP c4-highcpu-16, Xeon Platinum 8581C / Emerald Rapids |
| CPU state | 8 physical CPUs `0-7`, SMT off, one NUMA node, performance governor |
| topology | `1x8@0-7` |
| model | Qwen3-TTS 1.7B INT8, Ryan/English, seed 42, temperature 0 |
| binary/source | AMX binary SHA-256 `06b8cba62a2c5de6e2aa9c3cec6b45ceeb8513cf65e5220c97d7e44e5e16051`, build `a3e9ddd`, source fingerprint `a3e9ddd:clean` |
| profile | `amx-product`, profile update commit `26c20b6` |
| runtime | q4, Design-D INT8, fused residual, warm strip, ragged threshold 2, engine pool, prefix cache, synchronous output, `--max-queue 0` |
| admission point | cap 2 for the final conservative product point |
| strict preflight | PASS: `ragged-design-d-int8`, AMX Talker/CP/prefill/Q4, fused and Design-D ACTIVE, `QWEN_TTS_STREAM_LAYOUT=1` |

The raw KPI artifacts are on the host under `profiles/pf2-sl1-*`; the selected
audio samples were copied to `/private/tmp/pf2-sl1-06b-audio` for local
inspection. The wave header's generic `int8 dot` label is a caps-level leaf,
not the resolved matrix execution claim; strict profile preflight is the source
of truth for AMX/Design-D.

## Long-prefill root cause

The old tail was caused by full Talker prefill being executed inline during
admission, before the slot was installed. The scheduler could not select
established streams between the start and return of that call, while the
prefill also consumed the same engine-owned compute budget. Prefix cache reuse
does not provide a resumable variable-text cursor.

The rejected prefill helper still computed one complete prefill and added
multi-second TTFA tails; it was not cooperative prefill. A real cooperative
implementation would need request-owned token/layer/KV continuation state and a
safe continuation boundary. It was not implemented in this qualification.

## SL-1 causal A/B

`QWEN_TTS_STREAM_LAYOUT=1` is the official known-text dual-track layout: the
initial prompt ends after the control prefix, first text token and codec BOS;
the remaining known text is request-owned trailing state. It is not live text,
append-after-generation, ICL/clone resumability or a prefill pause/resume.

| workload | full-layout control | SL-1 treatment |
|---|---:|---:|
| C1 long TTFA p95 | 329 ms | 65 ms |
| C1 long prefill | 244–292 ms | 30–37 ms |
| C3 long TTFA p95 | 855 ms | 158 ms |
| accepted 2+1 injection max-gap | 513–538 ms | 245–289 ms |
| short/medium C3 regression | baseline | no material regression; fixed-buffer stalls remained zero |

The known-text treatment materially flattened startup and reduced accepted
long-arrival disturbance. The profile now makes this lane explicit; ICL/clone
and live incremental text remain outside it.

## Final C2 class results

Values are p50/p95 unless noted. All cells had zero errors, timeouts and
unexpected rejects; fixed-buffer stall rates at 100/250/500/1000 ms were zero
in these class waves.

| class | TTFA ms | STREAM_RTF | prebuffer ms | safe start ms | max gap p95 |
|---|---:|---:|---:|---:|---:|
| short | 111/115 | .626/.694 | 40/70 | 151/182 | 241 |
| medium | 112/122 | .630/.677 | 36/104 | 148/226 | 236 |
| long | 112/126 | .658/.671 | 39/64 | 157/176 | 277 |
| mixed | 182/462 | .625/.692 | 48/86 | 247/513 | 243 |

The mixed wave is still within the C2 continuity envelope; its p95 TTFA is the
largest class tail but remains below the 700 ms fallback ceiling.

## C2 sustained and overload evidence

The corrected five-minute C2 SOAK used the fixed analyzer, 72 completed
requests / 64 KPI samples, zero errors, zero rejects and zero timeouts. Across
the five windows, the worst window values were:

| metric | worst observed window p95 |
|---|---:|
| TTFA | 155 ms |
| STREAM_RTF | .816 |
| required prebuffer | 258 ms |
| safe play start | 365 ms |
| max gap | 345 ms |
| stall @100 / @250 / @500 / @1000 | 10% / 0% / 0% / 0% |
| coalesced reads | 0% |

The accepted long-arrival `2+1` probe completed all 6 established and 3
injected requests. The injected request was accepted; local post-injection
gaps were approximately 303–318 ms and fixed stalls at 250/500 ms were zero.
The cap2 `2+1` overload arm rejected the third request immediately in all three
repetitions and preserved the established streams. This is the production
overload behavior; no temporary third-slot policy is enabled.

The bounded synchronous slow-reader test on this same serving generation had
no observed disturbance to unrelated normal readers. Async output remains an
independently implemented, default-off option rather than a required C2 change.

## C3 evidence and decision

SL-1 made isolated C3 short/medium/long and a mixed all-bank screen healthy:
the all-bank screen had TTFA p95 578 ms, STREAM_RTF p95 .805, safe-play-start
p95 698 ms and zero fixed stalls at 250/500/1000 ms. This is the **highest
short-bank screen point**, not the full production point.

The five-minute C3 SOAK remained unstable at the tail: window STREAM_RTF p95
was approximately `1.024 / .909 / .913 / 1.006 / .874`, with pooled p95 just
over one and drift across windows. Therefore C3 is not promoted for the full
1.7B envelope even though the steady C3 arithmetic is close.

Final 1.7B classification:

* highest full-envelope GOOD: **C2**;
* highest short/isolated-bank screen: **C3**;
* first NOT GOOD full-envelope point: **C3**;
* C4 is not pursued on this host.

## Open arrival and quality contract

The final C2 SL-1 Poisson probes were intentionally short and are transition
evidence, not a claim of a long-run accepted-rate SLO:

| offered rate | HTTP 200 / intentional 503 | accepted TTFA p50/p95 | accepted STREAM p50/p95 |
|---:|---:|---:|---:|
| .03 req/s | 5 / 0 | 95/521 ms | .519/.768 |
| .06 req/s | 5 / 1 | 83/363 ms | .623/.699 |
| .10 req/s | 6 / 1 | 67/609 ms | .607/.850 |

The one non-200 in each latter arm was an intentional fail-fast 503, not an
inference error. No internal error, timeout or crash occurred. Small sample
sizes and cold-start tails mean no precise sustainable open-arrival rate is
claimed.

The current-generation gate is structural/dispatch based: exact model, voice,
language, seed/settings, valid non-empty PCM/WAV, strict resolved AMX profile,
no fallback and existing same-generation fused/residual parity. The legacy
CLI/librosa golden is explicitly not a server QL-1 oracle.

## Final verdict

`1.7B 8-core production point = C2, 1x8, q4, SL-1 known-text, Design-D INT8,
fused residual, synchronous output, cap2, fail-fast.`

The 1.7B box is not a full-envelope C3 production point. The remaining C3
limit is steady-state tail margin under three concurrent streams, not the
original long-input prefill startup term after SL-1. No further 1.7B tuning is
authorized in this generation.
