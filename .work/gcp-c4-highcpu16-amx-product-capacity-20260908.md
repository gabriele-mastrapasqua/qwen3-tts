# GCP C4 highcpu-16 AMX product capacity — 2026-09-08

## Identity and policy

Host: Xeon Platinum 8581C / Emerald Rapids, 8 physical cores `0-7`, SMT off,
one NUMA node, performance governor, measured full-host DRAM read roof about
110 GB/s. Binary SHA-256:
`06b8cba62a2c5de6e2aa9c3cec6b45ceeb8513cf65e5220c97d7e44e5e16051`.
Binary source tag: `a3e9ddd`; strict AMX product preflight passed with
`ragged-design-d-int8`, fused residual, warm strip, AMX Talker/CP/prefill/Q4,
and known-text `QWEN_TTS_STREAM_LAYOUT=1`.

The comparison is playback-aware. `STREAM_RTF < 1` alone is not sufficient;
the full envelope includes TTFA, safe play start, prebuffer, fixed-buffer stalls,
overload behavior and sustained drift. Cost/hour and streams per dollar are
UNKNOWN here because no grounded price was attached to this artifact.

## Final capacity table

| model | final topology | full-envelope GOOD | short-bank max GOOD | first NOT GOOD | TTFA p95 | STREAM p95 | prebuffer p95 | safe-start p95 | stall@250 | stall@500 | observed req/s / open-arrival | cost/h | GOOD streams/$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1.7B | `1x8`, cap2 | C2 | C3 screen-only | C3 full envelope | 462 ms C2 mixed | .692 C2 mixed | 86 ms C2 mixed | 513 ms C2 mixed | 0% | 0% | closed-loop; Poisson .03–.10 with prompt rejects at higher overlap | UNKNOWN | UNKNOWN |
| 0.6B | `1x8`, cap3 | C3 | C4 screen-only | C4 full envelope / C5 short | 185 ms C3 SOAK worst window | .861 C3 SOAK worst window | 377 ms C3 SOAK worst window | 471 ms C3 SOAK worst window | 2% pooled C3 SOAK | 0% | Poisson .05/.10 no rejects in short samples; .15+ is overload/reject transition | UNKNOWN | UNKNOWN |

The table deliberately separates full-envelope capacity from short-bank
headroom. The 0.6B C4 screen was below the exploratory .95 STREAM boundary in
the measured classes but long/medium C4 tails were around .943 and therefore
not promoted as a full-envelope point. C5/C6 fail the short-bank realtime and
stall envelope.

## 1.7B interpretation

SL-1 reduced known-text prefill positions and materially removed the long-input
startup/admission term: C1 long TTFA p95 fell from about 329 ms to 65 ms, and
C3 long TTFA p95 from about 855 ms to 158 ms. The remaining full-envelope C3
failure is not a reason to reopen AMX kernels or global batching: the C3 SOAK
tail crossed one in two windows and drifted. Use C2 for production comparison;
retain C3 only as a clearly labelled development/screen point.

## 0.6B interpretation

0.6B has enough headroom for full C3 operation on this 8-core host. C3 class
waves were healthy (short STREAM p95 about .658, long about .691), the corrected
five-minute SOAK passed with worst-window STREAM p95 .861, safe-play-start p95
471 ms, no @500 stall and only a small pooled @250 tail. A 2+1 long-arrival
probe accepted the third request in all repetitions; the third request stayed
near .67 STREAM_RTF and established-stream post-injection max gaps were
247–288 ms with no @250/@500 stalls. C3 `3+1` fail-fast rejected the fourth
request cleanly.

C4 is useful as a screen but not a full-envelope product point: medium/long
tails reached roughly .943 STREAM_RTF p95. C5 was the first clearly bad short
point, with STREAM p95 about 1.077 and @250 stalls in one third of requests.

The dominant 0.6B limit is now realtime cadence/overlap margin at C4+, not
long-prefill startup. This is materially different from the 1.7B C3 case.

## Open-arrival characterization

The 0.6B low-rate probes had 6/6 accepted at .05 req/s and 7/7 at .10 req/s,
with accepted STREAM p95 .579/.589 respectively. The original .15/.30/.45
probes entered the fail-fast region (accepted/rejected 9/6, 9/7, 7/9) while
accepted STREAM p95 remained .853/.899/.902. These are small transition probes,
not a sustained Poisson capacity rate; intentional 503s were separated from
internal errors and no internal failures were observed.

## Audio/quality and transport

The strict profile smoke produced valid non-empty mono 24 kHz 16-bit PCM WAVs;
local samples include short and long outputs with non-zero signal and plausible
durations. The server-generation quality contract passed: profile/dispatch and
precision were exact, output/container structure was valid, and no unexpected
fallback occurred. Legacy CLI/librosa goldens are not used as a server oracle.

The existing bounded slow-client test passed for the same serving generation;
the synchronous writer remains the reference. The async bounded writer remains
available but was not needed to rescue this 0.6B point.

## Recommended deployment points

* 1.7B: **C2, cap2**, with C3 reserved for development or a bounded input
  envelope only.
* 0.6B: **C3, cap3**, with C2 as the conservative fallback and C4 treated as
  a non-promoted screen.

Do not claim cost/GOOD-stream efficiency until hourly pricing is supplied and
the accepted-rate operating point is run long enough for a comparable economic
denominator.

Next cross-ISA target: **AMD/Turin VNNI first**, to test the current server
contract without AMX-only decoder assumptions; then Axion/Neoverse-V2 after the
VNNI product/common-control lanes are exercised. No further AMX optimization is
needed before that comparison.
