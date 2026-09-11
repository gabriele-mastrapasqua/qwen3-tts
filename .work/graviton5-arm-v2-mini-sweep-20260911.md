# Graviton5 Arm v2 mini-sweep and flow audit

Status: diagnostic handoff, not a qualification. The box was a clean
`964cb26` build of the public tree; no commit or push was made on the box.
The follow-up code and measurements below were performed locally and copied to
the box only for build/test/benchmark; the box still has no commit or push.

## Host and topology screen

The host is a 32-vCPU Neoverse-V3 Graviton5 instance, one NUMA node, 48 MiB
L3, with no SMT. `make doctor`, `--caps`, `--dispatch-map` and
`--self-test` passed. The short synchronized screen used the 1.7B model, the
Ryan voice, English, the short text bank, INT8 Arm DOTPROD, and two waves.

The control topology screen selected 4x8. 8x4 was consistently narrower; 2x16
was slower at C8 and was not a valid C12/C14 comparison with cap4 because its
two workers could admit only eight requests. The full qualification still has
to repeat the topology ladder with its final capacity contract.

## Optional-feature screen

The following are exploratory all-on numbers, not a promotion. The run enabled
`QWEN_SD_RES1_V2=1`, KAI, decoder lane split 4, elastic lane, two-slot cohort,
and BF16 pre-up. `STREAM` is p95 RTF.

| 4x8 | TTFA p95 | STREAM p95 | TOTAL p95 | result |
|---:|---:|---:|---:|---|
| C8  | 248 ms | 0.753 | 0.800 | pass, diagnostic |
| C12 | 347 ms | 0.834 | 0.988 | pass, diagnostic |
| C14 | 440 ms | 0.952 | 1.143 | near edge |
| C16 | 470 ms | 0.947 | 1.155 | hard TTFA edge |
| C18 | 529 ms | 1.514 | 1.718 | over the preferred gate |

The two-wave result suggests C16 is the useful short-wave ceiling on this
topology, while C18 is already overloaded. It is not a C12/C16 qualification:
the broader product-quality gate, long/Poisson workload and final profile
manifest are still missing. The corrected BF16 micro-A/B and paired C1 audio
gate are recorded below.

The isolation arms explain the gain:

* lane + multi-slot without BF16: STREAM p95 0.846 at C12 and 0.982 at C14;
* BF16-only without lane/multi-slot: 1.375 at C12 and 1.518 at C14;
* the all-on improvement is therefore mainly scheduling/cohort structure, not
  BF16 pre-up alone. BF16 remains default-off until its broader product gate
  passes, despite the corrected micro-A/B and paired C1 audio result below.

## Cost-map and allocator audit

Cost-map parity passed at C2 and C4: path IDs, leaf classes, optimized/BLAS
coverage, fallback and UNKNOWN counts matched between census-only and
census+cost-map arms. The observed decoder coverage was about 86.3% optimized
SDOT and 13.7% BLAS, with no fallback or UNKNOWN path.

The reliable map puts roughly 90--93% of decoder time in `conv_stack`; CP
`gate_up` is the largest Talker/CP subregion. Pool wait was 24.8% of total
dispatch at C2 and 27.1% at C4 in the first screen. This is a real completion
wait/barrier cost, not an unaccounted kernel: the follow-up marker patch now
records the single- and multi-slot Arm SDOT/VNNI workers. The repeated C2/C4
parity run stayed PASS with no UNKNOWN/fallback mismatch; at C2 the serve map
reports `conv_stack` 93.0%, pool wait 16.2% of pool dispatch and 8/8 workers
entered with 100% panel occupancy, while C4 reports 91.4%, 24.1% and 8/8 with
100% occupancy. The multi-slot kernel is therefore active; its previous
unaccounted row was an instrumentation gap. Nesting integrity remained zero
mismatch, zero overflow and zero unbalanced ends.

The profiler's separate `INERT_FLAG` finding was a parser false positive: it
matched the summary sentence rather than an ignored runtime flag. The
effective configuration reported all 27 supplied flags honoured and zero
ignored.

The scratch counters recorded zero spills. In the C4 diagnostic, 16 stream
states settled at four 32 MiB blocks (128 MiB reserved, about 106 MiB peak),
with a few first-use states growing to five, six or nine blocks. The arena is
reset and parked per thread for reuse, so these are first-use/shape growths,
not mallocs per layer. The lane mailbox and slot jobs are preallocated; its
heap fallback is only for a mailbox overrun, and none was observed.

## Remaining code-level candidates

The repeated BF16 pack candidate is implemented in all three decoder forward
forms (full, streaming and ragged). One prepared KAI LHS now feeds Q/K/V and
one feeds gate/up; O/down retain the existing single-projection path. The
prepared pointer is kept in the caller's KAI TLS scratch until the joined
projection group completes, so no worker barrier or cross-thread scratch
ownership is introduced. The first shape check was corrected to validate the
projection output rows (`qkv_dim`/`dec_inter`), not the frame count, and the
corrected Graviton5 build/self-test passed.

The corrected all-on FAST A/B at 4x8/C12 (three waves, with TTFB and TTFA)
showed the same directional result in both orders: first order candidate vs
baseline STREAM/TOTAL p95 `.845/.977` vs `.899/1.027`, TTFA p95 303 vs 310 ms;
second order `.830/.966` vs `.875/1.051`, TTFA p95 323 vs 317 ms. There were
zero errors/rejects. This supports keeping the code, but it is not a claim of
a universal percentage because C8 was noisier. The corrected paired C1 WAVs
were byte-identical: `ryan_r000_short.wav` SHA256
`eed3786ca90ec52d033ce0a7aace8b1caa517dbf5802ed06ba151470f1f8fdb4` and
`ryan_r001_short.wav` SHA256
`faf19c2f5f7584df13180d04fefc97adfcfdfc47e836ddbfbe49b9d40a5126bc` in both
arms.

The pool-spin diagnostic at C12/4x8 was not monotonic: spin 0/4096/16384/65536
gave STREAM p95 `.837/.838/.929/.903` and TTFA p95 254/320/313/292 ms in the
individual three-wave runs. Keep the Arm default at 65536; the short sweep does
not justify a pool policy change.

KV-cache `realloc` happens only when a stream grows, and returned audio is
allocated per emitted chunk by contract. Neither showed a hot-loop failure in
this screen. Do not pre-reserve every 128 MiB arena blindly: it could trade a
small first-use saving for multi-worker RSS growth.

Next order remains: baseline Arm qualification, then the full Graviton5
campaign with the final profile manifest. The BF16 prepared-LHS code has its
micro A/B and paired-audio gate, but BF16 pre-up and multi-slot remain
default-off in `configs/perf/arm-product.json` and the Axion JSON until the
broader product-quality/serving gates pass. No JSON defaults were changed by
this diagnostic.
