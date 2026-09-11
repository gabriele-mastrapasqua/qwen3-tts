# Graviton5 Arm v2 mini-sweep and flow audit

Status: diagnostic handoff, not a qualification. The box was a clean
`964cb26` build of the public tree; no commit or push was made on the box.

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
the BF16 quality gate, paired audio, repeated waves, long/Poisson workload and
final profile manifest are still missing.

The isolation arms explain the gain:

* lane + multi-slot without BF16: STREAM p95 0.846 at C12 and 0.982 at C14;
* BF16-only without lane/multi-slot: 1.375 at C12 and 1.518 at C14;
* the all-on improvement is therefore mainly scheduling/cohort structure, not
  BF16 pre-up alone. BF16 remains default-off until its quality gate passes.

## Cost-map and allocator audit

Cost-map parity passed at C2 and C4: path IDs, leaf classes, optimized/BLAS
coverage, fallback and UNKNOWN counts matched between census-only and
census+cost-map arms. The observed decoder coverage was about 86.3% optimized
SDOT and 13.7% BLAS, with no fallback or UNKNOWN path.

The reliable map puts roughly 90--93% of decoder time in `conv_stack`; CP
`gate_up` is the largest Talker/CP subregion. Pool wait was 24.8% of total
dispatch at C2 and 27.1% at C4. The multi-slot decoder kernel currently does
not emit the profiler's `workers/units` markers, so
`decoder.conv_int8.panels` reports work as UNACCOUNTED. This is an
instrumentation gap, not evidence of idle workers; nesting integrity remained
zero mismatch, zero overflow and zero unbalanced ends.

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

The streaming decoder still materializes f32 activations and asks KAI to pack
the same source activation for each projection. Q/K/V repack `x_norm` three
times and gate/up repack it twice; KAI returns f32 output each time. This is
not a literal persistent `f32 -> bf16 -> f32 -> bf16` activation chain, but it
is repeated f32-to-BF16 packing of identical data and is the cleanest next
candidate. It needs a prepared-LHS lifetime/cache change plus numerical and
paired-audio A/B before implementation.

KV-cache `realloc` happens only when a stream grows, and returned audio is
allocated per emitted chunk by contract. Neither showed a hot-loop failure in
this screen. Do not pre-reserve every 128 MiB arena blindly: it could trade a
small first-use saving for multi-worker RSS growth.

Next order remains: baseline Arm qualification, one measured prepared-LHS
reuse experiment, paired audio for BF16 and multi-slot separately, then the
full Graviton5 campaign. Keep `configs/perf/arm-product.json` and the Axion
JSON default-off for BF16/multi-slot until those gates pass.
