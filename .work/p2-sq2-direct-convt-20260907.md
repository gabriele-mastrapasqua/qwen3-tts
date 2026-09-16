# Task · SQ-2 direct ragged transposed-convolution slice

## Question

Can the ragged decoder remove the large transposed-convolution overlap
materialization while retaining the existing per-tap GEMM, carry semantics and
Design-D AMX path?

## Known facts

The control path allocated a full `[out_ch][input*stride+carry]` buffer, scattered
every per-tap GEMM into it, then copied the useful output and next carry.  The
candidate commit `7f3ff3f` adds the default-off `QWEN_SD_DIRECT_CONVT` arm.  It
accumulates each tap directly into the request-local useful output range and a
small carry workspace, then applies the old carry and bias in the same order.

## Unknowns

The server comparison is a short WAVE, built from a dirty development snapshot;
it is not a SOAK or a qualification result.  The slice removes one materialization
and its scatter/copy, but does not yet implement the complete SQ-1 strip pipeline
or eliminate all decoder fixed cost.

## Files/functions inspected

`qwen_tts_speech_decoder.c` (`rag_convt`, `rag_convt_direct`,
`conv_decoder_forward_streaming_batch`), `qwen_tts_kernels.c` (flag reporting),
`docs/feature-flags.md`, `tests/decode_quantum_bench.c`, and
`tests/serve_parallel_wave.py`.

## Evidence

### Local gates

`make clean && make blas -j4` passed on the local native build; `./qwen_tts
--self-test` passed with zero failures; `git diff --check`,
`python3 tools/check_repo_integrity.py`, and `python3 tools/check_plan.py` passed.
The integrity tool retained its pre-existing warning for missing
`docs/emotion-seeds.md` referenced by the Makefile.

### Correctness quality gate

On the AMX host, the control and direct server arms used the same 1.7B model,
English short/medium/long bank, seed base 4242, 2x6 with SMT off, engine pool,
batch cap 2, q8, ragged threshold 2, Design-D INT8 and warm range strip.  Four
corresponding output WAVs were byte-identical between `DIRECT_CONVT=0` and `=1`
(same lengths and SHA-256 for long, medium and short requests).

### Exact-shape diagnostic

The dirty candidate was built with `SIMD=amx` on the GCP `c4-standard-24`
Xeon Platinum 8581C, 12 physical CPUs online (`0-11`), SMT off, one socket and
one NUMA node.  Source fingerprint was `44c6d49-dirty:9f9beddb9cf1` and the
binary SHA-256 prefix was `159abc51960eedd7`.  Three-repetition p50 decoder
quantum timings, control → direct, were:

| ragged group | chunk 16 ms | change |
|---:|---:|---:|
| 1 | 161.7 → 158.3 | -2.1% |
| 2 | 322.1 → 309.2 | -4.0% |
| 3 | 549.4 → 509.2 | -7.3% |
| 4 | 788.3 → 716.1 | -9.2% |

The smaller chunks were neutral to mildly better.  This is a kernel/decoder
diagnostic, not serving evidence.

### Server path proof

A separate one-wave diagnostic enabled phase/ragged counters.  Ragged entries
reported `fallback=0` and `amx=panels` for representative shapes including
`M=192,K=1344,Kp=1536` (80/80 panels), `M=96,K=672,Kp=768` (240/240),
`M=384,K=2688,Kp=2816` (20/20), and `M=768,K=5376,Kp=5376` (4/4).
`[SDUP] path=ragged` reported `direct_convt=6`, proving the new direct helper
was reached in the batched server path rather than only in a per-slot run.

### Short server A/B

Both arms used 2x6, SMT off, engine pool, batch cap 2, q8, threshold 2, two
waves, C2/C4, no profiler and the same seed base 2027.  Receive coalescing was
4.5% at C2 and 4.9% at C4 in both arms.

| arm | C | STREAM_RTF p50/p95 | TTFA p50/p95 ms | required prebuffer p50/p95 ms | safe start p50/p95 ms | stall@250 / @500 | errors/rejects |
|---|---:|---:|---:|---:|---:|---|---|
| control | 2 | 0.591 / 0.603 | 345 / 365 | 27 / 34 | 362 / 385 | 0% / 0% | 0 / 0 |
| direct | 2 | 0.589 / 0.592 | 344 / 366 | 23 / 35 | 365 / 383 | 0% / 0% | 0 / 0 |
| control | 4 | 0.747 / 0.822 | 443 / 548 | 287 / 370 | 814 / 835 | 25% / 0% | 0 / 0 |
| direct | 4 | 0.760 / 0.819 | 443 / 485 | 299 / 336 | 779 / 784 | 25% / 0% | 0 / 0 |

The eight-request C4 sample is too small for a serving claim.  It shows no
regression and a possible startup/cadence improvement, but not a decisive
steady-state win.  `TOTAL_RTF` at C4 was 0.821/1.060 control versus
0.801/1.062 direct; this reinforces that the arm is not yet a qualification
candidate.

## Conclusion

**KEEP, experimental/default-off.**  The direct form is mathematically and
server-path validated, reduces the largest temporary transposed-convolution
buffer, and is locally faster on larger ragged groups.  The current short server
A/B is neutral-to-slightly favorable but inconclusive for production.  Do not
promote the flag or call C4 qualified from this evidence.

## Next action

Retain this slice as a clean base for the remaining SQ-1/SQ-2 work.  Measure the
next small-quantum fixed-cost component before changing scheduler or lead policy;
repeat a larger clean-commit server WAVE only if that measurement identifies a
material serving effect.
