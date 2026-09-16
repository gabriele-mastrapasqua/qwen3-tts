# P2 · per-slot direct transposed convolution — 2026-09-07

## Decision

Keep `QWEN_SD_DIRECT_CONVT=1` as an experimental, default-off arm. The
per-slot streaming helper is clean and reaches the real prefork/batched server,
but the short clean A/B is neutral at C4. It is not a serving promotion or a
C4 qualification result.

## Question and mechanism

The existing streaming transposed-convolution path materializes a full overlap
buffer, scatters every per-tap GEMM into it, then copies the useful output and
the next causal carry. The new helper accumulates directly into the useful
output and a small request-local carry workspace, then applies old carry and
bias in the same order. It keeps the existing per-tap GEMM; it does not claim
to remove all decoder fixed cost or to change the AMX conv1 kernel.

The implementation is in `qwen_tts_speech_decoder.c`, behind the existing
`QWEN_SD_DIRECT_CONVT` flag. A failed allocation returns `NULL` and preserves
the existing control path. The default remains off.

## Provenance and controls

- branch: `feature/x86-amx-vnni-oss`;
- runtime commit: `7d336aa` (`perf: remove per-slot decoder transpose materialization`);
- source fingerprint and AMX build: `7d336aa:clean`, `SIMD=amx`;
- host: GCP `c4-standard-24`, Xeon Platinum 8581C, one socket/NUMA,
  CPUs `0-11` online, SMT off;
- topology: 2 prefork workers × 6 physical cores, batch cap 2;
- model: Qwen3-TTS 1.7B INT8, `tests/load_texts_en.txt`, speaker `ryan`;
- workload: two synchronized waves at C2/C4, no profiler/census in KPI arms;
- fixed runtime: q8 (chunk 8, busy chunk 8), ragged threshold 2, Design-D
  INT8, warm strip, direct ragged depthwise path, engine decoder pool,
  `QWEN_BLAS_OWN=1`, prefix cache on;
- binary SHA-256 prefix: `34d1a6a9a6c075c7` (same binary for both arms).

The benchmark uses client-observed playback fields. Receive marks had a
coalesced-read share of 5.6% at C2 and 5.1% at C4 in both arms, so those values
are upper bounds on server-side lateness rather than direct socket-flush times.

## Local correctness gates

- clean native BLAS build and `./qwen_tts --self-test`: pass, zero failures;
- `make test-decode-quantum`: completed control and direct arms without a crash;
- local 0.6B streaming CLI control/direct output: byte-identical WAVs;
- `git diff --check`: pass before the implementation commit;
- focused native-code review of commit `7d336aa`: no actionable correctness,
  lifetime or concurrency defect found.

## Server A/B

The runner reported the requested flags in the engine startup line. Both arms
completed with zero errors, rejects and timeouts.

| arm | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p50/p95 ms | safe start p50/p95 ms | max gap p95 ms | stall@100 / @250 / @500 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control (`DIRECT_CONVT=0`) | 2 | 115/384 | 0.580/0.631 | 0.597/0.646 | 28/48 | 159/401 | 414 | 0% / 0% / 0% |
| direct (`DIRECT_CONVT=1`) | 2 | 115/352 | 0.570/0.579 | 0.585/0.593 | 26/27 | 142/374 | 387 | 0% / 0% / 0% |
| control (`DIRECT_CONVT=0`) | 4 | 425/510 | 0.774/0.810 | 0.815/0.945 | 265/329 | 689/775 | 554 | 88% / 0% / 0% |
| direct (`DIRECT_CONVT=1`) | 4 | 434/501 | 0.771/0.809 | 0.822/0.950 | 334/349 | 734/847 | 552 | 88% / 75% / 0% |

The C4 STREAM_RTF p95 delta is `-0.001` (0.13% relative), while TOTAL_RTF
p95 is slightly worse (`0.945` → `0.950`). The C2 improvement is based on
four requests per arm and is not independently persuasive. The direct arm's
C4 prebuffer and 250 ms fixed-buffer stall rate are worse in this sample. The
result is therefore **neutral/inconclusive for serving**, with no correctness
regression.

## Server-path proof

A separate C4 diagnostic enabled `QWEN_SD_PHASE=1` and
`QWEN_SD_RAG_STATS=1`; its KPI was not used as evidence because diagnostics
perturb timing. The server log contained 204 `per-slot` and 46 `ragged`
phase/update records. Representative ragged update records reported
`path=ragged direct_convt=6 direct_dwconv=22..28`; per-slot records reported
`path=per-slot direct_convt=6`. Thus the new helper executes in the actual
streaming server under both the per-slot and ragged batched decoder paths.

The same diagnostic exposed real decoder Design-D shapes such as
`M=768,N=256,K=5376`, `M=384,N=1280,K=2688`, `M=192,N=5120,K=1344`, and
`M=96,N=15360,K=672`. Startup `--caps`/`--dispatch-map` reported the clean
AMX build and the server flags selected Design-D INT8; the phase diagnostic
was used only to prove path reachability, not to publish a new AMX wall/MAC
share.

## Interpretation

The removed materialization is real, but it is not the dominant C4 serving
term under this workload. The control and direct C4 runs have essentially the
same measured CPU use, effective batch (`1.963` vs `1.964`) and STREAM_RTF.
This slice should remain available as a composable SQ-2 experiment while the
next P2 measurement targets the remaining decoder fixed cost. Do not promote
it, retune topology, or reopen BF16 based on this A/B.

## Next action

Continue P2 fixed-cost reduction only where a bounded measurement identifies a
larger target; keep the input-length result as the separate P3/PREFILL
requirement. If the next P2 slice is neutral again, move to the measured
decoder preparation/rendezvous bottleneck rather than accumulating more direct
copy-elimination flags.
