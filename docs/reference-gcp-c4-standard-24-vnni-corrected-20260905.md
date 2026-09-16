# GCP `c4-standard-24` — corrected VNNI/BF16-prefill screen

Date: 2026-09-05. This is an exploratory measurement of the current dirty local
branch, not a production qualification. The source was copied directly to the VM;
no GitHub fetch, clone, pull, or commit was used.

## What was corrected

The historical C4 VNNI campaign used `SIMD=avx512vnni`, which did not compile
`-mavx512bf16`. Consequently, `QWEN_PREFILL_MATMAT=1` could not select native BF16
matmat and the serving path fell back to generic BF16/F32 work. This run used the
current checkout copied from the Mac and rebuilt the Ingot library and engine on
Linux with:

```text
make -C third_party/ingot clean
make -B blas SIMD=avx512bf16 -j12
```

The resulting compiler line contains both `-mavx512vnni` and `-mavx512bf16`. AMX is
visible on the CPU but is deliberately not compiled in this binary: this report is
VNNI/AVX-512-BF16 only.

The binary records:

| item | value |
|---|---|
| VM | GCP `c4-standard-24`, `us-central1-a` |
| CPU | Intel Xeon Platinum 8581C @ 2.30 GHz |
| cores / vCPU | 12 physical / 24 logical |
| cache | 2 MiB L2 per core, 260 MiB shared L3 |
| RAM | 88.38 GiB |
| binary | `SIMD=avx512bf16`, SHA-256 `768138a0631c04651927fd4f6074580b61011112c2d7116db37e5b15cd71d71d` |
| source fingerprint | `620279c-dirty:47327bf1aa58` |
| model | `qwen3-tts-1.7b`, loaded from the VM |
| Ingot | clean-rebuilt on the VM with the Linux compiler |

## Dispatch proof

The profile-aware preflight passed caps, self-test, and dispatch gate. The resolved
map is archived in `profiles/gcp_c4_20260905/qwen_c4_vnni_bf16_preflight_20260905/`.
The relevant rows are:

```text
talker.prefill.matmat_bf16       compiled=yes supported=yes resolved=ON
  avx512_bf16_matmat_available (VDPBF16PS)
talker.prefill.f32_blas_fallback compiled=yes supported=yes resolved=OFF
  not taken: bf16 matmat selected
cp.prefill2                      compiled=yes supported=yes resolved=ON*
gate.int8.vnni                   compiled=yes supported=yes resolved=ON
gate.bf16.avx512                 compiled=yes supported=yes resolved=ON
gate.int8.amx                    compiled=no  supported=no  resolved=OFF
gate.bf16.amx                    compiled=no  supported=no  resolved=OFF
```

The serving profile was:

```text
OPENBLAS_THREAD_TIMEOUT=1
QWEN_CP_PREFILL2=1
QWEN_DECODER_BATCH=0
QWEN_POOL_SPIN=4096
QWEN_PREFILL_MATMAT=1
QWEN_PREFIX_CACHE=1
QWEN_STREAM_DECODE_CHUNK=8
QWEN_STREAM_DECODE_CHUNK_BUSY=0
QWEN_VNNI_GEMV_MR=2
```

No `QWEN_NO_VNNI`, `QWEN_NO_VNNI_*`, `QWEN_NO_BF16_MATMUL`, or AMX kill switch was
passed. The log line `quantized prefill OFF` refers to the separate quality-sensitive
`QWEN_PREFILL_QUANT` feature; it is not the BF16 matmat fallback discovered in the
old campaign.

## Measurement protocol

The corrected model screen used `tests/bench_suite.sh --only fast` and the same
`tests/serve_parallel_wave.py` harness for the isolated A/B arms:

- synchronized true-wave arrival, five waves;
- short bank, five texts, `--precision int8`, C=1 and C=4;
- topology `2x6`, no audio saving and no shape census;
- exact profile passed to every server, with only the tested flag overridden in an A/B arm;
- every server printed and the harness verified the requested `[FLAGS]` values;
- all cells completed with zero errors and zero rejects.

With SMT on, the observed execution domain was `2W6T_m0-11|12-23`; the run used
one logical sibling set per worker. A separate corrected fast run was made with
SMT off, then the VM was verified back at `control=on`, `active=1`, `online=0-23`.

## Corrected fast result and SMT screen

`TTFB` and `TTFA` were both recorded. In these runs they differ by less than a
millisecond, so the compact table shows TTFA; the JSON artifacts retain both.

| state | C | TTFA p50 / p95 | TOTAL_RTF p50 / p95 | STREAM_RTF p50 / p95 | prebuffer p50 | ctx switches/s | errors/rejects |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMT on, `2x6` | 1 | 129 / 167 ms | 0.644 / 0.703 | 0.617 / 0.647 | 32 ms | 2,363 | 0 / 0 |
| SMT on, `2x6` | 4 | 276 / 393 ms | 1.135 / 1.207 | 1.015 / 1.074 | 555 ms | 4,939 | 0 / 0 |
| SMT off, `2x6` | 1 | 128 / 158 ms | 0.641 / 0.697 | 0.607 / 0.636 | 31 ms | 2,225 | 0 / 0 |
| SMT off, `2x6` | 4 | 249 / 375 ms | 1.166 / 1.241 | 1.050 / 1.132 | 594 ms | 4,520 | 0 / 0 |

The corrected VNNI build is healthy, but neither base C4 row is stable below
`STREAM_RTF=1` at p95. The SMT difference is within the noise of this five-wave
screen and goes in opposite directions for TTFA and streaming RTF. It does not
justify treating SMT as a bandwidth lever; the independent hardware screen remains
about 101.5 GB/s Triad on 12 physical CPUs and about 106 GB/s with the full logical
mask. The clean SMT-off model result is useful as a tail baseline, not as a
qualification.

## A/B: `QWEN_POOL_SPIN`

Each arm was a separate server process with the same model, topology, bank, profile,
seed, and five-wave C1/C4 workload.

| `QWEN_POOL_SPIN` | C | TTFA p50 / p95 | TOTAL_RTF p50 | STREAM_RTF p50 / p95 | prebuffer p50 | ctx switches/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 141 / 174 ms | 0.742 | 0.699 / 0.740 | 76 ms | 52,040 |
| 0 | 4 | 291 / 401 ms | 1.207 | 1.077 / 1.142 | 643 ms | 33,606 |
| 4,096 (profile) | 1 | 128 / 165 ms | 0.657 | 0.615 / 0.658 | 32 ms | 2,294 |
| 4,096 (profile) | 4 | 256 / 441 ms | 1.141 | 1.043 / 1.093 | 577 ms | 4,736 |
| 65,536 | 1 | 127 / 156 ms | 0.649 | 0.606 / 0.655 | 30 ms | 454 |
| 65,536 | 4 | 277 / 377 ms | 1.108 | 1.014 / 1.057 | 527 ms | 934 |

`0` is clearly the bad arm: it parks/reawakens often and produces tens of thousands
of context switches per second. `65,536` is the strongest exploratory arm: it cuts
the observed C4 switch rate by about 80% versus 4,096 and improves C4 throughput,
but it still does not produce a sub-one C4 stream tail. Keep it as a candidate for a
longer repeat/soak, not as a promoted profile value yet.

## A/B: `QWEN_STREAM_DECODE_CHUNK`

The harness verified the requested flag in every server's `[FLAGS]` line. The change
also appears in the measured output: the p50 number of chunks falls from 9 to 7 to
6 as the chunk grows from 4 to 8 to 16/32.

| chunk | C | TTFA p50 / p95 | TOTAL_RTF p50 | STREAM_RTF p50 / p95 | chunks p50 | prebuffer p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 130 / 160 ms | 0.689 | 0.646 / 0.651 | 9 | 33 ms |
| 4 | 4 | 277 / 434 ms | 1.182 | 1.073 / 1.201 | 9 | 343 ms |
| 8 (profile) | 1 | 131 / 165 ms | 0.651 | 0.611 / 0.651 | 7 | 33 ms |
| 8 (profile) | 4 | 272 / 394 ms | 1.151 | 1.045 / 1.142 | 7 | 595 ms |
| 16 | 1 | 131 / 157 ms | 0.630 | 0.597 / 0.654 | 6 | 240 ms |
| 16 | 4 | 266 / 427 ms | 1.088 | 0.985 / 1.079 | 6 | 899 ms |
| 32 | 1 | 131 / 158 ms | 0.635 | 0.587 / 0.652 | 6 | 274 ms |
| 32 | 4 | 271 / 410 ms | 1.042 | 0.948 / 1.068 | 6 | 1,039 ms |

The larger chunks improve the p50 stream rate at C4, but they delay the useful
steady stream: p50 prebuffer rises from 595 ms at 8 to 899/1,039 ms at 16/32.
Even at 32, p95 `STREAM_RTF` is 1.068 and every C4 request reports an underrun in
this synchronized workload. This is a throughput/overhead trade, not yet a stable
realtime result. `16` is the more balanced next candidate; `32` is the raw p50
throughput candidate if a larger deliberate prebuffer is acceptable.

## Artifacts and next boundary

Local artifacts are under:

- `profiles/gcp_c4_20260905/qwen_c4_vnni_bf16_preflight_20260905/`
- `profiles/gcp_c4_20260905/corrected_vnni_fast_20260905/`
- `profiles/gcp_c4_20260905/corrected_vnni_fast_smt_off_20260905/`
- `profiles/gcp_c4_20260905/ab_pool_spin_20260905/`
- `profiles/gcp_c4_20260905/ab_stream_chunk_20260905/`

The existing profile `configs/perf/gcp-c4-standard-24-vnni-ttfa.json` remains the
unqualified baseline (`POOL_SPIN=4096`, `STREAM_DECODE_CHUNK=8`). No A/B value is
promoted here. The next meaningful step is a repeated/soak comparison of the small
candidate set, optionally combining the best independent arms only after the
single-variable results are frozen; the AMX build remains a separate campaign.
