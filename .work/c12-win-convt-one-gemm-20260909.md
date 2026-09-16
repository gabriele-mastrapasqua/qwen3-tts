# C12-WIN-2a — ConvT one-GEMM falsifier

Date: 2026-09-09
Host: Turin c8a.8xlarge-class, AMD EPYC 9R45, 32 physical cores
Profile family: `turin-c8a-32c-vnni-product`
Implementation source: temporary local change on top of `5813540` (reverted after test)
Decoder benchmark binary: `21ff8d09a82f5ad16f3176cf4de2d69ade679bc2227d723f22434fb339d421a0`

## Question

Can ConvTranspose replace the current `k` separate SGEMMs plus scatter with one
GEMM using a persistent `[out_ch][tap][in_ch]` weight layout, without increasing
the wall time of the small streaming decoder unit?

## Implementation

The diagnostic path was default-off and independently gated. It packed the six
ConvT weights once at load, expanded the strided input into a zero-filled
`[kernel*in_ch][full_len]` panel, ran one f32 GEMM, and preserved carry/bias order.
The existing per-tap path remained the control. The temporary runtime change was
removed after the falsifier; no new flag or persistent memory cost remains.

## Correctness

The existing decoder batch-parity binary was built on Turin. The repository default
0.6B model was not present on that checkout, so the target wrapper could not run its
default model. A direct run against the available 1.7B model with the diagnostic
path enabled passed:

```
worst max_abs_diff = 0.000000e+00
rms_diff          = 0.000000e+00
PASS (bit-identical)
```

This validates the tested streaming/ragged mapping, but does not rescue its cost.

## Decoder quantum benchmark

Existing `qwen_tts_decode_quantum` entry point, 1.7B model, 4 decoder threads,
five warm repetitions per cell. Values are p50 call wall in milliseconds.

| group / chunk | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| control B1 | 26.58 | 37.58 | 56.54 | 95.71 |
| one-GEMM B1 | 30.23 | 44.44 | 71.47 | 124.54 |
| control B2 | 55.28 | 76.90 | 112.32 | 190.19 |
| one-GEMM B2 | 61.08 | 90.53 | 142.99 | 250.34 |
| control B3 | 81.22 | 115.60 | 167.35 | 284.97 |
| one-GEMM B3 | 91.66 | 136.16 | 214.45 | 376.92 |
| control B4 | 108.59 | 152.65 | 225.83 | 389.65 |
| one-GEMM B4 | 122.69 | 181.96 | 285.27 | 501.06 |

The treatment is slower at every reported point: approximately +14–18% for
chunks 1–2 and +26–32% for chunks 4–8. The expansion panel and extra full output
materialization erase the intended GEMM amortization at these decoder geometries.

## Verdict

**REJECTED / REVERTED.** Correctness is clean, but the one-GEMM formulation is not
a performance candidate for the current f32 VNNI decoder. It was not promoted to a
server A/B and no server claim is made. A future ConvT attempt would need a different
dataflow that avoids the zero-expanded input panel or uses a native packed execution
kernel; simply consolidating the existing taps is not enough.
