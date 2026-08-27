# KleidiAI — vendored subset

Source: https://github.com/ARM-software/kleidiai at commit `495f652`.
License: **Apache-2.0** (`Apache-2.0.txt`), © Arm Limited and affiliates.

Only the files needed for the Q4_0 (`qsi4c32p`) × dynamic-int8-activation
(`qsi8d32p`) path are vendored, unmodified:

| file | role |
|---|---|
| `kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0` | RHS (weights) packer. `qsu4c32**s16s0**` is ggml's own nibble order — a GGUF `Q4_0` tensor feeds it with **zero conversion**. |
| `kai_lhs_quant_pack_qsi8d32p_f32` | LHS (activations) quantize+pack, f32 → per-32-block int8. |
| `kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm` | GEMM, SMMLA. Needs `+i8mm`. |
| `kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod` | GEMV (m=1), SDOT. |

This is the same kernel pair llama.cpp selects for `Q4_0` on aarch64
(`ggml/src/ggml-cpu/kleidiai/kernels.cpp`). Files are copied verbatim: do not edit
them here — re-copy from upstream instead, and bump the commit above.
