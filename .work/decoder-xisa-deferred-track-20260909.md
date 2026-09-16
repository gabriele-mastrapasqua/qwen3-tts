# DECODER-XISA — deferred cross-ISA decoder convergence

Status: DEFERRED. Do not execute during the current Turin C12-WIN ladder.

## Trigger and scope

Start only after the Turin C12-WIN work reaches a stable, evidence-backed checkpoint.
This is a dataflow-convergence track, not a new serving experiment and not permission to
reopen AMX/VNNI tuning while the paid host is active.

Current known state:

- x86 AVX-512 VNNI has the production reference `qwen_conv1d_int8_v2`, DL-2 elastic
  decoder lane and q4 units;
- x86 AMX has Design-D/direct and ragged paths, but not the same V2 dataflow contract;
- Arm/Linux has KleidiAI/i8mm/dotprod decoder paths, but does not automatically inherit
  the new direct-convolution/lane dataflow;
- old `DIRECT_*` flags are backend-specific and must not be treated as portable evidence.

## Ordered work

1. Define one common decoder state/dataflow: causal/dilated direct residual convolution,
   explicit streaming tails, persistent/prepacked weights where useful, direct epilogues,
   minimal ext/full/cut/materialization, efficient pre-upsample matmat, and amortized
   ConvT geometry. Separate this common layer from VNNI, AMX and KleidiAI execution leaves.
2. Audit RES1_V2 semantics: per-position activation quantization, weight scales, tap
   traversal, causal/dilated indexing, tail continuation, residual handling and numerical
   contract. Map the reference VNNI leaf to an AMX leaf and a native KleidiAI/i8mm/dotprod
   leaf; do not emulate another ISA's instructions.
3. Commonize post-V2 glue fixes above the leaf layer where ownership/layout permits:
   split input, remove full/cut copies and avoidable allocation, residual epilogue/snake
   improvements and final-conv vectorization. Keep the old path only as a parity fallback.
4. If Turin validates it, converge pre-upsample f32 weight traffic to a shape/precision
   dispatcher with native VNNI, AMX and Arm BF16/INT8 options plus exact fallback.
5. If Turin validates one-GEMM ConvT/repacked taps, make the logical layout common and
   provide native VNNI, AMX and KleidiAI leaves.
6. Re-qualify Arm after common changes, then re-evaluate AMX on the changed dataflow;
   old serving numbers are not evidence for the new architecture.

## Gates and non-goals

Each leaf requires deterministic self-test, streaming-continuation/reference comparison,
paired WAV/audio quality evidence, and playback-aware serving qualification before
promotion. Keep common dataflow, backend leaf, numerical semantics and streaming geometry
as separate review dimensions.

Do not implement live incremental text, global batching, a new scheduler, a portable
instruction emulation layer, or a wholesale three-decoder rewrite in this track. Future
measurements must preserve `MEASURED`, `DERIVED`, `PREDICTED` and `UNKNOWN` labels.
