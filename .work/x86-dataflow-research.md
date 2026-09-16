# X86 dataflow research — AWS Turin VNNI + GCP Sapphire Rapids AMX/VNNI

Task: X86-2
Task: X86-3
Task: X86-4
Task: X86-5
(these were P5.5-P5.9 before PLAN reconciled the private probes with the tracked
 document docs/x86-int8-dataflow-2026-09-05.md; ids renamed, content unchanged)

Question: why do the AWS Turin VNNI host and the GCP Sapphire Rapids AMX/VNNI host
show the same broad x86 serving weakness, even though one has AMX and the other does
not?

Known facts:

- The AWS c8a production C4 map measures Talker decode at 44.5% of request wall,
  Code Predictor at 34.1%, and the speech decoder at 18.6%.
- The AWS C4 sample profile is dominated by `int8_matmat_vnni_tile_m4n4` (49.7%),
  `int8_matvec_vnni_rowsum` (10.3%), decoder VNNI convolution (7.4%), BF16 matmat
  (6.8%), worker spin (9.9%) and CP barrier spin (4.5%).
- The same AWS map records activation quantisation, BF16/input staging and
  gather/scatter work at layer frequency, and decoder `col`/`rk` transformations at
  frame frequency.
- AWS c8a is AMD EPYC 9R45/Turin, 16 physical cores, AVX-512 VNNI/BF16 and no AMX.
  Its canonical C4 soak is about `STREAM_RTF` p50 0.993–1.006 and p95 1.057–1.069.
- GCP c4-standard-24 is Intel Xeon Platinum 8581C/Sapphire Rapids, 12 physical
  cores with SMT, AVX-512 VNNI/BF16 and AMX tile/INT8/BF16 capability. Existing
  GCP notes prove AMX reachability at larger batches, while production-like C4
  decode remains mostly VNNI because the effective worker batch is around 1–2.
- The current repository already contains a bounded oneDNN oracle on an x86 VNNI
  host. It shows B=1 cells close to the engine and several B>1 cells where oneDNN's
  persistent RHS layout is faster, but its activation input is already quantised;
  that is not a complete end-to-end comparison.

Unknowns:

- No idle-GCP oneDNN/benchdnn/OpenVINO run was performed in this investigation.
  The box was not idle, so no package installation, benchmark, or model run was
  attempted.
- We do not yet have a hardware-counter measurement of actual DRAM/cache traffic
  for activation staging, q8 panels, scales, output scatter, or AMX activation
  packing.
- The common dataflow hypothesis is strong from static code plus the AWS profile,
  but the size of each term on Sapphire Rapids still needs a clean idle-box run.

Files/functions inspected:

- `docs/runtime-map-c8a-c4.md`, `docs/x86-oracle.md`,
  `docs/x86-optimization.md`, and the public AWS/GCP reference notes.
- `qwen_tts_kernels.c`: `quantize_act_int8_col_avx512`,
  `qwen_matmat_int8`, `qwen_matmat_int8_qkv`, `qwen_matvec_int8`,
  `qwen_i8mm_run*`, `qwen_int8_quant_rows`, `sd_conv1d_worker`,
  VNNI/AMX weight caches, and the timing/phase diagnostics.
- `qwen_tts_talker.c`: `qwen_talker_quantize_int8`, `batch_gather`,
  `batch_scatter`, `qwen_batch_proj_q*`, and the Talker region preparation.
- `qwen_tts_code_predictor.c`: `qwen_cp_quantize_int8`, CP region preparation,
  batched MTP/lm-head paths and the QKV/gate-up/down calls.
- `qwen_tts_speech_decoder.c`: causal-convolution im2col, INT8 weight caching,
  transpose-convolution `rk` staging and decoder phase counters.
- `tests/onednn_w8a8_bench.c`: existing shape matrix and its separate RHS reorder
  and compute measurements.

## 1. Common AWS + GCP symptoms

The hardware is different, but the serving regime is not. Both machines expose
AVX-512 VNNI, and both spend the important decode work at B=1/B=2 per worker. AMX
only changes the matrix instruction available when the batch and shape gates are
large enough; it does not remove the input-layout, quantisation, output-layout or
pool costs before and after the instruction.

| concern | AWS Turin VNNI-only | GCP Sapphire Rapids VNNI + AMX | repeated pattern |
|---|---|---|---|
| dominant decode | Talker/CP INT8 VNNI | C4 Talker/CP VNNI; AMX reaches higher-B cells | small-B INT8 is the production path |
| useful batch | C4 B≈2 matmat plus B=1 GEMV | C4 worker batches below the AMX INT8 gate; B=1/B=2 dominates | client concurrency is not kernel B |
| activation work | per-call quantisation and gather/scatter | same x86 dispatcher and region preparation | runtime data transformation surrounds the dot product |
| weight preparation | INT8 conversion once; optional VNNI reorder | INT8 conversion once; optional AMX/VNNI reorder | prepack is layout-only, not a new low-bit representation |
| decoder | VNNI conv plus FP32/BLAS paths | same decoder structure when the conv path is selected | im2col/panel preparation remains separate |
| architectural result | high local worker-read utilisation, but C4 still near the realtime boundary | AMX capability does not automatically move C4 decode | ISA presence is not the same as mature dataflow |

This is evidence for a common x86 execution problem, not proof that both boxes have
the same absolute bandwidth or the same kernel bottleneck. The AWS map has measured
time shares; the current GCP evidence establishes capability and dispatch regimes,
not a new clean cross-box performance qualification.

## 2. Current dataflow and lifetime

### Weight side

`qwen_talker_quantize_int8` and `qwen_cp_quantize_int8` convert BF16 weights to
per-output-row INT8 plus one FP32 scale when the model is loaded. This is a load or
model-preparation event, not a per-projection conversion. The current code retains
the original BF16 tensors as well, so the INT8 path reduces the streamed weight
payload but does not make the model representation weight-only in the ARK sense.

The optional VNNI cache packs INT8 weights in an N16/K4 traversal and the optional
AMX cache packs BF16 or INT8 tiles. Both caches are keyed by source pointer and
shape (AMX also by element kind), and their stored byte count is approximately the
source byte count. They are layout reorders, not lower-bit conversion. The VNNI
lookup is shape/regime-gated; it is not the GEMV solution for every B. AMX packing
is also opt-in and only useful when the AMX runner is selected.

### Ordinary batched Talker/CP projection

The normal `qwen_batch_proj_q` path does the following for an INT8 projection when
the contiguous fast path is not delegated to another backend:

1. `batch_gather` reads slot-major FP32 rows and writes a `[cols][B]` staging matrix.
2. `qwen_matmat_int8` quantises each activation column. The AVX-512 column helper
   makes an absolute-value pass, then a second pass that scales/rounds and writes
   `qXt`; it also gathers when the input is still strided.
3. VNNI consumes `qXt` directly. AMX additionally repacks `qXt` to tile layout
   before the tile loop.
4. The integer result is scaled with per-row weight scales and per-column activation
   scales, then `batch_scatter` writes the output back to slot-major layout.

The QKV wrapper shares one `qXt` and one activation-scale array across Q/K/V. That
is already a real reuse point; it is wrong to count Q, K and V as three independent
activation quantisations in that fused path. It still pays the staging/layout work
around the shared buffer.

### Persistent Talker/CP regions

The region code makes the lifetime more explicit. `*_region_gather_quant` copies one
slot into the transposed `Xt` buffer and immediately calls the same per-column
quantiser into a shared `qx/sx` region buffer. The region then runs QKV, attention,
O-projection, gate-up and down with barriers and scatters the result back after each
stage.

This removes repeated pool entry, but it does not make activation preparation free:
the quantise/gather sequence occurs for each projection whose input changed. For VNNI
the row-block runner reads the already-built `qx`. For AMX, `qwen_i8mm_run` currently
packs the same B×K activation into per-thread scratch before calling the AMX runner;
the source comment explicitly describes this as O(B×K) work per thread. That is a
GCP-only extra on AMX, but the surrounding quantise/staging contract is common to
both x86 hosts.

### Decoder

For the standard INT8 causal convolution, `sd_conv1d_worker` builds a float im2col
panel of up to `SD_INT8_NC=128` positions, calls `qwen_int8_quant_rows`, and then
calls the VNNI panel GEMM. The INT8 weight, per-block scales and row sums are cached
by source pointer after the first use. On the current x86 path, the row quantiser is
not an AVX-512 fused gather/quantise kernel; the panel transformation is therefore
separate from the VNNI compute.

The transpose-convolution path still materialises the FP32 `rk` buffer, runs a BLAS
GEMM per kernel slice, and scatters into the output. The working-tree scratch fixes
reduce allocation churn in the stream path, but they do not remove the im2col, q8
panel, `rk`, scale or scatter bytes.

## 3. AWS runtime-map cross-check and the roofline caveat

The AWS statement that Talker is around 89% of its worker-local read roof is useful,
but it is not a proof that the dataflow is optimal. The repository's own diagnostics
make the denominator limitation explicit:

- `qwen_kernel_timing_report` calls its weight-byte field a **one-read lower bound**;
  it excludes non-weight traffic and does not measure DRAM bytes.
- `qwen_vnni_phase_report` labels `vnni_dot work_gb` as the one-read INT8 weight
  count, not measured hardware traffic.
- The AWS roof is local to the worker/CCX model used by that profile, not a complete
  useful-MAC roof for the request.

For one B×K activation matrix, a useful accounting model should at least separate:

| term | ordinary VNNI B=1/B=2 | AMX matmat or region |
|---|---|---|
| source FP32 read | one or more passes for absmax and quantisation | same |
| FP32 staging | gather write plus quantiser rereads when `[cols][B]` staging is used | same before tile pack |
| q8 write/read | q8 output write, then VNNI reads or AMX pack reads | q8 write, pack read and tile-layout write |
| weight | INT8 matrix traversal, plus scale/row-sum metadata | packed or source tile traversal, plus metadata |
| output | accumulator conversion/scale and output store | tile store, conversion/scale and output store |
| layout | gather and output scatter/transposes | gather/scatter plus AMX pack; region AMX can repeat pack per worker |

The exact cache residency determines which of these becomes DRAM traffic, so the
table is not a bandwidth measurement. It does show why a weight-only roof can hide a
large amount of useful-MAC-independent work. The decisive next number is:

> actual bytes moved per useful MAC, split into weight, activation preparation,
> temporary layout, metadata and output, compared with a oneDNN path measured from
> the same FP32 activation input.

The current instrumentation can provide wall-time and nominal weight bytes, but not
that full byte total. A clean experiment needs hardware counters or explicit phase
byte accounting, and must keep the complete activation contract identical on both
sides.

## 4. oneDNN oracle on the real Qwen shapes

The existing local oracle in `docs/x86-oracle.md` and
`tests/onednn_w8a8_bench.c` already covers the requested shape family:

- CP: `4096x1024`, `6144x1024`, `1024x3072`, plus the lm-head family.
- Talker: `4096x2048`, `12288x2048`, `2048x6144`.
- B=1, 2, 4, 8 and 16 where the harness supports the cell.

It reports persistent RHS reorder separately from compute and records the selected
oneDNN implementation. The prior oracle selected `brg_matmul:avx512_core_vnni`
and a `BA16a64b4a` RHS layout. The result is informative but not apples-to-apples:
the oneDNN harness receives pre-quantised activations, while the engine timing
includes its activation quantisation, row-sum handling, wrapper and pool behavior.

When the GCP box is genuinely idle, the bounded oracle should be run in an isolated
directory, with no loader or runtime changes in this branch. For every shape record:

1. source-to-q8 activation preparation;
2. oneDNN RHS reorder/prepack, cold and warm;
3. activation reorder, if any;
4. compute implementation and wall time;
5. post-ops/scales/output reorder;
6. nominal and measured bytes, thread count and physical-core placement.

`benchdnn` is useful for identifying VNNI/AMX/BRGEMM choices, while the small exact-
shape harness is needed for complete-path comparison. The goal is not to add oneDNN
as a dependency; it is to learn which layout and lifetime contract removes work.

References: [oneDNN benchdnn](https://github.com/uxlfoundation/oneDNN/tree/main/tests/benchdnn),
[oneDNN BRGEMM guide](https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_brgemm.html).

## 5. AutoRound / ARK / W4A8 question

The current x86 Q4 path is a useful baseline but should not be conflated with ARK:
Q4_0 weights are block-packed and the x86 path dynamically quantises the activation
to INT8, so the eligible path is approximately W4A8. It is a simple runtime format,
not an offline calibrated AutoRound representation.

The public ARK CPU material exposes packed weight-only operators including
`woqgemm` and `woqgemm_s8`, and supports CPU INT8/BF16/FP32 compute. It also describes
dynamic activation quantisation for some INT8 compute paths. That makes ARK useful as
a representation/dataflow oracle, but not evidence that CPU W4A8 is supported or
faster for this TTS model. The W4A8/Q2A8 examples must not be transferred from an
XPU path to Sapphire Rapids without a CPU-specific run.

AutoRound plus the vLLM/LLM Compressor integration is principally an offline
quantisation and model-format story. The public example emphasizes W4A16-style
deployment; it does not establish a Qwen3-TTS Xeon CPU result. W4A16 could reduce
weight traffic at B=1, while W8A16 or the current W8A8-like path may preserve more
quality, but the winner depends on activation conversion and output post-processing.

The useful comparison matrix is therefore:

| representation | likely contract | question |
|---|---|---|
| current INT8 | W8A8-like, per-row weight scale plus dynamic per-column activation scale | is activation preparation the dominant avoidable cost? |
| current Q4_0 x86 | simple W4A8-like runtime path | does lower weight traffic offset q8 activation/correction work? |
| AutoRound W4A16 | offline packed weight-only, FP16/BF16 activation likely | does B=1 weight traffic win without an expensive dequant path? |
| ARK W4/W8 CPU | persistent packed weight-only operator, possibly dynamic q8 activation | does its packing and accumulator contract map to the exact Qwen shapes? |
| explicit W8A8 recipe | calibrated per-channel/selected-layer policy | can it preserve audio while reducing scale/correction overhead? |

No loader or model precision change belongs in the current branch until a small
isolated oracle has: exact tensor import, audio-quality parity, warm B=1/B=2 latency,
packed-weight size, and a complete activation-preparation breakdown.

References: [vLLM AutoRound integration](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc),
[Intel ARK CPU README](https://github.com/intel/auto-round/blob/main/auto_round_extension/ark/README.md),
[AutoRound](https://github.com/intel/auto-round),
[IPEX CPU LLM guide](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/llm/inference/README.md),
[OpenVINO CPU device](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.rst),
[Intel Xeon/vLLM guide](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/A-Practical-Guide-to-CPU-Optimized-LLM-Deployment-on-Intel-Xeon/post/1737233).

## 6. Activation quantisation is the primary common hypothesis

The current VNNI/AMX preparation has this pass structure for a normal projection:

- pass A: read the FP32 activation column and find `absmax`;
- pass B: read it again, scale/round/clamp and write q8;
- VNNI: consume q8 directly;
- AMX: read q8 and write a second tiled activation layout;
- compute and scale/store the output;
- caller-side output scatter when the batch is not already in destination layout.

There are important exceptions that keep the audit honest:

- QKV shares one activation quantisation and one q8 buffer across Q/K/V.
- The current region path avoids a separate generic `batch_gather`, but still writes
  an FP32 transposed staging buffer before quantisation.
- The AMX in-region runner can repack the same q8 input once per worker thread.
- B=1 VNNI GEMV has a different row-block microkernel and no AMX tile pack.

The narrowest high-value candidate is therefore not “remove dynamic quantisation”
by changing numerical policy. It is to test a bit-preserving source-to-q8 path for
B=1/B=2 that avoids the intermediate FP32 staging write/read where possible:

1. first pass over the original slot layout for the same absmax result;
2. second pass over the original layout directly into q8;
3. preserve the current scale, rounding, zero handling and QKV sharing;
4. feed the existing VNNI/AMX runner and compare complete call wall time.

This may trade contiguous staged reads for strided source reads, so it is a measured
candidate, not an assumed win. Its value is that it tests the common x86 dataflow on
both Turin and Sapphire Rapids without changing weight precision or the kernel's
integer arithmetic.

## 7. Decoder im2col + `qwen_int8_quant_rows`

The decoder has a separate, more local opportunity. On x86 the causal-conv path does:

1. gather/pad input into float `colf`;
2. scan rows for scales;
3. rescan rows to quantise into `colq` and write per-row/block scales;
4. run the VNNI panel kernel with cached quantised weights and row sums.

A fused AVX-512 gather/quantise panel could remove the float panel or at least combine
the gather with one of the quantisation passes. It must preserve zero padding, block
scales, row sums and the correction term exactly. This is a good bounded decoder
experiment, but it is not the strongest explanation of the whole x86 gap: decoder
is 18.6% of AWS request wall and its VNNI conv is 7.4% of the C4 samples, while
Talker+CP dominate the request.

The stream scratch work already makes allocations grow-once. That removes allocator
churn, not the data movement; the two should not be credited as the same fix.

## 8. Persistent prepack quality

Current prepack audit:

- INT8 weights are quantised once at model load.
- VNNI prepack is an optional source-pointer/shape cache in an N16/K4 layout. It
  preserves roughly the same byte count and is only valid for aligned shapes.
- AMX prepack is an optional source-pointer/shape/kind cache in 16-row tile blocks,
  with K steps of 32 for BF16 and 64 for INT8.
- The packed weight is consumed directly by the corresponding tile/VNNI runner when
  its dispatch gate selects that path; otherwise the source layout remains in play.
- The parent-side caches can be inherited by prefork workers, but that is a lifetime
  optimization, not proof that every call avoids a reorder.

This is structurally behind oneDNN/KleidiAI's clearer persistent RHS contract, but
the previous VNNI prepack screen did not produce a stable server win. The next oracle
must distinguish “pack happened once” from “the selected hot kernel actually reads
the packed bytes and avoids all other layout work.”

## 9. Small-B x86 and assembly questions

The current production evidence says to prioritize B=1/B=2:

- AWS C4 uses B=2 VNNI matmat and substantial B=1 GEMV.
- GCP C4 effective worker batches remain below the default AMX INT8 B=4 gate, so
  AMX presence cannot dominate the steady decode path.
- The local oneDNN oracle is close at B=1 but often ahead at B>1. Because its input
  is prequantised, this points to a possible RHS/blocking advantage but does not
  isolate activation preparation.

The later assembly comparison should inspect the complete B=1/B=2 runner for:

- output-row accumulator blocking and inactive accumulators;
- scalar scale loads or reductions in the inner loop;
- activation reuse across output rows;
- N/K blocking and prefetch distance;
- register spills and tail handling;
- whether a packed RHS is actually used for the observed shape.

Do not spend another experiment proving that AMX enters at B>=4; that dispatch fact
is already established. The unresolved question is whether the VNNI small-B path and
its preparation contract do unnecessary work.

## 10. Plan append

The appended tasks are intentionally small and all point to this note:

- P5.5 / X86.1 — oneDNN/benchdnn oracle on the exact CP/Talker shapes, split reorder,
  activation preparation, compute and post-ops.
- P5.6 / X86.2 — audit activation quantisation, staging, pack and scatter lifetimes
  for the B=1/B=2 Talker/CP paths.
- P5.7 / X86.3 — evaluate a decoder im2col + `qwen_int8_quant_rows` fused-panel
  candidate, with exact scale/correction parity.
- P5.8 / X86.4 — compare the complete B=1/B=2 VNNI path against a mature x86
  small-B oracle and inspect the selected assembly/layout.
- P5.9 / X86.5 — verify persistent packed-weight lifetime and direct consumption
  against oneDNN/ARK-style reorder contracts.

## Final verdict

### Ranked likely inefficiencies

| rank | hypothesis | confidence | possible impact | code area | numerical risk |
|---|---|---|---|---|---|
| 1 | FP32 gather/staging plus two-pass dynamic activation preparation around each small-B projection; AMX adds another activation-layout pass | high from code, medium for exact wall share | high because Talker+CP are over 78% of AWS wall | `batch_gather`, region `*_gather_quant`, `quantize_act_int8_col`, `qwen_matmat_int8`, `qwen_i8mm_run*` | low if scale/rounding and QKV sharing remain bit-identical |
| 2 | VNNI B=1/B=2 row blocking and packed-RHS use are not matched to the real worker B | medium-high | high only if complete B=2 share is material; local oneDNN gap makes it credible | VNNI matvec/matmat runners and dispatch gates | low; integer accumulation can remain unchanged |
| 3 | Decoder float im2col and scalar x86 row quantisation are separate passes before VNNI conv | high locally, medium overall | medium; decoder is 18.6% of AWS wall | `sd_conv1d_worker`, `qwen_int8_quant_rows`, `sd_gemm_panel` | medium; exact block scales/correction need parity |
| 4 | Prepack is optional/layout-specific and may not be consumed by every hot shape; AMX region activation pack repeats per worker | medium | medium; larger at B>=4 or where RHS reuse is real, less at C4 B=1/2 | VNNI/AMX weight caches, `qwen_i8mm_run*` | low for reorder-only changes, but cache footprint is a risk |
| 5 | AutoRound/ARK W4A16/W4A8-style representation could reduce weight traffic | low-medium until import and CPU measurements exist | potentially high for B=1 bandwidth, but unknown after dequant/activation work | model format/loader and selected linear operators | high: audio quality, scale contract and loader compatibility |

### Strongest next candidate for Claude

Start with one complete, bit-preserving B=1/B=2 activation-preparation path for
Talker/CP: eliminate avoidable FP32 staging traffic while retaining the existing
two-pass scale/rounding semantics, shared QKV q8 input and current VNNI/AMX runner.
Measure wall time and bytes per useful MAC before widening it to all projections.

This candidate explains both boxes because it exists before the ISA-specific split:
Turin executes the VNNI branch and Sapphire Rapids executes VNNI for its C4 small-B
decode; AMX only changes the later compute/pack regime. It also tests the hypothesis
that “near the current read roof” can mean “bandwidth-bound inside an avoidable
dataflow,” not “the model has no remaining useful optimization.”

Do not make AutoRound, ARK, oneDNN or OpenVINO a runtime dependency from this result.
Use them as isolated oracles. Do not promote W4A16/W4A8 or a new activation scale
policy until the exact Qwen3-TTS audio path has quality and complete-path evidence.

Conclusion: the strongest current explanation for x86 looking less mature than
ARM/KleidiAI is not missing AMX. It is that the x86 path still exposes more of the
activation layout/quantisation contract to each projection and small-B call, while
mature backends make RHS blocking, LHS preparation, post-ops and lifetime a tighter
single contract. This is a testable hypothesis, not yet a measured causal proof.

Next action: wait for a genuinely idle GCP box, then run only the isolated oracle
matrix and record full preparation/reorder/compute/post-op costs. Until then, keep
the plan append-only and leave runtime/config/profile files untouched.

## GCP oneDNN phase oracle — 2026-09-05

The isolated standalone oneDNN run completed without installing anything else or
rebuilding KTKernel. It covered the six real projection shapes, B=1/2/4/8/16,
for INT8 and BF16 (60 rows total). The machine was idle before the run; execution
used 12 OpenMP threads pinned to cores 0–11. Every row selected
`brg_matmul:avx10_1_512_amx`.

The benchmark separates one-time primitive/weight work from per-call work. The
reported `act_prep` is host-side activation preparation: dynamic INT8
absmax/quantisation for INT8, and BF16 conversion for BF16. `compute` is the
warmed median matmul. `total` is activation preparation plus compute. There was
no oneDNN activation reorder in this descriptor (`src_reorder=0` for all rows),
so this is an oracle for the compute and dataflow cost, not proof that oneDNN
fuses our activation preparation.

### INT8 steady state

Times are microseconds; GMAC/s is the reported matmul throughput.

| shape | B | act prep | compute | total | GMAC/s |
|---|---:|---:|---:|---:|---:|
| CP qkv 4096×1024 | 1 | 3.71 | 18.48 | 22.83 | 226.9 |
| CP qkv 4096×1024 | 2 | 7.19 | 11.02 | 18.94 | 761.5 |
| CP qkv 4096×1024 | 4 | 12.94 | 11.91 | 24.39 | 1409.2 |
| CP qkv 4096×1024 | 8 | 25.88 | 10.49 | 38.03 | 3198.6 |
| CP down 1024×3072 | 1 | 10.42 | 6.51 | 21.54 | 482.9 |
| CP down 1024×3072 | 2 | 18.94 | 6.26 | 29.52 | 1004.4 |
| CP down 1024×3072 | 4 | 38.81 | 6.30 | 49.19 | 1996.6 |
| CP down 1024×3072 | 8 | 75.08 | 7.73 | 84.86 | 3257.5 |
| Talker gate/up 12288×2048 | 1 | 6.14 | 86.62 | 90.80 | 290.5 |
| Talker gate/up 12288×2048 | 2 | 14.58 | 101.24 | 119.84 | 497.2 |
| Talker gate/up 12288×2048 | 4 | 25.34 | 89.15 | 119.19 | 1129.2 |
| Talker gate/up 12288×2048 | 8 | 53.67 | 106.27 | 151.92 | 1894.5 |
| Talker down 2048×6144 | 1 | 18.99 | 43.09 | 64.96 | 292.0 |
| Talker down 2048×6144 | 2 | 44.68 | 50.10 | 97.28 | 502.3 |
| Talker down 2048×6144 | 4 | 89.38 | 51.04 | 147.01 | 986.2 |
| Talker down 2048×6144 | 8 | 148.10 | 45.60 | 202.39 | 2207.4 |

The remaining B=16 rows are in the isolated result CSV. The decisive pattern is
that INT8 activation preparation scales with B while AMX compute is nearly flat
for several shapes. At CP qkv B=8, preparation is 25.9 us versus 10.5 us compute;
at CP down B=8 it is 75.1 versus 7.7 us; at Talker down B=8 it is 148.1 versus
45.6 us. The preparation is therefore not a small wrapper cost at production
small-B shapes.

### BF16 control

| shape | B | conversion/prep | compute | total | GMAC/s |
|---|---:|---:|---:|---:|---:|
| CP qkv 4096×1024 | 1 | 0.09 | 34.66 | 34.66 | 120.1 |
| CP qkv 4096×1024 | 2 | 0.16 | 30.21 | 30.61 | 277.6 |
| CP qkv 4096×1024 | 4 | 0.37 | 32.50 | 35.95 | 516.2 |
| CP qkv 4096×1024 | 8 | 0.71 | 33.21 | 36.66 | 1010.3 |
| CP down 1024×3072 | 1 | 0.24 | 21.94 | 23.48 | 143.4 |
| CP down 1024×3072 | 2 | 0.53 | 26.72 | 29.18 | 235.5 |
| CP down 1024×3072 | 4 | 1.19 | 26.73 | 30.56 | 470.7 |
| CP down 1024×3072 | 8 | 2.37 | 28.38 | 36.04 | 886.7 |
| Talker gate/up 12288×2048 | 1 | 0.17 | 206.49 | 206.59 | 121.9 |
| Talker gate/up 12288×2048 | 2 | 0.33 | 211.00 | 212.09 | 238.5 |
| Talker gate/up 12288×2048 | 4 | 0.73 | 238.67 | 242.00 | 421.8 |
| Talker gate/up 12288×2048 | 8 | 1.33 | 205.96 | 209.04 | 977.5 |

BF16 conversion is negligible here, while BF16 compute is slower than INT8. This
isolates the INT8 activation preparation as a real additional cost rather than a
generic oneDNN call overhead.

### Persistent weight cost

Weight layout and oneDNN reorder are one-time costs, not per projection call. For
CP qkv INT8 B=1 the host layout cost was 15.73 ms and oneDNN reorder 0.710 ms for
a 4 MiB packed RHS. Talker gate/up INT8 was 87.03 ms plus 1.460 ms for 24 MiB;
the BF16 equivalent was 153.0 ms plus 2.839 ms for 48 MiB. Primitive creation was
roughly 0.14–0.48 ms. First execution can be much larger than steady state
(48–411 us in the sampled rows), so it must not be used as the serving number.
This confirms that oneDNN has a persistent packed-RHS contract; it does not
confirm that every current engine path consumes an equivalent layout directly.

### Comparison with existing engine measurements

The only existing engine-vs-oneDNN table is from the separate Scaleway AVX-512
VNNI oracle, not this GCP Sapphire Rapids host. Ratios below are
`our engine / oneDNN`, in microseconds, and must not be read as a same-host AMX
comparison:

| shape | B=1 | B=2 | B=4 | B=8 |
|---|---:|---:|---:|---:|
| CP qkv 4096×1024 | 6105/6784 = 0.90× | 8721/8010 = 1.09× | 11884/8585 = 1.38× | 29621/12482 = 2.37× |
| CP gate/up 6144×1024 | 8818/9220 = 0.96× | 11957/12955 = 0.92× | 15802/12260 = 1.29× | 41437/35153 = 1.18× |
| CP down 1024×3072 | 4596/4283 = 1.07× | 9801/4989 = 1.96× | 15410/7849 = 1.96× | 30140/10633 = 2.84× |

There is no same-host GCP engine timing for the Talker rows in the current
evidence, so those GCP rows remain oracle numbers rather than an invented ratio.
The cross-box result does not show a universal 2× oneDNN matmul advantage. It
does show that the wrapper/dataflow comparison is essential: the oneDNN compute
number excludes the dynamic activation preparation that dominates several total
INT8 rows.

### AutoRound/ARK boundary

Source inspection only, with no Torch installation and no AutoRound benchmark:

- W4A16 is a documented CPU/weight-only direction and ARK exposes persistent
  packed/repacked weight APIs; it is the most plausible B=1 weight-traffic oracle.
- CPU INT8/W8A8-style compute paths exist and include dynamic activation
  quantisation in some paths.
- CPU W4A8 is not established by the inspected docs; the visible W4A8 rescale
  material is XPU-specific. Do not assume it is a CPU format.
- No quality or speed claim is made for this Qwen3-TTS model without a loader and
  complete-path measurement. OpenVINO remains a later graph-level TODO.

### Updated verdict

The strongest next experiment remains one bit-preserving fused/avoided activation
preparation path for Talker/CP. It directly explains both Turin/VNNI and
Sapphire-Rapids/AMX observations: the same runtime transformation happens before
the ISA-specific matmul, and at B=1/2 the transformation can outweigh the
benefit of a faster kernel. Persistent packed weights are worth keeping as the
oracle design, but replacing the engine with oneDNN is not justified by these
numbers alone.

## ARK CPU W4A16 oracle — 2026-09-05

The bounded AutoRound/ARK test reached the kernel numbers without installing a
model stack or changing qwen-tts. PyTorch 2.14.0+cpu was isolated under the GCP
oracle directory; ARK used `repack_quantized_weight` followed by `woqgemm`.
The synthetic weights use symmetric INT4, group size 128, FP32 scales, and are
an execution-format oracle, not an AutoRound quality/calibration result.

All rows use 12 threads pinned to cores 0–11. Input activation starts as BF16;
the ARK CPU wrapper converts it to FP32 before `woqgemm`. There is no dynamic
INT8 activation quantisation. The public wrapper exposes the BestLA/ARK
`woqgemm` operation rather than a runtime kernel-name query. Its CPU binary
contains the AMX-BF16 `ActivationConverterFp32` W4 path, while the reported
dispatch label is therefore conservatively `ARK/BestLA CPU woqgemm`.

Times are microseconds. `total` includes BF16→FP32 conversion plus the ARK call;
`compute` uses an already-converted FP32 activation. Pack time and packed bytes
are one-time per shape and independent of B.

| shape | packed bytes | pack us | B | act prep us | compute us | total us |
|---|---:|---:|---:|---:|---:|---:|
| CP qkv 4096×1024 | 2,245,760 | 1,601 | 1 | 1.70 | 51.58 | 52.98 |
| CP qkv 4096×1024 | 2,245,760 | 1,601 | 2 | 1.77 | 52.22 | 56.09 |
| CP qkv 4096×1024 | 2,245,760 | 1,601 | 4 | 2.04 | 66.95 | 72.47 |
| CP down 1024×3072 | 1,723,520 | 865 | 1 | 1.60 | 34.11 | 36.95 |
| CP down 1024×3072 | 1,723,520 | 865 | 2 | 1.86 | 32.04 | 40.12 |
| CP down 1024×3072 | 1,723,520 | 865 | 4 | 2.59 | 38.30 | 49.25 |
| Talker gate/up 12288×2048 | 13,369,472 | 4,716 | 1 | 1.79 | 221.06 | 223.87 |
| Talker gate/up 12288×2048 | 13,369,472 | 4,716 | 2 | 2.24 | 240.71 | 245.97 |
| Talker gate/up 12288×2048 | 13,369,472 | 4,716 | 4 | 2.45 | 289.75 | 298.21 |
| Talker down 2048×6144 | 6,737,024 | 1,894 | 1 | 1.93 | 123.90 | 125.13 |
| Talker down 2048×6144 | 6,737,024 | 1,894 | 2 | 3.11 | 126.46 | 139.17 |
| Talker down 2048×6144 | 6,737,024 | 1,894 | 4 | 4.79 | 147.62 | 147.05 |

### W8A8 total comparison

The comparison baseline is the preceding same-host GCP oneDNN W8A8 phase oracle,
not a qwen-tts engine timing. Values are `W8A8 total → ARK W4A16 total`, followed
by the W4A16/W8A8 ratio:

| shape | B=1 | B=2 | B=4 |
|---|---:|---:|---:|
| CP qkv | 22.83 → 52.98 (2.32×) | 18.94 → 56.09 (2.96×) | 24.39 → 72.47 (2.97×) |
| CP down | 21.54 → 36.95 (1.72×) | 29.52 → 40.12 (1.36×) | 49.19 → 49.25 (1.00×) |
| Talker gate/up | 90.80 → 223.87 (2.47×) | 119.84 → 245.97 (2.05×) | 119.19 → 298.21 (2.50×) |
| Talker down | 64.96 → 125.13 (1.93×) | 97.28 → 139.17 (1.43×) | 147.01 → 147.05 (1.00×) |

The decisive result is negative for a blanket W4A16 replacement on these shapes:
the activation-preparation problem disappears, but ARK's FP32 weight-only path is
slower at CP qkv and Talker gate/up. It approaches parity only on the larger-K
CP/Talker down shapes at B=4. W4A16 still reduces packed weight traffic, but its
dequant/FP32-conversion/weight-only kernel cost is not automatically lower than
AMX INT8.

No W8A16 run was needed after W4A16 produced the requested CP and Talker numbers;
no CPU W4A8 path was chased. The isolated result CSV is
`results/autoround_w4a16_20260905.csv` on GCP. No repository source, Git index,
KTKernel build, or OpenVINO installation was touched.
