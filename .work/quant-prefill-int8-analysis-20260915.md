# Calibration-aware PTQ for the INT8 prefill — evidence review and redirection

Addendum to the `QUANT-PTQ` track. It answers the open hypothesis in
`.work/quantization-ptq-revisit.md` §2 and redirects the track. No code, no measurement,
no dependency change: this is a literature + own-source review that changes what the next
experiment should be.

## Task

Decide whether calibration/optimization-aware PTQ (AutoRound or equivalent) can make an
INT8 prefill quality-neutral against the production BF16 prefill, and if not, identify what
can.

## Question

`.work/quantization-ptq-revisit.md` §2 asks: "was the quality loss that killed the earlier
attempts inherent to the bit width, or an artifact of how the rounding was chosen?"

## Answer in one line

**Neither.** At 8 bits the rounding rule is worth ~0.03-0.08 accuracy points and AutoRound
loses to plain RTN on one of the two INT8 models Intel itself publishes; the bit width is
not the problem either, because our weight and activation granularity already equals the
best that INT8 hardware permits. The remaining suspect is **activation range on a small set
of prefix tokens**, which is a different defect with different, cheaper cures.

## Known facts — what the engine does today

Verified by reading the sources, not from memory.

| stage | scheme | site |
|---|---|---|
| Talker/CP weights, INT8 | symmetric, **one scale per output row**, `s = amax/127`, plain round-to-nearest, no calibration | `qwen_quantize_bf16_to_int8`, `qwen_tts_kernels.c:5796` |
| Activations, INT8 | symmetric, **per token (per column)**, `amax` over the **full K**, dynamic | `quantize_act_int8_col`, `qwen_tts_kernels.c:3686` |
| Talker weights, Q4_0 | per-32 block, fp16 scale, **weighted least-squares scale** `s* = Σw·v·q / Σw·q²`, `w = v²` | `qwen_quantize_bf16_to_q4_0`, `qwen_tts_kernels.c:6989` (`QWEN_Q4_NAIVE=1` restores absmax) |
| Prefill, production | BF16 (AMX-BF16 / AVX-512-BF16 / KleidiAI bf16), 16-token chunks | `prefill_proj_matmat`, `qwen_tts_talker.c:876` |
| Prefill, opt-in INT8 | the **same** INT8 weights as decode, activations quantized per token | `prefill_proj_matmat_i8`, `qwen_tts_talker.c:860`; `QWEN_PREFILL_INT8MM` |
| KV cache | BF16 (`uint16_t`) | `kv_cache_grow`, `qwen_tts_talker.c:283` |

`QWEN_PREFILL_INT8MM=1` covers **all six projections including `down_proj`** —
`qwen_tts_talker.c:1577-1580, 1671, 1710, 1755` (chunked prefill) and `:2050-2053, 2123,
2154, 2188` (range prefill). The Talker `lm_head` is **not** quantized by
`qwen_talker_quantize_int8` (only wq/wk/wv/wo/gate_up/down), which is the correct choice —
see the `lm_head` note in the evidence section.

So the production scheme is **W8 per-channel × A8 per-token dynamic, both symmetric,
both absmax RTN**.

## Known facts — what was already measured

From `docs/runtime-map-c8a-c4.md` (ranked interventions, item #1) and
`.work/quantization-ptq-revisit.md`:

- Admission prefill costs **108-110 ms** per admission on the c8a reference, ~1 admission/s
  per worker at C4, ~10 % of wall. Halving it is worth **-5 to -8 % soak STREAM p95**;
  scheduling alone is zero-sum (helper/priority measured -3 % STREAM for +100/+250 ms TTFA).
- `QWEN_PREFILL_INT8MM=1` measures **60 ms**, soak 0.96/1.01, TTFA -15 %,
  mel-corr vs bf16 **0.39-0.60**.
- The rejection verdict was **by ear**: "the audio stayed structurally valid — correct
  length, no artifacts, passing waveform-level checks — and still drifted audibly in
  pronunciation and speaker character."

**Correction to an earlier reading of this evidence.** The mel-corr 0.39-0.60 figure is a
weak signal on its own: measured against a BF16 run of a sampled autoregressive model, any
numeric perturbation changes the token trajectory and drives the correlation down
regardless of whether the audio is good — the same reasoning already recorded as
"non-determinism is benign". It is therefore **not** the evidence that rejected the path.
The ear verdict is, and it stands. Do not re-open this path on the argument that the metric
was wrong; the metric was merely uninformative, while the listening result was real.

## Unknowns

- No activation-range profile has ever been taken on the Talker. Nobody has published
  outlier statistics for a TTS Talker either: speech-token vocabularies, a long ICL voice
  prefix and steering-vector injection all change the activation distribution relative to a
  text LM. **This is the single largest unknown and it is cheap to close.**
- Whether the audible drift is dominated by the text prefix tokens, by `down_proj`, by the
  KV seam between an INT8-prefill prefix and BF16-prefill generated positions, or by a
  combination.
- Whether W8A8 is neutral on an acoustic-token model at 1.7B. No published evidence exists
  either way.

## Files/functions inspected

`qwen_tts_kernels.c` (`qwen_quantize_bf16_to_int8`, `quantize_act_int8_col`,
`quantize_act_int8_col_avx512`, `qwen_quantize_bf16_to_q4_0`), `qwen_tts_talker.c`
(`tk_qz`, `qwen_talker_quantize_int8`, `prefill_int8mm_enabled`, `prefill_proj_matmat`,
`prefill_proj_matmat_i8`, `prefill_proj_matmat_qkv`, `kv_cache_grow`,
`qwen_prefill_matmat_resolved`), `qwen_tts_code_predictor.c` (`qwen_cp_quantize_int8`),
`docs/quantization.md`, `docs/quant-sub4.md`, `docs/runtime-map-c8a-c4.md`,
`.work/quantization-ptq-revisit.md`, `tools/quant/fakequant_cp.py`, `tests/quant_ladder.py`.

---

# Evidence

## E1. AutoRound at 8 bits — CLOSED, negative

**What the algorithm is.** Per linear layer it learns three fp32 tensors — a per-weight-element
continuous rounding offset `V` (init 0, applied as `round_ste(W/s + V)`) and per-group clip
coefficients `α`, `β` that shrink the min/max used to derive the scale — by **SignSGD**
(`w ← w − lr·sign(g)`) on the **MSE between the quantized and FP outputs of a whole
transformer block**, with the already-quantized previous blocks feeding the input. Defaults:
`iters=200`, `lr = 1/iters = 5e-3`, `nsamples=128`, `seqlen=2048`, dataset `NeelNanda/pile-10k`,
`group_size=128`, symmetric. Cost ≈ 12 GPU-min for 8B, ≈ 2 h for 70B on an A100.
Paper: arXiv 2309.05516 (named **SignRound**; "AutoRound" is the toolkit), EMNLP 2024
Findings. Follow-up: **SignRoundV2**, arXiv 2512.04746 — adds a DeltaLoss sensitivity metric
driving per-layer bit allocation (`AutoScheme`), a llama.cpp-imatrix-inspired pre-tuning
scale search, and loss filtering.

It is mechanically distinct from the neighbours: GPTQ compensates the error of column *j*
by **updating the not-yet-quantized columns** through `H⁻¹`; AWQ **reparametrizes** the
layer with a per-input-channel scale folded away; AutoRound changes nothing but the
**round-up-or-down decision** plus clipping. That is why it composes with AWQ
(`--algs awq,signround`).

**The weight-only objection is wrong on current code, and it does not matter.** The natural
objection — "AutoRound cannot help because it never sees activation error" — is false at
HEAD: `auto_round/wrapper.py:517-545` fake-quantizes the activation **inside** the tuned
forward, so `V` is optimized against activation quantization error too, and `act_min_scale`
/ `act_max_scale` are learnable. The `INT8` scheme resolves to compressed-tensors
`int8_w8a8` with `weights: strategy="channel"` and
`input_activations: strategy="token", dynamic=True, symmetric=True`. The theoretical door
is open. The data closes it anyway:

Intel's own `docs/awq_details.md`, INT8/W8A8, 5-task average:

| model | config | AVG | time | VRAM |
|---|---|---:|---:|---:|
| Llama-3.1-8B-Instruct | BF16 | 70.42 | — | — |
| | RTN | **70.92** | 98 s | 0.85 GB |
| | AutoRound | **70.06** | 988.8 s | 13.38 GB |
| Qwen3-8B | BF16 | 72.38 | — | — |
| | RTN | 71.38 | 88 s | 0.75 GB |
| | AutoRound | **72.29** | 905.9 s | 12.37 GB |

One win, one loss, magnitude ±0.9 pt, sign unpredictable, at 10× the time and ~16× the
VRAM. Intel's own verbatim conclusion in that file: *"INT8/W8A8 is already a high-accuracy
setting in these experiments."* And `docs/fp8_block_acc.md`, 8-bit weights, RTN
(`--iters 0`) vs tuning, 12-task average: LLaMA-3-8B-Instruct **+0.0008**, Qwen3-8B
**+0.0003**.

Three structural corroborations, all in-repo:

1. `auto_round/autoround.py:282` — `if bits >= 8 and act_bits >= 8 and data_type == "int":`
   logs *"`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency"*.
   **The tool disables its own scale search at 8 bits.**
2. `export/formats/backends/gguf.py:135` — *"`iters=0` is recommended for bits>=8"*.
3. Model counts on the Hub: `Intel` org — **58** `int4-AutoRound` models, **1** `int8`; that
   one (`Intel/gemma-4-12B-it-int8-AutoRound`) is built with `--iters 0`, i.e. with
   AutoRound's tuning switched off, and publishes no accuracy table. `OPEA` — 71 int4, 0 int8.

**Neither paper evaluates 8 bits in five versions over three years.** 2309.05516v5 is
W4G-1/W4G128/W3G128/W2G128 throughout; `8-bit`/`W8`/`int8` appear only in related work.

**The headroom gradient explains why.** `docs/opt_rtn.md`, RTN → optimized-RTN average gain:
Llama-3.1-8B-I **4-bit +0.23, 3-bit +1.41, 2-bit +5.90**; Qwen3-8B +0.75 / +6.38 / **+13.10**.
No 8-bit row exists. MXFP recovery vs BF16 (`docs/auto_scheme_acc.md`) is already
**98.6 % at 6 bits**. The curve is flat before it reaches 8.

**Independent corroboration that 8-bit weight RTN is at the noise floor:**
- Dettmers & Zettlemoyer, arXiv 2212.09720 — 35,000 experiments, 19M-176B. Appendix C.3 is
  titled *"No scaling improvements for 6 to 8-bit models through quantization methods"*:
  *"we find that none of these methods improve bit-level scaling … the model parameters have
  enough precision."*
- ZeroQuant-V2, arXiv 2303.08302 — per-row RTN W8A16 is *"identical to W16A16 to 2 decimals"*,
  *"negligible accuracy loss (less than 0.05)"*; recommends **per-row W8A8 RTN for models <1B**.
- arXiv 2505.02214 (Qwen3) — at 8-bit weight-only, RTN/AWQ/GPTQ all give C4 PPL 13.3 vs
  FP16 13.3 on Qwen3-8B: *"At 8 bits, Qwen3 consistently maintains near lossless performance."*
- llama.cpp `q8_0`: LLaMA-7B F16 5.9066 → 5.9070 (Δ +0.0004), and `quantize_q8_0()` opens
  with `(void)quant_weights; // not used` — **the importance matrix is deliberately discarded
  at 8 bits**.
- GPTQ, arXiv 2210.17323 — reports no 8-bit experiments; §1: RTN *"works well for low
  compression targets, e.g., 8-bit weights."*
- QuaRot, arXiv 2404.00456 — plain RTN weight quantization *fully maintains* FP16 accuracy
  at 8 bits on Llama-2 7B/13B/70B.

**Where AutoRound disappoints even at its own target.** In Intel's own `docs/paper_acc.md`
at W4G128 it loses to AWQ/GPTQ/RTN on several models (V2-13B 60.85 vs GPTQ 61.00; V1-30B
63.20 vs AWQ 63.35; V1-65B 65.08 vs RTN 65.26). The paper concedes *"SignRound outperformed
all other methods in 83 out of 124 scenarios"* — it loses a third of its own comparisons —
and its best column "Ours*" is an **oracle over eight per-model hyperparameter configs**.
An independent non-Intel benchmark (Marie & Fujita, arXiv 2508.20893, 4 methods × 5 models ×
55 languages, COMET) finds a **2-bit collapse vs GGUF**: Qwen3-8B ja_JP **47.0 vs 83.7**,
Qwen3-1.7B ja_JP **33.1 vs 69.4**; their verdict is *"GGUF variants provide the most
consistent performance, even at 2-bit precision."* Fairness caveat: they did not use
`--enable_alg_ext` or `auto-round-best`, both of which target that regime.

**Speed is unchanged, confirmed.** At equal bits and group size AutoRound emits
byte-identical containers to GPTQ (`qweight` int32, `qzeros` int32, `scales` fp16, optional
`g_idx`), and the inference backend table dispatches on **packing format**, not on the
producing algorithm. The paper's own abstract claims *"avoiding additional inference
overhead."* **The speedup we hoped to buy does not exist to be bought: the only speed lever
is BF16 → INT8, which we already hold and already measured at 1.8×.**

## E2. Our granularity is already the state of the art for INT8

AutoRound's `INT8` scheme is per-channel weights + per-token dynamic activations. **That is
our scheme.** It is also the recipe behind the strongest published W8A8 result: arXiv
2411.02355 (>500k evaluations, full Llama-3.1 family) uses *dynamic per-token activation
quantization + symmetric per-channel weights via GPTQ* and recovers **99.75 % on average**,
*"1-3 % per task degradation, far lower than the 10 %+ drops reported in prior work"*.

This is forced by hardware, not chosen. `Σ_k (s_k·qa_k)(sw·qw_k) = sw·Σ_k s_k·qa_k·qw_k` —
a factor varying with `k` cannot leave an int32 accumulator. None of `VPDPBUSD`, `TDPB**D`,
`SDOT` or `SMMLA` takes a scale operand. **The finest granularity that preserves one exact
integer reduction is any scale constant along K: per-tensor, per-token for activations,
per-output-channel for weights. Nothing finer exists on any CPU**, and block-scaled integer
MAC does not exist in hardware anywhere (NVIDIA's `mma.kind::mxf*` block-scaled forms have
no int8 entry; Intel ISA ref. 319433-060 has zero hits for microscaling; Arm ACLE has none).

So the premise "our method is simpler and loses more data" is **false on granularity** and
**true but immaterial on scale selection**: our INT8 uses absmax where llama.cpp uses a
19-candidate search with a weighted-LS refit and our own Q4 uses weighted LS — but that
lever is worth ~0.03-0.08 pt at 8 bits (E1). **Port the Q4 LSQ scale to INT8 only as a
30-minute falsification, not as a project.** It is worth a lot at 2-4 bits and nothing here.

## E3. The speed ceiling: 2.0×, and we are already at it

| path | int8 : bf16 | source |
|---|---:|---|
| Arm N2 / V1 / V2 (Graviton4), `SMMLA` vs `BFMMLA` | **2.0×** | Neoverse SWOGs; 32 vs 16 MACs/instr, same TP |
| AMX-INT8 vs AMX-BF16 | **2.0×** | 16384 vs 8192 MACs/instr, both recip-TP 16; Intel 353107-001US: 2048 vs 1024 ops/cycle |
| Zen 4 / Zen 5 | **2.0×** | uops.info `VPDPBUSD` vs `VDPBF16PS` |
| Apple M4 SME, single P-core | **2.0×** | arXiv 2409.18779: SMOPA int8 4017 GOPS vs BFMOPA 2010 |

Our measured 108→60 ms is **1.8×**. There is no second speedup hiding behind a better
quantizer.

Two facts worth carrying:
- The "8×" figure sometimes quoted for Intel SPR/EMR is an artefact: `VDPBF16PS` is
  **4 µops at reciprocal throughput 2** on those cores = 16 MAC/cycle, i.e. **half the plain
  fp32 FMA rate**. Vector BF16 is a compute *loss* there; Intel's BF16 answer on SPR+ is AMX.
  Against fp32 FMA the honest VNNI ceiling is 4× theoretical, ~2.9× measured L1-resident.
- **M1 has neither i8mm nor BF16** (measured: `FEAT_I8MM 0`, `FEAT_BF16 0`, `FEAT_DotProd 1`;
  the cutoff is M1→M2, not M3→M4). Any `smmla`/`bfmmla` reasoning is Linux-box-only,
  consistent with the existing "Linux box first" rule.

## E4. The redirected hypothesis: activation range on prefix tokens

With the rounding rule (E1) and the granularity (E2) both excluded, and the ear verdict
standing, the remaining candidate is the **range** of specific activations.

**Mechanism.** arXiv 2405.14428 (*Mitigating Quantization Errors Due to Activation Spikes in
GLU-Based LLMs*) — our Talker is SwiGLU — localizes extreme magnitudes to the **input of
`down_proj`** (the output of `SiLU(gate)·up`), in **early and late layers**, concentrated on
a handful of tokens: **BOS, newline, apostrophe**. Non-GLU models (OPT, Pythia, Falcon, MPT)
do not show this. Their W8A8 per-tensor numbers: Llama-2-7B 69.61 → 62.08,
**Llama-2-13B 72.15 → 55.29**, Llama-2-70B 76.65 → 66.87. Related but distinct: arXiv
2402.17762 (*Massive Activations*) documents input-agnostic dimensions up to ~10⁵× the
median on BOS/delimiter tokens, acting as implicit attention bias, **present across model
sizes including small ones**.

**Why this splits prefill from decode in our engine.** Those spike-carrying tokens — BOS,
newline, apostrophe — live in the **text prefix**, i.e. exactly what the prefill processes.
The decode emits acoustic codes and never sees them. The asymmetry is therefore not
"prefill vs decode" as a numerical regime; it is **which tokens each path carries**.

**Why per-token scaling, normally the defence, works against us here.** A per-token absmax
is exactly the right tool when the outlier is a *token*. It is the wrong tool when the
outlier is a *channel within* a token: one channel at 100× consumes the range and leaves the
other ~6000 channels of that token at ~4 effective bits. And the token this happens to is
the attention sink — the highest-attention token in the sequence.

**This is also the one place the counter-argument does not reach.** arXiv 2605.20315
(*Mix-Quant: Quantized Prefilling, Precise Decoding*) argues prefill is the structurally
*safe* phase — *"During prefilling, the input context is fixed. Quantization errors may
affect the hidden states and the constructed KV cache, but they do not change the input
tokens being processed"*, whereas in decode *"a different sampled token … all future
predictions are conditioned on a different history"* — and runs prefill at **NVFP4 (4 bits)**
with BF16 decode for up to 3× prefill speedup at near-BF16 quality. Their defence against
KV poisoning is attention concentration (top 3.125 % of tokens carry 95.8 % of attention
mass, so errors on low-attention tokens are attenuated). **That defence does not cover the
sink token, which is the one the spike damages.** Their workload is also our inverse: long
prompts with short reasoning output, versus our short text prefix with very long acoustic
generation and a possible ICL voice prefix that the whole generation conditions on tightly.

**Note that this supersedes an earlier guess.** A "prefill error poisons the BF16 KV cache,
so prefill is inherently more sensitive than decode" hypothesis was considered and is
**not** supported: Mix-Quant addresses it directly and empirically rejects it, and our KV
stores BF16 values whose relative error is orders of magnitude smaller than what the
KV-quantization literature studies. Keep KV-in-BF16 as a cheap control arm, not as the
explanation.

**Scale context for our size class — contested, and worth stating honestly:**
- For, at our scale: SLMQuant (arXiv 2511.13023) measures **<0.5 %** at W8A8 for
  SmolLM-135M and Qwen2.5-0.5B; ZeroQuant-V2 finds activation quantization *easier* on
  smaller models and recommends per-row W8A8 RTN below 1B; LLM.int8() (arXiv 2208.07339)
  places emergent outlier coverage at a phase transition around 6.7B.
- Against, and it is about our exact family: arXiv 2505.02214 measures **Qwen3-1.7B**
  SmoothQuant W8A8 at Wiki2 PPL **9.39 → 9.65 (+2.8 %)** and C4 **13.4 → 13.8 (+3.0 %)**,
  concluding *"even under the w8a8 setting, there is already a noticeable degradation."*
  **Confound:** that study used SmoothQuant's default, per-tensor-leaning configuration,
  not the per-token-dynamic recipe we already run. Compare the OPT-1.3B row of LLM.int8():
  per-tensor absmax **+0.64 PPL**, vector-wise (per-row × per-column) **+0.02**. The gap
  between "+3 %" and "neutral" is very plausibly the granularity gap — testable on our bench.
- Unknown: no published outlier statistics or W8A8 result for a TTS Talker.

**Digit caveat:** the tables from 2505.02214, 2605.20315, 2405.14428 and 2411.02355 were
extracted automatically from HTML/PDF. Quoted sentences are verified; re-derive exact digits
from the papers before quoting them in a commit message or a release doc.

## E5. The measurement instrument is wrong, independently of the verdict

arXiv 2407.09141 (*Accuracy is Not All You Need*): across 6 quantization schemes, aggregate
accuracy vs BF16 is statistically indistinguishable on 7 tasks **while all schemes except
GPTQ W8A16 produce large numbers of "flips"** (answers changing correct↔incorrect, cancelling
in the mean). On perplexity specifically: it is the inverse geometric mean of token
probabilities, so **lower probability on some tokens is cancelled by higher probability on
others** — structurally blind to the damage that matters. They advocate **distance metrics:
KL-divergence and flip rate** against the FP16 model, on the grounds that a quantized model's
job is to be a drop-in replacement.

Corroborating: arXiv 2411.02355 measures text similarity to the BF16 output at **ROUGE-1 ≈
0.62 at 8B** under W8A8-INT, i.e. 99 %+ task recovery with a materially different token
stream. For a Talker whose token stream *is* the product, that distinction is the whole game.
arXiv 2508.20893 (PTQ in machine translation — the closest generative analogue to TTS) finds
perplexity-stable models still lose measurable COMET/BLEU, **unevenly across language pairs**
— directly relevant to our multilingual voices: a quantization neutral on English may not be
on Italian or Japanese, so test per language.

Also: arXiv 2505.10938 warns that conventional evaluation on short decodes gives quantization
error *"little opportunity to compound along the sequence axis"* — our
`-j1 --temperature 0 --seed 42` golden gate on short utterances is exactly that regime.

## E6. What the alternatives cost

- **SmoothQuant** (arXiv 2211.10438): migrate per-channel activation range into the weights,
  `s_j = max|X_j|^α / max|W_j|^(1−α)`, α≈0.5. Mathematically exact. The paper states the
  factor is fused into the **previous layer's parameters offline**, *"which does not incur
  kernel call overhead"*; the named exception is a residual-add input. For a Qwen-shaped
  block the `q/k/v` and `gate/up` scales fold into the preceding **RMSNorm weight vector**.
  *Derived, verify against our graph before relying on it:* the `o_proj` input scale should
  fold into `v_proj`'s output channels and the `down_proj` input scale into `up_proj`'s,
  because a diagonal scale commutes with the elementwise SiLU-gate product and with head-wise
  mixing. Granularity grades: **O1** per-token dynamic activations (what we run), O2
  per-tensor dynamic, **O3** per-tensor static — O3 costs **-0.8 %** average accuracy on
  OPT-175B even *with* smoothing. Needs calibration data; no runtime cost if it folds.
- **QFeM / QFeP** (arXiv 2405.14428): QFeM leaves the few spike-carrying linear layers
  unquantized, selected by a max/median ratio; QFeP precomputes a short prefix containing the
  spike tokens and holds its KV in FP16. Both are static decisions, both implementable in C,
  both far friendlier to AMX/VNNI tiling than dynamic column decomposition.
- **LLM.int8() mixed-precision decomposition**: recovers exactly, but the scatter/gather plus
  a ragged FP16 sub-GEMM breaks AMX/VNNI tiling. **Worst option for a hand-written kernel**,
  and per the 1.3B row we should not need it.
- **Rotations (QuaRot arXiv 2404.00456, SpinQuant arXiv 2405.16406)**: a 4-bit tool. QuaRot
  itself reports RTN is already lossless at 8 bits; SpinQuant notes the Hadamard choice
  matters less at W8A8. The online Hadamards cost **up to ~7 %** of the forward pass. The one
  exception worth remembering is the online Hadamard **before `down_proj`** — precisely the
  GLU spike site — which is a 1-D butterfly implementable in NEON/AVX-512 if E4's targeted
  fixes fail.
- **Per-K-block activation quantization** (blocks of 32 or 128 instead of whole-K per token):
  this is literally `q8_0` (32 int8 + one fp16 scale = 8.5 bpw). An outlier is confined to its
  own block of 32 rather than crushing the whole row. Overhead is O(1/B) — one int32→float
  convert, one FMA and one fp16→fp32 per 32 elements — and B=32 is where everyone converged
  (OCP MX, ggml `q8_0`/`q8_K`, KleidiAI `c32`/`d32`, oneDNN `groups={32,1}`). The literature
  frames finer-than-per-token at 8 bits as buying *robustness, not accuracy* — and robustness
  against outliers is exactly our defect. **No calibration data required**, which makes it
  cheaper to try than SmoothQuant.

## E7. Deployment reality, if AutoRound were ever wanted anyway

- **ARK** (`auto-round-lib`, the BesTLA-backed CPU kernels covering AMX-INT8 and AVX512-VNNI)
  ships **`manylinux_2_28_x86_64` wheels only** — no aarch64, no macOS, no Windows.
- **vLLM CPU** accepts AutoRound only by rewriting its quant method to `inc`, and the CPU
  dispatch is **W4, symmetric, GPTQ packing only**; W8A16 raises `NotImplementedError` on CPU
  despite the docs, and quantized MoE silently falls back to unquantized. The vLLM
  `auto_round.md` page (the v0.14.0 URL that started this investigation) has been **deleted
  from main** and folded into `inc.md`; its *"AutoRound applies weight-only quantization"*
  sentence is stale.
- **IPEX is archived** (2026-03-30) and auto-round **removed its IPEX backends at v0.13.0**.
- **ONNX Runtime: no path at all** — zero `onnx` paths in the repo.
- **Arm: GGUF → llama.cpp only.** 15 GGUF aliases over 11 real block types
  (`bf16,q4_0,q4_1,q5_0,q5_1,q8_0,q2_k,q3_k,q4_k,q5_k,q6_k`); **no `iq*` types at all**.
  Recommended recipe for GGUF is `--iters 0` except at 3 bits.
- The checkpoint itself is **plain GPTQ layout** (`qweight` int32 packed along in-features
  LSB-first, `qzeros` int32, `scales` fp16, optional `g_idx`) and a hand-written C loader can
  read it. Two traps: the `auto_round:auto_gptq` packing uses a **"zp−1"** convention, and
  nibble order differs across ecosystems — **ggml/KleidiAI pack element `j` and `j+16`
  together (distance 16)**, GPTQ/compressed-tensors pack adjacent elements, AWQ permutes
  `[0,2,4,6,1,3,5,7]`. Getting this wrong produces plausible-looking garbage.

## E8. Two findings that belong to other tracks

- **auto-round ships a Qwen3-TTS GGUF converter**:
  `auto_round/export/export_to_gguf/conversion/qwen3tts.py`, registered for
  `Qwen3TTSForConditionalGeneration` with `Qwen/Qwen3-TTS-12Hz-1.7B-Base` as the example.
  Its comments document a llama.cpp mapping: text-projection MLP folded into the embedding
  table, `codec_embedding` concatenated onto the text embedding with vocab extension
  (`codec_bos_id 2149`, `codec_eos_token_id 2150`), `code_predictor` →
  `MTMD_GEN_PROCESS_TYPE_GEN_CODE`, `code2wav` → `…GEN_WAV`. Worth reading regardless of the
  quantization verdict — it is an independent reading of our model's structure.
- **`ggml_quantize_chunk()` is a linkable C API**:
  `size_t ggml_quantize_chunk(enum ggml_type, const float *src, void *dst, int64_t start,
  int64_t nrows, int64_t n_per_row, const float *imatrix)`. One can link `ggml-quants.c`
  offline, pass an imatrix vector (mean `a²` per input column) and receive `q4_K`/`iq4_xs`/
  `q6_K` blocks in the exact documented layouts, then write kernels against those structs —
  no Python, no GGUF parsing. `ggml`'s `imatrix` is the **diagonal of the activation second
  moment** (`Σ_tokens a_j²`, off-diagonals deliberately discarded, unlike GPTQ's full `XᵀX`);
  measured gains on Mistral-7B PPL: Q2_K 6.7580 → 6.4268, Q4_K_S 5.7764 → 5.7428,
  Q6_K 5.7028 → 5.7011 — i.e. **0.2-0.8 % at 4-5 bpw, ~5 QErr points at 2-bit, and a
  precondition below 3 bpw.** Note again that `q8_0` ignores it.

---

# Ideas backlog

Ordered by recovered-quality per unit of cost. None started.

| id | idea | cost | why it is here |
|---|---|---|---|
| **QP-1** | **Profile the activation range** before anything else: per linear layer, per token position, `max/median` ratio, across languages, with and without a voice prefix, prefix vs generated positions. ~20 lines of C behind a flag. | hours | Closes the largest unknown. If `down_proj` inputs show the GLU signature on BOS/newline/apostrophe, E4 is confirmed and the cure is known. If they do not, E4 is dead and the track needs a new hypothesis. |
| **QP-2** | **Distance gate**: teacher-forced `KL(bf16 ‖ int8)` per decode step plus flip rate, on the BF16 token stream, on **long** utterances, **per language**, at temperature > 0. Reuse `tools/quant/fakequant_cp.py` + `tests/quant_ladder.py`. | 1 day | E5. Isolates prefill damage from trajectory divergence, which no end-to-end audio metric can do. Needed *before* any candidate, so that a verdict means something. Does **not** replace the ear gate (§4 of the parent note) — it makes ear time cheap by filtering candidates. |
| **QP-3** | **Per-K-block activation quantization in the prefill** (B=32, then 128). No calibration, no new format, reuses the per-32 machinery already written for Q4_0. | 1-2 days | E6. The targeted cure for E4 if the spike is channel-within-token. Cheapest real candidate. |
| **QP-4** | **QFeP: first N prefix tokens in BF16, INT8 from there.** N small, cost negligible. | 1 day | E6. Removes the spike tokens by construction; also removes the sink-token damage Mix-Quant's defence does not cover. |
| **QP-5** | **QFeM: exclude the 1-3 worst layers** (selected by QP-1's max/median ratio) from INT8 prefill, `down_proj` first. | 1 day | E6. Static, AMX/VNNI-friendly escape hatch. |
| **QP-6** | **SmoothQuant α-sweep folded into RMSNorm / `v_proj` / `up_proj`.** Verify foldability against our block graph first. | 1 week + calibration harness | E6. Free at runtime if it folds, but it is the first idea that needs calibration data — do it only if QP-3/4/5 fall short. |
| **QP-7** | **KV seam control arm**: INT8 prefill for everything except the K/V projections. | hours | Demoted from a hypothesis to a control (E4). Cheap enough to run inside the QP-3 A/B. |
| **QP-8** | **`iq4_nl` revisit** — belongs to the INT4 track, not this one. Same 4.5 bpw and the same 18-byte block as our Q4_0, a 16-entry LUT `{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}`; QErr 1.10 % vs Q4_0's 1.84 % on LLaMA-v2-7B. Our own memory records **+8.8 pt on the CP with the kernel parked**. AutoRound **cannot emit it**; llama.cpp can. | kernel only | `docs/quant-sub4.md` §5 already names this as the cheap follow-up outside that epic's scope. |
| **QP-9** | **Offline quantizer via `ggml_quantize_chunk()`** if the INT4 track ever restarts: link `ggml-quants.c`, feed an imatrix, consume the blocks in our own kernels. | days | E8. Avoids building a calibration pipeline from scratch. |

## Explicitly rejected — do not re-open without new evidence

| rejected | why | evidence |
|---|---|---|
| **AutoRound / SignRound at INT8** | no headroom; loses to RTN on 1 of 2 of Intel's own INT8 models; tool disables its own optimizations at `bits>=8`; papers never evaluate 8 bits; zero speed change | E1 |
| **Porting the Q4 weighted-LSQ scale to the INT8 weight quantizer as a project** | ~0.03-0.08 pt expected at 8 bits. Run it once as a 30-minute falsification; do not plan around it. It remains valuable at 2-4 bits. | E1, E2 |
| **Rotations (QuaRot / SpinQuant) at 8 bits** | 4-bit tools; ~7 % runtime cost for a problem per-token scaling already solves. Exception: the online Hadamard before `down_proj`, only if QP-3/4/5 fail. | E6 |
| **LLM.int8()-style dynamic column decomposition** | breaks AMX/VNNI tiling; QFeM is the implementation-friendly equivalent | E6 |
| **Expecting more than 2× from BF16 → INT8 prefill** | architectural ceiling on every ISA we run; we measure 1.8× already | E3 |
| **Re-opening the INT8-prefill rejection on the grounds that mel-corr was the wrong metric** | the metric was uninformative, but the rejection was an ear verdict and it stands | Known facts |
| **Quantizing the Talker `lm_head`** | rounding error in the logit projection changes *which acoustic token is sampled*; flagged explicitly in Red Hat's W8A8 guide. Currently not quantized — keep it that way. | E4 |

---

# Conclusion

1. `.work/quantization-ptq-revisit.md` §2 is **answered: NO** for AutoRound-style rounding
   optimization at 8 bits. The hypothesis was reasonable and is now closed by Intel's own
   measurements plus four independent corroborations. It remains **open and promising at
   2-4 bits**, which is a different track.
2. The parent note's core distinction survives and is reinforced: the earlier rejections
   falsify **implementations**, not the direction. But the implementation defect is not the
   one that note assumed. It is **not** the rounding rule and **not** the bit width — both
   are excluded. The live candidate is **activation range on a small set of prefix tokens**,
   with a specific site (`down_proj` input), a specific token class (BOS, newline,
   apostrophe), a specific mechanism (per-token absmax crushed by a channel outlier on the
   attention-sink token), and three cheap cures that need no calibration.
3. The prize is unchanged and real: the prefill is **intervention #1** of the c8a cost map
   (-5 to -8 % soak p95), and halving its bytes also halves what admission steals from
   playback-critical streams — the same mechanism as the Graviton5 contention post-mortem.
4. The gate in §4 of the parent note is correct and must not be weakened. QP-2 does not
   replace it; it makes it affordable by filtering candidates before they reach a listener.

# Next action

Run **QP-1** and **QP-2** — in that order, both diagnostic, neither touching a default. Do
not start QP-3/4/5 before QP-1 says which site and which tokens carry the range, and do not
start QP-6 at all until the no-calibration options have been measured. Keep the track behind
the C12-WIN items and the report qualification work, as the parent note specifies.

# References

**Method / algorithm**
- AutoRound / SignRound — arXiv 2309.05516 · EMNLP 2024 Findings <https://aclanthology.org/2024.findings-emnlp.662/> · repo <https://github.com/intel/auto-round>
- SignRoundV2 — arXiv 2512.04746
- GPTQ — arXiv 2210.17323 · AWQ — arXiv 2306.00978 · SmoothQuant — arXiv 2211.10438 (PMLR v202)
- QuaRot — arXiv 2404.00456 · SpinQuant — arXiv 2405.16406 · OmniQuant — arXiv 2308.13137
- Atom — arXiv 2310.19102 · QServe/QoQ — arXiv 2405.04532 · ZeroQuant-V2 — arXiv 2303.08302

**Why 8-bit has no headroom**
- Dettmers & Zettlemoyer, *The case for 4-bit precision* — arXiv 2212.09720, Appendix C.3
- LLM.int8() — arXiv 2208.07339
- Intel auto-round in-repo tables: `docs/awq_details.md` (INT8/W8A8), `docs/fp8_block_acc.md`, `docs/opt_rtn.md`, `docs/auto_scheme_acc.md`, `docs/paper_acc.md`

**Outliers and the GLU spike**
- Activation spikes in GLU-based LLMs (QFeM / QFeP) — arXiv 2405.14428
- Massive Activations in LLMs — arXiv 2402.17762
- Outliers and calibration sets have diminishing effect — arXiv 2405.20835

**Scale and family evidence**
- An Empirical Study of Qwen3 Quantization — arXiv 2505.02214
- "Give Me BF16 or Give Me Death?" — arXiv 2411.02355 (ACL 2025)
- SLMQuant — arXiv 2511.13023 · Evaluating Quantized LLMs — arXiv 2402.18158

**Prefill vs decode, KV**
- Mix-Quant: Quantized Prefilling, Precise Decoding — arXiv 2605.20315
- KVQuant — arXiv 2401.18079 · KVLinC — arXiv 2510.05373 · Outlier Tokens Tracing — arXiv 2505.10938
- Sarathi (chunked prefill) — arXiv 2308.16369

**Metrics**
- Accuracy is Not All You Need — arXiv 2407.09141
- The Uneven Impact of PTQ in Machine Translation — arXiv 2508.20893 (also the independent AutoRound 2-bit benchmark)

**TTS / autoregressive**
- BitTTS — arXiv 2506.03515 · RALL-E — arXiv 2404.03204 · Hallucination in LM-based TTS — arXiv 2508.15442

**Formats, kernels, hardware**
- llama.cpp `ggml/src/ggml-quants.c`, `ggml-common.h`; imatrix PRs #4861, #4897, #4930, #4969, #5590, #9400, #14842; KLD discussion #5263
- `iq4_nl` — llama.cpp PR #5590
- KleidiAI microkernel naming <https://github.com/ARM-software/kleidiai/blob/main/docs/microkernel_names.md>
- oneDNN quantization attributes; int8 computations guide (s8s8 saturation workaround)
- gemmlowp `doc/low-precision.md` · Jacob et al. — arXiv 1712.05877 · Nagel et al. — arXiv 2106.08295
- Intel AMX — 353107-001US; Optimization Manual 248966-049 §20.5.2; ISA ext. ref. 319433-054 / 319433-060
- Neoverse N2 / V2 Software Optimization Guides; Apple M4 SME — arXiv 2409.18779
- uops.info (`VPDPBUSD`, `VDPBF16PS`, `TDP*`)
- Red Hat, *Understanding W8A8 INT8 LLM quantization* <https://developers.redhat.com/articles/2026/09/07/understanding-w8a8-int8-llm-quantization>

**Provenance.** Compiled 2026-09-15 from three parallel literature/source reviews plus a
direct read of this repository's quantization and prefill sources. Where a table was
extracted automatically from a paper's HTML/PDF it is flagged in E4; quoted sentences were
verified against the source. AutoRound claims marked as in-repo were read at
`intel/auto-round` HEAD `f331bb0` (version 0.16.0, latest tag v0.15.1).
