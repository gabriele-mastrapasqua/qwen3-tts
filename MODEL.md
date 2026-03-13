# Qwen3-TTS — Model Reference

Models: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` and `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

This document describes the model architecture, weight format, tokenizer layout,
and inference algorithm needed to implement Qwen3-TTS from scratch.

---

## Architecture Overview

Qwen3-TTS is a text-to-speech model with four main components:
- **Talker (LLM)**: Qwen3-based autoregressive transformer predicting semantic audio tokens
- **Code Predictor (MTP)**: Small transformer generating residual codebook tokens
- **Speech Tokenizer Decoder**: Causal ConvNet converting discrete tokens to waveform
- **Speaker Encoding**: Speakers are represented as codec vocabulary tokens (e.g., serena=3066), not separate embeddings

**Pipeline:**
```
Text → BPE Tokenizer → Talker (28 layers) → Code Predictor (5 layers × 15 passes) → 16 discrete codes/frame → ConvNet Decoder (480× upsample) → 24 kHz WAV
```

### Model Variants

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| **Text hidden_size** | 2048 | 2048 |
| **Talker hidden_size** | 2048 | 1024 |
| **Talker layers** | 28 | 28 |
| **Talker heads** | 16 | 16 |
| **Talker KV heads** | 8 (GQA 2:1) | 8 (GQA 2:1) |
| **Talker head_dim** | 128 | 128 |
| **Talker q_dim** | 2048 (num_heads × head_dim) | 2048 (num_heads × head_dim) |
| **Talker intermediate** | 6144 | 3072 |
| **Text projection** | SiLU: 2048→2048→1024 | SiLU: 2048→2048→1024 |
| **Code Predictor hidden** | 1024 | 1024 |
| **Code Predictor layers** | 5 | 5 |
| **Code Predictor intermediate** | 3072 | 3072 |
| **Text vocab size** | 151,936 | 151,936 |
| **Codec vocab size** | 3,072 | 3,072 |
| **Codebooks** | 16 | 16 |
| **Codebook entries** | 2,048 | 2,048 |
| **Codebook dim** | 256 | 256 |
| **Frame rate** | 12.5 Hz | 12.5 Hz |
| **Output sample rate** | 24,000 Hz | 24,000 Hz |

---

## Speech Tokenizer (Qwen3-TTS-Tokenizer-12Hz)

The audio codec uses 16-layer Residual Vector Quantization (RVQ):

| Parameter | Value |
|-----------|-------|
| Frame rate | 12.5 Hz (80 ms per frame) |
| Codebooks | 16 layers |
| Entries per codebook | 2,048 |
| Entry dimension | 256 |
| Bitrate | ~2.2 kbps |
| Output sample rate | 24 kHz |

**Codebook 0** captures semantic content (trained with WavLM teacher alignment).
**Codebooks 1-15** capture acoustic detail, prosody, and speaker characteristics.

The decoder is a causal ConvNet (no lookahead), enabling streaming synthesis.

### Decoder Architecture

All speech tokenizer weights are **F32** (not BF16).

**Codebook tensors:**
```
decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum  [2048, 256]  (codebook 0)
decoder.quantizer.rvq_rest.vq.layers.{0..14}._codebook.embedding_sum  [2048, 256]  (codebooks 1-15)
```

**Pre-transformer:**
```
decoder.pre_transformer.layers.{0..7}.*   (8 layers with input_proj/output_proj)
```

**Decoder blocks** (with Snake activation using alpha/beta parameters):
```
decoder.decoder.{0..6}.*   (7 blocks: ConvTranspose1d + residual blocks with Snake activation)
```

**Upsample** (ConvNeXt blocks):
```
decoder.upsample.{0,1}.*   (ConvNeXt upsampling blocks)
```

```
16 codes per frame
    ↓
Codebook embedding lookup: 1 first + 15 rest tables × [2048, 256]
RVQ sum: sum all 16 embeddings → [N_frames, 256]
    ↓
Pre-transform: 8 transformer layers (self-attention + FFN, with input_proj/output_proj)
    ↓
Decoder blocks: 7 blocks with ConvTranspose1d + Snake activation (alpha/beta)
    Total: 480× upsampling
    ↓
Upsample: 2 ConvNeXt blocks
    ↓
Tanh activation → float samples in [-1, 1]
    ↓
24 kHz audio waveform
```

---

## Talker (LLM Backbone)

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| text_hidden_size | 2048 | 2048 |
| hidden_size | 2048 | 1024 |
| n_layers | 28 | 28 |
| n_heads | 16 | 16 |
| n_kv_heads | 8 (GQA 2:1) | 8 (GQA 2:1) |
| head_dim | 128 (NOT hidden_size/num_heads) | 128 (NOT hidden_size/num_heads) |
| q_dim (n_heads × head_dim) | 2048 | 2048 |
| intermediate_size | 6144 | 3072 |
| Norm | RMSNorm (eps=1e-6) | RMSNorm (eps=1e-6) |
| Position | M-RoPE (theta=1e6, interleaved) | M-RoPE (theta=1e6, interleaved) |
| mrope_section | [24, 20, 20] | [24, 20, 20] |
| Attention | causal | causal |
| Biases | NO (none in Talker) | NO |
| Activation | SwiGLU (SiLU) | SwiGLU (SiLU) |
| Text projection | SiLU: fc1 [2048→2048] → fc2 [2048→1024] | SiLU: fc1 [2048→2048] → fc2 [2048→1024] |

### Key feature: Q/K RMSNorm

Same as Qwen3-ASR decoder: per-head RMSNorm on Q and K after projection,
before RoPE:
```python
q = q_proj(h_norm)                          # [seq, n_heads * head_dim]
q = q.view(seq, n_heads, head_dim)          # [seq, 16, 128]
q = RMSNorm_per_head(q, q_norm_weight)      # normalize each head independently
# Then apply RoPE
```

The `q_norm` and `k_norm` weights have shape `[head_dim]` = `[128]`.

### RoPE (Interleaved style, NOT NeoX split-half)

**Important:** This model uses **interleaved RoPE** (also known as the original
rotary embedding style), NOT the NeoX/split-half style.

**M-RoPE** with `mrope_section=[24, 20, 20]` and `interleaved=true`:
```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))  # [64]
# mrope_section = [24, 20, 20] → split inv_freq into 3 sections
# section 0: dims 0..23  (24 pairs → 48 dims)  ← temporal position
# section 1: dims 24..43 (20 pairs → 40 dims)  ← spatial dim 1
# section 2: dims 44..63 (20 pairs → 40 dims)  ← spatial dim 2

# For TTS (text-only input), all three position dimensions are identical,
# so M-RoPE reduces to standard RoPE with the same position for all sections.

angles = positions * inv_freq  # [seq, 64]

# Interleaved rotation (NOT NeoX split-half):
# For each pair (x[..., 2i], x[..., 2i+1]):
#   result[..., 2i]   = x[..., 2i]   * cos[i] - x[..., 2i+1] * sin[i]
#   result[..., 2i+1] = x[..., 2i+1] * cos[i] + x[..., 2i]   * sin[i]
```

**Key difference from NeoX:** NeoX splits the vector in half `[x1, x2] = x[:64], x[64:]`
and rotates across halves. Interleaved rotates adjacent pairs `(x[0], x[1]), (x[2], x[3]), ...`.
Using the wrong style will produce completely incorrect results.

### Talker Forward Pass

**Text embedding and projection (before Talker layers):**
1. `text_emb = text_embedding_table[token_ids]` → `[seq, 2048]` (text_hidden_size)
2. `projected = SiLU(text_emb @ fc1^T + bias1) @ fc2^T + bias2` → `[seq, 1024]` (hidden_size)

Per-layer computation for hidden state `h` (dim=1024 for 0.6B):

1. **Input RMSNorm**: `x = RMSNorm(h, input_layernorm, eps=1e-6)` → `[seq, 1024]`
2. **QKV projections (GQA)**:
   - `q = x @ Wq^T` → `[seq, 2048]` → reshape `[seq, 16, 128]` (q_dim=2048 != hidden_size=1024)
   - `k = x @ Wk^T` → `[seq, 1024]` → reshape `[seq, 8, 128]`
   - `v = x @ Wv^T` → `[seq, 1024]` → reshape `[seq, 8, 128]`
3. **Per-head Q/K RMSNorm** (eps=1e-6, weight shape [128])
4. **RoPE** on Q and K (interleaved style, theta=1e6, M-RoPE sections [24,20,20])
5. **KV cache**: append K, V to per-layer cache
6. **Causal attention**: scale=1/sqrt(128), GQA repeat 2:1
7. **Output projection + residual**: `h = h + attn_out @ Wo^T` (Wo: [1024, 2048])
8. **Post-attention RMSNorm**: `h_norm = RMSNorm(h, post_attention_layernorm)`
9. **SwiGLU MLP + residual**:
   - `gate = silu(h_norm @ W_gate^T)`
   - `up = h_norm @ W_up^T`
   - `h = h + (gate * up) @ W_down^T`

After last layer: `h = RMSNorm(h, talker.model.norm.weight)`, then logits via `talker.codec_head`.

### Talker Output

The Talker produces tokens from the **codec vocabulary** (size 3,072), not the
text vocabulary. Each generated token is a codebook-0 (semantic) code for one
audio frame at 12.5 Hz.

**Special token IDs:**

Text-side tokens (in the text vocabulary 0-151,935):
```
tts_bos_token_id  = 151672   (beginning of speech — TEXT token ID)
tts_eos_token_id  = 151673   (end of speech — TEXT token ID)
```

Codec-side tokens (in the codec vocabulary 0-3,071):
```
codec_bos_id  = 2149   (beginning of codec generation)
codec_eos_id  = 2150   (end of codec generation)
codec_pad     = 2148   (padding)
```

Generation stops when `codec_eos_id` (2150) is produced.

---

## Code Predictor (MTP Module)

| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| n_layers | 5 |
| n_heads | 16 |
| n_kv_heads | 8 |
| head_dim | 128 |
| intermediate_size | 3072 |
| Norm | RMSNorm (eps=1e-6) |
| Activation | SwiGLU |
| Vocab size (per group) | 2,048 |
| Num heads (codebooks 1-15) | 15 |

For each audio frame, after the Talker produces codebook-0:
1. The Code Predictor (`talker.code_predictor.model.layers.{0..4}`) receives the Talker's hidden states
2. It runs **15 sequential forward passes** (one per remaining codebook)
3. Each pass generates one codebook token via `talker.code_predictor.lm_head.{g}` (codebooks 1-15, g=0..14)
4. Each pass is conditioned on previously generated codebook tokens via `talker.code_predictor.model.codec_embedding.{g}`

This produces the full 16-code vector per frame.

---

## Tokenizer (Qwen BPE)

### Text Token Special IDs
```
<|im_start|>    = 151644
<|im_end|>      = 151645
<|endoftext|>   = 151643
```

### TTS Special Token IDs (text vocabulary)
```
tts_bos_token_id = 151672   (beginning of speech — text-side)
tts_eos_token_id = 151673   (end of speech — text-side)
```

### Codec Control Token IDs (within codec vocab 0-3071)
```
codec_pad     = 2148
codec_bos_id  = 2149   (beginning of codec generation)
codec_eos_id  = 2150   (end of codec generation)
think         = 2154
no_think      = 2155
```

### Language IDs (within codec vocab)
```
Chinese    = 2055
English    = 2050
Japanese   = 2058
Korean     = 2064
German     = 2053
French     = 2061
Russian    = 2069
Portuguese = 2071
Spanish    = 2054
Italian    = 2070
```

### Token Encoding

Uses GPT-2 style byte-level BPE from `vocab.json`. The vocabulary maps
byte-encoded strings to token IDs. Characters are encoded using the GPT-2
bytes-to-unicode mapping.

To encode: convert UTF-8 text to byte sequences → map through GPT-2 byte
mapping → BPE merge → token IDs.

---

## Prompt Format

The prompt template for TTS (CustomVoice):
```
<|im_start|>assistant\n{text_to_synthesize}<|im_end|>\n
```

As token IDs:
```
PREFIX:  [151644, 77091, 198]           # <|im_start|>assistant\n
TEXT:    [... BPE-encoded text tokens ...]
SUFFIX:  [151645, 198]                   # <|im_end|>\n
```

Then the codec generation sequence begins with:
```
CODEC_START: [speaker_id, language_id, codec_bos_id]
```

The Talker autoregressively generates codec-0 tokens until `codec_eos_id` (2150).

### Dual-Track Architecture

The Talker processes text and audio tokens along the same sequence, but they
occupy separate vocabulary spaces:
- Text tokens: vocabulary 0-151,935
- Codec tokens: vocabulary 0-3,071 (separate embedding/head)

The text prompt is embedded via `talker.model.text_embedding` (dim=2048) and
projected to hidden_size (1024) via `talker.text_projection`. The codec BOS
triggers switching to `talker.model.codec_embedding` / `talker.codec_head`.

---

## Weight Format

### Files (1.7B-CustomVoice)
- `model.safetensors.index.json`: weight-to-shard mapping
- `model-*.safetensors`: main model shards, BF16
- `speech_tokenizer/model.safetensors`: speech decoder weights
- `vocab.json` + `merges.txt`: BPE tokenizer
- `config.json`: model configuration

### Files (0.6B-CustomVoice)
- `model.safetensors`: single file, BF16
- `speech_tokenizer/model.safetensors`: speech decoder weights
- Same tokenizer and config files

### Tensor Names

All main model weights are **BF16**. All speech tokenizer weights are **F32**.

**Text Embedding** (BF16):
```
talker.model.text_embedding.weight      [151936, 2048]
```

Note: text_hidden_size = 2048, which is projected down to hidden_size (1024 for
0.6B) via the text projection layers below.

**Text Projection** (BF16, SiLU activation between fc1 and fc2):
```
talker.text_projection.linear_fc1.weight  [2048, 2048]
talker.text_projection.linear_fc1.bias    [2048]
talker.text_projection.linear_fc2.weight  [1024, 2048]
talker.text_projection.linear_fc2.bias    [1024]
```

This is: `text_emb → fc1 → SiLU → fc2 → hidden_size`

**Codec Embedding + Head** (BF16):
```
talker.model.codec_embedding.weight     [3072, 1024]   (singular: codec_embedding, NOT codec_embeddings)
talker.codec_head.weight                [3072, 1024]
```

Note: No `spk_embeddings` tensor exists. Speakers are represented as codec
vocabulary tokens (e.g., serena=3066).

**Talker Layers** (prefix: `talker.model.layers.{i}.`, BF16):
```
input_layernorm.weight                  [1024]
self_attn.q_proj.weight                 [2048, 1024]   (n_heads×head_dim = 16×128 = 2048, NOT hidden_size)
self_attn.k_proj.weight                 [1024, 1024]   (n_kv_heads×head_dim = 8×128)
self_attn.v_proj.weight                 [1024, 1024]   (n_kv_heads×head_dim = 8×128)
self_attn.o_proj.weight                 [1024, 2048]   (hidden_size × q_dim)
self_attn.q_norm.weight                 [128]
self_attn.k_norm.weight                 [128]
post_attention_layernorm.weight         [1024]
mlp.gate_proj.weight                    [3072, 1024]
mlp.up_proj.weight                      [3072, 1024]
mlp.down_proj.weight                    [1024, 3072]
```

**Important:** head_dim = 128, so q_dim = num_heads × head_dim = 16 × 128 = 2048,
which is DIFFERENT from hidden_size = 1024 (for 0.6B). This means Q projection
output is larger than the hidden state input.

Plus `talker.model.norm.weight [1024]` (final norm). NO biases in Talker layers.

**Code Predictor Layers** (prefix: `talker.code_predictor.model.layers.{i}.`, BF16):
```
input_layernorm.weight                  [1024]
self_attn.q_proj.weight                 [2048, 1024]   (same head_dim=128 as talker)
self_attn.k_proj.weight                 [1024, 1024]
self_attn.v_proj.weight                 [1024, 1024]
self_attn.o_proj.weight                 [1024, 2048]
self_attn.q_norm.weight                 [128]
self_attn.k_norm.weight                 [128]
post_attention_layernorm.weight         [1024]
mlp.gate_proj.weight                    [3072, 1024]
mlp.up_proj.weight                      [3072, 1024]
mlp.down_proj.weight                    [1024, 3072]
```

**Code Predictor norm** (BF16):
```
talker.code_predictor.model.norm.weight [1024]
```

**Code Predictor heads and embeddings** (BF16):
```
talker.code_predictor.lm_head.{g}.weight              [2048, 1024]  (g=0..14, 15 heads)
talker.code_predictor.model.codec_embedding.{g}.weight [2048, 1024]  (g=0..14, 15 embeddings)
```

Note: 15 heads/embeddings (indices 0-14), one per residual codebook (codebooks 1-15).

**Speech Tokenizer Decoder** (all F32, in `speech_tokenizer/model.safetensors`):

Codebooks:
```
decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum  [2048, 256]  (codebook 0)
decoder.quantizer.rvq_rest.vq.layers.{0..14}._codebook.embedding_sum  [2048, 256]  (codebooks 1-15)
```

Pre-transformer:
```
decoder.pre_transformer.layers.{0..7}.*    (8 layers with input_proj/output_proj)
```

Decoder blocks (with Snake activation alpha/beta):
```
decoder.decoder.{0..6}.*    (7 blocks: ConvTranspose1d + residual blocks)
```

Upsample (ConvNeXt):
```
decoder.upsample.{0,1}.*    (2 ConvNeXt upsampling blocks)
```

---

## Decode Schedule

### Algorithm

1. **Build prompt**: Construct text token IDs with ChatML wrapping
2. **Embed text tokens**: Look up via `talker.model.text_embedding` table → [seq, 2048]
3. **Project text embeddings**: `text_projection` (fc1 → SiLU → fc2) → [seq, 1024]
4. **Talker prefill**: Feed projected embeddings through 28-layer Talker, build KV cache
5. **Inject codec start**: Embed speaker_id + language_id + codec_bos_id via `talker.model.codec_embedding`
6. **Autoregressive generation**: For each frame:
   a. Talker step: feed previous codec embedding → predict codebook-0 token via `talker.codec_head`
   b. Code Predictor: 15 passes → predict codebooks 1-15 via `talker.code_predictor.lm_head.{0..14}`
   c. Accumulate 16 codes for this frame
7. **Stop on codec_eos_id** (codec token 2150)
8. **Speech decoder**: Convert all accumulated codes → waveform
9. **Write WAV**: 24 kHz, 16-bit PCM, mono

### Sampling

Unlike ASR which uses greedy argmax, TTS requires sampling:
- Temperature: scale logits by `1/temperature` before softmax
- Top-k: keep only k highest-probability tokens
- Top-p (nucleus): keep smallest set of tokens whose cumulative probability >= p
- Repetition penalty: divide logits of previously generated tokens by penalty factor

Default parameters: temperature=0.5, top_k=50, top_p=1.0, repetition_penalty=1.05

The Code Predictor uses separate sampling parameters:
code_predictor_temperature=0.0 (greedy), code_predictor_top_k=1

---

## Performance Characteristics

### Compute Per Audio Frame (12.5 Hz = 80 ms of audio)

| Component | Layer evaluations | Matrix sizes (0.6B) |
|-----------|-------------------|---------------------|
| Talker | 28 layers × 1 pass | Q:[2048×1024], KV:[1024×1024], O:[1024×2048], FFN:[3072×1024] |
| Code Predictor | 5 layers × 15 passes | Q:[2048×1024], KV:[1024×1024], O:[1024×2048], FFN:[3072×1024] |
| Speech Decoder | 8 pre-transform + 7 conv | smaller matrices |
| **Total** | **103 layer evals/frame** | |

At 12.5 frames/second: **~1,288 layer evaluations per second of output audio**.

### BLAS Hotspots (in order of impact)

1. Talker QKV + output projections (28 layers, large matrices)
2. Talker SwiGLU FFN gate/up/down (28 layers, largest matrices)
3. Code Predictor QKV + FFN (75 evaluations per frame)
4. Pre-transform attention layers (8 layers per batch of frames)
5. ConvNet upsampling (moderate, can use im2col + sgemm)

---

## Paper Reference (arXiv:2601.15621)

Key facts from the Qwen3-TTS technical report:

**Training:** Multi-stage pipeline with GAN-based tokenizer pretraining,
LM backbone pretraining, and SFT for voice control.

**Languages:** 10 languages: Chinese, English, Japanese, Korean, German,
French, Russian, Portuguese, Spanish, Italian.

**Quality:** Zero-shot WER 1.24 (English), 0.77 (Chinese) on SEED benchmark.
Speaker similarity competitive with commercial TTS systems.

**Latency:** First-packet latency 97 ms (0.6B), 101 ms (1.7B).
RTF 0.313 at single concurrency (1.7B with torch.compile + CUDA Graph).

**CustomVoice:** 9 preset speakers with controllable speaking styles.
