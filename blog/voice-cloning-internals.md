# Voice Cloning in Pure C: How ECAPA-TDNN Turns Audio Into Identity

*Implementing a speaker encoder from scratch, finding hidden bugs through A/B testing, and learning that embedding dimension matters more than model size.*

## The Goal

Our [pure C TTS engine](https://github.com/gabriele-mastrapasqua/qwen3-tts) already
handled text-to-speech with preset voices — nine speakers across ten languages,
driven by discrete token IDs baked into the model during training. But preset voices
are limiting. Voice cloning lets you feed in a WAV file of *any* speaker and generate
new speech in their voice.

Qwen3-TTS supports this through a **Base** model variant (separate from the CustomVoice
model used for presets). The Base model includes an ECAPA-TDNN speaker encoder that
extracts a continuous embedding from reference audio, replacing the discrete speaker
token in the prompt. The rest of the pipeline — transformer, code predictor, speech
decoder — stays the same.

Implementing it required understanding how a neural network extracts "identity" from
raw audio, and the journey uncovered bugs that only surfaced when we compared models
side by side.

## From Audio to Identity: The ECAPA-TDNN Pipeline

The speaker encoder is an ECAPA-TDNN (*Emphasized Channel Attention, Propagation and
Aggregation in Time Delay Neural Network*), a model architecture designed specifically
for speaker verification. Its job: take variable-length audio and produce a fixed-size
vector that captures *who* is speaking, independent of *what* they're saying.

Here's the full pipeline, step by step:

### Step 1: Mel Spectrogram

Raw audio is a stream of amplitude samples at 24 kHz — about 24,000 numbers per
second, with no structure. The mel spectrogram transforms this into a time-frequency
representation that roughly matches how the human ear perceives sound:

```
Input:  [24000 × duration] raw samples
Output: [T × 128] mel spectrogram (T ≈ 94 frames/second)
```

Parameters: 1024-point FFT, hop size 256 (10.7ms per frame), 128 mel-spaced frequency
bins from 0 to 12 kHz. Each frame becomes a 128-dimensional vector capturing the
energy distribution across frequency bands — bass to treble.

For 30 seconds of audio, this produces about 2,810 frames — a matrix of shape
`[2810, 128]`.

### Step 2: TDNN + SE-Res2Net Blocks

Four convolutional blocks process the mel spectrogram sequentially, extracting
increasingly abstract speaker characteristics:

**Block 0: Initial TDNN** (Time Delay Neural Network)
```
Conv1d(128 → 512, kernel=5, dilation=1) + ReLU
```
A 1D convolution with kernel size 5, looking at ~50ms of context per frame.
Transforms the 128-dim mel features into a richer 512-dim representation.

**Blocks 1–3: SE-Res2Net**

Each block is a Squeeze-and-Excitation Res2Net — a residual network with
multi-scale processing and channel attention:

```
Input (512) → TDNN(512→512, k=3) → Res2Net(7 parallel branches, scale=8)
            → TDNN(512→512, k=1) → SE(channel attention) → + residual
```

The Res2Net splits the 512 channels into 8 groups of 64, processes them with
cascaded convolutions at increasing receptive fields, then recombines. This
captures speaker characteristics at multiple temporal scales simultaneously —
fine-grained (individual phonemes) and coarse (speaking rhythm, pitch contour).

The three blocks use dilations of 2, 3, and 4, giving effective receptive fields
that grow with depth. By the final block, each frame "sees" several hundred
milliseconds of context.

### Step 3: Multi-layer Feature Aggregation

The outputs of all three SE-Res2Net blocks are concatenated channel-wise and
processed through one more TDNN:

```
Concat([block1, block2, block3]) → [1536 × T]
TDNN(1536 → 1536, kernel=1)
```

This gives the network access to features at every level of abstraction
simultaneously — early blocks capture spectral details, later blocks capture
temporal patterns.

### Step 4: Attentive Statistics Pooling

This is the most important step — the one that collapses a variable-length
sequence into a fixed-size vector:

```
Input: [1536 × T] (variable length)

1. Compute mean and std across time → expand to [1536 × T] each
2. Concatenate [hidden, mean, std] → [4608 × T]
3. TDNN(4608 → 128) → tanh
4. Conv1d(128 → 1536) → softmax over time → attention weights [1536 × T]
5. Weighted mean: Σ(attention × hidden) → [1536]
6. Weighted std:  √Σ(attention × (hidden - mean)²) → [1536]
7. Concatenate [weighted_mean, weighted_std] → [3072]
```

The attention mechanism learns *which frames matter most* for identifying the
speaker. Not all speech is equally informative — sustained vowels reveal more
about vocal tract shape than fricatives or silence. The network learns to weight
informative frames higher.

This is also why **longer reference audio helps**: with more frames, the pooling
sees more variation in the speaker's voice. Monotone input yields a flatter
embedding; varied speech with questions, exclamations, and different intonations
captures the speaker's full vocal range. In our testing:

| Duration | Mel Frames | Observation |
|----------|-----------|-------------|
| 10s | ~940 | Recognizable but flat |
| 15s | ~1,400 | Good baseline |
| 30s | ~2,810 | Noticeably richer, better timbre match |
| 45s+ | ~4,200+ | Diminishing returns |

We settled on **30 seconds** as the default — it's the sweet spot between quality
and extraction speed.

### Step 5: Final Projection

```
Conv1d(3072 → enc_dim, kernel=1)
```

A linear projection from the 3072-dim pooled statistics to the model's embedding
dimension. This is where the two model sizes diverge:

| Model | enc_dim | Hidden Size |
|-------|---------|-------------|
| 0.6B-Base | 1024 | 1024 |
| 1.7B-Base | 2048 | 2048 |

The output is a single vector — 1024 or 2048 floats — that encodes the speaker's
identity. This vector is injected into the transformer prompt in place of the
discrete speaker token, and the rest of generation proceeds normally.

## The Hidden Bug: When 1024 ≠ 2048

After implementing the full ECAPA-TDNN pipeline, voice cloning worked well on the
0.6B model. The cloned voices sounded similar to the reference. But when we tested
the 1.7B model, the cloned voice sounded completely different — wrong timbre, wrong
pitch, nothing like the input speaker.

At first, we assumed it was a model quality difference. But the 1.7B should produce
*better* clones, not worse — it has more capacity to utilize the speaker embedding.

The bug was in one line:

```c
enc->enc_dim = 1024;  // hardcoded for 0.6B, wrong for 1.7B
```

The speaker encoder's final FC layer (step 5) produces `enc_dim` values. On the 0.6B
model, `enc_dim = 1024` — correct by coincidence. On the 1.7B, `enc_dim = 2048`,
but we were only computing 1024 values. The remaining 1024 values in the embedding
buffer were uninitialized memory.

When this truncated embedding was injected into the 2048-dim transformer hidden state:

```c
for (int j = 0; j < h; j++)  // h = 2048 on 1.7B
    dst[j] += ctx->speaker_embedding[j];  // buffer only has 1024 valid values!
```

The first half got correct speaker information. The second half got whatever garbage
was in memory — random floats that completely corrupted the speaker signal.

The fix was reading `enc_dim` from the model's `config.json` instead of hardcoding
it. The 0.6B had worked by accident because `enc_dim` happened to equal `hidden_size`
at that model size.

**Lesson:** When two model sizes "work" but one sounds wrong, check whether
shared code makes assumptions about dimensions. Matching by coincidence is not
matching by design.

## 0.6B vs 1.7B: Why Bigger Models Clone Better

After fixing the `enc_dim` bug, we ran A/B tests with the same reference audio,
same text, same seed. The 1.7B consistently produced more faithful voice clones.
Two reasons:

**Richer speaker embedding.** The 1.7B encodes the speaker as a 2048-dim vector vs
1024-dim for the 0.6B. Twice the dimensions means twice the capacity to capture
subtle vocal characteristics — not just fundamental frequency and formant structure,
but breathiness, nasality, speaking rhythm, and micro-patterns in how the speaker
transitions between phonemes.

**More transformer capacity.** The 1.7B Talker has `hidden=2048` and
`intermediate=6144` — four times the parameters of the 0.6B. This means it can
*use* the richer embedding more effectively. A detailed speaker embedding is only
useful if the model has enough capacity to condition its output on those details.

The 0.6B is still excellent for preset voices (CustomVoice model), where the speaker
is a discrete token that the model saw extensively during training. But for voice
cloning — reconstructing a voice from a single embedding of an unseen speaker — the
1.7B's extra capacity makes a clear difference.

## The .qvoice Format: Clone Once, Use Forever

Extracting a speaker embedding from audio isn't free. The mel spectrogram computation,
four convolutional blocks, attentive pooling, and FC projection take about 200ms for
30 seconds of audio. Not terrible, but unnecessary if you're cloning the same voice
repeatedly.

We designed the `.qvoice` format to store the extracted embedding for instant reuse.
The format has evolved through three versions:

- **v1/v2**: Speaker embedding + optional ICL data (~4-50 KB)
- **v3**: Added metadata (language, voice name, model size) + optional weight overrides
- **v3 + WDELTA**: Added compressed weight deltas for bit-identical cross-model output (~510 MB)

For the full story on cross-model voice injection and how we achieved bit-identical output
across different model variants, see [cross-model-voice-analysis.md](cross-model-voice-analysis.md).

The base format:

```
Offset  Size    Field
0       4       Magic: "QVCE"
4       4       Version (3)
12      N×4     Speaker embedding (N = enc_dim floats)
12+N×4  4       ref_text length
...     var     ref_text (UTF-8)
...     4       n_ref_frames
...     var     ref_codes (n_ref_frames × 16 int32)
```

Version 2 of the format (introduced after finding the `enc_dim` bug) stores the
embedding dimension in the header. When loading a `.qvoice` file, the tool checks
that the file's `enc_dim` matches the model's — a 0.6B `.qvoice` cannot be loaded
into a 1.7B model, because the embeddings were produced by different speaker encoders
with different output dimensions. The error message is explicit:

```
Error: .qvoice has enc_dim=1024 but model expects 2048
Re-create the .qvoice with the matching Base model.
```

The original v1 format (without `enc_dim` header) is still supported for backward
compatibility — it assumes the model's own `enc_dim`, which works if the file was
created with the same model.

A typical `.qvoice` file is 4–8 KB for x-vector-only mode (just the embedding) or
20–50 KB with ICL codec tokens (for higher-quality cloning with reference
transcription). Loading is essentially instantaneous — a single `fread` call.

## C Implementation Notes

Implementing ECAPA-TDNN in pure C meant writing every operation from scratch:

**1D convolutions with dilation.** The Res2Net blocks use dilated convolutions
(dilation 2, 3, 4). Unlike the causal convolutions in the speech decoder (left-only
padding), the speaker encoder uses standard symmetric "same" padding with reflection
at boundaries. Each conv is implemented as a nested loop over output channels,
input channels, kernel positions, and time steps.

**Res2Net multi-scale processing.** The 512 channels are split into 8 groups of 64.
Group 0 passes through unchanged. Groups 1–7 each go through a dilated Conv1d, and
each group's output is added to the next group's input before its convolution. This
cascaded structure creates features at increasingly large receptive fields.

**Squeeze-and-Excitation.** After Res2Net, the channel attention block computes:
global average pool → FC(512→128) → ReLU → FC(128→512) → sigmoid → scale channels.
This re-weights channels based on their global statistics — emphasizing channels
that carry speaker-distinctive information.

**Attentive Statistics Pooling.** The attention TDNN uses ReLU (from the TimeDelayNet
block) followed by tanh (from the ASP wrapper) — an unusual combination that we
initially got wrong. The Python code applies `self.tanh(self.tdnn(x))`, where
`self.tdnn` is a full TDNN block with ReLU activation. So the activation chain is
Conv → ReLU → tanh → Conv → softmax. Missing the tanh produces attention weights
with the wrong distribution.

All of these operations use scalar C with `-O3 -ffast-math`. The speaker encoder
runs once per voice clone (or zero times when loading from `.qvoice`), so SIMD
optimization wasn't worth the complexity. The total extraction time for 30 seconds
of audio is ~200ms — negligible compared to the generation itself.

## What We Learned

1. **Test every model size.** The `enc_dim` bug was invisible on 0.6B because the
   dimensions happened to match. It only surfaced on 1.7B. If we'd only tested one
   model size, we'd have shipped a broken voice clone that produced garbage speaker
   embeddings on the larger model.

2. **Longer reference audio helps, but not linearly.** The attentive statistics
   pooling benefits from diversity in the input — different intonations, pitch
   ranges, speaking styles. 30 seconds of varied speech beats 60 seconds of monotone
   reading. Quality of input matters more than quantity.

3. **Embedding dimension is quality.** The jump from 1024-dim to 2048-dim embeddings
   produced clearly better voice clones — more faithful timbre, better pitch matching,
   more natural prosody. This makes intuitive sense: encoding a human voice into 1024
   floats is lossy compression. 2048 floats capture twice the detail.

4. **File formats need version headers.** The v1 `.qvoice` format assumed a fixed
   `enc_dim`, which broke silently on model size mismatch. Adding a version field
   and dimension header to v2 turned a silent data corruption into a clear error
   message. Always design file formats for forward compatibility.

5. **A/B testing catches what unit tests miss.** The `enc_dim` bug wouldn't have been
   caught by comparing C output to Python output for a single model — both would
   produce the "correct" output for 0.6B. It took comparing 0.6B vs 1.7B voice
   clones, noticing the 1.7B sounded wrong, and tracing back through the pipeline
   to find the hardcoded dimension. Listen to your outputs.

6. **The encoder hears everything, not just the voice.** The ECAPA-TDNN processes the
   raw mel spectrogram — it has no concept of "voice" vs "background." When the
   reference audio contains background music, ambient noise, or other speakers, those
   sounds become part of the speaker embedding. The model faithfully reproduces them
   as artifacts in the generated audio — a low hum, a rhythmic texture, or subtle
   noise floor that shouldn't be there. The fix is upstream: use clean reference audio,
   or pre-process with a source separation tool like
   [demucs](https://github.com/facebookresearch/demucs) to isolate the voice before
   cloning. The speaker encoder itself is working correctly — it's just capturing
   exactly what you give it.

7. **Style control and voice cloning are separate worlds** — on the Base model.
   The Base model was never trained with both instruct tokens and voice clone prompt
   together. When both are present, instruct dominates and the cloned voice fades.
   But there's a solution: cross-model injection with weight deltas (see below).

## Cross-Model Voice Injection: Reusable Custom Voices

After implementing the basic voice clone, we discovered something remarkable: the Base
and CustomVoice models have **99.98% identical transformer weights** (cosine similarity
0.9999 per layer). The only differences are the 9 preset speaker embeddings in the codec
table and the ECAPA-TDNN speaker encoder that only the Base model has.

This near-identity opened the door to **perfect cross-model voice transfer**. We can
clone a voice on Base, then replay it on CustomVoice — with `--instruct` support —
by correcting for the tiny weight differences between models.

### The .qvoice v3 Format

The format evolved from a simple embedding dump to a full voice profile with metadata
and optional weight corrections:

```
.qvoice v3 structure:
  QVCE magic + version 3
  ├── Speaker embedding (ECAPA-TDNN, 1024 or 2048 floats)
  ├── Reference text + ICL codec tokens (optional)
  ├── META: language, voice name, source model size, flags
  │         → auto-sets language at load time, warns on mismatch
  ├── TPAD: source model's tts_pad/bos/eos embeddings (12 KB)
  ├── WOVR: text_projection + codec_embedding weights (16 MB)
  └── WDELTA: LZ4-compressed int16 deltas for all talker+CP weights (~785 MB)
              → bit-identical output on target CustomVoice model
```

### How We Got to Bit-Identical

The journey involved systematic elimination of divergence sources:

**Step 1: Embed override (TPAD, +12 KB)** — The `tts_pad_embed` (used every generation
frame) differs slightly between models due to micro-differences in `text_projection`
weights. Storing and overriding the source model's embed eliminates per-frame drift.
Result: mel correlation improved from baseline to 0.756.

**Step 2: Weight override (WOVR, +16 MB)** — Override `text_projection` and
`codec_embedding` entirely. Eliminates embedding-pipeline differences.
Result: mel correlation 0.711, RTF improved to 1.60 (fastest variant at the time).

**Step 3: Full weight delta (WDELTA, ~785 MB with LZ4)** — The remaining gap came from
the transformer layers themselves (28 talker layers + 5 code predictor layers). Despite
being 99.98% similar by cosine, the BF16 values differ at 87% of positions. We store
int16 deltas per tensor, compressed with LZ4 — at load time, the target model's weights
are corrected to exactly match the source model.
Result: **mel correlation 1.000, PCM bit-identical output**.

Key discoveries along the way:

- **Partial layer replacement makes things worse** — replacing only the 5 most-divergent
  layers (out of 28) drops quality below the no-replacement baseline. The transformer is
  a chain; mismatched interfaces at layer boundaries cause more harm than uniform small
  differences across all layers. It's all-or-nothing.

- **The Code Predictor matters** — even after replacing all 28 talker layers, codebooks
  5-15 still diverged. The CP has its own weights (86 tensors) and its own `gate_up_fused`
  buffer that must be rebuilt after weight override. Missing this caused us hours of
  debugging.

- **30 seconds of reference audio is the sweet spot** — more audio doesn't help because
  the ECAPA embedding stabilizes quickly (cosine > 0.999 between 30s and 47s embeddings).
  The bottleneck isn't embedding quality; it's the autoregressive butterfly effect during
  generation.

### Performance Comparison

All measurements on Apple M1 8-core, 0.6B model, Italian text:

| .qvoice Format | Size | Mel Corr | Wall Time | vs Preset | Bit-identical? |
|----------------|------|----------|----------|-----------|----------------|
| CV preset ryan (baseline) | — | N/A | 12.0s | — | — |
| Standard (TPAD+WOVR) | 16 MB | 0.711 | 10.6s | -12% | No |
| **Delta + LZ4 (WDELTA)** | **785 MB** | **1.000** | **12.8s** | **+7%** | **Yes** |
| Base `--ref-audio` | — | reference | 33.5s | +179% | — |

The delta format with LZ4 has only **+7% overhead** compared to using a preset voice.
This was a major optimization: the original zlib compression had +32% overhead (~4s
decompression for 494MB). Switching to LZ4 reduced decompression to ~1s for the same
data, despite a 54% larger file (785MB vs 510MB). The tradeoff is worth it.

### LZ4 vs Zlib: Why Faster Decompression Matters More Than File Size

We initially used zlib (gzip level 6) for delta compression. It produced compact files
but the decompression dominated load time:

| Compression | File (0.6B) | Decompress | Total Wall Time | vs Preset |
|-------------|------------|-----------|----------------|-----------|
| zlib | 510 MB | ~4s | 15.9s | +32% |
| **LZ4** | **785 MB** | **~1s** | **12.8s** | **+7%** |

LZ4 decompresses ~7.5x faster because it trades compression ratio for speed — the
algorithm is simpler and operates on larger blocks without back-references across the
entire stream. For our use case (one-time load at startup), decompression speed matters
far more than file size.

### Style Control with Cloned Voices (--instruct)

The ultimate test: can you clone a voice AND apply style control? With the 1.7B
CustomVoice model and a WDELTA .qvoice, yes:

```bash
# Clone once (needs 1.7B Base + CV)
./qwen_tts -d qwen3-tts-1.7b-base --ref-audio silvio.wav -l Italian \
    --voice-name "Silvio" --target-cv qwen3-tts-1.7b \
    --save-voice silvio_17b.qvoice

# Use with different styles
./qwen_tts -d qwen3-tts-1.7b --load-voice silvio_17b.qvoice \
    --text "Una notizia importante." \
    -I "Parla con voce triste e malinconica" -o sad.wav

./qwen_tts -d qwen3-tts-1.7b --load-voice silvio_17b.qvoice \
    --text "Una notizia importante." \
    -I "Parla con voce allegra e entusiasta" -o happy.wav
```

The voice identity is preserved across all styles — the instruct modulates prosody
(rhythm, emphasis, pacing) while the cloned timbre stays consistent. The effect is
subtle but real: sad speech is slower with lower energy, happy speech has more
variation and higher pitch.

This was made possible by the WDELTA format: the `.qvoice` transforms the CustomVoice
model's weights to match the Base model exactly, so the cloned voice is processed by
the same weights that originally produced it — but now with instruct support that only
CustomVoice provides.

### Creating and Using Reusable Voices

```bash
# One-time creation (needs both Base and CV models)
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio speaker.wav -l Italian \
    --voice-name "Mario" --target-cv qwen3-tts-0.6b \
    --save-voice voices/mario_06b.qvoice

# Use anywhere (only needs CV model + .qvoice file)
./qwen_tts -d qwen3-tts-0.6b --load-voice voices/mario_06b.qvoice \
    --text "Ciao, come va?" -o output.wav
# Language auto-set from metadata: Italian

# List available voices
./qwen_tts --list-voices voices/
#   mario_06b.qvoice   v3  [Mario]  lang=Italian  model=0.6B
```

Without `--target-cv`, the standard 16 MB format is created — the voice is recognizable
but prosody varies from the Base clone. With `--target-cv`, the ~785 MB LZ4 delta format
gives perfect fidelity with only +7% load overhead. Users choose the tradeoff they want.

The practical impact: you can clone any voice once, save it as a `.qvoice`, and share
the file. Anyone with a CustomVoice model can use it — with perfect fidelity, correct
language settings, and (on 1.7B) instruct-based style control. No Base model needed
at runtime.

For the full technical deep-dive on weight analysis, delta encoding, and how we traced
every source of divergence, see [cross-model-voice-analysis.md](cross-model-voice-analysis.md).

---

*This is part of the [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts)
project — a pure C inference engine for Qwen3-TTS text-to-speech models.*
