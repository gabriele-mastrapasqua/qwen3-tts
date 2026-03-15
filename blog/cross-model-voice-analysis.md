# Cross-Model Voice Injection: Anatomy of a 0.9999 Cosine Surprise

*When two "different" models turn out to be nearly identical, and the real divergence hides in the last 0.001% of the pipeline.*

## The Problem

We can clone any voice with the Base model — feed it 30 seconds of audio, and it reproduces that speaker with near-perfect fidelity. But the Base model doesn't support style control (`--instruct`), and only the CustomVoice variant lets you say things like "speak slowly and solemnly" or "whisper excitedly."

The obvious solution: extract the voice on Base, inject it into CustomVoice, get both clone quality and instruct support. We tried this with KV cache injection (dump the voice prefix from Base, load into CustomVoice). It works — the voice is recognizable, the language is correct, there are no artifacts. But something is off. The timbre shifts. The voice sounds like a mix of the cloned speaker and the model's internal idea of how speech should sound. About 90% fidelity, not 100%.

Why? We assumed the answer was obvious: the models have different weights. Our previous analysis suggested "80-88% different transformer weights." With that much divergence, 90% fidelity seemed impressive.

We were wrong about the weights.

## The Discovery: One Model, Two Skins

We wrote a script to compare every tensor in both models — all 402 shared tensors, weight by weight, using cosine similarity.

The results were shocking:

| Component | Tensors | Cosine | Status |
|-----------|---------|--------|--------|
| Talker transformer (28 layers × 11 weights) | 308 | 0.9998–1.0000 | **Identical** |
| Code Predictor (5 layers) | 86 | 0.9999 | **Identical** |
| Text embeddings (151K tokens) | 1 | 1.000000 | **Identical** |
| Codebook entries (0–2047) | — | 1.0000 | **Identical** |
| LayerNorm, Q/K norms | all | 1.0000 | **Identical** |
| codec_embedding (speaker tokens) | 1 | 0.6147 | **Different** |

**Every single transformer weight is nearly identical.** K projection, V projection, Q projection, O projection, MLP gate, MLP up, MLP down, LayerNorm — all cosine > 0.9998 across all 28 layers.

The "80-88% different" claim from earlier was wrong. What we'd actually measured back then was KV cache divergence (the *output* of the transformer), not the weights themselves. The weights are the same model.

The only real difference: **9 speaker preset tokens** in the codec embedding table.

```
Speaker token 2861 (aiden):  Base norm = 0.016  →  CV norm = 9.99
Speaker token 3061 (ryan):   Base norm = 0.016  →  CV norm = 9.41
Speaker token 3066 (serena): Base norm = 0.016  →  CV norm = 9.63
```

In the Base model, these tokens have near-zero norms (unused — Base uses ECAPA-TDNN continuous embeddings instead). In CustomVoice, they hold the 9 preset voice identities.

The models also differ in one other component: Base has 76 extra tensors for the ECAPA-TDNN speaker encoder. CustomVoice doesn't need it because it uses discrete tokens.

That's it. Same transformer, same code predictor, same speech decoder. Two models differing only in *how the speaker is specified*.

## If the Weights Are the Same, Why Does the Voice Change?

This is where it gets interesting. If the transformer is identical, injecting Base KV into CustomVoice should be transparent — the math is the same on both sides. But it's not.

We traced the divergence to three sources, all subtle:

### Source 1: `tts_pad_embed` — The Silent Accumulator

Every frame of autoregressive generation uses `tts_pad_embed` as the text-side input embedding. This embedding is computed at model load time:

```
text_embedding[TTS_PAD]  →  text_projection(fc1, SiLU, fc2)  →  tts_pad_embed
```

The text embedding table is identical between models (cosine 1.000000). But `text_projection.fc1` and `fc2` have cosine 0.99997 — not quite 1.0. This microscopic difference produces:

```
Base:  tts_pad_embed[:3] = [0.004777, -0.003806, -0.001696]
CV:    tts_pad_embed[:3] = [0.004978, -0.003639, -0.002433]
```

Cosine between them: 0.99996. Almost identical. But this embedding is added to the input at **every single frame** of generation — 60+ times for a 5-second utterance. The error doesn't cancel out; it accumulates through 28 transformer layers per frame, shifting the hidden state distribution slightly at each step.

### Source 2: Weight Micro-Differences in Middle Layers

While the overall weight cosine is 0.9999, the worst layers are in the middle of the network:

```
Layer 16 MLP up:   cosine = 0.999782  (worst)
Layer 17 MLP gate: cosine = 0.999787
Layer 16 MLP gate: cosine = 0.999793
Layer 18 MLP gate: cosine = 0.999795
```

These are small differences — the 4th decimal place — but they compound. A transformer layer's output feeds into the next layer's input. Over 28 layers, a 0.0002 per-layer error in the MLP pathway can shift the attention distribution measurably.

### Source 3: The Butterfly Effect of Autoregressive Sampling

This is the killer. Autoregressive generation is chaotic — the next token depends on all previous tokens. Even a tiny difference in logits can push the softmax past a decision boundary, selecting a different codec token. That different token gets embedded (from the identical codebook), but now the hidden state has diverged. The next step diverges more. Within 10-20 frames, the two models are generating completely different token sequences.

We verified this with frame-by-frame mel spectrogram correlation:

```
Frames 0-0.9s:   correlation 0.85  (similar, not yet diverged)
Frames 0.9-1.8s: correlation 0.79  (starting to drift)
Frames 1.8-2.7s: correlation 0.82  (some recovery)
Frames 2.7-3.6s: correlation 0.50  (fully diverged)
Frames 3.6-4.5s: correlation 0.66  (different prosody)
```

The voice *timbre* (who is speaking) remains similar throughout because the KV cache voice prefix locks the attention to the right speaker identity. But the *prosody* (rhythm, intonation, phrasing) diverges because each sampled token pushes the generation onto a different trajectory.

## Quantifying the Gap

We generated the same Italian text ("Ciao, come stai oggi? Spero che tu stia bene.") with Silvio's voice across three configurations, all with seed 42:

| Configuration | Mel Correlation vs Base Direct | Duration | Notes |
|--------------|-------------------------------|----------|-------|
| Base direct (reference) | 1.000 | 4.48s | Ground truth |
| Base KV reload (same model) | 0.847 | 4.32s | BF16 roundtrip noise |
| CV inject (.qvoice v3) | 0.649 | 5.20s | Cross-model |

Even same-model KV reload isn't bit-identical (0.85 not 1.0). The BF16 dump/load roundtrip introduces quantization noise that cascades through autoregressive generation. But the first 0.9 seconds have correlation 1.0000 — the divergence builds up only after several generation steps.

Cross-model injection scores 0.65 overall, but the frame-by-frame analysis reveals it starts at 0.85 (matching the same-model case) and degrades from there. The gap isn't in the voice identity — it's in the generation trajectory.

### Error Decomposition by Position

We compared KV cache values position by position between the Silvio voice clone (Base) and ryan preset (CustomVoice), both in Italian:

```
Position    Content       K cosine    V cosine    Notes
0           <|im_start|>  0.9997      0.9986      Nearly identical
1           assistant      0.9997      0.9966      Nearly identical
2           \n             0.9998      0.9990      Nearly identical
3           THINK          0.9998      0.9983      Nearly identical
4           THINK_BOS      0.9990      0.9927      Nearly identical
5           language_id    0.9909      0.9861      Slight drift (5th token in chain)
6           THINK_EOS      0.9787      0.9718      Accumulating (attends to 0-5)
7           SPEAKER        0.8266      0.5812      ← Different input embedding!
8           PAD            0.8608      0.4308      ← Sees different speaker at pos 7
```

Positions 0-6 (same tokens, same weights) still show cosine < 1.0 because of the weight micro-differences accumulating through attention. But the real gap is at positions 7-8:

- **Position 7 (speaker)**: K cosine 0.83, V cosine 0.58. This position receives fundamentally different input — ECAPA-TDNN embedding (30s of audio compressed to 1024 floats) vs preset speaker token embedding. Even though both go through the same transformer weights, the *direction* of the input is different.

- **Position 8 (PAD)**: K cosine 0.86, V cosine 0.43. This position attends to position 7, inheriting its divergence.

The error is **99-100% directional** (not magnitude). Scaling norms doesn't help — the vectors point in genuinely different directions because they encode different speakers via different mechanisms.

## .qvoice v3: Metadata That Prevents Mistakes

One practical problem we kept hitting: using an Italian voice with `-l English`, or forgetting which model created a voice file. This led to `.qvoice` v3, which adds a metadata section:

```
.qvoice v3 layout:
  QVCE magic + version 3
  ├── enc_dim + speaker_embedding (ECAPA-TDNN, 1024 or 2048 floats)
  ├── ref_text + ref_codes (ICL data, optional)
  └── META section (new in v3):
      ├── language_id + language_name     → auto-set at load time
      ├── source_model_hidden_size        → warn on size mismatch
      ├── ref_audio_duration              → how much audio was used
      ├── voice_name                      → human identifier
      └── flags (xvector/icl/base)
```

When loading a v3 voice, the engine automatically sets the language:

```
$ ./qwen_tts -d qwen3-tts-0.6b --load-voice silvio.qvoice --text "Ciao, come va?"
  Auto-set language from voice: Italian
  Voice: Silvio | Language: Italian | Model: 0.6B | Ref: 30s
```

And warns if you try to override with a mismatched language:

```
$ ./qwen_tts -d qwen3-tts-0.6b --load-voice silvio.qvoice -l English --text "Hello"
  WARNING: voice was created with language 'Italian' but you specified 'English'
    Voice fidelity may be reduced. Consider using -l Italian
```

The format is backward-compatible: v1 and v2 files load without issues, they just don't have metadata. `--list-voices` shows the full picture:

```
$ ./qwen_tts --list-voices qvoices/
  silvio_italian_17b.qvoice       v2    0 frames  8.0 KB
  silvio_italian_06b.qvoice       v2    0 frames  4.0 KB
  silvio_06b.qvoice               v3    0 frames  4.1 KB  [Silvio]  lang=Italian  model=0.6B
```

## What We Tried: Exhaustive Optimization

After the initial analysis, we systematically tested every approach to close the fidelity gap.

### TPAD: Overriding tts_pad/bos/eos Embeddings

Since `tts_pad_embed` was the dominant per-frame error source, we stored the Base model's computed embeddings directly in the `.qvoice` file (TPAD section, +12KB). When CustomVoice loads a Base voice, it overrides its own embeddings with the stored ones.

Result: **+2.3% mel correlation, -6.9% MSE**. The improvement was real but modest — the first 2 seconds of audio aligned much better (min correlation 0.33 → 0.62), but the butterfly effect still dominated after that.

### WOVR: Full Weight Override

We went further: store the Base model's `text_projection` weights and `codec_embedding` codebook entries in the `.qvoice` file (WOVR section, +16MB). This makes the voice file **completely self-contained** — no Base model needed at load time.

At load time, CustomVoice replaces its own text_projection and codec_embedding with the Base model's versions. This eliminates ALL per-frame weight differences in the embedding pipeline.

### Everything Else We Tested

| Approach | Result | Why |
|----------|--------|-----|
| Lower temperature (0.3) | **Much worse** (mel corr 0.19) | Sharper softmax amplifies tiny logit differences |
| Greedy warmup (first N frames) | **Much worse** | Greedy selects different tokens than temp 0.5, changing the entire trajectory |
| Reduced top-k (50→5) | **No effect** | The chosen token is already in top-5 at temp 0.5 |
| Longer reference (47s vs 30s) | **Worse** | ECAPA embeddings are already stable (cosine > 0.999 between 30s and 47s). More audio doesn't help |
| Preset voice interference | **Confirmed: none** | Codebook entries (0-2047) are bit-identical. Speaker presets are never looked up in voice_clone mode |
| Partial layer replacement (top 5) | **Much worse** (mel corr 0.59) | Interface mismatch: Base layer output → CV layer input is worse than all-CV |
| Partial layer replacement (top 10) | **Even worse** (mel corr 0.37) | More mixed layers = more mismatched interfaces |
| Full layer replacement (all 28) | **Perfect** (mel corr 1.00) | But this IS the Base model — 840MB of weights |
| BF16 delta compression | **Not feasible** (290MB gzipped) | 87% of values differ; cosine 0.9999 but BF16 deltas are large (mean 117-228) |

### The Multi-Seed Revelation

The most important experiment was running 5 different seeds and comparing:

```
Base vs Base (different seeds):    mean mel corr = 0.30
CV vs CV (different seeds):        mean mel corr = 0.41
Cross-model (same seed):           mean mel corr = 0.32
```

**Cross-model divergence equals natural seed variance.** The "voice shift" we hear between Base and CustomVoice is the same magnitude as the variation you get by simply changing the random seed on the same model. The voice *identity* (timbre, who is speaking) is preserved. The *prosody* (rhythm, intonation, phrasing) varies — but it varies just as much between different seeds on the same model.

Seed 42 happened to be a lucky outlier (mel corr 0.76). The mean across seeds tells the real story.

## The Final Picture: RTF and Quality

Four ways to use a cloned voice, from slowest to fastest:

| Config | File Size | RTF | Quality | Base Model Needed? |
|--------|----------|-----|---------|-------------------|
| `--ref-audio` (clone from WAV) | 30s WAV | 2.00 | Reference | Yes (at runtime) |
| Base `--load-voice` | 16MB .qvoice | 1.78 | **Bit-identical** to above | Yes (at runtime) |
| CV + TPAD only | 16KB .qvoice | 1.88 | Voice preserved, prosody varies | No |
| **CV + WOVR full** | **16MB .qvoice** | **1.60** | Voice preserved, prosody varies | **No** |

The standout finding: **Base model + .qvoice produces bit-identical output** (mel correlation 1.0000) compared to cloning from the original WAV file. Zero quality loss, 11% faster (skips ECAPA-TDNN extraction).

The **WOVR configuration is the fastest** at RTF 1.60 — 20% faster than cloning from WAV. The self-contained 16MB `.qvoice` file carries everything needed: speaker embedding, language metadata, voice name, tts embeddings, text_projection weights, and codec_embedding codebook. Any CustomVoice model can load it without the Base model being present.

## What This Means

The cross-model voice injection story has a clean ending:

1. **Clone once, use forever.** Extract a voice with the Base model, save as `.qvoice`. The file is self-contained and works on any compatible model.

2. **The quality gap is a mirage.** What sounds like a "different voice" on CustomVoice is actually just different prosody — the same variation you'd hear changing seeds on ANY model. The voice identity (timbre, pitch range, speaking style) is faithfully preserved.

3. **Speed bonus.** Cross-model injection on CustomVoice (RTF 1.60) is faster than direct voice cloning on Base (RTF 2.00). The `.qvoice` file eliminates both ECAPA extraction time and the minor overhead of loading the larger Base model.

4. **The fundamental limit is autoregressive sampling.** With weights that are 99.98% identical, the remaining 0.02% produces logit differences that cascade through sampling. This is structurally identical to changing the random seed — it's not a bug to fix, it's the nature of stochastic text-to-speech.

The `.qvoice` format makes voice cloning practical: extract once with a Base model (on any machine), share the 16MB file, and anyone can use that voice on any CustomVoice model with full quality, correct language, and style control via `--instruct`.

## The All-or-Nothing Trap

One last thing we tried: partial layer replacement. If the most divergent layers (14-18) cause the most error, why not store just those layers' weights in the `.qvoice`?

It makes things **worse**:

```
CV + WOVR (no layer replacement):     mel corr 0.711
CV + WOVR + top-5 layers from Base:   mel corr 0.590  ← worse!
CV + WOVR + top-10 layers from Base:  mel corr 0.371  ← much worse!
CV + WOVR + ALL 28 layers from Base:  mel corr 1.000  ← perfect
```

The transformer is a chain. Each layer's output feeds the next layer's input. When layer 14 produces "Base-style" output but layer 13 (CV) and layer 19 (CV) expect "CV-style" input/output, the interface mismatch at layer boundaries causes more damage than having uniformly-slightly-wrong weights throughout.

This is a fundamental property of deep networks: you can't mix weights from different training runs at the layer level. The representations at each layer boundary are model-specific. Even though the weights are 99.98% similar by cosine, the remaining 0.02% creates internal representations that are incompatible when mixed.

We also checked if the weight deltas could be compressed: 87% of BF16 values differ, and the deltas are large (mean absolute delta 117-228 per layer, not the ±1 we'd hoped). Gzipped, the delta patch is 290-370MB — essentially the full model.

The practical conclusion: **use the Base model for perfect voice clone** (it's the same speed), or **use CV + WOVR for portable voice files** (16MB, self-contained, slight prosody variation). There's no middle ground.

---

*This is part of the [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) project — a pure C inference engine for Qwen3-TTS text-to-speech models.*
