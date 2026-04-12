---
title: "I cloned a voice in pure C — and you can hear the difference between 4 KB, 16 MB, and 785 MB"
published: false
description: "Three ways to store a cloned voice, from a tiny embedding to a bit-identical weight delta. Listen to them side by side, then read how the pipeline works."
tags: c, machinelearning, tts, audio
---

*Part of [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) — a pure C inference engine for Qwen3-TTS.*

## TL;DR — listen first, read later

I took ~30 seconds of freely-licensed speech from public-domain recordings, ran it through an ECAPA-TDNN speaker encoder I implemented from scratch in C, and generated new speech in the same voice. The result is a `.qvoice` file you can share, reload, and mix with style prompts on a different model variant.

The interesting part isn't that it works — it's that **the same clone can be stored three ways, and each trade-off is audible**:

| Format      | File size | Fidelity         | What's inside                                   |
|-------------|-----------|------------------|-------------------------------------------------|
| `.bin`      | 4 KB      | ~60–70%          | Just the 1024-float speaker embedding           |
| `.qvoice` standard | 16 MB     | Good, prosody drifts | Embedding + `text_projection` + `codec_embedding` |
| `.qvoice` WDELTA   | **785 MB**    | **Bit-identical**    | LZ4-compressed weight deltas for all layers     |

Scroll down and hear them back to back.

---

## The reference voices (CC / public domain)

All input audio below is either public domain or Creative Commons. Sources and licenses are linked next to each sample.

All four clips are 30-second excerpts taken from the 30-second mark of the original
recording (skipping the LibriVox preamble), downmixed to 24 kHz mono PCM.

### 🇮🇹 Italian — *Galatea* by Anton Giulio Barrili

- **Reader:** Riccardo Fasol (solo)
- **Source:** [archive.org/details/galatea_0908_librivox](https://archive.org/details/galatea_0908_librivox)
- **License:** Public Domain (LibriVox)

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/it_galatea_fasol.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/it_galatea_fasol.wav">Download WAV</a>
</audio>

### 🇬🇧 English — *The Gifts of the Magi* by O. Henry

- **Reader:** Phil Chenevert (solo)
- **Source:** [archive.org/details/5belovedstories_ohenry_pc_librivox](https://archive.org/details/5belovedstories_ohenry_pc_librivox)
- **License:** CC0 / Public Domain

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/en_ohenry_chenevert.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/en_ohenry_chenevert.wav">Download WAV</a>
</audio>

### 🇪🇸 Spanish — *Don Quijote* by Miguel de Cervantes

- **Reader:** Lu (solo)
- **Source:** [archive.org/details/donquijote_2507_librivox](https://archive.org/details/donquijote_2507_librivox)
- **License:** Public Domain (LibriVox)

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/es_quijote_lu.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/es_quijote_lu.wav">Download WAV</a>
</audio>

### 🇫🇷 French — *Le dernier jour d'un condamné* by Victor Hugo

- **Reader:** Bidou (solo)
- **Source:** [archive.org/details/dernierjour_2203_librivox](https://archive.org/details/dernierjour_2203_librivox)
- **License:** Public Domain (LibriVox)

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/fr_hugo_bidou.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/fr_hugo_bidou.wav">Download WAV</a>
</audio>

---

## The three storage formats, side by side

Same reference audio, same text, same seed (42), same 0.6B CustomVoice model at generation time. Only the `.qvoice` format changes. Wall times measured on Apple M1 8-core, 16 GB RAM, 4 threads.

### Summary

| Format | File size | Create time | Generate wall | What's inside |
|--------|-----------|-------------|--------------|---------------|
| `.bin` (embedding only) | **4.0 KB** | ~20.8 s | ~11 s | 1024 ECAPA-TDNN floats |
| `.qvoice` standard (TPAD + WOVR) | **16.0 MB** | ~19.6 s | ~9.6 s | Embedding + source `text_projection` + `codec_embedding` + pad embeds |
| `.qvoice` WDELTA (LZ4 full delta) | **785 MB** | ~28.5 s | ~13.8 s | LZ4 int16 deltas for all 402 transformer + CP tensors |

Create time includes the full 30-s reference audio pass through ECAPA-TDNN (~200 ms) plus model load + weight-delta scan (WDELTA only). Generate wall is cold-start: mmap + optional LZ4 decompress + inference.

### 🇮🇹 Italian — *Galatea* / Fasol

Same text for all three: *"Buongiorno a tutti, oggi vi racconto una breve storia, con la voce clonata da una registrazione libera."*

**4 KB — `.bin`, embedding only.** A single 1024-float vector — the raw output of ECAPA-TDNN, nothing else. Recognizable voice, but prosody and micro-timing drift noticeably.

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_bin_4kb.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_bin_4kb.wav">Download WAV</a>
</audio>

**16 MB — standard `.qvoice` (TPAD + WOVR).** Adds the source model's `tts_pad` embedding and overrides of `text_projection` + `codec_embedding`. Good fidelity, small enough to share over chat, but the autoregressive "butterfly effect" still causes subtle prosody drift.

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_standard_16mb.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_standard_16mb.wav">Download WAV</a>
</audio>

**785 MB — WDELTA (LZ4-compressed full weight delta).** Stores int16 deltas for every transformer + code-predictor tensor, LZ4-compressed. At load time the CV model's weights are corrected to exactly match the Base model that did the cloning. Mel correlation **1.000**, PCM bit-identical.

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_wdelta_785mb.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_wdelta_785mb.wav">Download WAV</a>
</audio>

### 🇬🇧 English — O. Henry / Chenevert

Same text for all three: *"Hello everyone, today I am speaking with a voice cloned from a freely licensed recording."*

**4 KB — `.bin`:**
<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_bin_4kb.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_bin_4kb.wav">Download WAV</a>
</audio>

**16 MB — standard `.qvoice`:**
<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_standard_16mb.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_standard_16mb.wav">Download WAV</a>
</audio>

**785 MB — WDELTA:**
<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_wdelta_785mb.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_wdelta_785mb.wav">Download WAV</a>
</audio>

Listening side by side: the 4 KB version sounds like "someone vaguely similar", the 16 MB sounds close but with slightly different pacing, and the 785 MB is indistinguishable from running the Base model directly every time. In practice the 16 MB standard format is the sensible default for sharing; WDELTA is for cases where bit-identical reproducibility and instruct-style support matter more than disk space.

---

## Multilingual clones — one voice per language

Each language uses its own CC-licensed reference voice. The clone keeps the speaker's
timbre and prosody while speaking a completely different sentence. All generated at
0.6B-Base, seed 42.

### 🇮🇹 Italian — cloned from Fasol / *Galatea*
> "Buongiorno a tutti, oggi vi racconto una breve storia, con la voce clonata da una registrazione libera."

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_galatea.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_galatea.wav">Download WAV</a>
</audio>

### 🇬🇧 English — cloned from Chenevert / *O. Henry*
> "Hello everyone, today I am speaking with a voice cloned from a freely licensed recording."

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_ohenry.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_ohenry.wav">Download WAV</a>
</audio>

### 🇪🇸 Spanish — cloned from Lu / *Don Quijote*
> "Hola a todos, hoy les hablo con una voz clonada a partir de una grabación de dominio público."

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_quijote.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_quijote.wav">Download WAV</a>
</audio>

### 🇫🇷 French — cloned from Bidou / Hugo
> "Bonjour à tous, aujourd'hui je vous parle avec une voix clonée à partir d'un enregistrement libre."

<audio controls src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_hugo.wav">
  <a href="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_hugo.wav">Download WAV</a>
</audio>

---

Now the technical part — how we got there.

## From audio to identity: the ECAPA-TDNN pipeline

The speaker encoder is an ECAPA-TDNN (*Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network*), designed for speaker verification. Its job: take variable-length audio and produce a fixed-size vector that captures *who* is speaking, independent of *what* they're saying.

### Step 1 — Mel spectrogram

Raw 24 kHz audio → 1024-point FFT, hop 256, 128 mel bins → `[T, 128]` where T ≈ 94 frames/sec. For 30 s of audio, roughly `[2810, 128]`.

### Step 2 — TDNN + SE-Res2Net blocks

Four convolutional blocks process the mel spectrogram. The first is a plain 1D conv (128→512, k=5). The next three are Squeeze-and-Excitation Res2Net blocks: each splits 512 channels into 8 groups of 64, runs cascaded dilated convolutions (dilations 2, 3, 4) over them, and reweights channels with a small attention block. The effective receptive field grows to several hundred milliseconds by the last block.

### Step 3 — Multi-layer feature aggregation

The three SE-Res2Net outputs are concatenated channel-wise (→ `[1536, T]`) and passed through one more TDNN. The network now has access to features at every level of abstraction simultaneously.

### Step 4 — Attentive Statistics Pooling

The most important step — the one that collapses variable-length time into a fixed-size vector:

```
[1536, T] → mean, std across T → concat [hidden, mean, std] → [4608, T]
         → TDNN(4608→128) → tanh → Conv1d(128→1536) → softmax over T
         → weighted mean + weighted std → [3072]
```

Attention learns *which frames matter most*. Sustained vowels reveal more about vocal tract shape than fricatives or silence — the network weights them higher. This is also why **varied reference audio beats long monotone audio**: more variation, richer pooling. 30 s is our default sweet spot.

### Step 5 — Final projection

```
Conv1d(3072 → enc_dim, kernel=1)
```

| Model      | enc_dim | hidden |
|------------|---------|--------|
| 0.6B-Base  | 1024    | 1024   |
| 1.7B-Base  | 2048    | 2048   |

The output — 1024 or 2048 floats — replaces the discrete speaker token in the transformer prompt.

## The bug hidden by a coincidence

Voice cloning worked on 0.6B. On 1.7B it sounded completely wrong. The cause was a single line:

```c
enc->enc_dim = 1024;  // hardcoded — wrong for 1.7B
```

On 0.6B, `enc_dim == hidden == 1024` by coincidence. On 1.7B, `enc_dim == 2048`, so we were writing 1024 valid floats into a 2048-dim slot — the rest was uninitialized memory. The first half of the hidden state got a real speaker; the second half got garbage.

The fix was reading `enc_dim` from `config.json`. **Lesson:** when two model sizes "work" but one sounds wrong, check whether shared code accidentally matches by coincidence rather than by design.

## Why 1.7B clones better

After the fix, 1.7B consistently produced more faithful clones. Two reasons:

- **2048-dim embedding** vs 1024-dim — twice the capacity to capture breathiness, nasality, micro-timing in phoneme transitions.
- **4× transformer parameters** — the model can actually *use* the richer embedding.

A detailed speaker embedding is only useful if the model has the capacity to condition on those details.

## The `.qvoice` v3 format

Cloning isn't free (~200 ms of ECAPA-TDNN per 30 s of audio), and more importantly a raw embedding alone loses prosody. The `.qvoice` format stores everything needed to reproduce the clone:

```
QVCE magic + version 3
├── Speaker embedding        (1024 or 2048 floats)
├── Reference text + ICL codec tokens (optional)
├── META                     (language, voice name, source model size, flags)
├── TPAD                     (source model's tts_pad/bos/eos embeddings, 12 KB)
├── WOVR                     (text_projection + codec_embedding, 16 MB)
└── WDELTA                   (LZ4 int16 deltas for all talker+CP weights, ~785 MB)
```

Each section is optional. You pick the trade-off.

## How we got to bit-identical

Base and CustomVoice share **99.98 %** of transformer weights (cosine ≈ 0.9999 per layer). But BF16 values differ at 87 % of positions, and those micro-differences accumulate autoregressively. Closing the gap was a three-step elimination:

1. **TPAD (+12 KB)** — override the source model's `tts_pad_embed`. Mel correlation 0.756.
2. **WOVR (+16 MB)** — override `text_projection` and `codec_embedding` entirely. Mel correlation 0.711, RTF 1.60.
3. **WDELTA (+785 MB, LZ4)** — int16 deltas for every remaining layer. Mel correlation **1.000**, PCM bit-identical.

Two things bit us along the way:

- **Partial layer replacement is worse than none.** Replacing the 5 most-divergent layers out of 28 dropped quality below the no-replacement baseline. The transformer is a chain; mismatched interfaces at layer boundaries cost more than uniform small differences everywhere.
- **The Code Predictor has its own weights.** Even after replacing all 28 talker layers, codebooks 5–15 still diverged until we also deltaed the CP's 86 tensors and rebuilt its `gate_up_fused` buffer.

## LZ4 vs zlib

We started with zlib. It produced smaller files but decompression dominated load time.

| Compression | File (0.6B) | Decompress | Total wall | vs preset |
|-------------|-------------|-----------|------------|-----------|
| zlib        | 510 MB      | ~4 s      | 15.9 s     | +32 %     |
| **LZ4**     | **785 MB**  | **~1 s**  | **12.8 s** | **+7 %**  |

For a one-shot load at startup, decompression speed matters more than file size.

## Style control + cloned voice

On 1.7B + WDELTA you can finally combine `--instruct` with a cloned voice — something the Base model alone can't do, because it was never trained with both signals together:

```bash
./qwen_tts -d qwen3-tts-1.7b --load-voice silvio_17b.qvoice \
    --text "Una notizia importante." \
    -I "Parla con voce triste e malinconica" -o sad.wav

./qwen_tts -d qwen3-tts-1.7b --load-voice silvio_17b.qvoice \
    --text "Una notizia importante." \
    -I "Parla con voce allegra e entusiasta" -o happy.wav
```

Voice identity stays constant; instruct modulates rhythm, pacing, and emphasis.

## Commands, for the curious

```bash
# Clone once (needs Base + CV of same size)
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio speaker.wav -l Italian \
    --voice-name "Mario" --target-cv qwen3-tts-0.6b \
    --save-voice voices/mario_06b.qvoice

# Use anywhere (only needs CV + .qvoice)
./qwen_tts -d qwen3-tts-0.6b --load-voice voices/mario_06b.qvoice \
    --text "Ciao, come va?" -o output.wav
```

Without `--target-cv`, you get the 16 MB standard format. With it, the 785 MB WDELTA.

## What I learned

1. **Test every model size.** Dimension bugs hide behind coincidences.
2. **Longer audio helps, but not linearly.** Diversity beats duration.
3. **Embedding dimension *is* quality.** 1024 → 2048 is a clear audible jump.
4. **Version your file formats.** v1 `.qvoice` silently corrupted on size mismatch; v2+ errors loudly.
5. **A/B test by listening.** Unit tests pass on garbage outputs. Ears don't.
6. **The encoder captures everything, not just the voice** — background music, room noise, a second speaker. Clean your input or run demucs first.
7. **Style control and voice cloning live in separate worlds — until you bridge them with weight deltas.**

---

*Source, deeper dives, and benchmarks: [gabriele-mastrapasqua/qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts). For the full weight-analysis story behind WDELTA, see [cross-model-voice-analysis.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/blog/cross-model-voice-analysis.md).*
