---
title: "Qwen3-TTS voice cloning in a pure C engine — 785 MB, 16 MB, or 4 KB?"
published: false
description: "Clone a voice from 30 seconds of public-domain audio in pure C, then choose how to store it: bit-identical (785 MB), small-and-sharable (16 MB), or postcard-sized (4 KB). Listen to all three."
tags: c, machinelearning, tts, audio
---

*Part of [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) — a pure C inference engine for Qwen3-TTS.*

## TL;DR — turn any 30-second clip into a first-class Qwen3-TTS voice

Qwen3-TTS ships with **9 preset speakers**. That's it. You can't add your own, you can't use the 1.7B instruct feature on a cloned voice, and every new clone has to re-run the 200 ms ECAPA-TDNN encoder from scratch.

This post is about tearing that ceiling down.

With the pure-C engine at [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) you can:

- 🎙️ Clone **any voice** from 30 seconds of audio (ECAPA-TDNN speaker encoder implemented from scratch)
- 💾 Save it as a portable **`.qvoice` file** and load it anywhere — CLI, HTTP server, streaming pipeline, one-shot generation
- 🎛️ Combine a cloned voice with **`--instruct` style prompts** on 1.7B (sad / happy / angry / solemn) — something the Base model alone can't do
- 🎯 Get **bit-identical output** across runs, processes, and machines via the `WDELTA` weight-delta format

The `.qvoice` file is a new way to *extend* Qwen3-TTS's voice set: drop the file next to the binary, point `--load-voice` at it, and the model speaks with your voice like it was one of the originals.

That portability comes in three flavors. All produce the same voice identity; they differ in how much of the Base model's weight signature they carry along:

| Format | Size | Mel correlation vs Base | Fidelity | Works with instruct? | Use when… |
|--------|------|------------------------|----------|----------------------|-----------|
| 🥇 `.qvoice` **WDELTA** (LZ4 full delta) | **785 MB** | **1.000** (bit-identical) | Perfect, PCM-identical | ✅ yes (1.7B) | You're building a reusable voice asset — server, streaming, product |
| 🥈 `.qvoice` **standard** (TPAD + WOVR) | **16 MB** | **0.71** | Good; small prosody drift | ⚠️ Base only | Default for sharing — fits in chat, sounds right |
| 🥉 `.bin` **embedding only** | **4 KB** | *not measured* (~60–70 % subj.) | Voice drifts, timbre loose | ❌ no | You have 4 kilobytes to spend |

The headline: **WDELTA makes a cloned voice a first-class citizen of the CustomVoice model**. You clone once on Base, save a `.qvoice`, and the CV model loads it and treats it exactly like one of the nine built-in speakers — same latency, same server behavior, same streaming support, now with instruct-style control on top.

All audio below is hosted in the same repo — click ▶️ to play.

---

## Samples — listen for yourself

### 🇮🇹 Italian — *Galatea* / Riccardo Fasol · [LibriVox, PD](https://archive.org/details/galatea_0908_librivox)

> *"Buongiorno a tutti, oggi vi racconto una breve storia, con la voce clonata da una registrazione libera."*

📥 **Input reference** — 30 s from LibriVox · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/it_galatea_fasol.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/it_galatea_fasol.wav"></audio>

---

#### 🎤 Voice clone output — 3 storage formats

🥇 **Top — WDELTA, 785 MB** (mel 1.000, bit-identical) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_wdelta_785mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_wdelta_785mb.wav"></audio>

🥈 **Mid — standard `.qvoice`, 16 MB** (mel 0.71) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_standard_16mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_standard_16mb.wav"></audio>

🥉 **Light — `.bin`, 4 KB** (embedding only) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_bin_4kb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_it_bin_4kb.wav"></audio>

### 🇬🇧 English — *The Gifts of the Magi* / Phil Chenevert · [LibriVox, CC0](https://archive.org/details/5belovedstories_ohenry_pc_librivox)

> *"Hello everyone, today I am speaking with a voice cloned from a freely licensed recording."*

📥 **Input reference** · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/en_ohenry_chenevert.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/en_ohenry_chenevert.wav"></audio>

---

#### 🎤 Voice clone output — 3 storage formats

🥇 **Top — WDELTA, 785 MB** (mel 1.000) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_wdelta_785mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_wdelta_785mb.wav"></audio>

🥈 **Mid — standard `.qvoice`, 16 MB** (mel 0.71) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_standard_16mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_standard_16mb.wav"></audio>

🥉 **Light — `.bin`, 4 KB** · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_bin_4kb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_en_bin_4kb.wav"></audio>

### 🇪🇸 Spanish — *Don Quijote* / Lu · [LibriVox, PD](https://archive.org/details/donquijote_2507_librivox)

> *"Hola a todos, hoy les hablo con una voz clonada a partir de una grabación de dominio público."*

📥 **Input reference** · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/es_quijote_lu.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/es_quijote_lu.wav"></audio>

---

#### 🎤 Voice clone output — 3 storage formats

🥇 **Top — WDELTA, 785 MB** (mel 1.000) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_wdelta_785mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_wdelta_785mb.wav"></audio>

🥈 **Mid — standard `.qvoice`, 16 MB** (mel 0.71) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_standard_16mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_standard_16mb.wav"></audio>

🥉 **Light — `.bin`, 4 KB** · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_bin_4kb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_es_bin_4kb.wav"></audio>

### 🇫🇷 French — *Le dernier jour d'un condamné* / Bidou · [LibriVox, PD](https://archive.org/details/dernierjour_2203_librivox)

> *"Bonjour à tous, aujourd'hui je vous parle avec une voix clonée à partir d'un enregistrement libre."*

📥 **Input reference** · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/fr_hugo_bidou.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/fr_hugo_bidou.wav"></audio>

---

#### 🎤 Voice clone output — 3 storage formats

🥇 **Top — WDELTA, 785 MB** (mel 1.000) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_wdelta_785mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_wdelta_785mb.wav"></audio>

🥈 **Mid — standard `.qvoice`, 16 MB** (mel 0.71) · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_standard_16mb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_standard_16mb.wav"></audio>

🥉 **Light — `.bin`, 4 KB** · [wav](https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_bin_4kb.wav)
<audio controls preload="none" src="https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/samples/voice_clone_refs/outputs/out_fr_bin_4kb.wav"></audio>

### Cost table — how much you pay for each tier

All numbers on Apple M1 8-core, 16 GB RAM, 4 threads, 0.6B model, cold start.

| Format | File size | Create `.qvoice` | Generate (wall) | What's inside |
|--------|-----------|-----------------|----------------|---------------|
| `.bin` | 4 KB | ~20.8 s | ~11 s | 1024 ECAPA-TDNN floats |
| standard `.qvoice` | 16 MB | ~19.6 s | ~9.6 s | Embedding + `text_projection` + `codec_embedding` + pad embeds |
| WDELTA `.qvoice` | 785 MB | ~28.5 s | ~13.8 s | LZ4 int16 deltas for all 402 talker + CP tensors |

Verdict: **standard 16 MB is the sensible default.** Go to WDELTA only when you need bit-identical output or to combine a cloned voice with `--instruct` style control (1.7B). Go to `.bin` only if 4 KB is the whole budget.

All input clips are 30-second excerpts from the 30 s mark (skipping the LibriVox preamble), 24 kHz mono PCM. Full attribution in [`samples/voice_clone_refs/ATTRIBUTION.md`](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/samples/voice_clone_refs/ATTRIBUTION.md).

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
