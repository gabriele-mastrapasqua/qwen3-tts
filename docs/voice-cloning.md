# Voice Cloning

Clone any voice from a short reference audio clip using the Base model's built-in
ECAPA-TDNN speaker encoder.

**Requires a Base model** — the CustomVoice models (0.6B, 1.7B) do NOT support
voice cloning directly. To use cloned voices on CustomVoice (with style control,
server, streaming), see [Custom Voices](custom-voices.md).

## Quick Start

```bash
# Download the Base model (has speaker encoder for voice cloning)
./download_model.sh --model base-small   # 0.6B-Base (faster, good quality)
./download_model.sh --model base-large   # 1.7B-Base (slower, best clone quality)

# Clone a voice from a WAV file
./qwen_tts -d qwen3-tts-1.7b-base --text "Hello, this is my cloned voice." \
    --ref-audio reference.wav -o cloned.wav

# Clone with Italian text
./qwen_tts -d qwen3-tts-1.7b-base --text "Ciao, questa e la mia voce clonata." \
    --ref-audio reference.wav -l Italian -o cloned_it.wav
```

## Model Comparison

The 1.7B-Base produces significantly better voice clones than the 0.6B-Base.

| | 0.6B-Base | 1.7B-Base |
|---|---|---|
| **Speaker embedding dim** | 1024 | 2048 |
| **Transformer hidden** | 1024 | 2048 |
| **Clone fidelity** | Good | Best |
| **Speed (Apple M1)** | RTF ~1.5–1.7 | RTF ~3.2–4.1 |
| **Best for** | Fast cloning, acceptable quality | Maximum voice fidelity |

The 2048-dim speaker embedding captures twice the vocal detail (timbre, pitch contour,
breathiness, speaking rhythm) compared to the 0.6B's 1024-dim embedding. The larger
transformer also has more capacity to condition its output on speaker characteristics.

## Reference Audio

### Format

The reference audio **must be 24 kHz WAV** (PCM, mono or stereo, 16-bit or 32-bit).
Convert other formats with ffmpeg:

```bash
# Convert any audio file to 24 kHz mono WAV
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
ffmpeg -i input.opus -ar 24000 -ac 1 output.wav

# Even WAV files may need resampling (e.g., 16 kHz → 24 kHz)
ffmpeg -i voice_16k.wav -ar 24000 output.wav
```

This is required because the ECAPA-TDNN speaker encoder uses a mel spectrogram
computed at 24 kHz with specific FFT parameters (n_fft=1024, hop=256, 128 mels).
A mismatched sample rate produces incorrect mel features and a bad voice embedding.

### Duration

More reference audio generally produces better clones. The attentive pooling layer
benefits from seeing diverse speech patterns.

| Duration | Mel frames | Quality | Notes |
|----------|-----------|---------|-------|
| 5-10s | 470-940 | Good | Minimum for recognizable clone |
| 15-20s | 1400-1880 | Better | Covers basic vocal range |
| **30s** | **2810** | **Recommended** | Good balance of quality and speed |
| 45s+ | 4200+ | Best | Diminishing returns, slower extraction |

By default, the first **30 seconds** of reference audio are used. Use `--max-ref-duration 0`
to process the entire file, or set a custom limit (e.g., `--max-ref-duration 45`).

### Tips for Best Results

- **Use clean audio without background music or noise.** The speaker encoder processes the
  raw mel spectrogram and cannot separate voice from background — music, ambient noise, or
  other speakers will be captured as part of the speaker embedding and reproduced as artifacts
  in the output. If your reference has background noise, pre-process it with a voice separation
  tool (e.g., [demucs](https://github.com/facebookresearch/demucs)) before cloning.
- Include varied speech (questions, statements, different emotions) rather than monotone reading.
- 24 kHz WAV is ideal; other sample rates will be rejected.

## How the Speaker Encoder Works

The speaker encoder is an **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and
Aggregation in TDNN) network that converts reference audio into a fixed-size speaker embedding:

1. **Mel spectrogram**: The reference audio is converted to a 128-band mel spectrogram at 24 kHz
   (hop size 256 = ~94 frames/second).
2. **TDNN + SE-Res2Net blocks**: Four convolutional blocks extract speaker-characteristic features
   across time — capturing pitch, timbre, and speaking style.
3. **Attentive Statistics Pooling**: Computes a weighted mean and standard deviation over the
   **entire temporal sequence**. This is the key step: longer audio means the pooling sees more
   variation in the speaker's voice (different intonations, pitch ranges, speaking styles),
   producing a richer and more representative embedding.
4. **FC projection**: The pooled statistics (3072-dim) are projected to the final embedding
   dimension (1024 for 0.6B, 2048 for 1.7B) and injected into the transformer prompt.

For a full technical deep-dive, see the
[voice cloning internals blog post](../blog/voice-cloning-internals.md).

## Voice Clone Samples

| Input | Model | Cloned Output | Text |
|-------|-------|---------------|------|
| [reference (movie clip)](../samples/10s_back_down_the_road.wav) | 0.6B | [english](../samples/clone_output_en.wav) | *I love programming in C, it gives you complete control over the machine.* |
| [reference (movie clip)](../samples/10s_back_down_the_road.wav) | 0.6B | [italian](../samples/clone_output_it.wav) | *Buongiorno, questa e una dimostrazione della clonazione vocale.* |
| italian speaker | 0.6B | [italian](../samples/clone_silvio_constitution_06b.wav) | *Tutti i cittadini hanno il dovere di essere fedeli alla Repubblica...* |
| italian speaker | 1.7B | [italian](../samples/clone_silvio_constitution_17b.wav) | *Tutti i cittadini hanno il dovere di essere fedeli alla Repubblica...* |

The 1.7B clone captures the original speaker's timbre and pitch more faithfully than
the 0.6B — compare the two Italian constitution samples above.

## Quick Demo

```bash
# Clone from a sample WAV (outputs to samples/)
make demo-clone

# Clone from your own audio
make demo-clone REF=my_voice.wav

# Custom text too
make demo-clone REF=my_voice.wav TEXT="Hello from my cloned voice!"
```

## Performance

**Comparison** (Apple M1 8-core, 4 threads, 0.6B-Base, ~4s output):

| Mode | Prefill | Total | RTF | Notes |
|------|---------|-------|-----|-------|
| From WAV (`--ref-audio`) | 2.8s | 21.6s | 4.91 | Mel + speaker enc + speech enc + generate |
| From `.qvoice` (`--load-voice`) | 1.7s | 9.8s | 2.23 | Load file + generate (no audio processing) |

Saving to a `.qvoice` gives a **2x speedup** on subsequent generations by skipping
mel spectrogram extraction, speaker encoding, and speech encoding.
See [Custom Voices](custom-voices.md) for details.
