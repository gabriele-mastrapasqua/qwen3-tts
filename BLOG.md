# Building a Text-to-Speech Engine in Pure C

A few weeks ago I came across [antirez](https://github.com/antirez)'s [qwen-asr](https://github.com/antirez/qwen-asr) — a pure C implementation of Qwen2-Audio for speech recognition. No Python, no PyTorch, just C and raw model weights. I've always admired antirez's approach to software: minimal, focused, no unnecessary abstractions. Reading his code, I thought: *if ASR can be done in C, why not TTS?*

That's how this project started. [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) had just been released by the Qwen team — a state-of-the-art text-to-speech system supporting 10 languages, multiple voices, and two model sizes (0.6B and 1.7B parameters). The official implementation is Python/PyTorch. I wanted to see if I could run the entire pipeline — tokenizer, transformer, code predictor, speech decoder — in pure C with zero ML framework dependencies.

## The Architecture

Qwen3-TTS has three main components chained together:

1. **Talker** — A 28-layer Qwen3 transformer (basically an LLM) that takes tokenized text and generates one audio frame token per autoregressive step.
2. **Code Predictor** — A smaller 5-layer transformer that runs 15 sequential passes per frame, predicting 15 additional codebook entries for each audio frame.
3. **Speech Decoder** — A causal convolutional network that takes the 16 codebook indices per frame, looks them up in learned embeddings, and upsamples 480x to produce raw audio at 24 kHz.

The output of this pipeline is a standard WAV file. Simple in theory. Implementing it was anything but.

## What Made It Hard

### Reverse-engineering the pipeline

The official Python code is spread across multiple files with layers of abstraction from HuggingFace Transformers. There's no single "here's how inference works" document. I had to trace through the Python code step by step, running the PyTorch model and dumping intermediate tensors at every layer to understand what was actually happening.

Some things that tripped me up:

- **RoPE style confusion.** The model config says `"rope_interleaved": true`, but the actual Python code uses the "NeoX" split-half style by default (not interleaved). This took days to figure out — the attention scores were wrong in subtle ways that only showed up as garbled audio.

- **Causal convolutions everywhere.** The speech decoder's convolutions are all left-padded (causal), not the standard symmetric padding you'd expect. Missing this meant the decoder outputs were shifted and correlation with the reference was terrible.

- **Snake activation in log space.** The speech decoder uses Snake activations where the alpha and beta parameters are stored as logarithms. The formula is `x + (1/exp(beta)) * sin^2(exp(alpha) * x)`. Getting the exp() placement wrong produces audio that sounds like static.

- **Separate RVQ projections.** The first codebook and remaining 15 codebooks have separate output projection layers for dequantization. I initially used the same projection for all, which produced audio that was *almost* right but had persistent artifacts.

- **Special token encoding.** The prompt template uses ChatML-style tokens like `<|im_start|>`. These need to be BPE-encoded as raw text, not looked up as special token IDs. Getting this wrong meant the model received a completely different prompt than expected.

### BF16 without a GPU

The model weights are stored in BF16 (Brain Float 16). On a GPU you'd just load them directly. On CPU, every matrix multiplication needs BF16-to-F32 conversion. The trick from qwen-asr is elegant: memory-map the safetensors files and convert on the fly during computation using NEON/AVX SIMD intrinsics. A BF16-to-F32 conversion is literally just a left-shift by 16 bits — it's nearly free.

For the hot path (autoregressive token generation), I wrote fused BF16 matrix-vector kernels that read BF16 weights, convert to F32, multiply, and accumulate without ever materializing a full F32 weight matrix. This keeps memory bandwidth usage low and means the 0.6B model runs with only ~3 GB of RAM.

### Verification methodology

The hardest part of writing any ML inference engine from scratch is knowing whether your numbers are right. A single wrong sign, transposed dimension, or off-by-one error somewhere in a 28-layer transformer produces outputs that are "wrong" but not obviously wrong — they're just different floats.

My approach: run the Python reference and the C code side by side, comparing intermediate outputs at every stage:
1. First, verify the tokenizer produces identical token IDs.
2. Then verify embeddings match after the text projection.
3. Then verify attention outputs match layer by layer during prefill.
4. Then verify the final logits match (top-5 tokens and values).
5. Then verify Code Predictor outputs match.
6. Then verify Speech Decoder outputs match.
7. Finally, verify the full WAV output has correlation > 0.999 with the Python output.

Each step exposed bugs in the previous one. The final correlation between my C output and the Python reference is 0.999996 — essentially identical.

## Performance

The 0.6B model runs at about 0.7x realtime on a 4-core CPU (meaning 1 second of audio takes about 1.4 seconds to generate). The bottleneck is the Code Predictor — it runs 15 sequential autoregressive passes per audio frame, and each pass involves 5 transformer layers. That's 75 transformer forward passes per 80ms of audio.

Key optimizations:
- **Memory-mapped BF16 weights** — near-instant loading, no copy.
- **Fused NEON/AVX kernels** — BF16 matvec with 2-row fusion, directly reading from mmapped weights.
- **Multi-threaded dispatch** — 4 threads for the bandwidth-bound matvec operations.
- **Reused buffers** — eliminated ~1000+ malloc/free calls per generation by pre-allocating and reusing scratch buffers.
- **Fused argmax+matvec** — for greedy decoding, computes the argmax during the matrix-vector multiply without materializing the full logit vector.
- **BLAS for prefill** — the initial prompt processing uses cblas_sgemm for the large matrix-matrix multiplications.

I tried INT4 quantization for the 0.6B model but it was actually 20% slower — the matrices are small enough (1024-wide) that the overhead of unpacking quantized weights exceeds the bandwidth savings. BF16 is the sweet spot for this model size.

## The Result

The engine compiles with a single `make blas` command on macOS or Linux. It produces 24 kHz WAV files in 9 different voices across 10 languages. The audio quality matches the Python reference exactly (when using greedy decoding) and is indistinguishable in quality when using standard sampling.

Building this was a deep dive into every component of a modern TTS system — from BPE tokenization to transformer attention to convolutional audio synthesis. It reinforced something I've believed for a while: implementing ML models from scratch in C is not just an exercise — it produces genuinely useful, deployable software. A single static binary that does text-to-speech with no dependencies is something you can drop onto any Linux server and just run.

Thanks to the Qwen team for releasing the model weights, and to antirez for showing that this approach works.

The code is at [github.com/gabriele-mastrapasqua/qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts). MIT licensed.
