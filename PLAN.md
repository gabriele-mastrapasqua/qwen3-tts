# Qwen3-TTS: Pure C Implementation Plan

Implementazione in C puro del modello Qwen3-TTS (text-to-speech), seguendo lo stile
di [antirez/qwen-asr](https://github.com/antirez/qwen-asr): codice semplice, minimale,
con BLAS come unica dipendenza esterna (Accelerate su macOS, OpenBLAS su Linux).

---

## 1. Architettura del Modello Qwen3-TTS

Il modello ha **4 componenti principali** che formano una pipeline a cascata:

```
INPUT TEXT
    │
    ▼
[BPE Tokenizer]  ──▶  text token IDs (vocab 151,936)
    │
    + (opzionale) [Speaker Encoder / ECAPA-TDNN]  ──▶  speaker embedding
    │
    ▼
[TALKER: 28-layer Qwen3 Transformer]
    │  Autoregressive: predice il codebook 0 (semantico) per ogni frame
    │  GQA: 16 Q heads, 8 KV heads, head_dim=128
    │  SwiGLU FFN, RoPE (theta=1M)
    │
    ▼
[CODE PREDICTOR: 5-layer Transformer (MTP)]
    │  15 pass sequenziali per frame
    │  Genera codebook 1-15 (dettaglio acustico)
    │
    ▼
16 codici discreti per frame @ 12.5 Hz
    │
    ▼
[SPEECH TOKENIZER DECODER (ConvNet)]
    │  Codebook embedding lookup (16 × 2048 × 128)
    │  8 pre-transformer layers
    │  4 upsampling ConvNet blocks (480× totale)
    │  Causale (no lookahead) → streaming possibile
    │
    ▼
24 kHz AUDIO WAVEFORM  ──▶  file .wav
```

### 1.1 Talker (LLM Backbone)

Il cuore del sistema. Stessa famiglia Qwen3 del decoder ASR, con differenze chiave:

| Parametro | 0.6B | 1.7B |
|-----------|------|------|
| Hidden size | 1024 | 2048 |
| Layers | 28 | 28 |
| Q heads | 16 | 16 |
| KV heads (GQA) | 8 | 8 |
| Head dim | 128 | 128 |
| FFN intermediate | 3072 | 6144 |
| Activation | SwiGLU (SiLU) | SwiGLU (SiLU) |
| RoPE theta | 1,000,000 | 1,000,000 |
| Vocab (testo) | 151,936 | 151,936 |
| Vocab (codec) | 3,072 | 3,072 |

**Nota critica**: A differenza dell'ASR che fa solo greedy argmax, il TTS richiede
**sampling** (temperature=0.9, top_k=50, repetition_penalty=1.05).

### 1.2 Code Predictor (MTP Module)

Piccolo transformer secondario che completa i codebook residuali:

| Parametro | Valore |
|-----------|--------|
| Hidden size | 1024 |
| Layers | 5 |
| Q heads | 16 |
| KV heads | 8 |
| FFN intermediate | 3072 |
| Codebook groups | 16 |
| Codebook vocab | 2,048 per group |

**Pipeline per frame**: Il Talker produce 1 token (codebook 0) → il Code Predictor
fa 15 pass sequenziali per generare codebook 1-15. Totale: 28×1 + 5×15 = 103
layer evaluations per frame audio.

### 1.3 Speech Tokenizer Decoder (ConvNet)

Converte i 16 codici discreti in forma d'onda audio a 24 kHz:

- **Codebook embedding**: 16 tabelle × 2048 entries × 128 dim
- **Pre-transform**: 8 transformer layers (self-attention + FFN)
- **Upsampling blocks**: 4 blocchi con convoluzioni dilated 1D
  - Fattore totale: 480× (da 12.5 Hz a 24,000 Hz)
- **Fully causal**: nessun lookahead, quindi streaming nativo

### 1.4 Speaker Encoder (ECAPA-TDNN) — opzionale, per voice cloning

- Input: mel spectrogram dell'audio di riferimento
- 8 pre-transformer layers + 3 SE-Res2Net blocks
- Attentive statistics pooling → embedding 1024/2048 dim
- **Fase 1**: Skip, supportare solo CustomVoice (speaker ID predefiniti)

---

## 2. Confronto con qwen-asr: Cosa Riusare

### 2.1 Componenti Direttamente Riusabili (~60% del codice)

| Componente ASR | LOC | Riuso TTS | Note |
|----------------|-----|-----------|------|
| `qwen_asr_safetensors.c` | 394 | **Copia diretta** | Loader multi-shard, mmap, identico |
| `qwen_asr_kernels.c` | 1247 | **Copia diretta** | matmul, softmax, RMSNorm, SwiGLU, threading |
| `qwen_asr_kernels_neon.c` | 274 | **Copia diretta** | SIMD ARM per bf16 matvec |
| `qwen_asr_kernels_avx.c` | 502 | **Copia diretta** | SIMD x86 per bf16 matvec |
| `qwen_asr_kernels_impl.h` | ~50 | **Copia diretta** | Dispatch architettura |
| `qwen_asr_tokenizer.c` | 650 | **Adattare** | Stesso BPE vocab, aggiungere token TTS speciali |
| `qwen_asr_decoder.c` | 476 | **Base per Talker** | GQA, SwiGLU, RoPE, KV cache — stesso pattern |
| Makefile | ~60 | **Adattare** | Stessa struttura BLAS cross-platform |

### 2.2 Componenti Nuovi da Scrivere

| Componente TTS | LOC stimate | Descrizione |
|----------------|-------------|-------------|
| `qwen_tts.c` | ~1500 | Orchestrazione pipeline, prompt construction, generazione |
| `qwen_tts.h` | ~300 | Strutture dati, config, API pubblica |
| `qwen_tts_talker.c` | ~500 | Forward pass Talker (basato su decoder ASR) |
| `qwen_tts_code_predictor.c` | ~400 | Forward pass MTP, 15 pass sequenziali |
| `qwen_tts_speech_decoder.c` | ~600 | ConvNet decoder: embedding, conv1d, upsampling |
| `qwen_tts_audio.c` | ~200 | WAV writer (24kHz, 16-bit PCM) |
| `qwen_tts_sampling.c` | ~200 | Temperature, top-k, top-p, repetition penalty |
| `main.c` | ~200 | CLI parsing, usage |
| **Totale nuovo** | **~3900** | |
| **Totale riusato** | **~3100** | |
| **Totale progetto** | **~7000** | |

### 2.3 Differenze Chiave ASR → TTS

| Aspetto | ASR | TTS |
|---------|-----|-----|
| Input | Audio WAV → mel spectrogram | Testo (stringa UTF-8) |
| Output | Testo (token IDs → UTF-8) | Audio WAV (24 kHz) |
| Encoder audio | Conv2D stem + Transformer (bidirectional) | Nessuno (solo tokenizer BPE per testo) |
| Decoder LLM | Identico Qwen3 28-layer | Identico Qwen3 28-layer (**Talker**) |
| Post-LLM | — | Code Predictor (5 layer × 15 pass) |
| Vocoder | — | ConvNet decoder (480× upsample) |
| Sampling | Greedy argmax | Temperature + top-k + top-p |
| Speaker | — | Speaker embedding (ECAPA o ID predefinito) |
| Streaming | Audio chunks → testo | Testo → audio chunks (causale) |

---

## 3. Pipeline di Inferenza Dettagliata

### Step 1: Preparazione Input

```
Testo utente: "Ciao, come stai?"
    │
    ▼
ChatML formatting:
  <|im_start|>assistant\n{testo}<|im_end|>\n
    │
    ▼
BPE tokenize → [151644, ..., 151645, 198]
    │
    ▼
Prepend speaker tokens + language ID + tts_bos
    │
    ▼
Token embedding lookup (bf16 → f32)
```

### Step 2: Talker Autoregressive Generation

```
Per ogni frame audio (12.5 al secondo):
    │
    ▼
  Forward pass completo 28 layers:
    ├─ RMSNorm
    ├─ Q/K/V projections (bf16 matmul) ← BLAS hotspot
    ├─ Per-head Q/K RMSNorm
    ├─ RoPE (NeoX-style)
    ├─ Causal GQA attention + KV cache append
    ├─ Output projection ← BLAS hotspot
    ├─ Post-attention RMSNorm
    ├─ SwiGLU FFN (gate+up fused → down) ← BLAS hotspot
    └─ Residual connections
    │
    ▼
  Logits → sampling (temperature, top-k, top-p)
    │
    ▼
  Token codebook-0 per questo frame
```

### Step 3: Code Predictor (MTP)

```
Per ogni frame, dato il token codebook-0 dal Talker:
    │
    ▼
  Per i = 1..15:
    Forward pass 5 layers (stesso pattern: RMSNorm, GQA, SwiGLU)
    Logits → sampling → token codebook-i
    │
    ▼
  Output: 16 codici discreti per frame
```

### Step 4: Speech Tokenizer Decoder

```
Sequenza di frame [N_frames × 16 codici]:
    │
    ▼
  Codebook embedding lookup:
    Per ogni codebook k=0..15:
      emb_k = codebook_k[code_k]  (128 dim)
    sum_emb = Σ emb_k  (somma residuale RVQ)
    │
    ▼
  Pre-transform (8 transformer layers):
    [N_frames × 128] → self-attention + FFN
    │
    ▼
  Upsampling ConvNet (4 blocchi):
    Blocco 1: upsample ×5  (ConvTranspose1d)
    Blocco 2: upsample ×4
    Blocco 3: upsample ×4
    Blocco 4: upsample ×6
    Totale: 5 × 4 × 4 × 6 = 480×
    │
    ▼
  Output: [N_frames × 480] samples @ 24 kHz
    │
    ▼
  Scrivi WAV 24kHz 16-bit PCM
```

---

## 4. Strutture Dati Principali

```c
/* ── Config ── */
typedef struct {
    int hidden_size;          /* 1024 (0.6B) o 2048 (1.7B) */
    int num_layers;           /* 28 */
    int num_heads;            /* 16 */
    int num_kv_heads;         /* 8 */
    int head_dim;             /* 128 */
    int intermediate_size;    /* 3072 (0.6B) o 6144 (1.7B) */
    int vocab_size;           /* 151936 (testo) */
    int codec_vocab_size;     /* 3072 */
    int num_codebooks;        /* 16 */
    int codebook_dim;         /* 128 */
    int codebook_size;        /* 2048 */
    float rope_theta;         /* 1e6 */
    /* Code Predictor config */
    int cp_hidden_size;       /* 1024 */
    int cp_num_layers;        /* 5 */
    int cp_intermediate_size; /* 3072 */
    /* Speech decoder config */
    int sd_pre_layers;        /* 8 */
    int sd_upsample_rates[4]; /* {5, 4, 4, 6} */
    int sample_rate;          /* 24000 */
} qwen_tts_config_t;

/* ── Talker Layer (riuso da decoder ASR) ── */
typedef struct {
    uint16_t *wq_bf16, *wk_bf16, *wv_bf16, *wo_bf16;
    float *q_norm, *k_norm;           /* Per-head RMSNorm */
    float *input_norm, *post_attn_norm;
    uint16_t *gate_bf16, *up_bf16, *down_bf16;
    uint16_t *gate_up_fused_bf16;     /* Pre-fused per single-token */
} qwen_talker_layer_t;

/* ── Code Predictor Layer ── */
typedef struct {
    uint16_t *wq_bf16, *wk_bf16, *wv_bf16, *wo_bf16;
    float *q_norm, *k_norm;
    float *input_norm, *post_attn_norm;
    uint16_t *gate_bf16, *up_bf16, *down_bf16;
} qwen_cp_layer_t;

/* ── Speech Decoder ── */
typedef struct {
    /* 16 codebook embedding tables */
    float *codebook_emb[16];          /* [2048 × 128] ciascuno */
    /* 8 pre-transformer layers (self-attn + FFN) */
    /* ... layers ... */
    /* 4 upsampling blocks (ConvTranspose1d + dilated Conv1d) */
    /* ... conv weights, biases ... */
} qwen_speech_decoder_t;

/* ── Contesto Principale ── */
typedef struct {
    qwen_tts_config_t config;
    /* Weights */
    qwen_talker_layer_t talker_layers[28];
    uint16_t *tok_embeddings_bf16;    /* [151936 × hidden] */
    uint16_t *codec_embeddings_bf16;  /* [3072 × hidden] per talker codec head */
    float *talker_norm;               /* Final RMSNorm */
    qwen_cp_layer_t cp_layers[5];
    qwen_speech_decoder_t speech_dec;
    /* Safetensors */
    void *safetensors;
    char model_dir[512];
    /* KV Cache (Talker) */
    float *kv_cache_k, *kv_cache_v;
    int kv_len, kv_max;
    /* KV Cache (Code Predictor) — piccolo, resettato per frame */
    float *cp_kv_k, *cp_kv_v;
    /* Buffers di lavoro */
    float *dec_x, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_ffn_out;
    /* RoPE */
    float *rope_cos, *rope_sin;
    /* Sampling */
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
    /* Output */
    float *audio_buffer;              /* Waveform accumulata */
    int audio_len, audio_cap;
    /* Tokenizer */
    void *tokenizer;
    /* Callback streaming */
    void (*audio_cb)(float *samples, int n_samples, void *userdata);
    void *audio_cb_userdata;
    /* Perf stats */
    double perf_total_ms, perf_talker_ms, perf_cp_ms, perf_decode_ms;
    int perf_frames;
} qwen_tts_ctx_t;
```

---

## 5. Dove Ottimizzare con BLAS

Analisi del costo computazionale per frame audio (1 frame = 80ms di audio):

### 5.1 Hotspot per Importanza

| # | Operazione | Dimensioni matrice (1.7B) | Frequenza | Priorità BLAS |
|---|-----------|--------------------------|-----------|---------------|
| 1 | Talker Q/K/V proj | [2048 × 2048] × 3 | 1/frame | **Critica** |
| 2 | Talker output proj | [2048 × 2048] | 1/frame | **Critica** |
| 3 | Talker FFN gate+up | [2048 × 6144] × 2 | 1/frame | **Critica** |
| 4 | Talker FFN down | [6144 × 2048] | 1/frame | **Critica** |
| 5 | CP Q/K/V proj | [1024 × 1024] × 3 | 15/frame | **Alta** |
| 6 | CP FFN | [1024 × 3072] × 2 | 15/frame | **Alta** |
| 7 | Conv1d upsampling | Varie | 1/frame | Media |
| 8 | Pre-transform attn | [128 × 128] | 1/frame | Bassa |
| 9 | Codebook lookup | [16 × 128] | 1/frame | Trascurabile |

### 5.2 Strategia BLAS

```
                   ┌─────────────────────────────────┐
                   │     BLAS (Accelerate/OpenBLAS)   │
                   │                                   │
                   │  cblas_sgemm / cblas_sgemv        │
                   │  ──────────────────────────────   │
                   │  • Talker: tutte le linear proj   │
                   │  • Code Predictor: linear proj    │
                   │  • Speech Decoder pre-transform   │
                   │  • Prefill: batch matmul (sgemm)  │
                   │  • Single-token: matvec (sgemv)   │
                   └─────────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   │                     │
          ┌────────▼─────┐     ┌─────────▼────────┐
          │  SIMD Kernels │     │  Generic Fallback │
          │  (NEON / AVX) │     │  (pure C)         │
          │               │     │                   │
          │ • bf16→f32    │     │ • bf16→f32        │
          │ • matvec_bf16 │     │ • softmax         │
          │ • dot product │     │ • RMSNorm         │
          │ • argmax      │     │ • SiLU/GELU       │
          │ • activation  │     │ • RoPE            │
          └───────────────┘     └───────────────────┘
```

**Per le convoluzioni 1D** dell'upsampling: implementare come matmul tramite
im2col (ristrutturazione input) → BLAS sgemm. Alternativa: kernel diretto
SIMD per conv1d che è più cache-friendly per dimensioni piccole.

---

## 6. Struttura File del Progetto

```
qwen-tts/
├── Makefile                        # Build system (da qwen-asr, adattato)
├── PLAN.md                         # Questo file
├── main.c                          # CLI entry point (~200 LOC)
├── qwen_tts.h                      # Header principale, structs, API (~300 LOC)
├── qwen_tts.c                      # Orchestrazione pipeline (~1500 LOC)
├── qwen_tts_talker.c               # Talker forward pass (~500 LOC)
├── qwen_tts_code_predictor.c       # Code Predictor MTP (~400 LOC)
├── qwen_tts_speech_decoder.c       # ConvNet decoder + embedding (~600 LOC)
├── qwen_tts_audio.c                # WAV writer 24kHz (~200 LOC)
├── qwen_tts_sampling.c             # Sampling strategies (~200 LOC)
├── qwen_tts_tokenizer.c            # BPE tokenizer (da ASR, adattato) (~650 LOC)
├── qwen_tts_safetensors.c          # Safetensors loader (da ASR, copia) (~394 LOC)
├── qwen_tts_kernels.c              # Core math (da ASR, copia) (~1247 LOC)
├── qwen_tts_kernels_neon.c         # ARM NEON (da ASR, copia) (~274 LOC)
├── qwen_tts_kernels_avx.c          # x86 AVX (da ASR, copia) (~502 LOC)
├── qwen_tts_kernels_impl.h         # Arch dispatch (da ASR, copia) (~50 LOC)
└── python_reference.py             # Implementazione Python di riferimento
```

---

## 7. Piano di Implementazione per Fasi

### Fase 1: Scaffolding e Infrastruttura (riuso da ASR)

1. **Copiare e adattare** i file riusabili da `../qwen-asr`:
   - Kernels (math, NEON, AVX, impl.h) → rinominare prefisso `qwen_asr_` → `qwen_tts_`
   - Safetensors loader → copia diretta con rename
   - Tokenizer → copia con aggiunta token TTS speciali
   - Makefile → adattare nomi file sorgente

2. **Creare** `qwen_tts.h` con tutte le strutture dati (config, layers, ctx)

3. **Creare** `main.c` con parsing CLI:
   ```
   Usage: qwen_tts [options]
     --model-dir <path>    Model directory
     --text <string>       Text to synthesize
     --output <path>       Output WAV file (default: output.wav)
     --speaker <id>        Speaker ID (0-8, for CustomVoice)
     --temperature <f>     Sampling temperature (default: 0.9)
     --top-k <n>           Top-k sampling (default: 50)
     --top-p <f>           Top-p sampling (default: 1.0)
     --rep-penalty <f>     Repetition penalty (default: 1.05)
     --stream              Enable streaming output
   ```

### Fase 2: Model Loading

1. **Implementare** `qwen_tts.c:qwen_tts_load()`:
   - Caricare `config.json` per determinare dimensioni modello
   - Aprire safetensors (multi-shard)
   - Mappare weights del Talker (bf16, mmapped)
   - Mappare weights del Code Predictor
   - Caricare speech tokenizer decoder weights (dalla subdirectory `speech_tokenizer/`)
   - Pre-fondere gate+up weights del Talker (come fa ASR)
   - Allocare KV cache e buffers

2. **Implementare** mapping nomi tensor:
   - Talker: `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` etc.
   - Code Predictor: `model.subtalker.layers.{i}...`
   - Speech Decoder: `speech_tokenizer.decoder...`
   - Verificare nomi esatti dai safetensors ufficiali

### Fase 3: Talker (LLM Forward Pass)

1. **Implementare** `qwen_tts_talker.c`:
   - `qwen_talker_prefill()` — multi-token forward (prompt iniziale)
   - `qwen_talker_step()` — single-token forward (generazione autoregressive)
   - Riusare pattern identico al decoder ASR:
     - RMSNorm → QKV proj (bf16 matmul) → per-head QK norm → RoPE
     - Causal GQA attention → KV cache → output proj
     - Post-attn RMSNorm → SwiGLU FFN (fused gate+up)

2. **Implementare** `qwen_tts_sampling.c`:
   - Temperature scaling: `logits[i] /= temperature`
   - Top-k filtering: sort parziale, azzerare sotto k-esimo
   - Top-p (nucleus): sort per probabilità, azzerare sotto soglia cumulativa
   - Repetition penalty: `logits[id] /= penalty` per token già generati
   - Sampling: distribuzione categorica con random uniforme

### Fase 4: Code Predictor

1. **Implementare** `qwen_tts_code_predictor.c`:
   - Stesso pattern del Talker ma con 5 layer e hidden=1024
   - KV cache separato, resettato ad ogni frame
   - Loop interno: per ogni frame, 15 pass sequenziali
   - Ogni pass condizionato su hidden states del Talker + codebook precedenti
   - Sampling con parametri separati (`subtalker_temperature`, `subtalker_top_k`)

### Fase 5: Speech Tokenizer Decoder

1. **Implementare** `qwen_tts_speech_decoder.c`:
   - **Codebook embedding**: lookup + somma residuale RVQ (16 codebook)
   - **Pre-transform**: 8 self-attention + FFN layers su sequenza [N_frames × 128]
   - **Upsampling**: 4 blocchi ConvTranspose1d + Conv1d dilated
     - Implementare conv1d come loop diretto o im2col + BLAS
     - ConvTranspose1d (deconvoluzione) per upsampling
   - **Tanh finale** per clamping output a [-1, 1]

2. **Implementare** `qwen_tts_audio.c`:
   - WAV header writer (24 kHz, 16-bit PCM, mono)
   - Float32 → int16 conversion con clipping
   - Supporto streaming: scrivi chunks progressivamente

### Fase 6: Orchestrazione e CLI

1. **Completare** `qwen_tts.c`:
   - `qwen_tts_generate()` — pipeline completa testo → audio
   - Prompt construction con ChatML format
   - Loop di generazione: Talker step → Code Predictor → accumula codici
   - Dopo generazione completa: Speech Decoder → WAV output
   - EOS detection (token `tts_eos` = 2150 nel codec vocab)
   - Performance stats

2. **Modalità streaming**:
   - Generare N frame → decodificare chunk → callback audio
   - Il decoder causale permette chunk-by-chunk senza lookahead
   - Latenza primo pacchetto: ~100ms (target)

### Fase 7: Testing e Validazione

1. **Python reference** (`python_reference.py`):
   - Script Python minimale che esegue inferenza con il modello HuggingFace
   - Salva output intermedi (logits, codici, embeddings) per confronto
   - Genera WAV di riferimento per test di regressione

2. **Test di correttezza**:
   - Confrontare output C vs Python per stesse frasi
   - Verificare qualità audio soggettiva
   - Benchmark RTF (Real-Time Factor)

---

## 8. Approccio Implementativo: Stile Antirez

Principi da seguire, coerenti con il codice ASR:

1. **No abstraction layers inutili**: struct piatti, funzioni dirette
2. **Nomi chiari**: `qwen_talker_step()`, `qwen_codebook_lookup()`, non `forward()`
3. **Commenti "perché", non "cosa"**: spiegare le scelte, non il codice ovvio
4. **Memory-mapped weights**: bf16 direttamente da safetensors, zero-copy
5. **Allocazione esplicita**: malloc/free visibili, no hidden allocators
6. **Buffers riusati**: allocare una volta, espandere con doubling
7. **BLAS dove conta**: linear projections grandi, non overhead per lookup piccoli
8. **SIMD per i kernel caldi**: bf16→f32 matvec, dot products, activations
9. **Un file, una responsabilità**: talker, code_predictor, speech_decoder separati
10. **Build semplice**: un Makefile, `make blas` e via

---

## 9. Modelli da Supportare (Fase 1)

Per la prima versione, focus su:

- **Qwen3-TTS-12Hz-0.6B-CustomVoice** — più piccolo, 9 speaker predefiniti
- **Qwen3-TTS-12Hz-1.7B-CustomVoice** — migliore qualità, stessi speaker

Download:
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./models/0.6B
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./models/1.7B
```

**Fase 2** (future): supportare Base (voice cloning) e VoiceDesign.

---

## 10. Note su Performance CPU con BLAS

Il TTS è più compute-intensive dell'ASR per secondo di output:

- **ASR**: encoder (batch, una volta) + decoder (~5 token/s audio per token testo)
- **TTS**: 12.5 frame/s × (28 Talker layers + 5×15 CP layers) = 12.5 × 103 = **1287.5 layer evaluations/s**

Per il modello 1.7B su CPU, stimiamo:
- Singolo matvec [2048 × 2048]: ~0.1ms con BLAS ottimizzato su M3 Max
- Talker step completo: ~5-10ms (28 layers × ~0.2ms ciascuno)
- Code Predictor per frame: ~3-5ms (75 layer evaluations × ~0.05ms)
- Speech Decoder per frame: ~0.5-1ms
- **Totale per frame**: ~10-15ms → **RTF ~0.12-0.19** (target realtime: RTF < 1.0)

Su Apple Silicon con Accelerate: dovrebbe essere possibile realtime anche con 1.7B.
Su Linux con OpenBLAS: probabilmente OK con 0.6B, borderline con 1.7B.

---

## 11. Rischi e Mitigazioni

| Rischio | Impatto | Mitigazione |
|---------|---------|-------------|
| Speech decoder ConvNet complesso | Alto | Analizzare weights reali, implementare conv1d incrementalmente |
| Nomi tensor non documentati | Medio | Script Python per estrarre mappa nomi da modello HuggingFace |
| Qualità audio scarsa senza fine-tuning sampling | Medio | Parametri di sampling identici al reference Python |
| Memory usage elevato (1.7B) | Basso | mmap per weights, buffers minimali, stessa strategia ASR |
| Conv1d lento su CPU senza BLAS | Medio | im2col + sgemm, o kernel SIMD dedicato |

---

## 12. Primi Passi Concreti

1. Scaricare modello 0.6B-CustomVoice
2. Script Python per:
   - Elencare tutti i nomi tensor nel safetensors
   - Eseguire inferenza e salvare output intermedi
   - Generare WAV di riferimento
3. Copiare e adattare infrastruttura da qwen-asr
4. Implementare model loading (verificare struttura weights)
5. Implementare Talker step-by-step, validando vs Python
6. Implementare Code Predictor
7. Implementare Speech Decoder
8. Integrazione end-to-end
9. Ottimizzazione e benchmark
