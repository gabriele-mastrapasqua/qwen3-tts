# `.qvoice` anatomy, the `--icl-only` graft: weight + portability analysis

Answers PLAN A2 #1 (ICL vs WDELTA — peso + "funziona ovunque?") and #4 (peso per
salvare/riusare + compressione → mini `.qvoice-lite`). Measured 2026-06-09 on M1,
galatea_17b.qvoice (1.7B) + galatea_06b.qvoice (0.6B).

> ## ⭐ DEFAULT clone path = x-vector-only `.bin` (2026-06-18)
> The default way to clone is now an **8 KB x-vector-only `.bin`**:
> ```bash
> ./qwen_tts -d qwen3-tts-1.7b --load-voice voices/X.bin --xvector-only \
>   -l Italian --expr <emotion.expr> --instruct "<EN instruct>" --text "<IT text>" \
>   -T 1.3 -o out.wav
> ```
> Make the `.bin` from an existing qvoice (or straight from a ref recording):
> ```bash
> python3 tests/qvoice_to_xvec.py voices/X.qvoice
> # or, clone straight to a .bin from a ref recording (the engine, not the helper):
> ./qwen_tts -d qwen3-tts-1.7b-base --ref-audio ref_24k_mono.wav --xvector-only \
>   --save-voice voices/X.bin
> ```
> **Why x-vector-only is the default (ear-validated 2026-06-18):** the ICL `ref_codes`
> carry the reference RECORDING's **room acoustics** (a faint "muffled metallic / reverb")
> which gets re-injected on every generation, and an `.expr` **amplifies** it. The x-vector
> carries the speaker IDENTITY **without** the room → clean output, identity preserved, and
> it tolerates a **higher `.expr` weight** (the ICL amplification path is gone, so you can
> push the expr harder for the same emotional movement). **Key measured fact:** the lite
> `galatea_icl.qvoice` and the 2.8 GB `galatea_17b.qvoice` have **byte-identical x-vectors**
> (cosine 1.0000) — so the clean-vs-metallic difference was the **ICL ref_codes, NOT
> embedding quality**. Defaults: **T1.3, weight ~1.6–2.0** (`w2.5/T1.3` svaria; joy/disgust
> stay flat with weight → training ceiling, retrain `top_k=4`). Keep `--icl-only` (§5) as the
> alternative for **max timbre mimicry** from a studio-clean ref. See also
> `docs/csp-ft-emotion.md`.

## 1. What a `.qvoice` actually contains (byte breakdown)

Parser: `tests/qvoice_anatomy.py <file.qvoice>`. The file is sequential sections:

| section            | galatea_17b (1.7B)        | galatea_06b (0.6B)        | what it is |
|--------------------|---------------------------|---------------------------|------------|
| speaker embedding  | 8,192 B (2048 f)          | 4,096 B (1024 f)          | the **x-vector** (voice identity) |
| ref_text           | 0 B                       | 0 B                       | reference transcript (empty here) |
| **ref_codes (ICL)**| **0 frames / 0 B**        | **0 frames / 0 B**        | in-context codec tokens — **EMPTY** |
| META               | 104 B                     | 104 B                     | lang/name/model metadata |
| TPAD               | 24,584 B                  | 12,296 B                  | base tts_pad/bos/eos embeddings |
| WOVR               | **25,182,224 B (25 MB)**  | **16,789,520 B (16 MB)**  | base text_projection + codec_embedding override |
| **WDELTA bulk**    | **2,875.84 MB (99.17%)**  | **769.26 MB (97.96%)**    | 404 talker weight-delta tensors (LZ4 int16) |
| **TOTAL**          | **2,899.89 MB**           | **785.29 MB**             | |

### Key surprise: our voices are NOT "ICL" voices
All four of our `.qvoice` files (galatea/silvio, 0.6B/1.7B) store **0 ref_codes**.
They clone via **x-vector (speaker embedding) + WDELTA weight-delta**, *not* via
in-context-learning reference tokens. The "ICL prefix" name used in earlier notes was
loose — what `--icl-only` actually keeps from these files is the **8 KB x-vector** (+
the TPAD/WOVR overrides), not codec ref-tokens.

> The codebase still *supports* true ICL (ref_codes are read at load, line 1426-1441,
> and set `xvector_only=0`/ICL mode when present) — our creation flow just didn't store
> them. A future "ICL-format" voice would be tens of KB (≈24 KB ref_codes for 30 s).

## 2. The `--icl-only` graft: what it loads

`--icl-only` skips **only** the WFULL/WDELTA weight-swap (main.c:1617). It STILL applies
META + TPAD + **WOVR**. So the graft = CV talker weights (instruct-capable) + x-vector +
base text_proj/codec_emb (WOVR). It reads **~24.6 MB** of the 2.9 GB file; the 2.87 GB
WDELTA is never touched.

## 3. "Funziona ovunque?" — YES, all serving modes (measured)

Voice loads once at startup into the shared ctx (main.c:1334-1929), *before* dispatch to
serve/stream/batch → the graft is **mode-agnostic by construction**. Confirmed empirically
(galatea_17b graft, IT text, EN instruct, seed 42, T 1.0):

| mode                      | result                              | mel vs single |
|---------------------------|-------------------------------------|---------------|
| single                    | ✅ 133 fr, 10.64 s                  | —             |
| `--stream`                | ✅ TTFA 4.0 s                       | **1.00000**   |
| `--batch` (1 chunk)       | ✅                                  | **1.00000**   |
| `--batch` (3 chunks)      | ✅ 14.32 s, composed 3 spans        | (longer text) |
| `--serve` `/v1/tts`       | ✅ valid WAV, per-request instruct  | **1.00000**   |

Server reads `instruct`/`seed`/`temperature`/`language` per-request and **preserves the
grafted voice** in clone mode (server.c:272). For this cold single request the server was
**bit-identical** to the CLI (mel 1.0).

## 4. `.qvoice-lite` — how small can a reusable voice be? (#4)

Stripping WDELTA gives an immediate, safe **121× / 49×** shrink:

| variant                                   | 1.7B size   | 0.6B size   | vs full   | status |
|-------------------------------------------|-------------|-------------|-----------|--------|
| full `.qvoice` (WDELTA)                    | 2,900 MB    | 785 MB      | 1×        | shipped |
| **lite = x-vector + TPAD + WOVR (no WDELTA)** | **24.6 MB** | **16.4 MB** | **121× / 49×** | proven loadable via `--icl-only` (graft) |
| **pure-CV = x-vector only (no WOVR/WDELTA)**   | **~8 KB**   | **~4 KB**   | **~360,000× / ~200,000×** | coherent ✅, **different timbre (mel 0.49)** — needs ear-validation |

`QWEN_GRAFT_NO_WOVR=1` (with `--icl-only`) skips the WOVR override → pure-CV graft. It
produces fully coherent audio (normal EOS) but a **different voice** than the with-WOVR
graft (mel 0.49). So WOVR is not required for *coherence* but it *shapes the timbre*.

**The floor for a reusable graft voice therefore depends on one ear-decision:**
- if pure-CV (x-vector only) sounds good → a voice is **~8 KB** (ship thousands, send over
  the wire, embed in a request body).
- if WOVR is needed for fidelity → a voice is **~24.6 MB** (still 121× smaller than full).

Either way the WDELTA bulk (98–99% of the file) is **dead weight for the graft** and can
be dropped from a reusable-voice export. Building that exporter (`--save-voice-lite` or a
`tests/qvoice_strip.py`) is the productization step.

## Repro
```bash
# anatomy
python3 tests/qvoice_anatomy.py voices/galatea_17b.qvoice
# graft across modes
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea_17b.qvoice --icl-only \
  -l Italian --seed 42 -T 1.0 --instruct "<EN instruct>" --text "<IT text>" -o out.wav
./qwen_tts ... --stream ... ; ./qwen_tts ... --batch ... ; ./qwen_tts ... --serve 8011
# pure-CV graft (no WOVR)
QWEN_GRAFT_NO_WOVR=1 ./qwen_tts ... --icl-only ...
```

---

## 5. The TRUE ICL-lite voice format (ref_codes) — the recommended reusable clone

The graft (§2) keeps only the 8 KB x-vector and loses refinement (the clone sounds
"different"). The far better path is a **real ICL voice**: store the **reference codec
tokens (ref_codes)** and feed them in-context on the CustomVoice model. Result (M1,
2026-06-09): a clone **~identical to the full 2.9 GB qvoice by ear** at a fraction of the
size, **with the CV instruct path intact**.

### Why our old qvoices had 0 ref_codes
The save path only computes/stores ref_codes when **`--ref-text` is supplied** (main.c:1997).
Galatea/Silvio were created with `--ref-audio` but no `--ref-text` → x-vector + WDELTA only.

### Create it (on the Base model, which has the speech encoder)
```bash
./qwen_tts -d qwen3-tts-1.7b-base --ref-audio ref_24k_mono.wav \
  --ref-text "<accurate transcript of the reference audio>" \
  --save-voice voices/myvoice_icl.qvoice --voice-name MyVoice -l Italian
```
- No `--target-cv` → **no WDELTA**. File = x-vector + ref_text + ref_codes + META + WOVR (~24 MB
  @1.7B; the 24 MB is WOVR, the ref_codes themselves are ~24 KB for 30 s).
- The transcript is needed **once** (it's the ICL text↔voice anchor and gates code extraction);
  it is then baked into the file (~0.5 KB). Reuse needs no transcript. It need not be perfect.
- The audio→codes encode uses the audio only; the transcript is for the ICL prompt.

### Use it on the CustomVoice model (instruct-capable)
```bash
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/myvoice_icl.qvoice \
  -l Italian --instruct "<EN instruct>" --text "..." -o out.wav
```
Loads ref_codes → ICL mode on CV (log: `ICL: using N cached ref frames`). The CV talker
weights stay intact, so the instruct path still works. ICL inference code is model-agnostic
(qwen_tts.c:1145-1173); the Base-only guard (main.c:1335) blocks only *live* `--ref-audio`
extraction on CV, not loading pre-computed ref_codes.

### Onset/tail click fixes (ICL-specific)
The ICL decoder cold-starts at the reference→target boundary: the Talker is primed with the
reference frames (which are **not** decoded), so the first generated frame comes out already
loud → a leading **"tud"** transient (the non-ICL/qvoice path ramps from silence and is clean).
Tail likewise ends mid-energy. Fixes (default-on):
- **Onset**: drop the first generated frame(s) of decoder output in ICL mode — `QWEN_ICL_TRIM_FRAMES`
  (default **2** = 160 ms, ear-tuned; 0 disables). In the decoder thread (qwen_tts.c), so it
  covers streaming + non-streaming.
- **Tail**: asymmetric fade in the WAV writer — 5 ms in / **40 ms out** (qwen_tts_audio.c).

### Encode crash fix (`-ffast-math`)
The RVQ nearest-neighbor encode (qwen_tts_speech_encoder.c) SIGSEGVs under `-O3 -ffast-math`
on **1.7B** (a bad codegen / benign over-read that lands on an unmapped page; **harmless on
0.6B** by heap layout). Output is unaffected: ref_codes are **bit-identical** with/without the
flag (0/6000 differ). Fix: compile that one TU **without `-ffast-math`** (Makefile rule) — it's
an offline one-time path, not a hot loop. (Belt-and-suspenders root-cause with Valgrind on Linux
is a PLAN TODO — full-model ASan is unusably slow/stuck on M1.)

### Emotion caveat (still open) + the room-reverb cost of ref_codes
Emotion barely moves on an ICL voice even at T1.3 + extreme instruct: the ref_codes are a very
strong prosody anchor (that's *why* identity is ~perfect) and they damp the instruct. Trade-off:
stronger clone anchor → less emotion (x-vector graft emotes more but is a weaker/different voice).
Next lever: an `--icl-frames N` knob to dilute the anchor (fewer reference frames). Also: aggressive
sampling knobs (high rep_pen, Dawizzer-style) can break EOS → runaway generation; calibrate gently.

> **There is a second, acoustic cost to ref_codes (why x-vector-only is the DEFAULT now,
> 2026-06-18):** the ref_codes encode the reference RECORDING's **room acoustics** and re-inject
> that faint "muffled metallic / reverb" on every generation; an `.expr` amplifies it. We confirmed
> the x-vector itself is NOT the problem: the lite `galatea_icl.qvoice` and the 2.8 GB
> `galatea_17b.qvoice` carry **byte-identical x-vectors (cosine 1.0000)** → the clean-vs-metallic
> difference is the **ICL ref_codes, not embedding quality**. So x-vector-only (`--xvector-only`)
> is clean and tolerates higher `.expr` weight; ICL stays the choice when you need **max timbre
> mimicry** from a **studio-clean** ref where the room is already controlled.

### Size ladder (reusable voice)
| format | 1.7B | fidelity | emotion | notes |
|--------|------|----------|---------|-------|
| full WDELTA qvoice | 2.9 GB | perfect | ✗ frozen | the original |
| ICL-lite (ref_codes) | ~24 MB | ~perfect | weak | **max timbre mimicry** (studio-clean ref); carries the room reverb |
| **x-vector-only `.bin` (`--xvector-only`)** | **~8 KB** | **identity preserved, clean** | **✓ (tolerates high `.expr` weight)** | **DEFAULT (2026-06-18)** — no room reverb, byte-identical x-vector to the full qvoice |
