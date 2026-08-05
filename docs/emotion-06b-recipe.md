# THE 0.6B recipe — emotion, paralinguistics and cloning on the **small** model

> Companion to [`emotion-THE-recipe.md`](emotion-THE-recipe.md), which is the source of truth for the
> **1.7B**. This file is the source of truth for the **0.6B**, where the mechanism is *completely
> different*. Do not apply the 1.7B recipe here: on the 0.6B `--emotion` (steer/expr/COMBINE) is a
> **no-op** — the engine gates it on `hidden_size >= 2048` and it fails silently.
>
> Ear-validated 2026-08-05. Trail: `samples/tests/2026-08-05_06b_*`.

## The one idea

> **On the 0.6B, emotion is not an inference-time lever — it is a property of the VOICE.**

The small model has no steerable emotion subspace (measured: emotion identity sits early, at L0-6,
tangled with language; late layers are 0.59-collinear across emotions — see `plan_06b_emo.md` §1.3,
and the archived 2026-06-30 failures of steer / CSP-FT / COMBINE). Every attempt to *inject* a
direction failed.

But the 0.6B **clones voices very well**. So the emotion enters through the channel that already
works: you clone from *emotional* audio, and you get an emotional voice.

## The fast path — a cloned voice emotes with no setup at all

The emotional offset in ECAPA speaker space is largely **speaker-independent**. Measured on two
different speakers (a preset and a clone), per emotion:

| emotion | cos(d_speakerA, d_speakerB) | ‖d‖ / ‖x‖ |
|---|---|---|
| sad | +0.68 | 0.23 |
| anger | +0.66 | 0.26 |
| fear | +0.58 | 0.24 |
| surprise | +0.59 | 0.25 |
| disgust | +0.54 | 0.24 |
| joy | +0.48 | 0.18 |

Random 1024-dim vectors would sit near 0, so roughly half of the emotional shift is a shared
direction rather than a property of the person. Six averaged unit directions ship as
`presets/emovoice/dir_<tok>.bin` (4 KB each). At runtime the engine adds one to whatever x-vector is
loaded and restores the original norm — pure arithmetic, no extra model, no generation:

```bash
./qwen_tts -d qwen3-tts-0.6b --load-voice myvoice.qvoice --icl-only --int8 \
    -l Italian --emotion anger --text "[sigh] ..." -o out.wav
```

Dose with `--emotion-strength` (default **0.25**; 0.35 pushes harder). Ear-validated 2026-08-05, on a
cloned voice using a direction derived from a *different* speaker.

**Resolution order**: dedicated per-voice asset (below) → generic direction → explicit error. The
dedicated asset always wins when present; it is stronger and more faithful, because it is rendered
from that voice actually performing the emotion rather than displaced toward it.

## The high-quality path — a dedicated asset per (voice × emotion)

### Runtime — small model only, one flag, nothing else

```bash
./qwen_tts -d qwen3-tts-0.6b --load-voice <voice>_<emo>.bin --xvector-only \
  -l Italian --seed 42 --text "…" -o out.wav
```

No `--emotion`, no `--ml-steer`, no `--expr`, no `--instruct`, no ICL prefix. The 1.7B is **not** in
the inference path. Verified: `Config: hidden=1024`, `instruct=0`, no steer/expr line in the log,
and `-j1 --temperature 0` is bit-reproducible.

### Build time — once per (voice × emotion), produces a **4 KB** file

```bash
# 1. donor: ~25 s of the TARGET VOICE, emotional. Easiest source is the 1.7B, where emotion works —
#    but ANY emotional audio of that voice works, including a real recording.
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/<voice>_graft.qvoice --icl-only \
  -l Italian --seed 42 --emotion anger --text "<~25s of text>" -o donor.wav
#    (for a PRESET voice, use the preset directly: -s ryan --emotion anger)

# 2. extract the voice from that emotional audio (ECAPA encoder, 0.6B Base model)
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio donor.wav --save-voice <voice>_<emo>.bin
```

Works for **cloned voices and presets alike** — a preset simply becomes a voice asset too.

### Optional: the 16.8 MB graft instead of the bare 4 KB

The graft additionally carries TPAD + WOVR (prosody micro-detail: sighs, pauses). Because two graft
files of *different voices* are byte-identical except for the x-vector (measured: they differ only
in bytes 12..4158 of 16,806,036), you never re-export one — you **swap the x-vector in place**:

```bash
python3 tests/graft_set_xvector.py voices/<voice>_06b_graft.qvoice \
    <voice>_<emo>.bin  <voice>_<emo>_graft.qvoice
./qwen_tts -d qwen3-tts-0.6b --load-voice <voice>_<emo>_graft.qvoice --icl-only …
```

Same speed as the 4 KB file (RTF 0.72 vs 0.73 under int8), so the choice is purely about quality.
**Corollary for distribution**: TPAD/WOVR belong to the *model*, not the voice — ship one graft per
model plus **4 KB per (voice × emotion)** cell.

### Which one to use — ear verdict 2026-08-05

| emotion | 4 KB `.bin` | 16.8 MB graft |
|---|---|---|
| sad, joy | ✅ as good as the graft | ✅ |
| **anger** | ⚠️ **speeds up and swallows a short word** | ✅ clean |

High-arousal emotions compress the delivery, and the bare x-vector has no pause/prosody scaffolding to
hold the text together — the graft's TPAD+WOVR do. So: **`.bin` for the calm emotions, graft for anger**
(and check any other high-arousal cell before shipping it as a bare `.bin`). Since the graft costs the
same at runtime, when in doubt use the graft.

## Paralinguistics — unchanged, and it works on the 0.6B

Inline `[tag]` → onomatopoeia substitution (`qwen_tts_compose.c:para_pick`) happens at the **text**
level, so it has no dependency on hidden size. `[sigh] [laugh] [wow] [yawn] [scoff]` all fire on the
0.6B, on presets and on cloned voices, and compose with an emotional voice in a single generation.

> ⚠️ **The seed behaves differently here than on the 1.7B.** On the big model the per-tag seed was
> WIN/KO and had to be pinned (`docs/para-experiments.md`). On the 0.6B `[sigh]` is **robust across
> seeds** — every seed produces a sigh, the seed only changes *which* sigh:
> `s7` = short "eh!" · `s42` = "ehhh" · `s2024` = "awhhh" · `s123` = another variant.
> So on the small model the seed is a **variety knob**, not a risk to tune — for `[sigh]`.
>
> It is *not* uniform across tags: `[laugh]` fires only at **s2024** here (the 1.7B wants s7), and
> **seed 42 is a "yawn" attractor** on this model — the laugh, wow and yawn onomatopoeia all land on
> a yawn there. So `[yawn]` uses s42 and `[laugh]` avoids it. The engine picks the right table per
> model automatically; sweeps live in `samples/tests/2026-08-05_06b_rtf_seeds/seeds/`.

**General rule for this model**: the constants in `para_pick` (onomatopoeia × seed × temperature) and
all per-voice weights were tuned on the **1.7B**. The *method* transfers to the 0.6B; the *constants*
must be re-validated. A tag that misbehaves here is a mistuned constant until a seed sweep proves
otherwise.

## Speed — everything on, still sub-realtime

M1, `-j4`, quiet machine:

| configuration | bf16 | **int8** | int4 |
|---|---|---|---|
| emotional voice (4 KB) | 1.16 | **0.73** | **0.56** |
| emotional voice (16.8 MB graft) | 1.14 | **0.72** | — |
| **emotional voice + `[tag]`** | 1.18 | **0.78** | — |
| bare 0.6B (reference) | 1.17 | 0.69 | — |

The full stack costs **~0.09 RTF** over the bare model. `--int8` preserves the emotion by ear
("stessa anger potente"), so the recommended production config is **`--int8` + a 4 KB emotional
voice**: clone + emotion + paralinguistics together at **RTF ≈ 0.78**.

## Validated by ear (2026-08-05)

- **presets: all 9 ship with all 6 emotions**, each built from a donor in the language that voice
  speaks natively (ZH `vivian`/`uncle_fu`, JA `ono_anna`, KO `sohee`, EN the rest, IT `ryan`).
  Ear-checked across languages including Korean, Japanese and English — a preset simply becomes a
  voice asset too, the recipe is not clone-only
- **all 6 also verified through the shipped `--emotion` flag under `--int8`** (sad/joy/anger, then
  fear/disgust/surprise) ⇒ the sub-1.0 configuration is the *validated* one, not a speed-only mode
- **cloned voice (galatea): works**, with the anger caveat above
- **emotion + inline `[tag]` compose** in a single generation, on presets and clones

## What is NOT yet validated

- **paralinguistics beyond four tags**: `[wow]` is weak on the 0.6B (s2024 is the least bad) and
  `[giggle]`/`[scoff]` still carry the 1.7B constants, untested here. `[sigh]`, `[laugh]` and
  `[yawn]` are ear-validated. The tag sweep also only covers one voice, one language and four seeds
- **per-emotion identity drift**: anger moves the x-vector ~0.93 cosine from neutral *for the same
  speaker*, so each emotion pulls identity somewhere — worth watching across a full matrix
- **the generic directions are averaged over only two speakers**; more voices would make them more
  robust, but changing them requires re-validating by ear
- weakest cells by objective movement: `surprise` and `sad` on the galatea clone

## Why the obvious alternatives were rejected

- **steer / CSP-FT / COMBINE natively on the 0.6B** — all failed 2026-06-30, ear-validated. Not a
  tuning problem: there is no emotion subspace disjoint from language/timbre at half width.
- **importing directions from the 1.7B** — impossible by construction: the Code Predictor has
  identical *shapes* in the two models but **cosine ≈ 0** between their weights (orthogonal learned
  bases; Base↔CV of the same size is 0.9999 for comparison). This is also why the 2026-06-30 ridge
  map had negative R².
- **`--emo-ref` (ICL codec prefix from an emotional reference)** — implemented and it does transfer
  emotion, but the ref_codes carry the **donor's identity too** and override the loaded voice
  (`qwen_tts.c:1111`: *"ref_codes are BOTH the identity carrier AND the prosody template"*). It is
  usable only when the reference is already in the target voice — and then E6 above beats it on both
  fidelity and speed (RTF 0.73 vs 6.8-11.4). Timbre is not separable from the codes
  (`QWEN_TF_CB_KEEP` experiment).
