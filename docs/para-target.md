# Paralinguistics — TARGET / desiderata menu (what we want, what the pros do, what we have)

> Companion to `docs/para-experiments.md` (the WIN/KO test log). This doc is the **menu we aim for**: the set of
> `[tag]` events we want, with a clear definition of each, which pros offer it, whether WE have it (via the INLINE
> method), and the variants we want. Built from web research 2026-07-01 (§10.1/T1 of plan_emo_v3). Not one laugh —
> a *range*: multiple laughs/sighs (short/medium/long) + more vocal events, all in the active voice's own timbre.

## 1. How the pros do it (method matters)
| provider | how para is exposed | inline-in-text? | tag / cue vocabulary | notes |
|---|---|---|---|---|
| **ElevenLabs v3** | inline `[tags]` (trained model) | ✅ yes | `[laughs]/[laughing]`, `[sighs]/[sigh]`, `[clears throat]`, `[gulps]`, `[gasps]/[gasp]`, `[crying]`, `[whispers]`, `[shouts]`; cognitive `[pauses]/[hesitates]/[stammers]`; tone `[cheerfully]/[flatly]/[deadpan]/[playfully]` | biggest documented tag set; "experiment, more tags exist than documented" |
| **Bark** (Suno) | inline special tokens (trained) | ✅ yes | `[laughter]`, `[laughs]`, `[sighs]`, `[gasps]`, `[clears throat]`, `[music]`, `…` hesitation, `♪` song, CAPS = emphasis | tokens can be ignored / probabilistic |
| **Step-Audio-EditX** (Chinese) | 10 paralinguistic labels; edit-LLM | ✅ tag markers `[Laughter]`/`[Breathing]` | **breathing, laughter, surprise-oh, confirmation-en, uhm, surprise-ah, surprise-wa, sigh, question-ei, dissatisfaction-hnn** | CHINESE-trained yet multilingual → validates our CN-onomatopoeia approach |
| **CosyVoice 3** (Chinese) | inline tags **processed DURING generation** | ✅ yes (natural integration) | vocal-event tags | closest to OURS (in-generation, not post-hoc splice) |
| **Hume Octave** | natural-language **acting instructions** (`description`) + `speed` | ❌ no inline tags | — (describe delivery in prose, ≤100 chars) | instruction-based, not tag-based |
| **OpenAI gpt-4o-mini-tts** | natural-language **instructions** (tone/emotion) | ❌ no inline tags | — | instruction-based |

**Takeaways:** (1) The tag-in-text UX (ElevenLabs/Bark/Step/CosyVoice) is the industry direction — **matches ours**.
(2) Everyone else pays for it with **heavy training**; we get a subset **zero-train** via inline onomatopoeia +
seed. (3) A Chinese-trained model (Step-Audio-EditX) exposing exactly laughter/sigh/breathing/surprise/uhm
**confirms Chinese onomatopoeia is the right lever** (as we found: `哈哈哈`/`唉`/`嗯`). (4) Nobody exposes *named
duration variants* per event — an opening for us (`[sigh:long]`).

## 2. The TARGET menu — VOCAL family (what Qwen can plausibly do inline)
Legend: **HAVE** = shipped/validated (see para-experiments.md) · **CAND** = serendipitous candidate, validate ·
**TRY** = pros have it, not yet tested on Qwen · family per NVBench taxonomy (laugh/cry/scream/moan/sigh + subtypes).

| tag | should sound like | family | typical duration / when | pros with it | Qwen status | variants wanted |
|---|---|---|---|---|---|---|
| `[laugh]` | genuine laugh/giggle | laughter | 1–3s · joy, relief, mirth | EL, Bark, Step, CV | ✅ **HAVE** `哈哈哈` s7 | short giggle / medium / long belly-laugh; chuckle |
| `[chuckle]` | soft short laugh | laughter | ~0.6–1s · amused, wry | EL (implied) | TRY (`哈`? `呵呵`?) | 1 short |
| `[sigh]` | breathy exhale | sigh/resp. | 1–2s · resignation, tiredness, relief | EL, Bark, Step | ✅ **HAVE** `唉`(ryan/clone) / `ahh`(vivian) | short / medium / long; "defeated" |
| `[scoff]` / `[sneer]` | scornful short laugh | laughter | ~0.8s · contempt, derision | — (novel) | 🆕 **CAND** `哈哈` s42 (galatea) | 1 |
| `[pant]` / `[moan]` | panting / aroused exhale | moan | 1–2s · exertion, pleasure | — | 🆕 **CAND** `哈哈` s2024 (galatea) | 1–2 (careful: NSFW-adjacent) |
| `[yawn]` | tired open yawn | resp. | ~1.5s · sleepy, bored | — | TRY `haaa`/`哈啊` (native winner earlier) | 1 |
| `[mmm]` / `[pleasure]` | satisfied hum | moan | ~0.8s · savoring, assent | — | TRY `嗯` (CN, old macro) | 1 |
| `[gasp]` | sharp inhale | resp. | ~0.5s · surprise, shock | EL, Bark, Step (surprise-*) | TRY (Step: surprise-ah/oh/wa) | short / big |
| `[groan]` | low complaint | groan | ~1s · pain, annoyance | (taxonomy) | TRY | 1 |
| `[cry]` / `[sob]` | weeping | cry | 1–3s · grief | EL, Bark | TRY ⚠️ (laugh≈cry confusable — needs care) | 1 |
| `[huff]` / `[ugh]` | irritated puff | — | ~0.6s · annoyance | EL (implied) | 🟡 old DSP macro (`Uff…`/`Ugh…`) — re-do inline | 1 |

## 3. TARGET menu — ARTICULATORY / RESPIRATORY (hard — decoder ceiling)
| tag | pros with it | Qwen status | note |
|---|---|---|---|
| `[cough]` | EL(clears throat), Step | ❌ KO | decoder ceiling, even CN `咳咳` (para-experiments.md) |
| `[clear throat]` | EL, Step | 🟡 partial | `ahem` native was partial; retest inline |
| `[breath]` / `[breathing]` | Step (breathing) | TRY | Step has it trained; test inline `哈…`/soft `hhh` |
| `[sneeze]` | — | ❌ | no trigger found (truly absent) |
| `[whisper]` / `[shout]` | EL | (delivery) | not an event → do via `--rate`/`--volume`/instruct, not a para tag |

→ Rule: chase the **VOCAL family** (§2). Articulatory/§3 need training (or a new idea) — don't burn cycles.

## 4. What's genuinely OURS / novel (the pitch)
- **Zero-train inline** cross-voice para (the pros mostly train it) — event in the **voice currently in use**.
- **Chinese-onomatopoeia lever** cracks events Latin letters can't (laugh in IT: `哈哈哈` laughs, `hahaha` sighs).
- **Named duration/style VARIANTS** per event (`[sigh:short|long]`) — nobody exposes this.
- **Style-carryover**: an inline para tag can shift the MOOD/SPEED/PROSODY of the following speech → a mid-prompt
  style pivot (plan_emo_v3 §10.2/T5). Potentially a headline feature no tag-TTS advertises.

## Sources
ElevenLabs v3 audio tags · Bark (suno-ai) special tokens · Hume Octave acting instructions · OpenAI
gpt-4o-mini-tts instructions · Step-Audio-EditX technical report (10 paralinguistic labels) · CosyVoice3 tags
guide · NVBench / "Voices without words" nonverbal-vocalization taxonomy. (Full URLs in the session research log.)
