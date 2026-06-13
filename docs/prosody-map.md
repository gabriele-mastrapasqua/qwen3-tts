# Prosody layer-map of Qwen3-TTS (where prosody lives in the Talker)

Where, layer by layer, the Talker encodes **general prosody** (rhythm, intonation, pacing/
duration) — not just emotion. Measured 2026-06-12 with the `QWEN_ACT_MAP` instrument; it tells
us which layers a prosody-only LoRA should target (vs the emotion-only L16-26 band).

## Method

`QWEN_ACT_MAP=f.qamp ./qwen_tts ...` dumps the **mean Talker residual stream after every layer**
(+ final hidden) over a generation — a `[num_layers+1 × hidden]` fingerprint. `tests/act_map_diff.py`
contrasts two fingerprints → per-layer relative shift + cross-condition cosine separation.
`tests/act_map_prosody.sh` runs the whole experiment.

Contrasts vary **only prosody at constant (neutral) emotion**, per language (EN, IT):
- **Intonation**: same sentence as statement vs question (`?` only, no instruct).
- **Pacing**: same sentence + neutral vs slow vs fast instruct (neutral-instruct base cancels
  "instruct presence", isolating the pacing/duration shift).

### Rigor: capture at T0 (greedy)

At T0.9 the **seed-to-seed noise floor** (same text, different seed) was ~40–50% per layer — as
big as the weak contrasts, so intonation sat *at/below noise* and early-layer "signal" was partly
noise. **Re-running at T0 (greedy) is deterministic → noise floor = 0.0% at every layer → every
shift is pure signal.** Always capture the map at T0. (At T0.9 you must average over many sentences
to beat the floor.)

## Findings (T0, ryan 1.7B, noise-validated)

**1. Magnitude — prosody concentrates MID L07–L12 + final, but extends EARLIER (L00–L06) than emotion.**
- EN intonation & pacing: mid L07–12 + final, with L00/L03 (early) appearing.
- **Italian pacing is strongly EARLY**: slow `L03/L07/L08 ~46%, L06 43%`; fast `L00/L01/L03/L05 ~44–46%`.
- IT prosody is **bigger magnitude AND earlier** than EN.
- Compare emotion (measured 2026-06-08): magnitude mid **L06–11** + final. → prosody reaches earlier.

**2. Identity (which variant) — late for EN, mid for IT.**
- EN slow-vs-fast separated by **L20–24** (overlaps the emotion identity band L21–25).
- IT slow-vs-fast separated by **L08–12**.

**3. Prosody is strongly per-language.**
- Intonation EN-vs-IT cosine **0.095** (near-orthogonal); the language difference concentrates **L07–L10**.
- IT (the cross-lingually "switched" language) uses **earlier + larger** layers than EN.

## What this means (actionable)

| concern | band |
|---|---|
| Timbre / identity | decoder/codec (+ qvoice WDELTA) — `.expr` never touches it |
| **Emotion** | Talker **L16–26** (magnitude mid L06–11, identity late L21–25) |
| **General/linguistic prosody, pacing** | **L00–L12 + final**, extends EARLY (esp. Italian) |
| Prosody-variant identity | late L20–24 (shared with emotion) |

- The emotion `.expr` trains **L16–26** — it **misses the early prosody band L00–L12** where general/
  linguistic prosody and pacing live (esp. Italian).
- → A **"language-prosody / naturalness" LoRA should target a WIDER, EARLIER band (~L00–L12 + the
  late emotion/identity band)**, not L16–26 alone. Set the trainer's `--layers` accordingly.
- This explains two earlier observations: (a) the paralinguistic LoRA (L16–26) couldn't **place**
  nonverbals — the structural where/when lives EARLIER, outside its band → only a crude late
  override; (b) "plain Italian got richer" with an L16–26 adapter — it grazed the expressive band,
  but the big IT prosody is early (L00–L06), so real IT naturalness needs an early-band LoRA.

## Validated — the broad-band LoRA (2026-06-12, ear-confirmed)

Retraining the EMOVO Italian LoRA on the FULL band (`--layers 0-27`, r32) instead of emotion-only
L16-26 **won on all three axes at once** (`tests/lora_band_ab.sh`):
- **Emotion** stronger on presets (ryan, vivian).
- **Clone emotion** finally works — the broad band at **r32 closes the clone-gap that previously
  needed r64** (the early/prosody layers give the clone the reach it lacked). → r64 obsolete.
- **Per-language rendering**: a Chinese-native preset (vivian) renders Italian markedly better.

Loss fit much deeper (8.18→3.25 vs ~4.9 for L16-26). Adapter 90 MB (from 28 layers, **not** rank —
still r32). One r32 LoRA = emotion + clone + language rendering. Caveat: the deep fit slightly
**rushes emotional lines** — dose with `--expr-weight 0.5–0.7` or an earlier checkpoint if needed;
phonetics stayed clean (no early-layer word corruption). **Recipe: `--layers 0-27 --lora_r 32
--lora_alpha 64`** as the new default per-language pack (drop the per-voice rank distinction).

## Caveats

- **Single sentence per condition** — a content-specific component remains. T0 zero-noise makes the
  per-layer profile reliable for *that* sentence; generalize by averaging the fingerprint over many
  diverse sentences per condition.
- The shift % is **relative magnitude, correlational** — not causal. A layer-ablation/activation-patch
  would confirm a layer is *causally* responsible (vs merely correlated).

## Reproduce

```bash
tests/act_map_prosody.sh          # T0 (default) — captures + per-layer diffs to stdout
ACT_T=0.9 tests/act_map_prosody.sh   # the noisy version (needs sentence-averaging)
```
