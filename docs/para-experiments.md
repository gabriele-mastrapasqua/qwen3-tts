# Paralinguistic experiments log — WIN / KO by onomatopoeia × seed × language × emotion × speaker

> **READ THIS BEFORE running any `[tag]` / paralinguistic experiment.** It is the durable, reproducible
> record of which inline onomatopoeia-triggers produce which event (or fail) — per seed, language, emotion
> and speaker. The goal: a `[tag]` maps to a **precise, deterministic** para event that **keeps the voice
> timbre and mixes into the sentence** (INLINE, one generation — never a separate span / "splice"). Do NOT
> re-run a KO combo; extend the table with new experiments and promote the WINs into the `[tag]` mapping.

## Method (the only method we keep)
- **INLINE native-trigger**: the onomatopoeia goes **inside the sentence as plain text**, ONE generation,
  so the event is produced in the active voice (preset or clone) — same timbre, mixes naturally.
- ❌ **NEVER the split-span / steering-span** ("splice"): generating the event as a separate cold-prefill
  span mixes voices (sounds like a different speaker) — rejected by ear 2026-07-01. Dead, do not reuse.
- The emotion comes from `--emotion` (STEER on presets / COMBINE on clones). Instruct is an **optional
  booster** to push a specific event ("here you laugh") — a `[tag]` must map its event **with or without** it.

## Reproducible command template
```bash
./qwen_tts -d qwen3-tts-1.7b <VOICE> -l <LANG> -T 1.1 --seed <SEED> --emotion <EMO> \
    --text "<carrier sentence with the ONOMATOPOEIA inline>" -o out.wav
# preset:  <VOICE> = -s ryan            clone: <VOICE> = --load-voice voices/galatea_graft.qvoice --icl-only
```
Carriers used below:
- joy/laugh: `"Non ci posso credere, <O>, è la notizia più bella della mia vita!"`
- sad/sigh:  `"Ho perso tutto quello che avevo, <O> e adesso non so più cosa fare."`

---

## Findings — galatea (clone), Italian, T1.1, inline (ear 2026-07-01)

### 😮‍💨 SIGH  (`--emotion sad`)  — the vocal family works inline, cross-voice
| onomatopoeia | seed | verdict | ear note |
|---|---|---|---|
| `唉` (CN) | **42** | ✅ **WIN — best short** | ottimo `ehh` breve, controllo perfetto |
| `唉` (CN) | **7**  | ✅ **WIN — short/medium** | sospiro breve + `ehh`, buon controllo, poche pause |
| `唉` (CN) | **2024** | ✅ **WIN — long** | sospiro lungo `ahhhhh` e finisce la frase pulito |
| `ahh` | **2024** | ✅ **WIN — "defeated"** | sospira, pause, emozione, esce `ehhhh` sconfitto |
| `ahh` | 42 | 🟡 interesting | `ah!` sospirato, pausa, poi finisce la frase |
| `哈哈` (CN) | 7 | 🟡 KEEP as **sigh medium-long** | NON ride: sospiro lungo, molto bello (è un sigh, non un laugh) |
| `ahh` | 7 | ❌ KO | fa `eh eh eh` 3× (non un sigh) |
| `哈哈` (CN) | 42 | ❌ KO | svaria in tanti sospiri + `eueoeo`, non finisce la frase |
| `哈哈` (CN) | 2024 | ❌ KO | `ahhhh` svaria, rumore di fondo a fine frase, troppo lungo |

→ **Sigh mapping candidates (galatea IT):** short = `唉` s42 · medium = `唉` s7 (or `哈哈` s7) · long = `唉` s2024 / `ahh` s2024.

### 😄 LAUGH — ✅ SOLVED inline both English AND Italian/clone (2026-07-01)
**Two levers that crack it: (1) SHORT onomatopoeia (`haha`, not `hahaha` — the long form over-laughs into a
pant/"godimento"); (2) the RIGHT LANGUAGE of the onomatopoeia — Chinese `哈哈哈` makes the CLONE laugh in
Italian where Latin letters only sigh (user's CN hypothesis, confirmed). No event-instruct (it goes metallic).**

**English (ryan, `--emotion joy`):**
| onomatopoeia | T | seed | verdict | ear note |
|---|---|---|---|---|
| `haha` | **1.0** | **42** | ✅✅ **TOP WIN** | risata clean e breve `ehehe`, pulita, **finisce la frase** — the shippable EN laugh |
| `haha` | 0.9 | 42 | ✅ **WIN** | ride medio-lungo, ci sta bene |
| `hahaha 哈哈` (EN+ZH mix) | 1.1 | 42 | ✅ **WIN (TOP)** | risata media bella, finisce la frase |
| `hahaha` | 0.9/1.0/1.1 | 42/7 | ❌ KO | svaria in `ah ah ah` pant/"godimento", metallico, spesso non finisce la frase |
| `哈哈` | 1.0 | 42 | ❌ KO | ride troppo + ansima, si allunga, non finisce |
| `hahaha` + laugh-instruct | 1.1 | 42 | ❌ KO | l'event-instruct lo fa svariare metallico → NON aggiungere instruct al laugh |

**Italian — galatea clone (`--emotion joy`, T1.1):**
| onomatopoeia | seed | verdict | ear note |
|---|---|---|---|
| `哈哈哈` (CN) | **7** | ✅✅ **WIN — clone laughs in IT!** | ride, breve, **con la voce clonata** — CN cracks the "IT sighs" wall |
| `哈哈哈` (CN) | 42 | ❌ KO | non ride, iperventila/affannato poi finisce la frase |
| `hahaha` (Latin) | 7/42/2024 | ❌ KO | svaria / non ride (Latin sighs in IT) |
| `哈哈` (CN, 2 chars) | 7 | ❌ KO | `ah ah!` 2×, non ride (needs the 3rd char `哈哈哈`) |
| `ehehe` | 42/7 | ❌ KO | metallico |

**⇒ LAUGH mapping:** EN/preset → `haha` @ **T1.0** s42 · clone/IT → **`哈哈哈` (CN)** s7 · EN+ZH mix `hahaha 哈哈`
also TOP. Seed is decisive (`哈哈哈` s7 laughs, s42 hyperventilates) — always pin the validated seed. SHORT form
+ no event-instruct.

### 🀄 Chinese cross-voice para on the clone (galatea IT) — the rest
| event | trigger | seed | verdict | ear note |
|---|---|---|---|---|
| cough | `咳咳` (CN) | 42 | ❌ KO (as cough) | non tossisce; sospira pulito e finisce la frase (a clean sigh, not a cough) |
| mmm/pleasure | `嗯` (CN) | 42 | ❌ KO | non fa nulla para, forse non finisce la frase |

→ CN unlocks the **vocal** family (laugh now too, sigh already) but **articulatory** cough still hits the
decoder ceiling even in Chinese. Consistent with the whole project: vocal events achievable, articulatory not.

### 🎭 SERENDIPITOUS NEW-TAG candidates (galatea IT — keep for future tags)
These did NOT laugh but produced a **clean, distinct OTHER event in-voice** — promote to their own `[tag]` later:
| sound | trigger | seed | note |
|---|---|---|---|
| **scoff / sneer** (sbeffeggio) | `哈哈` (CN, 2-char) + laugh-instruct | 42 | `AHH!` sospirato = risata breve sprezzante/di scherno |
| **pant / aroused** (ansimo) | `哈哈` (CN, 2-char) + laugh-instruct | 2024 | `ah ah ah` poi ansima — panting/aroused vocalization |

---

## Cross-voice validation (2026-07-01) — `哈哈哈` s7 is the UNIVERSAL laugh; `唉` sigh fails on vivian
Validated the galatea wins on the ryan/vivian presets (EN + IT) to confirm the mapping generalizes.

### 😄 LAUGH `哈哈哈` (CN, 3-char) @ **seed 7** — WIN across voices AND languages
| voice · lang | seed | verdict | ear note |
|---|---|---|---|
| ryan · EN | 7 | ✅ **WIN** | 2 risate forti/lunghe belle; lieve metallico a fine |
| ryan · IT | 7 | ✅ **WIN** | risata lunga `ehehe`, lieve metallico ma poco |
| vivian · IT | 7 | ✅ **WIN — breve e precisa** | ride pulito, corto |
| galatea (clone) · IT | 7 | ✅ WIN (prior) | ride con voce clonata |
| ryan · IT | 42 | 🟡 interesting | ride ma un po' forzata (`ahahah ahhh ah`) |
| vivian · IT | 42 | ❌ KO | metallico, rallenta, allunga ogni `eh eh eh` |
| **`haha` (Latin) @ T1.0** · vivian · EN | 42 | ❌ KO | NON ride: sospira `ah ahhh` sfinita/godimento — `haha` is **ryan-specific**, doesn't generalize |

**⇒ LAUGH final mapping: `[laugh]` → `哈哈哈` (CN) @ seed 7 — ONE onomatopoeia, all voices + languages.**
Seed 7 is decisive (s42 forces/derails). `haha`@T1.0 stays a ryan-EN-only clean alt; `哈哈哈` s7 is universal
(mild metallic tail on ryan is the only nit → later: trim tail or seed-tune).

### 😮‍💨 SIGH — `唉` (CN) for ryan/clone; `ahh` (Latin) for vivian (fixed 2026-07-01)
| voice · lang | onomatopoeia | seed · T | verdict | ear note |
|---|---|---|---|---|
| ryan · IT | `唉` (CN) | 42 · 1.1 | ✅ **WIN** | perfetto `ehhh` sigh |
| galatea (clone) · IT | `唉` (CN) | 42 · 1.1 | ✅ WIN (prior) | `ehh` breve, controllo perfetto |
| **vivian · IT** | **`ahh`** | **7 · 1.1** | ✅✅ **TOP WIN** | sigh sospirato medio, molto bello |
| vivian · IT | `ahh` | 42 · 1.1 | ✅ **WIN** | sospiro breve |
| vivian · IT | `唉` (CN) | 42 · **0.9** | ✅ **WIN** | pulito (la temp più bassa calma il CN) |
| vivian · IT | `唉` (CN) | 42 · 1.1 | ❌ KO | ansima stanchezza/godimento, metallico (vivian over-does 唉 at T1.1) |

| ryan · IT | `ahh` | 7 / 42 · 1.1 | ✅ **WIN (alt)** | sospira pulito: s7 medio, s42 breve |

**⇒ SIGH mapping (FINAL) — VOICE-DEPENDENT (not universal!):** ryan/clone → **`唉` @ seed 42**; vivian → **`ahh`
@ seed 7** (vivian over-does `唉` at T1.1). ⚠️ NOTE: `ahh` @ seed 7 is a WIN on ryan/vivian but was **KO on
galatea** (`3× eh eh eh`) — do NOT use `ahh` s7 as a universal sigh. `ahh` s2024 is the galatea `ahh` win.

---

## ✅ FINAL inline `[tag]` mapping — SHIPPED in main.c (commit 2a6d661, 2026-07-01)
The **user writes the friendly tag `[laugh]`/`[sigh]`**; the engine rewrites it under the hood to the
onomatopoeia below, **COMMA-DELIMITED** (`", onom, "` — the pause that makes it a discrete event), pins the
validated seed, and generates ONCE. The user NEVER types Chinese. (`para_pick`/`para_inline_substitute` in main.c.)

| tag (user writes) | onomatopoeia (engine inserts) | seed | scope |
|---|---|---|---|
| **`[laugh]`** | `哈哈哈` (CN, 3-char) | **7** | universal — ryan EN/IT, vivian, galatea clone |
| **`[sigh]`** — ryan/clone | `唉` (CN) | **42** | ryan/clone |
| **`[sigh]`** — vivian | `ahh` (Latin) | **7** | vivian only (over-does `唉`) |

Method: inline substitution, ONE `--emotion` generation @ T1.1, comma-delimited, no event-instruct, no
steering-span. Seed pinned per-tag (laugh 7 / sigh 42) when the user gave no `--seed`. voice_class = vivian vs
ryan/clone/other. Nits: mild metallic tail on ryan laugh (later). `haha`@T1.0=alt laugh (ryan-EN only),
`ahh` s2024=alt sigh (galatea). Plain text (no tag) is untouched.

> ⚠️⚠️ **PROCESS RULE (2026-07-01, learned the hard way):** ALWAYS test a `[tag]` feature the way a REAL USER
> would — write the actual `[laugh]`/`[sigh]` tag in `--text` and let the engine substitute — **never** hand-paste
> the raw onomatopoeia. The first wiring pasted `哈哈哈` WITHOUT the delimiting commas and picked the wrong
> per-voice seed → all 3 clips were garbage. The `[tag]` path (commas + per-voice seed) is the only valid test.

---

## T3 — new-event discovery (2026-07-01, galatea clone IT, CN triggers from Step-Audio-EditX + related emotion)
Reinforces the SEED rule hard: groan works ONLY at s42 (s7 = nothing), yawn works ONLY at s42 (s7 weak),
while gasp/mmm work at s7 — **each (event × trigger × voice) has its own winning seed, always pin it.**
| new tag | trigger | emotion | seed | verdict | ear note |
|---|---|---|---|---|---|
| **`[gasp]`** | `啊` (CN) | surprise | **7** | ✅ **WIN** | "ah!" breve, stupore incredulo (s42 = più `eh!` secco, weaker) |
| **`[groan]` / `[growl]`** | `哼` (CN) | anger | **42** | ✅ **WIN** | rabbia + `arggg/grrr` — molto bello (⚠️ s7 = renders NOTHING) |
| **`[mmm]` / `[pleasure]`** | `嗯` (CN) | joy | **7** | ✅ **WIN** | "mmm" breve, "davvero delizioso" (s42 = drifts language, KO) |
| **`[yawn]`** | `哈啊` (CN) | sad/tired | **42** | ✅ **TOP WIN** | sbadiglio stanco `ahh` (s7 = weaker/less effective) |
| chuckle→**sigh** | `呵呵` (CN) | joy | 7 | ↪ **WIN (re-mapped)** | didn't chuckle — makes a nice sighed `awww`/uff → keep as a 2nd **sigh** trigger |
| cry→**yawn** | `呜呜` (CN) | sad | 7 | ↪ **WIN (re-mapped)** | didn't cry — makes a clean **yawn** → keep as a 2nd **yawn** trigger (s42 has slight tail noise) |
| gasp | `哇` (CN) | surprise | 42 | 🟡 undefined | "waaah" — nice but unclear which tag; park it |

> **PRINCIPLE (user 2026-07-01): keep a WIN even if it's a DIFFERENT event than the trigger name suggested.**
> The trigger→event map is EMPIRICAL — if `呜呜` makes a lovely yawn (not the cry we aimed for), it's a WIN for
> `[yawn]`, not a KO. Build the menu from what actually sounds good; re-label freely. (cry/laugh/yawn are
> acoustically confusable, so triggers cross over — harvest whatever lands clean.)

**⇒ NEW events to add to the map (pending cross-voice validation on ryan/vivian):** `[gasp]`→`啊` s7 ·
`[groan]`→`哼` s42 · `[mmm]`→`嗯` s7 · `[yawn]`→`哈啊` s42 (+ alt `呜呜`) · **`[sigh]` gains an alt trigger `呵呵`**.
Genuinely still-open: a real **cry** (crosses over to yawn/sigh). **chuckle is NOT a separate hunt** — it's a
short **laugh variant** (`[laugh:short]`, T4: shorter `哈`/`哈哈` + seed).

### CRY — dedicated hunt plan (user 2026-07-01: "a real cry almost came with strongly-pushed sad + steer in some sentences")
Cry needs its own sweep — it's the hardest (acoustically ≈ laugh/yawn, and low-arousal). Ideas to try:
- **Onomatopoeia candidates:** `呜呜` (wuwu — gave yawn here), `呜咽` (sob), `啜泣`, `哇` (loud wail — also surprise),
  Latin `uhuh`/`sob`/`snif`/`buaa`, and a broken `...ah...ah...`.
- **Strong sad push:** `--emotion sad` + a very vivid crying `--instruct` ("Break down crying, voice trembling and
  sobbing, on the verge of tears") — the user saw cry emerge under strongly-pushed sad.
- **WIDE seed pool** (the key — cry is seed-fragile): sweep {7, 42, 100, 256, 777, 2024, 1234, 88, 333} not just 7/42.
- **Sentence-dependent:** try several emotional carriers (grief-loaded text) — cry surfaced only in some sentences.
- Harvest anything clean even if re-labeled (per the PRINCIPLE above).

---

## Status legend
✅ WIN (promote to the `[tag]` map) · 🟡 interesting/partial (keep, needs a pick) · ❌ KO (do not re-run) · ↪ produced a different event.

_Extend this table with every new para experiment. When a WIN is stable, wire it into the `[tag]`→inline
mapping in main.c and note the commit here._
