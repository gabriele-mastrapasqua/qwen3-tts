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
| **`[yawn]`** — preset | `哈啊` (CN) | **7** | ryan / vivian / other presets |
| **`[yawn]`** — clone | `哈啊` (CN) | **42** | `--load-voice` clones |
| **`[wow]`** | `哇` (CN) | **7** | universal — "wow!" interjection (pair with `--emotion surprise`) |
| **`[giggle]`** | `嘿嘿` (CN) | **42** | universal — sly giggle (pair with `--emotion joy`) |
| **`[scoff]`** | `切` (CN) | **7** | disdain/scoff · **T1.0** (per-tag; 1.1 over-drives pitch) · pair with `--emotion disgust` |
| **`[phew]`** | `呼` preset / `唉` clone | ryan 7 · vivian 42 · clone 42 | relief exhale; clone READS 呼 → graceful `唉` sigh fallback |

`[yawn]` added 2026-07-07 (discovered + ear-validated via the E1 harness; wired w/ a preset-vs-clone
`voice_class` split). `[moan]`/`[throat]` stay ryan-only (unshipped, under research for a generalized
trigger); cry unsolved (needs FT).

### T4 laugh variants + moan/throat generic (2026-07-07, ear verdicts) — laugh ladder needs re-tune
- **T4 laugh (ryan EN, 哈×N) — ⚠️ named variants NOT cleanly achievable, PARKED.** Ear across a full re-hunt
  (哈/哈哈/哈哈哈哈/哈哈哈哈哈 × seeds {7,42,100,256,777,2024} + T0.8): **哈哈 s7 (2.4s) too long for a "short"**
  (model always renders a FULL laugh ~2.4s floor; single 哈 paradoxically gives 5-7s or derails; T0.8 → longer
  not shorter); **哈哈哈哈哈 s7 (11.7s) laughs well but too long**; **哈哈哈哈 s256 (7.7s) METALLIC/fails** (CLAP
  0.81 = false positive, precision 0.75). PATTERN: clean laughs are all **seed 7 and scale LONG with onom
  length**; shortening (other seed / fewer chars→derail or more chars@non-7→metallic) breaks them. ⇒ no clean
  crisp-short nor right-sized-long. Only the MEDIUM `哈哈哈` s7 (shipped `[laugh]`) is solid. Named `[laugh:short|
  long]` would need DSP (`--rate`, but it's global) or FT → **PARK T4, keep the single shipped `[laugh]`.**
- **[moan] generic:** ❌ **vivian `嗯` s256 = METALLIC** (starts as a moan, then laughs, then metallic mumble) —
  not shippable. Clone 嗯/哈啊 → hum/pant. moan does NOT generalize; ryan-only at best. PARK (research later).
- **[throat] generic:** ❌ KO — CNN14 (has "Throat clearing") P=0 on 咳/呵/嗯哼. Articulatory ceiling. PARK / FT.

### Broad exploration net (2026-07-07, ryan, semi-autonomous CLAP/CNN14 screen → ear) — NEW WINS
Wide onomatopoeia net across playful/disdain/surprise/exhale buckets; screener clustered, ear judged.
Audio: `samples/tests/2026-07-07_para_broad_explore/WINS/`. **New candidate tags (ryan-validated, cross-voice
+ naming pending):**
| proposed tag | onom | seed | ear verdict |
|---|---|---|---|
| **`[wow]`** | `哇` | 7 | ✅✅ **TOP** — perfect "wow!" (2.4s, crisp — the short interjection laugh couldn't do) |
| **`[oh]`** | `噢` | 7 | ✅✅ **TOP** — perfect "oh!" (2.6s) |
| **`[phew]`** (relief) | `呼` | 7 | ✅✅ **TOP** — relief sigh, distinct from sad `[sigh]` (7.0s) |
| **`[giggle]`** (sly) | `嘿嘿` | 42 | ✅✅ **TOP** — sly/knowing chuckle (4.8s) |
| **`[hey]`** (recognition) | `咦` | 7 | ✅ WIN — "hey, it's really you?" (2.7s), not a plain huh |
| **`[huff]`** (tired) | `嗤` | 7 | ✅ WIN — "uff uff" 2× exertion/tiredness pant |
| **`[scoff]`** (disdain) | `切` | 7 | ✅ WIN but **too strong** (emo raises pitch) — needs strength ↓ |
| — | `咯咯` | 7 | 🟡 cackle but METALLIC — reduce force |
| — | `唔` | 7 | 🟡 groan attempt, metallic/forced — reduce force |
| — | `嘻嘻` | 7 | ❌ KO — forced "eh eh eh" pant |
Recurring lesson: several wins are ear-good but **too forceful/metallic** (咯咯/唔/切) — the emotion push
over-drives them; a milder emotion / no-emotion take may clean them up (a strength knob for para). The clean
TOPs (哇/噢/呼/嘿嘿) don't need it. Next: cross-voice the TOPs → wire into para_pick (like `[yawn]`); strength-
tune the metallic ones.

### Strength-tune of the metallic wins (2026-07-07) — the lever is TEMPERATURE, not removing emotion
Verified (non-silent) the onomatopoeia WAS in the prompt; the "no-emo" versions still under-perform because
**the para needs the emotion to FIRE** (no-emo → the model reads the sentence flat). So the strength lever is
a MIDDLE temperature, not dropping the emotion:
- **`[scoff]` 切 → T1.0 + disgust = WIN** (T1.1 over-drove the pitch; no-emo did nothing; T0.9 too weak). SHIPPED
  with a per-tag temperature (para_pick now returns temp; `[scoff]`=1.0, others=1.1).
- **唔 (groan): DROPPED** — my carrier mismatch (swept with the gasp/surprise carrier "Wait… is that you?",
  wrong context for a groan). The real groan trigger is `哼` s42 (T3). Don't re-chase 唔.
- **咯咯 (cackle): DROPPED** — laughs too long/won't stop at any T. Not a clean cackle.


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

### T3-val — cross-voice validation on ryan + vivian (2026-07-01) → 4 new events CONFIRMED (per-voice seed)
The 4 discovery wins hold across voices, each with its own winning seed (pin per voice):
| tag | trigger | galatea | ryan | vivian |
|---|---|---|---|---|
| **`[gasp]`** | `啊` | s7 | **s42** ✅ "ahhh è incredibile" | **s42** ✅ |
| **`[groan]`** | `哼` | s42 | **s42** ✅ clean groan | **s42** ✅ (a sharp angry TSK) |
| **`[mmm]`** | `嗯` | s7 | **s7** ✅ | **s7** ✅ |
| **`[yawn]`** | `哈啊` | s42 | **s7** ✅ | **s7/s42** ✅ |

⇒ **mmm = s7 universal · groan = s42 universal · gasp/yawn = per-voice seed.** All 4 → promote to HAVE + map into
`para_pick` (per-voice seed like `[sigh]`). s7/s42 not-listed cells = metallic/errors (KO).

### New serendipitous candidates (this round — SAVE per the PRINCIPLE)
| candidate | trigger | voice · seed · emo | note |
|---|---|---|---|
| **angry laugh** (risata rabbiosa) | `哼` | ryan · s7 · anger | long scornful/stizzosa laugh — distinct from `[scoff]` (哈哈 s42) |
| **tsk / disapproval** | `哼` | vivian · s42 · anger | sharp angry tongue-click "TSK!" (groan-family variant) |
| **pleasure / aroused** | `嗯` | vivian · s42 · joy | "godimento ahhhwww" — ends metallic → explore other seeds on vivian |
| **`[aww]` tender-wonder** | `呜呜` | galatea · any seed · sad | soft cooing "uooo" — adoring/tender ("che bel cucciolo tenero"). NOT a cry. |

### T3-explore — 2-class seed rule CONFIRMED + the `嗯` seed→meaning goldmine + cross-lang (2026-07-01)
**A) 🤯 `嗯` (mmm) is a MULTI-EVENT trigger — the SEED selects the meaning** (vivian IT, joy). One CN char, many
events — this is the T4 variants engine in miniature:
| trigger · voice · seed | event | candidate tag |
|---|---|---|
| `嗯` vivian s100 | mmm of taste/pleasure (food), short | `[mmm]` (pleasure-taste) |
| `嗯` vivian s256 | aroused "gode, eh eh, mmm delizioso" | `[moan]` (pleasure-aroused) |
| `嗯` vivian s777 | scornful/dismissive "mmm!" | `[hmpf]` (disdain) |
| `嗯` vivian s1234 | dry "mm-mm, sì sì" confirmation | `[mhm]` (confirmation = Step confirmation-en) |
| `嗯` vivian s2024 | tired/sated sighed mmm | `[mmm]` (tired-pleasure) — a variant |
| `嗯` vivian s333 | ❌ laughs then long yawn, doesn't speak | KO |

**B) 2-CLASS SEED RULE — CONFIRMED on a 2nd clone (quijote ES):** clones share a seed, presets share another.
- **`[gasp]` `啊`: CLONE=s7 (galatea ✅ + quijote ✅), PRESET=s42 (ryan ✅ + vivian ✅)** — the clean 2-class split.
- `[groan]` `哼` = **s42 universal** (quijote → a "TSK" win too). `[mmm]` `嗯` "delizioso" = **s7 universal** (quijote win).
- `[yawn]` `哈啊`: CLONE=s42 / PRESET=s7 (quijote works at both, s7 a touch cleaner — slight tail noise either way).
- quijote s42 gasp = a nice **longer gasp variant** (sospirato). → **⇒ MAP RULE = per-event {clone-seed, preset-seed},
  not per-voice** (vivian only deviates on `[sigh]` onom = `ahh`). Test a 3rd clone later to lock it.

**C) Cross-language:** gasp `啊` + groan `哼` hold on **ryan-ENGLISH** ✅ (CN triggers are language-agnostic). mmm-EN
a touch metallic at the end; yawn-EN = a short "mini-yawn" (a yawn variant). → CN onomatopoeia works across langs.

### CRY — dedicated hunt plan (user 2026-07-01: "a real cry almost came with strongly-pushed sad + steer in some sentences")
Cry needs its own sweep — it's the hardest (acoustically ≈ laugh/yawn, and low-arousal). Ideas to try:
- **Onomatopoeia candidates:** `呜呜` (wuwu — gave yawn here), `呜咽` (sob), `啜泣`, `哇` (loud wail — also surprise),
  Latin `uhuh`/`sob`/`snif`/`buaa`, and a broken `...ah...ah...`.
- **Strong sad push:** `--emotion sad` + a very vivid crying `--instruct` ("Break down crying, voice trembling and
  sobbing, on the verge of tears") — the user saw cry emerge under strongly-pushed sad.
- **WIDE seed pool** (the key — cry is seed-fragile): sweep {7, 42, 100, 256, 777, 2024, 1234, 88, 333} not just 7/42.
- **Sentence-dependent:** try several emotional carriers (grief-loaded text) — cry surfaced only in some sentences.
- Harvest anything clean even if re-labeled (per the PRINCIPLE above).

**Hunt #1 result (2026-07-01, galatea, sad + crying instruct):** ❌ no real cry yet. Findings:
- `呜呜` × 9 seeds {7…333} came out **identical (3.9s each)** → the strong crying `--instruct` PINS the trajectory,
  so **seed does not vary it here** — for cry the lever is the ONOMATOPOEIA + the SENTENCE, not the seed. And `呜呜`
  performs as a soft **"uooo"** tender coo → logged as the `[aww]` candidate above, NOT a cry.
- `呜咽` and `sob` are **READ as words** ("wuyè" / "sob") — the model doesn't perform them. `哇` = "waaah", unclear.
- NEXT cry idea: vary the CARRIER/onomatopoeia (broken `...ah...ah...`, `singhiozzo`, sniffle `snif snif`) rather
  than seed; or accept that cry ≈ the hardest vocal event and park it. Real-cry may need FT (like the pros).

---

## 2026-07-07 — first AUTOMATED discovery pass (harness E1: `tools/para/para_sweep.sh` + CLAP judge)
Method change: generated a trigger×seed×lang grid and **auto-screened with CLAP** (`para_judge.py`, τ0.20),
so the ear only judged the shortlist (4 clips out of 24) instead of every clip. Voice: ryan, 1.7B, T1.1.
The CLAP screener is calibrated only for laugh/sigh → for new events its probs are RELATIVE shortlisting
signal; ear is decisive (verdicts below are the USER's ear, 2026-07-07).

| tag | trigger | voice · lang · seed · emo | CLAP | EAR verdict |
|---|---|---|---|---|
| **`[yawn]`** (tired) | `哈啊` | ryan · EN · **s7** · (no emo) | 0.36 WIN | ✅ **TOP** — sbadiglio di stanchezza. Nit: the following speech comes out *slightly faster* (minor). Re-confirms the shipped preset seed. |
| **`[moan]`** (NEW, pleasure) | `哈啊` | ryan · EN · **s42** · (no emo) | 0.29 WIN | ✅ **WIN** — a *pleasure/godimento* yawn (satisfied stretch), distinct from the tired `[yawn]`. Named **`[moan]`** (user-approved 2026-07-07) — pleasure/godimento vocalization, own tag. |
| **`[throat]` / tsk** (NEW trigger) | `嗯嗯` | ryan · IT · **s42** · disgust | 0.20 (labeled groan) | ✅ **TOP** — a "tsk-tsk" throat-clear (pulirsi la voce/gola). Serendipity: swept as *groan*, landed as **throat-clear** (per the PRINCIPLE, keep it). New trigger — the shipped groan stays `哼` s42. |
| groan | `嗯嗯` | ryan · EN · s42 · disgust | 0.24 WIN | ❌ separates the two "gr-gr" too much — not a clean groan. |
| gasp | `啊` | ryan · EN/IT · s7/42/2024 | all DRIFT/MISS | ❌ this pass — `啊` derailed to groan/yawn/laugh (note: `啊` DID win as gasp in the T3 runs above at the per-class seed; this carrier/lang combo didn't). |
| cry | `呜呜` | ryan · EN/IT · s7/42/2024 | P(cry)=0.00, →yawn | ❌ consistent with hunt #1: `呜呜` performs yawn/sigh-ish, never a cry. |

⇒ **3 saves (user-validated):** `[yawn]`=`哈啊` s7 (tired, re-confirmed) · `[moan]`=`哈啊` s42 (NEW) ·
`[throat]`=`嗯嗯` s42 IT (NEW, tsk throat-clear). Audio in `samples/tests/2026-07-07_{yawn,groan}_discovery/`.
**Pending:** cross-voice (vivian/clone) seed check before wiring into `para_pick`; then confirm `[moan]`
seed + whether `[throat]` needs its own preset/clone seed. Harness proved the discovery loop works.

### Cross-voice check (2026-07-07, vivian preset + galatea clone, CLAP screen) — ⚠️ ryan-SPECIFIC, did NOT generalize
Swept the 3 ryan wins on vivian + galatea to lock per-class seeds. They **do NOT transfer** (screener):
| trigger | vivian | galatea clone | read |
|---|---|---|---|
| `哈啊` (yawn/moan) | → **laugh** (P0.22–0.54, all seeds) | → nothing (top 'hum' ~0.00) | vivian LAUGHS on `哈啊`; clone at most hums — NOT a yawn |
| `嗯嗯` (throat/tsk) | → **hum** (P0.57–0.72) | → **hum** (P0.25–0.95) | `嗯`="mmm/hum" literal on these voices; the ryan tsk was voice-specific |
⇒ Like `haha` (ryan-EN-only laugh) and sigh needing `ahh` for vivian, **`[yawn]`/`[moan]`/`[throat]` are so far
ryan-only**. NOT wired into `para_pick` (would ship a broken tag on vivian/clones). Options: (a) per-voice
trigger discovery for vivian/clone (find their yawn/throat onom, as `ahh` was found for vivian sigh); (b) ship
ryan-gated; (c) park as ryan candidates. Audio: `samples/tests/2026-07-07_{yawn_xvoice,groan_throat_xvoice}/`.
CLAP uncalibrated for these events → ear should confirm the vivian-laughs / clone-hums reads before finalizing.

> ⚠️ **CORRECTION (2026-07-07): the yawn "KO" above is likely a CLAP artifact, not a real KO.** CLAP is
> calibrated ONLY for laugh+sigh; a breathy YAWN is acoustically ≈ a breathy laugh, so CLAP labeling
> vivian's `哈啊` "laugh 0.54" is probably a MISLABEL. **T3-val (2026-07-01) already ear-validated `哈啊`
> yawn on vivian (s7/s42) AND galatea clone (s42).** ⇒ `[yawn]` `哈啊` is most likely ALREADY universal;
> re-ear the existing `samples/tests/2026-07-07_yawn_xvoice/` clips to confirm, then wire. Lesson: do NOT
> trust the uncalibrated screener to REJECT an event it can't score — only to shortlist. Only `[throat]`
> (`嗯嗯`→hum on vivian/clone) genuinely needs a per-voice trigger.

### `[throat]` per-voice discovery (2026-07-07, vivian+clone, triggers `嗯哼`/`咳`/`呃`) — ❌ KO, articulatory ceiling
All 12 → hum/sigh/yawn, P≈0, and the CN triggers **derail the sentence** (garbled multilingual output:
"¡Oh no de ti no hago!", "云束株オリー"). No clean throat-clear on vivian/clone. Throat-clear is
**articulatory** (cough-family, which the doc already logs as decoder-ceiling KO) — ryan's `嗯嗯` tsk was a
lucky *vocal* rendering that doesn't reproduce. ⇒ **`[throat]` stays ryan-only / PARK** (real throat-clear
likely needs FT, same as cough/cry). Only the VOCAL family (laugh/sigh/yawn/moan/gasp/groan-哼) generalizes.

### ⇒ Session net (2026-07-07): what's wireable
- **`[yawn]` `哈啊`** — VOCAL, cross-voice OK per T3 ear (preset s7 / clone s42). Wireable into `para_pick`
  once the ear re-confirms the existing xvoice clips. NOTE: needs a **preset-vs-clone** seed split (s7/s42),
  which `para_pick`'s current `voice_class` (vivian-vs-rest) doesn't encode → small code add.
- **`[moan]` `哈啊` s42** — pleasure variant, ryan-validated; ear-check other voices before universal wiring.
- **`[throat]`** — ryan-only, articulatory ceiling → PARK.
- **cry** — EXHAUSTED (needs FT).

### Step-2 iteration (2026-07-07, ryan IT) — gasp alt-triggers KO, CRY hunt #2 KO (3rd fail)
- **gasp `倒吸` / `嘶` (sharp-inhale candidates):** all MISS/DRIFT (→yawn/hum, P≤0.10). No new gasp win.
  `啊` stays the gasp trigger (ear-validated in T3-val; CLAP just can't score gasp — a screener gap, not a
  trigger failure). gasp = DONE via `啊`.
- **CRY hunt #2 `呜呜呜` / `啊呜` / `呜哇` / `buaa` × {42,2024}:** **all P(cry)=0.00, drift to yawn/sigh** — same
  failure as hunt #1. Transcripts show breathy "Ah... Ah..." (yawn/sigh-like), never crying; some triggers
  read/garbled (`buaa`→"Boa"). **3rd automated failure across many triggers/seeds → CRY via inline
  onomatopoeia is EXHAUSTED.** Cry is the decoder-ceiling event; a real cry needs FT (like the pros) — park
  it. (Cross-cutting note: nearly every breathy TTS vocalization clusters as "yawn"/"sigh" acoustically —
  cry's low-arousal sob just isn't in the model's inline reach.)

---

## Status legend
✅ WIN (promote to the `[tag]` map) · 🟡 interesting/partial (keep, needs a pick) · ❌ KO (do not re-run) · ↪ produced a different event.

_Extend this table with every new para experiment. When a WIN is stable, wire it into the `[tag]`→inline
mapping in main.c and note the commit here._
