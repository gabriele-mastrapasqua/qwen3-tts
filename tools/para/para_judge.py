#!/usr/bin/env python3
"""
E1.1 — para_judge: offline no-ear screening for paralinguistic [tag] candidates.

Given a directory of TTS wavs (a sweep of trigger x seed x voice x language), score each
clip with an audio-event tagger + whisper ASR and emit a markdown table.

Taggers (pluggable, --tagger):
  * cnn14  — PANNs CNN14 AudioSet tagger. Calibrated 2026-07-07: TRUSTED for SIGH, blind to laugh.
  * clap   — msclap zero-shot (text-prompt) tagger. The fallback for events AudioSet misses (laugh).
  * both   — run both; the verdict routes each tag to its calibrated-best backend (TAG_BACKEND).

whisper ASR gives the transcript, to catch the "literal onomatopoeia spoken" failure (the model
READ 哈哈哈 as words instead of RENDERING a laugh sound).

Verdicts (plan_v4 E1.1):
  WIN_CAND   P(event) >= tau  AND the literal onomatopoeia is NOT in the transcript
  KO_LITERAL the onomatopoeia is spoken literally in the transcript (any P)
  DRIFT      a DIFFERENT target event class dominates (serendipity: flags a WIN for another tag)
  MISS       no target event fired and nothing literal -> just neutral speech

Screener, NOT the final judge (plan E1.2): trust its shortlists only after the calibration gate;
the ear stays final judge for promotions. CPU-first (keeps the M1 cool); --device mps/cuda optional.

Usage:
  python para_judge.py --wavs DIR --onom 哈哈哈 --tag laugh --tagger clap
  python para_judge.py --manifest sweep.json --tagger both --out report.md
  python para_judge.py --manifest calib.json --tagger both --calibrate
"""
import argparse
import glob
import json
import os
import sys

# ---- CNN14: our tags -> AudioSet class names (max prob over the group). ----
CNN14_CLASSES = {
    "laugh":  ["Laughter", "Giggle", "Snicker", "Chuckle, chortle", "Belly laugh", "Baby laughter"],
    "sigh":   ["Sigh"],
    "gasp":   ["Gasp"],
    "cough":  ["Cough"],
    "sniff":  ["Sniff", "Snort"],
    "sneeze": ["Sneeze"],
    "breath": ["Breathing"],
    "throat": ["Throat clearing"],
    "cry":    ["Crying, sobbing", "Whimper", "Wail, moan", "Baby cry, infant cry"],
    "groan":  ["Groan", "Grunt"],
    "pant":   ["Pant"],
    "hum":    ["Humming"],
}

# ---- CLAP zero-shot: our tags -> a natural-language prompt. A neutral anchor competes in the softmax. ----
CLAP_PROMPTS = {
    "laugh":  "the sound of a person laughing out loud",
    "sigh":   "the sound of a person sighing",
    "gasp":   "the sound of a person gasping in surprise",
    "cough":  "the sound of a person coughing",
    "sniff":  "the sound of a person sniffling",
    "sneeze": "the sound of a person sneezing",
    "breath": "the sound of heavy breathing",
    "throat": "the sound of a person clearing their throat",
    "cry":    "the sound of a person crying and sobbing",
    "groan":  "the sound of a person groaning",
    "pant":   "the sound of panting",
    "yawn":   "the sound of a person yawning",
    "hum":    "the sound of a person humming",
}
CLAP_NEUTRAL = "a person speaking normally"

# ---- Per-tag best backend, set by the E1.2 calibration (2026-07-07). Used when --tagger both. ----
TAG_BACKEND = {"sigh": "cnn14", "laugh": "clap", "throat": "cnn14", "cough": "cnn14", "sneeze": "cnn14"}
DEFAULT_BACKEND = "clap"   # for events AudioSet misses; sigh is the proven cnn14 exception


def eprint(*a):
    print(*a, file=sys.stderr)


# ---------------------------------------------------------------- taggers ----
class Cnn14Tagger:
    name = "cnn14"

    def __init__(self, device):
        from panns_inference import AudioTagging
        from panns_inference.config import labels
        eprint(f"[load] CNN14 (panns_inference) on {device} ...")
        self.at = AudioTagging(checkpoint_path=None, device=device)
        idx = {n: i for i, n in enumerate(labels)}
        self.resolved = {}
        for tag, names in CNN14_CLASSES.items():
            got = [idx[n] for n in names if n in idx]
            miss = [n for n in names if n not in idx]
            if miss:
                eprint(f"[warn] cnn14 tag '{tag}': no AudioSet class {miss} — ignored")
            if got:
                self.resolved[tag] = got

    def probs(self, wav):
        import librosa
        audio, _ = librosa.load(wav, sr=32000, mono=True)
        clipwise, _ = self.at.inference(audio[None, :])
        cw = clipwise[0]
        return {t: float(max(cw[i] for i in ix)) for t, ix in self.resolved.items()}


def _patch_torchaudio_soundfile():
    """torchaudio>=2.9 delegates load() to torchcodec; route it through soundfile instead
    (already a dep, no ffmpeg/torchcodec needed). Returns (channels, time) tensor + sr."""
    import torchaudio, torch, soundfile as sf
    def _load(path, *a, **k):
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        return torch.from_numpy(data.T).contiguous(), sr           # (C, T)
    torchaudio.load = _load


class ClapTagger:
    name = "clap"

    def __init__(self, device):
        _patch_torchaudio_soundfile()
        from msclap import CLAP
        eprint(f"[load] CLAP (msclap 2023) on {device} ...")
        self.m = CLAP(version="2023", use_cuda=(device == "cuda"))
        self.tags = list(CLAP_PROMPTS.keys())
        self.text_emb = self.m.get_text_embeddings(
            [CLAP_PROMPTS[t] for t in self.tags] + [CLAP_NEUTRAL])

    def probs(self, wav):
        import torch.nn.functional as F
        a = self.m.get_audio_embeddings([wav])
        sim = self.m.compute_similarity(a, self.text_emb)   # [1, T+1] scaled cosine
        p = F.softmax(sim, dim=-1)[0]
        return {t: float(p[i]) for i, t in enumerate(self.tags)}


def load_taggers(which, device):
    taggers = {}
    if which in ("cnn14", "both"):
        taggers["cnn14"] = Cnn14Tagger(device)
    if which in ("clap", "both"):
        taggers["clap"] = ClapTagger("cpu" if device == "mps" else device)  # msclap mps is patchy
    return taggers


def route_backend(tag, taggers):
    """Pick the backend for a tag: calibrated map if loaded, else any loaded one."""
    want = TAG_BACKEND.get(tag, DEFAULT_BACKEND)
    if want in taggers:
        return want
    return next(iter(taggers))


# --------------------------------------------------------------- ASR + verdict ----
def load_asr(model, device):
    import whisper
    eprint(f"[load] whisper '{model}' ...")
    return whisper.load_model(model, device=("cpu" if device == "mps" else device))


def transcribe(asr, wav):
    try:
        return (asr.transcribe(wav, fp16=False).get("text") or "").strip()
    except Exception as e:  # noqa
        eprint(f"[warn] ASR failed on {os.path.basename(wav)}: {e}")
        return ""


def verdict(p_backend, tag, backend, onom, transcript, tau):
    """Classify one clip using the routed backend's probs."""
    probs = p_backend[backend]
    literal = bool(onom) and onom in transcript
    p_target = probs.get(tag, 0.0) if tag else 0.0
    others = {t: p for t, p in probs.items() if t != tag}
    drift_tag, drift_p = (max(others.items(), key=lambda kv: kv[1]) if others else (None, 0.0))

    if literal:
        return "KO_LITERAL", f"'{onom}' spoken; {backend} P({tag})={p_target:.2f}"
    if tag and p_target >= tau:
        return "WIN_CAND", f"{backend} P({tag})={p_target:.2f} tau={tau}"
    if drift_p >= tau:
        return "DRIFT", f"{backend} other='{drift_tag}' P={drift_p:.2f} (wanted {tag} {p_target:.2f})"
    return "MISS", f"{backend} P({tag})={p_target:.2f} < tau; top='{drift_tag}' {drift_p:.2f}"


def load_items(args):
    if args.manifest:
        base = os.path.dirname(os.path.abspath(args.manifest))
        with open(args.manifest) as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("clips", [])
        for r in rows:
            r.setdefault("tag", args.tag)
            r.setdefault("onom", args.onom)
            if not os.path.isabs(r["file"]):
                r["file"] = os.path.normpath(os.path.join(base, r["file"]))
            yield r
    else:
        for wav in sorted(glob.glob(os.path.join(args.wavs, "**", "*.wav"), recursive=True)):
            yield {"file": wav, "tag": args.tag, "onom": args.onom}


def tau_for(backend, args):
    return args.tau_clap if backend == "clap" else args.tau


def main():
    ap = argparse.ArgumentParser(description="E1.1 para-judge — offline paralinguistic screener")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wavs", help="directory of wavs (recursive)")
    src.add_argument("--manifest", help="JSON list of clips w/ per-clip tag/onom/expected")
    ap.add_argument("--tag", help="default target event tag if not per-clip")
    ap.add_argument("--onom", help="default onomatopoeia for the literal-reading check (e.g. 哈哈哈)")
    ap.add_argument("--tagger", default="cnn14", choices=["cnn14", "clap", "both"])
    ap.add_argument("--tau", type=float, default=0.15, help="cnn14 event-prob threshold (default 0.15)")
    ap.add_argument("--tau-clap", dest="tau_clap", type=float, default=0.20,
                    help="clap softmax threshold (default 0.20, from E1.2 laugh calibration)")
    ap.add_argument("--asr-model", default="tiny", help="whisper model tiny/base/small")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--out", help="write markdown table here (else stdout)")
    ap.add_argument("--calibrate", action="store_true",
                    help="clips have `expected` WIN/KO -> print precision/recall (overall + per event)")
    args = ap.parse_args()

    items = list(load_items(args))
    if not items:
        eprint("no clips found"); sys.exit(1)
    eprint(f"[judge] {len(items)} clips | tagger={args.tagger} tau={args.tau}/{args.tau_clap} "
           f"asr={args.asr_model} device={args.device}")

    taggers = load_taggers(args.tagger, args.device)
    asr = load_asr(args.asr_model, args.device)

    rows = []
    for it in items:
        wav = it["file"]
        if not os.path.exists(wav):
            eprint(f"[skip] missing {wav}"); continue
        p_backend = {name: t.probs(wav) for name, t in taggers.items()}
        tx = transcribe(asr, wav)
        tag = it.get("tag")
        backend = route_backend(tag, taggers) if tag else next(iter(taggers))
        tau = tau_for(backend, args)
        v, detail = verdict(p_backend, tag, backend, it.get("onom"), tx, tau)
        rows.append({**it, "verdict": v, "detail": detail, "transcript": tx, "backend": backend,
                     "p_target": p_backend[backend].get(tag, 0.0)})
        eprint(f"  {v:10s} {os.path.basename(wav):40s} {detail}")

    order = {"WIN_CAND": 0, "DRIFT": 1, "MISS": 2, "KO_LITERAL": 3}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), -r["p_target"]))

    lines = ["| verdict | tag | backend | file | P(tag) | detail | transcript |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        tx = (r["transcript"][:36] + "…") if len(r["transcript"]) > 36 else r["transcript"]
        lines.append(f"| {r['verdict']} | {r.get('tag','')} | {r['backend']} | "
                     f"{os.path.basename(r['file'])} | {r['p_target']:.2f} | {r['detail']} "
                     f"| {tx.replace('|','/')} |")
    md = "\n".join(lines)

    if args.calibrate:
        def score(subset):
            tp = fp = tn = fn = 0
            for r in subset:
                exp = (r.get("expected") or "").upper()
                if exp not in ("WIN", "KO"):
                    continue
                pos_pred, pos_true = (r["verdict"] == "WIN_CAND"), (exp == "WIN")
                tp += pos_pred and pos_true; fp += pos_pred and not pos_true
                fn += (not pos_pred) and pos_true; tn += (not pos_pred) and not pos_true
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            return tp, fp, fn, tn, prec, rec
        md += "\n\n### Calibration (E1.2)\n| set | TP | FP | FN | TN | precision | recall |\n|---|---|---|---|---|---|---|"
        tags = sorted({r.get("tag") for r in rows if r.get("tag")})
        for label, subset in [("ALL", rows)] + [(t, [r for r in rows if r.get("tag") == t]) for t in tags]:
            tp, fp, fn, tn, prec, rec = score(subset)
            md += f"\n| {label} | {tp} | {fp} | {fn} | {tn} | {prec:.2f} | {rec:.2f} |"
        md += ("\n\nScreener earns 'trusted' status per event only if it agrees with the ear here. "
               "Tune --tau/--tau-clap, then re-run. Ear stays final judge for promotions.")

    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")
        eprint(f"[out] wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
