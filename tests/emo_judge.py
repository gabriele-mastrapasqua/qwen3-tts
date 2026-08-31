#!/usr/bin/env python3
"""Score generated TTS clips with the categorical SER JUDGE -> a RECOGNIZABILITY table (intended vs"""
import argparse, glob, os, sys, collections

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def _intended_from_name(path):
    """Pick the intended emotion from a filename (stem, or any LABEL token in it)."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem in LABELS:
        return stem
    toks = stem.replace("-", "_").split("_")
    for t in toks:
        if t in LABELS:
            return t
    alias = {"happy": "joy", "happiness": "joy", "neutrality": "neutral", "sad": "sadness", "angry": "anger"}
    for t in toks + [stem]:
        if t in alias:
            return alias[t]
    return None

def _collect(args):
    items = []
    if args.wav_dir:
        for p in sorted(glob.glob(os.path.join(args.wav_dir, "*.wav"))):
            emo = _intended_from_name(p)
            if emo:
                items.append((emo, p))
            else:
                print(f"[judge] WARN: cannot infer intended emotion from {os.path.basename(p)} (skipped)", file=sys.stderr)
    for spec in args.wav or []:
        emo, _, path = spec.partition("=")
        items.append((emo.strip().lower(), path))
    return items

def _print_table(results, tag):
    """results: list of (intended, predicted, confidence)."""
    hits = sum(1 for i, p, _ in results if i == p)
    n = len(results)
    per = collections.defaultdict(lambda: [0, 0])
    conf = collections.defaultdict(collections.Counter)
    for i, p, _ in results:
        per[i][1] += 1; per[i][0] += int(i == p); conf[i][p] += 1
    recalls = [h / t for h, t in (per[e] for e in per) if t]
    uar = sum(recalls) / len(recalls) if recalls else 0.0
    head = f" SER-JUDGE recognizability" + (f"  [{tag}]" if tag else "")
    print("=" * len(head)); print(head); print("=" * len(head))
    print(f"{'intended':10s} -> {'recognized':10s}  {'conf':>5}  hit")
    for i, p, c in results:
        print(f"{i:10s} -> {p:10s}  {c:5.2f}  {'OK' if i==p else 'x '}")
    print("-" * 34)
    print(f"accuracy {hits}/{n} = {hits/max(n,1):.2f}   UAR = {uar:.2f}")
    return {"tag": tag, "n": n, "acc": hits / max(n, 1), "uar": uar}

def judge(args):
    import numpy as np
    import torch
    import soundfile as sf
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    items = _collect(args)
    if not items:
        sys.exit("[judge] no clips found (use --wav-dir or --wav emotion=path)")
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(args.model).eval()
    id2lab = model.config.id2label
    sr = fe.sampling_rate

    def load(path):
        a, s = sf.read(path)
        if a.ndim > 1:
            a = a.mean(1)
        if s != sr:
            import librosa
            a = librosa.resample(a.astype("float32"), orig_sr=s, target_sr=sr)
        return a.astype("float32")

    results = []
    with torch.no_grad():
        for intended, path in items:
            x = fe(load(path), sampling_rate=sr, return_tensors="pt", padding=True)
            logits = model(**x).logits[0]
            prob = torch.softmax(logits, -1)
            k = int(prob.argmax())
            results.append((intended, str(id2lab[k]).lower(), float(prob[k])))
    _print_table(results, args.tag)

def _self_test():
    assert _intended_from_name("anger.wav") == "anger"
    assert _intended_from_name("ryan_joy_csp.wav") == "joy"
    assert _intended_from_name("happy.wav") == "joy"
    assert _intended_from_name("clip_0001.wav") is None
    fake = [("anger", "anger", .9), ("joy", "joy", .8), ("fear", "anger", .5),
            ("sadness", "sadness", .7), ("disgust", "disgust", .6),
            ("surprise", "surprise", .8), ("neutral", "sadness", .4)]
    s = _print_table(fake, "selftest")
    assert abs(s["acc"] - 5/7) < 1e-6, "accuracy aggregation wrong"
    assert 0 < s["uar"] <= 1.0
    print("SELF-TEST PASS — filename->intended parsing + recognizability/UAR aggregation correct.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", help="dir of the trained categorical SER judge (train_ser_judge.py output)")
    ap.add_argument("--wav-dir", help="dir of *.wav whose stem is the intended emotion")
    ap.add_argument("--wav", action="append", help="explicit intended=path (repeatable)")
    ap.add_argument("--tag", default="", help="label for this variant in the printout")
    a = ap.parse_args()
    if a.self_test:
        _self_test(); return
    if not a.model:
        ap.error("--model required (the trained judge dir) — or use --self-test")
    judge(a)

if __name__ == "__main__":
    main()
