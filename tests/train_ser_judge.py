#!/usr/bin/env python3
"""Train a CATEGORICAL 7-class Italian Speech-Emotion-Recognition JUDGE on Emozionalmente."""
import argparse, csv, json, os, sys, collections

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
ALIAS = {"joy": "joy", "happiness": "joy", "neutrality": "neutral", "neutral": "neutral"}

def norm(e):
    e = str(e).strip().lower()
    return ALIAS.get(e, e)

def _find_root(d):
    for c in (d, os.path.join(d, "emozionalmente")):
        if os.path.exists(os.path.join(c, "metadata", "samples.csv")):
            return c
    sys.exit(f"[ser] metadata/samples.csv not found under {d}")

def _read_split(root, which):
    """Return [(abs_wav_path, label_id)] for split in {train,dev,test}."""
    meta = os.path.join(root, "metadata")
    audio = os.path.join(root, "audio")
    path = os.path.join(meta, "split", f"{which}.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = norm(r["emotion_expressed"])
            if lab not in LAB2ID:
                continue
            wp = os.path.join(audio, r["file_name"])
            if os.path.exists(wp):
                rows.append((wp, LAB2ID[lab]))
    return rows

def train(args):
    import numpy as np
    import torch
    import soundfile as sf
    from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                              TrainingArguments, Trainer)

    root = _find_root(args.data_dir)
    tr = _read_split(root, "train"); dv = _read_split(root, "dev"); te = _read_split(root, "test")
    print(f"[ser] train {len(tr)}  dev {len(dv)}  test {len(te)}  base={args.base_model}", flush=True)
    print(f"[ser] train label dist: {collections.Counter(l for _,l in tr)}", flush=True)

    fe = AutoFeatureExtractor.from_pretrained(args.base_model)
    sr = fe.sampling_rate

    def load_wav(path):
        import librosa
        a, s = sf.read(path)
        if a.ndim > 1:
            a = a.mean(1)
        if s != sr:
            a = librosa.resample(a.astype("float32"), orig_sr=s, target_sr=sr)
        return a.astype("float32")

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            wp, lab = self.rows[i]
            return {"wav": load_wav(wp), "label": lab}

    def collate(batch):
        wavs = [b["wav"] for b in batch]
        feats = fe(wavs, sampling_rate=sr, return_tensors="pt", padding=True)
        feats["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return feats

    model = AutoModelForAudioClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS),
        label2id=LAB2ID, id2label={i: l for l, i in LAB2ID.items()})
    if args.freeze_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    def metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        recalls = []
        for c in range(len(LABELS)):
            m = labels == c
            if m.sum():
                recalls.append((preds[m] == c).mean())
        return {"acc": float((preds == labels).mean()), "uar": float(np.mean(recalls))}

    ta = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="uar", greater_is_better=True,
        logging_steps=20, warmup_ratio=0.1, fp16=torch.cuda.is_available(), report_to=[],
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=ta, train_dataset=DS(tr), eval_dataset=DS(dv),
                      data_collator=collate, compute_metrics=metrics)
    trainer.train()

    pred = trainer.predict(DS(te))
    np = __import__("numpy")
    preds = np.argmax(pred.predictions, axis=-1); labs = pred.label_ids
    conf = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for p, l in zip(preds, labs):
        conf[l][p] += 1
    report = {"labels": LABELS, "test_acc": float((preds == labs).mean()),
              "test_uar": float(np.mean([(preds[labs == c] == c).mean() for c in range(len(LABELS)) if (labs == c).sum()])),
              "confusion_rows_true_cols_pred": conf.tolist()}
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out); fe.save_pretrained(args.out)
    json.dump(report, open(os.path.join(args.out, "test_report.json"), "w"), indent=2)
    print(f"[ser] TEST acc {report['test_acc']:.3f}  UAR {report['test_uar']:.3f}  -> {args.out}", flush=True)
    print("[ser] confusion (rows=true, cols=pred):", LABELS, flush=True)
    for i, row in enumerate(conf.tolist()):
        print(f"  {LABELS[i]:9s} {row}", flush=True)

def _self_test():
    """No model/data: verify the label map + split-vocab normalization are consistent."""
    assert len(LABELS) == 7 and LAB2ID["anger"] == 0
    assert norm("joy") == "joy" and norm("happiness") == "joy"
    assert norm("neutrality") == "neutral" and norm("NEUTRAL") == "neutral"
    for l in LABELS:
        assert norm(l) in LAB2ID, f"label {l} not round-tripping"
    print("label set:", LABELS)
    print("SELF-TEST PASS — 7-class judge label map + Emozionalmente vocab normalization consistent.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-dir", help="Zenodo Emozionalmente dir (has metadata/split/*.csv + audio/)")
    ap.add_argument("--out", default="/root/qwen-ft/ser_judge_it")
    ap.add_argument("--base-model", default="jonatasgrosman/wav2vec2-large-xlsr-53-italian",
                    help="wav2vec2 base to fine-tune (Italian XLSR by default; any wav2vec2 works)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder", action="store_true", help="train only the head (faster, lower ceiling)")
    a = ap.parse_args()
    if a.self_test:
        _self_test(); return
    if not a.data_dir:
        ap.error("--data-dir required (or --self-test)")
    train(a)

if __name__ == "__main__":
    main()
