#!/usr/bin/env python3
"""Automatic emotion scoring via a pre-trained Speech-Emotion-Recognition model — replaces listening"""
import argparse, glob, os, sys
import numpy as np

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SR = 16000
DIMS = ("arousal", "dominance", "valence")

def load_model(device):
    import torch, torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = self.dropout(x); x = torch.tanh(self.dense(x)); x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            h = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.classifier(h)

    proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = EmotionModel.from_pretrained(MODEL_ID).to(device).eval()
    return proc, model

def read_16k(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        except Exception:
            import torch, torchaudio
            y = torchaudio.functional.resample(torch.from_numpy(y), sr, SR).numpy()
    return y

def score(proc, model, device, path):
    import torch
    y = read_16k(path)
    x = proc(y, sampling_rate=SR, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy()[0]
    return dict(zip(DIMS, out.tolist()))

def fmt(d):
    return "  ".join(f"{k[:3]}={d[k]:.3f}" for k in DIMS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral", help="neutral baseline wav (shifts are vs this)")
    ap.add_argument("--ab-dir", help="dir with neu_<cond>.wav and <emotion>_<cond>.wav")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()

    import torch
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0"
    print(f"[emo_score] device={device} model={MODEL_ID}", file=sys.stderr)
    proc, model = load_model(device)

    if a.ab_dir:
        conds, emos = [], []
        files = {}
        for f in sorted(glob.glob(os.path.join(a.ab_dir, "*.wav"))):
            b = os.path.basename(f)[:-4]
            parts = b.split("_")
            if len(parts) < 2:
                continue
            state, cond = parts[0], parts[1]
            reptag = "_".join(parts[2:])
            files[(state, cond, reptag)] = f
            if cond not in conds:
                conds.append(cond)
            if state != "neu" and state not in emos:
                emos.append(state)
        reps = sorted({k[2] for k in files})
        sc = {k: score(proc, model, device, v) for k, v in files.items()}
        nrep = len(reps)
        print(f"\nA/B AROUSAL+VALENCE SHIFT vs neutral (mean over {nrep} rep(s))  dir={a.ab_dir}")
        print(f"{'emotion':10s} | " + " | ".join(f"{c:^24s}" for c in conds))
        print("-" * (12 + 27 * len(conds)))
        for e in emos:
            cells = []
            for c in conds:
                dA, dV, n = [], [], 0
                for r in reps:
                    ek, nk = (e, c, r), ("neu", c, r)
                    if ek in sc and nk in sc:
                        dA.append(sc[ek]["arousal"] - sc[nk]["arousal"])
                        dV.append(sc[ek]["valence"] - sc[nk]["valence"]); n += 1
                if not n:
                    cells.append(f"{'--':^24s}"); continue
                cells.append(f"dAro{np.mean(dA):+.3f} dVal{np.mean(dV):+.3f}")
            print(f"{e:10s} | " + " | ".join(f"{x:^24s}" for x in cells))
        print("\n(anger/joy/fear -> arousal UP ; sad -> arousal+valence DOWN ; joy -> valence UP."
              " Bigger |shift| in the right direction = emotion lands harder.)")
        return

    if a.neutral:
        nb = score(proc, model, device, a.neutral)
        print(f"neutral  {fmt(nb)}  ({a.neutral})")
        for f in a.files:
            s = score(proc, model, device, f)
            ds = {k: s[k] - nb[k] for k in DIMS}
            print(f"{os.path.basename(f):16s} {fmt(s)}   shift {fmt(ds)}")
    else:
        for f in a.files:
            print(f"{os.path.basename(f):16s} {fmt(score(proc, model, device, f))}")

if __name__ == "__main__":
    main()
