#!/usr/bin/env python3
"""Score the ICL emotion-dilution sweep."""
import sys, os, glob
import numpy as np
import librosa

SR=24000; N_FFT=1024; HOP=256; N_MELS=128
D = sys.argv[1] if len(sys.argv)>1 else "samples/icl_emotion"

def logmel(y):
    m=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS)
    return librosa.power_to_db(m+1e-10)
def load(p):
    y,_=librosa.load(p,sr=SR,mono=True); return y
def corr(a,b):
    n=min(a.size,b.size); ma,mb=logmel(a[:n]),logmel(b[:n])
    k=min(ma.shape[1],mb.shape[1]); va,vb=ma[:,:k].ravel(),mb[:,:k].ravel()
    return float(np.corrcoef(va,vb)[0,1]) if va.std()>0 and vb.std()>0 else 0.0
def rms_db(y):
    r=np.sqrt(np.mean(y**2))+1e-9; return 20*np.log10(r)

caps=["full","200","120","80","50","30"]
cache={}
def g(tag,cap):
    p=f"{D}/{tag}_{cap}.wav"
    if p not in cache: cache[p]=load(p) if os.path.exists(p) else None
    return cache[p]

neu_full=g("neu","full")
print(f"{'cap':>5} | {'ang_mov':>7} {'sad_mov':>7} | {'identity':>8} | {'neu_dB':>6} {'ang_dB':>6} {'sad_dB':>6}")
print("-"*64)
for cap in caps:
    neu,ang,sad=g("neu",cap),g("ang",cap),g("sad",cap)
    if neu is None: continue
    am=corr(ang,neu) if ang is not None else float('nan')
    sm=corr(sad,neu) if sad is not None else float('nan')
    ident=corr(neu,neu_full) if neu_full is not None else float('nan')
    print(f"{cap:>5} | {am:7.3f} {sm:7.3f} | {ident:8.3f} | "
          f"{rms_db(neu):6.1f} {rms_db(ang) if ang is not None else 0:6.1f} {rms_db(sad) if sad is not None else 0:6.1f}")
print("\nmovement LOWER=emotes more; identity HIGHER=voice preserved; watch dB collapse")
