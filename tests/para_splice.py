#!/usr/bin/env python3
"""Splice a REAL paralinguistic clip into synthesized speech at a [tag] point."""
import sys, wave, struct, argparse
import numpy as np

def rd(p):
    w=wave.open(p,'rb'); n=w.getnframes(); sr=w.getframerate()
    a=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768.0
    if w.getnchannels()>1: a=a.reshape(-1,w.getnchannels()).mean(1)
    w.close(); return a,sr

def wr(p,a,sr):
    a=np.clip(a,-1,1); x=(a*32767).astype(np.int16)
    w=wave.open(p,'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes()); w.close()

def rms(a):
    return float(np.sqrt(np.mean(a**2))+1e-9)

def xfade(a,b,n):
    if n<=0 or len(a)==0 or len(b)==0: return np.concatenate([a,b])
    n=min(n,len(a),len(b)); t=np.linspace(0,1,n)
    fo=np.cos(t*np.pi/2); fi=np.sin(t*np.pi/2)
    mid=a[-n:]*fo+b[:n]*fi
    return np.concatenate([a[:-n],mid,b[n:]])

ap=argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("pre"); ap.add_argument("clip"); ap.add_argument("post")
ap.add_argument("--xfade",type=float,default=0.04); ap.add_argument("--gain",type=float,default=0.9)
ap.add_argument("--gap",type=float,default=0.05)
a=ap.parse_args()
pre,sr=rd(a.pre); clip,_=rd(a.clip); post,_=rd(a.post)
clip=clip*(rms(pre)/rms(clip))*a.gain
g=int(a.gap*sr); pad=np.zeros(g,dtype=np.float32)
nx=int(a.xfade*sr)
out=xfade(pre, np.concatenate([pad,clip,pad]), nx)
out=xfade(out, post, nx)
wr(a.out,out,sr)
print(f"  spliced -> {a.out}  ({len(out)/sr:.2f}s)")
