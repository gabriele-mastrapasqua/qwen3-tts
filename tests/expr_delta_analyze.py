#!/usr/bin/env python3
"""Analyze the expressivity fine-tune delta = (expr - base)."""
import argparse, os, sys
from collections import defaultdict
import torch
from safetensors import safe_open

def st_path(d):
    p = os.path.join(d, "model.safetensors")
    if not os.path.exists(p):
        sys.exit(f"no model.safetensors in {d}")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("expr_dir")
    ap.add_argument("--thresh", type=float, default=1e-4,
                    help="relative-L2 above which a tensor counts as 'changed'")
    args = ap.parse_args()

    bp, ep = st_path(args.base_dir), st_path(args.expr_dir)
    fb = safe_open(bp, framework="pt", device="cpu")
    fe = safe_open(ep, framework="pt", device="cpu")
    kb, ke = set(fb.keys()), set(fe.keys())
    if kb != ke:
        print(f"[warn] key sets differ: only-base={len(kb-ke)} only-expr={len(ke-kb)}")

    keys = sorted(kb & ke)
    changed = []
    fam_changed = defaultdict(lambda: [0, 0])
    fam_total = defaultdict(int)

    for k in keys:
        tb = fb.get_tensor(k).to(torch.float32)
        te = fe.get_tensor(k).to(torch.float32)
        fam = family(k)
        fam_total[fam] += 1
        if tb.shape != te.shape:
            print(f"[shape!] {k} {tuple(tb.shape)} vs {tuple(te.shape)}")
            continue
        bn = tb.norm().item()
        d = (te - tb)
        rel = (d.norm().item() / bn) if bn > 0 else (d.norm().item())
        if rel > args.thresh:
            numel = tb.numel()
            changed.append((k, rel, numel, numel * 2))
            fam_changed[fam][0] += 1
            fam_changed[fam][1] += numel
        del tb, te, d

    changed.sort(key=lambda x: -x[1])
    tot_params = sum(c[2] for c in changed)
    tot_bf16 = sum(c[3] for c in changed)

    print("\n=== CHANGED TENSORS (top 25 by relative-L2) ===")
    for k, rel, numel, _ in changed[:25]:
        print(f"  {rel:8.4f}  {numel/1e6:7.2f}M  {k}")
    print(f"  ... {max(0,len(changed)-25)} more changed tensors")

    print("\n=== BY FAMILY (changed / total tensors, changed params) ===")
    for fam in sorted(fam_total):
        nc, pc = fam_changed[fam]
        print(f"  {fam:40s}  {nc:3d}/{fam_total[fam]:<3d}  {pc/1e6:8.2f}M params")

    print("\n=== MICRO-FILE SIZE (changed tensors only) ===")
    print(f"  changed tensors : {len(changed)} / {len(keys)}")
    print(f"  changed params  : {tot_params/1e6:.1f}M")
    print(f"  bf16  : {tot_bf16/1e6:8.1f} MB")
    print(f"  int8  : {tot_params/1e6:8.1f} MB  (1 B/param + tiny scales)")
    print(f"  int4  : {tot_params/2/1e6:8.1f} MB  (0.5 B/param + scales)")
    li = sorted({layer_idx(k) for k, *_ in changed if layer_idx(k) is not None})
    print(f"\n  talker layer indices that changed: {li}")

def family(k):
    if ".layers." in k:
        parts = k.split(".layers.")[1].split(".", 1)
        sub = parts[1] if len(parts) > 1 else ""
        sub_class = ".".join(sub.split(".")[:2])
        return f"layers.*.{sub_class}"
    return k

def layer_idx(k):
    if ".layers." in k:
        try:
            return int(k.split(".layers.")[1].split(".")[0])
        except ValueError:
            return None
    return None

if __name__ == "__main__":
    main()
