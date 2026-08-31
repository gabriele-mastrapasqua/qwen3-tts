#!/usr/bin/env python3
"""Build a multi-layer Talker steer (.qlsteer) from two QWEN_ACT_MAP captures."""
import struct, sys, argparse, math

QAMP = 0x504D4151
QLST = 0x54534C51

def read_qamp(path):
    with open(path, "rb") as f:
        magic, L, D = struct.unpack("<Iii", f.read(12))
        if magic != QAMP: sys.exit(f"{path}: bad magic")
        flat = list(struct.unpack(f"<{L*D}f", f.read(L*D*4)))
    return L, D, flat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neutral"); ap.add_argument("emotion"); ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--unit-per-layer", action="store_true",
                    help="L2-normalize each layer's delta (so --ml-weight is uniform across layers)")
    ap.add_argument("--mean-center", action="store_true",
                    help="subtract the per-layer mean (DC offset) from the delta before normalizing")
    ap.add_argument("--project-out-neutral", action="store_true",
                    help="remove the component of the delta along the NEUTRAL activation direction "
                         "(the global energy/identity axis) → emotion-only direction, less timbre/energy bleed")
    ap.add_argument("--clean", action="store_true",
                    help="convenience: --mean-center + --project-out-neutral (idea-3 cleaned direction)")
    a = ap.parse_args()
    if a.clean: a.mean_center = True; a.project_out_neutral = True
    Ln, Dn, neu = read_qamp(a.neutral)
    Le, De, emo = read_qamp(a.emotion)
    if (Ln, Dn) != (Le, De): sys.exit(f"shape mismatch {Ln}x{Dn} vs {Le}x{De}")
    out = [0.0] * (Ln * Dn)
    print(f"per-layer delta L2 (emotion - neutral), {Ln} layers x {Dn}  "
          f"[mean_center={a.mean_center} project_out_neutral={a.project_out_neutral}]:")
    for l in range(Ln):
        d = [emo[l*Dn+i] - neu[l*Dn+i] for i in range(Dn)]
        nrm = math.sqrt(sum(x*x for x in d))
        if a.mean_center:
            m = sum(d) / Dn
            d = [x - m for x in d]
        if a.project_out_neutral:
            nl = [neu[l*Dn+i] for i in range(Dn)]
            nn = math.sqrt(sum(x*x for x in nl))
            if nn > 0:
                nhat = [x / nn for x in nl]
                dot = sum(d[i]*nhat[i] for i in range(Dn))
                d = [d[i] - dot*nhat[i] for i in range(Dn)]
        nrm2 = math.sqrt(sum(x*x for x in d))
        if a.unit_per_layer and nrm2 > 0:
            d = [x / nrm2 for x in d]
        d = [x * a.scale for x in d]
        for i in range(Dn): out[l*Dn+i] = d[i]
        tag = "final" if l == Ln-1 else f"L{l:02d}"
        print(f"  {tag}: ||delta||={nrm:7.2f}")
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Iii", QLST, Ln, Dn))
        f.write(struct.pack(f"<{Ln*Dn}f", *out))
    print(f"wrote {a.out}: {Ln} x {Dn}, scale={a.scale}, unit_per_layer={a.unit_per_layer}")

if __name__ == "__main__":
    main()
