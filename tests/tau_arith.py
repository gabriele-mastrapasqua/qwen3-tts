#!/usr/bin/env python3
"""Emotion TASK VECTOR via FLOAT weight arithmetic (the pro method, arXiv 2507.03382):"""
import argparse, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expr_extract import parse_header, read_bf16, bf16_to_f32

def f32_to_bf16(f):
    """round-to-nearest-even f32 -> bf16 (uint16 of the top 16 bits)."""
    u = f.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    return ((u + bias) >> 16).astype(np.uint16)

def is_f32_tensor(name):
    return any(s in name for s in ("_norm.weight", "layernorm.weight", ".bias")) or name.endswith(".norm.weight")

def main():
    import lz4.block
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir"); ap.add_argument("emotion_dir"); ap.add_argument("neutral_dir")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=1.0, help="bake this scale into tau (default 1.0; tune live via --expr-weight)")
    ap.add_argument("--lang", default="Italian")
    ap.add_argument("--hidden", type=int, default=2048)
    a = ap.parse_args()

    bp = os.path.join(a.base_dir, "model.safetensors")
    ep = os.path.join(a.emotion_dir, "model.safetensors")
    npth = os.path.join(a.neutral_dir, "model.safetensors")
    bh, bo = parse_header(bp); eh, eo = parse_header(ep); nh, no = parse_header(npth)

    keys = sorted(k for k in eh if k != "__metadata__")
    changed, n_f32, n_overflow = [], 0, 0
    stream = bytearray()
    for k in keys:
        if k not in bh or k not in nh:
            continue
        base = read_bf16(bp, bh, bo, k)
        emo = read_bf16(ep, eh, eo, k)
        neu = read_bf16(npth, nh, no, k)
        if base.shape != emo.shape or base.shape != neu.shape:
            print(f"[shape!] {k} skipped"); continue
        new_f = bf16_to_f32(base) + a.alpha * (bf16_to_f32(emo) - bf16_to_f32(neu))
        if not np.any(bf16_to_f32(emo) != bf16_to_f32(neu)):
            continue
        name_b = k.encode()
        if is_f32_tensor(k):
            payload = new_f.astype("<f4").tobytes()
            stream += struct.pack("<H", len(name_b)) + name_b
            stream += struct.pack("<I", len(payload)) + struct.pack("<B", 0)
            stream += struct.pack("<I", len(payload)) + payload
            changed.append((k, len(payload), len(payload))); n_f32 += 1
        else:
            new_u16 = f32_to_bf16(new_f)
            d32 = new_u16.astype(np.int32) - base.astype(np.int32)
            if np.any(d32 > 32767) or np.any(d32 < -32768):
                n_overflow += int(np.count_nonzero((d32 > 32767) | (d32 < -32768)))
            delta = d32.astype(np.int16)
            comp = lz4.block.compress(delta.tobytes(), mode="default", store_size=False)
            stream += struct.pack("<H", len(name_b)) + name_b
            stream += struct.pack("<I", base.nbytes) + struct.pack("<B", 4)
            stream += struct.pack("<I", len(comp)) + comp
            changed.append((k, base.nbytes, len(comp)))

    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"QEXP"); f.write(struct.pack("<I", 1))
        lb = a.lang.encode()[:15]; f.write(lb + b"\x00" * (16 - len(lb)))
        f.write(struct.pack("<I", 0)); f.write(b"WDLT")
        f.write(struct.pack("<I", a.hidden)); f.write(struct.pack("<I", len(changed)))
        f.write(stream)
    disk = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}  (alpha={a.alpha})")
    print(f"  tensors : {len(changed)} ({len(changed)-n_f32} bf16-delta + {n_f32} f32)  on disk {disk:.1f} MB")
    print(f"  overflow: {n_overflow} (should be ~0 now — float arithmetic keeps new near base)")

if __name__ == "__main__":
    main()
