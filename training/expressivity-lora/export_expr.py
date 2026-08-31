#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter into a .expr micro-file the C engine can load."""
import argparse, json, os, struct, sys
import numpy as np

def parse_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen

def read_tensor(path, hdr, data_off, name, np_dtype):
    e = hdr[name]
    s, en = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_off + s)
        raw = f.read(en - s)
    return np.frombuffer(raw, dtype=np_dtype).reshape(e["shape"])

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f32):
    return (f32.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("out")
    ap.add_argument("--lang", default="Italian")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--merge", action="store_true", help="bake dense delta (dtype 4) for A/B validation")
    ap.add_argument("--base", default="qwen3-tts-1.7b", help="base CV dir (needed for --merge)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.adapter_dir, "adapter_config.json")))
    r_cfg, alpha = cfg["r"], cfg["lora_alpha"]
    scale_default = alpha / r_cfg
    ap_path = os.path.join(args.adapter_dir, "adapter_model.safetensors")
    ah, ao = parse_header(ap_path)

    PREFIX = "base_model.model."
    mods = {}
    for k in ah:
        if k == "__metadata__":
            continue
        if ".lora_A.weight" in k:
            wn = k[len(PREFIX):].replace(".lora_A.weight", ".weight")
            mods.setdefault(wn, {})["A"] = read_tensor(ap_path, ah, ao, k, np.float32)
        elif ".lora_B.weight" in k:
            wn = k[len(PREFIX):].replace(".lora_B.weight", ".weight")
            mods.setdefault(wn, {})["B"] = read_tensor(ap_path, ah, ao, k, np.float32)

    names = sorted(mods)
    if args.merge:
        bh, bo = parse_header(os.path.join(args.base, "model.safetensors"))
        import lz4.block

    stream = bytearray()
    tot_payload = 0
    for wn in names:
        A = mods[wn]["A"]
        B = mods[wn]["B"]
        r, n_in = A.shape
        n_out = B.shape[0]
        name_b = wn.encode()
        if not args.merge:
            payload = struct.pack("<IIIf", r, n_in, n_out, scale_default)
            payload += A.astype("<f4").tobytes() + B.astype("<f4").tobytes()
            stream += struct.pack("<H", len(name_b)) + name_b
            stream += struct.pack("<I", len(payload))
            stream += struct.pack("<B", 5)
            stream += struct.pack("<I", len(payload)) + payload
            tot_payload += len(payload)
        else:
            delta = (scale_default * (B @ A)).astype(np.float32)
            cv = read_tensor(os.path.join(args.base, "model.safetensors"), bh, bo, wn, np.uint16).ravel()
            new = f32_to_bf16(bf16_to_f32(cv) + delta.ravel())
            d16 = (new.astype(np.int32) - cv.astype(np.int32)).astype(np.int16)
            comp = lz4.block.compress(d16.tobytes(), mode="default", store_size=False)
            stream += struct.pack("<H", len(name_b)) + name_b
            stream += struct.pack("<I", cv.nbytes)
            stream += struct.pack("<B", 4)
            stream += struct.pack("<I", len(comp)) + comp
            tot_payload += len(comp)

    n_extra = 0
    emb_key = next((k for k in ah if k.endswith("text_embedding.weight")), None)
    if emb_key:
        import lz4.block
        bh2, bo2 = parse_header(os.path.join(args.base, "model.safetensors"))
        trained = read_tensor(ap_path, ah, ao, emb_key, np.uint16).ravel()
        base_emb = read_tensor(os.path.join(args.base, "model.safetensors"), bh2, bo2,
                               "talker.model.text_embedding.weight", np.uint16).ravel()
        if trained.shape != base_emb.shape:
            print(f"  !! embedding shape mismatch {trained.shape} vs {base_emb.shape}, skipping")
        else:
            d16 = (trained.astype(np.int32) - base_emb.astype(np.int32)).astype(np.int16)
            comp = lz4.block.compress(d16.tobytes(), mode="default", store_size=False)
            nb = b"talker.model.text_embedding.weight"
            stream += struct.pack("<H", len(nb)) + nb
            stream += struct.pack("<I", base_emb.nbytes)
            stream += struct.pack("<B", 4) + struct.pack("<I", len(comp)) + comp
            n_extra = 1
            changed = int((d16 != 0).sum())
            d2 = d16.reshape(base_emb.shape[0] // 2048 if False else -1, 2048) if base_emb.size % 2048 == 0 else None
            try:
                d2 = d16.reshape(-1, 2048)
                tag_changed = int((d2[151646:151662] != 0).any(axis=1).sum())
            except Exception:
                tag_changed = -1
            print(f"  + text_embedding delta: {changed} changed elems, {len(comp)/1e6:.2f} MB comp "
                  f"| TAG rows 151646-151661 changed: {tag_changed}/16")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"QEXP")
        f.write(struct.pack("<I", 1))
        lb = args.lang.encode()[:15]
        f.write(lb + b"\x00" * (16 - len(lb)))
        f.write(struct.pack("<I", 0))
        f.write(b"WDLT")
        f.write(struct.pack("<I", args.hidden))
        f.write(struct.pack("<I", len(names) + n_extra))
        f.write(stream)

    disk = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({'merged dense' if args.merge else 'factored LoRA'})")
    print(f"  modules : {len(names)}  (r={r_cfg} alpha={alpha} scale={scale_default:.3f})")
    print(f"  on disk : {disk:.1f} MB")

if __name__ == "__main__":
    main()
