#!/usr/bin/env python3
"""Extract the pure x-vector (speaker embedding) from a .qvoice into a tiny legacy .bin."""
import argparse
import os
import struct
import sys

def extract(path):
    """Return (embedding_bytes, enc_dim, version) for a QVCE .qvoice file."""
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"QVCE":
            raise ValueError(f"{path} is not a .qvoice (QVCE) file")
        version = struct.unpack_from("<I", head, 4)[0]
        if version < 2:
            raise ValueError(f"{path} is QVCE v{version}; v>=2 (enc_dim header) required")
        enc_dim = struct.unpack_from("<I", head, 8)[0]
        emb = f.read(enc_dim * 4)
        if len(emb) != enc_dim * 4:
            raise ValueError(f"{path}: truncated embedding ({len(emb)} of {enc_dim*4} bytes)")
    return emb, enc_dim, version

def main():
    ap = argparse.ArgumentParser(description="Extract x-vector .bin from a .qvoice")
    ap.add_argument("qvoice", nargs="?", help="input .qvoice file")
    ap.add_argument("-o", "--out", help="output .bin (default: same name, .bin)")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not args.qvoice:
        ap.error("qvoice input required (or --self-test)")
    out = args.out or os.path.splitext(args.qvoice)[0] + ".bin"
    emb, enc_dim, version = extract(args.qvoice)
    with open(out, "wb") as f:
        f.write(emb)
    norm = sum(struct.unpack(f"<{enc_dim}f", emb)[i] ** 2 for i in range(enc_dim)) ** 0.5
    print(f"{args.qvoice} (QVCE v{version}, enc_dim={enc_dim})")
    print(f"  -> {out}  ({len(emb)} bytes, embedding L2 norm={norm:.4f})")
    print(f"  use: ./qwen_tts -d <model> --load-voice {out} --xvector-only -l <Lang> ...")

def _self_test():
    """Round-trip a synthetic QVCE header through extract() with no model/files needed."""
    import tempfile

    enc_dim = 8
    vals = [0.1 * i for i in range(enc_dim)]
    blob = b"QVCE" + struct.pack("<I", 3) + struct.pack("<I", enc_dim) + struct.pack(f"<{enc_dim}f", *vals)
    blob += b"\x00\x00\x00\x00" + b"junk after embedding"
    with tempfile.NamedTemporaryFile(suffix=".qvoice", delete=False) as tf:
        tf.write(blob)
        p = tf.name
    emb, ed, ver = extract(p)
    os.unlink(p)
    got = list(struct.unpack(f"<{ed}f", emb))
    assert ed == enc_dim and ver == 3, (ed, ver)
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, vals)), got
    assert len(emb) == enc_dim * 4
    print("SELF-TEST PASS — extracts exactly enc_dim floats, ignores trailing ref_text/ref_codes/WDELTA.")

if __name__ == "__main__":
    sys.exit(main())
