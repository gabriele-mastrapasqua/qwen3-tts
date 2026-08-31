#!/usr/bin/env python3
"""Offline quantize/dequantize of the Code Predictor weights."""
import argparse, json, os, re, shutil, struct, subprocess, sys, time
import numpy as np

CP = "talker.code_predictor."

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f32):
    """round-to-nearest-even, like the hardware."""
    bits = f32.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)

def fp16_rt(x):
    """fp16 storage roundtrip (scales are stored as fp16 in these formats)."""
    return x.astype(np.float16).astype(np.float32)

def round_c(x):
    """C roundf(): round half away from zero (np.rint is half-to-even)."""
    return np.copysign(np.floor(np.abs(x) + 0.5), x)

def safediv(a, b):
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=(b != 0))
    return out

def get_weights(v, mode):
    if mode == "x2":
        return v * v
    if mode == "abs":
        return np.abs(v)
    return np.ones_like(v)

def quant_q4_0(v, wmode):
    b = v.reshape(-1, 32)
    s = fp16_rt(np.abs(b).max(axis=1, keepdims=True) / 7.0)
    q = np.clip(round_c(safediv(b, s)), -8, 7)
    return (s * q).reshape(v.shape), 4.5

def quant_q3_0(v, wmode):
    b = v.reshape(-1, 32)
    s = fp16_rt(np.abs(b).max(axis=1, keepdims=True) / 3.0)
    q = np.clip(round_c(safediv(b, s)), -4, 3)
    return (s * q).reshape(v.shape), 3.5

def quant_q2_0(v, wmode):
    b = v.reshape(-1, 32)
    s = np.abs(b).max(axis=1, keepdims=True) / 1.5
    code = np.clip(np.rint(safediv(b, s) + 1.5), 0, 3)
    return ((code - 1.5) * s).reshape(v.shape), 2.5

def qx_search(vb, wb, nmax):
    """make_qx_quants-style symmetric weighted scale search on sub-blocks.
    vb, wb: [N, sub] f32. Returns (best q pattern [N,sub], best float scale [N])."""
    n = vb.shape[0]
    idx = np.argmax(np.abs(vb), axis=1)
    vmax = vb[np.arange(n), idx]
    best_err = np.full(n, np.inf, dtype=np.float64)
    best_s = np.zeros(n, dtype=np.float32)
    best_q = np.zeros_like(vb)
    nz = vmax != 0
    for k in range(-9, 10):
        isc = np.zeros(n, dtype=np.float32)
        isc[nz] = -(nmax + 0.1 * k) / vmax[nz]
        q = np.clip(np.rint(vb * isc[:, None]), -nmax, nmax - 1)
        num = (wb * vb * q).sum(axis=1)
        den = (wb * q * q).sum(axis=1)
        s = safediv(num, den.astype(num.dtype))
        err = (wb * (vb - s[:, None] * q) ** 2).sum(axis=1)
        better = err < best_err
        best_err[better] = err[better]
        best_s[better] = s[better]
        best_q[better] = q[better]
    return best_q, best_s

def quant_q4_0s(v, wmode):
    """Q4_0 layout (block 32, fp16 scale, q∈[-8,7]) but with the weighted scale
    SEARCH instead of naive absmax RTN. Same runtime format as the C kernels →
    if this wins, it ships as a loader-side change with ZERO new kernels."""
    b = v.reshape(-1, 32)
    w = get_weights(b, wmode)
    q, s = qx_search(b, w, nmax=8)
    s = fp16_rt(s)[:, None]
    return (s * q).reshape(v.shape), 4.5

def quant_q4_0s1(v, wmode):
    """q4_0s with a SINGLE scale candidate (signed absmax → -8) + closed-form
    weighted LSQ rescale — the near-free C-port variant (one pass, no sweep).
    Real-tensor RMSE: 8.93% vs 8.87% (full 19-candidate search), naive 10.18%."""
    b = v.reshape(-1, 32)
    w = get_weights(b, wmode)
    n = b.shape[0]
    idx = np.argmax(np.abs(b), axis=1)
    vmax = b[np.arange(n), idx]
    isc = np.zeros(n, dtype=np.float32)
    nz = vmax != 0
    isc[nz] = -8.0 / vmax[nz]
    q = np.clip(np.rint(b * isc[:, None]), -8, 7)
    num = (w * b * q).sum(axis=1)
    den = (w * q * q).sum(axis=1)
    s = fp16_rt(safediv(num, den.astype(num.dtype)))[:, None]
    return (s * q).reshape(v.shape), 4.5

IQ4NL_KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1,
                     13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
IQ4NL_MID = (IQ4NL_KV[:-1] + IQ4NL_KV[1:]) / 2.0

def quant_iq4_nl(v, wmode):
    """IQ4_NL-style: block 32, fp16 scale, 4-bit index into a NON-LINEAR 16-value
    LUT (llama.cpp kvalues). Same 4.5 bpw as Q4_0 — quality-only upgrade."""
    b = v.reshape(-1, 32)
    w = get_weights(b, wmode)
    n = b.shape[0]
    idx = np.argmax(np.abs(b), axis=1)
    vmax = b[np.arange(n), idx]
    best_err = np.full(n, np.inf)
    best_d = np.zeros(n, dtype=np.float32)
    best_i = np.zeros(b.shape, dtype=np.int64)
    nz = vmax != 0
    for anchor in (-127.0, 113.0):
        for k in range(-9, 10):
            d = np.zeros(n, dtype=np.float32)
            d[nz] = vmax[nz] / (anchor * (1.0 + 0.02 * k))
            t = safediv(b, d[:, None] * np.ones_like(b))
            qi = np.searchsorted(IQ4NL_MID, t)
            kv = IQ4NL_KV[qi]
            num = (w * b * kv).sum(axis=1)
            den = (w * kv * kv).sum(axis=1)
            ds = safediv(num, den.astype(num.dtype))
            err = (w * (b - ds[:, None] * kv) ** 2).sum(axis=1)
            better = err < best_err
            best_err[better] = err[better]
            best_d[better] = ds[better]
            best_i[better] = qi[better]
    d = fp16_rt(best_d)[:, None]
    return (d * IQ4NL_KV[best_i]).reshape(v.shape), 4.5

def qkx_asym(v, wmode, sub, qmax, smax):
    """Two-level asymmetric k-quant core (shared by q2_k/q4_k shapes):
    sub-blocks of `sub`, q∈[0,qmax], v ≈ d·sc·q − dmin·m with sc,m ∈ [0,smax]
    quantized against fp16 d/dmin per 256-superblock."""
    vb = v.reshape(-1, sub)
    wb = get_weights(vb, wmode)
    n = vb.shape[0]
    vmin = np.minimum(vb.min(axis=1), 0.0)
    vmax = vb.max(axis=1)
    s = np.maximum((vmax - vmin) / qmax, 0.0)
    mn = vmin.copy()
    best_err = np.full(n, np.inf)
    best_s = s.copy(); best_mn = mn.copy()
    for _ in range(6):
        q = np.clip(np.rint(safediv(vb - mn[:, None], s[:, None] * np.ones_like(vb))), 0, qmax)
        sw = wb.sum(axis=1); swq = (wb * q).sum(axis=1); swq2 = (wb * q * q).sum(axis=1)
        swv = (wb * vb).sum(axis=1); swvq = (wb * vb * q).sum(axis=1)
        det = sw * swq2 - swq * swq
        ok = det > 0
        s_new = np.where(ok, safediv(sw * swvq - swq * swv, det), s)
        mn_new = np.where(ok, safediv(swq2 * swv - swq * swvq, det), mn)
        s = np.maximum(s_new, 0.0)
        mn = np.minimum(mn_new, 0.0)
        err = (wb * (vb - s[:, None] * q - mn[:, None]) ** 2).sum(axis=1)
        better = err < best_err
        best_err[better] = err[better]; best_s[better] = s[better]; best_mn[better] = mn[better]
    s, mn = best_s, best_mn
    nsub = 256 // sub
    ss = s.reshape(-1, nsub); mm = (-mn).reshape(-1, nsub)
    d = fp16_rt(ss.max(axis=1) / smax)
    dmin = fp16_rt(mm.max(axis=1) / smax)
    sc_q = np.clip(np.rint(safediv(ss, d[:, None] * np.ones_like(ss))), 0, smax)
    m_q = np.clip(np.rint(safediv(mm, dmin[:, None] * np.ones_like(mm))), 0, smax)
    s_f = (d[:, None] * sc_q).reshape(-1)[:, None]
    mn_f = (-(dmin[:, None] * m_q)).reshape(-1)[:, None]
    q = np.clip(np.rint(safediv(vb - mn_f, s_f)), 0, qmax)
    return (s_f * q + mn_f).reshape(v.shape)

def quant_q4_k(v, wmode):
    """Q4_K-style: superblock 256 = 8 sub-blocks of 32, asymmetric q∈[0,15],
    6-bit sub-scales/mins vs fp16 d/dmin. 4.5 bpw — quality-only upgrade vs Q4_0."""
    assert v.shape[1] % 256 == 0
    return qkx_asym(v, wmode, sub=32, qmax=15, smax=63), 4.5

def quant_q3_k(v, wmode):
    """Superblock 256 = 16 sub-blocks of 16; 6-bit signed sub-scales vs fp16 d."""
    rows, cols = v.shape
    assert cols % 256 == 0, f"cols {cols} not divisible by 256"
    vb = v.reshape(-1, 16)
    wb = get_weights(vb, wmode)
    _, sc = qx_search(vb, wb, nmax=4)
    scs = sc.reshape(-1, 16)
    n = scs.shape[0]
    idx = np.argmax(np.abs(scs), axis=1)
    smax = scs[np.arange(n), idx]
    d = np.zeros(n, dtype=np.float32)
    nz = smax != 0
    d[nz] = fp16_rt(smax[nz] / -32.0)
    l = np.clip(np.rint(safediv(scs, np.where(nz, d, 1)[:, None] * np.ones_like(scs))), -32, 31)
    l[~nz] = 0
    s_final = (d[:, None] * l).reshape(-1)[:, None]
    q = np.clip(round_c(safediv(vb, s_final)), -4, 3)
    return (s_final * q).reshape(v.shape), 3.4375

def quant_q2_k(v, wmode):
    """Superblock 256 = 16 sub-blocks of 16; asymmetric: x ≈ d·sc·q − dmin·m, q∈[0,3]."""
    rows, cols = v.shape
    assert cols % 256 == 0
    vb = v.reshape(-1, 16)
    wb = get_weights(vb, wmode)
    n = vb.shape[0]
    vmin = np.minimum(vb.min(axis=1), 0.0)
    vmax = vb.max(axis=1)
    s = np.maximum((vmax - vmin) / 3.0, 0.0)
    mn = vmin.copy()
    best_err = np.full(n, np.inf)
    best_s = s.copy(); best_mn = mn.copy()
    for _ in range(6):
        q = np.clip(np.rint(safediv(vb - mn[:, None], s[:, None] * np.ones_like(vb))), 0, 3)
        sw = wb.sum(axis=1); swq = (wb * q).sum(axis=1); swq2 = (wb * q * q).sum(axis=1)
        swv = (wb * vb).sum(axis=1); swvq = (wb * vb * q).sum(axis=1)
        det = sw * swq2 - swq * swq
        ok = det > 0
        s_new = np.where(ok, safediv(sw * swvq - swq * swv, det), s)
        mn_new = np.where(ok, safediv(swq2 * swv - swq * swvq, det), mn)
        s = np.maximum(s_new, 0.0)
        mn = np.minimum(mn_new, 0.0)
        err = (wb * (vb - s[:, None] * q - mn[:, None]) ** 2).sum(axis=1)
        better = err < best_err
        best_err[better] = err[better]; best_s[better] = s[better]; best_mn[better] = mn[better]
    s, mn = best_s, best_mn
    ss = s.reshape(-1, 16); mm = (-mn).reshape(-1, 16)
    d = fp16_rt(ss.max(axis=1) / 15.0)
    dmin = fp16_rt(mm.max(axis=1) / 15.0)
    sc_q = np.clip(np.rint(safediv(ss, d[:, None] * np.ones_like(ss))), 0, 15)
    m_q = np.clip(np.rint(safediv(mm, dmin[:, None] * np.ones_like(mm))), 0, 15)
    s_f = (d[:, None] * sc_q).reshape(-1)[:, None]
    mn_f = (-(dmin[:, None] * m_q)).reshape(-1)[:, None]
    q = np.clip(np.rint(safediv(vb - mn_f, s_f)), 0, 3)
    return (s_f * q + mn_f).reshape(v.shape), 2.625

FORMATS = {
    "q4_0": quant_q4_0, "q3_0": quant_q3_0, "q2_0": quant_q2_0,
    "q3_k": quant_q3_k, "q2_k": quant_q2_k,
    "q4_0s": quant_q4_0s, "iq4_nl": quant_iq4_nl, "q4_k": quant_q4_k,
    "q4_0s1": quant_q4_0s1,
}

def cp_targets(names, scope, only):
    trans = [n for n in names
             if n.startswith(CP + "model.layers.") and n.endswith("_proj.weight")]
    heads = [n for n in names if re.match(re.escape(CP) + r"lm_head\.\d+\.weight$", n)]
    talker = [n for n in names
              if n.startswith("talker.model.layers.") and n.endswith("_proj.weight")]
    sel = {"all": trans + heads, "transformer": trans, "heads": heads,
           "talker": talker}[scope]
    if only:
        sel = [n for n in sel if re.search(only, n)]
    return sorted(sel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", required=True, choices=sorted(FORMATS))
    ap.add_argument("--scope", default="all",
                    choices=["all", "transformer", "heads", "talker"])
    ap.add_argument("--only", default=None, help="regex filter on tensor names")
    ap.add_argument("--weights", default="x2", choices=["x2", "abs", "none"],
                    help="error weighting in the k-quant scale search")
    args = ap.parse_args()

    src = os.path.join(args.model, "model.safetensors")
    hdr, data_start = read_header(src)
    names = [k for k in hdr if k != "__metadata__"]
    targets = cp_targets(names, args.scope, args.only)
    if not targets:
        sys.exit("no target tensors matched")

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "model.safetensors")
    if os.path.exists(dst):
        os.remove(dst)
    t0 = time.time()
    if subprocess.call(["cp", "-c", src, dst], stderr=subprocess.DEVNULL) != 0:
        shutil.copyfile(src, dst)
    print(f"cloned model.safetensors ({time.time()-t0:.1f}s)")

    for entry in os.listdir(args.model):
        if entry == "model.safetensors":
            continue
        link = os.path.join(args.out, entry)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(args.model, entry)), link)

    qfn = FORMATS[args.format]
    tot_sq = tot_ref = 0.0
    with open(dst, "r+b") as f:
        for name in targets:
            meta = hdr[name]
            assert meta["dtype"] == "BF16", f"{name}: {meta['dtype']}"
            off0, off1 = meta["data_offsets"]
            rows, cols = meta["shape"]
            f.seek(data_start + off0)
            raw = np.frombuffer(f.read(off1 - off0), dtype=np.uint16)
            v = bf16_to_f32(raw).reshape(rows, cols)
            dq, bpw = qfn(v, args.weights)
            sq = float(((dq - v) ** 2).sum()); ref = float((v ** 2).sum())
            tot_sq += sq; tot_ref += ref
            print(f"  {name.removeprefix(CP):48s} [{rows}x{cols}]  "
                  f"rel-RMSE {100*np.sqrt(sq/ref):6.3f}%")
            f.seek(data_start + off0)
            f.write(f32_to_bf16(np.ascontiguousarray(dq.reshape(-1))).tobytes())
    print(f"DONE format={args.format} scope={args.scope} weights={args.weights} "
          f"bpw={bpw} tensors={len(targets)} "
          f"overall rel-RMSE {100*np.sqrt(tot_sq/tot_ref):.3f}%  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
