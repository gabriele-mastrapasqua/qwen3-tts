import struct, sys, os

def parse(path):
    sz = os.path.getsize(path)
    f = open(path, 'rb')
    def rd(n): return f.read(n)
    def u32():
        b = f.read(4)
        return struct.unpack('<I', b)[0]
    out = {}
    magic = rd(4); assert magic == b'QVCE', magic
    ver = u32(); out['version'] = ver
    p_emb_start = f.tell()
    enc_dim = 1024
    if ver >= 2:
        enc_dim = u32()
    out['enc_dim'] = enc_dim
    spk_bytes = enc_dim*4
    f.seek(spk_bytes, 1)
    out['spk_emb_bytes'] = spk_bytes
    rtlen = u32()
    out['ref_text_len'] = rtlen
    rt = rd(rtlen).decode('utf-8','replace') if rtlen else ''
    out['ref_text'] = rt
    nrf = u32()
    out['n_ref_frames'] = nrf
    rc_bytes = nrf*16*4
    f.seek(rc_bytes, 1)
    out['ref_codes_bytes'] = rc_bytes
    # ICL prefix = spk_emb + ref_text + ref_codes + tiny headers
    icl_prefix_end = f.tell()
    out['icl_prefix_bytes'] = icl_prefix_end  # everything from start of file
    # META
    meta_start = f.tell()
    mm = rd(4)
    meta_bytes = 0
    if mm == b'META':
        # lang_id u32, lang_name 16, model_size u32, enc_dim u32, ref_dur f32, voice_name 64, flags u32
        body = 4+16+4+4+4+64+4
        f.seek(body, 1)
        meta_bytes = 4+body
        out['meta_voice'] = None
    else:
        f.seek(meta_start, 0)
    out['meta_bytes'] = meta_bytes
    # TPAD
    tp_start = f.tell()
    tm = rd(4)
    tpad_bytes = 0
    if tm == b'TPAD':
        th = u32()
        # 3 * hidden floats
        body = 3*th*4
        f.seek(body, 1)
        tpad_bytes = 4+4+body
        out['tpad_hidden'] = th
    else:
        f.seek(tp_start, 0)
    out['tpad_bytes'] = tpad_bytes
    # WOVR
    wo_start = f.tell()
    wm = rd(4)
    wovr_bytes = 0
    if wm == b'WOVR':
        wh = u32(); wth = u32(); wcv = u32()
        # fc1: th*th u16, fc1_b: th f32, fc2: h*th u16, fc2_b: h f32, ce: cv*h u16
        body = wth*wth*2 + wth*4 + wh*wth*2 + wh*4 + wcv*wh*2
        f.seek(body, 1)
        wovr_bytes = 4+12+body
        out['wovr_dims'] = (wh, wth, wcv)
    else:
        f.seek(wo_start, 0)
    out['wovr_bytes'] = wovr_bytes
    # remainder = WDELTA/WFULL bulk
    wdelta_start = f.tell()
    out['wdelta_bulk_bytes'] = sz - wdelta_start
    out['total_bytes'] = sz
    f.close()
    return out

for p in sys.argv[1:]:
    o = parse(p)
    print(f"\n=== {os.path.basename(p)} (v{o['version']}) ===")
    print(f"  total file              : {o['total_bytes']/1024/1024:10.2f} MB")
    print(f"  ── ICL PREFIX ──────────────────────────────")
    print(f"  speaker embedding       : {o['spk_emb_bytes']:>12,} B   ({o['enc_dim']} floats)")
    print(f"  ref_text ('{o['ref_text'][:40]}...')")
    print(f"      ref_text bytes      : {o['ref_text_len']:>12,} B")
    print(f"  ref_codes               : {o['ref_codes_bytes']:>12,} B   ({o['n_ref_frames']} frames = {o['n_ref_frames']/12.5:.1f}s, 16 cb)")
    icl_total = o['spk_emb_bytes'] + o['ref_text_len'] + o['ref_codes_bytes'] + 20
    print(f"  ICL prefix TOTAL        : {icl_total/1024:>12,.1f} KB")
    print(f"  ── weight-swap sections (skipped by --icl-only WDELTA) ──")
    print(f"  META                    : {o['meta_bytes']:>12,} B")
    print(f"  TPAD                    : {o['tpad_bytes']:>12,} B")
    print(f"  WOVR                    : {o['wovr_bytes']:>12,} B   {o.get('wovr_dims','')}")
    print(f"  WDELTA bulk             : {o['wdelta_bulk_bytes']/1024/1024:>12,.2f} MB  ({o['wdelta_bulk_bytes']/o['total_bytes']*100:.3f}% of file)")
    print(f"  ── 'qvoice-lite' (ICL+META+TPAD+WOVR, no WDELTA) ──")
    lite = icl_total + o['meta_bytes'] + o['tpad_bytes'] + o['wovr_bytes']
    print(f"  lite TOTAL              : {lite/1024:>12,.1f} KB   ({o['total_bytes']/lite:,.0f}x smaller than full)")
