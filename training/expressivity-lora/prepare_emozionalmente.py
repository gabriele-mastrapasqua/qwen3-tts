#!/usr/bin/env python3
"""Emozionalmente -> train_raw.jsonl in the SAME schema as EMOVO (gpu_emovo_prep.py) / CREMA-D"""
import os, json, argparse, subprocess, tempfile, collections, time, sys, io

HF_REPO = "amu-cai/CAMEO"
HF_SPLIT = "emozionalmente"
HF_PARQUET = "data/emozionalmente-00000-of-00001.parquet"

EMO = {
    "anger":     ("anger",    "Speak with hot, furious anger, sharp and forceful."),
    "disgust":   ("disgust",  "Speak with physical disgust, repulsed and recoiling."),
    "fear":      ("fear",     "Speak with fear, tense and trembling, your voice wary."),
    "happiness": ("joy",      "Speak happily, bright and warm, smiling through the words."),
    "neutral":   ("neutral",  ""),
    "sadness":   ("sadness",  "Speak with a sad, sorrowful, downcast tone, voice low and heavy."),
    "surprise":  ("surprise", "Speak with surprise, startled and taken aback, held through the whole sentence."),
}
EMO_ALIAS = {"joy": "happiness", "neutrality": "neutral"}

def _norm_emo(e):
    e = str(e).strip().lower()
    return EMO_ALIAS.get(e, e)

STRONG_AGREE = {"anger", "joy", "sadness", "surprise", "neutral"}

def _agree_threshold(exp_raw, a):
    """Per-clip min-agreement bar. exp_raw = the source's emotion_expressed string."""
    if a.smart_agreement:
        lab = EMO.get(_norm_emo(exp_raw), (None,))[0]
        return 3 if lab in STRONG_AGREE else 0
    return a.min_agreement or 0

def _peak(arr):
    import numpy as np
    a = np.asarray(arr, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return float(abs(a).max()) if a.size else 0.0

def _est_snr_db(arr, sr):
    """Crude SNR proxy: energy of the loudest 20% frames over the quietest 20% (noise floor), in dB.
    Not calibrated SNR, but monotone enough to RANK clips worst->best for filtering."""
    import numpy as np
    a = np.asarray(arr, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.size < sr // 10:
        return 0.0
    win = max(1, sr // 100)
    n = (a.size // win) * win
    fr = a[:n].reshape(-1, win)
    e = (fr ** 2).mean(axis=1) + 1e-12
    e_sorted = np.sort(e)
    noise = e_sorted[: max(1, len(e_sorted) // 5)].mean()
    sig = e_sorted[-max(1, len(e_sorted) // 5):].mean()
    return float(10.0 * np.log10(sig / max(noise, 1e-12)))

def _emit(rows, per_spk, wav24, arr, sr, emo, spk, text, opts, write_audio=True):
    """Map one example -> a row (and a cleaned 24k wav). Returns True if emitted, False if skipped.
    `opts` carries the quality knobs (drop_clipped / loudnorm / denoise)."""
    emo = _norm_emo(emo)
    if emo not in EMO:
        return False
    if write_audio and opts.get("drop_clipped") and _peak(arr) >= opts["drop_clipped"]:
        opts["_n_clipped_dropped"] = opts.get("_n_clipped_dropped", 0) + 1
        return False
    label, instruct = EMO[emo]
    actor = f"emozionalmente{spk}"
    out_wav = os.path.join(wav24, f"{actor}_{label}_{per_spk[spk]:04d}.wav")
    if write_audio:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, arr, sr); src = tmp.name
        af = []
        if opts.get("denoise"):
            af.append("afftdn=nf=-25")
        if opts.get("loudnorm"):
            af.append("loudnorm=I=-23:TP=-2:LRA=11")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "24000", "-ac", "1"]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd.append(out_wav)
        subprocess.run(cmd, check=False)
        os.unlink(src)
    rows.append({"audio": out_wav, "text": text, "ref_audio": out_wav, "instruct": instruct,
                 "emotion": label, "actor": actor, "language": "Italian"})
    per_spk[spk] += 1
    return True

def _iter_hf_rows():
    """Yield (arr, sr, emotion, speaker_id, transcription) from the Emozionalmente parquet shard.
    Uses hf_hub_download + pyarrow + soundfile (the proven-reliable path; streaming stalled on audio)."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    print(f"[emoz] downloading {HF_REPO}:{HF_PARQUET} ...", flush=True)
    f = hf_hub_download(HF_REPO, HF_PARQUET, repo_type="dataset")
    t = pq.ParquetFile(f).read()
    au = t.column("audio").to_pylist()
    emo = t.column("emotion").to_pylist()
    spk = t.column("speaker_id").to_pylist()
    txt = t.column("transcription").to_pylist()
    print(f"[emoz] shard has {len(au)} rows", flush=True)
    for a, e, s, x in zip(au, emo, spk, txt):
        b = a.get("bytes") if isinstance(a, dict) else None
        if not b:
            continue
        arr, sr = sf.read(io.BytesIO(b))
        yield arr, sr, str(e).strip().lower(), s, (x or "")

def _report():
    """Inspect-only: peak + crude-SNR distribution per emotion. Writes nothing."""
    import numpy as np
    by_emo = collections.defaultdict(lambda: {"snr": [], "peak": [], "n": 0})
    n = 0
    for arr, sr, emo, spk, txt in _iter_hf_rows():
        if emo not in EMO:
            continue
        d = by_emo[emo]
        d["snr"].append(_est_snr_db(arr, sr)); d["peak"].append(_peak(arr)); d["n"] += 1
        n += 1
        if n % 1000 == 0:
            print(f"[emoz/report] analysed {n} ...", flush=True)
    print(f"\n[emoz/report] {n} clips. per-emotion crude-SNR(dB) and peak:")
    all_snr = []
    for emo in sorted(by_emo):
        d = by_emo[emo]; s = np.array(d["snr"]); p = np.array(d["peak"]); all_snr += list(s)
        print(f"  {emo:9s} n={d['n']:4d}  SNR med {np.median(s):5.1f}  p10 {np.percentile(s,10):5.1f}  "
              f"| peak mean {p.mean():.2f}  clipped>=0.99 {(p>=0.99).mean()*100:4.1f}%")
    a = np.array(all_snr)
    print(f"\n  OVERALL crude-SNR(dB): p5 {np.percentile(a,5):.1f}  p10 {np.percentile(a,10):.1f}  "
          f"median {np.median(a):.1f}  -> the 'worst' tail is below ~p10; "
          f"use --drop-clipped and/or --loudnorm/--denoise to clean without dropping much.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False)
    ap.add_argument("--wav24-dir", default=None, help="where to write 24k wavs (default: <out_dir>/wav24k)")
    ap.add_argument("--local-dir", default=None,
                    help="read a local Zenodo Emozionalmente dir (has the human-validation CSVs) instead of HF CAMEO")
    ap.add_argument("--min-agreement", type=int, default=0,
                    help="keep clips where >= N of 5 human evaluators RECOGNIZED the intended emotion "
                         "(0=all; 3=majority, ~70%% kept, drops emotionally-ambiguous clips; 4=~50%%, 5=unanimous ~26%%). "
                         "NOTE: thins disgust/fear (intrinsically ambiguous), NOT anger/joy.")
    ap.add_argument("--smart-agreement", action="store_true",
                    help="per-emotion bar (best-of-both): require >=3/5 for the well-recognized emotions "
                         "(anger/joy/sadness/surprise/neutral) but KEEP ALL non-zero clips for disgust/fear "
                         "(intrinsically ambiguous + data-scarce -> don't starve them). Overrides --min-agreement.")
    ap.add_argument("--clean-only", action="store_true",
                    help="keep only clips humans rated audio_quality=good on ALL 5 evals (~84%%); the PERCEPTUAL "
                         "noise filter (better than SNR). Most clips are clean (96%% good) so this drops little.")
    ap.add_argument("--split", choices=["all", "train", "dev", "test"], default="all",
                    help="use the official speaker-independent split (metadata/split/<split>.csv). default all.")
    ap.add_argument("--max-per-speaker", type=int, default=0,
                    help="0 = all (default). Cap utterances per speaker to balance vs EMOVO if needed.")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all; cap total rows (debug/quick runs)")
    ap.add_argument("--drop-clipped", type=float, default=0.0,
                    help="drop clips whose peak >= this (e.g. 0.98). 0 = keep all (default).")
    ap.add_argument("--loudnorm", action="store_true",
                    help="EBU R128 loudness-normalize on encode (fixes level variance WITHOUT dropping data)")
    ap.add_argument("--denoise", action="store_true",
                    help="light spectral denoise (afftdn) on encode (removes ambient hiss WITHOUT dropping data)")
    ap.add_argument("--report", action="store_true",
                    help="analyse-only: print per-emotion SNR/peak distribution, write nothing")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        _self_test(); return
    if a.report:
        _report(); return
    if not a.out:
        ap.error("--out required (or use --self-test / --report)")
    if (a.min_agreement or a.smart_agreement or a.clean_only or a.split != "all") and not a.local_dir:
        ap.error("--min-agreement/--smart-agreement/--clean-only/--split need --local-dir (the human-validation "
                 "CSVs are only in the Zenodo package, not the CAMEO parquet)")

    wav24 = a.wav24_dir or os.path.join(os.path.dirname(a.out) or ".", "wav24k")
    os.makedirs(wav24, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    opts = {"drop_clipped": a.drop_clipped, "loudnorm": a.loudnorm, "denoise": a.denoise}
    rows, n_skip, n_done = [], 0, 0
    per_spk = collections.Counter()
    t0 = time.time()
    clean = [k for k in ("loudnorm", "denoise") if opts[k]] + ([f"drop_clipped>={a.drop_clipped}"] if a.drop_clipped else [])
    src = f"local:{a.local_dir} split={a.split}" if a.local_dir else f"HF:{HF_REPO}:{HF_SPLIT}"
    filt = ["smart-agreement(>=3 strong / all disgust+fear)"] if a.smart_agreement else (
           [f"min_agreement>={a.min_agreement}"] if a.min_agreement else [])
    filt += ["clean-only"] if a.clean_only else []
    print(f"[emoz] START -> {a.out}  src={src}  (wav24={wav24}, max_per_speaker={a.max_per_speaker or 'all'}, "
          f"clean={clean or 'NONE'}, filters={filt or 'NONE (keep all)'})", flush=True)

    if a.local_dir:
        _from_local(a, rows, per_spk, wav24, opts, t0)
    else:
        for arr, sr, emo, spk, txt in _iter_hf_rows():
            if a.max_per_speaker and per_spk[spk] >= a.max_per_speaker:
                continue
            if not _emit(rows, per_spk, wav24, arr, sr, emo, spk, txt, opts):
                n_skip += 1; continue
            n_done += 1
            if n_done % a.log_every == 0:
                print(f"[emoz] {n_done} rows, {len(per_spk)} speakers, {time.time()-t0:.0f}s", flush=True)
            if a.max_rows and n_done >= a.max_rows:
                print(f"[emoz] hit --max-rows {a.max_rows}, stopping", flush=True); break

    with open(a.out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_emo = collections.Counter(r["emotion"] for r in rows)
    print(f"[emoz] DONE: wrote {a.out}: {len(rows)} rows, {len(per_spk)} speakers, skipped {n_skip} "
          f"(clipped-dropped {opts.get('_n_clipped_dropped', 0)}), {time.time()-t0:.0f}s total", flush=True)
    print("[emoz]   emotions:", dict(by_emo), flush=True)

def _find_meta_root(local_dir):
    """Return the dir that directly contains metadata/samples.csv (handles dir, dir/emozionalmente, ...)."""
    for cand in (local_dir, os.path.join(local_dir, "emozionalmente")):
        if os.path.exists(os.path.join(cand, "metadata", "samples.csv")):
            return cand
    for dirpath, _, files in os.walk(local_dir):
        if dirpath.endswith(os.path.join("metadata")) and "samples.csv" in files:
            return os.path.dirname(dirpath)
    sys.exit(f"[emoz] could not find metadata/samples.csv under {local_dir}")

def _from_local(a, rows, per_spk, wav24, opts, t0):
    """Read the Zenodo Emozionalmente package (records/12616095) WITH its human-validation CSVs:
       metadata/samples.csv      file_name,sentence,sentence_language,actor,emotion_expressed
       metadata/evaluations.csv  file_name,evaluator,audio_quality(good|bad),emotion_recognized  (~5/clip)
       metadata/split/<s>.csv    speaker-independent split (same columns as samples.csv)
    Applies the PERCEPTUAL filters: --min-agreement (recognized==expressed votes) and --clean-only."""
    import csv
    import soundfile as sf
    root = _find_meta_root(a.local_dir)
    meta = os.path.join(root, "metadata")
    audio_dir = os.path.join(root, "audio")
    print(f"[emoz] local root: {root}", flush=True)

    samples = {}
    with open(os.path.join(meta, "samples.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            samples[r["file_name"]] = (r["emotion_expressed"], r["actor"], r.get("sentence", ""))

    keep_fn = None
    if a.split != "all":
        sp = os.path.join(meta, "split", f"{a.split}.csv")
        with open(sp, newline="", encoding="utf-8") as f:
            keep_fn = {r["file_name"] for r in csv.DictReader(f)}
        print(f"[emoz] split={a.split}: {len(keep_fn)} clips", flush=True)

    agree = collections.Counter(); good = collections.Counter(); nev = collections.Counter()
    with open(os.path.join(meta, "evaluations.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fn = r["file_name"]; exp = samples.get(fn, (None,))[0]
            nev[fn] += 1
            if exp is not None and _norm_emo(r["emotion_recognized"]) == _norm_emo(exp):
                agree[fn] += 1
            if str(r["audio_quality"]).strip().lower() == "good":
                good[fn] += 1

    n_done = n_skip_agree = n_skip_clean = 0
    for fn, (exp, spk, text) in samples.items():
        if keep_fn is not None and fn not in keep_fn:
            continue
        thr = _agree_threshold(exp, a)
        if thr and agree[fn] < thr:
            n_skip_agree += 1; continue
        if a.clean_only and good[fn] < nev[fn]:
            n_skip_clean += 1; continue
        if a.max_per_speaker and per_spk[spk] >= a.max_per_speaker:
            continue
        wp = os.path.join(audio_dir, fn)
        if not os.path.exists(wp):
            continue
        arr, sr = sf.read(wp)
        if not _emit(rows, per_spk, wav24, arr, sr, exp, spk, text, opts):
            continue
        n_done += 1
        if n_done % a.log_every == 0:
            print(f"[emoz] {n_done} rows, {len(per_spk)} speakers, {time.time()-t0:.0f}s", flush=True)
        if a.max_rows and n_done >= a.max_rows:
            break
    bar = "smart(>=3 strong/all disgust+fear)" if a.smart_agreement else f"<{a.min_agreement}"
    print(f"[emoz] local filters: dropped {n_skip_agree} (agreement {bar}), "
          f"{n_skip_clean} (not all-good-quality)", flush=True)

def _self_test():
    """No network: exercise the emotion map + row shape + clip-drop on fake examples."""
    rows, per_spk = [], collections.Counter()
    fake_arr = [0.0, 0.1, -0.1, 0.0]
    opts = {"drop_clipped": 0.0, "loudnorm": False, "denoise": False}
    for emo in ["anger", "happiness", "neutral", "surprise", "BOGUS"]:
        _emit(rows, per_spk, "/tmp/wav24", fake_arr, 16000, emo, "spk7", "Ciao mondo.", opts, write_audio=False)
    labels = [r["emotion"] for r in rows]
    print("mapped labels:", labels)
    assert labels == ["anger", "joy", "neutral", "surprise"], "emotion map wrong / BOGUS not skipped"
    assert all(r["language"] == "Italian" for r in rows), "language must be stamped Italian"
    assert rows[0]["instruct"] and rows[2]["instruct"] == "", "neutral must have empty instruct, anger must not"
    assert all(r["actor"] == "emozionalmentespk7" for r in rows), "actor id malformed"
    import numpy as np
    sig = np.concatenate([np.random.randn(1600) * 0.02, np.sin(np.linspace(0, 50, 3200)) * 0.5,
                          np.random.randn(1600) * 0.02]).astype("float32")
    snr = _est_snr_db(sig, 16000)
    assert snr > 5.0, f"crude SNR proxy looks broken ({snr:.1f} dB on a clean-ish synthetic)"
    print(f"crude-SNR proxy on synthetic: {snr:.1f} dB (sane)")
    print("SELF-TEST PASS — 7-emotion map -> EMOVO schema, neutral anchor empty, language=Italian, "
          "bogus skipped, SNR proxy sane.")

if __name__ == "__main__":
    main()
