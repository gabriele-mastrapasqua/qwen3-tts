#!/usr/bin/env python3
"""ASR-gate: an OBJECTIVE "is this take realistically OK?" judge for emotion seed-audition.

The emotion (anger/sad/...) does NOT change the WORDS, so for a known target sentence a clean take
transcribes back to (near) that text, while a BROKEN take — gibberish, a code-switch to Chinese, a
runaway repetition — transcribes to something far from it. So:

    score = CER( asr_transcript , target_text )      (lower = cleaner, more intelligible, right language)

This catches exactly the failures the duration+glitch picker can't see (gibberish at a normal duration,
language drift). It is ORTHOGONAL to the emotion, so it never penalizes an expressive-but-correct take.

ASR engine: qwen-asr (antirez's pure-C CPU Qwen3-ASR — same family/ethos as qwen-tts, MIT, no Python ML
deps; transcript on stdout with --silent). Build it (`make blas` in ../qwen-asr) + a model first.

USAGE:
  tests/audition_asr_gate.py --wav-dir samples/emotion_seeds/italian/ryan \\
      --text "Domani mattina ci vediamo davanti alla stazione." --lang Italian \\
      --asr-bin ../qwen-asr/qwen_asr --asr-model ../qwen-asr/qwen3-asr-0.6b
  # or a single cell's takes by glob:
  tests/audition_asr_gate.py --wav "samples/.../anger.seed*.wav" --text "..." --lang Italian ...

Per take it prints: seed | CER | OK/BROKEN | transcript. Plus the best (lowest-CER) seed = the
ASR-recommended pick. Pairs with the in-binary glitch+duration picker (Tier 1); this is Tier 2.

  --self-test   exercise the CER/normalize math with NO model/binary (CI-safe).
"""
import argparse, glob, os, re, subprocess, sys


def normalize(s):
    """lowercase, punctuation→space, collapse whitespace (same spirit as asr_regression.py 'norm')."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def levenshtein(a, b):
    """character-level edit distance (stdlib only)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp, ref):
    """normalized character error rate of hyp vs ref; clamped to [0, 1+]. 0 = perfect, ~1 = unrelated."""
    h, r = normalize(hyp), normalize(ref)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / max(1, len(r))


def seed_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def transcribe(asr_bin, asr_model, wav, lang, timeout):
    cmd = [asr_bin, "-d", asr_model, "-i", wav, "--silent"]
    if lang:
        cmd += ["--language", lang]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return ""
    except FileNotFoundError:
        print(f"ERROR: asr binary not found: {asr_bin}", file=sys.stderr)
        sys.exit(2)


def run_self_test():
    assert normalize("Ciao,  MONDO!!") == "ciao mondo"
    assert levenshtein("kitten", "sitting") == 3
    ref = "Domani mattina ci vediamo davanti alla stazione."
    assert cer(ref, ref) == 0.0, "identical → CER 0"
    assert cer("Domani mattina ci vediamo davanti alla stazione", ref) < 0.05, "punct-only diff → tiny"
    assert cer("你好世界 random gibberish here", ref) > 0.6, "gibberish/other-lang → large CER"
    assert cer("domani domani domani domani domani", ref) > 0.4, "runaway repetition → large CER"
    print("SELF-TEST PASSED (normalize + levenshtein + cer)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", help="dir of <stem>.seedNN.wav takes")
    ap.add_argument("--wav", help="glob of takes (quote it), e.g. 'cell/anger.seed*.wav'")
    ap.add_argument("--text", help="the target sentence the takes were asked to speak")
    ap.add_argument("--lang", default="", help="force ASR language (e.g. Italian) — recommended")
    ap.add_argument("--asr-bin", default="../qwen-asr/qwen_asr")
    ap.add_argument("--asr-model", default="../qwen-asr/qwen3-asr-0.6b")
    ap.add_argument("--ok-cer", type=float, default=0.25, help="CER <= this = OK, else BROKEN")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        run_self_test()
        return

    if not a.text:
        print("ERROR: --text (target sentence) is required", file=sys.stderr)
        sys.exit(2)
    files = []
    if a.wav_dir:
        files += sorted(glob.glob(os.path.join(a.wav_dir, "*.seed*.wav")))
    if a.wav:
        files += sorted(glob.glob(a.wav))
    files = sorted(set(files), key=seed_of)
    if not files:
        print("ERROR: no take wavs found (use --wav-dir or --wav)", file=sys.stderr)
        sys.exit(2)
    if not (os.path.exists(a.asr_bin) and os.path.isdir(a.asr_model)):
        print(f"ERROR: build qwen-asr + a model first ({a.asr_bin} / {a.asr_model} missing).\n"
              f"  cd ../qwen-asr && make blas && ./download_model.sh", file=sys.stderr)
        sys.exit(3)

    print(f"target: \"{a.text}\"  (lang={a.lang or 'auto'})  ok_cer<={a.ok_cer}")
    print(f"{'seed':>5} | {'CER':>5} | verdict | transcript")
    print("-" * 70)
    best = None
    for f in files:
        s = seed_of(f)
        hyp = transcribe(a.asr_bin, a.asr_model, f, a.lang, a.timeout)
        c = cer(hyp, a.text)
        verdict = "OK    " if c <= a.ok_cer else "BROKEN"
        print(f"{s:>5} | {c:>5.2f} | {verdict} | {hyp[:60]}")
        if best is None or c < best[1]:
            best = (s, c, f)
    if best:
        tag = "OK" if best[1] <= a.ok_cer else "all takes look BROKEN (lowest-CER shown)"
        print("-" * 70)
        print(f"ASR-recommended seed: {best[0]}  (CER {best[1]:.2f}, {tag})")


if __name__ == "__main__":
    main()
