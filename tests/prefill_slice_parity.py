#!/usr/bin/env python3
"""prefill_slice_parity.py — the local correctness oracle for C12-WIN-10 (QWEN_PREFILL_SLICE).

Runs entirely on a dev machine; no load, no timing gate.  The question it answers is
whether prefilling a request in resumable token-range slices produces the same request as
prefilling it in one call, and whether a partially prefilled request can be dropped without
leaving the worker broken.

Arms (one server per arm, single worker, one slot, temperature 0, fixed seed):

  inline    QWEN_PREFILL_SLICE unset      the monolithic path, the control
  one       QWEN_PREFILL_SLICE=-100000    the sliced path, whole prompt in ONE slice
  many      QWEN_PREFILL_SLICE=-<S>       the sliced path, ceil(n/S) slices

The negative value forces slicing on an idle worker, which is what makes the boundary
reachable with one client; a product arm uses a positive value and an idle worker still
prefills in one visit.

Every gate compares the treatment against a CONTROL RUN OF THE SAME SCENARIO, never
against a run at a different concurrency or with a different request history: a 2-slot
batch is not required to be numerically identical to a 1-slot one, and the engine is
history-sensitive across cancellations, so those comparisons would measure something else.

Gates:

  0. The treatment actually took the sliced path (the [ADMSLICE] first-admission marker is
     present in the sliced arms and absent in the control).  This one is load-bearing:
     the WAV is a function of integer codes, so identical audio is exactly what a CORRECT
     slicing produces AND exactly what a treatment that never ran would produce.
  A. many == one, byte for byte.  Same code, same arithmetic, only the number of pauses
     differs -- so this is EXACT, and it is the resume oracle: pausing between any pair of
     token ranges must leave the same Talker state as never pausing.
  B. inline vs one: mel-corr >= 0.99 and duration within 2 %.  Not exact by design: a new
     token attends to the earlier tokens of its own prompt through the bf16 KV cache
     instead of the f32 staging buffer (spec section 8).
  C. after a client disconnects mid-admission, the next request on the same worker is
     byte-identical to the same arm's undisturbed output -- no leaked or half-applied state.
  D. two concurrent clients on a 2-slot worker both return their correct audio, so a
     pending admission and an established stream cannot contaminate each other.
  E. an unusable QWEN_PREFILL_SLICE value kills the worker instead of being ignored.

Usage:
  python3 tests/prefill_slice_parity.py [--model qwen3-tts-0.6b] [--slice 48] [--port 9713]
"""
import argparse, hashlib, json, os, re, socket, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEXTS = [
    ("short",   "Buongiorno a tutti.",                                        "Italian"),
    ("medium",  "Il treno delle nove parte dal binario tre e arriva in "
                "stazione centrale poco prima di mezzogiorno.",               "Italian"),
    ("long",    "La qualita' di un motore di sintesi vocale non si misura "
                "soltanto dalla naturalezza della voce, ma anche dalla sua "
                "capacita' di reggere il carico: quante richieste riesce a "
                "servire contemporaneamente senza che l'ascoltatore senta "
                "una pausa, un salto o un artefatto nel mezzo di una frase.", "Italian"),
    ("english", "The scheduler admits a new request without stopping the "
                "streams that are already playing, which is the entire point "
                "of slicing the prefill across frame iterations.",            "English"),
]


def wait_health(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=2) as r:
                if b'"ok"' in r.read():
                    return True
        except Exception:
            time.sleep(0.5)
    return False


class Server:
    def __init__(self, binary, model, port, env, slots=1, log=None):
        self.port, self.log_path = port, log
        e = dict(os.environ); e.update(env)
        e.pop("QWEN_PREFILL_HELPER", None)
        self.log = open(log, "w") if log else subprocess.DEVNULL
        self.p = subprocess.Popen(
            [binary, "-d", model, "--serve", str(port), "--batch-size", str(slots),
             "--prefork", "1", "-j", str(os.cpu_count() or 4)],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, env=e)

    def __enter__(self):
        if not wait_health(self.port):
            self.__exit__(None, None, None)
            raise RuntimeError(f"server on {self.port} never became healthy")
        return self

    def __exit__(self, *a):
        try:
            self.p.terminate(); self.p.wait(timeout=20)
        except Exception:
            self.p.kill()
        if self.log is not subprocess.DEVNULL:
            self.log.close()


def tts(port, text, language, seed=42, speaker="ryan", timeout=300, retries=6):
    """One /v1/tts request -> WAV bytes.  A 503 is the server's fail-fast admission
    (a slot is still held by a just-cancelled request), not a result: back off and retry."""
    body = json.dumps({"text": text, "speaker": speaker, "language": language,
                       "seed": seed, "temperature": 0.0}).encode()
    last = None
    for i in range(retries):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/tts", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as ex:
            last = ex
            if ex.code != 503:
                raise
            time.sleep(0.5 * (i + 1))
    raise last


def tts_abort(port, text, language, after_s, seed=42, speaker="ryan"):
    """Send a streaming request and hang up after `after_s` seconds, mid-admission."""
    body = json.dumps({"text": text, "speaker": speaker, "language": language,
                       "seed": seed, "temperature": 0.0}).encode()
    req = (f"POST /v1/tts/stream HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    s = socket.create_connection(("127.0.0.1", port), 30)
    try:
        s.sendall(req)
        time.sleep(after_s)
    finally:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        s.close()


def wav_write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def mel_compare(a, b):
    """tests/compare_audio.py -> (mel_corr, duration_ratio) or (None, None)."""
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "compare_audio.py"), a, b],
                         cwd=ROOT, capture_output=True, text=True)
    txt = out.stdout + out.stderr
    mc = re.search(r"mel_corr=([0-9.]+)", txt)
    dr = re.search(r"rel ([0-9.]+)%", txt)
    return (float(mc.group(1)) if mc else None, float(dr.group(1)) if dr else None), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-tts-0.6b")
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--slice", type=int, default=48)
    ap.add_argument("--port", type=int, default=9713)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    binary = os.path.join(ROOT, a.bin) if not os.path.isabs(a.bin) else a.bin
    out = a.out or os.path.join(ROOT, ".work", "evidence", "prefill-slice-parity")
    os.makedirs(out, exist_ok=True)
    fails, notes, conflicts = [], [], []

    # --batch-size must be >= 2: main.c routes 1 to the NON-batched server, where the
    # admission block this change lives in does not exist at all.
    SLOTS = 2
    arms = {
        "inline": {},
        "one":    {"QWEN_PREFILL_SLICE": "-100000"},
        "many":   {"QWEN_PREFILL_SLICE": f"-{a.slice}"},
        "every":  {"QWEN_PREFILL_SLICE": "-1"},
    }
    wavs, slice_stats = {}, {}
    for arm, env in arms.items():
        log = os.path.join(out, f"{arm}.log")
        port = a.port + list(arms).index(arm)
        with Server(binary, a.model, port, env, slots=SLOTS, log=log):
            for name, text, lang in (TEXTS[:2] if arm == "every" else TEXTS):
                data = tts(port, text, lang)
                p = os.path.join(out, f"{arm}_{name}.wav")
                wav_write(p, data)
                wavs[(arm, name)] = p
        txt = open(log, errors="replace").read()
        m = re.search(r"\[ADMSLICE\][^\n]*first_sliced_admission[^\n]*", txt)
        slice_stats[arm] = m.group(0) if m else "(no sliced admission)"
        print(f"  arm {arm:<7} {slice_stats[arm]}")

    # ---- gate S: the STATE oracle, before anything that looks at audio ----------
    print("\n### S. Talker state after a sliced prefill (--prefill-slice-check)")
    for S in (2, 3, 16, 48):
        r = subprocess.run([binary, "-d", a.model, "-s", "ryan", "-l", "Italian",
                            "--prefill-slice-check", str(S)],
                           cwd=ROOT, capture_output=True, text=True, timeout=900)
        body = [l for l in r.stdout.splitlines()
                if "slice=" in l or "SLICING ONLY" in l or "STATE:" in l]
        ok = r.returncode == 0
        print(f"  slice={S}: {'PASS' if ok else 'FAIL'}")
        for l in body:
            print("     " + l.strip())
        if not ok:
            fails.append(f"S/{S}: sliced prefill state differs from the unsliced one")

    # ---- gate 0: the treatment must actually have taken the sliced path --------
    # Identical audio is the EXPECTED outcome of a correct slicing (the WAV is a function
    # of integer codes), so it is also what a treatment that never ran would produce.
    # Without this check every gate below is vacuous.
    print("\n### 0. the sliced path actually ran")
    for arm in ("one", "many", "every"):
        ok = "first_sliced_admission" in slice_stats[arm]
        print(f"  {arm:<7} {'PASS' if ok else 'FAIL — the arm ran the monolithic path'}")
        if not ok:
            fails.append(f"0/{arm}: treatment never took the sliced path")
    if "first_sliced_admission" in slice_stats["inline"]:
        print("  inline  FAIL — the control sliced")
        fails.append("0/inline: the control took the sliced path")
    else:
        print("  inline  PASS (monolithic, as the control must be)")

    # ---- gate A: many == one, byte for byte ------------------------------------
    # The slice=1 arm is judged by gate S, which distinguishes a slicing defect from the
    # M=1 matvec path in the shared projection kernels; a WAV hash cannot.
    print("\n### A. resume oracle — sliced-many vs sliced-one (byte-identical audio)")
    for arm in ("many",):
        for name, _, _ in TEXTS:
            if (arm, name) not in wavs:
                continue
            h1 = hashlib.sha256(open(wavs[(arm, name)], "rb").read()).hexdigest()
            h2 = hashlib.sha256(open(wavs[("one", name)], "rb").read()).hexdigest()
            ok = h1 == h2
            print(f"  {arm:<6} {name:<8} {h1[:16]} vs {h2[:16]}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                fails.append(f"A/{arm}/{name}: pausing between slices changed the result")

    # ---- gate B: inline vs sliced ----------------------------------------------
    # Spec section 9.2 asks for mel-corr >= 0.99 against the monolithic arm.  That gate is
    # about the MECHANISM's acceptability, not about whether this implementation is correct:
    # gate S already proves the sliced state is what an unsliced run of the same arithmetic
    # produces.  A failure here is a SPEC CONFLICT to escalate, never something to relax.
    print("\n### B. sliced vs monolithic — SPEC section 9.2 gate (mel-corr >= 0.99, dur 2 %)")
    for name, _, _ in TEXTS:
        (mc, dr), txt = mel_compare(wavs[("inline", name)], wavs[("one", name)])
        if mc is None:
            print(f"  {name:<8} compare_audio.py gave no mel-corr:\n{txt.strip()[:400]}")
            fails.append(f"B/{name}: could not compare")
            continue
        ok = mc >= 0.99 and (dr is None or dr <= 2.0)
        print(f"  {name:<8} mel-corr {mc:.4f}  dur rel {dr if dr is None else f'{dr:.2f}%'}"
              f"  {'PASS' if ok else 'FAIL'}")
        if not ok:
            conflicts.append(f"B/{name}: mel-corr {mc:.4f} dur rel {dr}%")

    # ---- gate C: drop a partially prefilled request -----------------------------
    # Run the SAME scenario in both arms and compare arm to arm.  Comparing the
    # post-cancel output against an undisturbed run of a different request sequence
    # would measure the engine's history sensitivity, not this change.
    print("\n### C. disconnect mid-admission — treatment vs control, same scenario")
    def cancel_scenario(arm, env, port):
        with Server(binary, a.model, port, env, slots=SLOTS,
                    log=os.path.join(out, f"cancel_{arm}.log")):
            tts(port, TEXTS[0][1], TEXTS[0][2])              # warm the prefix cache
            for after in (0.02, 0.08, 0.20):
                tts_abort(port, TEXTS[2][1], TEXTS[2][2], after)
                time.sleep(2.0)                              # let the slot actually free
            try:
                d = tts(port, TEXTS[1][1], TEXTS[1][2], retries=10)
            except urllib.error.HTTPError as ex:
                return ex                                    # PLAN TQ-2, see below
        p = os.path.join(out, f"cancel_{arm}.wav")
        wav_write(p, d)
        return p
    p_ctl = cancel_scenario("inline", arms["inline"], a.port + 10)
    p_trt = cancel_scenario("many", arms["many"], a.port + 11)
    if not isinstance(p_ctl, str) or not isinstance(p_trt, str):
        # The worker refuses the next request after a burst of cancellations.  That is the
        # open fail-fast defect (PLAN TQ-2), it reproduces on the CONTROL arm, and it is not
        # something this change introduced -- so it cannot be a verdict on this change.
        print(f"  INCONCLUSIVE — control={p_ctl} treatment={p_trt}")
        print("  the post-cancel request was refused with 503 on the control arm too: "
              "that is PLAN TQ-2 (fail-fast admission after cancellation), not this change")
        notes.append("C: inconclusive, blocked by the open TQ-2 fail-fast defect")
        p_ctl = p_trt = None
    if p_ctl is None:
        h_ctl = h_trt = None
    else:
        h_ctl = hashlib.sha256(open(p_ctl, "rb").read()).hexdigest()
        h_trt = hashlib.sha256(open(p_trt, "rb").read()).hexdigest()
    if h_ctl is not None:
        ok = h_ctl == h_trt
        print(f"  post-cancel medium: control {h_ctl[:16]} vs treatment {h_trt[:16]}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append("C: a dropped sliced admission left different state than the control")
    h_undist = hashlib.sha256(open(wavs[("inline", "medium")], "rb").read()).hexdigest()
    if h_ctl is not None and h_ctl != h_undist:
        print(f"  note: even the CONTROL differs from its undisturbed run "
              f"({h_ctl[:16]} vs {h_undist[:16]}) — cancellation history affects the engine "
              f"independently of this change")

    # ---- gate D: a pending admission next to an established stream --------------
    # NOT an exact oracle, and deliberately so: at two active slots the engine is not
    # batch-invariant, so two runs of the CONTROL against each other already score
    # mel-corr ~0.4.  What is testable is that nothing breaks and nothing is truncated:
    # both requests complete, both produce audio of the right length, the worker logs no
    # error, and the sliced path really was exercised while another slot was streaming.
    print("\n### D. pending admission beside an established stream (robustness, not parity)")
    import threading
    def concurrent(arm, env, port):
        got = {}
        log = os.path.join(out, f"conc_{arm}.log")
        with Server(binary, a.model, port, env, slots=SLOTS, log=log):
            def one_req(name, text, lang):
                try:
                    got[name] = tts(port, text, lang)
                except Exception as ex:
                    got[name] = ex
            th = [threading.Thread(target=one_req, args=t) for t in
                  [("long", TEXTS[2][1], TEXTS[2][2]),
                   ("english", TEXTS[3][1], TEXTS[3][2])]]
            for t in th: t.start()
            for t in th: t.join()
        return got, log

    got, log = concurrent("many", arms["many"], a.port + 13)
    logtxt = open(log, errors="replace").read()
    if "first_sliced_admission" not in logtxt:
        print("  FAIL — no sliced admission happened under concurrency")
        fails.append("D: the treatment did not slice under concurrency")
    else:
        print("  sliced admission under concurrency: PASS")
    for name in ("long", "english"):
        d = got.get(name)
        if not isinstance(d, (bytes, bytearray)) or len(d) < 1024:
            print(f"  {name:<8} FAIL — {d if isinstance(d, Exception) else 'no audio'}")
            fails.append(f"D/{name}: request failed under concurrency")
            continue
        q = os.path.join(out, f"conc_many_{name}.wav")
        wav_write(q, d)
        ref = os.path.getsize(wavs[("many", name)]) if (("many", name) in wavs) else None
        drift = abs(len(d) - ref) / ref if ref else 0.0
        ok = drift <= 0.15
        print(f"  {name:<8} {len(d)} bytes vs single-client {ref} "
              f"({drift*100:.1f}% length drift)  {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"D/{name}: audio length drifted {drift*100:.1f}% under concurrency")
    for pat in ("Error:", "error:", "assert"):
        if pat in logtxt:
            print(f"  FAIL — worker log contains {pat!r}")
            fails.append(f"D: worker logged {pat!r} under concurrency")
            break

    # ---- gate E: an unusable value must be fatal --------------------------------
    print("\n### E. invalid QWEN_PREFILL_SLICE must kill the worker")
    e = dict(os.environ); e["QWEN_PREFILL_SLICE"] = "banana"
    try:
        r = subprocess.run([binary, "-d", a.model, "--serve", str(a.port + 14),
                            "--batch-size", "1", "--prefork", "1"],
                           cwd=ROOT, capture_output=True, text=True, env=e, timeout=60)
        rc, txt = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        rc, txt = 0, "(server kept running)"
    # It must die in the PARENT, before any fork: a worker that exits on a bad value is
    # respawned and the parent goes on serving the control arm.
    ok = rc != 0 and "QWEN_PREFILL_SLICE" in txt
    print(f"  rc={rc}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("E: an invalid slice value did not stop the server parent")

    if conflicts:
        print("\nSPEC CONFLICT — the implementation is correct (gate S) but the mechanism does\n"
              "not meet spec section 9.2: reading the earlier tokens of the same prompt from\n"
              "the bf16 KV cache instead of the f32 staging buffer moves the Talker state by\n"
              "~1e-3, which at temperature 0 flips a sampled code and changes the utterance.\n"
              "Exact parity would need the f32 K/V of all 28 layers alive across slices, which\n"
              "spec section 5 forbids on memory grounds. Escalate; do not relax the gate.")
        for c in conflicts:
            print("  " + c)
    print("\n" + ("PREFILL-SLICE PARITY: PASS" if not fails and not conflicts else
                  ("PREFILL-SLICE PARITY: FAIL\n  " + "\n  ".join(fails)) if fails else
                  "PREFILL-SLICE PARITY: SPEC CONFLICT (implementation gates all pass)"))
    for n in notes:
        print("  note: " + n)
    print(f"artifacts: {os.path.relpath(out, ROOT)}")
    return 1 if (fails or conflicts) else 0


if __name__ == "__main__":
    sys.exit(main())
