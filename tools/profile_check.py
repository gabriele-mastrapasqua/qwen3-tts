#!/usr/bin/env python3
"""profile_check.py — provenance fingerprint of a build, and "is the last profile still valid?".

    tools/profile_check.py --fingerprint --bin ./qwen_tts [--model DIR] [--dispatch dispatch.json]
        prints a JSON fingerprint: binary sha256, build tag the binary reports, git commit,
        dirty flag, a hash of the dirty diff + untracked sources, compiler, host identity,
        the QWEN_*/OPENBLAS_*/OMP_* environment, optional model fingerprint.

    tools/profile_check.py --profiles profiles [--bin ./qwen_tts] [--model DIR]
        compares profiles/LATEST/manifest.json against the tree, binary, env and host as they
        are NOW, re-runs --dispatch-map and diffs every resolved value.  One ERROR line per
        mismatch, exit 1.  This is the first command of an optimisation session and the
        thing that protects a long session from compaction/amnesia: the profile you are
        reasoning from must describe the binary you are about to change.

The fingerprint is computed by ONE implementation (this file) for both directions, so the
comparison can never drift from what was recorded.
"""
import argparse, hashlib, json, os, platform, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_GLOBS = ("*.c", "*.h", "Makefile")
ENV_PREFIXES = ("QWEN_", "OPENBLAS_", "OMP_", "GOMP_", "KMP_", "MKL_")


def sh(cmd, default="", timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else default
    except Exception:
        return default


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    """One implementation of the identity: tools/source_fingerprint.sh (git where it exists,
    the shipped .source_fingerprint otherwise).  Never QWEN_SOURCE_COMMIT."""
    fp = sh("bash tools/source_fingerprint.sh", default="unknown")
    commit = fp.split(":", 1)[0]
    if not sh("git rev-parse --short HEAD", default=""):
        return {"source_commit": commit, "dirty": "yes" if "-dirty" in commit else ("no" if commit != "unknown" else "unknown"),
                "source_fingerprint": fp, "fingerprint_origin": "shipped .source_fingerprint (no git here)"}
    changed = sh("git status --porcelain --untracked-files=all", default="")
    dirty = "yes" if changed else "no"
    h = hashlib.sha256()
    h.update(sh("git diff HEAD", default="").encode())
    for line in sorted(changed.splitlines()):
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        full = os.path.join(ROOT, path)
        if os.path.isfile(full) and any(path.endswith(g.lstrip("*")) for g in SOURCE_GLOBS):
            h.update(path.encode()); h.update(sha256_file(full).encode())
    return {"source_commit": commit + ("-dirty" if changed else ""), "dirty": dirty,
            "source_fingerprint": fp, "fingerprint_origin": "git",
            "dirty_detail_sha256": h.hexdigest()}


def binary_state(bin_path):
    if not os.path.isfile(bin_path):
        return {"binary": bin_path, "binary_sha256": "MISSING", "binary_build_tag": "MISSING",
                "binary_simd": ""}
    caps = sh(f"'{bin_path}' --caps", default="")
    m = re.search(r"^\s*build:\s*(\S+)\s*·\s*SIMD=(\S+)", caps, re.M)
    fp = re.search(r"src=(\S+)", caps)
    return {"binary": bin_path, "binary_sha256": sha256_file(bin_path),
            "binary_build_tag": m.group(1) if m else "UNKNOWN",
            "binary_simd": m.group(2) if m else "UNKNOWN",
            "binary_source_fp": fp.group(1) if fp else "",
            "binary_mtime": os.path.getmtime(bin_path)}


def host_state():
    if sys.platform == "darwin":
        cpu = sh("sysctl -n machdep.cpu.brand_string", default=platform.processor())
        ncpu = sh("sysctl -n hw.ncpu", default="?")
    else:
        cpu = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2", default=platform.processor()).strip()
        ncpu = sh("nproc", default="?")
    return {"hostname": platform.node(), "os": platform.system() + " " + platform.release(),
            "arch": platform.machine(), "cpu_model": cpu, "logical_cpus": ncpu}


def env_state():
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}


def compiler_state():
    cc = os.environ.get("CC") or sh("make -s -f Makefile info 2>/dev/null | sed -n 's/^CC[ :=]*//p' | head -1", default="cc")
    return {"cc": cc, "cc_version": sh(f"{cc} --version 2>/dev/null | head -1", default="?")}


def model_state(model_dir):
    """Cheap and stable: config + the list of weight files with their sizes.  Hashing
    several GB of weights on every check is what would make people skip the check."""
    if not model_dir:
        return {}
    if not os.path.isdir(model_dir):
        return {"model": model_dir, "model_fingerprint": "MISSING"}
    h = hashlib.sha256()
    cfg = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg):
        h.update(open(cfg, "rb").read())
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            h.update(f"{name}:{os.path.getsize(p)}".encode())
    return {"model": model_dir, "model_fingerprint": h.hexdigest()}


def dispatch_state(bin_path, extra_env=None):
    if not os.path.isfile(bin_path):
        return {}
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    env = dict(os.environ); env["QWEN_DISPATCH_JSON"] = path
    if extra_env:
        env.update(extra_env)
    try:
        subprocess.run([bin_path, "--dispatch-map"], capture_output=True, text=True, timeout=120, env=env)
        return json.load(open(path))
    except Exception as e:
        return {"error": str(e)}
    finally:
        try: os.unlink(path)
        except OSError: pass


def resolved_map(doc):
    out = {}
    for f in doc.get("features", []):
        out[f["id"]] = f.get("resolved")
    for g in doc.get("gates", []):
        out[g["id"]] = "ON" if g.get("on") else "OFF"
    return out


def fingerprint(bin_path, model_dir=None, with_dispatch=True):
    fp = {"v": 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "epoch": time.time()}
    fp.update(git_state()); fp.update(binary_state(bin_path)); fp.update(host_state())
    fp.update(compiler_state()); fp["env"] = env_state(); fp.update(model_state(model_dir))
    if with_dispatch:
        d = dispatch_state(bin_path)
        fp["isa_class"] = d.get("isa_class", "?")
        fp["dispatch_resolved"] = resolved_map(d)
    return fp


def newest_source_mtime():
    newest, name = 0.0, ""
    for fn in os.listdir(ROOT):
        if fn.endswith((".c", ".h")) or fn == "Makefile":
            m = os.path.getmtime(os.path.join(ROOT, fn))
            if m > newest:
                newest, name = m, fn
    return newest, name


def check(profiles_dir, bin_path, model_dir):
    latest = os.path.join(profiles_dir, "LATEST")
    man = os.path.join(latest, "manifest.json")
    if not os.path.isfile(man):
        print(f"ERROR: no profile found ({man}); run `make cpu-check` first")
        return 1
    old = json.load(open(man))
    now = fingerprint(bin_path, model_dir or old.get("model"))
    errors = []

    def cmp(key, label):
        if old.get(key) != now.get(key):
            errors.append(f"{label}: profile={old.get(key)} current={now.get(key)}")

    cmp("binary_sha256", "binary SHA256 differs (rebuilt since the profile)")
    cmp("binary_source_fp", "fingerprint embedded in the binary differs")
    if now.get("binary_source_fp") and now.get("source_fingerprint") not in (None, "unknown") \
            and now["binary_source_fp"] != now["source_fingerprint"]:
        errors.append(f"binary was built from another tree: binary src={now['binary_source_fp']} tree={now['source_fingerprint']}")
    cmp("source_commit", "source commit differs")
    cmp("source_fingerprint", "dirty-source fingerprint differs (uncommitted edits changed)")
    cmp("cpu_model", "host CPU differs")
    cmp("logical_cpus", "host logical CPU count differs")
    cmp("binary_simd", "SIMD profile of the binary differs")
    if old.get("model_fingerprint") and now.get("model_fingerprint") \
            and old["model_fingerprint"] != now["model_fingerprint"]:
        errors.append(f"model fingerprint differs: profile={old.get('model')} current={now.get('model')}")

    oe, ne = old.get("env", {}), now.get("env", {})
    for k in sorted(set(oe) | set(ne)):
        if oe.get(k) != ne.get(k):
            errors.append(f"env differs: {k}: profile={oe.get(k, '<unset>')} current={ne.get(k, '<unset>')}")

    od, nd = old.get("dispatch_resolved", {}), now.get("dispatch_resolved", {})
    if not od:
        errors.append("no runtime dispatch map recorded in the profile (older profile format?)")
    for k in sorted(set(od) | set(nd)):
        if od.get(k) != nd.get(k):
            errors.append(f"resolved dispatch differs: {k}: profile={od.get(k)} current={nd.get(k)}")

    m, fn = newest_source_mtime()
    if old.get("epoch") and m > float(old["epoch"]):
        errors.append(f"last profile was produced before {fn} changed "
                      f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(m))})")

    real = os.path.realpath(latest)
    print(f"profile: {real}  ({old.get('utc', '?')})")
    if errors:
        for e in errors:
            print("ERROR:", e)
        print(f"PROFILE VALID: NO  ({len(errors)} mismatch{'es' if len(errors) != 1 else ''}) "
              f"-> run `make cpu-check` again before trusting any number from that profile")
        return 1
    print("PROFILE VALID: YES  (binary, source, host, env, dispatch map and model all match)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint", action="store_true")
    ap.add_argument("--profiles", default=os.path.join(ROOT, "profiles"))
    ap.add_argument("--bin", default=os.path.join(ROOT, "qwen_tts"))
    ap.add_argument("--model", default="")
    ap.add_argument("--no-dispatch", action="store_true")
    a = ap.parse_args()
    if a.fingerprint:
        print(json.dumps(fingerprint(a.bin, a.model, not a.no_dispatch), indent=1))
        return 0
    return check(a.profiles, a.bin, a.model)


if __name__ == "__main__":
    sys.exit(main())
