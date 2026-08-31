#!/usr/bin/env python3
"""Load, validate and resolve a server performance profile.

A profile answers "given this hardware and this objective, how should the engine be run?".
This turns it into the exact argv and environment, so the recommended values live in ONE
place instead of being restated in JSON, in shell scripts and in prose - which is how three
copies start disagreeing.

    tools/perf_profile.py validate                       # every profile in configs/perf
    tools/perf_profile.py show      recommended
    tools/perf_profile.py command   axion-16c-ttfa --model DIR --port 8000
    tools/perf_profile.py server-env axion-16c-ttfa      # comma form for the harness
    tools/perf_profile.py check-flags axion-16c-ttfa --log server.log

The last one closes the loop: the engine prints one [FLAGS] line naming every registered
variable it actually read, and this compares it against what the profile asked for. A flag
is on when the process says so, never when the invocation intended it.
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF = os.path.join(ROOT, "configs", "perf")


# ── a small validator for exactly the constructs schema.json uses ────────────────────
class Bad(Exception):
    pass


def _resolve(schema, root):
    while "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise Bad(f"unsupported $ref {ref}")
        node = root
        for part in ref[2:].split("/"):
            node = node[part]
        schema = node
    return schema


def check(node, schema, root, path="$"):
    schema = _resolve(schema, root)
    if "const" in schema:
        if node != schema["const"]:
            raise Bad(f"{path}: expected {schema['const']!r}, got {node!r}")
        return
    if "enum" in schema:
        if node not in schema["enum"]:
            raise Bad(f"{path}: {node!r} is not one of {schema['enum']}")
        return
    if "oneOf" in schema:
        for alt in schema["oneOf"]:
            try:
                check(node, alt, root, path)
                return
            except Bad:
                pass
        raise Bad(f"{path}: {node!r} matches none of the allowed forms")

    t = schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if types:
        ok = any(
            (ty == "object" and isinstance(node, dict))
            or (ty == "array" and isinstance(node, list))
            or (ty == "string" and isinstance(node, str))
            or (ty == "boolean" and isinstance(node, bool))
            or (ty == "integer" and isinstance(node, int) and not isinstance(node, bool))
            or (ty == "number" and isinstance(node, (int, float)) and not isinstance(node, bool))
            or (ty == "null" and node is None)
            for ty in types
        )
        if not ok:
            raise Bad(f"{path}: expected {'/'.join(types)}, got {type(node).__name__}")

    if isinstance(node, str) and "pattern" in schema:
        if not re.match(schema["pattern"], node):
            raise Bad(f"{path}: {node!r} does not match {schema['pattern']}")
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        if "minimum" in schema and node < schema["minimum"]:
            raise Bad(f"{path}: {node} < minimum {schema['minimum']}")

    if isinstance(node, dict):
        for req in schema.get("required", []):
            if req not in node:
                raise Bad(f"{path}: missing required field {req!r}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for k, v in node.items():
            if k in props:
                check(v, props[k], root, f"{path}.{k}")
            elif isinstance(extra, dict):
                check(v, extra, root, f"{path}.{k}")
            elif extra is False:
                raise Bad(f"{path}: unknown field {k!r}")
    if isinstance(node, list):
        if "minItems" in schema and len(node) < schema["minItems"]:
            raise Bad(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(node) > schema["maxItems"]:
            raise Bad(f"{path}: allows at most {schema['maxItems']} items")
        for i, v in enumerate(node):
            if "items" in schema:
                check(v, schema["items"], root, f"{path}[{i}]")


# ── profile loading ──────────────────────────────────────────────────────────────────
def load_raw(name):
    p = name if os.path.sep in name else os.path.join(PERF, f"{name}.json")
    if not p.endswith(".json"):
        p += ".json"
    if not os.path.exists(p):
        raise Bad(f"no such profile: {p}")
    with open(p) as f:
        return json.load(f), p


SECTIONS = ("hardware", "build", "server", "streaming", "runtime", "launch", "qualification")


def shape_check(raw, path):
    """A profile is either COMPLETE or an ALIAS. Nothing in between: a half-alias that
    carries a couple of its own values is the drift this format exists to prevent."""
    alias = raw.get("profile", {}).get("alias_of")
    present = [s for s in SECTIONS if s in raw]
    if alias:
        if present:
            raise Bad(f"{path}: alias_of is set, so {present} must be absent — an alias "
                      f"that repeats values is a copy that will drift")
    else:
        missing = [s for s in SECTIONS if s not in raw]
        if missing:
            raise Bad(f"{path}: not an alias, so these sections are required: {missing}")


def load(name, _seen=None):
    """Resolve aliases. An alias carries a pointer and nothing else."""
    _seen = _seen or []
    prof, path = load_raw(name)
    pid = prof.get("profile", {}).get("id", name)
    if pid in _seen:
        raise Bad(f"alias cycle: {' -> '.join(_seen + [pid])}")
    shape_check(prof, path)
    alias = prof.get("profile", {}).get("alias_of")
    if alias:
        return load(alias, _seen + [pid])
    return prof, path


# ── semantic checks the schema cannot express ────────────────────────────────────────
def semantic(prof, path, engine=None):
    errs = []
    hw, sv = prof["hardware"], prof["server"]
    w, t = sv["prefork_workers"], sv["threads_per_worker"]
    if isinstance(w, int) and isinstance(t, int):
        if w * t > hw["physical_cores"] and not prof.get("allow_oversubscription"):
            errs.append(f"{w} workers x {t} threads = {w*t} exceeds "
                        f"{hw['physical_cores']} physical cores")
    env = prof["runtime"]["environment"]
    for k, spec in env.items():
        if not re.match(r"^[A-Z][A-Z0-9_]*$", k):
            errs.append(f"environment key {k!r} is not a shell variable name")
        v = spec["value"]
        # The streaming block holds integers and they are turned into environment values by
        # environ(); here the schema has already established that a declared value is a
        # string or null, so a non-string is a schema failure and not this check's business.
        if v is not None and not isinstance(v, str):
            continue
        if v is not None and (v != v.strip() or " " in v or "\t" in v):
            errs.append(f"{k}={v!r} has whitespace; --server-env splits on COMMAS, and a "
                        f"space folds every later variable into this one's value")
    # a value fixed both on the command line and in the environment is two sources of truth
    cli = {a.split("=", 1)[0] for a in prof["launch"].get("extra_arguments", [])}
    for dup in cli & {"--prefork", "--prefork-threads", "--batch-size"}:
        errs.append(f"{dup} is set in extra_arguments and also in the server section")
    if engine:
        known = engine_known_flags(engine)
        if known:
            for k, spec in env.items():
                if spec["value"] is not None and k.startswith("QWEN_") and k not in known:
                    errs.append(f"{k} is not a variable this engine reports; it would be "
                                f"set but never declared, so no run could verify it")
    return errs


def engine_known_flags(binary):
    """The variables the engine declares in its [FLAGS] line, read from the source register."""
    src = os.path.join(ROOT, "qwen_tts_kernels.c")
    if not os.path.exists(src):
        return None
    with open(src) as f:
        txt = f.read()
    m = re.search(r"g_qwen_reported_flags\[\]\s*=\s*\{(.*?)\};", txt, re.S)
    return set(re.findall(r'"([A-Z0-9_]+)"', m.group(1))) if m else None


# ── resolution ───────────────────────────────────────────────────────────────────────
def argv(prof, model, port):
    sv = prof["server"]
    a = [prof["launch"]["executable"], "-d", model]
    if "int8" in prof["runtime"]["precision"].get("talker_weights", ""):
        a.append("--int8")
    a += ["--serve", str(port), "--batch-size", str(sv["batch_size"]),
          "--prefork", str(sv["prefork_workers"]),
          "--prefork-threads", str(sv["threads_per_worker"])]
    for key, flag in (("max_queue", "--max-queue"),
                      ("queue_timeout_ms", "--queue-timeout-ms"),
                      ("max_request_seconds", "--max-request-seconds")):
        v = sv.get(key)
        if isinstance(v, int):
            a += [flag, str(v)]
    return a + prof["launch"].get("extra_arguments", [])


# Fields that are declared elsewhere in the profile but reach the engine as environment
# variables. Without this mapping a value could sit in the file, be read by a human, and
# never be applied -- which is the same failure as forgetting the value entirely, except
# harder to notice because the file says the right thing.
STREAMING_ENV = {"decode_chunk": "QWEN_STREAM_DECODE_CHUNK",
                 "decode_chunk_busy": "QWEN_STREAM_DECODE_CHUNK_BUSY"}


def environ(prof):
    env = {k: s["value"] for k, s in prof["runtime"]["environment"].items()
           if s["value"] is not None}
    for key, var in STREAMING_ENV.items():
        v = prof.get("streaming", {}).get(key)
        if isinstance(v, int):
            # str(): the engine reports its flags as text, and comparing 8 against "8" made
            # check-flags disagree with a process that was configured exactly as asked.
            env.setdefault(var, str(v))  # an explicit runtime.environment entry still wins
    return env


def server_env(prof):
    return ",".join(f"{k}={v}" for k, v in sorted(environ(prof).items()))


def forbidden_env(prof):
    """Variables the profile declares with a null value: they must be ABSENT, not merely unset
    by us. OPENBLAS_NUM_THREADS is the case that matters -- the engine sizes OpenBLAS per
    worker with openblas_set_num_threads(), and backs off entirely when that variable is in
    the environment. Someone else's export therefore silently replaces the topology the
    profile qualified, and no table would show it."""
    return sorted(k for k, s in prof["runtime"]["environment"].items() if s["value"] is None)


# ── commands ─────────────────────────────────────────────────────────────────────────
def cmd_validate(a):
    files = sorted(f for f in os.listdir(PERF) if f.endswith(".json") and f != "schema.json")
    with open(os.path.join(PERF, "schema.json")) as f:
        schema = json.load(f)
    bad = 0
    for fn in files:
        name = fn[:-5]
        try:
            raw, path = load_raw(name)
            check(raw, schema, schema, "$")
            prof, _ = load(name)
            errs = semantic(prof, path, a.engine)
            if errs:
                bad += 1
                print(f"FAIL {fn}")
                for e in errs:
                    print(f"     {e}")
            else:
                tag = " (alias)" if raw["profile"].get("alias_of") else ""
                print(f"ok   {fn}{tag}")
        except Bad as e:
            bad += 1
            print(f"FAIL {fn}\n     {e}")
    print(f"\n{len(files)-bad}/{len(files)} profiles valid")
    return 1 if bad else 0


def cmd_show(a):
    prof, path = load(a.name)
    print(json.dumps(prof, indent=2))
    return 0


def cmd_command(a):
    prof, _ = load(a.name)
    env = environ(prof)
    line = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
    print((line + " " if line else "") + " ".join(argv(prof, a.model, a.port)))
    return 0


def cmd_server_env(a):
    print(server_env(load(a.name)[0]))
    return 0


def cmd_new(a):
    """Emit a skeleton for a new deployment, with everything unmeasured marked as such.

    Copying an existing profile is how a value from another machine becomes a claim about
    this one: the fields are all filled in, they all look deliberate, and nothing says which
    of them anybody actually measured. A skeleton starts from "unspecified" and makes filling
    a field a decision.
    """
    ref, _ = load(a.like)
    out = json.loads(json.dumps(ref))          # deep copy, then blank what is not portable
    out["profile"]["id"] = a.name
    out["profile"]["description"] = ("FILL IN: what this deployment is, and note that every value "
                                     "here must come from a measurement on it.")
    hw = out["hardware"]
    for k in ("cpu_family", "notes"):
        hw[k] = "unspecified"
    # The counts keep the reference's shape rather than becoming zero: a skeleton that fails
    # `validate` cannot sit in the tree while someone fills it in, and the honesty lives in
    # qualification.status rather than in an invalid number. The description says to replace
    # them, and the sweep is what replaces them.
    hw["notes"] = ("PLACEHOLDER counts, copied from the reference profile for shape only. Replace "
                   "them with what this machine reports before anything here is quoted.")
    out["qualification"] = {
        "status": "unqualified",
        "workload": "NONE YET — nothing here has been measured on this machine.",
        "benchmark_family": "TRUE_SIMULTANEOUS_WAVE",
        "model": "unspecified", "speaker": "unspecified",
        "notes": ("Run the break-in sweep in docs/serving-operations.md, then replace the topology, "
                  "the thread split and the environment with what it measured, and set status to "
                  "'qualified' with the numbers that earned it. Until then this file is a starting "
                  "point and says so."),
    }
    dest = os.path.join(PERF, a.name + ".json")
    if os.path.exists(dest) and not a.force:
        print(f"{dest} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    with open(dest, "w") as f:
        f.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {dest}")
    print("next: edit it, then `tools/perf_profile.py validate`")
    return 0


def cmd_forbidden_env(a):
    """Print the variables that must not be present in the environment of a qualifying run."""
    for k in forbidden_env(load(a.name)[0]):
        print(k)
    return 0


def cmd_check_flags(a):
    prof, _ = load(a.name)
    want = environ(prof)
    seen, found = {}, False
    with open(a.log, errors="replace") as f:
        for line in f:
            if line.startswith("[FLAGS]"):
                found = True
                seen = {}
                for tok in line.split()[2:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        seen[k] = v
    if not found:
        print(f"FAIL: no [FLAGS] line in {a.log} — the engine never declared, so nothing "
              f"about this run's configuration can be verified")
        return 1
    missing = {k: v for k, v in want.items() if k.startswith("QWEN_") and seen.get(k) != v}
    if missing:
        print(f"FAIL: engine reported {seen}, profile asked for {missing}")
        return 1
    print(f"ok: engine declares {' '.join(f'{k}={v}' for k, v in sorted(seen.items()))}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--engine", default=None); v.set_defaults(fn=cmd_validate)
    s = sub.add_parser("show"); s.add_argument("name"); s.set_defaults(fn=cmd_show)
    c = sub.add_parser("command")
    c.add_argument("name"); c.add_argument("--model", required=True); c.add_argument("--port", default=8000)
    c.set_defaults(fn=cmd_command)
    e = sub.add_parser("server-env"); e.add_argument("name"); e.set_defaults(fn=cmd_server_env)
    nw = sub.add_parser("new"); nw.add_argument("name")
    nw.add_argument("--like", default="axion-16c-ttfa",
                    help="profile to take the STRUCTURE from; every measured value is blanked")
    nw.add_argument("--force", action="store_true"); nw.set_defaults(fn=cmd_new)
    fb = sub.add_parser("forbidden-env"); fb.add_argument("name")
    fb.set_defaults(fn=cmd_forbidden_env)
    k = sub.add_parser("check-flags")
    k.add_argument("name"); k.add_argument("--log", required=True); k.set_defaults(fn=cmd_check_flags)
    a = ap.parse_args()
    try:
        return a.fn(a)
    except Bad as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
