#!/usr/bin/env python3
"""Cross-check ingot against an independent parse of the same file.

ingot's own tests build their fixtures with ingot's own understanding of the
format, so a shared misreading of the spec would pass unnoticed. This script
parses the safetensors header with nothing but struct+json and compares every
tensor's dtype, shape and byte count against what ingot-dump reports.

Usage: python3 tools/check_against_python.py <file.safetensors> [...]
Exit 0 when they agree.
"""
import json, struct, subprocess, sys, re, os
DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'ingot-dump')
path = sys.argv[1]
with open(path,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
ref = {}
for k,v in hdr.items():
    if k == '__metadata__': continue
    ref[k] = (v['dtype'], tuple(v['shape']), v['data_offsets'][1]-v['data_offsets'][0])
out = subprocess.run([DUMP,'-v',path], capture_output=True, text=True).stdout
got = {}
for line in out.splitlines():
    m = re.match(r'^  (\S+)\s+(\S+)\s+\[([^\]]*)\]\s+(\d+) B', line)
    if m:
        shape = tuple(int(x) for x in m.group(3).split(', ')) if m.group(3) else ()
        got[m.group(1)] = (m.group(2), shape, int(m.group(4)))
missing = set(ref) - set(got); extra = set(got) - set(ref)
diff = {k for k in ref.keys() & got.keys() if ref[k] != got[k]}
print(f"{path}: python-oracle {len(ref)} tensors, ingot {len(got)}; "
      f"missing={len(missing)} extra={len(extra)} mismatched={len(diff)}")
for k in list(diff)[:3]: print("   ", k, ref[k], "vs", got[k])
sys.exit(1 if (missing or extra or diff) else 0)
