#!/usr/bin/env python3
"""DEPRECATED SHIM — renamed to `tests/serve_parallel_wave.py`."""
import os, runpy, sys

print("⚠️  tests/kleidi_parallel_capacity.py is DEPRECATED — use tests/serve_parallel_wave.py",
      file=sys.stderr)
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_parallel_wave.py")
runpy.run_path(sys.argv[0], run_name="__main__")
