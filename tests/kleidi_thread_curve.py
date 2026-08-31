#!/usr/bin/env python3
"""DEPRECATED SHIM — renamed to `tests/serve_thread_curve.py`."""
import os, runpy, sys

print("⚠️  tests/kleidi_thread_curve.py is DEPRECATED — use tests/serve_thread_curve.py",
      file=sys.stderr)
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_thread_curve.py")
runpy.run_path(sys.argv[0], run_name="__main__")
