#!/usr/bin/env python3
"""DEPRECATED SHIM — renamed to `tests/serve_concurrency_matrix.py`."""
import os, runpy, sys

print("⚠️  tests/kleidi_server_matrix.py is DEPRECATED — use tests/serve_concurrency_matrix.py",
      file=sys.stderr)
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_concurrency_matrix.py")
runpy.run_path(sys.argv[0], run_name="__main__")
