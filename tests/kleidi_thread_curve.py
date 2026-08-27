#!/usr/bin/env python3
"""DEPRECATED SHIM — renamed to `tests/serve_thread_curve.py`.

The old name carried the backend that happened to be under test when the script was
written. What it measures is serving behaviour, which is backend-agnostic: the same tool
has to run against KleidiAI on ARM, VNNI or AMX on x86, or a plain scalar build. A name
that says the backend makes that reuse look wrong.

This shim forwards, so existing call sites and older notes keep working. It will be
removed once nothing references it.
"""
import os, runpy, sys

print("⚠️  tests/kleidi_thread_curve.py is DEPRECATED — use tests/serve_thread_curve.py",
      file=sys.stderr)
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_thread_curve.py")
runpy.run_path(sys.argv[0], run_name="__main__")
