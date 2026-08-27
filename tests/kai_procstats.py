"""DEPRECATED SHIM — renamed to `tests/serve_procstats.py`.

`kai_` named the KleidiAI campaign this was written during; the module reads /proc and
knows nothing about any backend. Importing this name still works and warns once.
"""
import warnings
warnings.warn("tests/kai_procstats.py is deprecated — import tests/serve_procstats.py",
              DeprecationWarning, stacklevel=2)
from serve_procstats import *          # noqa: F401,F403
from serve_procstats import (_read, proc_sample, worker_pids_from_log,   # noqa: F401
                             parse_prefork_stats, per_worker_rows, format_rows)
