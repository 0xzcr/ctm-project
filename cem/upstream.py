"""
Makes the pinned upstream CTM code (SakanaAI/continuous-thought-machines) importable.

The upstream repository is not an installable package, so its folder is put on sys.path. Every upstream
import in this project goes through this file. setup.sh fetches the code at the pinned commit.
"""
import os
import sys

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_FOLDER = os.path.join(PROJECT_FOLDER, "third_party", "continuous-thought-machines")

if not os.path.isdir(UPSTREAM_FOLDER):
    raise ImportError(f"the upstream CTM code is missing at {UPSTREAM_FOLDER}; run ./setup.sh first")

if UPSTREAM_FOLDER not in sys.path:
    sys.path.insert(0, UPSTREAM_FOLDER)
