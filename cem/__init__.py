"""
Clocked Event Memory (CEM): a diary for the Continuous Thought Machine.

The upstream CTM code (SakanaAI/continuous-thought-machines, which setup.sh fetches at the pinned commit) is not an
installable package, so importing cem puts its folder on the import path.
"""
import os
import sys

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_FOLDER = os.path.join(PROJECT_FOLDER, "third_party", "continuous-thought-machines")

if not os.path.isdir(UPSTREAM_FOLDER):
    raise ImportError(f"the upstream CTM code is missing at {UPSTREAM_FOLDER}; run ./setup.sh first")
if UPSTREAM_FOLDER not in sys.path:
    sys.path.insert(0, UPSTREAM_FOLDER)
