"""Pytest bootstrap for the backend suite.

Some tests import the repo-root ``bench`` package. When pytest runs from
``backend/`` (as CI does), the repo root is not on ``sys.path``, so those
imports fail at collection time. Insert it here so the suite runs
identically from any working directory.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
