"""Make the main choicebench repo's own ``tests`` package importable, so
this analysis package's tests can reuse its existing synthetic Stage-1
freeze fixture builders instead of duplicating them."""

from __future__ import annotations

import sys
from pathlib import Path

_CHOICEBENCH_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_CHOICEBENCH_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHOICEBENCH_REPO_ROOT))
