"""Portable entry points and model adapters for resident LOW/1 NPU training."""

from pathlib import Path
import sys

# A copied kit supplies its own runtime. Frozen runs instead have jade_lotus
# beside this package, so they need no mutable source-repository import path.
_vendor = Path(__file__).resolve().parent / "vendor"
if (_vendor / "jade_lotus").is_dir() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))

__version__ = "0.1.0"
