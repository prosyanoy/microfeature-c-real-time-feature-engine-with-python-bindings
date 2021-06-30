import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("MF_MODULE_DIR", str(ROOT / "build" / "python")))
sys.path.insert(0, str(ROOT / "benchmarks"))
