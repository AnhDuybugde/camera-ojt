"""Compatibility entry point for the shared Imou runtime."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camera_tracking.application import workstate

if __name__ == "__main__":
    workstate.main("imou")
else:
    sys.modules[__name__] = workstate
