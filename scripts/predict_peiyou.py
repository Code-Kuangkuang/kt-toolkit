"""Backward-compatible entry point for the AAAI2023 prediction script."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.predict_aaai2023 import app, main


if __name__ == "__main__":
    app()
