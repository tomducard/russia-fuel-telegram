#!/usr/bin/env python3
"""
Robust entry point for RFT CLI.
Bypasses installation issues by explicitly adding src/ to sys.path.
"""
import sys
import os
from pathlib import Path

# Add src folder to python path ensuring rft is importable
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from rft.cli import main
except ImportError as e:
    print(f"Error importing rft: {e}")
    print(f"Checked src path: {src_dir}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

if __name__ == "__main__":
    main()
