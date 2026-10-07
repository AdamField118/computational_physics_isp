"""Compatibility entry point for `python setup.py build_ext --inplace`."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    if sys.argv[1:] != ["build_ext", "--inplace"]:
        raise SystemExit("Use: python setup.py build_ext --inplace (or comphys-build nbody --language fortran)")
    root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, str(root / "scripts/build.py"),
                    "nbody", "--language", "fortran"], check=True)
