#!/usr/bin/env python3
"""
Simple unit test runner for the py-fitter `process.py` script.
Runs:
  python process.py --file test.txt --loc tape --fittype 2D
and collects output (CSV, PNG, PKL, NPZ) into `unit_test/`.

Usage: run this script from anywhere; it will execute process.py in its
own directory and collect outputs.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_FILE = "unit_test/test.txt"
UNIT_DIR = HERE / "unit_test"
LOG_FILE = UNIT_DIR / "process_output.log"

def main():
    UNIT_DIR.mkdir(exist_ok=True)

    file_path = HERE / TEST_FILE
    if not file_path.exists():
        print(f"Test file not found: {file_path}")
        sys.exit(2)

    # Run process.py from the py-fitter directory (use local files)
    cmd = [sys.executable, "process.py", "--file", str(file_path), "--loc", "tape", "--fittype", "2D"]
    print("Running:", " ".join(cmd))

    # Capture stdout/stderr
    with open(LOG_FILE, "wb") as lf:
        proc = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"process.py exited with code {proc.returncode}. See {LOG_FILE} for details.")
    else:
        print(f"process.py finished. See {LOG_FILE} for details.")

    # Derive csv basename same way as process.py: use filename base
    csv_base = os.path.splitext(file_path.name)[0]

    # Collect outputs that start with csv_base
    moved = []
    for p in HERE.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith(csv_base) and p.suffix.lower() in ('.csv', '.png', '.pkl', '.npz'):
            dest = UNIT_DIR / p.name
            try:
                shutil.move(str(p), str(dest))
                moved.append(dest.name)
            except Exception as e:
                print(f"Failed to move {p} -> {dest}: {e}")

    # Additionally, move any debug pngs or offspill outputs
    for p in HERE.glob('*_mom_mag_debug.png'):
        try:
            shutil.move(str(p), str(UNIT_DIR / p.name))
            moved.append(p.name)
        except Exception:
            pass

    print("Moved files into unit_test/:")
    for m in moved:
        print(" -", m)

    print("Unit test complete.")

if __name__ == '__main__':
    main()
