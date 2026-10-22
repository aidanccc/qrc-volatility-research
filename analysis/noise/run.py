"""Create a fresh noise-study run: compute, render, test. Existing runs are preserved."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists():
        raise SystemExit('Choose a new output directory; existing runs are preserved.')
    out.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT) + os.pathsep + str(HERE), MPLCONFIGDIR='/tmp/qrc-noise-mpl')
    stages = [('compute', [str(HERE / 'compute.py'), '--output', str(out)]),
              ('render', [str(HERE / 'render.py'), '--output', str(out)]),
              ('noise-tests', ['-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-v'])]
    for name, command in stages:
        print(f'{name}: running; log at {out / (name + ".log")}', flush=True)
        with (out / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Complete: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
