"""Fresh noise run: compute, then render. Usage: python -m qrc.noise.run --output results/NEW-DIR"""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists():
        raise SystemExit('Choose a new output directory.')
    out.mkdir(parents=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    for stage in ['compute', 'render']:
        print(f'{stage}: running; log at {out / (stage + ".log")}', flush=True)
        with (out / (stage + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', f'qrc.noise.{stage}', '--output', str(out)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Complete: {out}', flush=True)


if __name__ == '__main__':
    main()
