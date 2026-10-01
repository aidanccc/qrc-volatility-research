"""Create a fresh diagnostic run without modifying model or published artifacts."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compute import ROOT, digest, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise SystemExit('Choose a new output directory; existing runs are preserved.')
    out.mkdir(parents=True)
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    write_json(out/'baseline_hashes.json',{p:digest(ROOT/p) for p in files if p and (ROOT/p).is_file()})
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'analysis/diagnostics'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLCONFIGDIR='/tmp/qrc-diagnostics-mpl',XDG_CACHE_HOME='/tmp/qrc-diagnostics-cache')
    stages=[('compute',[str(ROOT/'analysis/diagnostics/compute.py'),'--output',str(out)]),
            ('render',[str(ROOT/'analysis/diagnostics/render.py'),'--output',str(out)]),
            ('baseline-tests',['-m','unittest','discover','-s','tests','-v']),
            ('diagnostic-tests',['-m','unittest','discover','-s','analysis/diagnostics','-p','test_*.py','-v']),
            ('verify',[str(ROOT/'analysis/diagnostics/verify.py'),'--output',str(out),'--check-render'])]
    for name,command in stages:
        print(f'{name}: running; log at {out / (name+".log")}',flush=True)
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Complete: {out / "report.html"}',flush=True)


if __name__=='__main__':main()
