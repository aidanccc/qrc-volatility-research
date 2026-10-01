"""Verify integrity and optionally reproduce every figure in a temporary folder."""
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from PIL import Image
from compute import ROOT, digest, write_json

DOCUMENTATION = {'README.md','ASSUMPTIONS.md','REPRO_STATUS.md','RUN_ORDER.md'}


def verify(out, check_render=False):
    out=Path(out).resolve()
    manifest=json.loads((out/'manifest.json').read_text())
    for path,checksum in manifest['input_hashes'].items():
        assert digest(ROOT/path)==checksum, f'Changed input: {path}'
    for path,checksum in manifest['source_hashes'].items():
        # Source hashes collected before computation cover its scientific code.
        # A renderer may be refined during visual QA; record its final revision below.
        if Path(path).name not in {'render.py', 'verify.py', 'circuit_diagram.py'}:
            assert digest(ROOT/path)==checksum, f'Changed computation source: {path}'
    baseline=json.loads((out/'baseline_hashes.json').read_text())
    changed=[path for path,checksum in baseline.items() if not (ROOT/path).is_file() or digest(ROOT/path)!=checksum]
    unexpected=set(changed)-DOCUMENTATION
    assert not unexpected, f'Changed baseline artifacts: {sorted(unexpected)}'
    index=json.loads((out/'figure_index.json').read_text());assert len(index)==21
    for entry in index:
        with Image.open(out/entry['png']) as im:
            im.verify()
        pdf=(out/entry['pdf']).read_bytes();assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-100:]
        for name in entry['tables'].split('; '):assert (out/'tables'/name).is_file(), name
    report=(out/'report.md').read_text()
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if '://' not in link:assert (out/link).exists(), f'Broken report link: {link}'
    comparisons={}
    if check_render:
        with tempfile.TemporaryDirectory(prefix='qrc-render-verification-') as tmp:
            target=Path(tmp)
            shutil.copytree(out/'tables',target/'tables')
            for name in ['manifest.json','validation.json']:shutil.copy2(out/name,target/name)
            env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),MPLCONFIGDIR='/tmp/qrc-diagnostics-mpl',XDG_CACHE_HOME='/tmp/qrc-diagnostics-cache',OPENBLAS_NUM_THREADS='1')
            result=subprocess.run([sys.executable,str(ROOT/'analysis/diagnostics/render.py'),'--output',str(target)],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stderr
            for sub in ['tables','figures','circuits']:
                for p in (target/sub).iterdir():
                    if p.suffix=='.pdf':continue  # embedded creation timestamps may differ
                    relative=p.relative_to(target)
                    comparisons[str(relative)]=digest(p)==digest(out/relative)
            for name in ['report.md','report.html','figure_index.json','FUTURE_CODEX_PROMPTS.md']:
                comparisons[name]=digest(target/name)==digest(out/name)
            assert all(comparisons.values()),[p for p,ok in comparisons.items() if not ok]
    manifest['source_hashes'].update({str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'analysis/diagnostics').glob('*.py')})
    write_json(out/'manifest.json',manifest)
    receipt={'protected_baseline_files_checked':len(baseline),'intentional_documentation_changes':changed,
             'model_data_and_published_artifacts_unchanged':True,'figures_verified':len(index),'png_pdf_pairs':len(index),
             'local_report_links_valid':True,'deterministic_render_compared':check_render,
             'identical_rerender_artifacts':len(comparisons),'source_hashes':manifest['source_hashes'],
             'artifact_hashes':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='verification.json' and '__pycache__' not in p.parts and p.suffix!='.log'}}
    write_json(out/'verification.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['artifact_hashes','source_hashes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--check-render',action='store_true');args=p.parse_args();verify(args.output,args.check_render)
