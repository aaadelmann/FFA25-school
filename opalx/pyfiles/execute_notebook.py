#!/usr/bin/env python3
"""Execute the School notebook in a fresh kernel; Docker is its visible default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ['FFA25_OPAL.ipynb','optimiseCO.py','probes_tune_calc.py','read_dat.py','plot_dat.py']


def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in LEGACY}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['docker','native'],help='Override the notebook in a local executed copy')
    parser.add_argument('--image',help='Override the Docker image in a local executed copy')
    parser.add_argument('--executable',type=Path,help='Explicit native executable; requires --backend native')
    parser.add_argument('--mpi-flags',help='Optional native launcher flags')
    args=parser.parse_args()
    if args.executable and args.backend!='native': parser.error('--executable requires --backend native')
    if args.backend=='native' and not args.executable: parser.error('--backend native requires --executable')
    if args.image and args.backend=='native': parser.error('--image is for Docker mode')
    if args.mpi_flags is not None:
        if args.backend!='native': parser.error('--mpi-flags requires --backend native')
        os.environ['OPALX_MPI_FLAGS']=args.mpi_flags
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('IPYTHONDIR',str(ROOT/'opalx/runs/.cache/ipython'))
    before=hashes()
    source=ROOT/'opalx/FFA26_OPALX.ipynb'
    nb=nbformat.read(source,as_version=4)
    overrides={}
    if args.backend: overrides['BACKEND']=args.backend
    if args.image: overrides['IMAGE']=args.image
    if args.executable: overrides['EXECUTABLE']=str(args.executable.expanduser().resolve())
    for cell in nb.cells:
        if cell.cell_type!='code': continue
        cell.outputs=[];cell.execution_count=None
        for name,value in overrides.items():
            cell.source=re.sub(r'^'+name+r'\s*=.*$',lambda match: name+' = '+repr(value),cell.source,flags=re.M)
    with tempfile.TemporaryDirectory(prefix='ffa-notebook-kernel-') as directory:
        spec=Path(directory)/'python3';spec.mkdir()
        (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher',
                                                             '-f','{connection_file}'],
                                                        display_name='Python 3',language='python')))
        manager=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        client=NotebookClient(nb,timeout=2400,resources={'metadata':{'path':str(ROOT/'opalx')}},km=manager)
        try:
            client.execute(cleanup_kc=True)
        except Exception:
            failed=ROOT/'opalx/runs/notebook-failed.ipynb'
            failed.parent.mkdir(parents=True,exist_ok=True)
            nbformat.write(nb,failed)
            raise
        finally:
            if manager.has_kernel: manager.shutdown_kernel(now=True)
    if before!=hashes(): raise RuntimeError('A legacy source file changed during notebook execution')
    changed_code=[]
    if overrides:
        # Keep the visible Docker default intact; the local copy shows the override.
        output=ROOT/'opalx/runs/notebook-override-executed.ipynb'
    else:
        # Preserve concurrent editor changes; only attach outputs to matching code.
        current=nbformat.read(source,as_version=4)
        executed={cell.id:cell for cell in nb.cells}
        for cell in current.cells:
            old=executed.get(cell.id)
            if cell.cell_type!='code': continue
            if old is None or old.cell_type!='code' or old.source!=cell.source:
                changed_code.append(cell.id)
                cell.outputs=[];cell.execution_count=None
            else:
                cell.outputs=old.outputs;cell.execution_count=old.execution_count
                if 'execution' in old.metadata: cell.metadata['execution']=old.metadata['execution']
        nb=current;output=source
    nbformat.validate(nb);nbformat.write(nb,output)
    if changed_code:
        raise RuntimeError('Code edited during execution; outputs preserved only for unchanged cells: '+str(changed_code))
    code=[cell for cell in nb.cells if cell.cell_type=='code']
    figures=sum('image/png' in item.get('data',{}) for cell in code for item in cell.outputs)
    if any(item.output_type=='error' for cell in code for item in cell.outputs):
        raise RuntimeError('Notebook contains error output')
    report=dict(notebook=str(output),executed_code_cells=len(code),figures=figures,
                original_files_sha256=before,original_files_unchanged=True,configuration_overrides=overrides)
    (ROOT/'opalx/runs/notebook-execution-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
