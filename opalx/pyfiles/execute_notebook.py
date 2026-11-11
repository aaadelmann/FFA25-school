#!/usr/bin/env python3
"""Execute the new school notebook with this Python environment and a fresh kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
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
    parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--mpi-flags',default=os.environ.get('OPALX_MPI_FLAGS',''))
    args=parser.parse_args()
    os.environ['OPALX_EXE']=str(args.executable.expanduser().resolve())
    os.environ['OPALX_MPI_FLAGS']=args.mpi_flags
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('IPYTHONDIR',str(ROOT/'opalx/runs/.cache/ipython'))
    before=hashes()
    source=ROOT/'opalx/FFA26_OPALX.ipynb'
    nb=nbformat.read(source,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code': cell.outputs=[];cell.execution_count=None
    with tempfile.TemporaryDirectory(prefix='ffa-notebook-kernel-') as directory:
        spec=Path(directory)/'python3';spec.mkdir()
        import sys
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
    # Keep edits made in the editor while the native processes were running.
    current=nbformat.read(source,as_version=4)
    executed={cell.id:cell for cell in nb.cells}
    changed_code=[]
    for cell in current.cells:
        old=executed.get(cell.id)
        if cell.cell_type!='code': continue
        if old is None or old.cell_type!='code' or old.source!=cell.source:
            changed_code.append(cell.id)
            cell.outputs=[];cell.execution_count=None
        else:
            cell.outputs=old.outputs;cell.execution_count=old.execution_count
            if 'execution' in old.metadata: cell.metadata['execution']=old.metadata['execution']
    nb=current
    nbformat.validate(nb)
    nbformat.write(nb,source)
    if changed_code:
        raise RuntimeError('Code edited during execution; outputs preserved only for unchanged cells: '+str(changed_code))
    code=[cell for cell in nb.cells if cell.cell_type=='code']
    figures=sum('image/png' in output.get('data',{}) for cell in code for output in cell.outputs)
    if any(output.output_type=='error' for cell in code for output in cell.outputs):
        raise RuntimeError('Notebook contains error output')
    report=dict(notebook=str(source),executed_code_cells=len(code),figures=figures,
                original_files_sha256=before,original_files_unchanged=True,
                executable=os.environ['OPALX_EXE'],mpi_flags=args.mpi_flags)
    (ROOT/'opalx/runs/notebook-execution-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
