"""Check the printed native COF matrix and reproduce the diagnostics in the notes."""
from pathlib import Path
import json,re,numpy as np
HERE=Path(__file__).resolve().parent
ref=json.loads((HERE/'one-turn-matrix-cof-refined.json').read_text())
base=json.loads((HERE/'one-turn-matrix-cof-default.json').read_text())
M=np.array(ref['matrix']);B=np.array(base['matrix']);assert ref['converged'] and base['converged']
text=(HERE.parent/'one-turn-matrix.tex').read_text()
body=re.search(r'\\boxed\{M\\simeq\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}',text,re.S)[1]
shown=np.array([[float(x) for x in row.strip().split('&')] for row in body.strip().split('\\\\')])
assert np.array_equal(shown,np.round(M,8))
assert np.all(M[:2,2:]==0) and np.all(M[2:,:2]==0)
P0=float(np.linalg.norm(ref['momentum_mc']))
report={'source':'Native COF DOP853; fixed-energy forward section return',
 'coordinate_order':['x_section_m','px_over_mc','y_section_m','py_over_mc'],
 'refined_dt_s':ref['dt_s'],'position_fd_m':5e-6,'momentum_fd_dimensionless':5e-6*P0,
 'reference_state':ref['coordinates'],'reference_pz_over_mc':ref['momentum_mc'][2],
 'display_agrees_with_native_json':True,'max_element_change':float(abs(M-B).max()),
 'max_element_change_index_1based':(np.array(np.unravel_index(abs(M-B).argmax(),M.shape))+1).tolist(),
 'determinants':[float(np.linalg.det(M[i:i+2,i:i+2])) for i in [0,2]],
 'full_tunes':ref['full_tunes'],
 'tune_changes':{k:ref['full_tunes'][k]-base['full_tunes'][k] for k in ref['full_tunes']}}
for i,plane in [(0,'horizontal'),(2,'vertical')]:
 A=M[i:i+2,i:i+2];q=float(np.arccos(np.trace(A)/(2*np.sqrt(np.linalg.det(A))))/(2*np.pi))
 assert abs(q-(ref['full_tunes'][plane]%1))<1e-12
assert max(abs(np.array(report['determinants'])-1))<1e-8
(HERE/'one-turn-matrix-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
