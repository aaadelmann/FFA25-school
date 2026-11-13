"""Reproduce native placement exports, check the placement formulas and draw plans.

Uses only copies of the three School inputs. No course inputs/notebooks are edited.
Run with ~/.venv-h6/bin/python make_geometry.py from this directory.
"""
from pathlib import Path
import argparse, ast, base64, hashlib, json, math, os, re, shutil, subprocess, sys, zlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE=Path(__file__).resolve().parent
SCHOOL=HERE.parent
sys.path.insert(0,str(SCHOOL/'pyfiles'))
from check_constants import evaluated_input
CASES=['school-16-cell','school-55deg-d-off','school-fourfold-3MeV']

def decode(path):
    values={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ['vertices_base64','numVertices','triangles','color']:
                values[name]=ast.literal_eval(node.value)
    points=np.frombuffer(zlib.decompress(base64.b64decode(values['vertices_base64'])),dtype='=f8').reshape(-1,3)
    assert len(points)==sum(values['numVertices'])
    edges=np.cumsum([0]+values['numVertices'])
    return [(points[edges[i]:edges[i+1]],values['triangles'][i],values['color'][i]) for i in range(len(edges)-1)]

def wrap(x):return np.angle(np.exp(1j*x))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,default=Path.home()/'git/opalx/build_openmp/bin/opalx')
    parser.add_argument('--work',type=Path,default=Path('/tmp/ffa-school-geometry-document'))
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    rows=[];exports=[];parameters=[]
    for case in CASES:
        source=SCHOOL/case/(case+'.in');text=source.read_text()
        constants,definitions,members=evaluated_input(text)
        directory=args.work/case;directory.mkdir(exist_ok=True)
        if not args.reuse:
            # One harmless field-free-space-charge tracking step triggers the
            # native placement writer. This is not an orbit/tune measurement.
            prefix=text[:text.index('DIST: DISTRIBUTION')]
            prefix=re.sub(r'REAL N_PARTICLES\s*=[^;]+;','REAL N_PARTICLES = 1;',prefix)
            prefix=re.sub(r'REAL BUNCH_CHARGE\s*=[^;]+;','REAL BUNCH_CHARGE = 0;',prefix)
            tail='''DIST: DISTRIBUTION, TYPE=GAUSS, NPARTDIST=1, SIGMAX=0, SIGMAY=0, SIGMAZ=0, SIGMAPX=0, SIGMAPY=0, SIGMAPZ=0;
SOURCE: EMISSIONSOURCE, DISTRIBUTION=DIST;
SOURCES: EMISSIONSOURCELIST=(SOURCE);
BEAM1: BEAM, PARTICLE=PROTON, MASS=P_MASS/1000, PC=P0/1000, NALLOC=1, SOURCES=SOURCES, BCHARGE=0, CHARGE=1;
FS: FIELDSOLVER, TYPE=NONE, NX=8, NY=8, NZ=8;
TRACK, LINE=SCHOOL, BEAM=BEAM1, DT={1e-12}, MAXSTEPS=1, ZSTOP=100;
RUN, METHOD="PARALLEL", FIELDSOLVER=FS;
ENDTRACK;
STOP;
'''
            (directory/'geometry.in').write_text(prefix+tail)
            env=dict(os.environ,OMP_NUM_THREADS='1',OMP_PROC_BIND='false')
            with (directory/'stdout.log').open('w') as out:
                subprocess.run([str(args.exe.resolve()),'--info','1','geometry.in'],cwd=directory,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=120)
        path=directory/'data/geometry_ElementPositions.py'
        meshes=decode(path); assert len(meshes)==len(members)
        r0=constants['R0'];delta=constants['SPIRAL_ANGLE'];remaining=list(members)
        mapped=[]
        for vertices,triangles,color in meshes:
            assert len(vertices)>0
            radius=np.hypot(vertices[:,0]+r0,vertices[:,2])
            phi=np.arctan2(vertices[:,2],vertices[:,0]+r0)
            is_volume=color==9
            psi=phi-math.tan(delta)*np.log(radius/r0) if is_volume else phi
            centre=np.angle(np.mean(np.exp(1j*psi)))
            candidates=[]
            for name in remaining:
                _,a=definitions[name]
                if bool(a['B0']!=0)!=is_volume:continue
                length=a.get('L')
                if length is None:
                    _,end=definitions[a['END_FIELD_MODEL']]
                    length=end['START_HALF_LENGTH']+end['END_HALF_LENGTH']
                entrance=-a['THETA'];expected=entrance+length/(2*r0)
                candidates.append((abs(wrap(centre-expected)),name,length,entrance))
            error,name,length,entrance=min(candidates)
            assert error<2e-12,(case,name,error)
            remaining.remove(name)
            expected=entrance+length/(2*r0)
            offsets=wrap(psi-expected)
            edge_error=max(abs(offsets.min()+length/(2*r0)),abs(offsets.max()-length/(2*r0)))
            assert edge_error<2e-12,(case,name,edge_error)
            a=definitions[name][1]
            assert np.allclose([a['X'],a['Y'],a['Z']],[r0*(math.cos(entrance)-1),0,r0*math.sin(entrance)],atol=2e-13,rtol=0)
            if is_volume:
                assert np.allclose([radius.min(),radius.max()],[r0-a['RADIAL_NEG_EXTENT'],r0+a['RADIAL_POS_EXTENT']],atol=2e-12,rtol=0)
            rows.append(dict(machine=case,element=name,entrance_rad=entrance,length_m=length,B0_T=a['B0'],mesh_center_error_rad=float(error),mesh_edge_error_rad=float(edge_error)))
            mapped.append((name,vertices,triangles,a['B0']))
        assert not remaining
        assert abs(sum(row['length_m'] for row in rows if row['machine']==case)-2*math.pi*r0)<1e-11
        parameters.append(constants);exports.append(mapped)
    pd.DataFrame(rows).to_csv(HERE/'placement-check.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    titles=['16 cells, 12 MeV','55° spiral, D off, 12 MeV','Fourfold, 3 MeV']
    for case,c,meshes,title in zip(CASES,parameters,exports,titles):
        fig,ax=plt.subplots(figsize=(5,5),layout="constrained")
        r0=c['R0'];theta=np.linspace(0,2*math.pi,501)
        ax.plot(r0*np.cos(theta),r0*np.sin(theta),'--',color='.55',lw=.65)
        for name,v,t,b in meshes:
            if t:
                faces=np.asarray(t,dtype=int).reshape(-1,3)
                top=[v[f][:,[0,2]]+np.array([r0,0]) for f in faces if np.all(v[f,1]>.1)]
                ax.add_collection(PolyCollection(top,facecolor='#2166ac' if b>0 else '#d6604d',edgecolors='none',alpha=.9))
            else:
                a=next(row for row in rows if row['machine']==case and row['element']==name)
                is_d=('_D' in name and not name.startswith('GAP'))
                ax.plot(v[:,0]+r0,v[:,2],color='#d6604d' if is_d else '.7',linestyle='--' if is_d else '-',lw=1.1 if is_d else .5)
        ax.scatter([r0],[0],s=14,color='black',zorder=10)
        ax.set(xlabel=r'$X+R_0$ [m]',ylabel=r'$Z$ [m]',title=title,xlim=(-5.85,5.85),ylim=(-5.85,5.85),aspect='equal')
        ax.tick_params(labelsize=8)
        fig.savefig(HERE/(case+'-plan.pdf'));plt.close(fig)
    summary=pd.DataFrame(rows).groupby('machine')[['mesh_center_error_rad','mesh_edge_error_rad']].max()
    print(summary.to_string());print('All 195 native element meshes agree with the input placement formulas.')
    # Keep compact native geometry sources alongside the report for reproducibility.
    target=HERE/'geometry-data';target.mkdir(exist_ok=True)
    for case in CASES:shutil.copy(args.work/case/'data/geometry_ElementPositions.py',target/(case+'_ElementPositions.py'))

if __name__=='__main__':main()
