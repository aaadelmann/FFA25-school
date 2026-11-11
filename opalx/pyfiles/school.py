"""OPALX school input, execution and plotting support.

Physics and full tunes come from OPALX. This module reads the bundled explicit
geometry templates, writes native inputs, and plots native output. It contains
no closed-orbit optimiser or frequency estimator.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from check_constants import evaluate

HERE = Path(__file__).resolve().parent.parent


def input_constants(text):
    """Read native REAL arithmetic for labels and launch checks, not orbit physics."""
    constants = {'PI': math.pi}
    source = re.sub(r'//[^\n]*', '', text)
    for name, expression in re.findall(r'\bREAL\s+(\w+)\s*=\s*([^;]+);', source):
        constants[name] = evaluate(expression, constants)
    return constants


@dataclass(frozen=True)
class Lattice:
    key: str
    label: str
    template: str

    @property
    def input_path(self):
        return HERE / self.template

    @property
    def constants(self):
        return input_constants(self.input_path.read_text())

    @property
    def energy_MeV(self):
        return self.constants['E0']

    @property
    def r0(self):
        return self.constants['R0']

    @property
    def periods(self):
        c = self.constants
        return round(2*math.pi/c['SUPERPERIOD_ANGLE']) if 'SUPERPERIOD_ANGLE' in c else int(c['N_CELLS'])

    @property
    def spiral(self):
        return self.constants['SPIRAL_ANGLE']

    @property
    def d_strength(self):
        return self.constants.get('BD')

    @property
    def p0(self):
        c = self.constants
        return c['P0']/c['P_MASS']

    @property
    def guess(self):
        c = self.constants
        return (c['COF_R_GUESS']-c['R0'], c['COF_PR_GUESS'], c['COF_Y_GUESS'], c['COF_PY_GUESS'])


CASES = (
    Lattice('16-cell', '16-cell ring · 12 MeV', 'school-16-cell/school-16-cell.in'),
    Lattice('55deg-d-off', '55° spiral · D off', 'school-55deg-d-off/school-55deg-d-off.in'),
    Lattice('fourfold', 'Fourfold ring · 3 MeV', 'school-fourfold-3MeV/school-fourfold-3MeV.in'),
)


def settings_table(case):
    """Show the controls actually read from this machine's input."""
    text = case.input_path.read_text()
    command = re.search(r'\bORBIT\s*:\s*COF\s*,([^;]+);', text, re.S)[1]
    integrator = re.search(r'TIMEINTEGRATOR\s*=\s*"([^"\n]+)"', command)[1]
    c = case.constants
    values = {'Input': case.template, 'Energy [MeV]': c['E0'], 'R0 [m]': c['R0'],
              'Field index k': c['K_VAL'], 'Spiral [deg]': math.degrees(c['SPIRAL_ANGLE']),
              'Optical periods': case.periods, 'COF integrator': integrator,
              'COF DT [ns]': c['COF_DT']*1e9, 'COF FDSTEP [m]': c['COF_FDSTEP'],
              'COF MAXSTEPS': int(c['COF_MAXSTEPS']),
              'COF XTOL [m]': c['COF_XTOL'], 'COF PTOL [p/(mc)]': c['COF_PTOL'],
              'Tracking DT [ns]': c['TRACK_DT']*1e9, 'Tracking turns': int(c['NUM_TURNS']),
              'HDF5 dump interval': int(c['DUMP_EVERY'])}
    return pd.DataFrame(values.items(), columns=['Parameter', 'Value'])


def find_executable(value=None):
    value = value or os.environ.get('OPALX_EXE') or shutil.which('opalx')
    if not value:
        raise FileNotFoundError('Set OPALX_EXE to an OPALX build with native FFA COF full tunes.')
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f'OPALX executable is unavailable: {path}')
    return path


def mpi_flags():
    return shlex.split(os.environ.get('OPALX_MPI_FLAGS', ''))


def _set_constants(text, values):
    for name,value in values.items():
        rendered = value if isinstance(value,str) else repr(float(value))
        pattern = r'\bREAL\s+'+re.escape(name)+r'\s*=\s*[^;]+;'
        text,count = re.subn(pattern, f'REAL {name} = {rendered};', text)
        if count != 1: raise ValueError(f'Expected one definition of {name}')
    return text


def lattice_text(case):
    """Retain this machine's native constants and geometry without substitutions."""
    source = re.sub(r'//[^\n]*','',case.input_path.read_text())
    lines = []
    for statement in source.split(';'):
        statement = ' '.join(statement.split())
        if (re.match(r'REAL\s+',statement)
                or re.match(r'\w+\s*:\s*(ASYMMETRIC_ENGE|SCALINGFFAMAGNET)\s*,',statement)
                or re.match(r'SCHOOL\s*:\s*RING\s*=',statement)):
            lines.append(statement+';')
    if sum(bool(re.search(r'\bEND_FIELD_MODEL\s*=',line)) for line in lines) != 32:
        raise ValueError('Expected exactly 32 physical magnets in the template')
    return '\n'.join(lines)


def _definitions(case, overrides):
    text = _set_constants(lattice_text(case),overrides)
    statements = [statement.strip()+';' for statement in text.split(';') if statement.strip()]
    return ('\n'.join(s for s in statements if s.startswith('REAL ')),
            '\n'.join(s for s in statements if not s.startswith('REAL ')))


def _source_lines(case, fromfile=False):
    distribution = ('DIST: DISTRIBUTION, TYPE=FROMFILE, FNAME="particles.dat", NPARTDIST=N_PARTICLES;'
                    if fromfile else 'DIST: DISTRIBUTION, TYPE=GAUSS, NPARTDIST=N_PARTICLES, '
                    'SIGMAX=0, SIGMAY=0, SIGMAZ=0, SIGMAPX=0, SIGMAPY=0, SIGMAPZ=0;')
    momentum = '' if fromfile else 'PC=P0/1000, '
    return [distribution,'SOURCE: EMISSIONSOURCE, DISTRIBUTION=DIST;',
            'SOURCES: EMISSIONSOURCELIST=(SOURCE);',
            'BEAM1: BEAM, PARTICLE=PROTON, MASS=P_MASS/1000, '+momentum+
            'NALLOC=N_PARTICLES, SOURCES=SOURCES, BCHARGE=BUNCH_CHARGE, CHARGE=P_CHARGE;']


def _fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _manifest(path, case, settings):
    info = dict(case=case.__dict__, settings=settings,
                template_sha256=hashlib.sha256(case.input_path.read_bytes()).hexdigest(),
                input_sha256=hashlib.sha256((path/'school.in').read_bytes()).hexdigest())
    (path/'manifest.json').write_text(json.dumps(info, indent=2)+'\n')



def format_input(text):
    """Use the school-16-cell layout without changing OPAL tokens or numbers."""
    items, buffer = [], []
    quoted = escaped = False
    i = 0
    while i < len(text):
        char = text[i]
        if not quoted and text.startswith('//', i):
            end = text.find('\n', i)
            if end < 0: end = len(text)
            items.append(('comment', text[i:end].rstrip()))
            buffer.append(' ')
            i = end
            continue
        if char == '"' and not escaped: quoted = not quoted
        if char == ';' and not quoted:
            statement = ''.join(buffer).strip()
            if statement: items.append(('statement', statement))
            buffer = []
        else: buffer.append(char)
        escaped = char == '\\' and not escaped if quoted else False
        i += 1
    if quoted or ''.join(buffer).strip(): raise ValueError('Incomplete native input statement')

    def parts(statement):
        answer, token, depth = [], [], 0
        quoted = escaped = False
        for char in statement:
            if char == '"' and not escaped: quoted = not quoted
            if not quoted:
                if char in '{([': depth += 1
                elif char in '})]': depth -= 1
            if char == ',' and not quoted and depth == 0:
                answer.append(''.join(token).strip()); token = []
            else: token.append(char)
            escaped = char == '\\' and not escaped if quoted else False
        answer.append(''.join(token).strip())
        return answer

    blocks, previous = [], None
    for kind, statement in items:
        if kind == 'comment':
            block, category = statement, 'comment'
        elif re.match(r'\w+\s*:\s*RING\s*=', statement):
            prefix, members = statement.split('(', 1)
            members = [value.strip() for value in members.rsplit(')', 1)[0].split(',')]
            lines, line = [], prefix.strip()+'('
            for index, member in enumerate(members):
                suffix = ',' if index+1 < len(members) else ');'
                if len(line)+len(member)+len(suffix) > 80 and not line.endswith('('):
                    lines.append(line); line = '        '
                line += member+suffix
            block, category = '\n'.join(lines+[line]), 'ring'
        else:
            tokens = parts(statement)
            command = re.sub(r'.*:\s*', '', tokens[0]).strip()
            breaks = {'SCALINGFFAMAGNET': {'RADIAL_NEG_EXTENT'},
                      'COF': {'MAXPATH', 'FDSTEP', 'SCALES'},
                      'TRACK': {'MAXSTEPS'},
                      'DUMPEMFIELDS': {'X_START','R_START','Y_START','Z_START',
                                       'PHI_START','T_START','FILE_NAME'}}.get(command, set())
            lines, line = [], ' '.join(tokens[0].split())
            for token in tokens[1:]:
                # Attribute values (including quoted filenames/arrays) stay verbatim.
                key = token.split('=', 1)[0].strip()
                if key in breaks:
                    lines.append(line+','); line = '       '+token
                else: line += ', '+token
            block = '\n'.join(lines+[line+';'])
            category = ('constant' if re.match(r'REAL\s+',command) else
                        'fringe' if command == 'ASYMMETRIC_ENGE' else
                        'source' if command in {'DISTRIBUTION','EMISSIONSOURCE'} or
                        tokens[0].strip().startswith('SOURCES:') else command)
        consecutive = category == previous and category in {'constant','fringe','source','comment'}
        stop = category == 'STOP'
        blocks.append(('\n' if consecutive or stop else '\n\n')+block if blocks else block)
        previous = category
    return ''.join(blocks).strip()+'\n'


def prepare_cof(case, path, dt=None, fd=None, guess_offset=(0.,0.,0.,0.)):
    """Copy the standalone solve verbatim; override controls only when requested."""
    if (dt is not None and dt <= 0) or (fd is not None and fd <= 0):
        raise ValueError('DT and FDSTEP must be positive')
    offset = np.asarray(guess_offset, dtype=float)
    if offset.shape != (4,) or not np.isfinite(offset).all():
        raise ValueError('guess_offset must contain four finite coordinates')
    text = case.input_path.read_text()
    overrides = {}
    if dt is not None: overrides['COF_DT'] = dt
    if fd is not None: overrides['COF_FDSTEP'] = fd
    if np.any(offset):
        x, px, y, py = np.asarray(case.guess)+offset
        overrides.update(COF_R_GUESS=f'R0+({x:.17g})', COF_PR_GUESS=px,
                         COF_Y_GUESS=y, COF_PY_GUESS=py)
    text = _set_constants(text, overrides)
    path = _fresh(path)
    (path/'school.in').write_text(text)
    c = input_constants(text)
    _manifest(path, case, dict(kind='COF', dt=c['COF_DT'], fd=c['COF_FDSTEP'],
                              guess_offset=offset.tolist()))
    return path


def execute(path, executable, flags=None, timeout=2400):
    ranks = 1
    path = Path(path).resolve()
    if (path/'run.log').exists(): raise FileExistsError('Use a fresh run directory; run.log exists')
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_PROC_BIND='false')
    env.pop('OPALX_TEST_PASSIVE_RING_PROBES', None)
    launcher = shlex.split(os.environ.get('OPALX_LAUNCHER', ''))
    options = mpi_flags() if flags is None else flags
    if options and not launcher: launcher = ['mpiexec', '-n', '1']
    command = [*launcher, *options, str(find_executable(executable)), '--info', '1', 'school.in']
    print(f'Running {path.name} on {ranks} rank(s); log: {path / "run.log"}', flush=True)
    with (path/'run.log').open('w') as stream:
        try:
            subprocess.run(command, cwd=path, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f'OPALX failed; inspect {path / "run.log"}\n'
                               +(path/'run.log').read_text()[-4000:]) from error
    manifest = json.loads((path/'manifest.json').read_text())
    manifest.update(command=command, ranks=ranks)
    (path/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def read_cof(case, path):
    path = Path(path)
    result = json.loads((path/'orbit.json').read_text())
    if not result.get('converged'): raise RuntimeError('COF did not converge')
    if not result.get('full_tunes') or result.get('stability') != 'STABLE':
        raise RuntimeError('Native full tunes unavailable: '+result.get('full_tunes_diagnostic',
                           'use a build with native full tunes; inspect DT/FDSTEP and run.log'))
    if abs(result['energy_MeV']-case.energy_MeV) > 1e-8:
        raise RuntimeError('COF energy does not match this exercise')
    residual = np.asarray(result['residual'])
    c = input_constants((path/'school.in').read_text())
    if max(abs(residual[[0,2]])) > c['COF_XTOL'] or max(abs(residual[[1,3]])) > c['COF_PTOL']:
        raise RuntimeError('Closed-orbit residual exceeds configured tolerances')
    for field in ['coordinates', 'position_m', 'momentum_mc']:
        if not np.isfinite(result[field]).all(): raise RuntimeError(f'Nonfinite {field}')
    return result


def native_report(path):
    prefixes = ('Status=', 'Coordinates (', 'Residual:', 'One-turn eigenanalysis:',
                'Mode ', 'Full tunes ', 'Phase samples=')
    return '\n'.join(line for line in (Path(path)/'run.log').read_text().splitlines()
                     if any(prefix in line for prefix in prefixes))


def results_table(results):
    return pd.DataFrame([dict(lattice=case.label, energy_MeV=result['energy_MeV'],
                              radius_m=case.r0+result['coordinates'][0],
                              pr_mc=result['coordinates'][1],
                              Qr=result['full_tunes']['horizontal'], Qy=result['full_tunes']['vertical'],
                              max_position_residual_m=max(abs(np.array(result['residual'])[[0,2]])))
                         for case, result in results])


def prepare_tracking(case, result, path, turns=None, dt=None, displacement=1e-3,
                     dump_every=None, maps=True):
    c = case.constants
    turns = int(c['NUM_TURNS']) if turns is None else turns
    dt = c['TRACK_DT'] if dt is None else dt
    dump_every = int(c['DUMP_EVERY']) if dump_every is None else dump_every
    if not isinstance(turns,int) or turns < 1 or turns > 100: raise ValueError('turns must be 1..100')
    if dt <= 0 or displacement <= 0 or not isinstance(dump_every,int) or dump_every < 1:
        raise ValueError('Positive DT, displacement and integer dump frequency required')
    path = _fresh(path)
    offsets = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(0,0)]
    position, momentum = np.array(result['position_m']), np.array(result['momentum_mc'])
    rows = []
    for radial, vertical in offsets:
        r = position+displacement*np.array([radial,vertical,0.])
        rows.append(' '.join(f'{v:.17g}' for v in [r[0],momentum[0],r[1],momentum[1],r[2],momentum[2]]))
    (path/'particles.dat').write_text('8\nx px y py z pz\n'+'\n'.join(rows)+'\n')
    constants,geometry = _definitions(case,dict(TRACK_DT=dt,NUM_TURNS=turns,DUMP_EVERY=dump_every,N_PARTICLES=8,BUNCH_CHARGE=0))
    lines = ['OPTION, ECHO=FALSE;',constants,
             'OPTION, ENABLEHDF5=TRUE, ENABLELINEARTRANSFERMAPS=FALSE, '
             'PSDUMPFREQ=DUMP_EVERY, STATDUMPFREQ=STAT_DUMP_FREQ, '
             'REPARTFREQ=REPART_FREQ, VERSION=INPUT_VERSION;',geometry,
             *_source_lines(case,fromfile=True)]
    if maps:
        lines += ['DUMPEMFIELDS, COORDINATE_SYSTEM=CARTESIAN, X_START=-R0-MAP_EDGE, '
                  'X_STEPS=MAP_CART_STEPS, DX=MAP_CART_STEP, Y_START=0, Y_STEPS=1, DY=MAP_Y_STEP, '
                  'Z_START=-MAP_EDGE, Z_STEPS=MAP_CART_STEPS, DZ=MAP_CART_STEP, '
                  'T_START=0, T_STEPS=1, DT=MAP_T_STEP, FILE_NAME="'+str(path/'field-cartesian.dat')+'";',
                  'DUMPEMFIELDS, COORDINATE_SYSTEM=CYLINDRICAL, CYL_ORIGIN_X=-R0, '
                  'R_START=MAP_R_MIN, R_STEPS=MAP_R_STEPS, DR=MAP_R_STEP, '
                  'PHI_START=0, PHI_STEPS=MAP_PHI_STEPS, DPHI=MAP_PHI_STEP, '
                  'Y_START=0, Y_STEPS=1, DY=MAP_Y_STEP, T_START=0, T_STEPS=1, DT=MAP_T_STEP, '
                  'FILE_NAME="'+str(path/'field-rphi.dat')+'";']
    lines += ['FS: FIELDSOLVER, TYPE=NONE, NX=N_GRID, NY=N_GRID, NZ=N_GRID;',
              'TRACK, LINE=SCHOOL, BEAM=BEAM1, DT={TRACK_DT}, MAXSTEPS=TRACK_MAXSTEPS, ZSTOP=TRACK_ZSTOP;',
              'RUN, METHOD="PARALLEL", FIELDSOLVER=FS, TURNS=NUM_TURNS;', 'ENDTRACK;', 'STOP;']
    (path/'school.in').write_text(format_input('\n'.join(lines)+'\n'))
    _manifest(path, case, dict(kind='TRACK', turns=turns, dt=dt, displacement=displacement,
                               dump_every=dump_every, maps=maps, input_rows=offsets,
                               closed_position_m=position.tolist(), closed_momentum_mc=momentum.tolist()))
    return path


def read_trajectories(case, path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    turns = manifest['settings']['turns']
    if manifest['ranks'] != 1: raise ValueError('This school reader expects a single-rank course run')
    if f'completed directed turn {turns},' not in (path/'run.log').read_text():
        raise RuntimeError('Requested directed turns did not complete')
    files = list(path.glob('*.h5'))
    if len(files) != 1: raise RuntimeError(f'Expected one HDF5 file, found {files}')
    frames = []
    with h5py.File(files[0], 'r') as stream:
        groups = sorted((key for key in stream if key.startswith('Step#')),
                        key=lambda key:int(key.split('#')[1]))
        for frame, key in enumerate(groups):
            group = stream[key]
            data = {name:group[name][...] for name in ['id','x','y','z','px','py','pz']}
            ids = data.pop('id').astype(int)
            if not np.array_equal(np.sort(ids),np.arange(8)):
                raise RuntimeError(f'Missing, duplicate or lost particles in {key}: {ids}')
            # Parallel-tracker HDF5 payloads use the moving reference chart.
            # RefPartR/RefPartP remain in laboratory coordinates. For these
            # planar rings local z follows RefPartP and local y is vertical.
            origin = np.asarray(group.attrs['RefPartR'], dtype=float).reshape(3)
            reference_p = np.asarray(group.attrs['RefPartP'], dtype=float).reshape(3)
            horizontal = np.hypot(reference_p[0], reference_p[2])
            if horizontal <= 0 or abs(reference_p[1]) > 1e-7*horizontal:
                raise RuntimeError('This trajectory reader requires a planar reference orbit')
            tangent = np.array([reference_p[0], 0., reference_p[2]])/horizontal
            normal = np.array([tangent[2], 0., -tangent[0]])
            rotation = np.column_stack([normal, [0.,1.,0.], tangent])
            local_r = np.column_stack([data[name] for name in ['x','y','z']])
            local_p = np.column_stack([data[name] for name in ['px','py','pz']])
            lab_r, lab_p = local_r @ rotation.T+origin, local_p @ rotation.T
            table = pd.DataFrame({name:lab_r[:,i] for i,name in enumerate(['x','y','z'])})
            for i,name in enumerate(['px','py','pz']): table[name] = lab_p[:,i]
            table['id'], table['frame'] = ids, frame
            table['time_s'] = float(np.asarray(group.attrs['TIME']).ravel()[0])
            table['row'] = ids
            frames.append(table)
    if len(frames) < 10: raise RuntimeError('Insufficient particle snapshots for plotting')
    data = pd.concat(frames,ignore_index=True).sort_values(['row','frame'])
    if not np.isfinite(data.to_numpy()).all(): raise RuntimeError('Nonfinite particle output')
    data['radius_m'] = np.hypot(data.x+case.r0,data.z)
    data['phi_rad'] = 0.
    for _, indices in data.groupby('row').groups.items():
        group = data.loc[indices]
        data.loc[indices,'phi_rad'] = np.unwrap(np.arctan2(group.z,group.x+case.r0))
    theta = np.arctan2(data.z,data.x+case.r0)
    data['pr_mc'] = data.px*np.cos(theta)+data.pz*np.sin(theta)
    data['p_norm_mc'] = np.sqrt(data.px**2+data.py**2+data.pz**2)
    # Detect a frame/schema mismatch against the short initial reference drift.
    first = data[(data.row==0)].iloc[0]
    expected = np.array(manifest['settings']['closed_position_m'])
    momentum = np.array(manifest['settings']['closed_momentum_mc'])
    predicted = expected+299792458.*first.time_s*momentum/np.sqrt(1+momentum@momentum)
    if np.linalg.norm(first[['x','y','z']].to_numpy(dtype=float)-predicted) > .02:
        raise RuntimeError('HDF5 coordinates are not in the expected global launch frame')
    data.to_csv(path/'trajectories.csv',index=False)
    return data


def read_field(path, cylindrical=False):
    columns = (['r','phi_deg','y','t_ns','Br','Bphi','By','Er','Ephi','Ey'] if cylindrical else
               ['x','y','z','t_ns','Bx','By','Bz','Ex','Ey','Ez'])
    lines = Path(path).read_text().splitlines()
    if len(lines) < 13 or lines[11].strip() != '0': raise ValueError('Unexpected DUMPEMFIELDS header')
    data = pd.read_csv(path,sep=r'\s+',skiprows=12,names=columns)
    if len(data)!=int(lines[0]) or not np.isfinite(data.to_numpy()).all():
        raise ValueError('Field map is incomplete/nonfinite')
    return data


def section_samples(case, data):
    """Interpolate HDF5 to equivalent sections for plots, never for native tunes."""
    groups = {}
    for row,frame in data.groupby('row'):
        if np.min(np.diff(frame.phi_rad)) < -1e-10:
            raise RuntimeError('Orbit azimuth is not forward monotonic')
        groups[row] = frame.drop_duplicates('phi_rad',keep='last').sort_values('phi_rad')
    lower = max(g.phi_rad.min() for g in groups.values())
    upper = min(g.phi_rad.max() for g in groups.values())
    step = 2*np.pi/case.periods
    indices = np.arange(math.ceil((lower+1e-10)/step), math.floor((upper-1e-10)/step)+1)
    targets = indices*step
    if len(targets)<case.periods: raise RuntimeError('Not enough common sections for phase plots')
    tables = []
    names = ['radius_m','pr_mc','y','py']
    reference = groups[0]
    baseline = {name:np.interp(targets,reference.phi_rad,reference[name]) for name in names}
    for row,g in groups.items():
        if not (np.diff(g.phi_rad)>0).all(): raise RuntimeError('Orbit azimuth is not forward monotonic')
        values = {name:np.interp(targets,g.phi_rad,g[name]) for name in names}
        table = pd.DataFrame(values)
        table['row'],table['section'],table['phi_rad'] = row,indices,targets
        for name in names: table['delta_'+name] = table[name]-baseline[name]
        tables.append(table)
    return pd.concat(tables,ignore_index=True)


def plot_fields_and_orbit(case, path, trajectories):
    path = Path(path)
    cart = read_field(path/'field-cartesian.dat')
    cyl = read_field(path/'field-rphi.dat',cylindrical=True)
    ref = trajectories[(trajectories.row==0)&(trajectories.phi_rad<=2*np.pi)]
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    grid = cart.pivot(index='z',columns='x',values='By')
    band = cyl[(cyl.r >= ref.radius_m.min()-.05)&(cyl.r <= ref.radius_m.max()+.05)]
    limit = float(abs(band.By).max()) if len(band) else float(abs(grid.to_numpy()).max())
    image = axes[0].pcolormesh(grid.columns+case.r0,grid.index,grid.to_numpy(),cmap='RdBu_r',
                              vmin=-limit,vmax=limit,shading='auto',rasterized=True)
    axes[0].plot(ref.x+case.r0,ref.z,color='black',lw=1,label='Tracked closed launch')
    axes[0].set(xlabel='x + R₀ [m]',ylabel='z [m]',aspect='equal',title='Median-plane native field')
    axes[0].legend(fontsize=8,loc='upper right')
    fig.colorbar(image,ax=axes[0],label='Bᵧ [T] · orbit-band scale',shrink=.8,extend='both')
    grid = cyl.pivot(index='r',columns='phi_deg',values='By')
    axes[1].pcolormesh(grid.columns,grid.index,grid.to_numpy(),cmap='RdBu_r',
                       vmin=-limit,vmax=limit,shading='auto',rasterized=True)
    axes[1].plot(np.degrees(ref.phi_rad),ref.radius_m,color='black',lw=1)
    axes[1].set(xlabel='Azimuth [degrees]',ylabel='Radius [m]',title='Scalloping over one revolution',
                xlim=(0,360),ylim=(ref.radius_m.min()-.03,ref.radius_m.max()+.03))
    fig.suptitle(case.label)
    fig.savefig(path/'fields-and-orbit.png',dpi=180)
    fig.savefig(path/'fields-and-orbit.pdf')
    return fig


def plot_oscillations(case, path, data):
    sections = section_samples(case,data)
    fig,axes = plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for row,color,label in [(1,'#12629a','+ radial'),(3,'#b65229','+ vertical'),(5,'#467644','+ both')]:
        g = sections[sections.row==row]
        turns = g.phi_rad/(2*np.pi)
        axes[0,0].plot(turns,1e3*g.delta_radius_m,'o-',ms=3,color=color,label=label)
        axes[0,1].plot(turns,1e3*g.delta_y,'o-',ms=3,color=color)
        axes[1,0].plot(1e3*g.delta_radius_m,1e3*g.delta_pr_mc,'o-',ms=3,color=color)
        axes[1,1].plot(1e3*g.delta_y,1e3*g.delta_py,'o-',ms=3,color=color)
    axes[0,0].set(xlabel='Revolutions',ylabel='Δr [mm]')
    axes[0,1].set(xlabel='Revolutions',ylabel='Δy [mm]')
    axes[1,0].set(xlabel='Δr [mm]',ylabel='Δpᵣ/(mc) [10⁻³]')
    axes[1,1].set(xlabel='Δy [mm]',ylabel='Δpᵧ/(mc) [10⁻³]')
    axes[0,0].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle(case.label+' · equivalent-section samples (interpolated HDF5)')
    fig.savefig(Path(path)/'oscillations.png',dpi=180)
    fig.savefig(Path(path)/'oscillations.pdf')
    sections.to_csv(Path(path)/'section-samples.csv',index=False)
    return fig
