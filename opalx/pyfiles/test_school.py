"""Reader/identity regressions with analytic synthetic HDF5, no OPALX required."""
import json
import os
import subprocess
import sys
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import textwrap

import h5py
import numpy as np
import nbformat
import execute_notebook

import school
from check_constants import evaluated_input


class SchoolReaderTest(unittest.TestCase):
    def fixture(self, root, name, ranks):
        case = school.CASES[0]
        result = dict(position_m=[0.,0.,0.],momentum_mc=[0.,0.,case.p0])
        path = school.prepare_tracking(case,result,root/name,turns=3,maps=False)
        manifest = json.loads((path/'manifest.json').read_text())
        manifest['ranks'] = ranks
        (path/'manifest.json').write_text(json.dumps(manifest))
        (path/'run.log').write_text('Synthetic reader fixture: completed directed turn 3,\n')
        with h5py.File(path/'synthetic.h5','w') as stream:
            ids = np.array([3,0,7,2,6,1,5,4])
            rows = ids
            for step,theta in enumerate(np.linspace(0,6*np.pi,73)):
                group = stream.create_group(f'Step#{step}')
                group.attrs['TIME'] = [theta*case.r0*np.sqrt(1+case.p0**2)/(299792458.*case.p0)]
                group.attrs['RefPartR'] = [case.r0*(np.cos(theta)-1),0.,case.r0*np.sin(theta)]
                group.attrs['RefPartP'] = [-case.p0*np.sin(theta),0.,case.p0*np.cos(theta)]
                for key,values in dict(id=ids,x=rows*1e-3,y=rows*1e-4,z=np.zeros(8),
                                       px=np.zeros(8),py=np.zeros(8),pz=np.full(8,case.p0)).items():
                    group[key] = values
        return path

    def test_moving_frame_reconstruction_and_particle_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory),'single',1)
            data = school.read_trajectories(school.CASES[0],path)
            for row,group in data.groupby('row'):
                np.testing.assert_allclose(group.radius_m,school.CASES[0].r0+row*1e-3,atol=2e-14,rtol=0)
                np.testing.assert_allclose(group.y,row*1e-4,atol=1e-15,rtol=0)
                np.testing.assert_allclose(group.pr_mc,0.,atol=1e-16,rtol=0)
                np.testing.assert_allclose(group.p_norm_mc,school.CASES[0].p0,atol=1e-16,rtol=0)
            self.assertAlmostEqual(data[data.row==0].phi_rad.max(),6*np.pi,places=12)

    def test_missing_or_duplicate_particle_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory),'missing',1)
            with h5py.File(path/'synthetic.h5','r+') as stream:
                stream['Step#10/id'][7] = 5
            with self.assertRaisesRegex(RuntimeError,'Missing, duplicate or lost'):
                school.read_trajectories(school.CASES[0],path)

    def test_each_machine_uses_its_own_native_input_without_parameter_replacement(self):
        for case in school.CASES:
            expected = evaluated_input(case.input_path.read_text())[1]
            actual = evaluated_input(school.lattice_text(case))[1]
            expected = {name:value for name,value in expected.items()
                        if value[0] in {'SCALINGFFAMAGNET','ASYMMETRIC_ENGE'}}
            self.assertEqual(actual, expected)
        self.assertNotEqual(school.CASES[0].input_path, school.CASES[1].input_path)

    def test_default_cof_is_exact_native_input_and_explicit_overrides_are_local(self):
        for case in school.CASES:
            source = case.input_path.read_text()
            expected = evaluated_input(source)
            with tempfile.TemporaryDirectory() as directory:
                default = school.prepare_cof(case, Path(directory)/'default')
                self.assertEqual((default/'school.in').read_text(), source)
                changed = school.prepare_cof(case, Path(directory)/'changed',
                                             dt=case.constants['COF_DT']/2,
                                             fd=case.constants['COF_FDSTEP']/2,
                                             guess_offset=(1e-3, 0., 0., 0.))
                actual = evaluated_input((changed/'school.in').read_text())
                self.assertEqual(actual[0]['COF_DT'], expected[0]['COF_DT']/2)
                self.assertEqual(actual[0]['COF_FDSTEP'], expected[0]['COF_FDSTEP']/2)
                self.assertAlmostEqual(actual[0]['COF_R_GUESS'], expected[0]['COF_R_GUESS']+1e-3)
                self.assertEqual(actual[1]['ORBIT'][1]['TIMEINTEGRATOR'], expected[1]['ORBIT'][1]['TIMEINTEGRATOR'])
                self.assertEqual(actual[0]['COF_MAXSTEPS'], expected[0]['COF_MAXSTEPS'])
                self.assertEqual(actual[2], expected[2])
                self.assertEqual(case.input_path.read_text(), source)

    def test_tracking_inherits_each_input_controls_and_closed_state(self):
        for case in school.CASES:
            result = dict(position_m=[-.02,0.,0.], momentum_mc=[.01,0.,case.p0])
            with tempfile.TemporaryDirectory() as directory:
                path = school.prepare_tracking(case, result, Path(directory)/'track', maps=False)
                actual = evaluated_input((path/'school.in').read_text())[0]
                for name in ['TRACK_DT', 'NUM_TURNS', 'DUMP_EVERY', 'TRACK_MAXSTEPS',
                             'COF_DT', 'COF_FDSTEP', 'STAT_DUMP_FREQ']:
                    self.assertEqual(actual[name], case.constants[name])
                particles = np.loadtxt(path/'particles.dat', skiprows=2)
                np.testing.assert_allclose(particles[:,[1,3,5]], np.tile(result['momentum_mc'],(8,1)))
                np.testing.assert_allclose(particles[0,[0,2,4]], result['position_m'])

    def test_native_constants_couple_energy_and_geometry(self):
        for case in school.CASES:
            text = school.lattice_text(case)
            constants,definitions,members = evaluated_input(text)
            self.assertAlmostEqual(constants['P0']/constants['P_MASS'],case.p0,places=13)
            if case.periods == 4:
                self.assertAlmostEqual(constants['SUPERPERIOD_ANGLE'],np.pi/2,places=13)
            scaled = school._set_constants(text,dict(R0=case.r0*1.1,E0=case.energy_MeV*1.2))
            new,new_definitions,new_members = evaluated_input(scaled)
            self.assertEqual(members,new_members)
            self.assertGreater(new['P0'],constants['P0'])
            for name,(kind,attrs) in definitions.items():
                if kind != 'SCALINGFFAMAGNET': continue
                for key in ['X','Z']:
                    self.assertAlmostEqual(new_definitions[name][1][key],1.1*attrs[key],places=12)
                self.assertAlmostEqual(new_definitions[name][1]['THETA'],attrs['THETA'],places=13)

    def test_wrapped_native_statements_keep_all_attributes(self):
        case = school.CASES[0]
        expected = school.lattice_text(case)
        source = (school.HERE/case.template).read_text()
        source = re.sub(r'//[^\n]*','',source)
        source = re.sub(r'=(?!=)', ' = ', source)
        wrapped = '\n'.join(textwrap.fill(statement.strip()+';',width=65,
                                          break_long_words=False,break_on_hyphens=False)
                            for statement in source.split(';') if statement.strip())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/case.template).parent.mkdir(parents=True)
            (root/case.template).write_text(wrapped)
            with patch.object(school,'HERE',root): actual = school.lattice_text(case)
        self.assertEqual(re.sub(r'\s+','',actual),re.sub(r'\s+','',expected))


    def test_docker_is_checked_and_pinned_independently_of_native_environment(self):
        image_info = [dict(Id='sha256:fixed-image', RepoDigests=['registry/opalx@sha256:fixed-digest'],
                           Architecture='arm64')]
        responses = [subprocess.CompletedProcess([],0,'linux\n',''),
                     subprocess.CompletedProcess([],0,json.dumps(image_info),''),
                     subprocess.CompletedProcess([],0,'OPAL Version 1.1.0, git rev. abcdef123456\n','')]
        with tempfile.TemporaryDirectory(prefix='school with spaces ') as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, {'OPALX_EXE':'/unwanted/local/opalx',
                                         'OPALX_LAUNCHER':'unwanted-launcher', 'OPALX_MPI_FLAGS':'--unwanted'}), \
                 patch.object(school.shutil,'which',return_value='/fake/docker'), \
                 patch.object(school.subprocess,'run',side_effect=responses) as run:
                execution = school.configure_execution(course_root=root)
            self.assertEqual(execution['backend'],'docker')
            self.assertEqual(execution['revision'],'abcdef123456')
            self.assertEqual(execution['image_digest'],'registry/opalx@sha256:fixed-digest')
            self.assertEqual(run.call_count,3)
            path = root/'nested run';path.mkdir()
            command = school.execution_command(path,execution)
            self.assertIn('sha256:fixed-image',command)
            self.assertNotIn(execution['image'],command)
            self.assertIn(f'type=bind,src={root},dst={root}',command)
            self.assertEqual(command[command.index('--workdir')+1],str(path))
            self.assertIn('OPALX_NP=1',command)
            self.assertIn('OMP_NUM_THREADS=1',command)
            self.assertIn('--pull=never',command)
            with self.assertRaisesRegex(ValueError,'inside the course root'):
                school.execution_command(root.parent,execution)
            # A run records the immutable selection, not just a mutable image tag.
            (path/'manifest.json').write_text('{}')
            with patch.object(school.subprocess,'run',return_value=subprocess.CompletedProcess([],0)):
                school.execute(path,execution)
            manifest = json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['execution'],execution)
            self.assertIn(execution['image_id'],manifest['command'])

    def test_docker_failure_never_falls_back_to_native_or_pulls(self):
        with patch.object(school.shutil,'which',return_value=None), \
             patch.object(school,'find_executable') as native:
            with self.assertRaisesRegex(FileNotFoundError,'Docker is unavailable'):
                school.configure_execution()
            native.assert_not_called()
        for failure,diagnostic in [(0,'Docker server is unavailable'),(1,'Docker image is unavailable')]:
            responses = [subprocess.CompletedProcess([],0,'linux','')]*failure
            responses += [subprocess.CalledProcessError(1,['docker'],stderr='unavailable')]
            with patch.object(school.shutil,'which',return_value='/fake/docker'), \
                 patch.object(school.subprocess,'run',side_effect=responses) as run, \
                 patch.object(school,'find_executable') as native:
                with self.assertRaisesRegex(RuntimeError,diagnostic):school.configure_execution()
                native.assert_not_called()
                self.assertFalse(any('pull' in c.args[0] for c in run.call_args_list))

    def test_native_mode_requires_explicit_executable(self):
        with patch.dict(os.environ, {'OPALX_EXE':sys.executable}):
            with self.assertRaisesRegex(ValueError,'explicit EXECUTABLE'):
                school.configure_execution(backend='native')
        with patch.dict(os.environ, {'OPALX_LAUNCHER':'', 'OPALX_MPI_FLAGS':''}), \
             patch.object(school.subprocess,'run',return_value=subprocess.CompletedProcess(
                 [],0,'OPAL Version 1.1.0, git rev. abcdef123456','')):
            execution = school.configure_execution(backend='native',executable=sys.executable)
        self.assertEqual(school.execution_command(school.HERE,execution)[0],str(Path(sys.executable).resolve()))
        self.assertNotIn('docker',execution)

    def test_docker_timeout_removes_only_the_current_named_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve();(root/'manifest.json').write_text('{}')
            execution = dict(backend='docker', docker='/fake/docker',course_root=str(root),image_id='sha256:fixed')
            with patch.object(school.subprocess,'run',side_effect=[
                    subprocess.TimeoutExpired(['docker'],1),subprocess.CompletedProcess([],0)]) as run:
                with self.assertRaisesRegex(RuntimeError,'timed out'):school.execute(root,execution,timeout=1)
            started = run.call_args_list[0].args[0]
            cleanup = run.call_args_list[1].args[0]
            name = started[started.index('--name')+1]
            self.assertTrue(name.startswith('ffa-school-run-'))
            self.assertEqual(cleanup,['/fake/docker','rm','-f',name])


    def test_cli_native_override_keeps_docker_notebook_and_labels_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'opalx/runs').mkdir(parents=True)
            for name in execute_notebook.LEGACY:(root/name).write_text('original')
            source=root/'opalx/FFA26_OPALX.ipynb'
            nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
                'BACKEND = "docker"\nIMAGE = "cpu"\nEXECUTABLE = None')])
            nbformat.write(nb,source);before=source.read_bytes()
            with patch.object(execute_notebook,'ROOT',root), \
                 patch.object(execute_notebook,'KernelManager') as manager, \
                 patch.object(execute_notebook,'NotebookClient'), \
                 patch.object(sys,'argv',['execute_notebook.py','--backend','native','--executable',sys.executable]):
                manager.return_value.has_kernel=False
                execute_notebook.main()
            self.assertEqual(source.read_bytes(),before)
            copy=nbformat.read(root/'opalx/runs/notebook-override-executed.ipynb',as_version=4)
            self.assertIn("BACKEND = 'native'",copy.cells[0].source)
            self.assertIn(str(Path(sys.executable).resolve()),copy.cells[0].source)


if __name__=='__main__': unittest.main()
