"""Reader/identity regressions with analytic synthetic HDF5, no OPALX required."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import textwrap

import h5py
import numpy as np

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


if __name__=='__main__': unittest.main()
