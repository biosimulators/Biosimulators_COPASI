""" Tests of the COPASI command-line interface

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Teja <akhilmteja@gmail.com>
:Date: 2020-12-14
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from biosimulators_copasi import __main__
from biosimulators_copasi.core import exec_sed_task, exec_sedml_docs_in_combine_archive, preprocess_sed_task
from biosimulators_copasi.data_model import KISAO_ALGORITHMS_MAP
from biosimulators_utils.archive.io import ArchiveReader
from biosimulators_utils.combine import data_model as combine_data_model
from biosimulators_utils.combine.exceptions import CombineArchiveExecutionError
from biosimulators_utils.combine.io import CombineArchiveWriter
from biosimulators_utils.config import get_config
from biosimulators_utils.report import data_model as report_data_model
from biosimulators_utils.report.io import ReportReader
from biosimulators_utils.simulator.exec import exec_sedml_docs_in_archive_with_containerized_simulator
from biosimulators_utils.simulator.specs import gen_algorithms_from_specs
from biosimulators_utils.sedml import data_model as sedml_data_model
from biosimulators_utils.sedml.io import SedmlSimulationWriter
from biosimulators_utils.sedml.utils import append_all_nested_children_to_doc
from biosimulators_utils.utils.core import are_lists_equal
from biosimulators_utils.warnings import BioSimulatorsWarning
from unittest import mock
import COPASI
import copy
import datetime
import dateutil.tz
import json
import numpy
import numpy.testing
import os
import shutil
import tempfile
import unittest


class CliTestCase(unittest.TestCase):
    DOCKER_IMAGE = 'ghcr.io/biosimulators/biosimulators_copasi/copasi:latest'
    NAMESPACES_L2V4 = {
        'sbml': 'http://www.sbml.org/sbml/level2/version4',
    }
    NAMESPACES_L3V1 = {
        'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
    }

    def setUp(self):
        self.dirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def test_exec_sed_task(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000560',
                    changes=[
                        sedml_data_model.AlgorithmParameterChange(
                            kisao_id='KISAO_0000209',
                            new_value='2e-6',
                        ),
                    ],
                ),
                initial_time=0.,
                output_start_time=10.,
                output_end_time=20.,
                number_of_points=20,
            ),
        )

        variables = [
            sedml_data_model.Variable(
                id='time',
                symbol=sedml_data_model.Symbol.time,
                task=task),
            sedml_data_model.Variable(
                id='A',
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']",
                target_namespaces=self.NAMESPACES_L2V4,
                task=task),
            sedml_data_model.Variable(
                id='C',
                target='/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id="C"]',
                target_namespaces=self.NAMESPACES_L2V4,
                task=task),
            sedml_data_model.Variable(
                id='DA',
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='DA']",
                target_namespaces=self.NAMESPACES_L2V4,
                task=task),
        ]

        variable_results, _ = exec_sed_task(task, variables)

        self.assertTrue(sorted(variable_results.keys()), sorted([var.id for var in variables]))
        self.assertEqual(variable_results[variables[0].id].shape, (task.simulation.number_of_points + 1,))
        numpy.testing.assert_almost_equal(
            variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
        )

        for results in variable_results.values():
            self.assertFalse(numpy.any(numpy.isnan(results)))

        # initial_time = output_start_time = output_end_time
        task.simulation.initial_time = 0.
        task.simulation.output_start_time = 0.
        task.simulation.output_end_time = 0.
        task.simulation.number_of_points = 10
        variable_results, _ = exec_sed_task(task, variables)
        self.assertTrue(numpy.all(variable_results['time'] == 0.))
        for results in variable_results.values():
            self.assertEqual(results.shape, (task.simulation.number_of_points + 1,))
            self.assertFalse(numpy.any(numpy.isnan(results)))

    def test_exec_sed_task_record_parameters(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000806.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000560',
                ),
                initial_time=0.,
                output_start_time=0.,
                output_end_time=10.,
                number_of_points=10,
            ),
        )

        variables = [
            sedml_data_model.Variable(
                id='time',
                symbol=sedml_data_model.Symbol.time,
                task=task),
            sedml_data_model.Variable(
                id='r',
                target="/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='r']",
                target_namespaces=self.NAMESPACES_L3V1,
                task=task),
            sedml_data_model.Variable(
                id='d_u',
                target="/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='d_u']",
                target_namespaces=self.NAMESPACES_L3V1,
                task=task),
        ]

        variable_results, _ = exec_sed_task(task, variables)

        self.assertTrue(sorted(variable_results.keys()), sorted([var.id for var in variables]))
        self.assertEqual(variable_results[variables[0].id].shape, (task.simulation.number_of_points + 1,))
        numpy.testing.assert_almost_equal(
            variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
        )

        for results in variable_results.values():
            self.assertFalse(numpy.any(numpy.isnan(results)))

        self.assertTrue(numpy.all(variable_results['r'] == variable_results['r'][0]))

    def test_exec_sed_task_correct_time_course_attrs(self):
        # test that initial time, output start time, output end time, number of points are correctly interpreted
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join('tests', 'fixtures', 'model.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000560',
                    changes=[
                        sedml_data_model.AlgorithmParameterChange(
                            kisao_id='KISAO_0000209',
                            new_value='2e-6',
                        ),
                    ],
                ),
                initial_time=0.,
                output_start_time=0.,
                output_end_time=20.,
                number_of_points=20,
            ),
        )

        variables = [
            sedml_data_model.Variable(
                id='time',
                symbol=sedml_data_model.Symbol.time,
                task=task),
            sedml_data_model.Variable(
                id='A',
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']",
                target_namespaces=self.NAMESPACES_L2V4,
                task=task),
        ]

        # initial time == 0, output start time == initial time
        task.simulation.initial_time = 0.
        task.simulation.output_start_time = 0.
        task.simulation.output_end_time = 20.
        task.simulation.number_of_points = 20
        full_variable_results, _ = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(
            full_variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
            rtol=1e-4)

        # initial time == 0, output start time != initial time
        task.simulation.initial_time = 0.
        task.simulation.output_start_time = 10.
        task.simulation.output_end_time = 20.
        task.simulation.number_of_points = 10
        second_half_variable_results, _ = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(
            second_half_variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
            rtol=1e-4)
        numpy.testing.assert_allclose(second_half_variable_results['A'], full_variable_results['A'][10:], rtol=1e-4)

        # initial time != 0, output start time == initial time
        task.simulation.initial_time = 5.
        task.simulation.output_start_time = 5.
        task.simulation.output_end_time = 25.
        task.simulation.number_of_points = 20
        offset_full_variable_results, _ = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(
            offset_full_variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
            rtol=1e-4)
        numpy.testing.assert_allclose(offset_full_variable_results['A'], full_variable_results['A'], rtol=1e-4)

        # initial time != 0, output start time != initial time
        task.simulation.initial_time = 5.
        task.simulation.output_start_time = 15.
        task.simulation.output_end_time = 25.
        task.simulation.number_of_points = 10
        offset_second_half_variable_results, _ = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(
            offset_second_half_variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
            rtol=1e-4)
        numpy.testing.assert_allclose(offset_second_half_variable_results['A'], offset_full_variable_results['A'][10:], rtol=1e-4)
        numpy.testing.assert_allclose(offset_second_half_variable_results['A'], second_half_variable_results['A'], rtol=1e-4)

        # negative initial time
        task.simulation.initial_time = -5.
        task.simulation.output_start_time = 5.
        task.simulation.output_end_time = 15.
        task.simulation.number_of_points = 10
        offset_second_half_variable_results, _ = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(
            offset_second_half_variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
            rtol=1e-4)

    def test_exec_sed_task_correct_time_course_attrs_2(self):
        # test that initial time, output start time, output end time, number of points are correctly interpreted
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join('tests', 'fixtures', 'model.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000560',
                ),
                initial_time=0.4,
                output_start_time=0.4,
                output_end_time=0.8,
                number_of_points=5,
            ),
        )

        variables = [
            sedml_data_model.Variable(
                id='time',
                symbol=sedml_data_model.Symbol.time,
                task=task),
        ]

        results, _ = exec_sed_task(task, variables)

        numpy.testing.assert_allclose(results['time'], numpy.linspace(0.4, 0.8, 5 + 1))

    def test_hybrid_rk45_partitioning(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join('tests', 'fixtures', 'BIOMD0000000634_url.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000563',
                    changes=[
                        sedml_data_model.AlgorithmParameterChange(
                            kisao_id='KISAO_0000534',
                            new_value=json.dumps(['p53mRNASynthesis']),
                        ),
                    ],

                ),
                initial_time=0.,
                output_start_time=0.,
                output_end_time=1.,
                number_of_points=20,
            ),
        )

        variables = [
            sedml_data_model.Variable(
                id='time',
                symbol=sedml_data_model.Symbol.time,
                task=task),
            sedml_data_model.Variable(
                id='Mdm2',
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='Mdm2']",
                target_namespaces=self.NAMESPACES_L2V4,
                task=task),
        ]

        results, _ = exec_sed_task(task, variables)

        numpy.testing.assert_almost_equal(
            results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
        )
        self.assertFalse(numpy.any(numpy.isnan(results['Mdm2'])))

    def test_exec_sed_task_with_changes(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000560',
                ),
                initial_time=0.,
                output_start_time=0.,
                output_end_time=10.,
                number_of_points=10,
            ),
        )
        model = task.model
        sim = task.simulation

        variable_ids = ['EmptySet', 'A', 'C', 'DA', 'DAp', "DR", "DRp", "MA", "MR", "R"]
        variables = []
        for variable_id in variable_ids:
            model.changes.append(sedml_data_model.ModelAttributeChange(
                id=variable_id,
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{}']/@initialConcentration".format(variable_id),
                target_namespaces=self.NAMESPACES_L2V4,
                new_value=None,
            ))
            variables.append(sedml_data_model.Variable(
                id=variable_id,
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{}']".format(variable_id),
                target_namespaces=self.NAMESPACES_L2V4,
                task=task,
            ))
        preprocessed_task = preprocess_sed_task(task, variables)

        model.changes = []
        results, _ = exec_sed_task(task, variables, preprocessed_task=preprocessed_task)
        with self.assertRaises(AssertionError):
            for variable_id in variable_ids:
                self.assertEqual(
                    results[variable_id][0:int(sim.number_of_points / 2 + 1)].shape,
                    results[variable_id][-int(sim.number_of_points / 2 + 1):].shape,
                )
                numpy.testing.assert_allclose(
                    results[variable_id][0:int(sim.number_of_points / 2 + 1)],
                    results[variable_id][-int(sim.number_of_points / 2 + 1):])

        sim.output_end_time = sim.output_end_time / 2
        sim.number_of_points = int(sim.number_of_points / 2)
        results2, _ = exec_sed_task(task, variables, preprocessed_task=preprocessed_task)

        for variable_id in variable_ids:
            numpy.testing.assert_allclose(results2[variable_id], results[variable_id][0:sim.number_of_points + 1])

        for variable_id in variable_ids:
            model.changes.append(sedml_data_model.ModelAttributeChange(
                id=variable_id,
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{}']/@initialConcentration".format(variable_id),
                target_namespaces=self.NAMESPACES_L2V4,
                new_value=results2[variable_id][-1],
            ))

        results3, _ = exec_sed_task(task, variables, preprocessed_task=preprocessed_task)

        with self.assertRaises(AssertionError):
            for variable_id in variable_ids:
                numpy.testing.assert_allclose(results3[variable_id], results2[variable_id])

        for variable_id in variable_ids:
            numpy.testing.assert_allclose(results3[variable_id], results[variable_id][-(sim.number_of_points + 1):], rtol=5e-6)

        task.model.changes = [
            sedml_data_model.ModelAttributeChange(
                id="model_change",
                target="/sbml:sbml",
                target_namespaces=self.NAMESPACES_L2V4,
                new_value=None,
            ),
        ]
        with self.assertRaises(ValueError):
            preprocess_sed_task(task, variables)

    def test_exec_sed_task_with_changes_discrete(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000806.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000027',
                ),
                initial_time=0.,
                output_start_time=0.,
                output_end_time=10.,
                number_of_points=10,
            ),
        )
        model = task.model
        sim = task.simulation

        model.changes = [
            sedml_data_model.ModelAttributeChange(
                id='UnInfected_Tumour_Cells_Xu',
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{}']/@initialConcentration".format(
                    'UnInfected_Tumour_Cells_Xu'),
                target_namespaces=self.NAMESPACES_L3V1,
                new_value=None,
            ),
            sedml_data_model.ModelAttributeChange(
                id='r',
                target="/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='{}']/@initialConcentration".format('r'),
                target_namespaces=self.NAMESPACES_L3V1,
                new_value=None,
            ),
            sedml_data_model.ModelAttributeChange(
                id='compartment',
                target="/sbml:sbml/sbml:model/sbml:listOfCompartments/sbml:compartment[@id='{}']/@size".format('compartment'),
                target_namespaces=self.NAMESPACES_L3V1,
                new_value=None,
            ),
        ]

        variables = [
            sedml_data_model.Variable(
                id='UnInfected_Tumour_Cells_Xu',
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{}']".format('UnInfected_Tumour_Cells_Xu'),
                target_namespaces=self.NAMESPACES_L3V1,
                task=task,
            ),
        ]

        preprocessed_task = preprocess_sed_task(task, variables)

    def test_exec_sed_task_error_handling(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join(self.dirname, 'model.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000560',
                    changes=[
                        sedml_data_model.AlgorithmParameterChange(
                            kisao_id='KISAO_0000209',
                            new_value='2e-6',
                        ),
                    ],
                ),
                initial_time=0.,
                output_start_time=10.,
                output_end_time=20.,
                number_of_points=20,
            ),
        )

        variables = [
        ]

        with open(task.model.source, 'w') as file:
            file.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
            file.write('<sbml2 xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">')
            file.write('  <model id="model">')
            file.write('  </model>')
            file.write('</sbml2>')

        with mock.patch('biosimulators_utils.sedml.validation.validate_model_with_language', return_value=([], [], None)):
            with self.assertRaisesRegex(ValueError, 'could not be imported'):
                exec_sed_task(task, variables)

        task.model.source = os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml')
        with mock.patch.object(COPASI.CCopasiTask, 'initializeRawWithOutputHandler', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Output handler could not be initialized'):
                exec_sed_task(task, variables)

        task.simulation.output_end_time = 20.1
        with self.assertRaisesRegex(NotImplementedError, 'integer number of time points'):
            exec_sed_task(task, variables)

        task.simulation.output_start_time = 20.
        task.simulation.output_end_time = 20.
        with self.assertRaisesRegex(NotImplementedError, 'must be greater'):
            exec_sed_task(task, variables)

        task.simulation.output_start_time = 10.
        variables = [
            sedml_data_model.Variable(
                id='time',
                symbol='urn:sedml:symbol:undefined',
                task=task),
            sedml_data_model.Variable(
                id='A',
                target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']",
                target_namespaces=self.NAMESPACES_L2V4,
                task=task),
        ]
        with self.assertRaisesRegex(NotImplementedError, 'symbols are not supported'):
            exec_sed_task(task, variables)

        variables = [
            sedml_data_model.Variable(
                id='A',
                target="/sbml:sbml/sbml:model",
                target_namespaces=self.NAMESPACES_L2V4,
                task=task),
        ]
        with self.assertRaisesRegex(ValueError, 'targets cannot be recorded'):
            exec_sed_task(task, variables)

        variables = []
        task.simulation.algorithm.changes[0].new_value = 'adf'
        with mock.patch.dict('os.environ', {'ALGORITHM_SUBSTITUTION_POLICY': 'NONE'}):
            with self.assertRaisesRegex(ValueError, 'is not a valid'):
                exec_sed_task(task, variables)

        with mock.patch.dict('os.environ', {'ALGORITHM_SUBSTITUTION_POLICY': 'SIMILAR_VARIABLES'}):
            with self.assertWarnsRegex(BioSimulatorsWarning, 'Unsuported value'):
                exec_sed_task(task, variables)

        task.simulation.algorithm.changes[0].kisao_id = 'KISAO_0000531'
        with mock.patch.dict('os.environ', {'ALGORITHM_SUBSTITUTION_POLICY': 'NONE'}):
            with self.assertRaisesRegex(NotImplementedError, 'is not supported'):
                exec_sed_task(task, variables)

        with mock.patch.dict('os.environ', {'ALGORITHM_SUBSTITUTION_POLICY': 'SIMILAR_VARIABLES'}):
            with self.assertWarnsRegex(BioSimulatorsWarning, 'Unsuported algorithm parameter'):
                exec_sed_task(task, variables)

    def test_exec_sed_task_copasi_error_handling(self):
        alg = sedml_data_model.Algorithm(kisao_id='KISAO_0000304')
        doc, archive_filename = self._build_combine_archive(algorithm=alg,
                                                            orig_model_filename='BIOMD0000000634_url.xml',
                                                            var_targets=[None, 'Mdm2', 'p53', 'Mdm2_p53'])

        out_dir = os.path.join(self.dirname, alg.kisao_id)

        task = copy.deepcopy(doc.tasks[0])
        task.model.source = os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000634_url.xml')
        task.simulation.algorithm.kisao_id = 'KISAO_0000560'
        variables = [data_set.data_generator.variables[0] for data_set in doc.outputs[0].data_sets]
        _, log = exec_sed_task(task, variables)
        self.assertEqual(log.algorithm, 'KISAO_0000560')

        task = copy.deepcopy(doc.tasks[0])
        task.model.source = os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000634_url.xml')
        task.simulation.algorithm.kisao_id = 'KISAO_0000304'
        variables = [data_set.data_generator.variables[0] for data_set in doc.outputs[0].data_sets]
        _, log = exec_sed_task(task, variables)
        self.assertNotEqual(log.algorithm, 'KISAO_0000304')

        task = copy.deepcopy(doc.tasks[0])
        task.model.source = os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml')
        task.simulation.algorithm.kisao_id = 'KISAO_0000304'
        variables = [sedml_data_model.Variable(id='time', task=task, symbol=sedml_data_model.Symbol.time.value)]
        _, log = exec_sed_task(task, variables)
        self.assertEqual(log.algorithm, 'KISAO_0000304')

        config = get_config()
        config.REPORT_FORMATS = [
            report_data_model.ReportFormat.h5,
            report_data_model.ReportFormat.csv,
        ]
        config.BUNDLE_OUTPUTS = True
        config.KEEP_INDIVIDUAL_OUTPUTS = True

        with self.assertRaises(CombineArchiveExecutionError):
            with mock.patch.object(COPASI.CCopasiTask, 'processRaw', return_value=False):
                _, log = exec_sedml_docs_in_combine_archive(archive_filename, out_dir, config=config)
            if log.exception:
                raise log.exception

    def test_exec_sedml_docs_in_combine_archive(self):
        doc, archive_filename = self._build_combine_archive()

        out_dir = os.path.join(self.dirname, 'out')

        config = get_config()
        config.REPORT_FORMATS = [
            report_data_model.ReportFormat.h5,
            report_data_model.ReportFormat.csv,
        ]
        config.BUNDLE_OUTPUTS = True
        config.KEEP_INDIVIDUAL_OUTPUTS = True

        _, log = exec_sedml_docs_in_combine_archive(archive_filename, out_dir, config=config)
        if log.exception:
            raise log.exception

        self._assert_combine_archive_outputs(doc, out_dir)

    def test_exec_sedml_docs_in_combine_archive_with_continuous_model_all_algorithms(self):
        # continuous model
        errored_algs = []
        for alg in gen_algorithms_from_specs(os.path.join(os.path.dirname(__file__), '..', 'biosimulators.json')).values():
            doc, archive_filename = self._build_combine_archive(algorithm=alg,
                                                                orig_model_filename='model.xml',
                                                                var_targets=[None, 'A', 'C', 'DA'])

            out_dir = os.path.join(self.dirname, alg.kisao_id)

            config = get_config()
            config.REPORT_FORMATS = [
                report_data_model.ReportFormat.h5,
                report_data_model.ReportFormat.csv,
            ]
            config.BUNDLE_OUTPUTS = True
            config.KEEP_INDIVIDUAL_OUTPUTS = True

            try:
                _, log = exec_sedml_docs_in_combine_archive(archive_filename, out_dir, config=config)
                if log.exception:
                    raise log.exception
                self._assert_combine_archive_outputs(doc, out_dir)
            except CombineArchiveExecutionError:
                errored_algs.append(alg.kisao_id)

        # fail because particle number too big for discrete methods
        self.assertEqual(sorted(errored_algs), sorted([
            'KISAO_0000027',
            'KISAO_0000029',
            'KISAO_0000039',
            'KISAO_0000048',
            'KISAO_0000561',
            'KISAO_0000562',
            'KISAO_0000563',
        ]))

    def test_exec_sedml_docs_in_combine_archive_with_stochastic_model_all_algorithms(self):
        # discrete/continuous model
        for alg in gen_algorithms_from_specs(os.path.join(os.path.dirname(__file__), '..', 'biosimulators.json')).values():
            doc, archive_filename = self._build_combine_archive(algorithm=alg,
                                                                orig_model_filename='BIOMD0000000634_url.xml',
                                                                var_targets=[None, 'Mdm2', 'p53', 'Mdm2_p53'])

            out_dir = os.path.join(self.dirname, alg.kisao_id)

            config = get_config()
            config.REPORT_FORMATS = [
                report_data_model.ReportFormat.h5,
                report_data_model.ReportFormat.csv,
            ]
            config.BUNDLE_OUTPUTS = True
            config.KEEP_INDIVIDUAL_OUTPUTS = True

            _, log = exec_sedml_docs_in_combine_archive(archive_filename, out_dir, config=config)
            if log.exception:
                raise log.exception
            self._assert_combine_archive_outputs(doc, out_dir)

    def _build_combine_archive(self, algorithm=None, orig_model_filename='model.xml', var_targets=[None, 'A', 'C', 'DA']):
        doc = self._build_sed_doc(algorithm=algorithm)

        for data_gen, target in zip(doc.data_generators, var_targets):
            if target is not None:
                data_gen.variables[0].target = "/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{}']".format(target)

        archive_dirname = os.path.join(self.dirname, 'archive')
        if not os.path.isdir(archive_dirname):
            os.mkdir(archive_dirname)

        model_filename = os.path.join(archive_dirname, 'model_1.xml')
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), 'fixtures', orig_model_filename),
            model_filename)

        sim_filename = os.path.join(archive_dirname, 'sim_1.sedml')
        SedmlSimulationWriter().run(doc, sim_filename)

        archive = combine_data_model.CombineArchive(
            contents=[
                combine_data_model.CombineArchiveContent(
                    'model_1.xml', combine_data_model.CombineArchiveContentFormat.SBML.value),
                combine_data_model.CombineArchiveContent(
                    'sim_1.sedml', combine_data_model.CombineArchiveContentFormat.SED_ML.value),
            ],
        )
        archive_filename = os.path.join(self.dirname,
                                        'archive.omex' if algorithm is None else 'archive-{}.omex'.format(algorithm.kisao_id))
        CombineArchiveWriter().run(archive, archive_dirname, archive_filename)

        return (doc, archive_filename)

    def _build_sed_doc(self, algorithm=None):
        if algorithm is None:
            algorithm = sedml_data_model.Algorithm(
                kisao_id='KISAO_0000304',
                changes=[
                    sedml_data_model.AlgorithmParameterChange(
                        kisao_id='KISAO_0000209',
                        new_value='0.0002',
                    ),
                ],
            )

        doc = sedml_data_model.SedDocument()
        doc.models.append(sedml_data_model.Model(
            id='model_1',
            source='model_1.xml',
            language=sedml_data_model.ModelLanguage.SBML.value,
            changes=[],
        ))
        doc.simulations.append(sedml_data_model.UniformTimeCourseSimulation(
            id='sim_1_time_course',
            algorithm=algorithm,
            initial_time=0.,
            output_start_time=0.1,
            output_end_time=0.2,
            number_of_points=20,
        ))
        doc.tasks.append(sedml_data_model.Task(
            id='task_1',
            model=doc.models[0],
            simulation=doc.simulations[0],
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_time',
            variables=[
                sedml_data_model.Variable(
                    id='var_time',
                    symbol=sedml_data_model.Symbol.time,
                    task=doc.tasks[0],
                ),
            ],
            math='var_time',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_A',
            variables=[
                sedml_data_model.Variable(
                    id='var_A',
                    target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@name='A']",
                    target_namespaces=self.NAMESPACES_L2V4,
                    task=doc.tasks[0],
                ),
            ],
            math='var_A',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_C',
            variables=[
                sedml_data_model.Variable(
                    id='var_C',
                    target='/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@name="C"]',
                    target_namespaces=self.NAMESPACES_L2V4,
                    task=doc.tasks[0],
                ),
            ],
            math='var_C',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_DA',
            variables=[
                sedml_data_model.Variable(
                    id='var_DA',
                    target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='DA']",
                    target_namespaces=self.NAMESPACES_L2V4,
                    task=doc.tasks[0],
                ),
            ],
            math='var_DA',
        ))
        doc.outputs.append(sedml_data_model.Report(
            id='report_1',
            data_sets=[
                sedml_data_model.DataSet(id='data_set_time', label='Time', data_generator=doc.data_generators[0]),
                sedml_data_model.DataSet(id='data_set_A', label='A', data_generator=doc.data_generators[1]),
                sedml_data_model.DataSet(id='data_set_C', label='C', data_generator=doc.data_generators[2]),
                sedml_data_model.DataSet(id='data_set_DA', label='DA', data_generator=doc.data_generators[3]),
            ],
        ))

        append_all_nested_children_to_doc(doc)

        return doc

    def _assert_combine_archive_outputs(self, doc, out_dir):
        self.assertEqual(set(['reports.h5', 'reports.zip', 'sim_1.sedml']).difference(set(os.listdir(out_dir))), set())

        report = doc.outputs[0]

        # check HDF report
        report_results = ReportReader().run(report, out_dir, 'sim_1.sedml/report_1', format=report_data_model.ReportFormat.h5)

        self.assertEqual(sorted(report_results.keys()), sorted([d.id for d in doc.outputs[0].data_sets]))

        sim = doc.tasks[0].simulation
        self.assertEqual(len(report_results[report.data_sets[0].id]), sim.number_of_points + 1)
        numpy.testing.assert_almost_equal(
            report_results['data_set_time'],
            numpy.linspace(sim.output_start_time, sim.output_end_time, sim.number_of_points + 1),
        )

        for data_set_result in report_results.values():
            self.assertFalse(numpy.any(numpy.isnan(data_set_result)))

        # check CSV report
        report_results = ReportReader().run(report, out_dir, 'sim_1.sedml/report_1', format=report_data_model.ReportFormat.csv)

        self.assertEqual(sorted(report_results.keys()), sorted([d.id for d in doc.outputs[0].data_sets]))

        sim = doc.tasks[0].simulation
        self.assertEqual(len(report_results[report.data_sets[0].id]), sim.number_of_points + 1)
        numpy.testing.assert_almost_equal(
            report_results['data_set_time'],
            numpy.linspace(sim.output_start_time, sim.output_end_time, sim.number_of_points + 1),
        )

        for data_set_result in report_results.values():
            self.assertFalse(numpy.any(numpy.isnan(data_set_result)))

    def test_exec_sedml_docs_in_combine_archive_real_example(self):
        archive_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'Ciliberto-J-Cell-Biol-2003-morphogenesis-checkpoint.omex')
        out_dir = os.path.join(self.dirname, 'out')

        config = get_config()
        config.REPORT_FORMATS = [
            report_data_model.ReportFormat.h5,
        ]
        config.BUNDLE_OUTPUTS = True
        config.KEEP_INDIVIDUAL_OUTPUTS = False

        _, log = exec_sedml_docs_in_combine_archive(archive_filename, out_dir, config=config)
        if log.exception:
            raise log.exception

        report = sedml_data_model.Report(
            data_sets=[
                sedml_data_model.DataSet(id='time', label='time'),
                sedml_data_model.DataSet(id='Cdh1', label='Cdh1'),
                sedml_data_model.DataSet(id='Trim', label='Trim'),
                sedml_data_model.DataSet(id='Clb', label='Clb'),
            ]
        )
        report_results = ReportReader().run(report, out_dir, 'simulation_1.sedml/report_1', format=report_data_model.ReportFormat.h5)

        self.assertEqual(len(report_results[report.data_sets[0].id]), 100 + 1)
        numpy.testing.assert_almost_equal(
            report_results['time'],
            numpy.linspace(0., 100., 100 + 1),
        )

        for data_set_result in report_results.values():
            self.assertFalse(numpy.any(numpy.isnan(data_set_result)))

    def test_raw_cli(self):
        with mock.patch('sys.argv', ['', '--help']):
            with self.assertRaises(SystemExit) as context:
                __main__.main()
                self.assertRegex(context.Exception, 'usage: ')

    def test_exec_sedml_docs_in_combine_archive_with_cli(self):
        doc, archive_filename = self._build_combine_archive()
        out_dir = os.path.join(self.dirname, 'out')
        env = self._get_combine_archive_exec_env()

        with mock.patch.dict(os.environ, env):
            with __main__.App(argv=['-i', archive_filename, '-o', out_dir]) as app:
                app.run()

        self._assert_combine_archive_outputs(doc, out_dir)

    def _get_combine_archive_exec_env(self):
        return {
            'REPORT_FORMATS': 'h5,csv'
        }

    def test_exec_sedml_docs_in_combine_archive_with_docker_image(self):
        doc, archive_filename = self._build_combine_archive()
        out_dir = os.path.join(self.dirname, 'out')
        docker_image = self.DOCKER_IMAGE
        env = self._get_combine_archive_exec_env()

        exec_sedml_docs_in_archive_with_containerized_simulator(
            archive_filename, out_dir, docker_image, environment=env, pull_docker_image=False)

        self._assert_combine_archive_outputs(doc, out_dir)
