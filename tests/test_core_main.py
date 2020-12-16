""" Tests of the COPASI command-line interface

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Teja <akhilmteja@gmail.com>
:Date: 2020-12-14
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from biosimulators_copasi import __main__
from biosimulators_copasi.core import exec_sed_task, exec_sedml_docs_in_combine_archive
from biosimulators_copasi.data_model import KISAO_ALGORITHMS_MAP
from biosimulators_utils.archive.io import ArchiveReader
from biosimulators_utils.combine import data_model as combine_data_model
from biosimulators_utils.combine.io import CombineArchiveWriter
from biosimulators_utils.report import data_model as report_data_model
from biosimulators_utils.report.io import ReportReader
from biosimulators_utils.simulator.exec import exec_sedml_docs_in_archive_with_containerized_simulator
from biosimulators_utils.simulator.specs import gen_algorithms_from_specs
from biosimulators_utils.sedml import data_model as sedml_data_model
from biosimulators_utils.sedml.io import SedmlSimulationWriter
from biosimulators_utils.sedml.utils import append_all_nested_children_to_doc
from biosimulators_utils.utils.core import are_lists_equal
from unittest import mock
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
            sedml_data_model.DataGeneratorVariable(id='time', symbol=sedml_data_model.DataGeneratorVariableSymbol.time),
            sedml_data_model.DataGeneratorVariable(id='A', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']"),
            sedml_data_model.DataGeneratorVariable(id='C', target='/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id="C"]'),
            sedml_data_model.DataGeneratorVariable(id='DA', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='DA']"),
        ]

        variable_results = exec_sed_task(task, variables)

        self.assertTrue(sorted(variable_results.keys()), sorted([var.id for var in variables]))
        self.assertEqual(variable_results[variables[0].id].shape, (task.simulation.number_of_points + 1,))
        numpy.testing.assert_almost_equal(
            variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
        )

        for results in variable_results.values():
            self.assertFalse(numpy.any(numpy.isnan(results)))

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
            sedml_data_model.DataGeneratorVariable(id='time', symbol=sedml_data_model.DataGeneratorVariableSymbol.time),
            sedml_data_model.DataGeneratorVariable(id='A', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']"),
        ]

        task.simulation.initial_time = 0.
        task.simulation.output_start_time = 0.
        task.simulation.output_end_time = 20.
        task.simulation.number_of_points = 20
        full_variable_results = exec_sed_task(task, variables)

        task.simulation.initial_time = 0.
        task.simulation.output_start_time = 10.
        task.simulation.output_end_time = 20.
        task.simulation.number_of_points = 10
        second_half_variable_results = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(second_half_variable_results['A'], full_variable_results['A'][10:], rtol=1e-4)

        task.simulation.initial_time = 5.
        task.simulation.output_start_time = 5.
        task.simulation.output_end_time = 25.
        task.simulation.number_of_points = 20
        offset_full_variable_results = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(offset_full_variable_results['A'], full_variable_results['A'], rtol=1e-4)

        task.simulation.initial_time = 5.
        task.simulation.output_start_time = 15.
        task.simulation.output_end_time = 25.
        task.simulation.number_of_points = 10
        offset_second_half_variable_results = exec_sed_task(task, variables)
        numpy.testing.assert_allclose(offset_second_half_variable_results['A'], offset_full_variable_results['A'][10:], rtol=1e-4)
        numpy.testing.assert_allclose(offset_second_half_variable_results['A'], second_half_variable_results['A'], rtol=1e-4)

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
            sedml_data_model.DataGeneratorVariable(id='time', symbol=sedml_data_model.DataGeneratorVariableSymbol.time),
            sedml_data_model.DataGeneratorVariable(id='Mdm2', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='Mdm2']"),
        ]

        results = exec_sed_task(task, variables)

        numpy.testing.assert_almost_equal(
            results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
        )
        self.assertFalse(numpy.any(numpy.isnan(results['Mdm2'])))

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

        with self.assertRaisesRegex(ValueError, 'could not be imported'):
            exec_sed_task(task, variables)

        task.model.source = os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml')
        task.simulation.output_end_time = 20.1
        with self.assertRaisesRegex(NotImplementedError, 'integer number of time points'):
            exec_sed_task(task, variables)

        task.simulation.output_end_time = 20.
        variables = [
            sedml_data_model.DataGeneratorVariable(id='time', symbol='urn:sedml:symbol:undefined'),
            sedml_data_model.DataGeneratorVariable(id='A', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']"),
        ]
        with self.assertRaisesRegex(NotImplementedError, 'symbols are not supported'):
            exec_sed_task(task, variables)

        variables = [
            sedml_data_model.DataGeneratorVariable(
                id='A', target="/sbml:sbml/sbml:model/sbml:listOfReactions/sbml:reaction[@id='Reaction1']"),
        ]
        with self.assertRaisesRegex(ValueError, 'targets could not be recorded'):
            exec_sed_task(task, variables)

    def test_exec_sedml_docs_in_combine_archive(self):
        doc, archive_filename = self._build_combine_archive()

        out_dir = os.path.join(self.dirname, 'out')
        exec_sedml_docs_in_combine_archive(archive_filename, out_dir,
                                           report_formats=[
                                               report_data_model.ReportFormat.h5,
                                               report_data_model.ReportFormat.csv,
                                           ],
                                           bundle_outputs=True,
                                           keep_individual_outputs=True)

        self._assert_combine_archive_outputs(doc, out_dir)

    def test_exec_sedml_docs_in_combine_archive_with_continuous_model_all_algorithms(self):
        # continuous model
        errored_algs = []
        for alg in gen_algorithms_from_specs(os.path.join(os.path.dirname(__file__), '..', 'biosimulators.json')).values():
            doc, archive_filename = self._build_combine_archive(algorithm=alg,
                                                                orig_model_filename='model.xml',
                                                                var_targets=[None, 'A', 'C', 'DA'])

            out_dir = os.path.join(self.dirname, alg.kisao_id)
            try:
                exec_sedml_docs_in_combine_archive(archive_filename, out_dir,
                                                   report_formats=[
                                                       report_data_model.ReportFormat.h5,
                                                       report_data_model.ReportFormat.csv,
                                                   ],
                                                   bundle_outputs=True,
                                                   keep_individual_outputs=True)
                self._assert_combine_archive_outputs(doc, out_dir)
            except RuntimeError:
                errored_algs.append(alg.kisao_id)

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
        errored_algs = []
        for alg in gen_algorithms_from_specs(os.path.join(os.path.dirname(__file__), '..', 'biosimulators.json')).values():
            doc, archive_filename = self._build_combine_archive(algorithm=alg,
                                                                orig_model_filename='BIOMD0000000634_url.xml',
                                                                var_targets=[None, 'Mdm2', 'p53', 'Mdm2_p53'])

            out_dir = os.path.join(self.dirname, alg.kisao_id)
            try:
                exec_sedml_docs_in_combine_archive(archive_filename, out_dir,
                                                   report_formats=[
                                                       report_data_model.ReportFormat.h5,
                                                       report_data_model.ReportFormat.csv,
                                                   ],
                                                   bundle_outputs=True,
                                                   keep_individual_outputs=True)
                self._assert_combine_archive_outputs(doc, out_dir)
            except RuntimeError:
                errored_algs.append(alg.kisao_id)

        self.assertEqual(sorted(errored_algs), sorted([
            'KISAO_0000039',
            'KISAO_0000304',
            'KISAO_0000561',
            'KISAO_0000562'
        ]))

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

        updated = datetime.datetime(2020, 1, 2, 1, 2, 3, tzinfo=dateutil.tz.tzutc())
        archive = combine_data_model.CombineArchive(
            contents=[
                combine_data_model.CombineArchiveContent(
                    'model_1.xml', combine_data_model.CombineArchiveContentFormat.SBML.value, updated=updated),
                combine_data_model.CombineArchiveContent(
                    'sim_1.sedml', combine_data_model.CombineArchiveContentFormat.SED_ML.value, updated=updated),
            ],
            updated=updated,
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
                sedml_data_model.DataGeneratorVariable(
                    id='var_time',
                    symbol=sedml_data_model.DataGeneratorVariableSymbol.time,
                    task=doc.tasks[0],
                    model=doc.models[0],
                ),
            ],
            math='var_time',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_A',
            variables=[
                sedml_data_model.DataGeneratorVariable(
                    id='var_A',
                    target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@name='A']",
                    task=doc.tasks[0],
                    model=doc.models[0],
                ),
            ],
            math='var_A',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_C',
            variables=[
                sedml_data_model.DataGeneratorVariable(
                    id='var_C',
                    target='/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@name="C"]',
                    task=doc.tasks[0],
                    model=doc.models[0],
                ),
            ],
            math='var_C',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_DA',
            variables=[
                sedml_data_model.DataGeneratorVariable(
                    id='var_DA',
                    target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='DA']",
                    task=doc.tasks[0],
                    model=doc.models[0],
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
        self.assertEqual(set(os.listdir(out_dir)), set(['reports.h5', 'reports.zip', 'sim_1.sedml']))

        # check HDF report
        report = ReportReader().run(out_dir, 'sim_1.sedml/report_1', format=report_data_model.ReportFormat.h5)

        self.assertEqual(sorted(report.index), sorted([d.id for d in doc.outputs[0].data_sets]))

        sim = doc.tasks[0].simulation
        self.assertEqual(report.shape, (len(doc.outputs[0].data_sets), sim.number_of_points + 1))
        numpy.testing.assert_almost_equal(
            report.loc['data_set_time', :].to_numpy(),
            numpy.linspace(sim.output_start_time, sim.output_end_time, sim.number_of_points + 1),
        )

        self.assertFalse(numpy.any(numpy.isnan(report)))

        # check CSV report
        report = ReportReader().run(out_dir, 'sim_1.sedml/report_1', format=report_data_model.ReportFormat.csv)

        self.assertEqual(sorted(report.index), sorted([d.id for d in doc.outputs[0].data_sets]))

        sim = doc.tasks[0].simulation
        self.assertEqual(report.shape, (len(doc.outputs[0].data_sets), sim.number_of_points + 1))
        numpy.testing.assert_almost_equal(
            report.loc['data_set_time', :].to_numpy(),
            numpy.linspace(sim.output_start_time, sim.output_end_time, sim.number_of_points + 1),
        )

        self.assertFalse(numpy.any(numpy.isnan(report)))

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
