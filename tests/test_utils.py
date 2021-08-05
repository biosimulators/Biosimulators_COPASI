from biosimulators_copasi import data_model
from biosimulators_copasi import utils
from biosimulators_utils.data_model import ValueType
from kisao.exceptions import AlgorithmCannotBeSubstitutedException
from kisao.warnings import AlgorithmSubstitutedWarning
from unittest import mock
import COPASI
import os
import unittest


class UtilsTestCase(unittest.TestCase):
    def test_get_algorithm_id(self):
        self.assertEqual(utils.get_algorithm_id('KISAO_0000027'), ('KISAO_0000027', COPASI.CTaskEnum.Method_stochastic))

        self.assertEqual(utils.get_algorithm_id('KISAO_0000560'), ('KISAO_0000560', COPASI.CTaskEnum.Method_deterministic))

        with self.assertWarns(AlgorithmSubstitutedWarning):
            self.assertEqual(utils.get_algorithm_id('KISAO_0000088'), ('KISAO_0000560', COPASI.CTaskEnum.Method_deterministic))

        with self.assertWarns(AlgorithmSubstitutedWarning):
            self.assertEqual(utils.get_algorithm_id('KISAO_0000089'), ('KISAO_0000560', COPASI.CTaskEnum.Method_deterministic))

        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'NONE'}):
            with self.assertRaises(AlgorithmCannotBeSubstitutedException):
                utils.get_algorithm_id('KISAO_0000088')

        with self.assertRaises(AlgorithmCannotBeSubstitutedException):
            utils.get_algorithm_id('KISAO_0000450')

        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SAME_MATH'}):
            self.assertEqual(utils.get_algorithm_id('KISAO_0000561', events=False)[0], 'KISAO_0000561')
        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SIMILAR_APPROXIMATIONS'}):
            self.assertEqual(utils.get_algorithm_id('KISAO_0000561', events=False)[0], 'KISAO_0000561')
        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SAME_MATH'}):
            self.assertEqual(utils.get_algorithm_id('KISAO_0000561', events=True)[0], 'KISAO_0000561')
        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SIMILAR_APPROXIMATIONS'}):
            self.assertEqual(utils.get_algorithm_id('KISAO_0000561', events=True)[0], 'KISAO_0000563')

    def test_set_function_boolean_parameter(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_task = copasi_data_model.getTask('Time-Course')
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000560')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()
        parameter = method.getParameter('Integrate Reduced Model')

        utils.set_algorithm_parameter_value('KISAO_0000560', method, 'KISAO_0000216', 'true')
        self.assertEqual(parameter.getBoolValue(), True)

        utils.set_algorithm_parameter_value('KISAO_0000560', method, 'KISAO_0000216', '0')
        self.assertEqual(parameter.getBoolValue(), False)

    def test_set_function_integer_parameter(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_task = copasi_data_model.getTask('Time-Course')
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000027')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        utils.set_algorithm_parameter_value('KISAO_0000027', method, 'KISAO_0000415', '100')

        parameter = method.getParameter('Max Internal Steps')
        self.assertEqual(parameter.getIntValue(), 100)

    def test_set_function_float_parameter(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_task = copasi_data_model.getTask('Time-Course')
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000560')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        utils.set_algorithm_parameter_value('KISAO_0000560', method, 'KISAO_0000209', '100.1')

        parameter = method.getParameter('Relative Tolerance')
        self.assertEqual(parameter.getDblValue(), 100.1)

    def test_set_function_step_float_parameter(self):
        # KISAO_0000561
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_task = copasi_data_model.getTask('Time-Course')
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000561')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        utils.set_algorithm_parameter_value('KISAO_0000561', method, 'KISAO_0000483', '1e-12')

        parameter = method.getParameter('Runge Kutta Stepsize')
        self.assertEqual(parameter.getDblValue(), 1e-12)

        self.assertEqual(method.getParameter('Internal Steps Size'), None)

        # KISAO_0000566
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_task = copasi_data_model.getTask('Time-Course')
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000566')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        utils.set_algorithm_parameter_value('KISAO_0000566', method, 'KISAO_0000483', '1e-11')

        parameter = method.getParameter('Internal Steps Size')
        self.assertEqual(parameter.getDblValue(), 1e-11)

        self.assertEqual(method.getParameter('Runge Kutta Stepsize'), None)

    def test_set_function_seed_integer_parameter(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_task = copasi_data_model.getTask('Time-Course')
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000048')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        parameter = method.getParameter('Use Random Seed')
        assert(parameter.setBoolValue(False))
        self.assertFalse(parameter.getBoolValue())

        utils.set_algorithm_parameter_value('KISAO_0000048', method, 'KISAO_0000488', '90')

        parameter = method.getParameter('Random Seed')
        self.assertEqual(parameter.getIntValue(), 90)

        parameter = method.getParameter('Use Random Seed')
        self.assertTrue(parameter.getBoolValue())

    def test_set_function_partioning_list_parameter(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        assert copasi_data_model.importSBML(os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml'))
        copasi_task = copasi_data_model.getTask('Time-Course')
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000563')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        utils.set_algorithm_parameter_value('KISAO_0000563', method, 'KISAO_0000534', '["Reaction1", "Reaction2"]')

        parameter = method.getParameter('Partitioning Strategy')
        self.assertEqual(parameter.getStringValue(), 'User specified Partition')

        with self.assertRaises(ValueError):
            utils.set_algorithm_parameter_value('KISAO_0000563', method, 'KISAO_0000534', '["Rxn1", "Rxn2"]')

    def test_test_set_algorithm_parameter_value_errors(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        _, algorithm_id = utils.get_algorithm_id('KISAO_0000027')
        copasi_task = copasi_data_model.getTask('Time-Course')
        assert(copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        with self.assertRaises(NotImplementedError):
            utils.set_algorithm_parameter_value('KISAO_0000027', method, 'KISAO_0000000', '100')

        with self.assertRaises(NotImplementedError):
            utils.set_algorithm_parameter_value('KISAO_0000027', method, 'KISAO_0000216', '100')

        with self.assertRaises(ValueError):
            utils.set_algorithm_parameter_value('KISAO_0000027', method, 'KISAO_0000488', '100.')

    def test_all_parameters_for_all_algorithms(self):
        for parameter_kisao_id, param_props in data_model.KISAO_PARAMETERS_MAP.items():
            for algorithm_kisao_id in param_props['algorithms']:
                copasi_data_model = COPASI.CRootContainer.addDatamodel()
                _, algorithm_id = utils.get_algorithm_id(algorithm_kisao_id)
                copasi_task = copasi_data_model.getTask('Time-Course')
                assert(copasi_task.setMethodType(algorithm_id))
                method = copasi_task.getMethod()

                if param_props['type'] == ValueType.boolean:
                    value = 'true'
                elif param_props['type'] == ValueType.integer:
                    value = '2'
                elif param_props['type'] == ValueType.float:
                    value = '1e-8'
                else:
                    continue

                utils.set_algorithm_parameter_value(algorithm_kisao_id, method, parameter_kisao_id, value)

                if isinstance(param_props['name'], str):
                    parameter_name = param_props['name']
                else:
                    parameter_name = param_props['name'][algorithm_kisao_id]
                parameter = method.getParameter(parameter_name)

                if param_props['type'] == ValueType.boolean:
                    self.assertEqual(parameter.getBoolValue(), True)
                elif param_props['type'] == ValueType.integer:
                    self.assertEqual(parameter.getIntValue(), 2)
                elif param_props['type'] == ValueType.float:
                    self.assertEqual(parameter.getDblValue(), 1e-8)

    def test_get_copasi_model_object_by_sbml_id(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_data_model.importSBML(os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000806.xml'))
        copasi_model = copasi_data_model.getModel()

        object = utils.get_copasi_model_object_by_sbml_id(copasi_model, 'UnInfected_Tumour_Cells_Xu', data_model.Units.continuous)
        self.assertEqual(
            object.getCN().getString(),
            'CN=Root,Model=Eftimie2019-Macrophages Plasticity,Vector=Compartments[compartment],Vector=Metabolites[UnInfected_Tumour_Cells(Xu)],Reference=Concentration')

        object = utils.get_copasi_model_object_by_sbml_id(copasi_model, 'UnInfected_Tumour_Cells_Xu', data_model.Units.discrete)
        self.assertEqual(
            object.getCN().getString(),
            'CN=Root,Model=Eftimie2019-Macrophages Plasticity,Vector=Compartments[compartment],Vector=Metabolites[UnInfected_Tumour_Cells(Xu)],Reference=ParticleNumber')

        object = utils.get_copasi_model_object_by_sbml_id(copasi_model, 'compartment', data_model.Units.continuous)
        self.assertEqual(
            object.getCN().getString(),
            'CN=Root,Model=Eftimie2019-Macrophages Plasticity,Vector=Compartments[compartment],Reference=Volume')

        object = utils.get_copasi_model_object_by_sbml_id(
            copasi_model, 'Uninfected_tumour_cell_logistic_growth', data_model.Units.continuous)
        self.assertEqual(
            object.getCN().getString(),
            'CN=Root,Model=Eftimie2019-Macrophages Plasticity,Vector=Reactions[Uninfected tumour cell logistic growth],Reference=Flux')

    def test_get_copasi_model_obj_sbml_ids(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_data_model.importSBML(os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000806.xml'))
        copasi_model = copasi_data_model.getModel()

        ids = utils.get_copasi_model_obj_sbml_ids(copasi_model)
        self.assertEqual(len(ids), 6 + 32 + 25 + 1)
        self.assertIn('UnInfected_Tumour_Cells_Xu', ids)
