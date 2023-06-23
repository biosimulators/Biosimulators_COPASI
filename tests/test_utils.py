import basico

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
        self.assertEqual(utils.get_algorithm('KISAO_0000027'), data_model.GibsonBruckAlgorithm())

        self.assertEqual(utils.get_algorithm('KISAO_0000560'), data_model.LsodaAlgorithm())

        with self.assertWarns(AlgorithmSubstitutedWarning):
            self.assertEqual(utils.get_algorithm('KISAO_0000088'), data_model.LsodaAlgorithm())

        with self.assertWarns(AlgorithmSubstitutedWarning):
            self.assertEqual(utils.get_algorithm('KISAO_0000089'), data_model.LsodaAlgorithm())

        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'NONE'}):
            with self.assertRaises(AlgorithmCannotBeSubstitutedException):
                utils.get_algorithm('KISAO_0000088')

        with self.assertRaises(AlgorithmCannotBeSubstitutedException):
            utils.get_algorithm('KISAO_0000450')

        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SAME_MATH'}):
            self.assertEqual(utils.get_algorithm('KISAO_0000561', False).KISAO_ID, 'KISAO_0000561')
        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SIMILAR_APPROXIMATIONS'}):
            self.assertEqual(utils.get_algorithm('KISAO_0000561', False).KISAO_ID, 'KISAO_0000561')
        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SAME_MATH'}):
            with self.assertRaises(ValueError):
                utils.get_algorithm('KISAO_0000561', True)
        with mock.patch.dict(os.environ, {'ALGORITHM_SUBSTITUTION_POLICY': 'SIMILAR_APPROXIMATIONS'}):
            self.assertEqual(utils.get_algorithm('KISAO_0000561', True).KISAO_ID, 'KISAO_0000563')

    def test_set_function_boolean_parameter(self):
        state: bool = True
        alg = data_model.SDESolveRI5Algorithm(force_physical_correctness=state)

        # Create empty Data model
        basico.create_datamodel()

        # build map to change algorithm, and apply the change
        replacement_settings = {"method": alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check whether the change applied
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(alg.NAME, task_settings["method"]["name"])
        saved_value = alg.force_physical_correctness.get_value()
        self.assertEqual(saved_value, task_settings["method"][alg.force_physical_correctness.NAME])

    def test_set_function_integer_parameter(self):
        state: int = 2023
        alg = data_model.SDESolveRI5Algorithm(max_internal_steps=state)

        # Create empty Data model
        basico.create_datamodel()

        # build map to change algorithm, and apply the change
        replacement_settings = {"method": alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check whether the change applied
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(alg.NAME, task_settings["method"]["name"])
        saved_value = alg.max_internal_steps.get_value()
        self.assertEqual(saved_value, task_settings["method"][alg.max_internal_steps.NAME])

    def test_set_function_float_parameter(self):
        state: float = 22.43
        alg = data_model.SDESolveRI5Algorithm(step_size=state)

        # Create empty Data model
        basico.create_datamodel()

        # build map to change algorithm, and apply the change
        replacement_settings = {"method": alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check whether the change applied
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(alg.NAME, task_settings["method"]["name"])
        saved_value = alg.step_size.get_value()
        self.assertEqual(saved_value, task_settings["method"][alg.step_size.NAME])

    def test_set_function_step_float_parameter(self):
        runge_kutta_step_size: float = 1e-12
        internal_steps_size: float = 1e-11
        runge_kutta_alg = data_model.HybridRungeKuttaAlgorithm(step_size=runge_kutta_step_size)
        ri5_alg = data_model.SDESolveRI5Algorithm(step_size=internal_steps_size)

        # Create empty Data model
        basico.create_datamodel()

        # HybridRungeKuttaAlgorithm (KISAO_0000561)
        # build map to change algorithm, and apply the change
        replacement_settings = {"method": runge_kutta_alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check whether the change applied
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(runge_kutta_alg.NAME, task_settings["method"]["name"])
        saved_value = runge_kutta_alg.step_size.get_value()
        self.assertEqual(saved_value, task_settings["method"][runge_kutta_alg.step_size.NAME])
        with self.assertRaises(KeyError):
            value = task_settings["method"][ri5_alg.step_size.NAME]

        # SDESolveRI5Algorithm (KISAO_0000566)
        # build map to change algorithm, and apply the change
        replacement_settings = {"method": ri5_alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check whether the change applied
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(ri5_alg.NAME, task_settings["method"]["name"])
        saved_value = ri5_alg.step_size.get_value()
        self.assertEqual(saved_value, task_settings["method"][ri5_alg.step_size.NAME])
        with self.assertRaises(KeyError):
            value = task_settings["method"][runge_kutta_alg.step_size.NAME]

    def test_set_function_seed_integer_parameter(self):
        state: float = 90
        alg = data_model.AdaptiveSSATauLeapAlgorithm()

        # Create empty Data model
        basico.create_datamodel()

        # build map to change algorithm, and apply the change
        replacement_settings = {"method": alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check if we have the algorithm, and the default is not a random seed.
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(alg.NAME, task_settings["method"]["name"])
        self.assertFalse(task_settings["method"]["Use Random Seed"])

        # Now set the random seed parameter:
        alg.random_seed.set_value(state)
        replacement_settings = {"method": alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check whether the change applied
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(alg.random_seed.get_value(), task_settings["method"][alg.random_seed.NAME])

    def test_set_function_partitioning_list_parameter(self):
        state: float = 90
        alg = data_model.HybridRK45Algorithm()

        # Build model from sbml
        basico.load_model(os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml'))

        # build map to change algorithm, and apply the change
        replacement_settings = {"method": alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check whether the change applied
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(alg.random_seed.get_value(), task_settings["method"][alg.random_seed.NAME])

        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        assert copasi_data_model.importSBML(os.path.join(os.path.dirname(__file__), 'fixtures', 'model.xml'))
        copasi_task = copasi_data_model.getTask('Time-Course')
        algorithm_id = utils.get_algorithm_id('KISAO_0000563').copasi_algorithm_code
        assert (copasi_task.setMethodType(algorithm_id))
        method = copasi_task.getMethod()

        utils.set_algorithm_parameter_value('KISAO_0000563', method, 'KISAO_0000534', '["Reaction1", "Reaction2"]')

        parameter = method.getParameter('Partitioning Strategy')
        self.assertEqual(parameter.getStringValue(), 'User specified Partition')

        with self.assertRaises(ValueError):
            utils.set_algorithm_parameter_value('KISAO_0000563', method, 'KISAO_0000534', '["Rxn1", "Rxn2"]')

    def test_test_set_algorithm_parameter_value_errors(self):
        state: float = 90
        alg: data_model.GibsonBruckAlgorithm = data_model.GibsonBruckAlgorithm()

        # Create empty Data model
        basico.create_datamodel()

        # build map to change algorithm, and apply the change
        replacement_settings = {"method": alg.get_method_settings()}
        basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

        # Check if we have the algorithm.
        task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
        self.assertEqual(alg.NAME, task_settings["method"]["name"])

        # Now check for errors
        with self.assertRaises(AttributeError):
            alg: data_model.LsodaAlgorithm  # Fake cast
            alg.integrate_reduced_model.set_value(True)

        with self.assertRaises(ValueError):
            alg: data_model.GibsonBruckAlgorithm
            alg.random_seed.set_value(True)

    def test_all_parameters_for_all_algorithms(self):
        alg: data_model.GibsonBruckAlgorithm = data_model.GibsonBruckAlgorithm()

        basico.create_datamodel()
        member: data_model.CopasiAlgorithmType
        for name, member in data_model.CopasiAlgorithmType.__members__.items():
            alg = member.value()

            for param in alg.get_parameters_by_kisao().values():
                val_type = param.get_value_type(param)
                if val_type == bool:  # can't use isinstance because PEP 285.
                    param.set_value(True)
                elif val_type == int:  # can't use isinstance because PEP 285.
                    param.set_value(2)
                elif val_type == float:  # can't use isinstance because PEP 285.
                    param.set_value(1e-8)
                else:
                    continue

            replacement_settings = {"method": alg.get_method_settings()}
            basico.set_task_settings(basico.T.TIME_COURSE, replacement_settings)

            # confirm our settings applied
            task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
            self.assertEqual(alg.NAME, task_settings["method"]["name"])
            alg_params = list(alg.get_parameters_by_kisao().values())
            param_name_to_value = {param.NAME: param.get_value() for param in alg_params}

            for basico_param_name, basico_param_value in task_settings["method"].items():
                if basico_param_name == "name" or basico_param_name == "Subtype":
                    continue  # The above are not parameters to check
                if basico_param_name == "Use Random Seed" or basico_param_name == "Partitioning Strategy":
                    continue  # We automatically set this, and don't keep track of it.
                if basico_param_name not in param_name_to_value.keys():
                    raise NotImplementedError
                value = param_name_to_value[basico_param_name]
                if value != basico_param_value:
                    raise ValueError

    def test_get_copasi_model_object_by_sbml_id(self):
        species_concentration_id, species_concentration_cn = (
            "UnInfected_Tumour_Cells_Xu",
            'CN=Root,Model=Eftimie2019-Macrophages Plasticity,Vector=Compartments[compartment],'
            + 'Vector=Metabolites[UnInfected_Tumour_Cells(Xu)],Reference=Concentration')
        compartment_id, compartment_cn = (
            "compartment",
            'CN=Root,Model=Eftimie2019-Macrophages Plasticity,'
            + 'Vector=Compartments[compartment],Reference=Volume')
        reaction_id, reaction_cn = (
            "Uninfected_tumour_cell_logistic_growth",
            'CN=Root,Model=Eftimie2019-Macrophages Plasticity,'
            + 'Vector=Reactions[Uninfected tumour cell logistic growth],Reference=Flux')

        # load SBML
        basico.load_model(os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000806.xml'))
        species = basico.get_species(sbml_id=species_concentration_id)
        compartment = basico.get_compartments(sbml_id=compartment_id)
        reaction = basico.get_reactions(sbml_id=reaction_id)

        index: str = species.index.values[0]
        index_cn = basico.get_cn(utils.format_to_copasi_species_concentration_name(index))
        self.assertEqual(index_cn, species_concentration_cn)

        index: str = compartment.index.values[0]
        index_cn = basico.get_cn(utils.format_to_copasi_compartment_name(index))
        self.assertEqual(index_cn, compartment_cn)

        index: str = reaction.index.values[0]
        index_cn = basico.get_cn(utils.format_to_copasi_reaction_name(index))
        self.assertEqual(index_cn, reaction_cn)

    def test_get_copasi_model_obj_sbml_ids(self):
        copasi_data_model = COPASI.CRootContainer.addDatamodel()
        copasi_data_model.importSBML(os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000806.xml'))
        copasi_model = copasi_data_model.getModel()

        ids = utils.get_copasi_model_obj_sbml_ids(copasi_model)
        self.assertEqual(len(ids), 6 + 32 + 25 + 1)
        self.assertIn('UnInfected_Tumour_Cells_Xu', ids)

    def test_check_all_algorithm_args(self):
        basico.create_datamodel()
        algorithms = [data_model.GibsonBruckAlgorithm(),
                      data_model.DirectMethodAlgorithm(),
                      data_model.TauLeapAlgorithm(),
                      data_model.AdaptiveSSATauLeapAlgorithm(),
                      data_model.LsodaAlgorithm(),
                      data_model.Radau5Algorithm(),
                      data_model.HybridLsodaAlgorithm(),
                      data_model.HybridRungeKuttaAlgorithm(),
                      data_model.HybridRK45Algorithm(),
                      data_model.SDESolveRI5Algorithm()]

        for alg in algorithms:
            basico.set_task_settings(basico.T.TIME_COURSE, {"method": {"name": alg.NAME}})
            task_settings = basico.get_task_settings(basico.T.TIME_COURSE)
            self.assertTrue(task_settings["method"]["name"] == alg.NAME)
