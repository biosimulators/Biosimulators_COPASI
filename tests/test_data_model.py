from biosimulators_copasi import data_model
from biosimulators_utils.data_model import ValueType
import COPASI
import json
import os
import unittest


class DataModelTestCase(unittest.TestCase):
    def test_algorithms(self):
        for kisao_id, props in data_model.KISAO_ALGORITHMS_MAP.items():
            self.assertRegex(kisao_id, r'^KISAO_\d{7}$')
            self.assertIsInstance(props['name'], str)
            self.assertIsInstance(props['id'], str)
            self.assertIsInstance(getattr(COPASI.CTaskEnum, 'Method_' + props['id']), int)
            self.assertIsInstance(props['default_units'], data_model.Units)

    def test_parameters(self):
        for kisao_id, props in data_model.KISAO_PARAMETERS_MAP.items():
            self.assertRegex(kisao_id, r'^KISAO_\d{7}$')
            self.assertIsInstance(props['name'], (str, dict))
            if isinstance(props['name'], dict):
                for alg_kisao_id, name in props['name'].items():
                    self.assertIn(alg_kisao_id, data_model.KISAO_ALGORITHMS_MAP.keys())
                    self.assertIsInstance(name, str)
            self.assertIsInstance(props['type'], ValueType)
            for alg in props['algorithms']:
                self.assertIn(alg, data_model.KISAO_ALGORITHMS_MAP.keys())

    def test_data_model_matches_specifications(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'biosimulators.json'), 'r') as file:
            specs = json.load(file)

        self.assertEqual(
            set(data_model.KISAO_ALGORITHMS_MAP.keys()),
            set(alg_specs['kisaoId']['id'] for alg_specs in specs['algorithms']))

        alg_params = {}
        for param_kisao_id, param_props in data_model.KISAO_PARAMETERS_MAP.items():
            for alg_kisao_id in param_props['algorithms']:
                if alg_kisao_id not in alg_params:
                    alg_params[alg_kisao_id] = set()
                alg_params[alg_kisao_id].add((param_kisao_id, param_props['type'].value))

        for alg_specs in specs['algorithms']:
            alg_kisao_id = alg_specs['kisaoId']['id']
            alg_props = data_model.KISAO_ALGORITHMS_MAP[alg_kisao_id]

            param_spec_kisao_ids = set((param_specs['kisaoId']['id'], param_specs['type']) for param_specs in alg_specs['parameters'])

            self.assertEqual(alg_params[alg_kisao_id], param_spec_kisao_ids)
