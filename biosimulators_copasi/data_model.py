""" Data model for maps from KiSAO terms to COPASI algorithms and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from biosimulators_utils.data_model import ValueType
import collections
import enum

__all__ = ['Units', 'KISAO_ALGORITHMS_MAP', 'KISAO_PARAMETERS_MAP']


class Units(str, enum.Enum):
    """ Function for getting simulation results """
    discrete = 'discrete'
    continuous = 'continuous'


KISAO_ALGORITHMS_MAP = collections.OrderedDict([
    ('KISAO_0000027', {
        'name': 'Gibson + Bruck',
        'id': 'stochastic',
        'default_units': Units.discrete,
        'supports_events': True,
    }),
    ('KISAO_0000029', {
        'name': 'direct method',
        'id': 'directMethod',
        'default_units': Units.discrete,
        'supports_events': True,
    }),
    ('KISAO_0000039', {
        'name': 'tau leap method',
        'id': 'tauLeap',
        'default_units': Units.discrete,
        'supports_events': False,
    }),
    ('KISAO_0000048', {
        'name': 'adaptive SSA + tau leap',
        'id': 'adaptiveSA',
        'default_units': Units.discrete,
        'supports_events': True,
    }),
    ('KISAO_0000560', {
        'name': 'LSODA/LSODAR',
        'id': 'deterministic',
        'default_units': Units.continuous,
        'supports_events': True,
    }),
    ('KISAO_0000304', {
        'name': 'RADAU5',
        'id': 'RADAU5',
        'default_units': Units.continuous,
        'supports_events': False,
    }),
    ('KISAO_0000562', {
        'name': 'hybrid (lsoda)',
        'id': 'hybridLSODA',
        'default_units': Units.discrete,
        'supports_events': False,
    }),
    ('KISAO_0000561', {
        'name': 'hybrid (runge kutta)',
        'id': 'hybrid',
        'default_units': Units.discrete,
        'supports_events': False,
    }),
    ('KISAO_0000563', {
        'name': 'hybrid (RK-45)',
        'id': 'hybridODE45',
        'default_units': Units.discrete,
        'supports_events': True,
    }),
    ('KISAO_0000566', {
        'name': 'SDE Solve (RI5)',
        'id': 'stochasticRunkeKuttaRI5',
        'default_units': Units.continuous,
        'supports_events': True,
    }),
])


KISAO_PARAMETERS_MAP = {
    'KISAO_0000209': {
        'name': 'Relative Tolerance',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000560', 'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000304'],
    },
    'KISAO_0000211': {
        'name': 'Absolute Tolerance',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000560', 'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000304',
                       'KISAO_0000566'],
    },
    'KISAO_0000216': {
        'name': 'Integrate Reduced Model',
        'type': ValueType.boolean,
        'algorithms': ['KISAO_0000560', 'KISAO_0000562', 'KISAO_0000304'],
    },
    'KISAO_0000415': {
        'name': 'Max Internal Steps',
        'type': ValueType.integer,
        'algorithms': ['KISAO_0000048', 'KISAO_0000560', 'KISAO_0000027', 'KISAO_0000029',
                       'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000561', 'KISAO_0000304', 'KISAO_0000566', 'KISAO_0000039'],
    },
    'KISAO_0000467': {
        'name': 'Max Internal Step Size',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000560', 'KISAO_0000562'],
    },
    'KISAO_0000488': {
        'name': 'Random Seed',
        'type': ValueType.integer,
        'algorithms': ['KISAO_0000048', 'KISAO_0000027', 'KISAO_0000029', 'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000561',
                       'KISAO_0000039'],
    },
    'KISAO_0000228': {
        'name': 'Epsilon',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000048', 'KISAO_0000039'],
    },
    'KISAO_0000203': {
        'name': 'Lower Limit',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000562', 'KISAO_0000561'],
    },
    'KISAO_0000204': {
        'name': 'Upper Limit',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000562', 'KISAO_0000561'],
    },
    'KISAO_0000205': {
        'name': 'Partitioning Interval',
        'type': ValueType.integer,
        'algorithms': ['KISAO_0000562', 'KISAO_0000561'],
    },
    'KISAO_0000559': {
        'name': 'Initial Step Size',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000304'],
    },
    'KISAO_0000483': {
        'name': {
            'KISAO_0000561': 'Runge Kutta Stepsize',
            'KISAO_0000566': 'Internal Steps Size',
        },
        'type': ValueType.float,
        'algorithms': ['KISAO_0000561', 'KISAO_0000566'],
    },
    'KISAO_0000565': {
        'name': 'Tolerance for Root Finder',
        'type': ValueType.float,
        'algorithms': ['KISAO_0000566'],
    },
    'KISAO_0000567': {
        'name': 'Force Physical Correctness',
        'type': ValueType.boolean,
        'algorithms': ['KISAO_0000566'],
    },
    'KISAO_0000534': {
        'name': 'Deterministic Reactions',
        'type': ValueType.list,
        'algorithms': ['KISAO_0000563'],
    },
}
