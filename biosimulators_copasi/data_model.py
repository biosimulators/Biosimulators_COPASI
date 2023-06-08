""" Data model for maps from KiSAO terms to COPASI algorithms and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from __future__ import annotations
from typing import Union
from biosimulators_utils.data_model import ValueType
import collections
import enum

__all__ = ['Units', 'KISAO_ALGORITHMS_MAP', 'KISAO_PARAMETERS_MAP']


class Units(str, enum.Enum):
    """ Function for getting simulation results """
    discrete = 'discrete'
    continuous = 'continuous'

class CopasiAlgorithmParameter:
    KISAO_ID: str
    ID: str
    NAME: str

    def get_value(self) -> Union[int, str, float, bool, list]:
        raise NotImplementedError

    def __eq__(self, other: CopasiAlgorithmParameter) -> bool:
        if not isinstance(other, CopasiAlgorithmParameter):
            return False
        other_value = other.get_value()
        if type(other_value) != type(self.get_value()):
            return False
        return other.get_value() == self.get_value()

    def __ne__(self, other: RelativeToleranceParameter) -> bool:
        return not self.__eq__(other)

class RelativeToleranceParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000209"
    ID: str = "rel_tol"
    NAME: str = "Relative Tolerance"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class AbsoluteToleranceParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000211"
    ID: str = "abs_tol"
    NAME: str = "Relative Tolerance"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class IntegrateReducedModel(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000216"
    ID: str = "irm"
    NAME: str = "Integrate Reduced Model"

    def __init__(self, value: bool):
        self._value = value

    def get_value(self) -> bool:
        return self._value

class MaximumInternalSteps(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000415"
    ID: str = "max_steps"
    NAME: str = "Max Internal Steps"

    def __init__(self, value: int):
        self._value = value

    def get_value(self) -> int:
        return self._value

class MaximumInternalStepSize(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000467"
    ID: str = "max_step_size"
    NAME: str = "Max Internal Step Size"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class RandomSeed(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000488"
    ID: str = "random_seed"
    NAME: str = "Random Seed"

    def __init__(self, value: int):
        self._value = value

    def get_value(self) -> int:
        return self._value

class Epsilon(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000228"
    ID: str = "epsilon"
    NAME: str = "Epsilon"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class LowerLimit(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000203"
    ID: str = "lower_lim"
    NAME: str = "Lower Limit"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class UpperLimit(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000204"
    ID: str = "upper_lim"
    NAME: str = "Upper Limit"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class PartitioningInterval(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000205"
    ID: str = "partitioning_interval"
    NAME: str = "Partitioning Interval"

    def __init__(self, value: int):
        self._value = value

    def get_value(self) -> int:
        return self._value

class InitialStepSize(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000559"
    ID: str = "init_step_size"
    NAME: str = "Initial Step Size"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class StepSize(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000483"
    ID: str = None
    NAME: str = None

    def get_value(self) -> float:
        raise NotImplementedError

class RungeKuttaStepSize(StepSize):
    ID: str = "rk_step_size"
    NAME: str = "Runge-Kutta Stepsize"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class InternalStep(StepSize):
    ID: str = "internal_step_size"
    NAME: str = "Internal Steps Size"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class ToleranceForRootFinder(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000565"
    ID: str = "root_finder_tolerance"
    NAME: str = "Tolerance for Root Finder"

    def __init__(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value

class ForcePhysicalCorrectness(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000567"
    ID: str = "root_finder_tolerance"
    NAME: str = "Force Physical Correctness"

    def __init__(self, value: bool):
        self._value = value

    def get_value(self) -> bool:
        return self._value

class DeterministicReactions(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000534"
    ID: str = "root_finder_tolerance"
    NAME: str = "Deterministic Reactions"

    def __init__(self, value: list):
        self._value = value

    def get_value(self) -> list:
        return self._value


class CopasiAlgorithmType(enum.Enum):
    GIBSON_BRUCK = "stochastic"
    DIRECT_METHOD = "directmethod"
    TAU_LEAP = "tauleap"
    ADAPTIVE_SSA_TAU_LEAP = "adaptivesa"
    LSODA = "lsoda"
    RADAU5 = "radau5"
    HYBRID_LSODA = "hybridlsoda"
    # HYBRID_RUNGE_KUTTA = ""  # Not yet implemented in basico! See `baisco.task_timecourse.__method_name_to_type`
    HYBRID_RK45 = "hybridode45"
    SDE_SOLVE_RI5 = "sde"

class CopasiAlgorithm:
    KISAO_ID: str
    ID: str
    NAME: str
    CAN_SUPPORT_EVENTS: bool
    def get_kisao_id(self) -> str:
        raise NotImplementedError

    def get_copasi_id(self) -> str:
        raise NotImplementedError

    def get_copasi_name(self) -> str:
        raise NotImplementedError

    def get_unit_set(self) -> Units:
        raise NotImplementedError


class GibsonBruckAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000027"
    ID: str = "stochastic"
    NAME: str = "Gibson + Bruck"
    CAN_SUPPORT_EVENTS: bool = True
    def __init__(self, units: Units = Units.discrete):
        self._units = units
        self.max_internal_steps = MaximumInternalSteps()
        self.random_seed = RandomSeed()

    def get_kisao_id(self) -> str:
        raise GibsonBruckAlgorithm.KISAO_ID

    def get_copasi_id(self) -> str:
        raise GibsonBruckAlgorithm.ID

    def get_copasi_name(self) -> str:
        return GibsonBruckAlgorithm.NAME

    def get_unit_set(self) -> Units:
        return self._units


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
