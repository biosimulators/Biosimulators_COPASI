""" Data model for maps from KiSAO terms to COPASI algorithms and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from __future__ import annotations
from typing import Union, get_type_hints

from biosimulators_utils.sedml.data_model import UniformTimeCourseSimulation, Variable

import basico
import pandas
from biosimulators_utils.data_model import ValueType
import collections
import enum
import math

__all__ = [
    'Units',
    'CopasiAlgorithmParameter',
    'RelativeToleranceParameter',
    'AbsoluteToleranceParameter',
    'IntegrateReducedModelParameter',
    'MaximumInternalStepsParameter',
    'MaximumInternalStepSizeParameter',
    'RandomSeedParameter',
    'EpsilonParameter',
    'LowerLimitParameter',
    'UpperLimitParameter',
    'PartitioningIntervalParameter',
    'InitialStepSizeParameter',
    'StepSizeParameter',
    'RungeKuttaStepSizeParameter',
    'InternalStepParameter',
    'ToleranceForRootFinderParameter',
    'ForcePhysicalCorrectnessParameter',
    'DeterministicReactionsParameter',
    'CopasiAlgorithmType',
    'CopasiAlgorithm',
    'GibsonBruckAlgorithm',
    'DirectMethodAlgorithm',
    'TauLeapAlgorithm',
    'AdaptiveSSATauLeapAlgorithm',
    'LsodaAlgorithm',
    'Radau5Algorithm',
    'HybridLsodaAlgorithm',
    'HybridRungeKuttaAlgorithm',
    'HybridRK45Algorithm',
    'SDESolveRI5Algorithm'
]


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

    def set_value(self, new_value: Union[int, str, float, bool, list]):
        raise NotImplementedError

    @staticmethod
    def get_value_type(cls: CopasiAlgorithmParameter):
        func = cls.get_value
        type_map = get_type_hints(func)
        return type_map["return"]

    def get_override_repr(self) -> dict:
        if self.get_value() is None:
            return {}

        return {self.NAME: self.get_value()}

    def __eq__(self, other: CopasiAlgorithmParameter) -> bool:
        if not isinstance(other, CopasiAlgorithmParameter):
            return False
        other_value = other.get_value()
        if isinstance(other_value, type(self.get_value())):
            return False
        return other.get_value() == self.get_value()

    def __ne__(self, other: RelativeToleranceParameter) -> bool:
        return not self.__eq__(other)


class RelativeToleranceParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000209"
    ID: str = "r_tol"
    NAME: str = "Relative Tolerance"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class AbsoluteToleranceParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000211"
    ID: str = "a_tol"
    NAME: str = "Absolute Tolerance"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class IntegrateReducedModelParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000216"
    ID: str = "irm"
    NAME: str = "Integrate Reduced Model"

    def __init__(self, value: bool = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> bool:
        return self._value

    def set_value(self, new_value: bool):
        if new_value is not None and not (type(new_value) == bool):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class MaximumInternalStepsParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000415"
    ID: str = "max_steps"
    NAME: str = "Max Internal Steps"

    def __init__(self, value: int = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> int:
        return self._value

    def set_value(self, new_value: int):
        if new_value is not None and not (type(new_value) == int):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class MaximumInternalStepSizeParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000467"
    ID: str = "max_step_size"
    NAME: str = "Max Internal Step Size"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class RandomSeedParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000488"
    ID: str = "random_seed"
    NAME: str = "Random Seed"

    def __init__(self, value: int = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> int:
        return self._value

    def set_value(self, new_value: int):
        if new_value is not None and not (type(new_value) == int):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value

    def get_override_repr(self) -> dict:
        if self.get_value() is None:
            return {}

        return {
            self.NAME: self.get_value(),
            "Use Random Seed": True
        }


class EpsilonParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000228"
    ID: str = "epsilon"
    NAME: str = "Epsilon"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class LowerLimitParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000203"
    ID: str = "lower_lim"
    NAME: str = "Lower Limit"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class UpperLimitParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000204"
    ID: str = "upper_lim"
    NAME: str = "Upper Limit"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class PartitioningIntervalParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000205"
    ID: str = "partitioning_interval"
    NAME: str = "Partitioning Interval"

    def __init__(self, value: int = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> int:
        return self._value

    def set_value(self, new_value: int):
        if new_value is not None and not (type(new_value) == int):  # can't use isinstance because PEP 285:
            raise ValueError
        self._value = new_value


class InitialStepSizeParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000559"
    ID: str = "init_step_size"
    NAME: str = "Initial Step Size"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285:
            raise ValueError
        self._value = new_value


class StepSizeParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000483"
    ID: str = None
    NAME: str = None

    def get_value(self) -> float:
        raise NotImplementedError

    def set_value(self, new_value: float):
        raise NotImplementedError


class RungeKuttaStepSizeParameter(StepSizeParameter):
    ID: str = "rk_step_size"
    NAME: str = "Runge Kutta Stepsize"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class InternalStepParameter(StepSizeParameter):
    ID: str = "internal_step_size"
    NAME: str = "Internal Steps Size"

    def __init__(self, value: float = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class ToleranceForRootFinderParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000565"
    ID: str = "root_finder_tolerance"
    NAME: str = "Tolerance for Root Finder"

    def __init__(self, value: float):
        self._value = None
        self.set_value(value)

    def get_value(self) -> float:
        return self._value

    def set_value(self, new_value: float):
        if new_value is not None and not (type(new_value) == float):  # can't use isinstance because PEP 285
            raise ValueError
        self._value = new_value


class ForcePhysicalCorrectnessParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000567"
    ID: str = "force_physical_correctness"
    NAME: str = "Force Physical Correctness"

    def __init__(self, value: bool = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> bool:
        return self._value

    def set_value(self, new_value: bool):
        if new_value is not None and not (type(new_value) == bool):  # can't use isinstance because PEP 285:
            raise ValueError
        self._value = new_value


class DeterministicReactionsParameter(CopasiAlgorithmParameter):
    KISAO_ID: str = "KISAO_0000534"
    ID: str = "deterministic_reactions"
    NAME: str = "Deterministic Reactions"

    def __init__(self, value: list = None):
        self._value = None
        self.set_value(value)

    def get_value(self) -> list:
        return self._value

    def set_value(self, new_value: list):
        if new_value is not None and not isinstance(new_value, list):
            raise ValueError
        basico.get_reactions()
        self._value = new_value if new_value is not None else []


class CopasiAlgorithm:
    KISAO_ID: str
    ID: CopasiAlgorithmType
    NAME: str
    CAN_SUPPORT_EVENTS: bool

    def get_parameters_by_kisao(self) -> dict[str, CopasiAlgorithmParameter]:
        return {
            getattr(self, member).KISAO_ID: getattr(self, member)
            for member in dir(self)
            if isinstance(getattr(self, member), CopasiAlgorithmParameter)
        }

    def get_copasi_id(self) -> str:
        raise NotImplementedError

    def get_unit_set(self) -> Units:
        raise NotImplementedError

    def get_overrides(self) -> dict:
        raise NotImplementedError

    def get_method_settings(self) -> dict[str, str]:
        settings: dict[str, str] = {"name": self.NAME}
        settings.update(self.get_overrides())
        return settings

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (True if (
            self.KISAO_ID == other.KISAO_ID,
            self.ID == other.ID,
            self.NAME == other.NAME,
            self.CAN_SUPPORT_EVENTS == other.CAN_SUPPORT_EVENTS
        ) else False)


class GibsonBruckAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000027"
    ID: str = "stochastic"
    NAME: str = "Stochastic (Gibson + Bruck)"
    CAN_SUPPORT_EVENTS: bool = True

    def __init__(self, max_internal_steps: int = None, random_seed: int = None, units: Units = Units.discrete):
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.random_seed = RandomSeedParameter(random_seed)
        self._units = units

    def get_copasi_id(self) -> str:
        return GibsonBruckAlgorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.random_seed.get_override_repr())
        return overrides


class DirectMethodAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000029"
    ID: str = "directmethod"
    NAME: str = "Stochastic (Direct method)"
    CAN_SUPPORT_EVENTS: bool = True

    def __init__(self, max_internal_steps: int = None, random_seed: int = None, units: Units = Units.discrete):
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.random_seed = RandomSeedParameter(random_seed)
        self._units = units

    def get_copasi_id(self) -> str:
        return DirectMethodAlgorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.random_seed.get_override_repr())
        return overrides


class TauLeapAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000039"
    ID: str = "tauleap"
    NAME: str = "Stochastic (τ-Leap)"
    CAN_SUPPORT_EVENTS: bool = False

    def __init__(self, max_internal_steps: int = None, random_seed: int = None, epsilon: float = None,
                 units: Units = Units.discrete):
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.random_seed = RandomSeedParameter(random_seed)
        self.epsilon = EpsilonParameter(epsilon)
        self._units = units

    def get_copasi_id(self) -> str:
        return TauLeapAlgorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.random_seed.get_override_repr())
        overrides.update(self.epsilon.get_override_repr())
        return overrides


class AdaptiveSSATauLeapAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000048"
    ID: str = "adaptivesa"
    NAME: str = "Stochastic (Adaptive SSA/τ-Leap)"
    CAN_SUPPORT_EVENTS: bool = True

    def __init__(self, max_internal_steps: int = None, random_seed: int = None, epsilon: float = None,
                 units: Units = Units.discrete):
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.random_seed = RandomSeedParameter(random_seed)
        self.epsilon = EpsilonParameter(epsilon)
        self._units = units

    def get_copasi_id(self) -> str:
        return AdaptiveSSATauLeapAlgorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.random_seed.get_override_repr())
        overrides.update(self.epsilon.get_override_repr())
        return overrides


class LsodaAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000560"
    ID: str = "lsoda"
    NAME: str = "Deterministic (LSODA)"
    CAN_SUPPORT_EVENTS: bool = True

    def __init__(self, relative_tolerance: float = None, absolute_tolerance: float = None,
                 integrate_reduced_model: bool = None, max_internal_steps: int = None,
                 max_internal_step_size: float = None, units: Units = Units.continuous):
        self._units = units
        self.relative_tolerance = RelativeToleranceParameter(relative_tolerance)
        self.absolute_tolerance = AbsoluteToleranceParameter(absolute_tolerance)
        self.integrate_reduced_model = IntegrateReducedModelParameter(integrate_reduced_model)
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.max_internal_step_size = MaximumInternalStepSizeParameter(max_internal_step_size)

    def get_copasi_id(self) -> str:
        return LsodaAlgorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.relative_tolerance.get_override_repr())
        overrides.update(self.absolute_tolerance.get_override_repr())
        overrides.update(self.integrate_reduced_model.get_override_repr())
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.max_internal_step_size.get_override_repr())
        return overrides


class Radau5Algorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000304"
    ID: str = "radau5"
    NAME: str = "Deterministic (RADAU5)"
    CAN_SUPPORT_EVENTS: bool = False

    def __init__(self, relative_tolerance: float = None, absolute_tolerance: float = None,
                 integrate_reduced_model: bool = None, max_internal_steps: int = None, initial_step_size: float = None,
                 units: Units = Units.continuous):
        self.relative_tolerance = RelativeToleranceParameter(relative_tolerance)
        self.absolute_tolerance = AbsoluteToleranceParameter(absolute_tolerance)
        self.integrate_reduced_model = IntegrateReducedModelParameter(integrate_reduced_model)
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.initial_step_size = InitialStepSizeParameter(initial_step_size)
        self._units = units

    def get_copasi_id(self) -> str:
        return Radau5Algorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.relative_tolerance.get_override_repr())
        overrides.update(self.absolute_tolerance.get_override_repr())
        overrides.update(self.integrate_reduced_model.get_override_repr())
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.initial_step_size.get_override_repr())
        return overrides


class HybridLsodaAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000562"
    ID: str = "hybridlsoda"
    NAME: str = "Hybrid (LSODA)"
    CAN_SUPPORT_EVENTS: bool = False

    def __init__(self, relative_tolerance: float = None, absolute_tolerance: float = None,
                 integrate_reduced_model: bool = None, max_internal_steps: int = None,
                 max_internal_step_size: float = None, random_seed: int = None, lower_limit: float = None,
                 upper_limit: float = None, partitioning_interval: float = None, units: Units = Units.discrete):
        self.relative_tolerance = RelativeToleranceParameter(relative_tolerance)
        self.absolute_tolerance = AbsoluteToleranceParameter(absolute_tolerance)
        self.integrate_reduced_model = IntegrateReducedModelParameter(integrate_reduced_model)
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.max_internal_step_size = MaximumInternalStepSizeParameter(max_internal_step_size)
        self.random_seed = RandomSeedParameter(random_seed)
        self.lower_limit = LowerLimitParameter(lower_limit)
        self.upper_limit = UpperLimitParameter(upper_limit)
        self.partitioning_interval = PartitioningIntervalParameter(partitioning_interval)
        self._units = units

    def get_copasi_id(self) -> str:
        return HybridLsodaAlgorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.relative_tolerance.get_override_repr())
        overrides.update(self.absolute_tolerance.get_override_repr())
        overrides.update(self.integrate_reduced_model.get_override_repr())
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.max_internal_step_size.get_override_repr())
        overrides.update(self.random_seed.get_override_repr())
        overrides.update(self.lower_limit.get_override_repr())
        overrides.update(self.upper_limit.get_override_repr())
        overrides.update(self.partitioning_interval.get_override_repr())
        return overrides


class HybridRungeKuttaAlgorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000561"
    ID: str = "hybrid"
    NAME: str = "Hybrid (Runge-Kutta)"
    CAN_SUPPORT_EVENTS: bool = False

    def __init__(self, max_internal_steps: int = None, random_seed: int = None, lower_limit: float = None,
                 upper_limit: float = None, step_size: float = None, partitioning_interval: float = None,
                 units: Units = Units.discrete):
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.random_seed = RandomSeedParameter(random_seed)
        self.lower_limit = LowerLimitParameter(lower_limit)
        self.upper_limit = UpperLimitParameter(upper_limit)
        self.step_size = RungeKuttaStepSizeParameter(step_size)
        self.partitioning_interval = PartitioningIntervalParameter(partitioning_interval)
        self._units = units

    def get_copasi_id(self) -> str:
        return HybridRungeKuttaAlgorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.random_seed.get_override_repr())
        overrides.update(self.lower_limit.get_override_repr())
        overrides.update(self.upper_limit.get_override_repr())
        overrides.update(self.step_size.get_override_repr())
        overrides.update(self.partitioning_interval.get_override_repr())
        return overrides


class HybridRK45Algorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000563"
    ID: str = "hybridode45"
    NAME: str = "Hybrid (RK-45)"
    CAN_SUPPORT_EVENTS: bool = True

    def __init__(self, relative_tolerance: float = None, absolute_tolerance: float = None,
                 max_internal_steps: int = None, random_seed: int = None, deterministic_reactions: list = None,
                 units: Units = Units.discrete):
        self.relative_tolerance = RelativeToleranceParameter(relative_tolerance)
        self.absolute_tolerance = AbsoluteToleranceParameter(absolute_tolerance)
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.random_seed = RandomSeedParameter(random_seed)
        self.deterministic_reactions = DeterministicReactionsParameter(deterministic_reactions)
        self._units = units

    def get_copasi_id(self) -> str:
        return HybridRK45Algorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.relative_tolerance.get_override_repr())
        overrides.update(self.absolute_tolerance.get_override_repr())
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.random_seed.get_override_repr())
        overrides.update(self.deterministic_reactions.get_override_repr())
        return overrides


class SDESolveRI5Algorithm(CopasiAlgorithm):
    KISAO_ID: str = "KISAO_0000566"
    ID: str = "sde"
    NAME: str = "SDE Solver (RI5)"
    CAN_SUPPORT_EVENTS: bool = True

    def __init__(self, absolute_tolerance: float = None, max_internal_steps: int = None, step_size: float = None,
                 tolerance_for_root_finder: float = None, force_physical_correctness: bool = None,
                 units: Units = Units.continuous):
        self.absolute_tolerance = AbsoluteToleranceParameter(absolute_tolerance)
        self.max_internal_steps = MaximumInternalStepsParameter(max_internal_steps)
        self.step_size = InternalStepParameter(step_size)
        self.tolerance_for_root_finder = ToleranceForRootFinderParameter(tolerance_for_root_finder)
        self.force_physical_correctness = ForcePhysicalCorrectnessParameter(force_physical_correctness)
        self._units = units

    def get_copasi_id(self) -> str:
        return SDESolveRI5Algorithm.ID

    def get_unit_set(self) -> Units:
        return self._units

    def get_overrides(self) -> dict:
        overrides = {}
        overrides.update(self.absolute_tolerance.get_override_repr())
        overrides.update(self.max_internal_steps.get_override_repr())
        overrides.update(self.step_size.get_override_repr())
        overrides.update(self.tolerance_for_root_finder.get_override_repr())
        overrides.update(self.force_physical_correctness.get_override_repr())
        return overrides


class CopasiAlgorithmType(enum.Enum):
    GIBSON_BRUCK = GibsonBruckAlgorithm
    DIRECT_METHOD = DirectMethodAlgorithm
    TAU_LEAP = TauLeapAlgorithm
    ADAPTIVE_SSA_TAU_LEAP = AdaptiveSSATauLeapAlgorithm
    LSODA = LsodaAlgorithm
    RADAU5 = Radau5Algorithm
    HYBRID_LSODA = HybridLsodaAlgorithm
    HYBRID_RUNGE_KUTTA = HybridRungeKuttaAlgorithm
    HYBRID_RK45 = HybridRK45Algorithm
    SDE_SOLVE_RI5 = SDESolveRI5Algorithm

class CopasiMappings:
    @staticmethod
    def format_to_copasi_reaction_name(sbml_name: str):
        return f"({sbml_name}).Flux"

    @staticmethod
    def format_to_copasi_species_concentration_name(sbml_name: str):
        return f"[{sbml_name}]"

    @staticmethod
    def format_to_copasi_compartment_name(sbml_name: str):
        return f"Compartments[{sbml_name}].Volume"

    @staticmethod
    def format_to_copasi_parameter_name(sbml_name: str):
        return f"Values[{sbml_name}]"

    @staticmethod
    def map_sedml_to_copasi(variables: list[Variable], for_output: bool = True) -> dict[Variable, str]:
        sed_to_id: dict[Variable, str]
        id_to_name: dict[str, str]

        sed_to_id = CopasiMappings.map_sedml_to_sbml_ids(variables)
        id_to_name = CopasiMappings._map_sbml_id_to_copasi_name(for_output)
        return {sedml_var: id_to_name[sed_to_id[sedml_var]] for sedml_var in sed_to_id}

    @staticmethod
    def map_sedml_to_sbml_ids(variables: list[Variable]) -> dict[Variable, str]:
        sedml_var_to_sbml_id: dict[Variable, str] = {}
        sedml_var_to_sbml_id.update(CopasiMappings._map_sedml_symbol_to_sbml_id(variables))
        sedml_var_to_sbml_id.update(CopasiMappings._map_sedml_target_to_sbml_id(variables))
        return sedml_var_to_sbml_id

    @staticmethod
    def _map_sbml_id_to_copasi_name(for_output: bool) -> dict[str, str]:
        # NB: usually, sbml_id == copasi name, with exceptions like "Time"
        compartments: pandas.DataFrame = basico.get_compartments()
        metabolites: pandas.DataFrame = basico.get_species()
        reactions: pandas.DataFrame = basico.get_reactions()
        parameters: pandas.DataFrame = basico.get_parameters()
        compartment_mapping: dict[str, str]
        metabolites_mapping: dict[str, str]
        reactions_mapping: dict[str, str]
        parameters_mapping: dict[str, str]
        sbml_id_to_sbml_name_map: dict[str, str]

        # Create mapping
        if for_output:
            compartment_mapping = \
                {compartments.at[row, "sbml_id"]: CopasiMappings.format_to_copasi_compartment_name(str(row))
                 for row in compartments.index} if compartments is not None else {}
            metabolites_mapping = \
                {metabolites.at[row, "sbml_id"]: CopasiMappings.format_to_copasi_species_concentration_name(str(row))
                 for row in metabolites.index} if metabolites is not None else {}
            reactions_mapping = \
                {reactions.at[row, "sbml_id"]: CopasiMappings.format_to_copasi_reaction_name(str(row))
                 for row in reactions.index} if reactions is not None else {}
            parameters_mapping = \
                {parameters.at[row, "sbml_id"]: CopasiMappings.format_to_copasi_parameter_name(str(row))
                 for row in parameters.index} if parameters is not None else {}
        else:
            compartment_mapping = {compartments.at[row, "sbml_id"]: str(row) for row in compartments.index}
            metabolites_mapping = {metabolites.at[row, "sbml_id"]: str(row) for row in metabolites.index}
            reactions_mapping = {reactions.at[row, "sbml_id"]: str(row) for row in reactions.index}
            parameters_mapping = {parameters.at[row, "sbml_id"] : str(row) for row in parameters.index}

        # Combine mappings
        sbml_id_to_sbml_name_map = {"Time": "Time"}
        sbml_id_to_sbml_name_map.update(compartment_mapping)
        sbml_id_to_sbml_name_map.update(metabolites_mapping)
        sbml_id_to_sbml_name_map.update(reactions_mapping)
        sbml_id_to_sbml_name_map.update(parameters_mapping)
        return sbml_id_to_sbml_name_map

    @staticmethod
    def _map_sedml_symbol_to_sbml_id(variables: list[Variable]) -> dict[Variable, str]:
        symbol_mappings = {"kisao:0000832": "Time", "urn:sedml:symbol:time": "Time", "symbol.time": "Time"}
        symbolic_variables: list[Variable]
        raw_symbols: list[str]
        symbols: list[Union[str, None]]

        # Process the variables
        symbolic_variables = [variable for variable in variables if variable.symbol is not None]
        raw_symbols = [str(variable.symbol).lower() for variable in variables if variable.symbol is not None]
        symbols = [symbol_mappings.get(variable, None) for variable in raw_symbols]

        if None in symbols:
            raise ValueError(f"BioSim COPASI is unable to interpret symbol '{raw_symbols[symbols.index(None)]}'")

        return {sedml_var: copasi_name for sedml_var, copasi_name in zip(symbolic_variables, symbols)}

    @staticmethod
    def _map_sedml_target_to_sbml_id(variables: list[Variable]) -> dict[Variable, str]:
        target_based_variables: list[Variable]
        raw_targets: list[str]
        targets: list[str]

        target_based_variables = [variable for variable in variables if list(variable.to_tuple())[2] is not None]
        raw_targets = [str(list(variable.to_tuple())[2]) for variable in target_based_variables]
        targets = [CopasiMappings._extract_id_from_xpath(target) for target in raw_targets]

        return {sedml_var: copasi_name for sedml_var, copasi_name in zip(target_based_variables, targets)}

    @staticmethod
    def _extract_id_from_xpath(target: str):
        beginning_index: int = (target.find('@id=\''))
        beginning_index = beginning_index + 5 if beginning_index is not -1 else (target.find('@id=\"')) + 5
        end_index: int = target.find('\']')
        end_index: int = end_index if end_index is not -1 else target.find('\"]')
        return target[beginning_index:end_index]


class BasicoInitialization:
    def __init__(self, algorithm: CopasiAlgorithm, variables: list[Variable]):
        self.algorithm = algorithm
        self._sedml_var_to_copasi_name: dict[Variable, str] = CopasiMappings.map_sedml_to_copasi(variables)
        self._sim = None
        self.init_time_offset = None
        self._duration_arg = None
        self._step_size = None
        self.number_of_steps = None
        self._length_of_output = None

    def configure_simulation_settings(self, sim: UniformTimeCourseSimulation):
        self._sim: UniformTimeCourseSimulation = sim
        self.init_time_offset: float = self._sim.initial_time
        self._duration_arg: float = self._sim.output_end_time - self.init_time_offset  # COPASI is kept in the dark
        self._step_size: float = BasicoInitialization._calc_simulation_step_size(self._sim)
        self.number_of_steps: int = int(self._duration_arg / self._step_size)
        self._length_of_output: int = int((self._sim.output_end_time - self._sim.output_start_time) / self._step_size)
        self._length_of_output += 1

    def get_simulation_configuration(self) -> dict:
        # Create the configuration basico needs to initialize the time course task
        problem = {
            "AutomaticStepSize": False,
            "StepNumber": self.number_of_steps,
            "StepSize": self._step_size,
            "Duration": self._duration_arg,
            "OutputStartTime": self._sim.output_start_time - self.init_time_offset
        }
        method = self.algorithm.get_method_settings()
        return {
            "problem": problem,
            "method": method
        }

    def get_run_configuration(self) -> dict:
        return {
            "output_selection": list(self._sedml_var_to_copasi_name.values()),
            "use_initial_values": True,
        }

    def get_expected_output_length(self) -> int:
        return self._length_of_output

    def get_copasi_name(self, sedml_var: Variable) -> str:
        return self._sedml_var_to_copasi_name.get(sedml_var)

    def get_kisao_id_for_kisao_algorithm(self) -> str:
        return self.algorithm.KISAO_ID

    def get_copasi_algorithm_id(self) -> str:
        return self.algorithm.get_copasi_id()

    @staticmethod
    def _calc_simulation_step_size(sim: UniformTimeCourseSimulation) -> int:
        if sim.output_end_time - sim.output_start_time < 0:
            raise ValueError('Output end time must be greater than the output start time.')

        try:
            time_diff = sim.output_end_time - sim.output_start_time
            if time_diff == 0:
                raise ZeroDivisionError  # We want to have exactly 1 step, and that's what our except block does
            step_size_arg = time_diff / sim.number_of_steps
        except ZeroDivisionError:  # sim.output_end_time == sim.output_start_time
            step_size_arg = sim.number_of_points

        return step_size_arg
