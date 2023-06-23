""" Data model for maps from KiSAO terms to COPASI algorithms and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from __future__ import annotations
from typing import Union, get_type_hints

import basico
from biosimulators_utils.data_model import ValueType
import collections
import enum

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
