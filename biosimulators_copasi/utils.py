""" Utilities for working with the maps from KiSAO ids to COPASI methods and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from __future__ import annotations

import builtins
import math
import types
from typing import Union
from biosimulators_copasi.data_model import *
from biosimulators_utils.combine.data_model import CombineArchiveContentFormat
from biosimulators_utils.combine.io import CombineArchiveReader, CombineArchiveWriter
from biosimulators_utils.config import get_config, Config  # noqa: F401
from biosimulators_utils.data_model import ValueType
from biosimulators_utils.simulator.utils import get_algorithm_substitution_policy
from biosimulators_utils.sedml.data_model import Variable, UniformTimeCourseSimulation, Algorithm, AlgorithmParameterChange
from biosimulators_utils.utils.core import validate_str_value, parse_value
from kisao.data_model import AlgorithmSubstitutionPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from kisao.utils import get_preferred_substitute_algorithm_by_ids
import COPASI
import basico
import itertools
import libsedml
import lxml
import os
import shutil
import tempfile
import pandas



__all__ = [
    'set_algorithm_parameter_value',
    'get_copasi_model_object_by_sbml_id',
    'get_copasi_model_obj_sbml_ids',
    'fix_copasi_generated_combine_archive',
]


def create_algorithm_instance(self, algorithm_type: CopasiAlgorithmType):
    if algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass
    elif algorithm_type == CopasiAlgorithmType.GIBSON_BRUCK:
        pass

class CopasiAlgorithm_OLD:
    def __init__(self, kisao_id: str, copasi_algorithm_code: int, copasi_algorithm_name: str):
        self.kisao_id = kisao_id
        self.copasi_algorithm_code = copasi_algorithm_code
        self.copasi_algorithm_name = copasi_algorithm_name

    def to_tuple(self):
        return self.kisao_id, self.copasi_algorithm_code, self.copasi_algorithm_name


class BasicoInitialization:
    def __init__(self, sim: UniformTimeCourseSimulation, algorithm: CopasiAlgorithm, variables: list[Variable]):
        self.algorithm = algorithm
        self._sedml_var_to_copasi_name: dict[Variable, str] = _map_sedml_to_copasi(variables)
        self._sim = sim
        self._duration_arg: float = self._sim.output_end_time - self._sim.initial_time
        self.number_of_steps = _calc_number_of_simulation_steps(self._sim, self._duration_arg)
        self._step_size = self._duration_arg/self.number_of_steps

    def get_simulation_configuration(self) -> dict:
        # Create the configuration basico needs to initialize the time course task
        problem = {
            "AutomaticStepSize": False,
            "StepNumber": self.number_of_steps,
            "StepSize": self._step_size,
            "Duration": self._duration_arg,
            "OutputStartTime": self._sim.output_start_time
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
        return self.number_of_steps + 1

    def get_COPASI_name(self, sedml_var: Variable) -> str:
        return self._sedml_var_to_copasi_name.get(sedml_var)

    def get_KiSAO_id_for_KiSAO_algorithm(self) -> str:
        return self.algorithm.KISAO_ID

    def get_COPASI_algorithm_ID(self) -> str:
        return self.algorithm.get_copasi_id()


def get_algorithm(kisao_id: str, events_were_requested: bool = False, config: Config = None) -> CopasiAlgorithm:
    """ Get the algorithm wrapper for a COPASI algorithm

        Args:
            kisao_id (:obj:`str`): KiSAO algorithm id
            events_were_requested (:obj:`bool`, optional): whether an algorithm that supports
                events is needed
            config (:obj:`Config`, optional): configuration

        Returns:
           :obj:`CopasiAlgorithm`: The copasi algorithm deemed suitable
        """
    # This step may not be necessary anymore

    algorithm_kisao_to_class_map: dict [str, CopasiAlgorithm] = {
        CopasiAlgorithmType[alg_name].value.KISAO_ID : CopasiAlgorithmType[alg_name].value
        for alg_name, _ in CopasiAlgorithmType.__members__.items()
    }

    legal_alg_kisao_ids = [
        kisao for kisao, obj in algorithm_kisao_to_class_map.items()
        if not events_were_requested or obj.CAN_SUPPORT_EVENTS
    ]

    if kisao_id in legal_alg_kisao_ids:
        constructor = algorithm_kisao_to_class_map[kisao_id]
        return constructor()  # it is, in fact, callable

    substitution_policy = get_algorithm_substitution_policy(config=config)
    try:
        alt_kisao_id = get_preferred_substitute_algorithm_by_ids(kisao_id, legal_alg_kisao_ids, substitution_policy)
    except NotImplementedError:
        other_hybrid_methods = ['KISAO_0000561', 'KISAO_0000562']
        similar_approx = AlgorithmSubstitutionPolicy.SIMILAR_APPROXIMATIONS
        selected_substitution_policy = ALGORITHM_SUBSTITUTION_POLICY_LEVELS[substitution_policy]
        needed_substitution_policy = ALGORITHM_SUBSTITUTION_POLICY_LEVELS[similar_approx]
        substitution_policy_is_sufficient = selected_substitution_policy >= needed_substitution_policy
        if events_were_requested and kisao_id in other_hybrid_methods and substitution_policy_is_sufficient:
            alt_kisao_id = 'KISAO_0000563'  # Hybrid Runge Kutta RK45 method
        else:
            alt_kisao_id = kisao_id  # Admit defeat, this will cause a ValueError

    if alt_kisao_id in legal_alg_kisao_ids:
        constructor = algorithm_kisao_to_class_map[alt_kisao_id]
        return constructor() # this too is, in fact, callable

    raise ValueError(f"No suitable equivalent for '{kisao_id}' could be found with the provided substitution policy")

def convert_sedml_deterministic_reactions_to_copasi(kisao_id: str, value: Union[bool, int, float, list]) -> list[str]:
    pass

def set_algorithm_parameter_values(copasi_algorithm: CopasiAlgorithm, requested_changes: list)\
        -> tuple[list[AlgorithmParameterChange],list[AlgorithmParameterChange]]:
    """ Set a parameter of a COPASI simulation function

    Args:
        algorithm_kisao_id (:obj:`str`): KiSAO algorithm id
        algorithm_function (:obj:`types.FunctionType`): algorithm function
        parameter_kisao_id (:obj:`str`): KiSAO parameter id
        value (:obj:`string`): parameter value

    Returns:
        :obj:`dict`: names of the COPASI parameters that were set and their values
    """
    param_dict: dict[str, CopasiAlgorithmParameter] = copasi_algorithm.get_parameters_by_kisao()

    change: AlgorithmParameterChange
    legal_changes, illegal_changes = [], []
    for change in requested_changes:
        legal_changes.append(change) if param_dict.get(change.kisao_id) is not None else illegal_changes.append(change)

    for change in legal_changes:
        target = param_dict.get(change.kisao_id)
        try:
            target.set_value(change.new_value)
        except ValueError:
            illegal_changes.append(change)

    if builtins.len(illegal_changes) == 0:
        return [], []

    unsupported_parameters = []
    bad_parameters = []
    for change in illegal_changes:
         bad_parameters.append(change) if change in legal_changes else unsupported_parameters.append(change)
    return unsupported_parameters, bad_parameters

    unsupported_message = ""
    bad_message = ""
    # Based on the logic above:
    #   - A parameter with a bad value is first labeled legal, but then found to be illegal, so it ends up in both lists
    #   - A parameter that is unsupported is immediately labeled as invalid, so it will only exist in the illegal list
    unsupported_parameters = [change.kisao_id for change in illegal_changes if change not in legal_changes]
    if builtins.len(unsupported_parameters) > 0:
        unsupported_list_str = f"[{', '.join(unsupported_parameters)}]"
        s: str
        be: str
        if builtins.len(unsupported_parameters) > 1:
            (s, be) = "s", "are"
        else:
            (s, be) = "", "is"
        unsupported_message = f"Parameter{s} '{unsupported_list_str}' {be} not supported, or a bad value.\n\n"

    bad_parameters = [change.kisao_id for change in illegal_changes if change in legal_changes]
    if builtins.len(bad_parameters) > 0:
        bad_list_str = f"[{', '.join(bad_parameters)}]"
        have: str
        a: str
        s: str
        if builtins.len(bad_parameters) > 1:
            (have, a, s) = "have", "", "s"
        else:
            (have, a, s) = "has", "a", ""
        bad_message = f"Parameter{s} '{bad_list_str}' {have} {a} bad value{s}.\n\n"

    valid_parameters = "".join([f'\t- "{params.NAME}"({kisao})\n' for kisao, params in param_dict.items()])
    valid_message = f"COPASI supports the following parameters with proper values:\n{valid_parameters}"

    error_message = unsupported_message + bad_message + valid_message
    if builtins.len(bad_parameters) == 0 and builtins.len(unsupported_parameters) > 0:
        raise NotImplementedError(error_message)
    elif builtins.len(unsupported_parameters) == 0 and builtins.len(bad_parameters) > 0:
        raise ValueError(error_message)
    else:
        raise RuntimeError(error_message)


def get_copasi_model_object_by_sbml_id(model: COPASI.CModel, id: str, units: Units):
    """ Get a COPASI model object by its SBML id

    Args:
        model (:obj:`COPASI.CModel`): model
        id (:obj:`str`): SBML id
        units (:obj:`Units`): desired units for the object

    Returns:
        :obj:`COPASI.CCompartment`, :obj:`COPASI.CMetab`, :obj:`COPASI.CModelValue`, or :obj:`COPASI.CReaction`:
            model object
    """
    for object in model.getMetabolites():
        if object.getSBMLId() == id:
            if units == Units.discrete:
                return object.getValueReference()
            else:
                return object.getConcentrationReference()

    for object in model.getModelValues():
        if object.getSBMLId() == id:
            return object.getValueReference()

    for object in model.getCompartments():
        if object.getSBMLId() == id:
            return object.getValueReference()

    for object in model.getReactions():
        if object.getSBMLId() == id:
            return object.getFluxReference()

    return None

def pre_process_changes(model: COPASI.CModel, sbml_ids: list[str], units: Units) -> COPASI.CDataObject:
    """ Get a COPASI model object by its SBML id

    Args:
        model (:obj:`COPASI.CModel`): model
        sbml_ids (:obj:`list` of :obj:`str`): SBML ids
        units (:obj:`Units`): desired units for the object

    Returns:
        :obj:`COPASI.CCompartment`, :obj:`COPASI.CMetab`, :obj:`COPASI.CModelValue`, or :obj:`COPASI.CReaction`:
            model object
    """

    metabolites = model.getMetabolites()
    model_vals = model.getModelValues()
    compartments = model.getCompartments()
    reactions = model.getReactions()

    entity: COPASI.CModelEntity
    metabolite_map: dict[str, COPASI.CMetab] = {entity.getSBMLId(): entity for entity in metabolites}
    model_val_map: dict[str, COPASI.CModelValue] = {entity.getSBMLId(): entity for entity in model_vals}
    compartment_map: dict[str, COPASI.CCompartment] = {entity.getSBMLId(): entity for entity in compartments}
    reaction_map: dict[str, COPASI.CReaction] = {entity.getSBMLId(): entity for entity in reactions}

    for sbml_id in sbml_ids:
        if sbml_id in metabolite_map.keys():
            entity: COPASI.CMetab = metabolite_map[sbml_id]
            return entity.getValueReference() if units == Units.discrete else entity.getConcentrationReference()

        if sbml_id in model_val_map.keys():
            entity: COPASI.CModelValue = model_val_map[sbml_id]
            return entity.getValueReference()

        if sbml_id in compartment_map.keys():
            entity: COPASI.CCompartment = compartment_map[sbml_id]
            return entity.getValueReference()

        if sbml_id in reaction_map.keys():
            entity: COPASI.CReaction = reaction_map[sbml_id]
            return entity.getFluxReference()

    return None


def get_copasi_model_obj_sbml_ids(model):
    """ Get the SBML id of each object of a COPASI model

    Args:
        model (:obj:`COPASI.CModel`): model

    Returns:
        :obj:`list` of :obj:`str: SBML id of each object of the model
    """
    ids = []

    for object in itertools.chain(
        model.getMetabolites(),
        model.getModelValues(),
        model.getCompartments(),
        model.getReactions()
    ):
        ids.append(object.getSBMLId())

    return ids


def fix_copasi_generated_combine_archive(in_filename, out_filename, config=None):
    """ Utility function that corrects COMBINE/OMEX archives generated by COPASI so they are compatible
    with other tools.

    All currently released versions of COPASI export COMBINE archive files. However, these archives
    presently diverge from the specifications of the SED-ML format.

    * Format in OMEX manifests is not a valid PURL media type URI
    * SED-ML files lack namespaces for SBML

    Args:
        in_filename (:obj:`str`): path to a COMBINE archive to correct
        out_filename (:obj:`str`): path to save correctd COMBINE archive
        config (:obj:`Config`, optional): BioSimulators-utils configuration
    """
    config = config or get_config()
    archive_dirname = tempfile.mkdtemp()
    try:
        archive = CombineArchiveReader().run(in_filename, archive_dirname, config=config)
    except Exception:
        shutil.rmtree(archive_dirname)
        raise

    # correct URI for COPASI application format
    for content in archive.contents:
        if content.format == 'application/x-copasi':
            content.format = CombineArchiveContentFormat.CopasiML
            # potentially issue warning messages if needed
            break

    # add SBML namespace to SED-ML file
    ns = None
    for content in archive.contents:
        if content.format == 'http://identifiers.org/combine.specifications/sbml':
            with open(os.path.join(archive_dirname, content.location), 'rb') as sbml:
                root = lxml.etree.parse(sbml)
                # get default ns
                ns = root.getroot().nsmap[None]
                break

    if ns:
        for content in archive.contents:
            if content.format == 'http://identifiers.org/combine.specifications/sed-ml':
                sedml_file = os.path.join(archive_dirname, content.location)
                doc = libsedml.readSedMLFromFile(sedml_file)
                sedml_ns = doc.getSedNamespaces().getNamespaces()
                if not sedml_ns.hasPrefix('sbml'):
                    sedml_ns.add(ns, 'sbml')
                    libsedml.writeSedMLToFile(doc, sedml_file)
                    # potentially issue warning message here, that the sedml file had no SBML prefix and it was added
                    break

    try:
        CombineArchiveWriter().run(archive, archive_dirname, out_filename)
    finally:
        shutil.rmtree(archive_dirname)

def _map_sedml_to_copasi(variables: list[Variable], for_output: bool = True) -> dict[Variable, str]:
    sed_to_id: dict[Variable, str]
    id_to_name: dict[str, str]

    sed_to_id = _map_sedml_to_sbml_ids(variables)
    id_to_name = _map_sbml_id_to_copasi_name(for_output)
    return {sedml_var: id_to_name[sed_to_id[sedml_var]] for sedml_var in sed_to_id}

def _map_sedml_to_sbml_ids(variables: list[Variable]) -> dict[Variable, str]:
    sedml_var_to_sbml_id: dict[Variable, str] = {}
    sedml_var_to_sbml_id.update(_map_sedml_symbol_to_sbml_id(variables))
    sedml_var_to_sbml_id.update(_map_sedml_target_to_sbml_id(variables))
    return sedml_var_to_sbml_id

def _map_sbml_id_to_copasi_name(for_output: bool) -> dict[str, str]:
    # NB: usually, sbml_id == copasi name, with exceptions like "Time"
    compartments: pandas.DataFrame = basico.get_compartments()
    metabolites: pandas.DataFrame = basico.get_species()
    reactions: pandas.DataFrame = basico.get_reactions()
    compartment_mapping: dict[str, str]
    metabolites_mapping: dict[str, str]
    reactions_mapping: dict[str, str]
    sbml_id_to_sbml_name_map: dict[str, str]

    # Create mapping
    if for_output:
        compartment_mapping = \
            {compartments.at[row, "sbml_id"]: format_to_copasi_compartment_name(str(row))
             for row in compartments.index}
        metabolites_mapping = \
            {metabolites.at[row, "sbml_id"]: format_to_copasi_species_concentration_name(str(row))
             for row in metabolites.index}
        reactions_mapping = \
            {reactions.at[row, "sbml_id"]: format_to_copasi_reaction_name(str(row))
             for row in reactions.index}
    else:
        compartment_mapping = {compartments.at[row, "sbml_id"]: str(row) for row in compartments.index}
        metabolites_mapping = {metabolites.at[row, "sbml_id"]: str(row) for row in metabolites.index}
        reactions_mapping = {reactions.at[row, "sbml_id"]: str(row) for row in reactions.index}

    # Combine mappings
    sbml_id_to_sbml_name_map = {"Time": "Time"}
    sbml_id_to_sbml_name_map.update(compartment_mapping)
    sbml_id_to_sbml_name_map.update(metabolites_mapping)
    sbml_id_to_sbml_name_map.update(reactions_mapping)
    return sbml_id_to_sbml_name_map

def _map_sedml_symbol_to_sbml_id(variables: list[Variable]) -> dict[Variable, str]:
    symbol_mappings = {"kisao:0000832": "Time", "urn:sedml:symbol:time": "Time"}
    symbolic_variables: list[Variable]
    raw_symbols: list[str]
    symbols: list[Union[str, None]]

    # Process the variables
    symbolic_variables = [variable for variable in variables if variable.symbol is not None]
    raw_symbols = [str(variable.symbol).lower() for variable in variables if variable.symbol is not None]
    symbols = [symbol_mappings.get(variable, None) for variable in raw_symbols]

    if None in symbols:
        raise ValueError(f"BioSim COPASI is unable to interpret symbol '{raw_symbols[symbols.index(None)]}'")

    return {sedml_var:copasi_name for sedml_var, copasi_name in zip(symbolic_variables, symbols)}

def _map_sedml_target_to_sbml_id(variables: list[Variable]) -> dict[Variable, str]:
    targetable_variables: list[Variable]
    raw_targets: list[str]
    targets: list[str]

    targetable_variables = [variable for variable in variables if list(variable.to_tuple())[2] is not None]
    raw_targets = [str(list(variable.to_tuple())[2]) for variable in targetable_variables]
    targets = [target[(target.find('@id=\'') + 5):target.find('\']')] for target in raw_targets]

    return {sedml_var:copasi_name for sedml_var, copasi_name in zip(targetable_variables, targets)}

def _calc_number_of_simulation_steps(sim: UniformTimeCourseSimulation, duration: float) -> int:
    try:
        step_number_arg = sim.number_of_points * duration / (sim.output_end_time - sim.output_start_time)
    except ZeroDivisionError:  # sim.output_end_time == sim.output_start_time
        if sim.output_start_time != sim.initial_time:
            raise ValueError('Output end time must be greater than the output start time.')
        step_number_arg = sim.number_of_points

    if step_number_arg != math.floor(step_number_arg):
        raise TypeError('Time course must specify an integer number of time points')

    return int(step_number_arg)


def format_to_copasi_reaction_name(sbml_name: str):
    return f"({sbml_name}).Flux"


def format_to_copasi_species_concentration_name(sbml_name: str):
    return f"[{sbml_name}]"


def format_to_copasi_compartment_name(sbml_name: str):
    return f"Compartments[{sbml_name}].Volume"
