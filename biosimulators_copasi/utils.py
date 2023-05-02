""" Utilities for working with the maps from KiSAO ids to COPASI methods and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from types import FunctionType #PR: 55 
from typing import Dict, List, Tuple, Union, Optional #PR: 55
from biosimulators_copasi.data_model import KISAO_ALGORITHMS_MAP, KISAO_PARAMETERS_MAP, Units
from biosimulators_utils.combine.data_model import CombineArchiveContentFormat
from biosimulators_utils.combine.io import CombineArchiveReader, CombineArchiveWriter
from biosimulators_utils.config import get_config, Config  # noqa: F401
from biosimulators_utils.data_model import ValueType
from biosimulators_utils.simulator.utils import get_algorithm_substitution_policy
from biosimulators_utils.utils.core import validate_str_value, parse_value
from kisao.data_model import AlgorithmSubstitutionPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from kisao.utils import get_preferred_substitute_algorithm_by_ids
import COPASI
import itertools
import libsedml
import lxml
import os
import shutil
import tempfile

__all__ = [
    'get_algorithm_id',
    'set_algorithm_parameter_value',
    'get_copasi_model_object_by_sbml_id',
    'get_copasi_model_obj_sbml_ids',
    'fix_copasi_generated_combine_archive',
]


def get_algorithm_id(kisao_id: str, events: bool = False, config: Optional[Config] = None) -> Tuple[str, int]:
    """ Get the COPASI id for an algorithm

    Args:
        kisao_id (:obj:`str`): KiSAO algorithm id
        events (:obj:`bool`, optional): whether an algorithm that supports
            events is needed
        config (:obj:`Config`, optional): configuration

    Returns:
        :obj:`tuple`:

            * :obj:`str`: KiSAO id of algorithm to execute
            * :obj:`int`: COPASI id for algorithm
    """
    possible_alg_kisao_ids = [
        id
        for id, props in KISAO_ALGORITHMS_MAP.items() # noqa python:S3776
        if not events or props['supports_events'] # noqa python:S3776
    ]

    substitution_policy = get_algorithm_substitution_policy(config=config)
    try:
        exec_kisao_id = get_preferred_substitute_algorithm_by_ids(
            kisao_id, possible_alg_kisao_ids,
            substitution_policy=substitution_policy)
    except NotImplementedError: # noqa python:S3776
        if ( # noqa python:S3776
            events
            and kisao_id in ['KISAO_0000561', 'KISAO_0000562']
            and (
                ALGORITHM_SUBSTITUTION_POLICY_LEVELS[substitution_policy] >=
                ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgorithmSubstitutionPolicy.SIMILAR_APPROXIMATIONS]
            )
        ):
            exec_kisao_id = 'KISAO_0000563'
        else: # noqa python:S3776
            exec_kisao_id = kisao_id

    alg = KISAO_ALGORITHMS_MAP[exec_kisao_id]
    return (exec_kisao_id, getattr(COPASI.CTaskEnum, 'Method_' + alg['id']))


def set_algorithm_parameter_value(algorithm_kisao_id: str, algorithm_function: FunctionType, parameter_kisao_id: str, value: str) -> Dict:
    """ Set a parameter of a COPASI simulation function

    Args:
        algorithm_kisao_id (:obj:`str`): KiSAO algorithm id
        algorithm_function (:obj:`types.FunctionType`): algorithm function
        parameter_kisao_id (:obj:`str`): KiSAO parameter id
        value (:obj:`string`): parameter value

    Returns:
        :obj:`dict`: names of the COPASI parameters that were set and their values
    """
    algorithm_name = KISAO_ALGORITHMS_MAP.get(algorithm_kisao_id, {}).get('name', 'N/A')

    parameter_attrs = KISAO_PARAMETERS_MAP.get(parameter_kisao_id, None)
    if parameter_attrs is None: # noqa python:S3776
        supported_parameters = '\n  - '.join(sorted('{}: {}'.format(id, val['name']) for id, val in KISAO_PARAMETERS_MAP.items())) # noqa python:S3776
        raise NotImplementedError("Parameter '{}' is not supported. COPASI supports the following parameters:\n  - {}".format(
            parameter_kisao_id, supported_parameters))

    if isinstance(parameter_attrs['name'], str): # noqa python:S3776
        parameter_name = parameter_attrs['name']
    else: # noqa python:S3776
        parameter_name = parameter_attrs['name'].get(algorithm_kisao_id, 'N/A')
    parameter = algorithm_function.getParameter(parameter_name)
    if not isinstance(parameter, COPASI.CCopasiParameter): # noqa python:S3776
        alg_params = []
        for param_id, param_props in KISAO_PARAMETERS_MAP.items(): # noqa python:S3776
            if algorithm_kisao_id in param_props['algorithms']: # noqa python:S3776
                alg_params.append('{}: {}'.format(param_id, param_props['name']))

        raise NotImplementedError("".join([
            "Algorithm {} ({}) does not support parameter {} ({}). ".format(
                algorithm_kisao_id, algorithm_name,
                parameter_kisao_id, parameter_name),
            "The algorithm supports the following parameters:\n  - {}".format(
                "\n  - ".join(sorted(alg_params))),
        ]))

    if parameter_attrs['type'] == ValueType.boolean: # noqa python:S3776
        param_type_name = 'Boolean'
        param_set_method = parameter.setBoolValue

    elif parameter_attrs['type'] == ValueType.integer: # noqa python:S3776
        param_type_name = 'integer'
        param_set_method = parameter.setIntValue

    elif parameter_attrs['type'] == ValueType.float: # noqa python:S3776
        param_type_name = 'float'
        param_set_method = parameter.setDblValue

    elif parameter_attrs['type'] == ValueType.list: # noqa python:S3776
        param_type_name = 'list'
        param_set_method = None

    else:  # noqa python:S3776 # pragma: no cover # unreachable because the above cases cover all used parameter types
        raise NotImplementedError("Parameter type '{}' is not supported".format(parameter_attrs['type']))

    if not validate_str_value(value, parameter_attrs['type']): # noqa python:S3776
        raise ValueError("'{}' is not a valid {} value for parameter {}".format(value, param_type_name, parameter_kisao_id))

    parsed_value = parse_value(value, parameter_attrs['type'])

    args = {}

    # Set values of basic parameters
    if param_set_method: # noqa python:S3776
        if not param_set_method(parsed_value): # noqa python:S3776 # pragma: no cover # unreachable due to above error checking
            raise RuntimeError('Value of parameter {} ({}) could not be set'.format(parameter_kisao_id, parameter_name))
        args[parameter_name] = parsed_value

    # if the parameter is the random number generator seed (KISAO_0000488), turn on the flag to use it
    if parameter_kisao_id == 'KISAO_0000488': # noqa python:S3776
        use_rand_seed_parameter = algorithm_function.getParameter('Use Random Seed')
        if not isinstance(use_rand_seed_parameter, COPASI.CCopasiParameter): # noqa python:S3776 # pragma: no cover # unreachable due to above error checking
            raise NotImplementedError("Random seed could not be turned on for algorithm {} ({})".format(
                algorithm_kisao_id, algorithm_name))
        if not use_rand_seed_parameter.setBoolValue(True): # noqa python:S3776  # pragma: no cover # unreachable because :obj:`True` is a valid input
            raise RuntimeError("Value of parameter '{}' could not be set".format("Use Random Seed"))
        args['Use Random Seed'] = True

    # set the partitioning strategy parameter
    if parameter_kisao_id == 'KISAO_0000534': # noqa python:S3776
        # set partitioning strategy to user specified
        if not algorithm_function.getParameter('Partitioning Strategy').setStringValue('User specified Partition'): # noqa python:S3776
            raise RuntimeError("'Partitioning Strategy' parameter could not be set for {} ({})".format(
                algorithm_kisao_id, algorithm_name))  # pragma: no cover # unreachable due to earlier validation
        args['Partitioning Strategy'] = 'User specified Partition'

        # build mapping from the SBML ids of reactions to their common names
        object_data_model = algorithm_function.getObjectDataModel()
        sbml_id_to_common_name = {}
        for reaction in object_data_model.getModel().getReactions(): # noqa python:S3776
            if not isinstance(reaction, COPASI.CReaction): # noqa python:S3776
                raise RuntimeError(
                    'Reaction must be an instance of CReaction'
                )  # pragma: no cover # unreachable because getReactions returns instances of :obj:`COPASI.CReaction`
            sbml_id_to_common_name[reaction.getSBMLId()] = COPASI.CRegisteredCommonName(reaction.getCN())

        # clean up any previously defined partitioning
        parameter.clear()
        args[parameter_name] = []

        # set partitioning
        for sbml_rxn_id in parsed_value: # noqa python:S3776
            rxn_common_name = sbml_id_to_common_name.get(sbml_rxn_id, None)
            if not rxn_common_name: # noqa python:S3776
                raise ValueError("The value of {} ({}) must be a list of the ids of reactions. '{}' is not an id of a reaction.".format(
                    parameter_kisao_id, parameter_name, sbml_rxn_id))

            sub_parameter = COPASI.CCopasiParameter('Reaction', COPASI.CCopasiParameter.Type_CN)
            if not sub_parameter.setCNValue(rxn_common_name): # noqa python:S3776
                raise NotImplementedError("Partitioning cannot not be set via reaction common names."
                                          )  # pragma: no cover # unreachable due to above validation
            parameter.addParameter(sub_parameter)

            args[parameter_name].append(rxn_common_name)

    return args


def get_copasi_model_object_by_sbml_id(model: COPASI.CModel, id: str, units: Units) -> Union[COPASI.CCompartment, COPASI.CMetab, COPASI.CModelValue, COPASI.CReaction]:
    """ Get a COPASI model object by its SBML id

    Args:
        model (:obj:`COPASI.CModel`): model
        id (:obj:`str`): SBML id
        units (:obj:`Units`): desired units for the object

    Returns:
        :obj:`COPASI.CCompartment`, :obj:`COPASI.CMetab`, :obj:`COPASI.CModelValue`, or :obj:`COPASI.CReaction`:
            model object
    """
    for object in model.getMetabolites(): # noqa python:S3776
        if object.getSBMLId() == id: # noqa python:S3776
            if units == Units.discrete: # noqa python:S3776
                return object.getValueReference()
            else: # noqa python:S3776
                return object.getConcentrationReference()

    for object in model.getModelValues(): # noqa python:S3776
        if object.getSBMLId() == id: # noqa python:S3776
            return object.getValueReference()
 
    for object in model.getCompartments(): # noqa python:S3776
        if object.getSBMLId() == id: # noqa python:S3776
            return object.getValueReference()

    for object in model.getReactions(): # noqa python:S3776
        if object.getSBMLId() == id: # noqa python:S3776
            return object.getFluxReference()

    return None


def get_copasi_model_obj_sbml_ids(model: COPASI.CModel) -> List[str]:
    """ Get the SBML id of each object of a COPASI model

    Args:
        model (:obj:`COPASI.CModel`): model

    Returns:
        :obj:`list` of :obj:`str: SBML id of each object of the model
    """
    ids = []

    for object in itertools.chain( # noqa python:S3776
        model.getMetabolites(),
        model.getModelValues(),
        model.getCompartments(),
        model.getReactions()
    ):
        ids.append(object.getSBMLId())

    return ids


def fix_copasi_generated_combine_archive(in_filename: str, out_filename: str, config: Optional[Config]=None) -> None:
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
    config = config or get_config() # noqa python:S3776
    archive_dirname = tempfile.mkdtemp()
    try:
        archive = CombineArchiveReader().run(in_filename, archive_dirname, config=config)
    except Exception: # noqa python:S3776
        shutil.rmtree(archive_dirname)
        raise

    # correct URI for COPASI application format
    for content in archive.contents: # noqa python:S3776
        if content.format == 'application/x-copasi':  # noqa python:S3776
            content.format = CombineArchiveContentFormat.CopasiML
            # potentially issue warning messages if needed
            break

    # add SBML namespace to SED-ML file
    ns = None
    for content in archive.contents: # noqa python:S3776
        if content.format == 'https://identifiers.org/combine.specifications/sbml': # noqa python:S3776
            with open(os.path.join(archive_dirname, content.location), 'rb') as sbml: # noqa python:S3776
                root = lxml.etree.parse(sbml)
                # get default ns
                ns = root.getroot().nsmap[None]
                break

    if ns: # noqa python:S3776
        for content in archive.contents: # noqa python:S3776
            if content.format == 'https://identifiers.org/combine.specifications/sed-ml': # noqa python:S3776
                sedml_file = os.path.join(archive_dirname, content.location)
                doc = libsedml.readSedMLFromFile(sedml_file)
                sedml_ns = doc.getSedNamespaces().getNamespaces()
                if not sedml_ns.hasPrefix('sbml'): # noqa python:S3776
                    sedml_ns.add(ns, 'sbml')
                    libsedml.writeSedMLToFile(doc, sedml_file)
                    # potentially issue warning message here, that the sedml file had no SBML prefix and it was added
                    break

    try:
        CombineArchiveWriter().run(archive, archive_dirname, out_filename)
    finally:
        shutil.rmtree(archive_dirname)
