""" Utilities for working with the maps from KiSAO ids to COPASI methods and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from .data_model import KISAO_ALGORITHMS_MAP, KISAO_PARAMETERS_MAP
from biosimulators_utils.data_model import ValueType
from biosimulators_utils.simulator.data_model import AlgorithmSubstitutionPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from biosimulators_utils.simulator.exceptions import AlgorithmCannotBeSubstitutedException
from biosimulators_utils.simulator.utils import get_algorithm_substitution_policy
from biosimulators_utils.simulator.warnings import AlgorithmSubstitutedWarning
from biosimulators_utils.utils.core import validate_str_value, parse_value
import COPASI
import warnings

__all__ = ['get_algorithm_id', 'set_algorithm_parameter_value']


def get_algorithm_id(kisao_id):
    """ Get the COPASI id for an algorithm

    Args:
        kisao_id (:obj:`str`): KiSAO algorithm id

    Returns:
        :obj:`int`: COPASI id for algorithm
    """
    requested_kisao_id = kisao_id

    substitution_policy = get_algorithm_substitution_policy()
    if kisao_id in ['KISAO_0000088', 'KISAO_0000089']:
        alg_name = 'LSODA' if kisao_id == 'KISAO_0000088' else 'LSODAR'
        if (
            ALGORITHM_SUBSTITUTION_POLICY_LEVELS[substitution_policy]
            >= ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgorithmSubstitutionPolicy.SIMILAR_VARIABLES]
        ):
            warnings.warn('Hybrid LSODA/LSODAR method (KISAO_0000560) will be used rather than {} ({}).'.format(
                alg_name, kisao_id),
                AlgorithmSubstitutedWarning)
            kisao_id = 'KISAO_0000560'
        else:
            raise AlgorithmCannotBeSubstitutedException((
                '{} ({}) cannot be substituted to the hybrid LSODA/LSODAR method (KISAO_0000560) '
                'under the current algorithm substitution policy {}.').format(
                alg_name, kisao_id, substitution_policy.name
            ))

    alg = KISAO_ALGORITHMS_MAP.get(kisao_id, None)
    if alg is None:
        raise NotImplementedError(
            "Algorithm with KiSAO id '{}' is not supported".format(requested_kisao_id))
    return getattr(COPASI.CTaskEnum, 'Method_' + alg['id'])


def set_algorithm_parameter_value(algorithm_kisao_id, algorithm_function, parameter_kisao_id, value):
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
    if parameter_attrs is None:
        raise NotImplementedError("Parameter '{}' is not supported. COPASI supports the following parameters:\n  - {}".format(
            parameter_kisao_id, '\n  - '.join(sorted('{}: {}'.format(id, val['name']) for id, val in KISAO_PARAMETERS_MAP.items()))))

    if isinstance(parameter_attrs['name'], str):
        parameter_name = parameter_attrs['name']
    else:
        parameter_name = parameter_attrs['name'].get(algorithm_kisao_id, 'N/A')
    parameter = algorithm_function.getParameter(parameter_name)
    if not isinstance(parameter, COPASI.CCopasiParameter):
        alg_params = []
        for param_id, param_props in KISAO_PARAMETERS_MAP.items():
            if algorithm_kisao_id in param_props['algorithms']:
                alg_params.append('{}: {}'.format(param_id, param_props['name']))

        raise NotImplementedError("".join([
            "Algorithm {} ({}) does not support parameter {} ({}). ".format(
                algorithm_kisao_id, algorithm_name,
                parameter_kisao_id, parameter_name),
            "The algorithm supports the following parameters:\n  - {}".format(
                "\n  - ".join(sorted(alg_params))),
        ]))

    if parameter_attrs['type'] == ValueType.boolean:
        param_type_name = 'Boolean'
        param_set_method = parameter.setBoolValue

    elif parameter_attrs['type'] == ValueType.integer:
        param_type_name = 'integer'
        param_set_method = parameter.setIntValue

    elif parameter_attrs['type'] == ValueType.float:
        param_type_name = 'float'
        param_set_method = parameter.setDblValue

    elif parameter_attrs['type'] == ValueType.list:
        param_type_name = 'list'
        param_set_method = None

    else:  # pragma: no cover # unreachable because the above cases cover all used parameter types
        raise NotImplementedError("Parameter type '{}' is not supported".format(parameter_attrs['type']))

    if not validate_str_value(value, parameter_attrs['type']):
        raise ValueError("'{}' is not a valid {} value for parameter {}".format(value, param_type_name, parameter_kisao_id))

    parsed_value = parse_value(value, parameter_attrs['type'])

    args = {}

    # Set values of basic parameters
    if param_set_method:
        if not param_set_method(parsed_value):  # pragma: no cover # unreachable due to above error checking
            raise RuntimeError('Value of parameter {} ({}) could not be set'.format(parameter_kisao_id, parameter_name))
        args[parameter_name] = parsed_value

    # if the parameter is the random number generator seed (KISAO_0000488), turn on the flag to use it
    if parameter_kisao_id == 'KISAO_0000488':
        use_rand_seed_parameter = algorithm_function.getParameter('Use Random Seed')
        if not isinstance(use_rand_seed_parameter, COPASI.CCopasiParameter):  # pragma: no cover # unreachable due to above error checking
            raise NotImplementedError("Random seed could not be turned on for algorithm {} ({})".format(
                algorithm_kisao_id, algorithm_name))
        if not use_rand_seed_parameter.setBoolValue(True):  # pragma: no cover # unreachable because :obj:`True` is a valid input
            raise RuntimeError("Value of parameter '{}' could not be set".format("Use Random Seed"))
        args['Use Random Seed'] = True

    # set the partitioning strategy parameter
    if parameter_kisao_id == 'KISAO_0000534':
        # set partitioning strategy to user specified
        if not algorithm_function.getParameter('Partitioning Strategy').setStringValue('User specified Partition'):
            raise RuntimeError("'Partitioning Strategy' parameter could not be set for {} ({})".format(
                algorithm_kisao_id, algorithm_name))  # pragma: no cover # unreachable due to earlier validation
        args['Partitioning Strategy'] = 'User specified Partition'

        # build mapping from the SBML ids of reactions to their common names
        object_data_model = algorithm_function.getObjectDataModel()
        sbml_id_to_common_name = {}
        for reaction in object_data_model.getModel().getReactions():
            if not isinstance(reaction, COPASI.CReaction):
                raise RuntimeError(
                    'Reaction must be an instance of CReaction'
                )  # pragma: no cover # unreachable because getReactions returns instances of :obj:`COPASI.CReaction`
            sbml_id_to_common_name[reaction.getSBMLId()] = COPASI.CRegisteredCommonName(reaction.getCN())

        # clean up any previously defined partitioning
        parameter.clear()
        args[parameter_name] = []

        # set partitioning
        for sbml_rxn_id in parsed_value:
            rxn_common_name = sbml_id_to_common_name.get(sbml_rxn_id, None)
            if not rxn_common_name:
                raise ValueError("The value of {} ({}) must be a list of the ids of reactions. '{}' is not an id of a reaction.".format(
                    parameter_kisao_id, parameter_name, sbml_rxn_id))

            sub_parameter = COPASI.CCopasiParameter('Reaction', COPASI.CCopasiParameter.Type_CN)
            if not sub_parameter.setCNValue(rxn_common_name):
                raise NotImplementedError("Partitioning cannot not be set via reaction common names."
                                          )  # pragma: no cover # unreachable due to above validation
            parameter.addParameter(sub_parameter)

            args[parameter_name].append(rxn_common_name)

    return args
