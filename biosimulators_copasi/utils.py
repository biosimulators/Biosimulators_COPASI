""" Utilities for working with the maps from KiSAO ids to COPASI methods and their arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from __future__ import annotations

import builtins
from typing import Optional

from biosimulators_copasi.data_model import CopasiAlgorithmType, CopasiAlgorithm, CopasiAlgorithmParameter
from biosimulators_utils.combine.data_model import CombineArchiveContentFormat
from biosimulators_utils.combine.io import CombineArchiveReader, CombineArchiveWriter
from biosimulators_utils.config import get_config, Config  # noqa: F401
from biosimulators_utils.simulator.utils import get_algorithm_substitution_policy
from biosimulators_utils.sedml.data_model import AlgorithmParameterChange
from kisao.data_model import AlgorithmSubstitutionPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from kisao.utils import get_preferred_substitute_algorithm_by_ids
import libsedml
import lxml
import os
import shutil
import tempfile

__all__ = [
    'get_algorithm',
    'convert_sedml_reactions_to_copasi_reactions',
    'set_algorithm_parameter_values',
    'fix_copasi_generated_combine_archive',
]


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

    algorithm_kisao_to_class_map: dict[str, CopasiAlgorithm] = {
        CopasiAlgorithmType[alg_name].value.KISAO_ID: CopasiAlgorithmType[alg_name].value
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
        return constructor()  # this too is, in fact, callable

    raise ValueError(f"No suitable equivalent for '{kisao_id}' could be found with the provided substitution policy")


def convert_sedml_reactions_to_copasi_reactions(sedml_reactions: list[str]) -> list[str]:
    pass


def set_algorithm_parameter_values(copasi_algorithm: CopasiAlgorithm, requested_changes: list) \
        -> tuple[list[AlgorithmParameterChange], list[AlgorithmParameterChange]]:
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


def fix_copasi_generated_combine_archive(in_filename: str, out_filename: str, config: Optional[Config] = None) -> None:
    """ Utility function that corrects COMBINE/OMEX archives generated by COPASI so they are compatible
    with other tools.

    All currently released versions of COPASI export COMBINE archive files. However, these archives
    presently diverge from the specifications of the SED-ML format.

    * Format in OMEX manifests is not a valid PURL media type URI
    * SED-ML files lack namespaces for SBML

    Args:
        in_filename (:obj:`str`): path to a COMBINE archive to correct
        out_filename (:obj:`str`): path to save corrected COMBINE archive
        config (:obj:`Config`, optional): BioSimulators-utils configuration
    """
    config = config or get_config()
    archive_directory_name = tempfile.mkdtemp()
    try:
        archive = CombineArchiveReader().run(in_filename, archive_directory_name, config=config)
    except Exception:
        shutil.rmtree(archive_directory_name)
        raise

    # correct URI for COPASI application format
    for content in archive.contents:
        if content.format == 'application/x-copasi':
            content.format = CombineArchiveContentFormat.CopasiML
            # potentially issue warning messages if needed
            break

    # add SBML namespace to SED-ML file
    namespace = None
    for content in archive.contents:
        if content.format == 'http://identifiers.org/combine.specifications/sbml':
            with open(os.path.join(archive_directory_name, content.location), 'rb') as sbml:
                root = lxml.etree.parse(sbml)
                # get default namespace
                namespace = root.getroot().nsmap[None]
                break

    if namespace:
        for content in archive.contents:
            if content.format == 'http://identifiers.org/combine.specifications/sed-ml':
                sedml_file = os.path.join(archive_directory_name, content.location)
                doc = libsedml.readSedMLFromFile(sedml_file)
                sedml_ns = doc.getSedNamespaces().getNamespaces()
                if not sedml_ns.hasPrefix('sbml'):
                    sedml_ns.add(namespace, 'sbml')
                    libsedml.writeSedMLToFile(doc, sedml_file)
                    # potentially issue warning message here, that the sedml file had no SBML prefix and it was added
                    break

    try:
        CombineArchiveWriter().run(archive, archive_directory_name, out_filename)
    finally:
        shutil.rmtree(archive_directory_name)
