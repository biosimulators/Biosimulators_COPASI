""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-11-17
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""
from __future__ import annotations

from typing import Optional, Tuple, Union, List, Dict

import biosimulators_utils.combine.exec as bsu_combine
import biosimulators_utils.sedml.exec as bsu_exec
import biosimulators_utils.config as bsu_config
import biosimulators_utils.simulator.utils as bsu_sim_utils
import biosimulators_utils.utils.core as bsu_util_core
import pandas

import biosimulators_copasi.data_model as data_model
import biosimulators_copasi.utils as utils

from biosimulators_utils.config import get_config, Config  # noqa: F401
from biosimulators_utils.log.data_model import TaskLog, StandardOutputErrorCapturerLevel, SedDocumentLog, \
    CombineArchiveLog  # noqa: F401
from biosimulators_utils.viz.data_model import VizFormat  # noqa: F401
from biosimulators_utils.report.data_model import ReportFormat, VariableResults, SedDocumentResults, \
    ReportResults  # noqa: F401
from biosimulators_utils.sedml.data_model import \
    Algorithm, Task, Model, Simulation, ModelLanguage, ModelChange, ModelAttributeChange, \
    UniformTimeCourseSimulation, SteadyStateSimulation, Variable, SedDocument  # noqa: F401
from biosimulators_utils.sedml import validation
from biosimulators_utils.utils.core import raise_errors_warnings
from biosimulators_utils.warnings import warn, BioSimulatorsWarning
from kisao.data_model import AlgorithmSubstitutionPolicy as AlgSubPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from biosimulators_copasi.data_model import Units

import basico
import COPASI
import lxml
import numpy
import os
import tempfile
import sys

__all__ = ['get_simulator_version', 'exec_sedml_docs_in_combine_archive', 'exec_sed_doc',
           'exec_sed_task', 'preprocess_sed_task']

CURRENT_PLATFORM = sys.platform
DEFAULT_STDOUT_LEVEL = StandardOutputErrorCapturerLevel.python if "Darwin" in CURRENT_PLATFORM else StandardOutputErrorCapturerLevel.c  # noqa python:S3776
proper_args: dict = {}


def get_simulator_version():
    """ Get the version of COPASI

    Returns:
        :obj:`str`: version
    """
    return basico.__version__


def exec_sedml_docs_in_combine_archive(archive_filename: str, out_dir: str, config: Optional[Config] = None,
                                       fix_copasi_generated_combine_archive: Optional[bool] = None) -> \
        Tuple[SedDocumentResults, CombineArchiveLog]:
    """ Execute the SED tasks defined in a COMBINE/OMEX archive and save the outputs

    Args:
        archive_filename (:obj:`str`): path to COMBINE/OMEX archive
        out_dir (:obj:`str`): path to store the outputs of the archive

            * CSV: directory in which to save outputs to files
              ``{ out_dir }/{ relative-path-to-SED-ML-file-within-archive }/{ report.id }.csv``
            * HDF5: directory in which to save a single HDF5 file (``{ out_dir }/reports.h5``),
              with reports at keys ``{ relative-path-to-SED-ML-file-within-archive }/{ report.id }`` within the HDF5 file

        config (:obj:`Config`, optional): BioSimulators common configuration
        should_fix_copasi_generated_combine_archive (:obj:`bool`, optional): Whether to make COPASI-generated COMBINE archives
            compatible with the specifications of the OMEX manifest and SED-ML standards

    Returns:
        :obj:`tuple`:

            * :obj:`SedDocumentResults`: results
            * :obj:`CombineArchiveLog`: log
    """
    # if fix_copasi_generated_combine_archive is None:
    #     should_fix_copasi_generated_combine_archive = os.getenv('FIX_COPASI_GENERATED_COMBINE_ARCHIVE',
    #                                                             '0').lower() in ['1', 'true']

    if fix_copasi_generated_combine_archive:
        archive_filename = _get_copasi_fixed_archive(archive_filename)

    result = bsu_combine.exec_sedml_docs_in_archive(exec_sed_doc, archive_filename, out_dir,
                                                    apply_xml_model_changes=True, config=config)

    if fix_copasi_generated_combine_archive:
        os.remove(archive_filename)

    return result


def exec_sed_doc(doc: Union[SedDocument, str], working_dir: str, base_out_path: str, rel_out_path: Optional[str] = None,
                 apply_xml_model_changes: bool = True, log: Optional[SedDocumentLog] = None,
                 indent: int = 0, pretty_print_modified_xml_models: bool = False,
                 log_level: Optional[StandardOutputErrorCapturerLevel] = DEFAULT_STDOUT_LEVEL,
                 config: Optional[Config] = None) -> Tuple[ReportResults, SedDocumentLog]:
    """ Execute the tasks specified in a SED document and generate the specified outputs

    Args:
        doc (:obj:`SedDocument` or :obj:`str`): SED document or a path to SED-ML file which defines a SED document
        working_dir (:obj:`str`): working directory of the SED document (path relative to which models are located)
        base_out_path (:obj:`str`): path to store the outputs

            * CSV: directory in which to save outputs to files
              ``{base_out_path}/{rel_out_path}/{report.id}.csv``
            * HDF5: directory in which to save a single HDF5 file (``{base_out_path}/reports.h5``),
              with reports at keys ``{rel_out_path}/{report.id}`` within the HDF5 file

        rel_out_path (:obj:`str`, optional): path relative to :obj:`base_out_path` to store the outputs
        apply_xml_model_changes (:obj:`bool`, optional): if :obj:`True`, apply any model changes specified in the SED-ML file before
            calling :obj:`task_executer`.
        log (:obj:`SedDocumentLog`, optional): log of the document
        indent (:obj:`int`, optional): degree to indent status messages
        pretty_print_modified_xml_models (:obj:`bool`, optional): if :obj:`True`, pretty print modified XML models
        log_level (:obj:`StandardOutputErrorCapturerLevel`, optional): level at which to log output
        config (:obj:`Config`, optional): BioSimulators common configuration
        simulator_config (:obj:`SimulatorConfig`, optional): tellurium configuration

    Returns:
        :obj:`tuple`:

            * :obj:`ReportResults`: results of each report
            * :obj:`SedDocumentLog`: log of the document
    """
    result = bsu_exec.exec_sed_doc(exec_sed_task, doc, working_dir, base_out_path,
                                   rel_out_path=rel_out_path,
                                   apply_xml_model_changes=apply_xml_model_changes,
                                   log=log,
                                   indent=indent,
                                   pretty_print_modified_xml_models=pretty_print_modified_xml_models,
                                   log_level=log_level,
                                   config=config)
    return result


def exec_sed_task(task: Task, variables: List[Variable], preprocessed_task: Optional[Dict] = None,
                  log: Optional[TaskLog] = None, config: Optional[Config] = None) -> Tuple[VariableResults, TaskLog]:
    ''' Execute a task and save its results

    Args:
        task (:obj:`Task`): task
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        preprocessed_task (:obj:`dict`, optional): preprocessed information about the task, including possible
            model changes and variables. This can be used to avoid repeatedly executing the same initialization
            for repeated calls to this method.
        log (:obj:`TaskLog`, optional): log for the task
        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`tuple`:

            :obj:`VariableResults`: results of variables
            :obj:`TaskLog`: log

    Raises:
        :obj:`ValueError`: if the task or an aspect of the task is not valid, or the requested output variables
            could not be recorded
        :obj:`NotImplementedError`: if the task is not of a supported type or involves an unsupported feature
    '''
    config: Config = config or bsu_config.get_config()

    if config.LOG and not log:
        log: TaskLog = TaskLog()

    if preprocessed_task is None:
        preprocessed_task = preprocess_sed_task(task, variables, config)

    if not isinstance(task.simulation, (UniformTimeCourseSimulation, SteadyStateSimulation)):
        raise NotImplementedError(f"Simulation type `{str(type(task.simulation))}` not currently supported")

    # Continued Initialization: Give preprocessed task the new task
    preprocessed_task.configure_simulation_settings(task.simulation)

    # Penultimate Initialization: Apply parameter overrides to model
    _apply_model_changes(task.model, preprocessed_task.algorithm)

    # Final Initialization: Process model changes
    _load_algorithm_parameters(task.simulation, preprocessed_task.algorithm, config)

    # prepare task
    basico.set_task_settings(preprocessed_task.task_type, preprocessed_task.get_simulation_configuration())

    # Execute Simulation
    data: pandas.DataFrame
    dh: COPASI.CDataHandler
    columns: "list"
    dh, columns = preprocessed_task.generate_data_handler(preprocessed_task.get_output_selection())
    preprocessed_task.basico_data_model.addInterface(dh)
    if preprocessed_task.task_type == basico.T.STEADY_STATE:
        basico.run_steadystate(**(preprocessed_task.get_run_configuration()))
    else:
        basico.run_time_course(**(preprocessed_task.get_run_configuration()))
        # data = basico.run_time_course_with_output(**(preprocessed_task.get_run_configuration()))
    data = basico.get_data_from_data_handler(dh, columns)
    preprocessed_task.basico_data_model.removeInterface(dh)

    # Process output 'data'
    copasi_output_length, _ = data.shape
    variable_results = VariableResults()
    offset = preprocessed_task.init_time_offset

    try:
        for variable in variables:
            data_target = preprocessed_task.get_copasi_name(variable)
            series: pandas.Series
            try:
                series = data.loc[:, data_target]
            except KeyError as e:
                raise RuntimeError("Unable to find output", e)
            # Check for duplicates (yes, that can happen)
            if isinstance(series, pandas.DataFrame):
                _, num_cols = series.shape
                first_sub_series: pandas.Series = series.iloc[:, 0]
                for i in range(1, num_cols):
                    if not first_sub_series.equals(series.iloc[:, i]):
                        raise RuntimeError("Different data sets for same variable")
                series: pandas.Series = first_sub_series
            variable_results[variable.id] = numpy.full(copasi_output_length, numpy.nan)
            for index, value in enumerate(series):
                adjusted_value = value if data_target != "Time" else value + offset
                variable_results[variable.id][index] = adjusted_value
    except Exception as e:
        raise e

    # log action
    if config.LOG:
        log.algorithm = preprocessed_task.get_kisao_id_for_kisao_algorithm()
        log.simulator_details = {
            'methodName': preprocessed_task.get_copasi_algorithm_id(),
            'parameters': None,
        }

    # return results and log
    return variable_results, log


def get_copasi_error_message(sim: Simulation, details=None):
    """ Get an error message from COPASI

    Args:
        sim (:obj:`Simulation`): KiSAO id of algorithm of failed simulation
        details (:obj:`str`, optional): details of of error

    Returns:
        :obj:`str`: COPASI error message
    """
    error_msg = f"Simulation with name {sim.name}(id='{sim.id}')"
    if not details:
        details = COPASI.CCopasiMessage.getLastMessage().getText()
    if details:
        details = '\n'.join(line[min(2, len(line)):] for line in details.split('\n')
                            if not (line.startswith('>') and line.endswith('<')))
        error_msg += ':\n\n  ' + details.replace('\n', '\n  ')
    return error_msg


def preprocess_sed_task(task: Task, variables: list[Variable],
                        config: Config = None) -> data_model.BasicoInitialization:
    """ Preprocess a SED task, including its possible model changes and variables. This is useful for avoiding
    repeatedly initializing tasks on repeated calls of :obj:`exec_sed_task`.

    Args:
        task (:obj:`Task`): task
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`BasicoInitialization`: prepared information about the task
    """
    config: Config = config if config is not None else bsu_config.get_config()

    # Get model and simulation
    model: Model = task.model
    sim: Simulation = task.simulation

    # Validate provided simulation description
    _validate_sedml(config, task, model, sim, variables)

    # Confirm UTC Simulation
    if not isinstance(sim, (UniformTimeCourseSimulation, SteadyStateSimulation)):
        raise ValueError("BioSimulators-COPASI can only handle UTC and Steady State "
                         "Simulations in this API for the time being")
    utc_sim: Union[UniformTimeCourseSimulation, SteadyStateSimulation] = sim

    # instantiate model
    basico_data_model: COPASI.CDataModel
    try:
        basico_data_model = \
           basico.import_sbml(model.source, annotations_to_remove=[('initialValue', 'http://copasi.org/initialValue')])
    except COPASI.CCopasiException as e:
        raise ValueError(f"SBML '{model.source}' could not be imported into COPASI;\n\t", e)

    # process solution algorithm
    has_events: bool = basico_data_model.getModel().getNumEvents() >= 1
    copasi_algorithm = utils.get_algorithm(utc_sim.algorithm.kisao_id, has_events, config=config)

    # Create and return preprocessed simulation settings
    preprocessed_info = data_model.BasicoInitialization(basico_data_model, copasi_algorithm, variables, has_events)
    return preprocessed_info


def _get_copasi_fixed_archive(archive_filename: bytes | str):
    temp_archive_file: int
    temp_archive_filename: bytes

    temp_archive_file, temp_archive_filename = tempfile.mkstemp()
    os.close(temp_archive_file)
    utils.fix_copasi_generated_combine_archive(archive_filename, temp_archive_filename)
    return temp_archive_filename


def _validate_sedml(config: Config, task: Task, model: Model, sim: Simulation, variables: list[Variable]):
    # Prepare error messages
    invalid_task: str = f'Task `{task.id}` is invalid.'
    invalid_model_lang: str = f'Language for model `{model.id}` is not supported.'
    invalid_model_changes: str = f'Changes for model `{model.id}` are not supported.'
    invalid_sim_type: str = f'{sim.__class__.__name__,} `{sim.id}` is not supported.'
    invalid_sim: str = 'Simulation `{sim.id}` is invalid.'
    invalid_data_gen: str = f'Data generator variables for task `{task.id}` are invalid.'

    # run validations
    if config.VALIDATE_SEDML:
        task_errors = validation.validate_task(task)
        model_lang_errors = validation.validate_model_language(model.language, ModelLanguage.SBML)
        model_change_type_errors = validation.validate_model_change_types(model.changes, (ModelAttributeChange,))
        valid_types = (UniformTimeCourseSimulation, SteadyStateSimulation)
        simulation_type_errors = validation.validate_simulation_type(sim, valid_types)
        model_change_errors_list = validation.validate_model_changes(model)
        simulation_errors_list = validation.validate_simulation(sim)
        data_generator_errors_list = validation.validate_data_generator_variables(variables)

        # pass results to raise errors and warnings method
        bsu_util_core.raise_errors_warnings(task_errors, error_summary=invalid_task)
        bsu_util_core.raise_errors_warnings(model_lang_errors, error_summary=invalid_model_lang)
        bsu_util_core.raise_errors_warnings(model_change_type_errors, error_summary=invalid_model_changes)
        bsu_util_core.raise_errors_warnings(*model_change_errors_list, error_summary=invalid_model_changes)
        bsu_util_core.raise_errors_warnings(simulation_type_errors, error_summary=invalid_sim_type)
        bsu_util_core.raise_errors_warnings(*simulation_errors_list, error_summary=invalid_sim)
        bsu_util_core.raise_errors_warnings(*data_generator_errors_list, error_summary=invalid_data_gen)

    if config.VALIDATE_SEDML_MODELS:
        model_errors = validation.validate_model(model, [], working_dir='.')
        bsu_util_core.raise_errors_warnings(*model_errors, error_summary=f'Model `{model.id}` is invalid.',
                                            warning_summary=f'Model `{model.id}` may be invalid.')

    # Necessary validation for COPASI's use:
    model_change_error_message = f'Changes for model `{model.id}` are not supported.'
    model_etree = lxml.etree.parse(model.source)
    validation.validate_target_xpaths(model.changes, model_etree, attr='id')
    validation.validate_target_xpaths(variables, model_etree, attr='id')
    errors_model_changes = validation.validate_model_change_types(model.changes, (ModelAttributeChange,))
    raise_errors_warnings(errors_model_changes, error_summary=model_change_error_message)


def _apply_model_changes(sedml_model: Model, copasi_algorithm: utils.CopasiAlgorithm) \
        -> tuple[list[ModelAttributeChange], list[ModelChange]]:
    legal_changes: list[ModelAttributeChange] = []
    illegal_changes: list[ModelChange] = []

    # If there's no changes, get out of here
    if not sedml_model.changes:
        return legal_changes, illegal_changes

    # check if there's anything but ChangeAttribute type changes
    change: ModelChange
    for change in sedml_model.changes:
        legal_changes.append(change) if isinstance(change, ModelAttributeChange) else illegal_changes.append(change)
    pseudo_variable_map = {change: Variable(change.id, change.name, change.target, change.target_namespaces)
                           for change in legal_changes}
    variable_to_target_sbml_id = data_model.CopasiMappings.map_sedml_to_sbml_ids(list(pseudo_variable_map.values()))
    change_to_sbml_id_map = {change: variable_to_target_sbml_id[pseudo_variable_map[change]]
                             for change in legal_changes}
    units = copasi_algorithm.get_unit_set()
    model_change: ModelAttributeChange
    for model_change in change_to_sbml_id_map.keys():
        sbml_id = change_to_sbml_id_map[model_change]
        metabolite = basico.get_species(sbml_id=sbml_id)
        metabolite = metabolite if metabolite is not None else basico.get_species(sbml_id)  # Pretend it's the name
        if metabolite is not None:
            metab_map = basico.as_dict(metabolite)
            if units == Units.continuous:
                basico.set_species(metab_map["name"], initial_concentration=model_change.new_value)
            else:
                basico.set_species(metab_map["name"], initial_particle_number=model_change.new_value)
            continue

        compartment = basico.get_compartments(sbml_id=sbml_id)
        if compartment is not None:
            compart_map = basico.as_dict(compartment)
            basico.set_compartment(compart_map["name"], initial_size=model_change.new_value)
            continue

        # reaction = basico.get_reactions(sbml_id=sbml_id)
        # if reaction is not None:
        #    basico.set_reaction(sbml_id=sbml_id, ???)
        #    continue

        illegal_changes.append(model_change)
        raise ValueError(f"Change [id='{model_change.id}', name='{model_change.name}', " +
                         f"target='{model_change.target}', value='{model_change.new_value}' is invalid!")

    return legal_changes, illegal_changes


def _load_algorithm_parameters(sim: Simulation, copasi_algorithm: utils.CopasiAlgorithm, config: Config = None):
    # Load the algorithm parameter changes specified by `simulation.algorithm_parameter_changes`
    algorithm_substitution_policy: AlgSubPolicy = bsu_sim_utils.get_algorithm_substitution_policy(config=config)
    requested_algorithm: Algorithm = sim.algorithm
    if copasi_algorithm.KISAO_ID != requested_algorithm.kisao_id:
        return

    unsupported_parameters, bad_parameters = utils.set_algorithm_parameter_values(copasi_algorithm,
                                                                                  requested_algorithm.changes)
    if len(unsupported_parameters) + len(bad_parameters) == 0:
        return  # No problems

    selected_sub_policy = ALGORITHM_SUBSTITUTION_POLICY_LEVELS[algorithm_substitution_policy]
    zero_substitution_policy = ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgSubPolicy.NONE]
    # If we were asked for a "zero-tolerance" substitution policy, we need to raise an exception
    # Otherwise, just create some warnings

    for change in unsupported_parameters:
        parameter_str = f'Parameter `{change.kisao_id}`'
        if selected_sub_policy <= zero_substitution_policy:
            raise NotImplementedError(parameter_str + " is not supported\n", change)
        warn(parameter_str + " was ignored:\n", BioSimulatorsWarning)

    for change in bad_parameters:
        if selected_sub_policy == zero_substitution_policy:
            raise ValueError(f"'{change}' is not a valid change with a 'NONE' substitution policy.")
        warning_message = 'Invalid or unsupported value `{}` for algorithm parameter `{}` encountered:\n'
        warn(warning_message.format(change.new_value, change.kisao_id), BioSimulatorsWarning)

    return
