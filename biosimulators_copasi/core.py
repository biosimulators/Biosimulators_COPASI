""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-11-17
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""
from __future__ import annotations
import typing
from typing import Tuple, List

import biosimulators_utils.combine.exec as bsu_combine
import biosimulators_utils.sedml.exec as bsu_exec
import biosimulators_utils.config as bsu_config
import biosimulators_utils.simulator.utils as bsu_sim_utils
import biosimulators_utils.utils.core as bsu_util_core
import pandas

import biosimulators_copasi.utils as utils

from biosimulators_utils.config import get_config, Config  # noqa: F401
from biosimulators_utils.log.data_model import CombineArchiveLog, TaskLog, \
    StandardOutputErrorCapturerLevel, SedDocumentLog  # noqa: F401
from biosimulators_utils.viz.data_model import VizFormat  # noqa: F401
from biosimulators_utils.report.data_model import ReportFormat, VariableResults, SedDocumentResults  # noqa: F401
from biosimulators_utils.sedml.data_model import (Task, Model, Simulation, ModelLanguage, ModelChange,
                                                  ModelAttributeChange, UniformTimeCourseSimulation,  # noqa: F401
                                                  Variable, Symbol, SedDocument)
from biosimulators_utils.sedml import validation
from biosimulators_utils.utils.core import raise_errors_warnings
from biosimulators_utils.warnings import warn, BioSimulatorsWarning
from kisao.data_model import AlgorithmSubstitutionPolicy as AlgSubPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from biosimulators_copasi.data_model import KISAO_ALGORITHMS_MAP, Units

#from .utils import (get_algorithm_id, set_algorithm_parameter_value, get_copasi_model_object_by_sbml_id, get_copasi_model_obj_sbml_ids)
import basico
import COPASI
import lxml
import math
import numpy
import os
import tempfile

__all__ = ['exec_sedml_docs_in_combine_archive', 'exec_sed_doc', 'exec_sed_task', 'preprocess_sed_task']

proper_args: dict = {}
def exec_sedml_docs_in_combine_archive(archive_filename: str, out_dir: str, config: Config = None,
                                       should_fix_copasi_generated_combine_archive: bool = None) -> tuple:
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
    if should_fix_copasi_generated_combine_archive is None:
        should_fix_copasi_generated_combine_archive = os.getenv('FIX_COPASI_GENERATED_COMBINE_ARCHIVE',
                                                                '0').lower() in ['1', 'true']

    if should_fix_copasi_generated_combine_archive:
        archive_filename = _get_copasi_fixed_archive(archive_filename)

    result = bsu_combine.exec_sedml_docs_in_archive(exec_sed_doc, archive_filename, out_dir,
                                                    apply_xml_model_changes=True, config=config)

    if should_fix_copasi_generated_combine_archive:
        os.remove(archive_filename)

    return result


def exec_sed_doc(doc: SedDocument | str, working_dir: str, base_out_path: str, rel_out_path: str = None,
                 apply_xml_model_changes: bool = True, log: SedDocumentLog = None, indent: int = 0,
                 pretty_print_modified_xml_models: bool = False,
                 log_level: StandardOutputErrorCapturerLevel = StandardOutputErrorCapturerLevel.c,
                 config: Config = None):
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
    return bsu_exec.exec_sed_doc(exec_sed_task, doc, working_dir, base_out_path,
                                 rel_out_path=rel_out_path,
                                 apply_xml_model_changes=apply_xml_model_changes,
                                 log=log,
                                 indent=indent,
                                 pretty_print_modified_xml_models=pretty_print_modified_xml_models,
                                 log_level=log_level,
                                 config=config)


def exec_sed_task(task: Task, variables: list[Variable], preprocessed_task: dict = None,
                  log: TaskLog = None, config: Config = None):
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
        preprocessed_task = alt_preprocess_sed_task(task, variables, config)

    # Execute Simulation
    settings_map: dict = preprocessed_task.get_simulation_configuration()
    data: pandas.DataFrame = basico.run_time_course_with_output(**settings_map)

    # Process output 'data'
    actual_output_length, _ = data.shape
    expected_output_length = preprocessed_task.get_expected_output_length()
    if expected_output_length != actual_output_length:
        msg = f"Length of output does not match expected amount: {actual_output_length} (vs {expected_output_length})"
        raise RuntimeError(msg)

    variable_results = VariableResults()
    for variable in variables:
        variable_results[variable.id] = numpy.full(actual_output_length, numpy.nan)
        data_target = preprocessed_task.get_COPASI_name(variable)
        series: pandas.Series = data.loc[:, data_target]
        for index, value in enumerate(series):
            variable_results[variable.id][index] = value

    # # We need this stuff, but we'll figure out what it's doing first...
    # if sim.output_end_time == sim.output_start_time and sim.output_start_time == sim.initial_time:
    #    for variable in variables:
    #        variable_results[variable.id] = numpy.concatenate((
    #            variable_results[variable.id][0:1],
    #            numpy.full((sim.number_of_points,), variable_results[variable.id][1]),
    #        ))

    # log action
    if config.LOG:
        log.algorithm = preprocessed_task.get_KiSAO_id_for_KiSAO_algorithm()
        log.simulator_details = {
            'methodName': preprocessed_task.get_COPASI_algorithm_name(),
            'methodCode': preprocessed_task.get_COPASI_algorithm_code(),
            'parameters': None,
        }

    # return results and log
    return variable_results, log


def get_copasi_error_message(algorithm_kisao_id, details=None):
    """ Get an error message from COPASI

    Args:
        algorithm_kisao_id (:obj:`str`): KiSAO id of algorithm of failed simulation
        details (:obj:`str`, optional): details of of error

    Returns:
        :obj:`str`: COPASI error message
    """
    error_msg = 'Simulation with algorithm {} ({}) failed'.format(
        algorithm_kisao_id, KISAO_ALGORITHMS_MAP.get(algorithm_kisao_id, {}).get('name', 'N/A'))
    if not details:
        details = COPASI.CCopasiMessage.getLastMessage().getText()
    if details:
        details = '\n'.join(line[min(2, len(line)):] for line in details.split('\n')
                            if not (line.startswith('>') and line.endswith('<')))
        error_msg += ':\n\n  ' + details.replace('\n', '\n  ')
    return error_msg


def alt_preprocess_sed_task(task: Task, variables: list[Variable], config: Config = None) -> utils.BasicoInitialization:
    """ Preprocess a SED task, including its possible model changes and variables. This is useful for avoiding
    repeatedly initializing tasks on repeated calls of :obj:`exec_sed_task`.

    Args:
        task (:obj:`Task`): task
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`dict`: preprocessed information about the task
    """
    config: Config = config or bsu_config.get_config()

    # Get model and simulation
    model: Model = task.model
    sim: Simulation = task.simulation

    # Validate provided simulation description
    if config.VALIDATE_SEDML:
        _validate_sedml(task, model, sim, variables)

    model_change_error_message = f'Changes for model `{model.id}` are not supported.'
    model_etree = lxml.etree.parse(model.source)
    model_change_target_sbml_id_map: dict = validation.validate_target_xpaths(model.changes, model_etree, attr='id')
    variable_target_sbml_id_map: dict = validation.validate_target_xpaths(variables, model_etree, attr='id')
    raise_errors_warnings(validation.validate_model_change_types(model.changes, (ModelAttributeChange,)),
                          error_summary=model_change_error_message)

    if config.VALIDATE_SEDML_MODELS:
        raise_errors_warnings(*validation.validate_model(model, [], working_dir='.'),
                              error_summary=f'Model `{model.id}` is invalid.',
                              warning_summary=f'Model `{model.id}` may be invalid.')

    # instantiate model
    basico_data_model: COPASI.CDataModel = basico.load_model(model.source)

    # process solution algorithm
    exec_alg_kisao_id: str
    alg_copasi_id: int
    alg_copasi_str_id: str
    alg_kisao_id: str = sim.algorithm.kisao_id
    has_events: bool = basico_data_model.getModel().getNumEvents() >= 1
    #exec_alg_kisao_id, alg_copasi_id, alg_copasi_str_id = utils.get_algorithm_id(alg_kisao_id, has_events, config)
    algorithm_info = utils.get_algorithm_id(alg_kisao_id, has_events, config)

    # process model changes
    _apply_model_changes(model, basico_data_model, model_change_target_sbml_id_map, algorithm_info.kisao_id)

    # Initialize solver settings
    method_arg = algorithm_info.copasi_algorithm_name
    if not isinstance(sim, UniformTimeCourseSimulation):
        raise ValueError("BioSimulators-COPASI can only handle UTC Simulations in this API for the time being")
    utc_sim: UniformTimeCourseSimulation = sim
    duration_arg: float = sim.output_end_time - sim.initial_time

    # Create simulation settings
    settings_map = {
        #"output_selection": list(sedml_var_to_copasi_name.values()),
        "use_initial_values": True,
        "update_model": False,
        "method": method_arg,
        "duration": duration_arg,
        "start_time": utc_sim.output_start_time,
        "step_number": _calc_number_of_simulation_steps(sim, duration_arg)
    }

    # return preprocessed info
    preprocessed_info = {
        'task': "",
        'model': {

        },
        'simulation': {

        },
    }
    preprocessed_info = utils.BasicoInitialization()
    return preprocessed_info

def preprocess_sed_task(task: Task, variables: list[Variable], config: Config = None):
    """ Preprocess a SED task, including its possible model changes and variables. This is useful for avoiding
    repeatedly initializing tasks on repeated calls of :obj:`exec_sed_task`.

    Args:
        task (:obj:`Task`): task
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`dict`: preprocessed information about the task
    """
    config: Config = config or bsu_config.get_config()

    model: Model = task.model
    sim: Simulation = task.simulation

    if config.VALIDATE_SEDML:
        _validate_sedml(task, model, sim, variables)

    model_etree = lxml.etree.parse(model.source)
    model_change_target_sbml_id_map: dict = validation.validate_target_xpaths(model.changes, model_etree, attr='id')
    variable_target_sbml_id_map: dict = validation.validate_target_xpaths(variables, model_etree, attr='id')

    if config.VALIDATE_SEDML_MODELS:
        raise_errors_warnings(*validation.validate_model(model, [], working_dir='.'),
                              error_summary=f'Model `{model.id}` is invalid.',
                              warning_summary=f'Model `{model.id}` may be invalid.')

    # Read the SBML-encoded model located at `os.path.join(working_dir, model_filename)`
    basico_data_model: COPASI.CDataModel = basico.load_model(model.source)
    if not basico_data_model:
        #copasi_error_message = get_copasi_error_message(sim.algorithm.kisao_id).replace('\n', "\n  ")
        raise ValueError(f"`{model.source}` could not be imported.")
    basico.set_model_name(f"{model.name}_{task.name}")
    reactions = basico.get_reactions()

    # determine the algorithm to execute
    alg_kisao_id: str = sim.algorithm.kisao_id
    has_events: bool = basico_data_model.getModel().getNumEvents() >= 1
    _, _, alg_copasi_str_id = utils.get_algorithm_id(alg_kisao_id, events=has_events, config=config)

################################################################
    # Read the SBML-encoded model located at `os.path.join(working_dir, model_filename)`
    #copasi_data_model: COPASI.CDataModel = COPASI.CRootContainer.addDatamodel()
    #if not copasi_data_model.importSBML(model.source):
        #copasi_error_message = get_copasi_error_message(sim.algorithm.kisao_id).replace('\n', "\n  ")
        #raise ValueError(f"`{model.source}` could not be imported:\n\n  {copasi_error_message}")
    #copasi_model: COPASI.CModel = copasi_data_model.getModel()

    copasi_data_model = basico_data_model  # Since basico and python-copasi both access the C++ api, this should work
    copasi_model: COPASI.CModel = copasi_data_model.getModel()
    # determine the algorithm to execute
    exec_alg_kisao_id: str
    alg_copasi_id: int
    alg_kisao_id: str = sim.algorithm.kisao_id
    has_events: bool = copasi_model.getNumEvents() >= 1
    exec_alg_kisao_id, alg_copasi_id, alg_copasi_str_id = utils.get_algorithm_id(alg_kisao_id,
                                                                                 events=has_events, config=config)

    # initialize COPASI task
    copasi_task: COPASI.CCopasiTask = copasi_data_model.getTask('Time-Course')

    # Load the algorithm specified by `simulation.algorithm`
    if not copasi_task.setMethodType(alg_copasi_id):
        raise RuntimeError(f'Unable to initialize function for {exec_alg_kisao_id}')
        # pragma: no cover # unreachable because :obj:`get_algorithm_id` returns valid COPASI method ids
    copasi_method: COPASI.CCopasiMethod = copasi_task.getMethod()

    # Apply the algorithm parameter changes specified by `simulation.algorithm_parameter_changes`
    method_parameters = {}
    algorithm_substitution_policy: AlgSubPolicy = bsu_sim_utils.get_algorithm_substitution_policy(config=config)
    if exec_alg_kisao_id == alg_kisao_id:
        for change in sim.algorithm.changes:
            try:
                change_args = utils.set_algorithm_parameter_value(exec_alg_kisao_id, copasi_method,
                                                                  change.kisao_id, change.new_value)
                for key, val in change_args.items():
                    method_parameters[key] = val
            except NotImplementedError as exception:
                if (
                        ALGORITHM_SUBSTITUTION_POLICY_LEVELS[algorithm_substitution_policy]
                        > ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgSubPolicy.NONE]
                ):
                    warn('Unsupported algorithm parameter `{}` was ignored:\n  {}'.format(
                        change.kisao_id, str(exception).replace('\n', '\n  ')),
                        BioSimulatorsWarning)
                else:
                    raise
            except ValueError as exception:
                if (
                        ALGORITHM_SUBSTITUTION_POLICY_LEVELS[algorithm_substitution_policy]
                        > ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgSubPolicy.NONE]
                ):
                    warn('Unsuported value `{}` for algorithm parameter `{}` was ignored:\n  {}'.format(
                        change.new_value, change.kisao_id, str(exception).replace('\n', '\n  ')),
                        BioSimulatorsWarning)
                else:
                    raise

    # validate model changes
    model_change_obj_map = {}

    invalid_changes = []

    units = KISAO_ALGORITHMS_MAP[exec_alg_kisao_id]['default_units']
    for change in model.changes:
        target_sbml_id = model_change_target_sbml_id_map[change.target]
        copasi_model_obj = utils.get_copasi_model_object_by_sbml_id(copasi_model, target_sbml_id, units)
        if copasi_model_obj is None:
            invalid_changes.append(change.target)
        else:
            model_obj_parent = copasi_model_obj.getObjectParent()

            if isinstance(model_obj_parent, COPASI.CCompartment):
                set_func = model_obj_parent.setInitialValue
                ref = model_obj_parent.getInitialValueReference()

            elif isinstance(model_obj_parent, COPASI.CModelValue):
                set_func = model_obj_parent.setInitialValue
                ref = model_obj_parent.getInitialValueReference()

            elif isinstance(model_obj_parent, COPASI.CMetab):
                if units == Units.discrete:
                    set_func = model_obj_parent.setInitialValue
                    ref = model_obj_parent.getInitialValueReference()
                else:
                    set_func = model_obj_parent.setInitialConcentration
                    ref = model_obj_parent.getInitialConcentrationReference()

            elif isinstance(model_obj_parent, COPASI.CReaction):
                if units == Units.discrete:
                    set_func = model_obj_parent.setInitialValue
                    ref = model_obj_parent.getInitialValueReference()
                else:
                    set_func = model_obj_parent.setInitialConcentration
                    ref = model_obj_parent.getInitialConcentrationReference()

            model_change_obj_map[change.target] = (set_func, ref)

    if invalid_changes:
        raise ValueError(''.join([
            'The following change targets are invalid:\n  - {}\n\n'.format(
                '\n  - '.join(sorted(invalid_changes)),
            ),
            'Targets must have one of the following ids:\n  - {}'.format(
                '\n  - '.join(sorted(utils.get_copasi_model_obj_sbml_ids(copasi_model))),
            ),
        ]))

    # set up observables of the task
    # due to a COPASI bug, :obj:`COPASI.CCopasiTask.initializeRawWithOutputHandler` must be called after
    # :obj:`COPASI.CCopasiTask.setMethodType`
    variable_common_name_map = {}

    invalid_symbols = []
    invalid_targets = []

    for variable in variables:
        copasi_model_obj_common_name: str = None
        if variable.symbol:
            if variable.symbol == Symbol.time.value:
                copasi_model_obj_common_name = copasi_model.getValueReference().getCN().getString()
            else:
                invalid_symbols.append(variable.symbol)

        else:
            target_sbml_id: str = variable_target_sbml_id_map[variable.target]

            copasi_model_obj: COPASI.CDataObject = utils.get_copasi_model_object_by_sbml_id(copasi_model, target_sbml_id,
                                                                  KISAO_ALGORITHMS_MAP[exec_alg_kisao_id][
                                                                      'default_units'])
            if copasi_model_obj is None:
                invalid_targets.append(variable.target)
            else:
                copasi_model_obj_common_name = copasi_model_obj.getCN().getString()

        if copasi_model_obj_common_name is not None:
            variable_common_name_map[(variable.target, variable.symbol)] = COPASI.CRegisteredCommonName(
                copasi_model_obj_common_name)

    if invalid_symbols:
        raise NotImplementedError("".join([
            "The following variable symbols are not supported:\n  - {}\n\n".format(
                '\n  - '.join(sorted(invalid_symbols)),
            ),
            "Symbols must be one of the following:\n  - {}".format(Symbol.time),
        ]))

    if invalid_targets:
        raise ValueError(''.join([
            'The following variable targets cannot be recorded:\n  - {}\n\n'.format(
                '\n  - '.join(sorted(invalid_targets)),
            ),
            'Targets must have one of the following ids:\n  - {}'.format(
                '\n  - '.join(sorted(utils.get_copasi_model_obj_sbml_ids(copasi_model))),
            ),
        ]))

    # Execute simulation
    copasi_task.setScheduled(True)

    copasi_problem = copasi_task.getProblem()
    copasi_problem.setTimeSeriesRequested(False)
    copasi_problem.setAutomaticStepSize(False)
    copasi_problem.setOutputEvent(False)

    # return preprocessed info
    preprocessed_info = {
        'task': copasi_task,
        'model': {
            'model': copasi_model,
            'model_change_obj_map': model_change_obj_map,
            'variable_common_name_map': variable_common_name_map,
        },
        'simulation': {
            'algorithm_kisao_id': exec_alg_kisao_id,
            'algorithm_copasi_id': alg_copasi_id,
            'method_name': KISAO_ALGORITHMS_MAP[exec_alg_kisao_id]['id'],
            'method_parameters': method_parameters,
        },
    }

    return preprocessed_info


def _fix_time_name(var: Variable, new_name: str) -> Variable:
    list_args = list(var.to_tuple())
    list_args[1] = new_name
    args = tuple(list_args)
    return Variable(*args)

def _get_copasi_fixed_archive(archive_filename: bytes | str):
    temp_archive_file: int
    temp_archive_filename: bytes

    temp_archive_file, temp_archive_filename = tempfile.mkstemp()
    os.close(temp_archive_file)
    utils.fix_copasi_generated_combine_archive(archive_filename, temp_archive_filename)
    return temp_archive_filename


def _validate_sedml(task: Task, model: Model, sim: Simulation, variables: list[Variable]):
    # Prepare error messages
    invalid_task: str = f'Task `{task.id}` is invalid.'
    invalid_model_lang: str = f'Language for model `{model.id}` is not supported.'
    invalid_model_changes: str = f'Changes for model `{model.id}` are not supported.'
    invalid_sim_type: str = f'{sim.__class__.__name__,} `{sim.id}` is not supported.'
    invalid_sim: str = 'Simulation `{sim.id}` is invalid.'
    invalid_data_gen: str = f'Data generator variables for task `{task.id}` are invalid.'

    # run validations
    task_errors = validation.validate_task(task)
    model_lang_errors = validation.validate_model_language(model.language, ModelLanguage.SBML)
    model_change_type_errors = validation.validate_model_change_types(model.changes, (ModelAttributeChange,))
    simulation_type_errors = validation.validate_simulation_type(sim, (UniformTimeCourseSimulation,))
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


def _apply_model_changes(model: Model, basico_data_model: COPASI.CDataModel, model_change_target_sbml_id_map: dict,
                         exec_alg_kisao_id: str) -> tuple[list[ModelAttributeChange], list[ModelChange]]:

    legal_changes: list[ModelAttributeChange] = []
    illegal_changes: list[ModelChange] = []

    # If there's no changes, get out of here
    if not model.changes:
        return legal_changes, illegal_changes

    # check if there's anything but ChangeAttribute type changes
    change: ModelChange
    for change in model.changes:
        legal_changes.append(change) if isinstance(change, ModelAttributeChange) else illegal_changes.append(change)


    units = KISAO_ALGORITHMS_MAP[exec_alg_kisao_id]['default_units']
    c_model = basico_data_model.getModel()
    model_change: ModelAttributeChange
    for model_change in legal_changes:
        target_sbml_id = model_change_target_sbml_id_map[model_change.target]
        copasi_model_obj = utils.get_copasi_model_object_by_sbml_id(c_model, target_sbml_id, units)
        if copasi_model_obj is None:
            illegal_changes.append(model_change.target)

    return legal_changes, illegal_changes
