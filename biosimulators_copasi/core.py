""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-11-17
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from biosimulators_utils.combine.exec import exec_sedml_docs_in_archive
from biosimulators_utils.config import get_config, Config  # noqa: F401
from biosimulators_utils.log.data_model import CombineArchiveLog, TaskLog, StandardOutputErrorCapturerLevel  # noqa: F401
from biosimulators_utils.viz.data_model import VizFormat  # noqa: F401
from biosimulators_utils.report.data_model import ReportFormat, VariableResults, SedDocumentResults  # noqa: F401
from biosimulators_utils.sedml.data_model import (Task, ModelLanguage, ModelAttributeChange, UniformTimeCourseSimulation,  # noqa: F401
                                                  Variable, Symbol)
from biosimulators_utils.sedml import validation
from biosimulators_utils.sedml.exec import exec_sed_doc as base_exec_sed_doc
from biosimulators_utils.simulator.utils import get_algorithm_substitution_policy
from biosimulators_utils.utils.core import raise_errors_warnings
from biosimulators_utils.warnings import warn, BioSimulatorsWarning
from kisao.data_model import AlgorithmSubstitutionPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from .data_model import KISAO_ALGORITHMS_MAP, Units
from .utils import (get_algorithm_id, set_algorithm_parameter_value,
                    get_copasi_model_object_by_sbml_id, get_copasi_model_obj_sbml_ids,
                    fix_copasi_generated_combine_archive as fix_copasi_generated_combine_archive_func)
import COPASI
import lxml
import math
import numpy
import os
import tempfile

__all__ = ['exec_sedml_docs_in_combine_archive', 'exec_sed_doc', 'exec_sed_task', 'preprocess_sed_task']


def exec_sedml_docs_in_combine_archive(archive_filename, out_dir, config=None, fix_copasi_generated_combine_archive=None):
    """ Execute the SED tasks defined in a COMBINE/OMEX archive and save the outputs

    Args:
        archive_filename (:obj:`str`): path to COMBINE/OMEX archive
        out_dir (:obj:`str`): path to store the outputs of the archive

            * CSV: directory in which to save outputs to files
              ``{ out_dir }/{ relative-path-to-SED-ML-file-within-archive }/{ report.id }.csv``
            * HDF5: directory in which to save a single HDF5 file (``{ out_dir }/reports.h5``),
              with reports at keys ``{ relative-path-to-SED-ML-file-within-archive }/{ report.id }`` within the HDF5 file

        config (:obj:`Config`, optional): BioSimulators common configuration
        fix_copasi_generated_combine_archive (:obj:`bool`, optional): Whether to make COPASI-generated COMBINE archives
            compatible with the specifications of the OMEX manifest and SED-ML standards

    Returns:
        :obj:`tuple`:

            * :obj:`SedDocumentResults`: results
            * :obj:`CombineArchiveLog`: log
    """
    if fix_copasi_generated_combine_archive is None:
        fix_copasi_generated_combine_archive = os.getenv('FIX_COPASI_GENERATED_COMBINE_ARCHIVE', '0').lower() in ['1', 'true']

    if fix_copasi_generated_combine_archive:
        temp_archive_file, temp_archive_filename = tempfile.mkstemp()
        os.close(temp_archive_file)
        fix_copasi_generated_combine_archive_func(archive_filename, temp_archive_filename)
        archive_filename = temp_archive_filename

    result = exec_sedml_docs_in_archive(exec_sed_doc, archive_filename, out_dir,
                                        apply_xml_model_changes=True,
                                        config=config)
    if fix_copasi_generated_combine_archive:
        os.remove(temp_archive_filename)

    return result


def exec_sed_doc(doc, working_dir, base_out_path, rel_out_path=None,
                 apply_xml_model_changes=True,
                 log=None, indent=0, pretty_print_modified_xml_models=False,
                 log_level=StandardOutputErrorCapturerLevel.c, config=None):
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
    return base_exec_sed_doc(exec_sed_task, doc, working_dir, base_out_path,
                             rel_out_path=rel_out_path,
                             apply_xml_model_changes=apply_xml_model_changes,
                             log=log,
                             indent=indent,
                             pretty_print_modified_xml_models=pretty_print_modified_xml_models,
                             log_level=log_level,
                             config=config)


def exec_sed_task(task, variables, preprocessed_task=None, log=None, config=None):
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
        :obj:`NotImplementedError`: if the task is not of a supported type or involves an unsuported feature
    '''
    config = config or get_config()

    if config.LOG and not log:
        log = TaskLog()

    if preprocessed_task is None:
        preprocessed_task = preprocess_sed_task(task, variables, config=config)

    model = task.model
    sim = task.simulation

    # initialize COPASI task
    copasi_model = preprocessed_task['model']['model']

    # modify model
    if model.changes:
        raise_errors_warnings(validation.validate_model_change_types(model.changes, (ModelAttributeChange,)),
                              error_summary='Changes for model `{}` are not supported.'.format(model.id))
        model_change_obj_map = preprocessed_task['model']['model_change_obj_map']
        changed_objects = COPASI.ObjectStdVector()
        for change in model.changes:
            model_obj_set_func, ref = model_change_obj_map[change.target]
            new_value = float(change.new_value)
            model_obj_set_func(new_value)
            changed_objects.push_back(ref)

        copasi_model.compileIfNecessary()
        copasi_model.updateInitialValues(changed_objects)

    # initialize simulation
    copasi_model.setInitialTime(sim.initial_time)
    copasi_model.forceCompile()

    copasi_task = preprocessed_task['task']
    copasi_problem = copasi_task.getProblem()
    copasi_problem.setOutputStartTime(sim.output_start_time)
    copasi_problem.setDuration(sim.output_end_time - sim.initial_time)
    if sim.output_end_time == sim.output_start_time:
        if sim.output_start_time == sim.initial_time:
            step_number = sim.number_of_points
        else:
            raise NotImplementedError('Output end time must be greater than the output start time.')
    else:
        step_number = (
            sim.number_of_points
            * (sim.output_end_time - sim.initial_time)
            / (sim.output_end_time - sim.output_start_time)
        )
    if step_number != math.floor(step_number):
        raise NotImplementedError('Time course must specify an integer number of time points')
    else:
        step_number = int(step_number)
    copasi_problem.setStepNumber(step_number)

    # setup data handler
    copasi_data_handler = COPASI.CDataHandler()
    variable_common_name_map = preprocessed_task['model']['variable_common_name_map']
    for variable in variables:
        common_name = variable_common_name_map[(variable.target, variable.symbol)]
        copasi_data_handler.addDuringName(common_name)
    if not copasi_task.initializeRawWithOutputHandler(COPASI.CCopasiTask.OUTPUT_DURING, copasi_data_handler):
        raise RuntimeError("Output handler could not be initialized:\n\n  {}".format(
            get_copasi_error_message(sim.algorithm.kisao_id).replace('\n', "\n  ")))

    # Execute simulation
    result = copasi_task.processRaw(True)
    warning_details = copasi_task.getProcessWarning()
    if warning_details:
        alg_kisao_id = preprocessed_task['simulation']['algorithm_kisao_id']
        warn(get_copasi_error_message(alg_kisao_id, warning_details), BioSimulatorsWarning)
    if not result:
        alg_kisao_id = preprocessed_task['simulation']['algorithm_kisao_id']
        error_details = copasi_task.getProcessError()
        raise RuntimeError(get_copasi_error_message(alg_kisao_id, error_details))

    # collect simulation predictions
    number_of_recorded_points = copasi_data_handler.getNumRowsDuring()

    if (
        variables
        and number_of_recorded_points != (sim.number_of_points + 1)
        and (sim.output_end_time != sim.output_start_time or sim.output_start_time != sim.initial_time)
    ):
        raise RuntimeError('Simulation produced {} rather than {} time points'.format(
            number_of_recorded_points, sim.number_of_points)
        )  # pragma: no cover # unreachable because COPASI produces the correct number of outputs

    variable_results = VariableResults()
    for variable in variables:
        variable_results[variable.id] = numpy.full((number_of_recorded_points,), numpy.nan)

    for i_step in range(number_of_recorded_points):
        step_values = copasi_data_handler.getNthRow(i_step)
        for variable, value in zip(variables, step_values):
            variable_results[variable.id][i_step] = value

    if sim.output_end_time == sim.output_start_time and sim.output_start_time == sim.initial_time:
        for variable in variables:
            variable_results[variable.id] = numpy.concatenate((
                variable_results[variable.id][0:1],
                numpy.full((sim.number_of_points,), variable_results[variable.id][1]),
            ))

    copasi_data_handler.cleanup()
    copasi_data_handler.close()

    # log action
    if config.LOG:
        log.algorithm = preprocessed_task['simulation']['algorithm_kisao_id']
        log.simulator_details = {
            'methodName': preprocessed_task['simulation']['method_name'],
            'methodCode': preprocessed_task['simulation']['algorithm_copasi_id'],
            'parameters': preprocessed_task['simulation']['method_parameters'],
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


def preprocess_sed_task(task, variables, config=None):
    """ Preprocess a SED task, including its possible model changes and variables. This is useful for avoiding
    repeatedly initializing tasks on repeated calls of :obj:`exec_sed_task`.

    Args:
        task (:obj:`Task`): task
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`dict`: preprocessed information about the task
    """
    config = config or get_config()

    model = task.model
    sim = task.simulation

    if config.VALIDATE_SEDML:
        raise_errors_warnings(validation.validate_task(task),
                              error_summary='Task `{}` is invalid.'.format(task.id))
        raise_errors_warnings(validation.validate_model_language(model.language, ModelLanguage.SBML),
                              error_summary='Language for model `{}` is not supported.'.format(model.id))
        raise_errors_warnings(validation.validate_model_change_types(model.changes, (ModelAttributeChange,)),
                              error_summary='Changes for model `{}` are not supported.'.format(model.id))
        raise_errors_warnings(*validation.validate_model_changes(model),
                              error_summary='Changes for model `{}` are invalid.'.format(model.id))
        raise_errors_warnings(validation.validate_simulation_type(sim, (UniformTimeCourseSimulation, )),
                              error_summary='{} `{}` is not supported.'.format(sim.__class__.__name__, sim.id))
        raise_errors_warnings(*validation.validate_simulation(sim),
                              error_summary='Simulation `{}` is invalid.'.format(sim.id))
        raise_errors_warnings(*validation.validate_data_generator_variables(variables),
                              error_summary='Data generator variables for task `{}` are invalid.'.format(task.id))

    model_etree = lxml.etree.parse(model.source)
    model_change_target_sbml_id_map = validation.validate_target_xpaths(model.changes, model_etree, attr='id')
    variable_target_sbml_id_map = validation.validate_target_xpaths(variables, model_etree, attr='id')

    if config.VALIDATE_SEDML_MODELS:
        raise_errors_warnings(*validation.validate_model(model, [], working_dir='.'),
                              error_summary='Model `{}` is invalid.'.format(model.id),
                              warning_summary='Model `{}` may be invalid.'.format(model.id))

    # Read the SBML-encoded model located at `os.path.join(working_dir, model_filename)`
    copasi_data_model = COPASI.CRootContainer.addDatamodel()
    if not copasi_data_model.importSBML(model.source):
        raise ValueError("`{}` could not be imported:\n\n  {}".format(
            model.source, get_copasi_error_message(sim.algorithm.kisao_id).replace('\n', "\n  ")))
    copasi_model = copasi_data_model.getModel()

    # determine the algorithm to execute
    alg_kisao_id = sim.algorithm.kisao_id
    exec_alg_kisao_id, alg_copasi_id = get_algorithm_id(alg_kisao_id, events=copasi_model.getNumEvents() >= 1, config=config)

    # initialize COPASI task
    copasi_task = copasi_data_model.getTask('Time-Course')

    # Load the algorithm specified by `simulation.algorithm`
    if not copasi_task.setMethodType(alg_copasi_id):
        raise RuntimeError('Unable to initialize function for {}'.format(exec_alg_kisao_id)
                           )  # pragma: no cover # unreachable because :obj:`get_algorithm_id` returns valid COPASI method ids
    copasi_method = copasi_task.getMethod()

    # Apply the algorithm parameter changes specified by `simulation.algorithm_parameter_changes`
    method_parameters = {}
    algorithm_substitution_policy = get_algorithm_substitution_policy(config=config)
    if exec_alg_kisao_id == alg_kisao_id:
        for change in sim.algorithm.changes:
            try:
                change_args = set_algorithm_parameter_value(exec_alg_kisao_id, copasi_method,
                                                            change.kisao_id, change.new_value)
                for key, val in change_args.items():
                    method_parameters[key] = val
            except NotImplementedError as exception:
                if (
                    ALGORITHM_SUBSTITUTION_POLICY_LEVELS[algorithm_substitution_policy]
                    > ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgorithmSubstitutionPolicy.NONE]
                ):
                    warn('Unsuported algorithm parameter `{}` was ignored:\n  {}'.format(
                        change.kisao_id, str(exception).replace('\n', '\n  ')),
                        BioSimulatorsWarning)
                else:
                    raise
            except ValueError as exception:
                if (
                    ALGORITHM_SUBSTITUTION_POLICY_LEVELS[algorithm_substitution_policy]
                    > ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgorithmSubstitutionPolicy.NONE]
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
        copasi_model_obj = get_copasi_model_object_by_sbml_id(copasi_model, target_sbml_id, units)
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

            model_change_obj_map[change.target] = (set_func, ref)

    if invalid_changes:
        raise ValueError(''.join([
            'The following change targets are invalid:\n  - {}\n\n'.format(
                '\n  - '.join(sorted(invalid_changes)),
            ),
            'Targets must have one of the following ids:\n  - {}'.format(
                '\n  - '.join(sorted(get_copasi_model_obj_sbml_ids(copasi_model))),
            ),
        ]))

    # set up observables of the task
    # due to a COPASI bug, :obj:`COPASI.CCopasiTask.initializeRawWithOutputHandler` must be called after
    # :obj:`COPASI.CCopasiTask.setMethodType`
    variable_common_name_map = {}

    invalid_symbols = []
    invalid_targets = []

    for variable in variables:
        copasi_model_obj_common_name = None
        if variable.symbol:
            if variable.symbol == Symbol.time.value:
                copasi_model_obj_common_name = copasi_model.getValueReference().getCN()
            else:
                invalid_symbols.append(variable.symbol)

        else:
            target_sbml_id = variable_target_sbml_id_map[variable.target]

            copasi_model_obj = get_copasi_model_object_by_sbml_id(copasi_model, target_sbml_id,
                                                                  KISAO_ALGORITHMS_MAP[exec_alg_kisao_id]['default_units'])
            if copasi_model_obj is None:
                invalid_targets.append(variable.target)
            else:
                copasi_model_obj_common_name = copasi_model_obj.getCN().getString()

        if copasi_model_obj_common_name is not None:
            variable_common_name_map[(variable.target, variable.symbol)] = COPASI.CRegisteredCommonName(copasi_model_obj_common_name)

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
                '\n  - '.join(sorted(get_copasi_model_obj_sbml_ids(copasi_model))),
            ),
        ]))

    # Execute simulation
    copasi_task.setScheduled(True)

    copasi_problem = copasi_task.getProblem()
    copasi_problem.setTimeSeriesRequested(False)
    copasi_problem.setAutomaticStepSize(False)
    copasi_problem.setOutputEvent(False)

    # return preprocessed info
    return {
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
