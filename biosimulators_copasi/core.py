""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-11-17
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from biosimulators_utils.combine.exec import exec_sedml_docs_in_archive
from biosimulators_utils.log.data_model import CombineArchiveLog, TaskLog  # noqa: F401
from biosimulators_utils.viz.data_model import VizFormat  # noqa: F401
from biosimulators_utils.report.data_model import ReportFormat, VariableResults  # noqa: F401
from biosimulators_utils.sedml.data_model import (Task, ModelLanguage, UniformTimeCourseSimulation,  # noqa: F401
                                                  Variable, Symbol)
from biosimulators_utils.sedml import validation
from biosimulators_utils.sedml.exec import exec_sed_doc
from biosimulators_utils.simulator.utils import get_algorithm_substitution_policy
from biosimulators_utils.utils.core import raise_errors_warnings
from biosimulators_utils.warnings import warn, BioSimulatorsWarning
from kisao.data_model import AlgorithmSubstitutionPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from .data_model import KISAO_ALGORITHMS_MAP
from .utils import (get_algorithm_id, set_algorithm_parameter_value,
                    get_copasi_model_object_by_sbml_id, get_copasi_model_obj_sbml_ids)
import COPASI
import functools
import math
import numpy


__all__ = ['exec_sedml_docs_in_combine_archive', 'exec_sed_task']


def exec_sedml_docs_in_combine_archive(archive_filename, out_dir,
                                       report_formats=None, plot_formats=None,
                                       bundle_outputs=None, keep_individual_outputs=None):
    """ Execute the SED tasks defined in a COMBINE/OMEX archive and save the outputs

    Args:
        archive_filename (:obj:`str`): path to COMBINE/OMEX archive
        out_dir (:obj:`str`): path to store the outputs of the archive

            * CSV: directory in which to save outputs to files
              ``{ out_dir }/{ relative-path-to-SED-ML-file-within-archive }/{ report.id }.csv``
            * HDF5: directory in which to save a single HDF5 file (``{ out_dir }/reports.h5``),
              with reports at keys ``{ relative-path-to-SED-ML-file-within-archive }/{ report.id }`` within the HDF5 file

        report_formats (:obj:`list` of :obj:`ReportFormat`, optional): report format (e.g., csv or h5)
        plot_formats (:obj:`list` of :obj:`VizFormat`, optional): report format (e.g., pdf)
        bundle_outputs (:obj:`bool`, optional): if :obj:`True`, bundle outputs into archives for reports and plots
        keep_individual_outputs (:obj:`bool`, optional): if :obj:`True`, keep individual output files

    Returns:
        :obj:`CombineArchiveLog`: log
    """
    sed_doc_executer = functools.partial(exec_sed_doc, exec_sed_task)
    return exec_sedml_docs_in_archive(sed_doc_executer, archive_filename, out_dir,
                                      apply_xml_model_changes=True,
                                      report_formats=report_formats,
                                      plot_formats=plot_formats,
                                      bundle_outputs=bundle_outputs,
                                      keep_individual_outputs=keep_individual_outputs)


def exec_sed_task(task, variables, log=None):
    ''' Execute a task and save its results

    Args:
       task (:obj:`Task`): task
       variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
       log (:obj:`TaskLog`, optional): log for the task

    Returns:
        :obj:`tuple`:

            :obj:`VariableResults`: results of variables
            :obj:`TaskLog`: log

    Raises:
        :obj:`ValueError`: if the task or an aspect of the task is not valid, or the requested output variables
            could not be recorded
        :obj:`NotImplementedError`: if the task is not of a supported type or involves an unsuported feature
    '''
    log = log or TaskLog()

    model = task.model
    sim = task.simulation

    raise_errors_warnings(validation.validate_task(task),
                          error_summary='Task `{}` is invalid.'.format(task.id))
    raise_errors_warnings(validation.validate_model_language(model.language, ModelLanguage.SBML),
                          error_summary='Language for model `{}` is not supported.'.format(model.id))
    raise_errors_warnings(validation.validate_model_change_types(model.changes, ()),
                          error_summary='Changes for model `{}` are not supported.'.format(model.id))
    raise_errors_warnings(*validation.validate_model_changes(task.model),
                          error_summary='Changes for model `{}` are invalid.'.format(model.id))
    raise_errors_warnings(validation.validate_simulation_type(sim, (UniformTimeCourseSimulation, )),
                          error_summary='{} `{}` is not supported.'.format(sim.__class__.__name__, sim.id))
    raise_errors_warnings(*validation.validate_simulation(sim),
                          error_summary='Simulation `{}` is invalid.'.format(sim.id))
    raise_errors_warnings(*validation.validate_data_generator_variables(variables),
                          error_summary='Data generator variables for task `{}` are invalid.'.format(task.id))
    target_x_paths_ids = validation.validate_variable_xpaths(variables, model.source, attr='id')

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
    exec_alg_kisao_id, alg_copasi_id = get_algorithm_id(alg_kisao_id, events=copasi_model.getNumEvents() >= 1)

    # initialize COPASI task
    copasi_task = copasi_data_model.getTask('Time-Course')

    # Load the algorithm specified by `simulation.algorithm`
    if not copasi_task.setMethodType(alg_copasi_id):
        raise RuntimeError('Unable to initialize function for {}'.format(exec_alg_kisao_id)
                           )  # pragma: no cover # unreachable because :obj:`get_algorithm_id` returns valid COPASI method ids
    copasi_method = copasi_task.getMethod()

    # Apply the algorithm parameter changes specified by `simulation.algorithm_parameter_changes`
    method_parameters = {}
    algorithm_substitution_policy = get_algorithm_substitution_policy()
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

    # set up observables of the task
    # due to a COPASI bug, :obj:`COPASI.CCopasiTask.initializeRawWithOutputHandler` must be called after
    # :obj:`COPASI.CCopasiTask.setMethodType`

    copasi_data_handler = COPASI.CDataHandler()

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
            target_sbml_id = target_x_paths_ids[variable.target]

            copasi_model_obj = get_copasi_model_object_by_sbml_id(copasi_model, target_sbml_id,
                                                                  KISAO_ALGORITHMS_MAP[exec_alg_kisao_id]['default_units'])
            if copasi_model_obj is None:
                invalid_targets.append(variable.target)
            else:
                copasi_model_obj_common_name = copasi_model_obj.getCN().getString()

        if copasi_model_obj_common_name is not None:
            copasi_data_handler.addDuringName(COPASI.CRegisteredCommonName(copasi_model_obj_common_name))

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

    if not copasi_task.initializeRawWithOutputHandler(COPASI.CCopasiTask.OUTPUT_DURING, copasi_data_handler):
        raise RuntimeError("Output handler could not be initialized:\n\n  {}".format(
            get_copasi_error_message(sim.algorithm.kisao_id).replace('\n', "\n  ")))

    # Execute simulation
    copasi_task.setScheduled(True)

    copasi_problem = copasi_task.getProblem()
    copasi_model.setInitialTime(sim.initial_time)
    copasi_problem.setOutputStartTime(sim.output_start_time - sim.initial_time)
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
    copasi_problem.setTimeSeriesRequested(False)
    copasi_problem.setAutomaticStepSize(False)
    copasi_problem.setOutputEvent(False)

    result = copasi_task.processRaw(True)
    warning_details = copasi_task.getProcessWarning()
    if warning_details:
        warn(get_copasi_error_message(exec_alg_kisao_id, warning_details), BioSimulatorsWarning)
    if not result:
        error_details = copasi_task.getProcessError()
        raise RuntimeError(get_copasi_error_message(exec_alg_kisao_id, error_details))

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
            if variable.symbol == Symbol.time.value:
                value += sim.initial_time

            variable_results[variable.id][i_step] = value

    if sim.output_end_time == sim.output_start_time and sim.output_start_time == sim.initial_time:
        for variable in variables:
            variable_results[variable.id] = numpy.concatenate((
                variable_results[variable.id][0:1],
                numpy.full((sim.number_of_points,), variable_results[variable.id][1]),
            ))

    # log action
    log.algorithm = exec_alg_kisao_id
    log.simulator_details = {
        'methodName': KISAO_ALGORITHMS_MAP[exec_alg_kisao_id]['id'],
        'methodCode': alg_copasi_id,
        'parameters': method_parameters,
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
