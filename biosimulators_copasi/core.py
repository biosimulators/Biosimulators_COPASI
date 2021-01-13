""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-11-17
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from biosimulators_utils.combine.exec import exec_sedml_docs_in_archive
from biosimulators_utils.log.data_model import CombineArchiveLog, TaskLog  # noqa: F401
from biosimulators_utils.plot.data_model import PlotFormat  # noqa: F401
from biosimulators_utils.report.data_model import ReportFormat, DataGeneratorVariableResults  # noqa: F401
from biosimulators_utils.sedml.data_model import (Task, ModelLanguage, UniformTimeCourseSimulation,  # noqa: F401
                                                  DataGeneratorVariable, DataGeneratorVariableSymbol)
from biosimulators_utils.sedml import validation
from biosimulators_utils.sedml.exec import exec_sed_doc
from .data_model import KISAO_ALGORITHMS_MAP
from .utils import get_algorithm_id, set_algorithm_parameter_value
import COPASI
import functools
import math
import numpy
import warnings


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
        plot_formats (:obj:`list` of :obj:`PlotFormat`, optional): report format (e.g., pdf)
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
       variables (:obj:`list` of :obj:`DataGeneratorVariable`): variables that should be recorded
       log (:obj:`TaskLog`, optional): log for the task

    Returns:
        :obj:`tuple`:

            :obj:`DataGeneratorVariableResults`: results of variables
            :obj:`TaskLog`: log

    Raises:
        :obj:`ValueError`: if the task or an aspect of the task is not valid, or the requested output variables
            could not be recorded
        :obj:`NotImplementedError`: if the task is not of a supported type or involves an unsuported feature
    '''
    log = log or TaskLog()

    validation.validate_task(task)

    model = task.model
    validation.validate_model_language(model.language, ModelLanguage.SBML)
    validation.validate_model_change_types(model.changes, ())

    sim = task.simulation
    validation.validate_simulation_type(sim, (UniformTimeCourseSimulation, ))
    validation.validate_uniform_time_course_simulation(sim)
    validation.validate_data_generator_variables(variables)
    target_x_paths_ids = validation.validate_data_generator_variable_xpaths(variables, model.source, attr='id')

    # Read the SBML-encoded model located at `os.path.join(working_dir, model_filename)`
    copasi_data_model = COPASI.CRootContainer.addDatamodel()
    if not copasi_data_model.importSBML(task.model.source):
        raise ValueError("'{}' could not be imported:\n\n  {}".format(
            task.model.source, get_copasi_error_message(sim.algorithm.kisao_id).replace('\n', "\n  ")))

    # Load the algorithm specified by `simulation.algorithm`
    alg_kisao_id = sim.algorithm.kisao_id
    alg_copasi_id = get_algorithm_id(alg_kisao_id)
    copasi_task = copasi_data_model.getTask('Time-Course')
    if not copasi_task.setMethodType(alg_copasi_id):
        raise RuntimeError('Unable to initialize function for {}'.format(alg_kisao_id)
                           )  # pragma: no cover # unreachable because :obj:`get_algorithm_id` returns valid COPASI method ids
    method = copasi_task.getMethod()

    # Apply the algorithm parameter changes specified by `simulation.algorithm_parameter_changes`
    method_parameters = {}
    for change in sim.algorithm.changes:
        change_args = set_algorithm_parameter_value(alg_kisao_id, method,
                                                    change.kisao_id, change.new_value)
        for key, val in change_args.items():
            method_parameters[key] = val

    # Execute simulation
    copasi_task.setScheduled(True)

    model = copasi_data_model.getModel()

    problem = copasi_task.getProblem()
    model.setInitialTime(sim.initial_time)
    problem.setOutputStartTime(sim.output_start_time - sim.initial_time)
    problem.setDuration(sim.output_end_time - sim.initial_time)
    step_number = sim.number_of_points * (sim.output_end_time - sim.initial_time) / (sim.output_end_time - sim.output_start_time)
    if step_number != math.floor(step_number):
        raise NotImplementedError('Time course must specify an integer number of time points')
    else:
        step_number = int(step_number)
    problem.setStepNumber(step_number)
    problem.setTimeSeriesRequested(True)
    problem.setAutomaticStepSize(False)
    problem.setOutputEvent(False)

    result = copasi_task.process(True)
    warning_details = copasi_task.getProcessWarning()
    if warning_details:
        warnings.warn(get_copasi_error_message(alg_kisao_id, warning_details), UserWarning)
    if not result:
        error_details = copasi_task.getProcessError()
        raise RuntimeError(get_copasi_error_message(alg_kisao_id, error_details))

    time_series = copasi_task.getTimeSeries()
    number_of_recorded_points = time_series.getRecordedSteps()
    if number_of_recorded_points != (sim.number_of_points + 1):
        raise RuntimeError('Simulation produced {} rather than {} time points'.format(
            number_of_recorded_points, sim.number_of_points)
        )  # pragma: no cover # unreachable because COPASI produces the correct number of outputs

    # collect simulation predictions
    sbml_id_to_i_time_series = {}
    for i_time_series in range(0, time_series.getNumVariables()):
        time_series_sbml_id = time_series.getSBMLId(i_time_series, copasi_data_model)
        sbml_id_to_i_time_series[time_series_sbml_id] = i_time_series

    get_data_function = getattr(time_series, KISAO_ALGORITHMS_MAP[alg_kisao_id]['get_data_function'].value)
    variable_results = DataGeneratorVariableResults()
    unpredicted_symbols = []
    unpredicted_targets = []
    for variable in variables:
        if variable.symbol:
            if variable.symbol == DataGeneratorVariableSymbol.time:
                variable_result = numpy.linspace(sim.output_start_time, sim.output_end_time, number_of_recorded_points)
            else:
                unpredicted_symbols.append(variable.symbol)
                variable_result = numpy.full((number_of_recorded_points,), numpy.nan)

        else:
            target_sbml_id = target_x_paths_ids[variable.target]
            i_time_series = sbml_id_to_i_time_series.get(target_sbml_id, None)
            variable_result = numpy.full((number_of_recorded_points,), numpy.nan)
            if i_time_series is None:
                unpredicted_targets.append(variable.target)
            else:
                for i_step in range(0, time_series.getRecordedSteps()):
                    variable_result[i_step] = get_data_function(i_step, i_time_series)

        variable_results[variable.id] = variable_result

    if unpredicted_symbols:
        raise NotImplementedError("".join([
            "The following variable symbols are not supported:\n  - {}\n\n".format(
                '\n  - '.join(sorted(unpredicted_symbols)),
            ),
            "Symbols must be one of the following:\n  - {}".format(DataGeneratorVariableSymbol.time),
        ]))

    if unpredicted_targets:
        raise ValueError(''.join([
            'The following variable targets could not be recorded:\n  - {}\n\n'.format(
                '\n  - '.join(sorted(unpredicted_targets)),
            ),
            'Targets must have one of the following ids:\n  - {}'.format(
                '\n  - '.join(sorted(sbml_id_to_i_time_series.keys())),
            ),
        ]))

    # log action
    log.algorithm = alg_kisao_id
    log.simulator_details = {
        'methodName': KISAO_ALGORITHMS_MAP[alg_kisao_id]['id'],
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
    if details is None:
        details = COPASI.CCopasiMessage.getLastMessage().getText()
    if details:
        details = '\n'.join(line[min(2, len(line)):] for line in details.split('\n')
                            if not (line.startswith('>') and line.endswith('<')))
        error_msg += ':\n\n  ' + details.replace('\n', '\n  ')
    return error_msg
