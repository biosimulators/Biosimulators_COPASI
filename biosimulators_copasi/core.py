""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-11-17
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from Biosimulations_utils.simulation.data_model import TimecourseSimulation, SimulationResultsFormat
from Biosimulations_utils.simulator.utils import exec_simulations_in_archive
import COPASI
import json
import numpy
import os
import pandas
import re
import types


__all__ = ['exec_combine_archive', 'exec_simulation']

KISAO_ALGORITHMS_MAP = {
    'KISAO_0000027': {
        'name': 'Gibson + Bruck',
        'id': COPASI.CTaskEnum.Method_stochastic,
        'get_data_function': 'getData',
    },
    'KISAO_0000029': {
        'name': 'direct method',
        'id': COPASI.CTaskEnum.Method_directMethod,
        'get_data_function': 'getData',
    },
    'KISAO_0000039': {
        'name': 'tau leap method',
        'id': COPASI.CTaskEnum.Method_tauLeap,
        'get_data_function': 'getData',
    },
    'KISAO_0000048': {
        'name': 'adaptive SSA + tau leap',
        'id': COPASI.CTaskEnum.Method_adaptiveSA,
        'get_data_function': 'getData',
    },
    'KISAO_0000560': {
        'name': 'LSODA/LSODAR',
        'id': COPASI.CTaskEnum.Method_deterministic,
        'get_data_function': 'getConcentrationData',
    },
    'KISAO_0000088': {
        'name': 'LSODA',
        'id': COPASI.CTaskEnum.Method_deterministic,
        'get_data_function': 'getConcentrationData',
    },
    'KISAO_0000089': {
        'name': 'LSODAR',
        'id': COPASI.CTaskEnum.Method_deterministic,
        'get_data_function': 'getConcentrationData',
    },
    'KISAO_0000304': {
        'name': 'RADAU5',
        'id': COPASI.CTaskEnum.Method_RADAU5,
        'get_data_function': 'getConcentrationData',
    },
    'KISAO_0000561': {
        'name': 'hybrid (runge kutta)',
        'id': COPASI.CTaskEnum.Method_hybrid,
        'get_data_function': 'getData',
    },
    'KISAO_0000562': {
        'name': 'hybrid (lsoda)',
        'id': COPASI.CTaskEnum.Method_hybridLSODA,
        'get_data_function': 'getData',
    },
    'KISAO_0000563': {
        'name': 'hybrid (RK-45)',
        'id': COPASI.CTaskEnum.Method_hybridODE45,
        'get_data_function': 'getData',
    },
    'KISAO_0000566': {
        'name': 'SDE Solve (RI5)',
        'id': COPASI.CTaskEnum.Method_stochasticRunkeKuttaRI5,
        'get_data_function': 'getConcentrationData',
    },
}

KISAO_PARAMETERS_MAP = {
    'KISAO_0000209': {
        'name': 'Relative Tolerance',
        'type': 'float',
        'algorithms': ['KISAO_0000560', 'KISAO_0000088', 'KISAO_0000089', 'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000304'],
    },
    'KISAO_0000211': {
        'name': 'Absolute Tolerance',
        'type': 'float',
        'algorithms': ['KISAO_0000560', 'KISAO_0000088', 'KISAO_0000089', 'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000304',
                       'KISAO_0000566'],
    },
    'KISAO_0000216': {
        'name': 'Integrate Reduced Model',
        'type': 'bool',
        'algorithms': ['KISAO_0000560', 'KISAO_0000088', 'KISAO_0000089', 'KISAO_0000562', 'KISAO_0000304'],
    },
    'KISAO_0000415': {
        'name': 'Max Internal Steps',
        'type': 'int',
        'algorithms': ['KISAO_0000048', 'KISAO_0000560', 'KISAO_0000088', 'KISAO_0000089', 'KISAO_0000027', 'KISAO_0000029',
                       'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000561', 'KISAO_0000304', 'KISAO_0000566', 'KISAO_0000039'],
    },
    'KISAO_0000467': {
        'name': 'Max Internal Step Size',
        'type': 'float',
        'algorithms': ['KISAO_0000560', 'KISAO_0000088', 'KISAO_0000089', 'KISAO_0000562'],
    },
    'KISAO_0000488': {
        'name': 'Random Seed',
        'type': 'int',
        'algorithms': ['KISAO_0000048', 'KISAO_0000027', 'KISAO_0000029', 'KISAO_0000562', 'KISAO_0000563', 'KISAO_0000561',
                       'KISAO_0000039'],
    },
    'KISAO_0000228': {
        'name': 'Epsilon',
        'type': 'float',
        'algorithms': ['KISAO_0000048', 'KISAO_0000039'],
    },
    'KISAO_0000203': {
        'name': 'Lower Limit',
        'type': 'int',
        'algorithms': ['KISAO_0000562', 'KISAO_0000561'],
    },
    'KISAO_0000204': {
        'name': 'Upper Limit',
        'type': 'int',
        'algorithms': ['KISAO_0000562', 'KISAO_0000561'],
    },
    'KISAO_0000205': {
        'name': 'Partitioning Interval',
        'type': 'int',
        'algorithms': ['KISAO_0000562', 'KISAO_0000561'],
    },
    'KISAO_0000559': {
        'name': 'Initial Step Size',
        'type': 'float',
        'algorithms': ['KISAO_0000304'],
    },
    'KISAO_0000483': {
        'name': {
            'KISAO_0000561': 'Runge Kutta Stepsize',
            'KISAO_0000566': 'Internal Steps Size',
        },
        'type': 'float',
        'algorithms': ['KISAO_0000561', 'KISAO_0000566'],
    },
    'KISAO_0000565': {
        'name': 'Tolerance for Root Finder',
        'type': 'float',
        'algorithms': ['KISAO_0000566'],
    },
    'KISAO_0000567': {
        'name': 'Force Physical Correctness',
        'type': 'bool',
        'algorithms': ['KISAO_0000566'],
    },
    # 'KISAO_0000534': {
    #    'name': 'Deterministic Reactions',
    #    'type': 'string[]',
    #    'algorithms': ['KISAO_0000563'],
    # },
}


def get_algorithm_id(kisao_id):
    """ Get the COPASI id for an algorithm

    Args:
        kisao_id (:obj:`str`): KiSAO algorithm id

    Returns:
        :obj:`int`: COPASI id for algorithm
    """
    alg = KISAO_ALGORITHMS_MAP.get(kisao_id, None)
    if alg is None:
        raise NotImplementedError(
            "Algorithm with KiSAO id '{}' is not supported".format(kisao_id))
    return alg['id']


def set_function_parameter(algorithm_kisao_id, algorithm_function, parameter_kisao_id, value):
    """ Set a parameter of a COPASI simulation function

    Args:
        algorithm_kisao_id (:obj:`str`): KiSAO algorithm id
        algorithm_function (:obj:`types.FunctionType`): algorithm function
        parameter_kisao_id (:obj:`str`): KiSAO parameter id
        value (:obj:`string`): parameter value
    """
    parameter_attrs = KISAO_PARAMETERS_MAP.get(parameter_kisao_id, None)
    if parameter_attrs is None:
        NotImplementedError("Parameter '{}' is not supported".format(parameter_kisao_id))

    if isinstance(parameter_attrs['name'], str):
        parameter_name = parameter_attrs['name']
    else:
        parameter_name = parameter_attrs['name']['algorithm_kisao_id']
    parameter = algorithm_function.getParameter(parameter_name)
    if not isinstance(parameter, COPASI.CCopasiParameter):
        NotImplementedError("Parameter '{}' is not supported for algorithm '{}'".format(
            parameter_kisao_id, algorithm_kisao_id))

    if parameter_attrs['type'] == 'bool':
        assert(parameter.setBoolValue(value.lower() == 'true' or value.lower() == '1'))
    elif parameter_attrs['type'] == 'int':
        assert(parameter.setIntValue(int(value)))
    elif parameter_attrs['type'] == 'float':
        assert(parameter.setDblValue(float(value)))
    # elif parameter_attrs['type'] == 'string[]':
        # TODO: handle Partitioning Strategy / Deterministic Reactions for hybrid RK-45
    else:
        raise NotImplementedError("Parameter type '{}' is not supported".format(parameter_attrs['type']))

    # if the parameter is the random number generator seed (KISAO_0000488), turn on the flag to use it
    if parameter_kisao_id == 'KISAO_0000488':
        use_rand_seed_parameter = algorithm_function.getParameter('Use Random Seed')
        if not isinstance(use_rand_seed_parameter, COPASI.CCopasiParameter):
            raise NotImplementedError("Random seed could not be turned on for algorithm '{}'".format(algorithm_kisao_id))
        use_rand_seed_parameter.setBoolValue(True)


def exec_combine_archive(archive_file, out_dir):
    """ Execute the SED tasks defined in a COMBINE archive and save the outputs

    Args:
        archive_file (:obj:`str`): path to COMBINE archive
        out_dir (:obj:`str`): directory to store the outputs of the tasks
    """
    exec_simulations_in_archive(archive_file, exec_simulation, out_dir, apply_model_changes=True)


def exec_simulation(model_filename, model_sed_urn, simulation, working_dir, out_filename, out_format):
    ''' Execute a simulation and save its results

    Args:
       model_filename (:obj:`str`): path to the model
       model_sed_urn (:obj:`str`): SED URN for the format of the model (e.g., `urn:sedml:language:sbml`)
       simulation (:obj:`TimecourseSimulation`): simulation
       working_dir (:obj:`str`): directory of the SED-ML file
       out_filename (:obj:`str`): path to save the results of the simulation
       out_format (:obj:`SimulationResultsFormat`): format to save the results of the simulation (e.g., `HDF5`)
    '''
    # check that model is encoded in SBML
    if model_sed_urn != "urn:sedml:language:sbml":
        raise NotImplementedError("Model language with URN '{}' is not supported".format(model_sed_urn))

    # check that simulation is a time course simulation
    if not isinstance(simulation, TimecourseSimulation):
        raise NotImplementedError('{} is not supported'.format(simulation.__class__.__name__))

    # check that model parameter changes have already been applied (because handled by :obj:`exec_simulations_in_archive`)
    if simulation.model_parameter_changes:
        raise NotImplementedError('Model parameter changes are not supported')

    # check that the desired output format is supported
    if out_format != SimulationResultsFormat.HDF5:
        raise NotImplementedError("Simulation results format '{}' is not supported".format(out_format))

    # Read the model located at `os.path.join(working_dir, model_filename)` in the format
    # with the SED URN `model_sed_urn`.
    data_model = COPASI.CRootContainer.addDatamodel()
    if not data_model.importSBML(model_filename):
        raise ValueError("'{}' could not be imported".format(model_filename))

    # Load the algorithm specified by `simulation.algorithm`
    algorithm_id = get_algorithm_id(simulation.algorithm.kisao_term.id)
    task = data_model.getTask('Time-Course')
    assert(task.setMethodType(algorithm_id))
    method = task.getMethod()

    # Apply the algorithm parameter changes specified by `simulation.algorithm_parameter_changes`
    for parameter_change in simulation.algorithm_parameter_changes:
        set_function_parameter(simulation.algorithm.kisao_term.id, method,
                               parameter_change.parameter.kisao_term.id, parameter_change.value)

    # Execute simulation
    task.setScheduled(True)

    model = data_model.getModel()

    problem = task.getProblem()
    model.setInitialTime(simulation.start_time)
    problem.setOutputStartTime(simulation.output_start_time)
    problem.setDuration(simulation.end_time - simulation.start_time)
    problem.setStepNumber(int(simulation.num_time_points *
                              (simulation.end_time - simulation.start_time) /
                              (simulation.end_time - simulation.output_start_time)))
    problem.setTimeSeriesRequested(True)
    problem.setAutomaticStepSize(False)
    problem.setOutputEvent(False)

    try:
        result = task.process(True)
        assert(result)
        time_series = task.getTimeSeries()
        assert(time_series.getRecordedSteps() == simulation.num_time_points + 1)
    except:
        error_msg = 'Simulation failed.'
        if COPASI.CCopasiMessage.size() > 0:
            error_msg += '\n\n' + COPASI.CCopasiMessage.getAllMessageText(True)
        raise ValueError(error_msg)

    # collect simulation predictions
    time = numpy.linspace(simulation.output_start_time, simulation.end_time,
                          simulation.num_time_points + 1).reshape((simulation.num_time_points + 1, 1))
    data = numpy.full((simulation.num_time_points + 1, len(simulation.model.variables)), numpy.nan)

    vars = sorted(simulation.model.variables, key=lambda var: var.target)
    var_id_to_idx = {}
    for i_var, var in enumerate(vars):
        match = re.match(r"^/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species\[@id='(.*?)'\]$", var.target)
        if not match:
            raise ValueError("Unable to parse SED-ML target '{}'".format(var.target))
        var_id_to_idx[match.group(1)] = i_var

    get_data_function = getattr(time_series, 'getConcentrationData')
    for i_var in range(0, time_series.getNumVariables()):
        var_id = time_series.getTitle(i_var)
        i_report_var = var_id_to_idx.get(var_id, None)
        if i_report_var is None:
            continue

        for i_step in range(0, time_series.getRecordedSteps()):
            data[i_step, i_report_var] = get_data_function(i_step, i_var)

    i_missing_vars = numpy.where(numpy.any(numpy.isnan(data), axis=0))[0].tolist()
    if i_missing_vars:
        raise ValueError('Some targets could not be recorded:\n  - {}'.format(
            '\n  - '.join(vars[i_missing_var].target for i_missing_var in i_missing_vars)))

    # save results to file
    results_df = pandas.DataFrame(numpy.concatenate((time, data), 1), columns=['time'] + [var.id for var in vars])
    results_df.to_csv(out_filename, index=False)
