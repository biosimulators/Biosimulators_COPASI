"""-*- coding: utf-8 -*-
Copyright (C) 2017 - 2018 by Pedro Mendes, Virginia Tech Intellectual
Properties, Inc., University of Heidelberg, and University of
of Connecticut School of Medicine.
All rights reserved.

Copyright (C) 2010 - 2016 by Pedro Mendes, Virginia Tech Intellectual
Properties, Inc., University of Heidelberg, and The University
of Manchester.
All rights reserved.

Copyright (C) 2009 by Pedro Mendes, Virginia Tech Intellectual
Properties, Inc., EML Research, gGmbH, University of Heidelberg,
and The University of Manchester.
All rights reserved.


This is an example on how to import an sbml file
create a report for a time course simulation
and run a time course simulation
"""

import sys
import os
import json
import requests
from COPASI import *

# create a datamodel
try:
    dataModel = CRootContainer.addDatamodel()
except:
    dataModel = CCopasiRootContainer.addDatamodel()


ALGORITHMS_MAP = {
    "Method_DsaLsodar": CTaskEnum.Method_DsaLsodar,
    "Method_RADAU5": CTaskEnum.Method_RADAU5,
    "Method_stochastic": CTaskEnum.Method_stochastic,
    "Method_directMethod": CTaskEnum.Method_directMethod,
    "Method_tauLeap": CTaskEnum.Method_tauLeap,
    "Method_adaptiveSA": CTaskEnum.Method_adaptiveSA,
    "Method_hybrid": CTaskEnum.Method_hybrid,
    "Method_hybridLSODA": CTaskEnum.Method_hybridLSODA,
    "Method_hybridODE45": CTaskEnum.Method_hybridODE45,
    "Method_stochasticRunkeKuttaRI5": CTaskEnum.Method_stochasticRunkeKuttaRI5
}


# Get environment variables
JOB_ID = os.getenv('JOB_ID')
INITIAL_TIME = os.getenv('INITIAL_TIME')
NUMBER_OF_POINTS = os.getenv('NUMBER_OF_POINTS')
OUTPUT_START_TIME = os.getenv('OUTPUT_START_TIME')
OUTPUT_END_TIME = os.getenv('OUTPUT_END_TIME')
JOBHOOK_URL = os.getenv('JOBHOOK_URL')
ALGORITHM = os.getenv('ALGORITHM')


# TODO: Combine print, error and jobhook_request_builder in single method
def main(args):
    # the only argument to the main routine should be the name of an SBML file
    if len(args) != 1:
        sys.stderr.write("Usage: copasi_sim  SBMLFILE\n")
        print(jobhook_request_builder("Usage: copasi_sim  SBMLFILE\n", error=True))
        return 1

    filename = args[0]
    try:
        # load the model
        if not dataModel.importSBML(filename):
            print("Couldn't load {0}:".format(filename))
            print(jobhook_request_builder("Couldn't load {0}:".format(filename)))
            print(CCopasiMessage.getAllMessageText())
            print(jobhook_request_builder(CCopasiMessage.getAllMessageText()))
    except:
        sys.stderr.write("Error while importing the model from file named \"" + filename + "\".\n")
        print(jobhook_request_builder("Error while importing the model from file named \"" + filename + "\".\n", error=True))
        return 1

    model = dataModel.getModel()
    assert model is not None

    # get the trajectory task object
    trajectoryTask = dataModel.getTask("Time-Course")
    assert (isinstance(trajectoryTask, CTrajectoryTask))

    trajectoryTask.setMethodType(ALGORITHMS_MAP[ALGORITHM])

    # activate the task so that it will be run when the model is saved
    # and passed to CopasiSE
    trajectoryTask.setScheduled(True)

    # create a new report that captures the time course result
    report = create_report(model)
    # set the report for the task
    trajectoryTask.getReport().setReportDefinition(report)
    # set the output filename
    trajectoryTask.getReport().setTarget("result.ida")
    # don't append output if the file exists, but overwrite the file
    trajectoryTask.getReport().setAppend(False)

    # get the problem for the task to set some parameters
    problem = trajectoryTask.getProblem()
    assert (isinstance(problem, CTrajectoryProblem))

    # simulate 100 steps
    problem.setStepNumber(int(NUMBER_OF_POINTS))
    # start at time 0
    dataModel.getModel().setInitialTime(int(INITIAL_TIME))
    # simulate a duration of 10 time units
    problem.setDuration(int(OUTPUT_END_TIME) - int(OUTPUT_START_TIME))
    # tell the problem to actually generate time series data
    problem.setTimeSeriesRequested(True)
    # tell the problem, that we want exactly 100 simulation steps (not automatically controlled)
    problem.setAutomaticStepSize(False)
    # tell the problem, that we don't want additional output points for event assignments
    problem.setOutputEvent(True)

    # set some parameters for the LSODA method through the method
    method = trajectoryTask.getMethod()

    parameter = method.getParameter("Absolute Tolerance")
    assert parameter is not None
    assert parameter.getType() == CCopasiParameter.Type_UDOUBLE
    parameter.setValue(1.0e-12)

    try:
        # now we run the actual trajectory
        result = trajectoryTask.process(True)
    except:
        sys.stderr.write("Error. Running the time course simulation failed.\n")
        print(jobhook_request_builder("Error. Running the time course simulation failed.\n", error=True))
        # check if there are additional error messages
        if CCopasiMessage.size() > 0:
            # print the messages in chronological order
            sys.stderr.write(CCopasiMessage.getAllMessageText(True))
            print(jobhook_request_builder(CCopasiMessage.getAllMessageText(True), error=True))
        return 1
    if not result:
        sys.stderr.write("Error. Running the time course simulation failed.\n")
        print(jobhook_request_builder("Error. Running the time course simulation failed.\n", error=True))
        # check if there are additional error messages
        if CCopasiMessage.size() > 0:
            # print the messages in chronological order
            sys.stderr.write(CCopasiMessage.getAllMessageText(True))
            print(jobhook_request_builder(CCopasiMessage.getAllMessageText(True), error=True))
        return 1

    # look at the timeseries
    print_results(trajectoryTask)
    

def print_results(trajectoryTask):
    timeSeries = trajectoryTask.getTimeSeries()
    # we simulated 100 steps, including the initial state, this should be
    # 101 step in the timeseries
    # assert timeSeries.getRecordedSteps() == 101
    print("The time series consists of {0} steps.".format(timeSeries.getRecordedSteps()))
    print(jobhook_request_builder("The time series consists of {0} steps.".format(timeSeries.getRecordedSteps())))
    print("Each step contains {0} variables.".format(timeSeries.getNumVariables()))
    print(jobhook_request_builder("Each step contains {0} variables.".format(timeSeries.getNumVariables())))
    print("\nThe final state is: ")
    print(jobhook_request_builder("\nThe final state is: "))
    iMax = timeSeries.getNumVariables()
    lastIndex = timeSeries.getRecordedSteps() - 1
    for i in range(0, iMax):
        # here we get the particle number (at least for the species)
        # the unit of the other variables may not be particle numbers
        # the concentration data can be acquired with getConcentrationData
        print("  {0}: {1}".format(timeSeries.getTitle(i), timeSeries.getData(lastIndex, i)))
        print(jobhook_request_builder("  {0}: {1}".format(timeSeries.getTitle(i), timeSeries.getData(lastIndex, i))))
    # the CTimeSeries class now has some new methods to get all variable titles
    # as a python list (getTitles())
    # and methods to get the complete time course data for a certain variable based on
    # the variables index or the corresponding model object.
    # E.g. to get the particle numbers of the second variable as a python list
    # you can use getDataForIndex(1) and to get the concentration data you use
    # getConcentrationDataForIndex(1)
    # To get the complete particle number data for the second metabolite of the model
    # you can use getDataForObject(model.getMetabolite(1)) and to get the concentration
    # data you use getConcentrationDataForObject.
    # print timeSeries.getTitles()
    # print timeSeries.getDataForIndex(1)
    # print timeSeries.getDataForObject(model)


def create_report(model):
    # create a report with the correct filename and all the species against
    # time.
    reports = dataModel.getReportDefinitionList()
    # create a report definition object
    report = reports.createReportDefinition("Report", "Output for timecourse")
    # set the task type for the report definition to timecourse
    report.setTaskType(CTaskEnum.Task_timeCourse)
    # we don't want a table
    report.setIsTable(False)
    # the entries in the output should be separated by a ", "
    report.setSeparator(CCopasiReportSeparator(", "))
    # we need a handle to the header and the body
    # the header will display the ids of the metabolites and "time" for
    # the first column
    # the body will contain the actual timecourse data
    header = report.getHeaderAddr()
    body = report.getBodyAddr()
    body.push_back(
        CRegisteredCommonName(CCommonName(dataModel.getModel().getCN().getString() + ",Reference=Time").getString()))
    body.push_back(CRegisteredCommonName(report.getSeparator().getCN().getString()))
    header.push_back(CRegisteredCommonName(CDataString("time").getCN().getString()))
    header.push_back(CRegisteredCommonName(report.getSeparator().getCN().getString()))
    iMax = model.getMetabolites().size()
    for i in range(0, iMax):
        metab = model.getMetabolite(i)
        assert metab is not None
        # we don't want output for FIXED metabolites right now
        if metab.getStatus() != CModelEntity.Status_FIXED:
            # we want the concentration oin the output
            # alternatively, we could use "Reference=Amount" to get the
            # particle number
            body.push_back(
                CRegisteredCommonName(metab.getObject(CCommonName("Reference=Concentration")).getCN().getString()))
            # add the corresponding id to the header
            header.push_back(CRegisteredCommonName(CDataString(metab.getSBMLId()).getCN().getString()))
            # after each entry, we need a separator
            if i != iMax - 1:
                body.push_back(CRegisteredCommonName(report.getSeparator().getCN().getString()))
                header.push_back(CRegisteredCommonName(report.getSeparator().getCN().getString()))
    return report


def jobhook_request_builder(msg: str, error=False):
    try:
        info_type = {True: 'INFO', False: 'ERROR'}[error]
        req_data = {
            'jobId': JOB_ID,
            'infoType': info_type,
            'simulator': 'copasi'
            'message': msg
        }
        return requests.post(JOBHOOK_URL, json.dumps(req_data))
    except BaseException as ex:
        print('Exception occurred: ', str(ex))


if __name__ == '__main__':
    main(sys.argv[1:])
