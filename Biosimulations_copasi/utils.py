import COPASI as copasi

def create_time_course_report(dataModel):
    """Create Report definition for Time course simulation task
    
    :param dataModel: Data model of root container being used to hold related tasks
    :type: copasi.CDataModel
    :return: Report definition for time course simulation  
    :rtype: copasi.CReportDefinition
    """    
    model = dataModel.getModel()
    # create a report with the correct filename and all the species against
    # time.
    reports = dataModel.getReportDefinitionList()
    # create a report definition object
    report = reports.createReportDefinition("Time-Course", "Output for timecourse")
    # set the task type for the report definition to timecourse
    report.setTaskType(copasi.CTaskEnum.Task_timeCourse)
    # we don't want a table
    report.setIsTable(False)
    # the entries in the output should be separated by a ", "
    report.setSeparator(copasi.CCopasiReportSeparator(", "))
    # we need a handle to the header and the body
    # the header will display the ids of the metabolites and "time" for
    # the first column
    # the body will contain the actual timecourse data
    header = report.getHeaderAddr()
    body = report.getBodyAddr()
    body.push_back(
        copasi.CRegisteredCommonName(copasi.CCommonName(dataModel.getModel().getCN().getString() + ",Reference=Time").getString()))
    body.push_back(copasi.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    header.push_back(copasi.CRegisteredCommonName(copasi.CDataString("time").getCN().getString()))
    header.push_back(copasi.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    iMax = model.getMetabolites().size()

    for i in range(0, iMax):
        metab = model.getMetabolite(i)
        assert metab is not None
        # we don't want output for FIXED metabolites right now
        if metab.getStatus() != copasi.CModelEntity.Status_FIXED:
            # we want the concentration oin the output
            # alternatively, we could use "Reference=Amount" to get the
            # particle number
            body.push_back(
                copasi.CRegisteredCommonName(metab.getObject(copasi.CCommonName("Reference=Concentration")).getCN().getString()))
            # add the corresponding id to the header
            header.push_back(copasi.CRegisteredCommonName(copasi.CDataString(metab.getSBMLId()).getCN().getString()))
            # after each entry, we need a separator
            if i != iMax - 1:
                body.push_back(copasi.CRegisteredCommonName(report.getSeparator().getCN().getString()))
                header.push_back(copasi.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    return report