""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-04-12
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

import importlib
import libcombine
import os
import shutil
import tempfile
import zipfile
import libsedml
import COPASI as copasi
import sys
from .utils import create_time_course_report
import pandas as pd
importlib.reload(libcombine)


__all__ = ['exec_combine_archive']


def exec_combine_archive(archive_file, out_dir):
    """Execute the SED tasks defined in a COMBINE archive and save the outputs

    :param archive_file: path to COMBINE archive
    :type archive_file: str
    :param out_dir: directory to store the outputs of the tasks
    :type out_dir: str
    :raises FileNotFoundError: When the combine archive is not found
    :raises IOError: When file is not an OMEX combine archive
    """
    # check that archive exists and is in zip format
    if not os.path.isfile(archive_file):
        raise FileNotFoundError("File does not exist: {}".format(archive_file))

    if not zipfile.is_zipfile(archive_file):
        raise IOError("File is not an OMEX Combine Archive in zip format: {}".format(archive_file))

    try:
        archive_file = os.path.abspath(archive_file)
        out_dir = os.path.abspath(out_dir)
        # Create temp directory
        tmp_dir = tempfile.mkdtemp()

        # Get list of contents from Combine Archive
        archive = libcombine.CombineArchive()
        is_initialised = archive.initializeFromArchive(archive_file)
        is_extracted = archive.extractTo(tmp_dir)
        manifest = archive.getManifest()
        contents = manifest.getListOfContents()

        if not is_initialised or not is_extracted:
            sys.exit("Problem while initialising/extract combine archive")

        # Get location of all SEDML files
        sedml_locations = list()
        for content in contents:
            if content.isFormat('sedml'):
                sedml_locations.append(content.getLocation())


        # run all sedml files
        for sedml_location in sedml_locations:
            sedml_path = os.path.join(tmp_dir, sedml_location)
            sedml_out_dir = os.path.join(out_dir, os.path.splitext(sedml_location)[0])

            sedml_doc = libsedml.readSedMLFromFile(sedml_path)
            tasks = sedml_doc.getListOfTasks()
            task_name_list = [task.getId() for task in tasks]

            if not os.path.isdir(sedml_out_dir):
                os.makedirs(sedml_out_dir)

            # Create a base Copasi container to hold all the Tasks
            try:
                data_model = copasi.CRootContainer.addDatamodel()
            except BaseException as ex:
                data_model = copasi.CRootContainer.getUndefinedFunction()
            data_model.importSEDML(sedml_path)

            report = create_time_course_report(data_model)
            # Run all Tasks
            task_name_index = 0
            for task_index in range(0, len(data_model.getTaskList())):
                task = data_model.getTaskList().get(task_index)
                # Get Name and Class of task as string
                task_str = str(task)
                try:
                    # Get name of Task
                    task_name = task_str.split('"')[1]
                except IndexError:
                    # Get Class name if Task name is not present
                    task_name = task_str.split("'")[1].split("*")[0]
                    task_name = task_name[:len(task_name) - 1]
                # Set output file for the task
                if task_name == 'Time-Course':
                    task.setScheduled(True)
                    # task.getReport().setReportDefinition(report)
                    report_def = task.getReport().compile('')
                    if not report_def:
                        print('No Report definition found in SEDML, setting to a default definition')
                        task.getReport().setReportDefinition(report)
                    sedml_task_name = task_name_list[task_name_index]
                    report_path = os.path.join(sedml_out_dir, f'{sedml_task_name}.csv')
                    task.getReport().setTarget(report_path)
                    task_name_index = task_name_index + 1
                    print(f'Generated report for Simulation "{sedml_task_name}": {report_path}')
                    # If file exists, don't append in it, overwrite it.
                    task.getReport().setAppend(False)
                    # Initialising the task with default values
                    task.initialize(119)
                    # TODO: Few tasks run but no Report is generated (Like Time Course)
                    # @body: Create report generation methods for such tasks.
                    # Run the task
                    task.process(True)
                    try:
                        pd.read_csv(report_path).drop(" ", axis=1).to_csv(report_path, index=False)
                    except KeyError as k:
                        print(f"No trailing commas were found in {report_path}")
                    df = pd.read_csv(report_path)
                    cols = list(df.columns)
                    new_cols = list()
                    for col in cols:
                        new_cols.append(col.split()[-1])
                    df.columns = new_cols
                    df.to_csv(report_path, index=False)

    finally:
        shutil.rmtree(tmp_dir)
