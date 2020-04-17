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
import warnings
import zipfile
import COPASI as copasi
import sys
importlib.reload(libcombine)


__all__ = ['exec_combine_archive']


def exec_combine_archive(archive_file, out_dir):
    """ Execute the SED tasks defined in a COMBINE archive and save the outputs

    Args:
        archive_file (:obj:`str`): path to COMBINE archive
        out_dir (:obj:`str`): directory to store the outputs of the tasks
    """

    # check that archive exists and is in zip format
    if not os.path.isfile(archive_file):
        raise FileNotFoundError("File does not exist: {}".format(archive_file))

    if not zipfile.is_zipfile(archive_file):
        raise IOError("File is not an OMEX Combine Archive in zip format: {}".format(archive_file))

    try:
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
            if not os.path.isdir(sedml_out_dir):
                os.makedirs(sedml_out_dir)

            # Create a base Copasi container to hold all the Tasks
            try:
                data_model = copasi.CRootContainer.addDatamodel()
            except:
                data_model = copasi.CRootContainer.getUndefinedFunction()
            data_model.importSEDML(sedml_path)

            # Run all Tasks
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
                    task_name = task_name[:len(task_name)-1]
                # Set output file for the task
                task.getReport().setTarget(os.path.join(sedml_out_dir, f'{task_name}.txt'))
                # Initialising the task with default values
                task.initialize(119)
                # Run the task
                task.process(True)

    finally:
        shutil.rmtree(tmp_dir)
