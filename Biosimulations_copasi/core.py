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
# from COPASI import * as copasi
import COPASI as copasi
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

    # extract files from archive and simulate
    try:
        # Create temp directory
        tmp_dir = tempfile.mkdtemp()
        
        # Create root container
        # data_model = copasi.CRootContainer().addDatamodel()

        # open omex archive
        # data_model.openCombineArchive(fileName=archive_file)  --> Would only work for single files (single sedml, sbml or cps)

        # Process combine archive
        archive = libcombine.CombineArchive()
        archive.initializeFromArchive(archive_file)
        archive.extractTo(tmp_dir)
        manifest = archive.getManifest()
        contents = manifest.getListOfContents()

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
            data_model = copasi.CRootContainer().addDatamodel()
            data_model.importSEDML(sedml_path)
            for i in range(0, len(data_model.getTaskList())):
                task = data_model.getTaskList().get(i)
                task.getReport().setTarget(sedml_out_dir)
                task.initialize(55)
                task.process(True)

    finally:
        shutil.rmtree(tmp_dir)



