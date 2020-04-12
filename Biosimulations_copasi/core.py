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
import COPASI as cps
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
        # Create root container
        data_model = cps.CRootContainer().addDatamodel()

        # open omex archive
        data_model.openCombineArchive(fileName=archive_file)

        model_analyser = cps.CModelAnalyzer(data_model.getModel())

        # run all sedml files
        for sedml_location in sedml_locations:
            sedml_path = os.path.join(tmp_dir, sedml_location)
            sedml_out_dir = os.path.join(out_dir, os.path.splitext(sedml_location)[0])
            if not os.path.isdir(sedml_out_dir):
                os.makedirs(sedml_out_dir)
            factory = tellurium.sedml.tesedml.SEDMLCodeFactory(sedml_path,
                                                               workingDir=os.path.dirname(sedml_path),
                                                               createOutputs=True,
                                                               saveOutputs=True,
                                                               outputDir=sedml_out_dir,
                                                               )
            factory.executePython()
    finally:
        shutil.rmtree(tmp_dir)



