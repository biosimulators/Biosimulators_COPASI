import COPASI

from ._version import __version__  # noqa: F401
# :obj:`str`: version

from biosimulators_copasi.core import exec_sed_task, preprocess_sed_task, exec_sed_doc, exec_sedml_docs_in_combine_archive  # noqa: F401

__all__ = [
    '__version__',
    'get_simulator_version',
    'exec_sed_task',
    'preprocess_sed_task',
    'exec_sed_doc',
    'exec_sedml_docs_in_combine_archive',
]


def get_simulator_version():
    """ Get the version of COPASI

    Returns:
        :obj:`str`: version
    """
    return COPASI.__version__
