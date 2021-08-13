import COPASI

from ._version import __version__  # noqa: F401
# :obj:`str`: version

from .core import exec_sedml_docs_in_combine_archive  # noqa: F401


def get_simulator_version():
    """ Get the version of COPASI

    Returns:
        :obj:`str`: version
    """
    return COPASI.__version__
