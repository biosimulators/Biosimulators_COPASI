""" BioSimulators-compliant command-line interface to the `COPASI <http://copasi.org/>`_ simulation program.

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from . import get_simulator_version
from ._version import __version__
from .core import exec_sedml_docs_in_combine_archive
from biosimulators_utils.simulator.cli import build_cli

App = build_cli('biosimulators-copasi', __version__,
                'COPASI', get_simulator_version(), 'http://copasi.org',
                exec_sedml_docs_in_combine_archive,
                )


def main():
    with App() as app:
        app.run()
