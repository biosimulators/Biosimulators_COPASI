""" BioSimulators-compliant command-line interface to the `COPASI <http://copasi.org/>`_ simulation program.

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from ._version import __version__
from .core import exec_sedml_docs_in_combine_archive
from biosimulators_utils.simulator.cli import build_cli
from biosimulators_utils.simulator.data_model import AlgorithmSubstitutionPolicy
from biosimulators_utils.simulator.environ import ENVIRONMENT_VARIABLES
import COPASI

App = build_cli('copasi', __version__,
                'COPASI', COPASI.__version__, 'http://copasi.org',
                exec_sedml_docs_in_combine_archive,
                environment_variables=[
                    ENVIRONMENT_VARIABLES[AlgorithmSubstitutionPolicy]
                ])


def main():
    with App() as app:
        app.run()
