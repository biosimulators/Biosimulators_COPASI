""" BioSimulators-compliant command-line interface to the `COPASI <http://copasi.org/>`_ simulation program.

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Akhil Marupilla <akhilmteja@gmail.com>
:Date: 2020-12-13
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

import cement
import termcolor
import COPASI
from . import get_simulator_version
from ._version import __version__
from .core import exec_sedml_docs_in_combine_archive
from .utils import fix_copasi_generated_combine_archive as fix_copasi_generated_combine_archive_func
from biosimulators_utils.config import get_config
from biosimulators_utils.simulator.cli import build_cli
from biosimulators_utils.simulator.data_model import EnvironmentVariable
from biosimulators_utils.simulator.environ import ENVIRONMENT_VARIABLES as DEFAULT_ENVIRONMENT_VARIABLES

ENVIRONMENT_VARIABLES = list(DEFAULT_ENVIRONMENT_VARIABLES.values())
ENVIRONMENT_VARIABLES.append(
    EnvironmentVariable(
        name='FIX_COPASI_GENERATED_COMBINE_ARCHIVE',
        description=(
            'Whether to make COPASI-generated COMBINE archives compatible with the '
            'specifications of the OMEX manifest and SED-ML standards.'
        ),
        options=['0', '1'],
        default='0',
        more_info_url='https://docs.biosimulators.org/Biosimulators_COPASI/source/biosimulators_copasi.html',
    )
)

App = build_cli('biosimulators-copasi', __version__,
                'COPASI', COPASI.__version__ + f" (BASICO: {get_simulator_version()})", 'http://copasi.org',
                exec_sedml_docs_in_combine_archive,
                environment_variables=ENVIRONMENT_VARIABLES,
                )


def main():
    with App() as app:
        app.run()


class FixCopasiGeneratedCombineArchiveController(cement.Controller):
    """ Controller for fixing COPASI-generated COMBINE archives """

    class Meta:
        label = 'base'
        help = 'Fix a COPASI-generated COMBINE/OMEX archive'
        description = (
            'Correct a COPASI-generated COMBINE/OMEX archive to be consistent with '
            'the specifications of the OMEX manifest and SED-ML formats'
        )
        arguments = [
            (
                ['-i', '--in-file'],
                dict(
                    type=str,
                    required=True,
                    help='Path to COMBINE/OMEX file to correct',
                ),
            ),
            (
                ['-o', '--out-file'],
                dict(
                    type=str,
                    required=True,
                    help='Path to save the corrected archive',
                ),
            ),
        ]

    @cement.ex(hide=True)
    def _default(self):
        args = self.app.pargs
        config = get_config()
        try:
            fix_copasi_generated_combine_archive_func(args.in_file, args.out_file)
        except Exception as exception:
            if config.DEBUG:
                raise
            raise SystemExit(termcolor.colored(str(exception), 'red')) from exception


class FixCopasiGeneratedCombineArchiveApp(cement.App):
    """ Command line application for fixing COPASI-generated COMBINE/OMEX archives """
    class Meta:
        label = 'fix-copasi-generated-combine-archive'
        base_controller = 'base'
        handlers = [
            FixCopasiGeneratedCombineArchiveController,
        ]


def fix_copasi_generated_combine_archive():
    with FixCopasiGeneratedCombineArchiveApp() as app:
        app.run()
