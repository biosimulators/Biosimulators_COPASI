Tutorial
========

BioSimulators-COPASI is available as a command-line program and as a command-line program encapsulated into a Docker image.


Creating COMBINE/OMEX archives and encoding simulation experiments into SED-ML
------------------------------------------------------------------------------

Information about how to create COMBINE/OMEX archives which can be executed by BioSimulators-COPASI is available at `BioSimulators <https://biosimulators.org/help>`_.

A list of the algorithms and algorithm parameters supported by COPASI is available at `BioSimulators <https://biosimulators.org/simulators/copasi>`_.

Conventions for values of changes to algorithm parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This package follows SED-ML conventions for Boolean-, integer- and float-valued parameters.

New values of parameter KISAO_0000534 (list of deterministic reactions) of KISAO_0000563 (hybrid RK-45 method) should be JSON-encoded lists of the SBML ids of the reactions which should be deterministically simulated using the RK-45 portion of the method. For example, the following SED-ML indicates that the RK-45 partition should consist of the reactions that have the SBML ids ``reaction_1`` and ``reaction_2``.

.. code-block:: text

    <algorithm kisaoID="KISAO:0000563">
      <algorithmParameter kisaoID="KISAO:0000534" value="[\"reaction_1\", \"reaction_2\"]" />
    </algorithm>


Command-line program for executing COMBINE/OMEX archives
--------------------------------------------------------

The command-line program can be used to execute COMBINE/OMEX archives that describe simulations as illustrated below.

.. code-block:: text

    usage: biosimulators-copasi [-h] [-d] [-q] -i ARCHIVE [-o OUT_DIR] [-v]

    BioSimulators-compliant command-line interface to the COPASI <http://copasi.org> simulation program.

    optional arguments:
      -h, --help            show this help message and exit
      -d, --debug           full application debug mode
      -q, --quiet           suppress all console output
      -i ARCHIVE, --archive ARCHIVE
                            Path to OMEX file which contains one or more SED-ML-
                            encoded simulation experiments
      -o OUT_DIR, --out-dir OUT_DIR
                            Directory to save outputs
      -v, --version         show program's version number and exit

For example, the following command could be used to execute the simulations described in ``./modeling-study.omex`` and save their results to ``./``:

.. code-block:: text

    biosimulators-copasi -i ./modeling-study.omex -o ./


Docker image with a command-line entrypoint
-------------------------------------------

The entrypoint to the Docker image supports the same command-line interface described above.

For example, the following command could be used to use the Docker image to execute the same simulations described in ``./modeling-study.omex`` and save their results to ``./``:

.. code-block:: text

    docker run \
        --tty \
        --rm \
        --mount type=bind,source="$(pwd),target=/tmp/working-dir \
        ghcr.io/biosimulators/copasi:latest \
            -i /tmp/working-dir/modeling-study.omex \
            -o /tmp/working-dir

Command-line program for correcting COMBINE/OMEX archives created by COPASI
---------------------------------------------------------------------------

The ``fix-copasi-generated-combine-archive`` command-line program can be used to align COMBINE/OMEX archives created by COPASI with the specifications of the OMEX manifest and SED-ML formats.

.. code-block:: text

    usage: fix-copasi-generated-combine-archive [-h] [-d] [-q] -i IN_FILE -o OUT_FILE

    Correct a COPASI-generated COMBINE/OMEX archive to be consistent with the specifications of the OMEX manifest and SED-ML formats

    optional arguments:
      -h, --help            show this help message and exit
      -d, --debug           full application debug mode
      -q, --quiet           suppress all console output
      -i IN_FILE, --in-file IN_FILE
                            Path to COMBINE/OMEX file to correct
      -o OUT_FILE, --out-file OUT_FILE
                            Path to save the corrected archive

For example, the following command could be used to correct the example COPASI-generated archive in the ``tests/fixtures`` directory:

.. code-block:: text

    fix-copasi-generated-combine-archive \
        -i tests/fixtures/copasi-34-export.omex \
        -o tests/fixtures/copasi-34-export-fixed.omex
