![Latest version](https://img.shields.io/github/v/tag/biosimulators/Biosimulators_COPASI)
[![PyPI](https://img.shields.io/pypi/v/biosimulators_copasi)](https://pypi.org/project/biosimulators_copasi/)
[![Docker image](https://github.com/biosimulators/Biosimulators_COPASI/workflows/Publish%20Docker%20To%20Hub/badge.svg)](https://github.com/biosimulators/Biosimulators_COPASI/actions?query=workflow%3A%22Publish+Docker+To+Hub%22)
[![Docker image build](https://github.com/biosimulators/Biosimulators_COPASI/workflows/Build%20Docker%20image/badge.svg)](https://github.com/biosimulators/Biosimulators_COPASI/actions?query=workflow%3A%22Build+Docker+image%22)
[![Unit tests](https://github.com/biosimulators/Biosimulators_COPASI/workflows/Unit%20tests/badge.svg)](https://github.com/biosimulators/Biosimulators_COPASI/actions?query=workflow%3A%22Unit+tests%22)
[![Documentation](https://img.shields.io/github/license/biosimulators/Biosimulators_COPASI?badges-awesome-green.svg)](https://biosimulators.github.io/Biosimulators_COPASI/)
[![Issues](https://img.shields.io/github/issues/biosimulators/Biosimulators_COPASI)](https://github.com/biosimulators/Biosimulators_COPASI/issues)
[![License](https://img.shields.io/github/license/biosimulators/Biosimulators_COPASI?badges-awesome-green.svg)](https://github.com/biosimulators/Biosimulators_COPASI/blob/dev/LICENSE)


# BioSimulators-COPASI 
BioSimulators-compliant command-line interface and Docker image for the [COPASI](http://copasi.org/) simulation program.

This command-line interface and Docker image enable users to use COPASI to execute [COMBINE/OMEX archives](https://combinearchive.org/) that describe one or more simulation experiments (in [SED-ML format](https://sed-ml.org)) of one or more models (in [SBML format](http://sbml.org])).

A list of the algorithms and algorithm parameters supported by COPASI is available at [BioSimulators](https://biosimulators.org/simulators/copasi).

A simple web application and web service for using COPASI to execute COMBINE/OMEX archives is also available at [runBioSimulations](https://run.biosimulations.org).

## Contents
* [Installation](#installation)
* [Usage](#local-usage)
* [Documentation](#documentation)
* [License](#license)
* [Development team](#development-team)
* [Questions and comments](#questions-and-comments)

## Installation

### Install Python package
```
pip install git+https://github.com/biosimulators/Biosimulators_COPASI
```

### Install Docker image
```
docker pull ghcr.io/biosimulators/copasi
```

## Usage

### Local usage
```
usage: copasi [-h] [-d] [-q] -i ARCHIVE [-o OUT_DIR] [-v]

BioSimulators-compliant command-line interface to the COPASI simulation program <http://copasi.org>.

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
```

### Usage through Docker container
```
docker run \
  --tty \
  --rm \
  --mount type=bind,source="$(pwd)"/tests/fixtures,target=/root/in,readonly \
  --mount type=bind,source="$(pwd)"/tests/results,target=/root/out \
  ghcr.io/biosimulators/copasi:latest \
    -i /root/in/BIOMD0000000297.omex \
    -o /root/out
```

## Documentation
Documentation is available at https://biosimulators.github.io/Biosimulators_COPASI/.

## License
This package is released under the [MIT license](LICENSE).

## Development team
This package was developed by the [Center for Reproducible Biomedical Modeling](http://reproduciblebiomodels.org).

## Questions and comments
Please contact the [BioSimulators Team](mailto:info@biosimulators.org) with any questions or comments.
  