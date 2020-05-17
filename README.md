# Biosimulations_COPASI 
BioSimulations-compliant command-line interface to the [COPASI](http://copasi.org/) simulation program.

[![Docker image](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/workflows/Publish%20Docker%20To%20Hub/badge.svg)](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/actions?query=workflow%3A%22Publish+Docker+To+Hub%22)
[![Docker image build](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/workflows/Build%20Docker%20image/badge.svg)](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/actions?query=workflow%3A%22Build+Docker+image%22)
[![Unit tests](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/workflows/Unit%20tests/badge.svg)](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/actions?query=workflow%3A%22Unit+tests%22)
[![Issues](https://img.shields.io/github/issues/reproducible-biomedical-modeling/Biosimulations_COPASI?logo=GitHub)](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/issues)
[![License](https://img.shields.io/github/license/reproducible-biomedical-modeling/Biosimulations_COPASI?badges-awesome-green.svg&logo=GitHub)](https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI/blob/master/LICENSE)

## Contents
* [Installation](#installation)
* [Usage](#local-usage)
* [License](#license)
* [Development team](#development-team)
* [Questions and comments](#questions-and-comments)

## Installation

### Install Python package
```
pip install git+https://github.com/reproducible-biomedical-modeling/Biosimulations_COPASI
```

### Install Docker image
```
docker pull crbm/biosimulations_copasi
```

## Local usage
```
usage: copasi [-h] [-d] [-q] -i ARCHIVE [-o OUT_DIR] [-v]

BioSimulations-compliant command-line interface to the COPASI simulation program <http://copasi.org>.

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

## Usage through Docker container
```
docker run \
  --tty \
  --rm \
  --mount type=bind,source="$(pwd)"/tests/fixtures,target=/root/in,readonly \
  --mount type=bind,source="$(pwd)"/tests/results,target=/root/out \
  crbm/biosimulations_copasi:latest \
    -i /root/in/BIOMD0000000297.omex \
    -o /root/out
```

## License
This package is released under the [MIT license](LICENSE).

## Development team
This package was developed by the [Center for Reproducible Biomedical Modeling](http://reproduciblebiomodels.org).

## Questions and comments
Please contact the [Center for Reproducible Biomedical Modeling](mailto:info@reproduciblebiomodels.org) with any questions or comments.
  