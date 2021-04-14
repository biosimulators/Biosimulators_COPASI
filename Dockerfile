# Build image:
#   docker build --tag ghcr.io/biosimulators/biosimulators_copasi/copasi:4.30.240 --tag ghcr.io/biosimulators/biosimulators_copasi/copasi:latest .
#
# Run image:
#   docker run \
#     --tty \
#     --rm \
#     --mount type=bind,source="$(pwd)"/tests/fixtures,target=/root/in,readonly \
#     --mount type=bind,source="$(pwd)"/tests/results,target=/root/out \
#     ghcr.io/biosimulators/copasi:latest \
#       -i /root/in/BIOMD0000000297.omex \
#       -o /root/out

# Base OS
FROM python:3.7.9-slim-buster

ARG VERSION="0.1.22"
ARG SIMULATOR_VERSION=4.30.240

# metadata
LABEL \
    org.opencontainers.image.title="COPASI" \
    org.opencontainers.image.version="${SIMULATOR_VERSION}" \
    org.opencontainers.image.description="Open-source software package for the simulation and analysis of biochemical networks and their dynamics." \
    org.opencontainers.image.url="http://copasi.org/" \
    org.opencontainers.image.documentation="http://copasi.org/Support/User_Manual/" \
    org.opencontainers.image.source="https://github.com/biosimulators/Biosimulators_COPASI" \
    org.opencontainers.image.authors="BioSimulators Team <info@biosimulators.org>" \
    org.opencontainers.image.vendor="BioSimulators Team" \
    org.opencontainers.image.licenses="Artistic-2.0" \
    \
    base_image="python:3.7.9-slim-buster" \
    version="${VERSION}" \
    software="COPASI" \
    software.version="${SIMULATOR_VERSION}" \
    about.summary="Open-source software package for the simulation and analysis of biochemical networks and their dynamics." \
    about.home="http://copasi.org/" \
    about.documentation="http://copasi.org/Support/User_Manual/" \
    about.license_file="https://github.com/copasi/COPASI/blob/develop/license.txt" \
    about.license="SPDX:Artistic-2.0" \
    about.tags="kinetic modeling,dynamical simulation,systems biology,biochemical networks,SBML,SED-ML,COMBINE,OMEX,XPP,Berkeley Madonna,BioSimulators" \
    maintainer="BioSimulators Team <info@biosimulators.org>"

# Copy code for command-line interface into image and install it
COPY . /root/Biosimulators_COPASI
RUN pip install /root/Biosimulators_COPASI \
    && rm -rf /root/Biosimulators_COPASI
RUN pip install "python_copasi==${SIMULATOR_VERSION}"
ENV ALGORITHM_SUBSTITUTION_POLICY=SIMILAR_VARIABLES \
    VERBOSE=0 \
    MPLBACKEND=PDF

# Entrypoint
ENTRYPOINT ["copasi"]
CMD []
