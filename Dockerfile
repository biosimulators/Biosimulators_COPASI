# Build image:
#   docker build --tag crbm/biosimulations_copasi:4.27.214 --tag crbm/biosimulations_copasi:latest .
#
# Run image:
#   docker run \
#     --tty \
#     --rm \
#     --mount type=bind,source="$(pwd)"/tests/fixtures,target=/root/in,readonly \
#     --mount type=bind,source="$(pwd)"/tests/results,target=/root/out \
#     crbm/biosimulations_copasi:latest \
#       -i /root/in/BIOMD0000000297.omex \
#       -o /root/out

# Base OS
FROM python:3.7

# metadata
LABEL base_image="python:3.7"
LABEL version="4.28.226"
LABEL software="COPASI"
LABEL software.version="4.28.226"
LABEL about.summary="Open-source software package for the simulation and analysis of biochemical networks and their dynamics."
LABEL about.home="http://copasi.org/"
LABEL about.documentation="http://copasi.org/"
LABEL about.license_file="https://github.com/copasi/COPASI/blob/develop/license.txt"
LABEL about.license="SPDX:Artistic License 2.0"
LABEL about.tags="kinetic modeling,dynamical simulation,systems biology,BNGL,SED-ML,COMBINE,OMEX,XPP,Berkeley Madonna"
LABEL maintainer="Gnaneswara Marupilla <marupilla@uchc.edu>"

# Install requirements
RUN pip install -U pip \
    && pip install -U setuptools

# Copy code for command-line interface into image and install it
COPY . /root/Biosimulations_copasi
RUN pip install /root/Biosimulations_copasi

# Entrypoint
ENTRYPOINT ["copasi"]
CMD []
