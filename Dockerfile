# Build image:
#   docker build --tag crbm/biosimulations_copasi:4.27.214--tag crbm/biosimulations_copasi:latest .
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
FROM ubuntu:16.04

# Install requirements
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && pip3 install -U pip \
    && pip3 install -U setuptools \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy code for command-line interface into image and install it
COPY . /root/Biosimulations_copasi
RUN pip3 install /root/Biosimulations_copasi

# Entrypoint
ENTRYPOINT ["copasi"]
CMD []
