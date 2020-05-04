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

# Install requirements
RUN pip install -U pip \
    && pip install -U setuptools

# Copy code for command-line interface into image and install it
COPY . /root/Biosimulations_copasi
RUN pip install /root/Biosimulations_copasi

# Entrypoint
ENTRYPOINT ["copasi"]
CMD []
