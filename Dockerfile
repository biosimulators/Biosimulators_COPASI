FROM ubuntu:latest


RUN apt update && apt install python3 python3-pip python3-dev -y
# RUN apt update && apt add --no-cache python3-dev

# WORKDIR command is not persisted when converted to Singularity image, add --pwd in Singularity while running
WORKDIR /usr/local/app/copasi/
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
ADD src/ ./src

ENTRYPOINT [ "python3",  "/usr/local/app/copasi/src/run.py"]