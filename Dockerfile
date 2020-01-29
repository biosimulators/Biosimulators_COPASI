FROM ubuntu:latest


RUN apt update && apt install python3 python3-pip python3-dev -y
# RUN apt update && apt add --no-cache python3-dev
WORKDIR /usr/local/app/copasi/
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
ADD src/ ./

ENTRYPOINT [ "python3",  "src/copasi/copasi_simulator.py"]