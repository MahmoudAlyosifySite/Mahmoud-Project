# Reproduces the paper's experiments on any machine with an NVIDIA GPU.
#   docker build -t spend-labels-last .
#   docker run --gpus all -v $PWD/results:/work/results spend-labels-last bash scripts/reproduce.sh
FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dlt dlt
COPY scripts scripts
COPY jobs jobs
COPY tests tests
ENV DLT_DATA=/work/data
