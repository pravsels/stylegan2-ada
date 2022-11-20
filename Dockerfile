
ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.10-tf1-py3
FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive

# install other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-pip  \
    python-tk \
    nano && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]
