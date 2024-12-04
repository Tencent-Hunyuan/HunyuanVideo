FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir torch==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install flash_attn
RUN pip3 install --no-cache-dir flash-attn
RUN python3 -m pip install ninja

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
