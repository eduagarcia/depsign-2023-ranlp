FROM nvcr.io/nvidia/pytorch:22.03-py3

ENV HF_DATASETS_CACHE="/workspace/cache/huggingface/datasets"
ENV HUGGINGFACE_HUB_CACHE="/workspace/cache/huggingface/hub"
ENV WANDB_CACHE_DIR="/workspace/cache/wandb"

ENV HOME=/workspace
RUN echo $PWD
ADD . $HOME
WORKDIR $HOME

RUN apt update
RUN pip install -r requirements.txt