FROM nvcr.io/nvidia/pytorch:21.03-py3
ARG PYTHON_VERSION=3.8

RUN apt-get update
RUN apt-get install htop -y
RUN apt-get install screen -y

RUN pip install \
seaborn==0.11.1 \
pyquaternion==0.9.9 \
orjson==3.5.1 \
ncls==0.0.57 \
dill==0.3.3 \
pathos==0.2.7 \
nuscenes-devkit==1.1.2 \
pykalman==0.9.5\
python-louvain\
pip install networkx[default]

RUN pip uninstall tensorboard nvidia-tensorboard -y
RUN pip install tensorboard

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
RUN apt-get install -y python3-opencv

WORKDIR /workspace/ScePT-plus-plus
