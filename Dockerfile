FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget git nano

RUN conda --version

WORKDIR /root

RUN conda install tqdm -f
RUN conda update conda
RUN conda install pip
RUN pip install bvhsdk
RUN conda --version
RUN conda install numpy
RUN conda install scipy scikit-learn
RUN conda install notebook=6.5
RUN conda install matplotlib=3.1