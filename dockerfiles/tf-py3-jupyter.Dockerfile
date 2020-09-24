ARG DOCKER_ENV=cpu

FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG DOCKER_ENV

ADD . /develop

# Needed for string testing
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y git nano graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install tensorflow_datasets && \
    pip3 install seaborn eli5 shap pydot pdpbox sklearn opencv-python IPython prettytable gdown==3.10.0 numpy Pillow && \
    pip3 uninstall -y tensorflow && \
    pip3 uninstall -y tensorflow-gpu && \
    pip3 install tensorflow-gpu  && \
    pip3 install --upgrade tensorflow-probability  && \
    pip3 install -q -U tb-nightly && \
    pip3 install --upgrade tensorflow-hub

WORKDIR /develop
