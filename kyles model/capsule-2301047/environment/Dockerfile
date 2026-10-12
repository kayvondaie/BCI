# hash:sha256:f928b63f4f34cdfd9aa00059f8c741571a8c68ebd441234c8f0d9cb427f23ff4
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.9.2 \
    numpy==2.0.2 \
    scikit-learn==1.5.2 \
    scipy==1.13.1 \
    statsmodels==0.14.4 \
    torch==2.5.1
