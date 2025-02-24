# syntax=docker/dockerfile:1

### base
FROM apache/airflow:2.10.4-python3.12 AS base


### builder
FROM base AS builder

ARG AZURE_ARTIFACTS_TOKEN

ENV PIP_EXTRA_INDEX_URL=https://${AZURE_ARTIFACTS_TOKEN}@pkgs.dev.azure.com/mostertpim/BayesianNetwork/_packaging/BayesianNetwork/pypi/simple/

WORKDIR /pip

COPY requirements.txt ./requirements.txt

# Fetch the Airflow packages already installed in the base image
RUN pip freeze | grep apache-airflow > airflow-constraints.txt

# Install dependencies, but don't change Airflow packages
# RUN --mount=type=cache,target=/home/airflow/.cache/pip \
RUN pip install \
    -c airflow-constraints.txt \
    -r ./requirements.txt


### runner
FROM base

# Copy Python environment from builder
COPY --from=builder /home/airflow/.local /home/airflow/.local

