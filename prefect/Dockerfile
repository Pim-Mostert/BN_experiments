FROM prefecthq/prefect:3.0.11-python3.10

ENV PIP_INDEX_URL="https://<redacted>@pkgs.dev.azure.com/mostertpim/BayesianNetwork/_packaging/BayesianNetwork/pypi/simple/"
ENV PREFECT_API_KEY="<redacted>"
ENV PREFECT_API_URL="https://api.prefect.cloud/api/accounts/be1d6ab5-7fa2-40b6-aaff-0feaee45abfc/workspaces/ec73bdf7-382c-4918-81ce-ea8909841322"
ENV PREFECT_HOME="/prefect"


COPY requirements.txt /opt/prefect/mnist/requirements.txt
RUN python -m pip install -r /opt/prefect/mnist/requirements.txt
COPY serve.py /opt/prefect/mnist/
WORKDIR /opt/prefect/mnist/

ENTRYPOINT ["python", "serve.py"]
