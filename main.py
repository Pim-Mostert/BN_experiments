import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from app_config import config
from analysis.pipeline_experiment import pipeline_experiment

credential = DefaultAzureCredential()

# Get a handle to the workspace
azure_ml_config = config.azure_ml

ml_client = MLClient(
    credential=credential,
    subscription_id=azure_ml_config.subscription_id,
    resource_group_name=azure_ml_config.resource_group_name,
    workspace_name=azure_ml_config.workspace_name,
)

# Create job
preprocess_options = {
    "gamma": 0.000001,
}

pipeline_job = pipeline_experiment(
    preprocess_options_serialized=json.dumps(preprocess_options)
);

# Submit the job
submitted_job = ml_client.jobs.create_or_update(
    job=pipeline_job,
    description="hoi",
    compute="gpu-cluster")

pass