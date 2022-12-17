from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from analysis.azure_ml_pipeline import aml_pipeline

from app_config import config

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
pipeline_job = aml_pipeline();

# Submit the job
submitted_job = ml_client.jobs.create_or_update(
    job=pipeline_job,
    compute="gpu-cluster3")

pass