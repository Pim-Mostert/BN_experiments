from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from app_config import config
from azure_ml.pipeline_experiment import pipeline_experiment

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
pipeline_job = pipeline_experiment(0.000001)

# Submit the job
submitted_job = ml_client.jobs.create_or_update(pipeline_job)

pass