from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from app_config import config
from azure_ml.hoi_pipeline import hoi_pipeline

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
pipeline_job = hoi_pipeline("pim")

# Submit the job
submitted_job = ml_client.jobs.create_or_update(pipeline_job)

pass