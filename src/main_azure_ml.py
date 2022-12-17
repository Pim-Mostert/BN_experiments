from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure_ml_pipeline import aml_pipeline

credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="cfadadd6-c606-42f4-a9a8-49fe686dfacf",
    resource_group_name="rg-azure-ml",
    workspace_name="pim_azure_ml",
)

# Create job
pipeline_job = aml_pipeline("cpu");

# Submit the job
submitted_job = ml_client.jobs.create_or_update(
    job=pipeline_job,
    compute="gpu-cluster3")

