from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

from app_config import config
from src.azure_ml_component import aml_pipeline

credential = DefaultAzureCredential()

# Get a handle to the workspace
azure_ml_config = config.azure_ml

ml_client = MLClient(
    credential=credential,
    subscription_id=azure_ml_config.subscription_id,
    resource_group_name=azure_ml_config.resource_group_name,
    workspace_name=azure_ml_config.workspace_name,
)

env_docker_image = Environment(
    image="bd6edd4ab1f64cc6b843dd398eba3c02.azurecr.io/pim:4",
    name="pim",
    description="Environment created for Pim",
)
ml_client.environments.create_or_update(env_docker_image)

