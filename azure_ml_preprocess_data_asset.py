from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from app_config import config
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

credential = DefaultAzureCredential()

# Get a handle to the workspace
azure_ml_config = config.azure_ml

ml_client = MLClient(
    credential=credential,
    subscription_id=azure_ml_config.subscription_id,
    resource_group_name=azure_ml_config.resource_group_name,
    workspace_name=azure_ml_config.workspace_name,
)

# Supported paths include:
# local: './<path>/<file>'
# blob:  'https://<account_name>.blob.core.windows.net/<container_name>/<path>/<file>'
# ADLS gen2: 'abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>/<file>'
# Datastore: 'azureml://datastores/<data_store_name>/paths/<path>/<file>'
my_path = 'azureml://datastores/workspaceblobstore/paths/azureml/27b6683f-d20d-47ab-b64e-38799547035e_output_data_output_file:1'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FILE,
    description="MNIST dataset morphed into evidence structure",
    name="Evidence_MNIST",
    version="1"
)

ml_client.data.create_or_update(my_data)