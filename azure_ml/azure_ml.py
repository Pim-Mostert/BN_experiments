from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.dsl import pipeline
from mldesigner import command_component, Input, Output

credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="cfadadd6-c606-42f4-a9a8-49fe686dfacf",
    resource_group_name="rg-azure-ml",
    workspace_name="pim_azure_ml",
)

# Create command
conda_env = dict(
    # note that mldesigner package must be included.
    conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
)

@command_component(
    environment=conda_env
)
def step_component(
    name: Input(type="string"),
    output_file: Input(type="string"),
    output_dir: Output(type="uri_folder"),
):
    print(f'Hoi {name}')
    
    with open(f'{output_dir}/{output_file}', "w") as file:
        file.write(f'Groetjes van {name}')
 
# Create pipeline
@pipeline(default_compute="cpu-cluster")
def step_pipeline(name):
    node = step_component(name=name, output_file="groetjes.txt")

# Create job
pipeline_job = step_pipeline("Pim")

# Submit the job
submitted_job = ml_client.jobs.create_or_update(pipeline_job)
