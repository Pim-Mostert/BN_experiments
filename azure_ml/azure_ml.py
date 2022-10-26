from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import dsl

credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="cfadadd6-c606-42f4-a9a8-49fe686dfacf",
    resource_group_name="rg-azure-ml",
    workspace_name="pim_azure_ml",
)

# Configuration
compute_target = "cpu-cluster"
job_environment = "AzureML-pytorch-1.9-ubuntu18.04-py37-cpu-inference:4"

# Create command
step_component = command(
    name="test",
    # display_name="Test met step.py",
    # description="reads a .xl input, split the input to train and test",
    inputs={
        "name": Input(type="string"),
        "output_file": Input(type="string"),
    },
    outputs={
        "output_dir": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code="./azure_ml",
    command="""python step.py --name ${{inputs.name}} --output_file ${{inputs.output_file}} --output_dir ${{outputs.output_dir}}""",
    environment=job_environment,
    compute=compute_target
)

# Create pipeline
@dsl.pipeline(
    default_compute=compute_target,
)
def step_pipeline(name):
    node = step_component(name=name, output_file="groetjes.txt")

# Create job
# step_job = step_component(name="Pim")
pipeline_job = step_pipeline("Pim")

# Submit the job
submitted_job = ml_client.jobs.create_or_update(pipeline_job)
