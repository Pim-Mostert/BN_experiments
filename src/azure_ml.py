from io import BytesIO
import os
import pickle
import shutil
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import torch
from azure_ml_pipeline import aml_pipeline

credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="cfadadd6-c606-42f4-a9a8-49fe686dfacf",
    resource_group_name="rg-azure-ml",
    workspace_name="pim_azure_ml",
)

def submit_job():
    # Create job
    pipeline_job = aml_pipeline();

    # Submit the job
    submitted_job = ml_client.jobs.create_or_update(
        job=pipeline_job,
        compute="gpu-cluster3")
    
    job_name = submitted_job.name
    
    print(f'Submitted job: {job_name}')
    
    return job_name

def get_output(job_name: str):
    download_path = "azure_ml_downloads"
    shutil.rmtree(download_path, ignore_errors=True)
    
    file_path = os.path.join(download_path, "named-outputs/output_folder/output.p")
    
    ml_client.jobs.download(name=job_name, download_path=download_path, all=True)

    with open(file_path, "rb") as file:
        output = CPU_Unpickler(file).load()
        
    return output
    
#https://github.com/pytorch/pytorch/issues/16797
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location='cpu')
        else: 
            return super().find_class(module, name)
    