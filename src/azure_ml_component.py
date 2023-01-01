from analysis import analysis
from mldesigner import command_component, Input, Output
import pickle
import torch
import os

@command_component(
    environment="azureml:pim:4"
)
def aml_component(
    torch_device: Input(type="string", default="cuda"),
    output_folder: Output(type="uri_folder")
):
    result = analysis(torch.device(torch_device))

    output_file = os.path.join(output_folder, "output.p")
    
    with open(output_file , 'wb') as file:
        pickle.dump(result, file)

