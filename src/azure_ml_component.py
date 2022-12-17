from analysis import analysis
from mldesigner import command_component, Input, Output
import pickle
import torch

@command_component(
    environment="azureml:pim:4"
)
def aml_component(
    torch_device: Input(type="string", default="cuda"),
    output_file: Output(type="uri_file")
):
    result = analysis(torch.device(torch_device))

    with open(output_file, 'wb') as file:
        pickle.dump(result, file)

