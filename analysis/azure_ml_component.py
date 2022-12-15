from analysis import analysis
from mldesigner import command_component, Output
import pickle

@command_component(
    environment="azureml:pim:4"
)
def aml_component(
    output_file: Output(type="uri_file")
):
    result = analysis()

    with open(output_file, 'wb') as file:
        pickle.dump(result, file)

