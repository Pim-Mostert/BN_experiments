import json
from mldesigner import command_component, Input, Output
import pickle
from azure_ml.analysis.preprocess_mnist import preprocess_mnist

@command_component(
    environment="azureml:pim:4"
)
def preprocess_mnist_component(
    preprocess_options_serialized: Input(type="string"),
    output_file: Output(type="uri_file") = None
):
    preprocess_options = json.loads(preprocess_options_serialized)
    
    evidence = preprocess_mnist(preprocess_options.gamma)
    
    with open(output_file, 'wb') as file:
        pickle.dump(evidence, file)
