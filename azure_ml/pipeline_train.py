from azure.ai.ml.dsl import pipeline
from azure_ml.component_preprocess_mnist import preprocess_mnist_component
from azure_ml.component_train import component_train

# Create pipeline
@pipeline(
    default_compute="gpu-cluster"
)
def pipeline_train(evidence_file: str):
    train = component_train(evidence_file=evidence_file)
    
    return {
        "bayesian_network_output_file": train.outputs.output_file
    }

