from azure.ai.ml.dsl import pipeline
from azure_ml.component_preprocess_mnist import preprocess_mnist_component

# Create pipeline
@pipeline(
    default_compute="gpu-cluster"
)
def pipeline_preprocess_mnist(gamma: float):
    # Preprocess
    preprocess = preprocess_mnist_component(gamma=gamma)
    
    return {
        "bayesian_network_output_file": preprocess.outputs.output_file
    }

