from azure.ai.ml.dsl import pipeline
from azure_ml.component_preprocess_mnist import preprocess_mnist_component

# Create pipeline
@pipeline(
    default_compute="cpu-cluster"
)
def pipeline_experiment(gamma: float):
    preprocess = preprocess_mnist_component(gamma=gamma)

