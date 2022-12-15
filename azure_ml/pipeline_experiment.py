from azure.ai.ml.dsl import pipeline
from azure_ml.component_preprocess_mnist import preprocess_mnist_component
from azure_ml.component_train import component_train

# Create pipeline
@pipeline(
    default_compute="gpu-cluster"
)
def pipeline_experiment(gamma: float):
    preprocess = preprocess_mnist_component(gamma=gamma)
    
    train = component_train(evidence_file=preprocess.outputs.output_file)

