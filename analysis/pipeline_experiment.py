from azure.ai.ml.dsl import pipeline
from analysis.preprocess_mnist.component_preprocess_mnist import preprocess_mnist_component
from analysis.train.component_train import component_train

# Create pipeline
@pipeline()
def pipeline_experiment(
    preprocess_options_serialized,
):
    preprocess = preprocess_mnist_component(preprocess_options_serialized=preprocess_options_serialized)
    
    train = component_train(evidence_file=preprocess.outputs.output_file)

