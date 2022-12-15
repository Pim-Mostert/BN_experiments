from azure.ai.ml.dsl import pipeline
from azure_ml.component_train import component_train
from azure.ai.ml import Input

# Create pipeline
@pipeline()
def pipeline_train():
    path = 'azureml:Evidence_MNIST:1'
    evidence_file = Input(type="uri_file", path=path)
    
    train = component_train(evidence_file=evidence_file)

