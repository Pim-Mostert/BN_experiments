from azure.ai.ml.dsl import pipeline

from azure_ml_component import aml_component

@pipeline()
def aml_pipeline():
    node = aml_component()
