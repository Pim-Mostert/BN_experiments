from azure.ai.ml.dsl import pipeline

from analysis.azure_ml_component import aml_component

@pipeline()
def aml_pipeline():
    node = aml_component()
