from azure.ai.ml.dsl import pipeline

from azure_ml_component import aml_component

@pipeline()
def aml_pipeline(torch_device: str = "cuda"):
    node = aml_component(torch_device=torch_device)

    return {
        "output_folder": node.outputs.output_folder
    }