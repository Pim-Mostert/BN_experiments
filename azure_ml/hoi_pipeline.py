from azure.ai.ml.dsl import pipeline
from azure_ml.hoi_component import hoi_component

# Create pipeline
@pipeline(default_compute="cpu-cluster")
def hoi_pipeline(name):
    node = hoi_component(name=name, output_file="groetjes.txt")

