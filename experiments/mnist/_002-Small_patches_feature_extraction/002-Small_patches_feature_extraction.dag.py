from pathlib import Path
from airflow.models.param import Param
from dags_common.experiment import create_batch_dag
from bayesian_network.common.torch_settings import TorchSettings

experiment_id = "002-Small_patches_feature_extraction"
notebook_path = Path(__file__).parents[0] / "002-Small_patches_feature_extraction.py"

dag = create_batch_dag(
    experiment_id,
    notebook_path,
    batch_params={
        "TORCH_SETTINGS": Param(
            default=[dict(TorchSettings(device="cpu", dtype="float64"))],
            type="array",
        ),
        "NUM_CLASSES": Param(
            default=[10],
            type="array",
            items={"type": "integer"},
        ),
    },
)
