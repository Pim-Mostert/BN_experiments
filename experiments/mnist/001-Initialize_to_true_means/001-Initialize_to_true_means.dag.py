from pathlib import Path
from airflow.models.param import Param
from dags_common.experiment import create_batch_dag
from bayesian_network.common.torch_settings import TorchSettings

experiment_id = "001-Initialize_to_true_means"
notebook_path = Path(__file__).parents[0] / "001-Initialize_to_true_means.py"

dag = create_batch_dag(
    experiment_id,
    notebook_path,
    batch_params={
        "TORCH_SETTINGS": Param(
            default=[dict(TorchSettings(device="cpu", dtype="float64"))],
            type="array",
        ),
        "BATCH_SIZE": Param(
            default=[100],
            type="array",
            items={"type": "integer"},
        ),
        "LEARNING_RATE": Param(
            default=[0.01, 0.1],
            type="array",
            items={"type": "number"},
        ),
        "TRUE_MEANS_NOISE": Param(
            default=[0.1, 0.5],
            type="array",
            items={"type": "number"},
        ),
        "NUM_ITERATIONS": Param(
            default=[200],
            type="array",
            items={"type": "integer"},
        ),
        "GAMMA": Param(
            default=[0.001],
            type="array",
            items={"type": "number"},
            description="Normalization factor of MNIST data",
        ),
    },
)
