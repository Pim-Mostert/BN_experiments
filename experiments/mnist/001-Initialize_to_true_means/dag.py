from pathlib import Path
from airflow.models.param import Param
from dags_common import create_experiment_dag
from bayesian_network.common.torch_settings import TorchSettings

experiment_id = Path(__file__).parents[0].stem
notebook_path = Path(__file__).parents[0] / f"{experiment_id}.py"

dag = create_experiment_dag(
    experiment_id,
    notebook_path,
    experiment_params={
        "TORCH_SETTINGS": Param(
            default=dict(TorchSettings(device="cpu", dtype="float64")),
            type="object",
        ),
        "BATCH_SIZE": Param(
            default=[100, 1000],
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
            default=200,
            type="integer",
        ),
        "GAMMA": Param(
            default=0.001,
            type="number",
            description="Normalization factor of MNIST data",
        ),
    },
)
