from dataclasses import asdict
from pathlib import Path
from airflow.models.param import Param
from dags_common import create_experiment_dag
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.optimizers.em_batch_optimizer import EmBatchOptimizerSettings

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
            default=50,
            type="integer",
        ),
        "EM_BATCH_OPTIMIZER_SETTINGS": Param(
            default=asdict(EmBatchOptimizerSettings(num_iterations=100, learning_rate=0.01)),
            type="object",
        ),
        "GAMMA": Param(
            default=0.001,
            type="number",
            description="Normalization factor of MNIST data",
        ),
    },
)
