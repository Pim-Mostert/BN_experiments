from pathlib import Path
from airflow.models.param import Param
from dags_common.experiment import create_batch_dag

experiment_id = "009_mnist_sequential"
notebook_path = Path(__file__).parents[0] / "009_mnist_sequential.py"

dag = create_batch_dag(
    experiment_id,
    notebook_path,
    batch_params={
        "BATCH_SIZE": Param(
            default=[1000],
            type="array",
            items={"type": "integer"},
        ),
        "LEARNING_RATE": Param(
            default=[0.1],
            type="array",
            items={"type": "number"},
        ),
        "REGULARIZATION": Param(
            default=[0.001],
            type="array",
            items={"type": "number"},
        ),
        "NUM_EPOCHS": Param(
            default=[10],
            type="array",
            items={"type": "integer"},
        ),
        "NUM_SUBSET": Param(
            default=[5000],
            type="array",
            items={"type": "integer"},
        ),
    },
)
