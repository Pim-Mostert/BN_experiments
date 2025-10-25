from pathlib import Path
from airflow.models.param import Param
from dags_common.experiment import create_batch_dag

experiment_id = "006-Logistic_regression_evaluator"
notebook_path = Path(__file__).parents[0] / "001-Logistic_regression_evaluator.py"

dag = create_batch_dag(
    experiment_id,
    notebook_path,
    batch_params={
        "NUM_CLASSES": Param(
            default=[10],
            type="array",
            items={"type": "integer"},
        ),
        "NUM_FEATURES": Param(
            default=[5],
            type="array",
            items={"type": "integer"},
        ),
        "Y_FEATURE_NODES": Param(
            default=[True, False],
            type="array",
            items={"type": "boolean"},
        ),
        "F_FEATURE_NODES": Param(
            default=[True, False],
            type="array",
            items={"type": "boolean"},
        ),
        "Q_FEATURE_NODES": Param(
            default=[True, False],
            type="array",
            items={"type": "boolean"},
        ),
    },
)
