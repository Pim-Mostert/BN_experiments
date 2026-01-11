from pathlib import Path
from airflow.models.param import Param
from dags_common.experiment import create_batch_dag

experiment_id = "005-Hierarchical_full_data"
notebook_path = Path(__file__).parents[0] / "001-Hierarchical_full_data.py"

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
    },
)
