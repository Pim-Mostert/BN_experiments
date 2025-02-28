from pathlib import Path

from dags_common import create_experiment_dag
from airflow.models.param import Param

experiment_id = Path(__file__).parents[0].stem
notebook_path = Path(__file__).parents[0] / f"{experiment_id}.py"

dag = create_experiment_dag(
    experiment_id,
    notebook_path,
    experiment_params={
        "name": Param(
            default="Pim & Klaas",
            type="string",
            description="Your name.",
        ),
        "age": Param(
            default=35,
            type="integer",
            description="Your age.",
        ),
    },
)
