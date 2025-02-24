from pathlib import Path
from airflow.models.param import Param
from dags_common import create_experiment_dag

experiment_id = Path(__file__).parents[0].stem
notebook_path = Path(__file__).parents[0] / f"{experiment_id}.py"

dag = create_experiment_dag(
    experiment_id,
    notebook_path,
    experiment_params={
        "device": Param(
            default="cpu",
            type="string",
            enum=["cpu", "cuda", "mps"],
            description="PyTorch device",
        ),
        "selected_num_observations": Param(1000, type="integer"),
        "num_iterations": Param(10, type="integer"),
        "gamma": Param(0.00001, type="number"),
    },
)
