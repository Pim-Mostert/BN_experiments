from pathlib import Path
from airflow.models.param import Param
from dags_common import create_experiment_dag

experiment_id = Path(__file__).parents[0].stem
notebook_path = Path(__file__).parents[0] / f"{experiment_id}.py"

dag = create_experiment_dag(
    experiment_id,
    notebook_path,
    experiment_params={
        "DEVICE": Param(
            default="cpu",
            type="string",
            enum=["cpu", "cuda", "mps"],
            description="PyTorch device",
        ),
        "DTYPE": Param(
            default="float64",
            type="string",
            enum=["float64", "float32"],
            description="PyTorch dtype",
        ),
        "NUM_ITERATIONS": Param(
            default=100,
            type="integer",
            description="Number of iterations for EM Batch Optimizer",
        ),
        "LEARNING_RATE": Param(
            default=0.01,
            type="number",
            description="Learning rate for EM Batch Optimizer",
        ),
        "BATCH_SIZE": Param(
            default=50,
            type="integer",
            description="Batch size for EM Batch Optimizer",
        ),
        "SELECTED_NUM_OBSERVATIONS": Param(
            default=2000,
            type="integer",
            description="Select subset of MNIST dataset",
        ),
        "GAMMA": Param(
            default=0.001,
            type="number",
            description="Normalization factor of MNIST data",
        ),
    },
)
