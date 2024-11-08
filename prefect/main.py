import asyncio
from prefect import flow
from prefect.task_runners import ThreadPoolTaskRunner
from bayesian_network.common.torch_settings import TorchSettings

import torch

from analyses.mnist.experiment import experiment


@flow(task_runner=ThreadPoolTaskRunner(max_workers=10))
async def main():
    torch_settings = TorchSettings(torch.device("cpu"), torch.float64)

    await asyncio.gather(
        experiment(torch_settings, selected_num_observations=1000),
        experiment(torch_settings, selected_num_observations=2000),
        experiment(torch_settings, selected_num_observations=3000),
        experiment(torch_settings, selected_num_observations=4000),
        experiment(torch_settings, selected_num_observations=5000),
        experiment(torch_settings, selected_num_observations=6000),
    )
