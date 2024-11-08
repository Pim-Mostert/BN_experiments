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
        asyncio.to_thread(
            experiment(torch_settings, selected_num_observations=1000)
        ),
        asyncio.to_thread(
            experiment(torch_settings, selected_num_observations=2000)
        ),
        asyncio.to_thread(
            experiment(torch_settings, selected_num_observations=3000)
        ),
        asyncio.to_thread(
            experiment(torch_settings, selected_num_observations=4000)
        ),
        asyncio.to_thread(
            experiment(torch_settings, selected_num_observations=5000)
        ),
        asyncio.to_thread(
            experiment(torch_settings, selected_num_observations=6000)
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
