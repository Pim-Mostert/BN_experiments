import mlflow
import torch
import torchvision
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence, IEvidenceBatches
from bayesian_network.optimizers.common import OptimizerLogger
from torch.utils.data import DataLoader
from torchvision import transforms


class MnistEvidenceBatches(IEvidenceBatches):
    def __init__(
        self,
        torch_settings: TorchSettings,
        batch_size,
        gamma: float,
    ):
        self.torch_settings = torch_settings
        self.gamma = gamma
        self.batch_size = batch_size

        mnist = torchvision.datasets.MNIST(
            "./experiments/mnist",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        self._data_loader = DataLoader(
            dataset=mnist,
            batch_size=batch_size,
            shuffle=True,
        )

        self.mnist_dimensions = mnist.data.shape[1:3]

    def next(self) -> Evidence:
        _, (data, _) = next(enumerate(self._data_loader))

        height, width = data.shape[2:4]
        num_features = height * width
        num_observations = data.shape[0]

        # Morph into evidence structure
        data = data.reshape([num_observations, num_features])

        evidence = Evidence(
            [torch.stack([1 - x, x]).T for x in data.T * (1 - self.gamma) + self.gamma / 2],
            self.torch_settings,
        )

        return evidence


class MLflowOptimizerLogger(OptimizerLogger):
    def log_iteration(self, iteration: int, ll: float):
        super().log_iteration(iteration, ll)

        log = self._logs[iteration]

        mlflow.log_metric(
            key="ll",
            value=log.ll,
            step=iteration,
            timestamp=int(log.ts.timestamp()),
        )
