from dataclasses import dataclass
from functools import lru_cache
import tempfile

import mlflow

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.optimizers.common import IEvaluator


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple


class FeaturesDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        chunk_size: int,
        chunk_paths: Dict[int, Path],
        chunk_cache_maxsize: int,
    ):
        self._num_samples = num_samples
        self._chunk_size = chunk_size
        self._chunk_paths = chunk_paths
        self._chunk_cache: dict[int, torch.Tensor] = {}
        self._get_chunk = lru_cache(maxsize=chunk_cache_maxsize)(self._load_chunk)

    def __len__(self):
        return self._num_samples

    def _load_chunk(self, chunk_id: int) -> torch.Tensor:
        chunk = torch.load(self._chunk_paths[chunk_id])

        self._chunk_cache[chunk_id] = chunk

        return chunk

    def __getitem__(self, idx: int):
        if idx >= self._num_samples:
            raise ValueError(f"idx should be smaller than num_samples {self._num_samples}")

        chunk_id = idx // self._chunk_size
        sample_id = idx % self._chunk_size

        data, labels = self._get_chunk(chunk_id)

        return data[sample_id], labels[sample_id]

    @classmethod
    def create(
        cls,
        data_loader: DataLoader,
        transform: Callable[[torch.Tensor], Evidence],
        inference_machine: IInferenceMachine,
        feature_nodes: List[Node],
        cache_path: Path,
        cache_maxsize: int,
    ):
        cache_path.mkdir(exist_ok=True)
        chunk_paths: dict[int, Path] = {}

        num_samples = 0
        for i, (data, labels) in enumerate(data_loader):
            evidence = transform(data)
            inference_machine.enter_evidence(evidence)
            Z = inference_machine.infer_single_nodes(feature_nodes)
            Z = torch.cat(Z, dim=1)

            chunk_paths[i] = cache_path / f"chunk_{i}.pt"
            torch.save((Z, labels), chunk_paths[i].as_posix())

            num_samples += len(data)

            logging.info("Finished batch %s/%s", i, len(data_loader))

        if not data_loader.batch_size:
            raise Exception()

        return cls(
            num_samples=num_samples,
            chunk_size=data_loader.batch_size,
            chunk_paths=chunk_paths,
            chunk_cache_maxsize=cache_maxsize,
        )


@dataclass
class LogisticRegressionEvaluatorSettings:
    torch_settings: TorchSettings
    feature_nodes: List[Node]
    learning_rate: float
    epochs: int
    train_batch_size: int
    test_batch_size: int
    num_classes: int
    should_evaluate: Callable[[int, int], bool] | None = None


class LogisticRegressionModel(nn.Module):
    def __init__(self, torch_settings: TorchSettings, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(
            input_dim,
            num_classes,
            dtype=torch_settings.dtype,
            device=torch_settings.device,
        )

    def forward(self, x):
        return self.linear(x)


class LogisticRegressionEvaluator(IEvaluator):
    def __init__(
        self,
        settings: LogisticRegressionEvaluatorSettings,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        transform: Callable[[torch.Tensor], Evidence],
        mnist_train_loader,
        mnist_test_loader,
    ):
        self._settings = settings
        self._inference_machine_factory = inference_machine_factory
        self._transform = transform
        self._mnist_train_loader = mnist_train_loader
        self._mnist_test_loader = mnist_test_loader

        self._accuracies: dict[tuple[int, int], float] = {}

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        if self._settings.should_evaluate:
            if not self._settings.should_evaluate(epoch, iteration):
                return

        model, train_loss = self._train(network)

        logging.info("Finished training , loss: %s", train_loss)

        accuracy = self._classify(model, network)
        self._accuracies[(epoch, iteration)] = accuracy

        logging.info(
            "Evaluated for epoch %s, iteration: %s, accuracy: %s",
            epoch,
            iteration,
            accuracy,
        )

    def _train(self, network: BayesianNetwork):
        with tempfile.TemporaryDirectory() as cache_dir:
            feature_dataset = FeaturesDataset.create(
                data_loader=self._mnist_train_loader,
                transform=self._transform,
                cache_path=Path(cache_dir),
                cache_maxsize=2,
                feature_nodes=self._settings.feature_nodes,
                inference_machine=self._inference_machine_factory(network),
            )

            x, _ = feature_dataset[0]

            model = LogisticRegressionModel(
                input_dim=len(x),
                num_classes=self._settings.num_classes,
                torch_settings=self._settings.torch_settings,
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=self._settings.learning_rate)

            train_loader = DataLoader(
                dataset=feature_dataset,
                batch_size=self._settings.train_batch_size,
                shuffle=False,
            )

            # %% Training loop
            total_loss = 0
            for epoch in range(self._settings.epochs):
                total_loss = 0
                model.train()
                for data, labels in train_loader:
                    labels = labels.to(self._settings.torch_settings.device)

                    # Train logistic regression model
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                logging.info(
                    "Finished training epoch %s/%s, loss: %s",
                    epoch,
                    self._settings.epochs,
                    total_loss,
                )

        return model, total_loss

    def _classify(self, model: LogisticRegressionModel, network: BayesianNetwork):
        with tempfile.TemporaryDirectory() as cache_dir:
            feature_dataset = FeaturesDataset.create(
                data_loader=self._mnist_test_loader,
                transform=self._transform,
                cache_path=Path(cache_dir),
                cache_maxsize=2,
                feature_nodes=self._settings.feature_nodes,
                inference_machine=self._inference_machine_factory(network),
            )

            test_loader = DataLoader(
                dataset=feature_dataset,
                batch_size=self._settings.test_batch_size,
            )

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    labels = labels.to(self._settings.torch_settings.device)

                    # Predictions
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

        return accuracy

    @property
    def log_likelihoods(self) -> Dict[Tuple[int, int], float]:
        raise NotImplementedError

    @property
    def accuracies(self):
        return self._accuracies


class MLflowLogisticRegressionEvaluator(LogisticRegressionEvaluator):
    def __init__(
        self,
        settings: LogisticRegressionEvaluatorSettings,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        transform: Callable[[torch.Tensor], Evidence],
        mnist_train_loader,
        mnist_test_loader,
        iterations_per_epoch,
    ):
        super().__init__(
            settings, inference_machine_factory, transform, mnist_train_loader, mnist_test_loader
        )

        self._iterations_per_epoch = iterations_per_epoch

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        if self._settings.should_evaluate:
            if not self._settings.should_evaluate(epoch, iteration):
                return

        super().evaluate(epoch, iteration, network)

        accuracy = self.accuracies[(epoch, iteration)]

        mlflow.log_metric(
            key="logreg_accuracy",
            step=epoch * self._iterations_per_epoch + iteration,
            value=accuracy,
        )


class CompositeEvaluator(IEvaluator):
    def __init__(self, evaluators: List[IEvaluator]):
        self._evaluators = evaluators

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        for evaluator in self._evaluators:
            evaluator.evaluate(epoch, iteration, network)

    @property
    def log_likelihoods(self) -> Dict[Tuple[int, int], float]:
        raise NotImplementedError
