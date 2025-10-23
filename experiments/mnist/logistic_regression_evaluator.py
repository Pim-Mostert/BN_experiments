from dataclasses import dataclass
from functools import lru_cache
import tempfile

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import EvidenceLoader
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

        chunk = self._get_chunk(chunk_id)

        return chunk[sample_id]

    @classmethod
    def create(
        cls,
        evidence_loader: EvidenceLoader,
        inference_machine: IInferenceMachine,
        feature_nodes: List[Node],
        cache_path: Path,
        cache_maxsize: int,
    ):
        cache_path.mkdir(exist_ok=True)
        chunk_paths: dict[int, Path] = {}

        for i, evidence in enumerate(evidence_loader):
            inference_machine.enter_evidence(evidence)
            Z = inference_machine.infer_single_nodes(feature_nodes)
            Z = torch.stack(Z).permute([1, 0, 2]).flatten(start_dim=1, end_dim=2)

            chunk_paths[i] = cache_path / f"chunk_{i}.pt"
            torch.save(Z, chunk_paths[i].as_posix())

            logging.info("Finished batch %s/%s", i, len(evidence_loader))

        if not evidence_loader._data_loader.batch_size:
            raise Exception()

        return cls(
            num_samples=evidence_loader.num_observations,
            chunk_size=evidence_loader._data_loader.batch_size,
            chunk_paths=chunk_paths,
            chunk_cache_maxsize=cache_maxsize,
        )


class MnistEnriched(Dataset):
    def __init__(self, mnist: Dataset, features: FeaturesDataset):
        if not len(mnist) == len(features):  # pyright: ignore[reportArgumentType]
            raise Exception()

        self._mnist = mnist
        self._features = features

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        x, target = self._mnist[idx]
        y = self._features[idx]

        return torch.concat([x, y]), target


@dataclass
class LogisticRegressionEvaluatorSettings:
    torch_settings: TorchSettings  # Ensure TorchSettings is a valid class and not self-referential
    # iterations_per_epoch: int
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
        evidence_loader_factory: Callable[[Dataset], EvidenceLoader],
        mnist_train,
        mnist_test,
    ):
        self._settings = settings
        self._inference_machine_factory = inference_machine_factory
        self._evidence_loader_factory = evidence_loader_factory
        self._mnist_train = mnist_train
        self._mnist_test = mnist_test

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        if self._settings.should_evaluate:
            if not self._settings.should_evaluate(epoch, iteration):
                return

        model, train_loss = self._train(network)

        logging.info("Finished trainig , loss: %s", train_loss)

        accuracy = self._classify(model, network)

        logging.info(
            "Evaluated for epoch %s, iteration: %s, accuracy: %s",
            epoch,
            iteration,
            accuracy,
        )

        # mlflow.log_metric(
        #     key="logreg_accuracy",
        #     step=epoch * self._settings.iterations_per_epoch + iteration,
        #     value=accuracy,
        # )

    def _train(self, network: BayesianNetwork):
        evidence_loader = self._evidence_loader_factory(self._mnist_train)

        with tempfile.TemporaryDirectory() as cache_dir:
            feature_dataset = FeaturesDataset.create(
                evidence_loader=evidence_loader,
                cache_path=Path(cache_dir),
                cache_maxsize=2,
                feature_nodes=self._settings.feature_nodes,
                inference_machine=self._inference_machine_factory(network),
            )

            mnist_enriched = MnistEnriched(self._mnist_train, feature_dataset)
            x, _ = mnist_enriched[0]

            model = LogisticRegressionModel(
                input_dim=len(x),
                num_classes=self._settings.num_classes,
                torch_settings=self._settings.torch_settings,
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=self._settings.learning_rate)

            train_loader = DataLoader(
                dataset=mnist_enriched,
                batch_size=self._settings.train_batch_size,
                shuffle=True,
            )

            # %% Training loop
            for epoch in range(self._settings.epochs):
                total_loss = 0
                model.train()
                for i, (data, labels) in enumerate(train_loader):
                    labels = labels.to(self._settings.torch_settings.device)

                    # Train logistic regression model
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    # logging.info(
                    #     "Finished trainig epoch %s, batch %s",
                    #     epoch,
                    #     i,
                    # )

                logging.info(
                    "Finished trainig epoch %s/%s, loss: %s",
                    epoch,
                    self._settings.epochs,
                    total_loss,
                )

        return model, total_loss  # pyright: ignore[reportPossiblyUnboundVariable]

    def _classify(self, model: LogisticRegressionModel, network: BayesianNetwork):
        evidence_loader = self._evidence_loader_factory(self._mnist_test)

        with tempfile.TemporaryDirectory() as cache_dir:
            feature_dataset = FeaturesDataset.create(
                evidence_loader=evidence_loader,
                cache_path=Path(cache_dir),
                cache_maxsize=2,
                feature_nodes=self._settings.feature_nodes,
                inference_machine=self._inference_machine_factory(network),
            )

            mnist_enriched = MnistEnriched(self._mnist_test, feature_dataset)

            test_loader = DataLoader(
                dataset=mnist_enriched,
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


class CompositeEvaluator(IEvaluator):
    def __init__(self, evaluators: List[IEvaluator]):
        self._evaluators = evaluators

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        for evaluator in self._evaluators:
            evaluator.evaluate(epoch, iteration, network)

    @property
    def log_likelihoods(self) -> Dict[Tuple[int, int], float]:
        raise NotImplementedError
