from dataclasses import dataclass
from functools import lru_cache
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import Evidence, EvidenceLoader
from bayesian_network.optimizers.common import IEvaluator


import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple


@dataclass
class LogisticRegressionEvaluatorSettings:
    torch_settings: TorchSettings
    iterations_per_epoch: int
    feature_nodes: List[Node]
    learning_rate: float
    epochs: int
    height: int
    width: int
    num_classes: int
    should_evaluate: Callable[[int, int], bool] | None = None


class LogisticRegressionEvaluator(IEvaluator):
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

    class FeatureDataset(Dataset):
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
            print(f"Loaded file {self._chunk_paths[chunk_id]}")

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
            inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
            network: BayesianNetwork,
            feature_nodes: List[Node],
            cache_path: Path,
            cache_maxsize: int,
        ):
            cache_path.mkdir(exist_ok=True)
            chunk_paths: dict[int, Path] = {}

            inference_machine = inference_machine_factory(network)
            for i, evidence in enumerate(evidence_loader):
                inference_machine.enter_evidence(evidence)
                Z = inference_machine.infer_single_nodes(feature_nodes)
                Z = torch.stack(Z)[:, :, :-1].permute([1, 0, 2]).flatten(start_dim=1, end_dim=2)

                chunk_paths[i] = cache_path / f"chunk_{i}.pt"
                torch.save(Z, chunk_paths[i].as_posix())

                print(f"Finished batch {i}/{len(evidence_loader)}")

            if not evidence_loader._data_loader.batch_size:
                raise Exception()

            return LogisticRegressionEvaluator.FeatureDataset(
                num_samples=evidence_loader.num_observations,
                chunk_size=evidence_loader._data_loader.batch_size,
                chunk_paths=chunk_paths,
                chunk_cache_maxsize=cache_maxsize,
            )

    def __init__(
        self,
        *,
        settings: LogisticRegressionEvaluatorSettings,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        train_loader: DataLoader,
        test_loader: DataLoader,
        transform: Callable[[torch.Tensor], Evidence],
    ):
        self._settings = settings
        self._inference_machine_factory = inference_machine_factory
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._transform = transform

        input_dim = self._settings.height * self._settings.width + len(self._settings.feature_nodes)
        self._model_factory = lambda: LogisticRegressionEvaluator.LogisticRegressionModel(
            input_dim=input_dim,
            num_classes=self._settings.num_classes,
            torch_settings=self._settings.torch_settings,
        )

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        if self._settings.should_evaluate:
            if not self._settings.should_evaluate(epoch, iteration):
                return

        model = self._model_factory()

        train_loss = self._train(model, network)

        logging.info("Finished trainig , loss: %s", train_loss)

        accuracy = self._classify(model, network)

        logging.info(
            "Evaluated for epoch %s, iteration: %s, accuracy: %s",
            epoch,
            iteration,
            accuracy,
        )

        mlflow.log_metric(
            key="logreg_accuracy",
            step=epoch * self._settings.iterations_per_epoch + iteration,
            value=accuracy,
        )

    def _train(self, model: LogisticRegressionModel, network: BayesianNetwork):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self._settings.learning_rate)

        # %% Training loop
        for epoch in range(self._settings.epochs):
            model.train()
            total_loss = 0
            for i, (data, labels) in enumerate(self._train_loader):
                labels = labels.to(self._settings.torch_settings.device)

                # Infer hidden states
                inference_machine = self._inference_machine_factory(network)
                evidence = self._transform(data)
                inference_machine.enter_evidence(evidence)
                Z = inference_machine.infer_single_nodes(self._settings.feature_nodes)
                Z = torch.stack(Z)[:, :, :-1].permute([1, 0, 2]).flatten(start_dim=1, end_dim=2)

                data = torch.concat([data, Z], dim=1)

                # Train logistic regression model
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                logging.info(
                    "Finished trainig epoch %s, batch %s",
                    epoch,
                    i,
                )

            logging.info(
                "Finished trainig epoch %s/%s, loss: %s",
                epoch,
                self._settings.epochs,
                total_loss,
            )

    def _classify(self, model: LogisticRegressionModel, network: BayesianNetwork):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in self._test_loader:
                labels = labels.to(self._settings.torch_settings.device)

                # Infer hidden states
                inference_machine = self._inference_machine_factory(network)
                evidence = self._transform(data)
                inference_machine.enter_evidence(evidence)
                Z = inference_machine.infer_single_nodes(self._settings.feature_nodes)
                Z = torch.stack(Z)[:, :, :-1].permute([1, 0, 2]).flatten(start_dim=1, end_dim=2)

                data = torch.concat([data, Z], dim=1)

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
