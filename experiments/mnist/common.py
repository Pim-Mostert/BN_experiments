from typing import Callable
import mlflow
from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.optimizers.common import OptimizerLogger, BatchEvaluator


class MLflowOptimizerLogger(OptimizerLogger):
    def __init__(
        self,
        iterations_per_epoch: int,
        should_log: Callable[[int, int], bool] | None = None,
    ):
        super().__init__(should_log)

        self._iterations_per_epoch = iterations_per_epoch

    def log(self, epoch: int, iteration: int, ll: float):
        super().log(epoch, iteration, ll)

        log = self._logs[-1]

        mlflow.log_metric(
            key="ll_train",
            value=log.ll,
            step=epoch * self._iterations_per_epoch + iteration,
            timestamp=int(log.ts.timestamp()),
        )


class MLflowBatchEvaluator(BatchEvaluator):
    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        if self._should_evaluate:
            if not self._should_evaluate(epoch, iteration):
                return

        super().evaluate(epoch, iteration, network)

        ll_eval = self._log_likelihoods[(epoch, iteration)]

        mlflow.log_metric(
            key="ll_eval",
            value=ll_eval,
            step=iteration,
        )
