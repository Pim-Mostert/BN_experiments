import mlflow
from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.optimizers.common import OptimizerLogger, BatchEvaluator


class MLflowOptimizerLogger(OptimizerLogger):
    def log(self, epoch: int, iteration: int, ll: float):
        super().log(epoch, iteration, ll)

        log = self._logs[-1]

        mlflow.log_metric(
            key="ll",
            value=log.ll,
            step=iteration,
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
