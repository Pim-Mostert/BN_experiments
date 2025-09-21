import mlflow
from bayesian_network.optimizers.common import OptimizerLogger


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
