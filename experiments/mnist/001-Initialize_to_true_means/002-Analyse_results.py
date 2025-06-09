# %% Imports

from mlflow import MlflowClient
import pandas as pd
import seaborn as sns
import numpy as np

# %% Configure MLflow

client = MlflowClient(tracking_uri="http://localhost:9000")

experiment_id = "2"
parent_run_id = "75e6d5bb1eed4f739ef255af91f6ccc7"  # Wide search

child_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
)

# %% Collect log-likelihood history per child run

metric_key = "ll"

dfs = []
for run in child_runs:
    history = client.get_metric_history(run.info.run_id, metric_key)

    df = pd.DataFrame(
        [
            (h.step, h.value)
            for h in sorted(
                history,
                key=lambda x: x.step,
            )
        ],
        columns=["step", "ll"],
    )
    df["run_id"] = run.info.run_id
    df["batch_size"] = np.float64(run.data.params["BATCH_SIZE"])
    df["learning_rate"] = np.float64(run.data.params["LEARNING_RATE"])
    df["true_means_noise"] = np.float64(run.data.params["TRUE_MEANS_NOISE"])

    df = df.reindex(
        columns=["run_id", "batch_size", "learning_rate", "true_means_noise", "step", "ll"]
    )

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# %%

plot_data = data[(data["learning_rate"] == 0.01) & (data["true_means_noise"] == 0)]

sns.lineplot(data=plot_data, x="step", y="ll")

# %%
