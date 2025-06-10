# %% Imports

from dataclasses import dataclass
from mlflow import MlflowClient
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %% Configure ipykernel

# %config InlineBackend.figure_format = 'svg'

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


@dataclass
class PlotDataFilter:
    key: str
    value: float


def plot_ll(hue_key, filter1: PlotDataFilter, filter2: PlotDataFilter):
    plot_data = data[(data[filter1.key] == filter1.value) & (data[filter2.key] == filter2.value)]

    plt.figure(figsize=(10, 6))
    plt.title(f"{filter1.key}: {filter1.value}, {filter2.key}: {filter2.value}")
    sns.lineplot(data=plot_data, x="step", y="ll", hue=hue_key, palette="Spectral")


plot_ll(
    hue_key="batch_size",
    filter1=PlotDataFilter(key="learning_rate", value=0.01),
    filter2=PlotDataFilter(key="true_means_noise", value=0.4),
)

plot_ll(
    hue_key="learning_rate",
    filter1=PlotDataFilter(key="batch_size", value=2000),
    filter2=PlotDataFilter(key="true_means_noise", value=0.4),
)

plot_ll(
    hue_key="learning_rate",
    filter1=PlotDataFilter(key="batch_size", value=50),
    filter2=PlotDataFilter(key="true_means_noise", value=0.4),
)

plot_ll(
    hue_key="true_means_noise",
    filter1=PlotDataFilter(key="batch_size", value=2000),
    filter2=PlotDataFilter(key="learning_rate", value=0.01),
)

plot_ll(
    hue_key="learning_rate",
    filter1=PlotDataFilter(key="batch_size", value=1000),
    filter2=PlotDataFilter(key="true_means_noise", value=1),
)

# %%

filter = (
    ((data["learning_rate"] == 0.01) | (data["learning_rate"] == 0.5))
    & ((data["batch_size"] == 100) | (data["batch_size"] == 2000))
    & ((data["true_means_noise"] == 0) | (data["true_means_noise"] == 0.8))
)

plot_data = data[filter].copy()
plot_data["group"] = plot_data.apply(
    lambda row: f"LR: {row['learning_rate']}; BS: {row['batch_size']}; TMN: {row['true_means_noise']}",
    axis=1,
)


plt.figure(figsize=(10, 6))
sns.lineplot(
    data=plot_data,
    x="step",
    y="ll",
    hue="group",
)

# %%
