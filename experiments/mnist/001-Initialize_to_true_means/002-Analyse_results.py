# %% Imports

from mlflow import MlflowClient
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %% Configure MLflow

client = MlflowClient(tracking_uri="http://localhost:9000")

experiment_id = "1"
parent_run_id = "c31ed399867b47c79b943110d29faee5"

child_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
)

# %% Collect log-likelihood history per child run

dfs = []
for run in child_runs:
    train_metrics = client.get_metric_history(run.info.run_id, "ll_train")
    eval_metrics = client.get_metric_history(run.info.run_id, "ll_eval")

    df = pd.DataFrame(
        [(h.step, h.value) for h in sorted(train_metrics, key=lambda x: x.step)],
        columns=["step", "ll"],
    )
    df["mode"] = "train"

    df_eval = pd.DataFrame(
        [(h.step, h.value) for h in sorted(eval_metrics, key=lambda x: x.step)],
        columns=["step", "ll"],
    )
    df_eval["mode"] = "eval"

    df = pd.concat([df, df_eval])

    df["run_id"] = run.info.run_id
    df["batch_size"] = np.float64(run.data.params["BATCH_SIZE"])
    df["learning_rate"] = np.float64(run.data.params["LEARNING_RATE"])
    df["true_means_noise"] = np.float64(run.data.params["TRUE_MEANS_NOISE"])

    df = df.reindex(
        columns=[
            "run_id",
            "batch_size",
            "learning_rate",
            "true_means_noise",
            "mode",
            "step",
            "ll",
        ]
    )

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

data["epoch"] = data["step"] / (60000 / data["batch_size"])

# %%

# fmt: off
filter = (
    1 
    & (data["batch_size"] == 1000) 
    & (data["mode"] == "eval") 
    # & (data["epoch"] > 3)
)
# fmt: on

plot_data = data[filter].copy()
plot_data["group"] = plot_data.apply(
    lambda row: f"Mode: {row['mode']}; LR: {row['learning_rate']}; BS: {row['batch_size']}; TMN: {row['true_means_noise']}",
    axis=1,
)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=plot_data,
    x="epoch",
    y="ll",
    hue="group",
)

# %%

final_ll = (
    data.sort_values("step").groupby(["batch_size", "learning_rate", "true_means_noise"]).last()
)

g = sns.catplot(
    final_ll,
    x="true_means_noise",
    y="ll",
    hue="learning_rate",
    col="batch_size",
    kind="bar",
)

g.set(ylim=(-162, -140))
# %%
