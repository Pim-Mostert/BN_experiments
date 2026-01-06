# %% Imports

from mlflow import MlflowClient
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% Configure MLflow

client = MlflowClient(tracking_uri="http://localhost:9000")

experiment_id = "5"
parent_run_id = "bdc064f84c074356b364808835afaef7"

child_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
)

# %% Collect log-likelihood history per child run

dfs = []
weight_plots = {}

for run in [child_run for child_run in child_runs if child_run.info.status == "FINISHED"]:
    train_metrics = client.get_metric_history(run.info.run_id, "ll_train")
    eval_metrics = client.get_metric_history(run.info.run_id, "ll_eval")
    dof_metrics = client.get_metric_history(run.info.run_id, "DoF")

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
    df["num_classes"] = run.data.params["NUM_CLASSES"]
    df["num_features"] = run.data.params["NUM_FEATURES"]
    df["DoF"] = dof_metrics[0].value
    df["epoch"] = df["step"] / (60000 / int(run.data.params["BATCH_SIZE"]))

    df = df.reindex(
        columns=[
            "run_id",
            "num_classes",
            "num_features",
            "DoF",
            "mode",
            "step",
            "ll",
        ]
    )

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)


# %% Training log-likelihood over time

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=data,
    x="step",
    y="ll",
    hue="num_classes",
    style="num_features",
)

# %% Final eval log-likelihood

final_ll = (
    data[(data["mode"] == "eval")]
    .sort_values("step")
    .groupby(["num_classes", "num_features"])
    .last()
).reset_index()
final_ll["num_classes"] = pd.to_numeric(final_ll["num_classes"])
final_ll["num_features"] = pd.to_numeric(final_ll["num_features"])

pivot_df = final_ll.pivot(index="num_classes", columns="num_features", values="ll")

sns.heatmap(data=pivot_df)

# %% DoF

final_ll["log10_DoF"] = np.log10(final_ll["DoF"].to_numpy())
plt.figure()
sns.lineplot(
    final_ll,
    x="log10_DoF",
    y="ll",
    palette="Set1",
    hue="num_classes",
    marker="o",
)

plt.figure()
sns.lineplot(
    final_ll,
    x="log10_DoF",
    y="ll",
    palette="Set1",
    hue="num_features",
    marker="o",
)


# %%
