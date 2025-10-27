# %% Imports

from mlflow import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Configure MLflow

client = MlflowClient(tracking_uri="http://localhost:9000")

experiment_id = "6"
parent_run_id = "3be2841a90844a0db562cb25e8f062df"

child_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
)

# %% Collect log-likelihood history per child run

dfs = []
weight_plots = {}

for run in [child_run for child_run in child_runs if child_run.info.status == "FINISHED"]:
    train_metrics = client.get_metric_history(run.info.run_id, "ll_train")

    df = pd.DataFrame(
        [(h.step, h.value) for h in sorted(train_metrics, key=lambda x: x.step)],
        columns=["step", "ll"],
    )

    df["run_id"] = run.info.run_id
    df["num_classes"] = run.data.params["NUM_CLASSES"]
    df["num_features"] = run.data.params["NUM_FEATURES"]
    df["Y"] = run.data.params["Y_FEATURE_NODES"]
    df["F"] = run.data.params["F_FEATURE_NODES"]
    df["Q"] = run.data.params["Q_FEATURE_NODES"]
    df = (
        df.sort_values("step")
        .groupby(
            [
                "num_classes",
                "num_features",
                "Y",
                "F",
                "Q",
            ]
        )
        .last()
    ).reset_index()
    df = df.drop(columns=["step"])

    df["DoF"] = run.data.params["DoF"]

    logreg_metrics = client.get_metric_history(run.info.run_id, "logreg_accuracy")
    df["accuracy"] = logreg_metrics[0].value

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data["num_features"] = pd.to_numeric(data["num_features"])
data["num_classes"] = pd.to_numeric(data["num_classes"])
data["DoF"] = pd.to_numeric(data["DoF"])

data = data.reindex(
    columns=[
        "run_id",
        "Y",
        "F",
        "Q",
        "num_classes",
        "num_features",
        "DoF",
        "ll",
        "accuracy",
    ]
)

# %% Accuracy

filter = (data["num_features"] == 20) & (data["num_classes"] == 20)
data[filter].groupby(["Y", "F", "Q"])["accuracy"].first(skipna=False)

# %% Inspect accuracy vs ll

X = data.copy()
X["group"] = X.apply(
    lambda row: f"Y: {int(row['Y'] == 'True')}; F: {int(row['F'] == 'True')}; Q: {int(row['Q'] == 'True')}",
    axis=1,
)

plt.figure()
sns.lineplot(
    X,
    x="ll",
    y="accuracy",
    hue="group",
    palette="Set1",
)
plt.ylim([85, 100])

# %% Inspect accuracy vs DoF

X = data.copy()
X["group"] = X.apply(
    lambda row: f"Y: {int(row['Y'] == 'True')}; F: {int(row['F'] == 'True')}; Q: {int(row['Q'] == 'True')}",
    axis=1,
)

plt.figure()
sns.lineplot(
    X,
    x="DoF",
    y="accuracy",
    hue="group",
    palette="Set1",
)
plt.ylim([85, 100])

# %%
