# %% Imports

from pathlib import Path
import tempfile
import mlflow
from mlflow import MlflowClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# %% Configure MLflow

mlflow.set_tracking_uri("http://localhost:9000")
client = MlflowClient(tracking_uri="http://localhost:9000")

experiment_id = "2"
parent_run_id = "3569f1f234aa48dd8781edf18e1d34e2"

child_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
)

# %% Collect log-likelihood history per child run

dfs = []
weight_plots = {}

for run in child_runs:
    train_metrics = client.get_metric_history(run.info.run_id, "ll_train")
    eval_metrics = client.get_metric_history(run.info.run_id, "ll_eval")

    with tempfile.TemporaryDirectory() as tmpdirname:
        client.download_artifacts(run.info.run_id, "weights.png", dst_path=tmpdirname)

        file_path = (Path(tmpdirname) / "weights.png").as_posix()
        weight_plots[run.data.params["NUM_CLASSES"]] = Image.open(file_path)

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

    df = df.reindex(
        columns=[
            "run_id",
            "num_classes",
            "mode",
            "step",
            "ll",
        ]
    )

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

data["epoch"] = data["step"] / 60

# %% Training log-likelihood over time

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=data,
    x="epoch",
    y="ll",
    hue="num_classes",
)

# %% Final eval log-likelihood

final_ll = data[(data["mode"] == "eval")].sort_values("step").groupby(["num_classes"]).last()

g = sns.catplot(
    final_ll,
    x="num_classes",
    y="ll",
    kind="bar",
    order=data["num_classes"].unique(),
)

# %% Weights

for num_classes in sorted([int(x) for x in weight_plots]):
    img = weight_plots[str(num_classes)]

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"num_classes: {num_classes}")

# %%
