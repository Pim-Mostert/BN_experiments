# %%
from prefect import flow
from prefect.runner.storage import GitRepository

if __name__ == "__main__":

    github_repo = GitRepository(
        url="https://github.com/Pim-Mostert/BN_experiments",
        branch="try-prefect-as-tool",
    )

    flow.from_source(
        source=github_repo,
        entrypoint="analyses/mnist/main.py:experiment",
    ).serve(name="experiment", limit=10)
