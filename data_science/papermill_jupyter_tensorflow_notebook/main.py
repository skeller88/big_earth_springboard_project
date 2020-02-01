
import papermill as pm


def run_notebook():
    pm.execute_notebook(
        '/app/data_science/model.ipynb',
        '/app/data_science/model_output.ipynb',
        parameters=dict(),
        log_output=True
    )


if __name__ == "__main__":
    run_notebook()
