"""Comparación de runs de MLflow y exportación a comparacion_runs.csv (Parte 7)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src.config import PROJECT_ROOT, configure_logging, load_params

LOGGER = configure_logging()
COMPARISON_CSV_PATH = PROJECT_ROOT / "comparacion_runs.csv"


def export_runs_comparison(output_path: Path = COMPARISON_CSV_PATH) -> pd.DataFrame:
    """Consulta MLflow con search_runs, ordena por rmse_test ASC y exporta tabla limpia."""
    params = load_params()
    tracking_uri = str(params["train"]["mlflow_tracking_uri"])
    experiment_name = str(params["train"]["experimento_mlflow"])

    mlflow.set_tracking_uri(tracking_uri)
    runs_df = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.rmse_test ASC"],
    )
    if runs_df.empty:
        raise RuntimeError(f"No se encontraron runs en el experimento '{experiment_name}'.")

    columns_map = {
        "run_id": "run_id",
        "tags.mlflow.runName": "run_name",
        "params.p": "p",
        "params.d": "d",
        "params.q": "q",
        "params.P": "P",
        "params.D": "D",
        "params.Q": "Q",
        "params.s": "s",
        "params.usar_exogena": "usar_exogena",
        "params.dias_entrenamiento": "dias_entrenamiento",
        "params.n_train": "n_train",
        "params.n_test": "n_test",
        "params.converged": "converged",
        "metrics.rmse_test": "rmse_test",
        "metrics.mae_test": "mae_test",
        "metrics.mape_test": "mape_test",
        "metrics.aic": "aic",
        "metrics.bic": "bic",
        "tags.dvc_data_hash": "dvc_data_hash",
        "tags.git_commit": "git_commit",
        "tags.fuente": "fuente",
        "tags.test_start": "test_start",
        "tags.test_end": "test_end",
    }
    available_cols = [col for col in columns_map if col in runs_df.columns]
    clean_df = runs_df[available_cols].rename(columns=columns_map)
    clean_df.to_csv(output_path, index=False)

    LOGGER.info("Tabla comparativa exportada con %d runs en %s", len(clean_df), output_path)
    return clean_df


if __name__ == "__main__":
    table = export_runs_comparison()
    cols_to_show = [
        "run_name",
        "usar_exogena",
        "rmse_test",
        "mae_test",
        "mape_test",
        "aic",
        "bic",
        "dvc_data_hash",
        "git_commit",
    ]
    print(table[[c for c in cols_to_show if c in table.columns]].to_string(index=False))

