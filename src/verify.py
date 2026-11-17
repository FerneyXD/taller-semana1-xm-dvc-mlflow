"""Auditoría y verificación automatizada del pipeline (Parte 6 del taller)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import statsmodels
import dvc
import yaml

from src.config import (
    METRICS_PATH,
    PROJECT_ROOT,
    RAW_CSV_PATH,
    RAW_DVC_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    configure_logging,
    load_params,
)

LOGGER = configure_logging()


def verify_no_data_leakage(train_path: Path = TRAIN_CSV_PATH, test_path: Path = TEST_CSV_PATH) -> dict[str, Any]:
    """Comprueba split temporal estricto: sin solapamiento, sin huecos y sin nulos."""
    train_df = pd.read_csv(train_path, parse_dates=["timestamp"], index_col="timestamp")
    test_df = pd.read_csv(test_path, parse_dates=["timestamp"], index_col="timestamp")

    gap_hours = (test_df.index.min() - train_df.index.max()).total_seconds() / 3600.0
    train_diffs = train_df.index.to_series().diff().dropna().dt.total_seconds()
    test_diffs = test_df.index.to_series().diff().dropna().dt.total_seconds()

    passed = (
        train_df.index.max() < test_df.index.min()
        and gap_hours == 1.0
        and bool((train_diffs == 3600.0).all())
        and bool((test_diffs == 3600.0).all())
        and not train_df.isna().any().any()
        and not test_df.isna().any().any()
    )
    return {
        "check": "split_temporal_sin_fuga",
        "passed": passed,
        "train_range": [train_df.index.min().isoformat(), train_df.index.max().isoformat()],
        "test_range": [test_df.index.min().isoformat(), test_df.index.max().isoformat()],
        "gap_hours_between_train_and_test": gap_hours,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }


def verify_manual_rmse_vs_mlflow(tracking_uri: str, experiment_name: str) -> dict[str, Any]:
    """Recalcula a mano con NumPy el RMSE de un run desde su artefacto CSV y contrasta con MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.rmse_test ASC"],
    )
    if runs.empty:
        raise RuntimeError(f"No hay runs registrados en el experimento '{experiment_name}'.")

    top_run = runs.iloc[0]
    run_id = str(top_run["run_id"])
    run_name = str(top_run["tags.mlflow.runName"])
    mlflow_rmse = float(top_run["metrics.rmse_test"])
    mlflow_mae = float(top_run["metrics.mae_test"])
    mlflow_mape = float(top_run["metrics.mape_test"])

    local_csv = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="predicciones_test.csv"))
    pred_df = pd.read_csv(local_csv)
    y_true = pred_df["demanda_real_mwh"].to_numpy(dtype=float)
    y_pred = pred_df["pronostico_mwh"].to_numpy(dtype=float)

    manual_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    manual_mae = float(np.mean(np.abs(y_true - y_pred)))
    manual_mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

    abs_diff_rmse = abs(manual_rmse - mlflow_rmse)
    return {
        "check": "recalculo_manual_rmse_numpy_vs_mlflow",
        "passed": abs_diff_rmse < 1e-9,
        "run_id": run_id,
        "run_name": run_name,
        "mlflow_rmse": mlflow_rmse,
        "manual_numpy_rmse": manual_rmse,
        "abs_diff_rmse": abs_diff_rmse,
        "mlflow_mae": mlflow_mae,
        "manual_numpy_mae": manual_mae,
        "mlflow_mape": mlflow_mape,
        "manual_numpy_mape": manual_mape,
    }


def verify_traceability_git_and_dvc(tracking_uri: str, experiment_name: str) -> dict[str, Any]:
    """Verifica que git_commit y dvc_data_hash de cada run coincidan con Git y su archivo .dvc."""
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    details: list[dict[str, Any]] = []
    all_passed = True

    for _, row in runs.iterrows():
        run_name = str(row.get("tags.mlflow.runName", ""))
        git_commit = str(row.get("tags.git_commit", ""))
        logged_dvc_hash = str(row.get("tags.dvc_data_hash", ""))

        git_cat = subprocess.run(
            ["git", "show", f"{git_commit}:data/xm_demanda.csv.dvc"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit_exists = git_cat.returncode == 0
        dvc_md5_in_commit = None
        if commit_exists:
            dvc_meta = yaml.safe_load(git_cat.stdout)
            dvc_md5_in_commit = dvc_meta["outs"][0]["md5"]

        matches = commit_exists and (dvc_md5_in_commit == logged_dvc_hash)
        if not matches:
            all_passed = False

        details.append(
            {
                "run_name": run_name,
                "git_commit": git_commit[:10],
                "commit_exists_in_git": commit_exists,
                "logged_dvc_data_hash": logged_dvc_hash,
                "dvc_hash_in_git_commit": dvc_md5_in_commit,
                "matches": matches,
            }
        )

    return {
        "check": "trazabilidad_git_y_dvc",
        "passed": all_passed and len(details) > 0,
        "total_runs_checked": len(details),
        "runs": details,
    }


def verify_domain_and_libraries(raw_csv: Path = RAW_CSV_PATH) -> dict[str, Any]:
    """Verifica unidades XM (MWh), alineación horaria 00:00..23:00 y versiones instaladas."""
    df = pd.read_csv(raw_csv, parse_dates=["timestamp"], index_col="timestamp")
    hours_present = sorted(df.index.hour.unique().tolist())
    hourly_mean = df.groupby(df.index.hour)["demanda_mwh"].mean()
    peak_hour = int(hourly_mean.idxmax())
    valley_hour = int(hourly_mean.idxmin())

    passed = (
        hours_present == list(range(24))
        and 6000.0 < float(df["demanda_mwh"].mean()) < 15000.0
        and peak_hour in (18, 19, 20)
        and valley_hour in (2, 3, 4, 5)
    )
    return {
        "check": "dominio_xm_unidades_y_versiones",
        "passed": passed,
        "hours_present_count": len(hours_present),
        "mean_demand_mwh": round(float(df["demanda_mwh"].mean()), 2),
        "peak_hour_bogota": peak_hour,
        "valley_hour_bogota": valley_hour,
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "statsmodels": statsmodels.__version__,
            "mlflow": mlflow.__version__,
            "dvc": dvc.__version__,
        },
        "exog_assumption": "Temperatura horaria observada en Bogotá (Open-Meteo, evaluación ex-post).",
    }


def run_all_verifications() -> dict[str, Any]:
    """Ejecuta todas las verificaciones de la Parte 6 y retorna el reporte consolidado."""
    params = load_params()
    tracking_uri = str(params["train"]["mlflow_tracking_uri"])
    experiment_name = str(params["train"]["experimento_mlflow"])

    report = {
        "verificacion_1_sin_fuga": verify_no_data_leakage(),
        "verificacion_2_rmse_numpy": verify_manual_rmse_vs_mlflow(tracking_uri, experiment_name),
        "verificacion_3_trazabilidad": verify_traceability_git_and_dvc(tracking_uri, experiment_name),
        "verificacion_4_dominio_y_apis": verify_domain_and_libraries(),
    }
    report["all_passed"] = all(item["passed"] for item in report.values())
    return report


if __name__ == "__main__":
    result = run_all_verifications()
    print(json.dumps(result, indent=2, ensure_ascii=False))

