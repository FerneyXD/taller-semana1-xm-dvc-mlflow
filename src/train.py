"""Etapa train del pipeline DVC: ajuste SARIMAX (o baseline), evaluación y registro en MLflow."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    PARAMS_PATH,
    PREDICTIONS_CSV_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    configure_logging,
    get_git_commit_hash,
    load_params,
    read_dvc_data_hash,
)

LOGGER = configure_logging()
SEASONAL_PERIOD_HOURS = 24
CONTEXT_TRAIN_HOURS = 72


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, MAE y MAPE (%) sobre el horizonte de prueba."""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud y no estar vacíos.")
    if np.any(y_true == 0):
        raise ValueError("y_true contiene ceros; no es posible calcular MAPE.")

    errors = y_true - y_pred
    return {
        "rmse_test": float(np.sqrt(np.mean(errors**2))),
        "mae_test": float(np.mean(np.abs(errors))),
        "mape_test": float(np.mean(np.abs(errors / y_true)) * 100.0),
    }


def forecast_seasonal_naive_24h(train_series: pd.Series, horizon: int) -> np.ndarray:
    """Pronostica h horas repitiendo el último ciclo observado de 24h sin usar datos de test."""
    if len(train_series) < SEASONAL_PERIOD_HOURS:
        raise ValueError("La serie de entrenamiento debe tener al menos 24 observaciones.")
    last_cycle = train_series.iloc[-SEASONAL_PERIOD_HOURS:].to_numpy(dtype=float)
    repeats = int(np.ceil(horizon / SEASONAL_PERIOD_HOURS))
    return np.tile(last_cycle, repeats)[:horizon]


def build_sarimax_run_name(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    usar_exogena: bool,
) -> str:
    """Construye el nombre de run según convención: sarimax_{exog|sinexog}_pdq_PDQs."""
    exog_tag = "exog" if usar_exogena else "sinexog"
    p, d, q = order
    P, D, Q, s = seasonal_order
    return f"sarimax_{exog_tag}_{p}{d}{q}_{P}{D}{Q}_{s}"


def save_forecast_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    run_name: str,
    rmse_test: float,
    output_path: Path,
) -> None:
    """Guarda gráfica de pronóstico vs valor real con intervalo de predicción del 95%."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = train_df.iloc[-CONTEXT_TRAIN_HOURS:]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(context.index, context["demanda_mwh"], label="Entrenamiento (últimas 72h)", color="#5b6e82")
    ax.plot(test_df.index, pred_df["demanda_real_mwh"], label="Real (prueba 48h)", color="#1f4e79", linewidth=2)
    ax.plot(
        test_df.index,
        pred_df["pronostico_mwh"],
        label=f"Pronóstico ({run_name})",
        color="#d95f02",
        linestyle="--",
        linewidth=2,
    )
    ax.fill_between(
        test_df.index,
        pred_df["ic_inferior_95"],
        pred_df["ic_superior_95"],
        color="#d95f02",
        alpha=0.18,
        label="Intervalo de predicción 95%",
    )
    ax.axvline(test_df.index.min(), color="#333333", linestyle=":", linewidth=1.2, label="Corte train/test")
    ax.set_title(f"Pronóstico 48h vs Real — {run_name} | RMSE = {rmse_test:.2f} MWh")
    ax.set_ylabel("Demanda eléctrica (MWh)")
    ax.set_xlabel("Fecha y hora (America/Bogota)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_diagnostics_plot(fitted_model: Any, output_path: Path) -> None:
    """Guarda los 4 gráficos estándar de diagnóstico de residuales de statsmodels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = fitted_model.plot_diagnostics(figsize=(12, 8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fit_and_evaluate_sarimax(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    usar_exogena: bool,
) -> tuple[Any, pd.DataFrame, dict[str, float]]:
    """Ajusta SARIMAX y genera pronóstico de prueba, intervalos y métricas."""
    exog_train = train_df[["temperatura_c"]] if usar_exogena else None
    exog_test = test_df[["temperatura_c"]] if usar_exogena else None

    model = SARIMAX(
        endog=train_df["demanda_mwh"],
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False, maxiter=100)

    forecast_obj = fitted.get_forecast(steps=len(test_df), exog=exog_test)
    y_pred = forecast_obj.predicted_mean.to_numpy(dtype=float)
    conf_int = forecast_obj.conf_int(alpha=0.05)
    y_true = test_df["demanda_mwh"].to_numpy(dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["aic"] = float(fitted.aic)
    metrics["bic"] = float(fitted.bic)

    pred_df = pd.DataFrame(
        {
            "timestamp": test_df.index,
            "demanda_real_mwh": y_true,
            "pronostico_mwh": y_pred,
            "ic_inferior_95": conf_int.iloc[:, 0].to_numpy(dtype=float),
            "ic_superior_95": conf_int.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    return fitted, pred_df, metrics


def log_run_to_mlflow(
    run_name: str,
    params_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict[str, float],
    converged: bool,
    artifact_paths: list[Path],
) -> str:
    """Registra parámetros, métricas, artefactos, tags y datasets en el servidor MLflow."""
    train_cfg = params_cfg["train"]
    prep_cfg = params_cfg["prepare"]
    order = tuple(int(v) for v in train_cfg["orden"])
    seasonal_order = tuple(int(v) for v in train_cfg["orden_estacional"])
    usar_exogena = bool(train_cfg["usar_exogena"])

    dvc_hash = read_dvc_data_hash()
    git_commit = get_git_commit_hash()

    mlflow.set_tracking_uri(str(train_cfg["mlflow_tracking_uri"]))
    mlflow.set_experiment(str(train_cfg["experimento_mlflow"]))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "dvc_data_hash": dvc_hash,
                "git_commit": git_commit,
                "fuente": str(train_cfg["fuente"]),
                "train_start": train_df.index.min().isoformat(),
                "train_end": train_df.index.max().isoformat(),
                "test_start": test_df.index.min().isoformat(),
                "test_end": test_df.index.max().isoformat(),
            }
        )
        mlflow.log_params(
            {
                "modelo": str(train_cfg["modelo"]),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "P": seasonal_order[0],
                "D": seasonal_order[1],
                "Q": seasonal_order[2],
                "s": seasonal_order[3],
                "usar_exogena": usar_exogena,
                "supuesto_exogena": (
                    "temperatura_observada_ex_post_bogota" if usar_exogena else "sin_exogena"
                ),
                "dias_entrenamiento": int(prep_cfg["dias_entrenamiento"]),
                "horas_prueba": int(prep_cfg["horas_prueba"]),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "converged": bool(converged),
            }
        )
        mlflow.log_metrics(metrics)

        train_dataset = mlflow.data.from_pandas(
            train_df.reset_index(),
            source="data/processed/train.csv",
            name=f"xm_demanda_train_{dvc_hash[:8]}",
            targets="demanda_mwh",
        )
        test_dataset = mlflow.data.from_pandas(
            test_df.reset_index(),
            source="data/processed/test.csv",
            name=f"xm_demanda_test_{dvc_hash[:8]}",
            targets="demanda_mwh",
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="testing")

        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        return run.info.run_id


def run_train() -> dict[str, Any]:
    """Ejecuta la etapa de entrenamiento según params.yaml (baseline o SARIMAX)."""
    params = load_params()
    train_cfg = params["train"]
    model_type = str(train_cfg["modelo"]).lower()

    train_df = pd.read_csv(TRAIN_CSV_PATH, parse_dates=["timestamp"], index_col="timestamp").asfreq("h")
    test_df = pd.read_csv(TEST_CSV_PATH, parse_dates=["timestamp"], index_col="timestamp").asfreq("h")

    PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "baseline":
        y_pred = forecast_seasonal_naive_24h(train_df["demanda_mwh"], horizon=len(test_df))
        y_true = test_df["demanda_mwh"].to_numpy(dtype=float)
        metrics = compute_regression_metrics(y_true, y_pred)
        pred_df = pd.DataFrame(
            {"timestamp": test_df.index, "demanda_real_mwh": y_true, "pronostico_mwh": y_pred}
        )
        pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)
        output = {"modelo": "baseline", "n_train": int(len(train_df)), "n_test": int(len(test_df)), **metrics}
    elif model_type == "sarimax":
        order = tuple(int(v) for v in train_cfg["orden"])
        seasonal_order = tuple(int(v) for v in train_cfg["orden_estacional"])
        usar_exogena = bool(train_cfg["usar_exogena"])
        run_name = build_sarimax_run_name(order, seasonal_order, usar_exogena)

        LOGGER.info(
            "Ajustando %s | order=%s | seasonal_order=%s | exog=%s | n_train=%d",
            run_name,
            order,
            seasonal_order,
            usar_exogena,
            len(train_df),
        )
        fitted, pred_df, metrics = fit_and_evaluate_sarimax(
            train_df, test_df, order, seasonal_order, usar_exogena
        )
        pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

        forecast_fig = FIGURES_DIR / "pronostico_vs_real.png"
        diag_fig = FIGURES_DIR / "diagnostico_residuales.png"
        save_forecast_plot(train_df, test_df, pred_df, run_name, metrics["rmse_test"], forecast_fig)
        save_diagnostics_plot(fitted, diag_fig)

        converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
        run_id = log_run_to_mlflow(
            run_name=run_name,
            params_cfg=params,
            train_df=train_df,
            test_df=test_df,
            metrics=metrics,
            converged=converged,
            artifact_paths=[forecast_fig, diag_fig, PREDICTIONS_CSV_PATH, PARAMS_PATH],
        )
        output = {
            "modelo": run_name,
            "mlflow_run_id": run_id,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "converged": converged,
            **metrics,
        }
    else:
        raise ValueError(f"Tipo de modelo no soportado en params.yaml: '{model_type}'")

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Entrenamiento finalizado: %s", output)
    return output


if __name__ == "__main__":
    run_train()
