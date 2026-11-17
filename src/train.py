"""Etapa train (Parte 4): baseline naive estacional (24h) con cálculo de métricas."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    METRICS_PATH,
    PREDICTIONS_CSV_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    configure_logging,
    load_params,
)

LOGGER = configure_logging()
SEASONAL_PERIOD_HOURS = 24


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, MAE y MAPE (%) sobre el horizonte de prueba."""
    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / y_true)) * 100.0)
    return {
        "rmse_test": round(rmse, 4),
        "mae_test": round(mae, 4),
        "mape_test": round(mape, 4),
    }


def forecast_seasonal_naive_24h(train_series: pd.Series, horizon: int) -> np.ndarray:
    """Pronostica h horas repitiendo el último ciclo observado de 24h sin usar datos de test."""
    if len(train_series) < SEASONAL_PERIOD_HOURS:
        raise ValueError("La serie de entrenamiento debe tener al menos 24 observaciones.")
    last_cycle = train_series.iloc[-SEASONAL_PERIOD_HOURS:].to_numpy(dtype=float)
    repeats = int(np.ceil(horizon / SEASONAL_PERIOD_HOURS))
    return np.tile(last_cycle, repeats)[:horizon]


def run_baseline_train() -> dict[str, Any]:
    """Entrena/evalúa el baseline estacional 24h y guarda métricas y predicciones."""
    params = load_params()
    train_df = pd.read_csv(TRAIN_CSV_PATH, parse_dates=["timestamp"], index_col="timestamp")
    test_df = pd.read_csv(TEST_CSV_PATH, parse_dates=["timestamp"], index_col="timestamp")

    y_pred = forecast_seasonal_naive_24h(train_df["demanda_mwh"], horizon=len(test_df))
    y_true = test_df["demanda_mwh"].to_numpy(dtype=float)
    metrics = compute_regression_metrics(y_true, y_pred)

    pred_df = pd.DataFrame(
        {
            "timestamp": test_df.index,
            "demanda_real_mwh": y_true,
            "pronostico_mwh": y_pred,
        }
    )
    PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

    output = {
        "modelo": str(params["train"]["modelo"]),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **metrics,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Baseline naive 24h evaluado: %s", output)
    return output


if __name__ == "__main__":
    run_baseline_train()
