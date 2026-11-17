"""Etapa prepare del pipeline DVC: split temporal estricto train/test sin fuga."""

from __future__ import annotations

import pandas as pd

from src.config import (
    RAW_CSV_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    configure_logging,
    load_params,
)

LOGGER = configure_logging()


def load_consolidated_dataset(csv_path=RAW_CSV_PATH) -> pd.DataFrame:
    """Carga el CSV consolidado y valida índice horario regular sin nulos."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos versionado: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    if df.empty:
        raise ValueError("El archivo de datos consolidado está vacío.")
    if df[["demanda_mwh", "temperatura_c"]].isna().any().any():
        raise ValueError("El dataset de entrada contiene valores nulos inesperados.")

    diffs = df.index.to_series().diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise ValueError("El índice temporal del dataset no es horario regular (freq=1h).")
    return df


def split_temporal_train_test(
    df: pd.DataFrame,
    dias_entrenamiento: int,
    horas_prueba: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide la serie en train y test estrictamente por tiempo sin mezclar fechas."""
    if dias_entrenamiento <= 0 or horas_prueba <= 0:
        raise ValueError("dias_entrenamiento y horas_prueba deben ser enteros positivos.")

    train_hours = dias_entrenamiento * 24
    min_required = train_hours + horas_prueba
    if len(df) < min_required:
        raise ValueError(
            f"Se requieren al menos {min_required} horas, pero el CSV solo tiene {len(df)}."
        )

    test_df = df.iloc[-horas_prueba:].copy()
    train_pool = df.iloc[:-horas_prueba]
    train_df = train_pool.iloc[-train_hours:].copy()

    if train_df.index.max() >= test_df.index.min():
        raise RuntimeError("Fuga temporal detectada: train y test se solapan.")
    if test_df.index.min() - train_df.index.max() != pd.Timedelta(hours=1):
        raise RuntimeError("Existe discontinuidad entre el fin de train y el inicio de test.")

    return train_df, test_df


def run_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta la etapa de preparación según params.yaml y guarda train/test."""
    params = load_params()
    prep_cfg = params["prepare"]
    dias_entrenamiento = int(prep_cfg["dias_entrenamiento"])
    horas_prueba = int(prep_cfg["horas_prueba"])

    df = load_consolidated_dataset()
    train_df, test_df = split_temporal_train_test(df, dias_entrenamiento, horas_prueba)

    TRAIN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV_PATH, index=True)
    test_df.to_csv(TEST_CSV_PATH, index=True)

    LOGGER.info(
        "Split completado | train: %d filas (%s -> %s) | test: %d filas (%s -> %s)",
        len(train_df),
        train_df.index.min(),
        train_df.index.max(),
        len(test_df),
        test_df.index.min(),
        test_df.index.max(),
    )
    return train_df, test_df


if __name__ == "__main__":
    run_prepare()
