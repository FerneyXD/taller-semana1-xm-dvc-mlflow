"""Ingesta real de demanda horaria XM (DemaCome) y temperatura horaria Open-Meteo."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import requests
from pydataxm import pydataxm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    BOGOTA_LATITUDE,
    BOGOTA_LONGITUDE,
    BOGOTA_TIMEZONE,
    FIGURES_DIR,
    KWH_TO_MWH_FACTOR,
    OPEN_METEO_ARCHIVE_URL,
    QUALITY_REPORT_PATH,
    RAW_CSV_PATH,
    configure_logging,
)

LOGGER = configure_logging()
XM_CHUNK_DAYS = 25


def parse_iso_date(date_text: str) -> dt.date:
    """Convierte una cadena YYYY-MM-DD en objeto date validando su formato."""
    try:
        return dt.date.fromisoformat(date_text)
    except ValueError as exc:
        raise ValueError(f"Fecha inválida '{date_text}'. Use formato YYYY-MM-DD.") from exc


def fetch_xm_demand_raw(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Consulta DemaCome del Sistema en XM dividiendo en bloques para evitar límites de API."""
    if start_date > end_date:
        raise ValueError("start_date no puede ser posterior a end_date.")

    api = pydataxm.ReadDB()
    chunks: list[pd.DataFrame] = []
    cursor = start_date

    while cursor <= end_date:
        chunk_end = min(cursor + dt.timedelta(days=XM_CHUNK_DAYS - 1), end_date)
        LOGGER.info("Consultando XM DemaCome/Sistema desde %s hasta %s", cursor, chunk_end)
        chunk = api.request_data("DemaCome", "Sistema", cursor, chunk_end)
        if chunk is not None and not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end + dt.timedelta(days=1)

    if not chunks:
        raise RuntimeError(f"XM no devolvió registros para el rango {start_date}..{end_date}.")

    return pd.concat(chunks, ignore_index=True)


def transform_xm_wide_to_hourly(raw_df: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Convierte tabla diaria XM (Values_Hour01..24 en kWh) a serie horaria regular en MWh."""
    hour_columns = [f"Values_Hour{hour:02d}" for hour in range(1, 25)]
    missing_cols = [col for col in hour_columns + ["Date"] if col not in raw_df.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas esperadas en respuesta XM: {missing_cols}")

    long_df = raw_df.melt(
        id_vars=["Date"],
        value_vars=hour_columns,
        var_name="hour_label",
        value_name="demanda_kwh",
    )
    long_df["hour_num"] = long_df["hour_label"].str.extract(r"(\d+)$").astype(int) - 1
    long_df["timestamp"] = pd.to_datetime(long_df["Date"]) + pd.to_timedelta(
        long_df["hour_num"], unit="h"
    )
    long_df["demanda_mwh"] = pd.to_numeric(long_df["demanda_kwh"], errors="coerce") / KWH_TO_MWH_FACTOR

    duplicate_count = int(long_df.duplicated(subset=["timestamp"]).sum())
    if duplicate_count > 0:
        LOGGER.warning("Se detectaron %d marcas temporales duplicadas en XM; se conserva la última.", duplicate_count)
        long_df = long_df.drop_duplicates(subset=["timestamp"], keep="last")

    series = long_df.set_index("timestamp")["demanda_mwh"].sort_index()
    full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq="h")
    series = series.reindex(full_index)
    series.index.name = "timestamp"

    gap_count = int(series.isna().sum())
    if gap_count > 0:
        LOGGER.warning("Se detectaron %d huecos horarios en demanda XM; interpolación temporal.", gap_count)
        series = series.interpolate(method="time").bfill().ffill()

    return series, {"duplicados_xm": duplicate_count, "huecos_xm_imputados": gap_count}


def fetch_open_meteo_temperature(
    start_date: dt.date,
    end_date: dt.date,
    target_index: pd.DatetimeIndex,
) -> tuple[pd.Series, int]:
    """Obtiene temperatura horaria (2m) de Bogotá en Open-Meteo alineada al índice objetivo."""
    params = {
        "latitude": BOGOTA_LATITUDE,
        "longitude": BOGOTA_LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m",
        "timezone": BOGOTA_TIMEZONE,
    }
    LOGGER.info("Consultando Open-Meteo Bogotá (%s a %s)", start_date, end_date)
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly", {})
    if "time" not in hourly or "temperature_2m" not in hourly:
        raise ValueError("Respuesta de Open-Meteo sin campos 'hourly.time' o 'hourly.temperature_2m'.")

    temp_series = pd.Series(
        data=pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
        index=pd.to_datetime(hourly["time"]),
        name="temperatura_c",
    ).sort_index()
    temp_series = temp_series[~temp_series.index.duplicated(keep="last")]
    aligned = temp_series.reindex(target_index)

    gap_count = int(aligned.isna().sum())
    if gap_count > 0:
        LOGGER.warning("Se detectaron %d huecos en temperatura Open-Meteo; interpolación temporal.", gap_count)
        aligned = aligned.interpolate(method="time").bfill().ffill()

    return aligned, gap_count


def compute_quality_report(df: pd.DataFrame, cleaning_stats: dict[str, int]) -> dict[str, Any]:
    """Genera resumen de control de calidad: rango, conteos, nulos y valores atípicos (IQR)."""
    q1 = float(df["demanda_mwh"].quantile(0.25))
    q3 = float(df["demanda_mwh"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (df["demanda_mwh"] < lower_bound) | (df["demanda_mwh"] > upper_bound)

    time_diffs = df.index.to_series().diff().dropna()
    is_regular_hourly = bool((time_diffs == pd.Timedelta(hours=1)).all())

    return {
        "n_observaciones": int(len(df)),
        "fecha_inicio": df.index.min().isoformat(),
        "fecha_fin": df.index.max().isoformat(),
        "indice_horario_regular": is_regular_hourly,
        "nulos_demanda": int(df["demanda_mwh"].isna().sum()),
        "nulos_temperatura": int(df["temperatura_c"].isna().sum()),
        "demanda_min_mwh": round(float(df["demanda_mwh"].min()), 3),
        "demanda_max_mwh": round(float(df["demanda_mwh"].max()), 3),
        "demanda_media_mwh": round(float(df["demanda_mwh"].mean()), 3),
        "temperatura_min_c": round(float(df["temperatura_c"].min()), 2),
        "temperatura_max_c": round(float(df["temperatura_c"].max()), 2),
        "temperatura_media_c": round(float(df["temperatura_c"].mean()), 2),
        "atipicos_iqr_demanda": int(outlier_mask.sum()),
        "limites_iqr_demanda": [round(lower_bound, 2), round(upper_bound, 2)],
        **cleaning_stats,
    }


def save_overview_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Genera y guarda la figura de serie horaria de demanda eléctrica y temperatura."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    axes[0].plot(df.index, df["demanda_mwh"], color="#1f4e79", linewidth=0.9)
    axes[0].set_title(
        f"Demanda comercial horaria SIN Colombia (XM) — {df.index.min().date()} a {df.index.max().date()}"
    )
    axes[0].set_ylabel("Demanda (MWh)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["temperatura_c"], color="#c55a11", linewidth=0.9)
    axes[1].set_title("Temperatura horaria Bogotá (Open-Meteo, 2m)")
    axes[1].set_ylabel("Temperatura (°C)")
    axes[1].set_xlabel("Fecha y hora (America/Bogota)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_ingestion(
    start_date: dt.date,
    end_date: dt.date,
    output_csv: Path = RAW_CSV_PATH,
    output_figure: Path = FIGURES_DIR / "serie_demanda_temperatura.png",
    report_path: Path = QUALITY_REPORT_PATH,
) -> dict[str, Any]:
    """Orquesta la ingesta consolidada de XM y Open-Meteo y guarda los artefactos."""
    raw_xm = fetch_xm_demand_raw(start_date, end_date)
    demand_series, xm_stats = transform_xm_wide_to_hourly(raw_xm)

    actual_start = demand_series.index.min().date()
    actual_end = demand_series.index.max().date()
    temp_series, temp_gaps = fetch_open_meteo_temperature(
        actual_start, actual_end, demand_series.index
    )

    consolidated = pd.DataFrame(
        {
            "demanda_mwh": demand_series,
            "temperatura_c": temp_series,
        },
        index=demand_series.index,
    )
    consolidated.index.name = "timestamp"

    cleaning_stats = {**xm_stats, "huecos_temperatura_imputados": temp_gaps}
    report = compute_quality_report(consolidated, cleaning_stats)

    if report["nulos_demanda"] > 0 or report["nulos_temperatura"] > 0:
        raise ValueError("El dataset consolidado aún contiene valores nulos.")
    if not report["indice_horario_regular"]:
        raise ValueError("El índice resultante no es horario regular.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(output_csv, index=True)
    save_overview_plot(consolidated, output_figure)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Ingesta completada: %d filas (%s -> %s) guardadas en %s",
        report["n_observaciones"],
        report["fecha_inicio"],
        report["fecha_fin"],
        output_csv,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Define los argumentos de línea de comandos para la ingesta."""
    parser = argparse.ArgumentParser(description="Ingesta horaria real de XM y Open-Meteo.")
    parser.add_argument("--start-date", required=True, help="Fecha inicial YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Fecha final YYYY-MM-DD.")
    parser.add_argument("--output-csv", default=str(RAW_CSV_PATH), help="Ruta de salida del CSV.")
    parser.add_argument(
        "--output-figure",
        default=str(FIGURES_DIR / "serie_demanda_temperatura.png"),
        help="Ruta de salida de la figura.",
    )
    return parser


def main() -> None:
    """Punto de entrada CLI para ejecutar la ingesta."""
    args = build_arg_parser().parse_args()
    report = run_ingestion(
        start_date=parse_iso_date(args.start_date),
        end_date=parse_iso_date(args.end_date),
        output_csv=Path(args.output_csv),
        output_figure=Path(args.output_figure),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
