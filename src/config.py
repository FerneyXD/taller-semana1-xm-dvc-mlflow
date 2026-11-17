"""Configuración centralizada, rutas y utilidades de trazabilidad del proyecto."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_CSV_PATH = DATA_DIR / "xm_demanda.csv"
RAW_DVC_PATH = DATA_DIR / "xm_demanda.csv.dvc"
TRAIN_CSV_PATH = PROCESSED_DIR / "train.csv"
TEST_CSV_PATH = PROCESSED_DIR / "test.csv"
PREDICTIONS_CSV_PATH = PROCESSED_DIR / "predicciones_test.csv"

PARAMS_PATH = PROJECT_ROOT / "params.yaml"
METRICS_PATH = PROJECT_ROOT / "metrics.json"
QUALITY_REPORT_PATH = DATA_DIR / "calidad_ingesta.json"

BOGOTA_LATITUDE = 4.6097
BOGOTA_LONGITUDE = -74.0817
BOGOTA_TIMEZONE = "America/Bogota"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
KWH_TO_MWH_FACTOR = 1000.0


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura el formato estándar de logs para los scripts del pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("taller_xm")


def load_params(params_path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Carga y valida el archivo de parámetros YAML del proyecto."""
    if not params_path.exists():
        raise FileNotFoundError(f"No existe el archivo de parámetros: {params_path}")
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if not isinstance(params, dict):
        raise ValueError(f"Formato inválido en {params_path}: se esperaba un diccionario YAML.")
    return params


def read_dvc_data_hash(dvc_file: Path = RAW_DVC_PATH) -> str:
    """Obtiene el hash MD5 exacto registrado en el archivo .dvc de los datos."""
    if not dvc_file.exists():
        raise FileNotFoundError(f"No se encontró el puntero DVC: {dvc_file}")
    with dvc_file.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    outs = metadata.get("outs", [])
    if not outs or "md5" not in outs[0]:
        raise ValueError(f"El archivo {dvc_file} no contiene un hash MD5 válido en 'outs'.")
    return str(outs[0]["md5"])


def get_git_commit_hash(repo_root: Path = PROJECT_ROOT) -> str:
    """Retorna el hash SHA-1 completo del commit HEAD actual de Git."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
