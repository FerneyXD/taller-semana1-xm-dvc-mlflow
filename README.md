# Taller Semana 1 — Pronóstico de demanda eléctrica XM con DVC y MLflow

Flujo reproducible de punta a punta para pronosticar la demanda comercial horaria del Sistema Interconectado Nacional (SIN) de Colombia (XM `DemaCome`) con temperatura exógena de Bogotá (Open-Meteo), versionamiento de datos en **DVC (remoto AWS S3)** y trazabilidad de experimentos en **MLflow (backend SQLite)**.

## Estructura del repositorio

```text
├── data/
│   ├── xm_demanda.csv.dvc          # Puntero DVC al snapshot activo en S3 (sin datos en Git)
│   └── calidad_ingesta.json        # Reporte de control de calidad de la ingesta
├── figures/
│   ├── serie_demanda_temperatura.png
│   ├── pronostico_vs_real.png      # Pronóstico 48h vs real del modelo elegido
│   └── diagnostico_residuales.png  # Diagnóstico de residuales del modelo elegido
├── src/
│   ├── config.py                   # Rutas, parámetros y helpers de trazabilidad Git/DVC
│   ├── ingest.py                   # Ingesta real XM (pydataxm) + Open-Meteo
│   ├── prepare.py                  # Etapa DVC: split temporal estricto train/test
│   ├── train.py                    # Etapa DVC: baseline/SARIMAX + registro en MLflow
│   ├── verify.py                   # Auditoría automatizada (Parte 6)
│   └── compare.py                  # Exportación de tabla comparativa MLflow (Parte 7)
├── dvc.yaml                        # Pipeline reproducible (prepare -> train)
├── dvc.lock                        # Estado bloqueado de dependencias y salidas
├── params.yaml                     # Hiperparámetros del pipeline
├── metrics.json                    # Métricas de la última ejecución
├── comparacion_runs.csv            # Comparativa exportada desde MLflow
├── NOTAS.md                        # Documentación completa de entregables (Partes 1–8)
└── requirements.txt                # Dependencias fijadas (Python 3.12, pandas<3)
```

## Reproducción rápida

```bash
# 1. Crear entorno e instalar dependencias
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 2. Levantar servidor MLflow local (SQLite)
.venv/Scripts/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

# 3. Descargar datos desde remoto DVC (AWS S3) y reproducir pipeline
.venv/Scripts/dvc pull
.venv/Scripts/dvc repro
```

Consulte [NOTAS.md](NOTAS.md) para los hashes de datos, resultados comparativos, auditoría técnica y prueba de reproducibilidad.
