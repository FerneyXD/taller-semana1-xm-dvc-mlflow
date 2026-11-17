# Taller Semana 1 — Pronóstico de demanda eléctrica XM con DVC y MLflow en AWS

Flujo reproducible de punta a punta para pronosticar la demanda comercial horaria del Sistema Interconectado Nacional (SIN) de Colombia (XM `DemaCome`) con temperatura exógena de Bogotá (Open-Meteo), versionamiento de datos en **DVC (remoto AWS S3)** y trazabilidad de experimentos en **MLflow desplegado en AWS EC2 (`t3.medium`) con artefactos en S3**.

## Infraestructura en AWS (`us-east-1`, cuenta `509780086678`)

- **Servidor MLflow UI / API (IP Elástica fija):** `http://54.243.189.124` (`eipalloc-02c85de3f78109a57`, puerto `80` con Nginx + gzip -> MLflow `:5000`; instancia `i-079e351d5f0b88938`, `t3.medium`, 4 GB RAM + 2 GB swap)
- **Experimento MLflow:** `http://54.243.189.124/#/experiments/1` (`s1_taller_xm`)
- **Artefactos MLflow (S3):** `s3://dvc-taller-xm-509780086678/mlflow-artifacts/`
- **Remoto DVC (S3):** `s3://dvc-taller-xm-509780086678/dvcstore/`

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

# 2. Verificar servidor MLflow desplegado en AWS EC2
curl http://54.243.189.124/health

# 3. Descargar datos desde remoto DVC (AWS S3) y reproducir pipeline
.venv/Scripts/dvc pull
.venv/Scripts/dvc repro
```

Consulte [NOTAS.md](NOTAS.md) para los hashes de datos, resultados comparativos, auditoría técnica y prueba de reproducibilidad.
