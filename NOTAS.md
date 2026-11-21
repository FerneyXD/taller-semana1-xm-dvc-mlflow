# NOTAS.md — Taller Semana 1: Pronóstico de demanda eléctrica con XM, DVC y MLflow

---

## 1. Entorno y configuración inicial (Parte 1 — Desplegado en AWS)

- **Repositorio Git:** rama `main` (`user.name = FerneyXD`, `user.email = ferney.guillen01@usa.edu.co`)
- **Python y dependencias clave:** Python `3.12.13`, `pandas==2.3.3` (`<3` obligatorio para `pydataxm`), `pydataxm==0.3.19`, `statsmodels==0.15.0`, `mlflow==3.18.0`, `dvc==3.67.1`, `dvc-s3==3.3.0`, `numpy==2.5.5`, `matplotlib==3.11.2`.
- **Remoto DVC (AWS S3):** `aws_s3 -> s3://dvc-taller-xm-509780086678/dvcstore` (región `us-east-1`, cuenta AWS `509780086678`).
- **Servidor MLflow en AWS EC2:** `http://3.93.69.202` (IP pública dinámica sin Elastic IP para evitar costos en reposo; Nginx puerto `80` con compresión gzip -> MLflow `:5000`, CORS habilitado)
  - **Instancia EC2:** `i-079e351d5f0b88938` (`MLflow-Taller-XM-Server`, `t3.medium`, 4 GB RAM + 2 GB swap, Ubuntu 22.04, key pair `vockey`, rol IAM `LabInstanceProfile`, security group `sg-0e995d518d3b63532` con puertos 22, 80 y 5000).
  - **Backend Store:** `sqlite:////opt/mlflow/mlflow.db` gestionado por `systemd` (`mlflow-server.service`, `--workers 1`, `--allowed-hosts "*"`, `--cors-allowed-origins "*"`).
  - **Artifact Store (AWS S3):** `s3://dvc-taller-xm-509780086678/mlflow-artifacts` (`--serve-artifacts`, 32 archivos verificados en S3).
- **Experimento MLflow:** `s1_taller_xm` (`http://3.93.69.202/#/experiments/1`).
- **Seguridad:** Ninguna credencial ni llave privada (`*.pem`, `.env`, `mlflow.db`) fue versionada en Git (protegidas en `.gitignore`).

---

## 2. Ingesta real de XM y temperatura + control de calidad (Parte 2)

- **Fuente demanda:** `pydataxm.ReadDB().request_data("DemaCome", "Sistema", inicio, fin)` en bloques de 25 días. XM entrega filas diarias con columnas `Values_Hour01..Values_Hour24` en **kWh**; se transformó a formato largo con `Values_Hour01 -> 00:00` hasta `Values_Hour24 -> 23:00` y se dividió por `1000` para expresar en **MWh**.
- **Fuente exógena:** Open-Meteo Archive API (`temperature_2m` horaria para Bogotá: lat `4.6097`, lon `-74.0817`, zona horaria `America/Bogota`), alineada exactamente al índice horario de XM.
- **Hallazgo real de control de calidad:** Al hacer polling hasta `2026-11-14` (`v2`), el control de calidad detectó **48 valores atípicos IQR** (demanda media `~2,350 MWh` el 13 y 14 de noviembre vs `~10,970 MWh` en días cerrados). Esto corresponde a días en **liquidación preliminar incompleta** en XM (rezago operativo de ~3–4 días). Por ello se construyó `v3` únicamente con **días cerrados completos** (`2026-08-26` a `2026-11-12`):
  - **Observaciones:** `1,896` horas (79 días cerrados)
  - **Índice horario regular (`1h`):** `True` · **Duplicados:** `0` · **Huecos imputados:** `0`
  - **Nulos:** `0` en demanda, `0` en temperatura · **Atípicos IQR:** `0`
  - **Demanda (MWh):** min `7,565.90`, media `10,320.10`, máx `12,832.27`
  - **Temperatura Bogotá (°C):** min `9.00`, media `15.70`, máx `24.00`
  - **Gráfica guardada:** `figures/serie_demanda_temperatura.png`

---

## 3. Snapshots de datos con DVC (Parte 3)

Se versionaron **tres snapshots reales** de `data/xm_demanda.csv` y los tres se encuentran almacenados en el remoto de Amazon S3 (`s3://dvc-taller-xm-509780086678/dvcstore`):

| Versión | Commit Git | Rango horario | Filas | Tamaño (bytes) | Hash MD5 (DVC) | Observación |
| :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **v1** | `776869a` | `2026-08-29 00:00` → `2026-11-11 23:00` ($T_0$) | 1,800 | 70,871 | `4eb914f51a1e0a4bf96a2267544c6c57` | Snapshot inicial (75 días cerrados, 0 atípicos) |
| **v2** | `1d3223c` | `2026-08-29 00:00` → `2026-11-14 23:00` ($T_1$) | 1,872 | 73,691 | `ad10044a870463f9987a5c0a522a72ea` | Polling +72h reales XM; reveló 48h preliminares incompletas |
| **v3** | `86c1238` | `2026-08-26 00:00` → `2026-11-12 23:00` ($T_{cerrado}$) | 1,896 | 74,691 | `133d48931a88253b9da7d2b5878e80bb` | 79 días cerrados limpios (0 nulos, 0 atípicos); dataset activo |

### Evidencia de `dvc diff` y viaje en el tiempo
- `dvc diff 776869a 1d3223c --show-hash` → `Modified: 4eb914f5..ad10044a data\xm_demanda.csv` (+72 filas, +2,820 bytes).
- `dvc diff 776869a 86c1238 --show-hash` → `Modified: 4eb914f5..133d4893 data\xm_demanda.csv` (+96 filas, +3,820 bytes).
- **Viaje en el tiempo:**
  - `git checkout 776869a` + `dvc checkout` → `data/xm_demanda.csv` restaurado con **1,800 filas** (`2026-08-29 00:00:00` a `2026-11-11 23:00:00`).
  - `git checkout main` + `dvc checkout` → `data/xm_demanda.csv` restaurado con **1,896 filas** (`2026-08-26 00:00:00` a `2026-11-12 23:00:00`).

---

## 4. Pipeline reproducible `dvc.yaml` y baseline (Parte 4)

### Grafo del pipeline (`dvc dag`)
```text
data/xm_demanda.csv.dvc  ──▶  prepare  ──▶  train
```

### ¿Por qué el segundo `dvc repro` no ejecuta nada?
En la primera ejecución, DVC calcula y registra en `dvc.lock` los hashes MD5 de todas las dependencias (`deps`), valores de `params.yaml` usados por cada etapa y salidas (`outs`). En la segunda ejecución, DVC compara los hashes actuales contra `dvc.lock`: como ni los datos, ni el código, ni los parámetros ni las salidas cambiaron, determina que ambas etapas están vigentes (`Stage 'prepare' didn't change, skipping` / `Stage 'train' didn't change, skipping`) y evita cómputo redundante.

### Predicción antes de ejecutar vs resultado observado
- **(a) Cambiar un parámetro exclusivo de `train` (`train.usar_exogena: false -> true`):**
  - *Predicción:* `prepare` no depende de `train.usar_exogena`, así que se omite; solo se recalcula `train`.
  - *Observado:* `Stage 'prepare' didn't change, skipping` → `Running stage 'train'` (**coincide 100%**).
- **(b) Cambiar un parámetro de `prepare` (`prepare.dias_entrenamiento: 45 -> 40`):**
  - *Predicción:* se recalcula `prepare` (genera nuevos `train.csv` / `test.csv`), y al cambiar el hash de esas dependencias también se invalida y recalcula `train`.
  - *Observado:* `Running stage 'prepare'` (`n_train=960`) → `Running stage 'train'` (**coincide 100%**).

### Resultado del Baseline naive estacional (24h, commit `b0683f7`)
- Horizonte de prueba: 48h (`2026-11-11 00:00` a `2026-11-12 23:00`, repitiendo el último ciclo observado de 24h sin fuga).
- **`rmse_test`:** `170.6939 MWh` · **`mae_test`:** `149.9995 MWh` · **`mape_test`:** `1.4082 %`

---

## 5. Comparación de runs en MLflow AWS (Partes 5 y 7)

Tabla exportada desde MLflow en AWS EC2 (`http://3.93.69.202`) mediante `src/compare.py` (`mlflow.search_runs(..., order_by=["metrics.rmse_test ASC"])`) en `comparacion_runs.csv` (incluye las 6 configuraciones sobre `v3` y la extensión opcional de la Parte 5.4 sobre `v1`, con todos los artefactos persistidos en `s3://dvc-taller-xm-509780086678/mlflow-artifacts/`):

| # | Run Name | Dataset | Exógena | `rmse_test` (MWh) | `mae_test` (MWh) | `mape_test` (%) | AIC | BIC | `dvc_data_hash` | `git_commit` |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| — | *Baseline naive 24h* | v3 | No | *170.69* | *150.00* | *1.41* | — | — | `133d4893...` | `b0683f7` |
| 1 | **`sarimax_sinexog_210_111_24`** | **v3** | **No** | **270.50** | **216.34** | **1.88** | **12,014.04** | **12,038.72** | `133d4893...` | `662d55d` |
| 2 | `sarimax_exog_210_111_24` | v3 | Sí | 271.92 | 217.90 | 1.89 | 12,015.79 | 12,045.41 | `133d4893...` | `8644190` |
| 3 | `sarimax_sinexog_210_111_24` *(ext. v1)* | v1 | No | 274.06 | 227.93 | 2.06 | 11,979.38 | 12,004.06 | `4eb914f5...` | `6f41a8d` |
| 4 | `sarimax_sinexog_111_110_24` | v3 | No | 359.16 | 313.30 | 2.92 | 12,298.29 | 12,318.04 | `133d4893...` | `bdebc7d` |
| 5 | `sarimax_exog_111_110_24` | v3 | Sí | 359.48 | 313.40 | 2.92 | 12,300.19 | 12,324.88 | `133d4893...` | `d92cacf` |
| 6 | `sarimax_sinexog_101_111_24` | v3 | No | 461.75 | 405.61 | 3.57 | 12,096.59 | 12,121.27 | `133d4893...` | `940fcb0` |
| 7 | `sarimax_exog_101_111_24` | v3 | Sí | 463.35 | 407.48 | 3.59 | 12,097.89 | 12,127.51 | `133d4893...` | `dc7cd52` |

*(Supuesto declarado para runs con exógena: se utilizó temperatura horaria observada ex-post de Bogotá en el horizonte de prueba de 48h.)*

---

## 6. Conclusión y elección del modelo (Parte 7 — máximo 10 líneas)

1. La temperatura de Bogotá **no mejora** el pronóstico en ninguno de los tres órdenes probados (+1.42, +0.32 y +1.60 MWh de RMSE frente a su par sin exógena), porque una sola ciudad fría (~15.7 °C) explica poco de la demanda nacional agregada del SIN.
2. Frente al **baseline naive 24h** (`170.69 MWh`), ningún SARIMAX con estacionalidad única $s=24$ logró mejorarlo (el mejor SARIMAX obtiene `270.50 MWh`, +99.81 MWh / +58.5% más error), debido a que el horizonte de prueba cayó en dos días laborales consecutivos (miércoles–jueves) muy similares al martes previo, mientras SARIMAX($s=24$) arrastra el efecto de fines de semana sin modelar el ciclo semanal ($s=168$).
3. El menor AIC/BIC **no siempre garantiza** menor error en prueba: aunque el ganador coincide en menor AIC/BIC (`12,014.04` / `12,038.72`) y menor `rmse_test`, el orden `(1,0,1)(1,1,1,24)` logra mejor AIC/BIC (`12,096.59`) que `(1,1,1)(1,1,0,24)` (`12,298.29`) pero produce un RMSE mucho mayor (`461.75` vs `359.16 MWh`).
4. **Modelo elegido:** entre las configuraciones SARIMAX se selecciona **`sarimax_sinexog_210_111_24`** (`662d55d`, `dvc_data_hash = 133d48931a88253b9da7d2b5878e80bb`) por alcanzar el menor RMSE (`270.50 MWh`), MAE (`216.34 MWh`), MAPE (`1.88%`) y AIC/BIC, y por no requerir pronósticos futuros de temperatura en operación.
5. Como siguiente iteración para superar al baseline naive 24h se recomienda incorporar dummies de tipo de día (laboral/sábado/domingo/festivo) o estacionalidad múltiple ($s=24, 168$) y temperatura ponderada de ciudades cálidas (Barranquilla, Cali, Medellín).

---

## 7. Verificación del código generado con IA (Parte 6)

Auditoria automatizada ejecutada con `python -m src.verify` (`all_passed: true`):

1. **Verificación de ausencia de fuga de información (`split_temporal_sin_fuga`: PASS):**
   - `train`: `2026-09-27T00:00:00` a `2026-11-10T23:00:00` (`1,080` filas).
   - `test`: `2026-11-11T00:00:00` a `2026-11-12T23:00:00` (`48` filas).
   - Brecha exacta entre el fin de `train` y el inicio de `test`: **`1.0` hora**, sin solapamiento, frecuencia constante de 3,600 s y cero imputaciones usando estadísticas de prueba.
2. **Recálculo manual con NumPy del RMSE frente a MLflow (`recalculo_manual_rmse_numpy_vs_mlflow`: PASS):**
   - Run auditado: `sarimax_sinexog_210_111_24` (`run_id = f478194418334ff8a2ee3da1b9b8109f`), descargando su artefacto `predicciones_test.csv` directamente desde MLflow AWS / S3.
   - `mlflow_rmse = 270.5018010439814` vs `manual_numpy_rmse = 270.5018010439815` → **diferencia absoluta = `1.14e-13`** (precisión de punto flotante). MAE (`216.337863`) y MAPE (`1.877651%`) también coinciden exactamente.
3. **Verificación de trazabilidad Git + DVC (`trazabilidad_git_y_dvc`: PASS — 7/7 runs):**
   - Para los 7 runs de `s1_taller_xm`, se ejecutó `git show <git_commit>:data/xm_demanda.csv.dvc` y se comprobó que el commit existe en Git y que su `md5` coincide exactamente con el tag `dvc_data_hash` en MLflow (`133d48931a88253b9da7d2b5878e80bb` en los 6 runs de v3 y `4eb914f51a1e0a4bf96a2267544c6c57` en el run de extensión v1).
4. **Verificación de dominio XM, APIs y supuestos (`dominio_xm_unidades_y_versiones`: PASS):**
   - `DemaCome` del SIN colombiano llega en kWh (`~10.3 millones kWh/h`) y tras dividir entre 1,000 queda en rango físico real (`10,320.10 MWh` promedio).
   - Alineación horaria verificada con las 24 horas `0..23`: pico nacional en la **hora 19** (19:00–20:00 COT) y valle en la **hora 3** (03:00–04:00 COT), consistente con la curva típica colombiana.
   - Se verificó la existencia real de las APIs usadas (`pydataxm.ReadDB.request_data`, `SARIMAX.get_forecast`, `mlflow.data.from_pandas`, `mlflow.log_input`, `mlflow.search_runs`) en las versiones instaladas (`pandas 2.3.3`, `statsmodels 0.15.0`, `mlflow 3.18.0`, `dvc 3.67.1`). Durante la ejecución se corrigió el esquema `source` de `mlflow.data.from_pandas` porque MLflow 3.18 no resuelve el esquema `dvc://` de forma nativa.

---

## 8. Prueba de reproducibilidad del modelo ganador (Parte 8 — AWS EC2 + S3)

- **Modelo objetivo:** `sarimax_sinexog_210_111_24`
- **Run original MLflow AWS:** `f478194418334ff8a2ee3da1b9b8109f`
- **Trazabilidad objetivo:** `git_commit = 662d55d51216fc57283138fa3326d59783131fec` · `dvc_data_hash = 133d48931a88253b9da7d2b5878e80bb`

### Procedimiento ejecutado desde cero
1. Borrado total de `data/xm_demanda.csv`, `data/processed/*` y `.dvc/cache/`.
2. Recuperación de código y parámetros exactos: `git checkout 662d55d51216fc57283138fa3326d59783131fec`.
3. Descarga de datos desde Amazon S3: `dvc pull data/xm_demanda.csv.dvc` → `1 file fetched and 1 file added`; MD5 verificado en disco: `133d48931a88253b9da7d2b5878e80bb`.
4. Reejecución completa de `prepare` y `train`: `dvc repro -f` → nuevo run MLflow AWS `537e7bb77e114b31827e03d20192ce16` (etiquetado `reproducibility_check=true`).
5. Retorno a la rama principal: `git checkout main` y restauración de caché con `dvc pull`.

### Comparación de métricas: Original vs Reproducido

| Métrica | Run Original (`f4781944...`) | Run Reproducido (`537e7bb7...`) | Diferencia absoluta |
| :--- | :---: | :---: | :---: |
| `rmse_test` (MWh) | `270.5018010439814` | `270.5018010439814` | `0.0` |
| `mae_test` (MWh) | `216.3378630008841` | `216.3378630008841` | `0.0` |
| `mape_test` (%) | `1.8776509158332` | `1.8776509158332` | `0.0` |
| `aic` | `12014.036251282108` | `12014.036251282108` | `0.0` |
| `bic` | `12038.717964961279` | `12038.717964961279` | `0.0` |

**Resultado:** Reproducibilidad **bit a bit exacta** (`diferencia = 0.0` en todas las métricas) reconstruyendo desde Git + Amazon S3 contra MLflow en AWS EC2 (`http://3.93.69.202`).

---

## 9. Checklist final de entrega

- [x] Repositorio Git con historial de commits claro y atómico (datos fuera de Git, en remoto DVC S3).
- [x] Tres versiones de los datos de XM en el remoto DVC S3 con hashes distintos documentados arriba (`v1`, `v2`, `v3`).
- [x] Pipeline `dvc.yaml` con etapas `prepare` y `train` y su `dvc.lock`.
- [x] 7 runs comparables en MLflow (`s1_taller_xm`, +1 run de verificación de reproducibilidad) con métricas, artefactos, datasets y tags de trazabilidad.
- [x] Tabla comparativa exportada en `comparacion_runs.csv` y conclusión de máximo 10 líneas eligiendo el modelo.
- [x] Verificación del código generado con IA (Parte 6) documentada con evidencia en la sección 7.
- [x] Prueba de reproducibilidad (Parte 8) ejecutada desde cero y documentada con evidencia en la sección 8.
