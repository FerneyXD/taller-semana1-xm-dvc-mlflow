# Taller Semana 1 — Pronóstico de demanda eléctrica con ingesta real, DVC y MLflow
**Electiva Maestría en IA · Series de tiempo · Sesión 3 (taller de 4 horas)**

---

## 1. Contexto y objetivo

Una empresa distribuidora necesita pronosticar la demanda eléctrica de las próximas 48 horas y, sobre todo, poder **demostrar con qué datos y con qué configuración se produjo cada pronóstico**. En las sesiones 1 y 2 exploraron la serie y propusieron un modelo SARIMAX; hoy lo convierten en un flujo reproducible de punta a punta:

* Traer la serie **real** de demanda del sistema eléctrico colombiano desde **XM** con `pydataxm` y una temperatura exógena con **Open-Meteo**.
* Versionar cada snapshot de datos con **DVC** y encadenar el trabajo en un pipeline (`dvc.yaml`).
* Ajustar varias configuraciones de SARIMAX y registrar cada run en **MLflow**, con la versión exacta de datos y de código.
* Comparar los resultados, elegir un modelo y **reproducirlo desde cero** a partir de lo registrado.

> *Prerrequisito:* Mecánica básica de DVC vista en la Sesión 2 (`init`, `add`, `push`, `checkout`). Material de apoyo: notebook `taller_intro_dvc` del repositorio del curso.

---

## 2. Qué deben entregar (Checklist)

- [ ] Un repositorio Git del taller con historial de commits claro (los datos **no** van en Git: van en el remoto de DVC).
- [ ] Dos o más versiones de los datos de XM en el remoto de DVC, con hashes distintos (anotados en `NOTAS.md`).
- [ ] Un pipeline `dvc.yaml` con al menos dos etapas (`prepare` y `train`) y su `dvc.lock`.
- [ ] Cuatro o más runs comparables en MLflow (mismas métricas y artefactos) con los tags de trazabilidad.
- [ ] Una tabla comparativa de los runs y una conclusión de máximo 10 líneas que elija un modelo.
- [ ] La verificación del código generado con IA (Parte 6), documentada en `NOTAS.md`.
- [ ] La prueba de reproducibilidad del modelo elegido, documentada en `NOTAS.md`.

---

## 3. Cronograma (240 minutos)

| Tiempo | Parte | Minutos |
| :--- | :--- | :---: |
| 0:00 – 0:15 | 1. Arranque: repositorio, DVC y servidor MLflow | 15 |
| 0:15 – 0:45 | 2. Ingesta real de XM y temperatura | 30 |
| 0:45 – 1:10 | 3. Snapshots de datos con DVC (v1, v2) | 25 |
| 1:10 – 1:20 | *Descanso* | 10 |
| 1:20 – 1:50 | 4. Pipeline reproducible con `dvc.yaml` y baseline | 30 |
| 1:50 – 2:50 | 5. SARIMAX y runs en MLflow (mínimo 4 configuraciones) | 60 |
| 2:50 – 3:10 | 6. Verificación del código generado con IA | 20 |
| 3:10 – 3:25 | 7. Comparación de runs y elección del modelo | 15 |
| 3:25 – 3:50 | 8. Prueba de reproducibilidad | 25 |
| 3:50 – 4:00 | Cierre y verificación del checklist | 10 |
| **Total** | | **240 min** |

---

## 4. Antes de empezar

### Requisitos del entorno:
* Python 3.10 o superior con `pandas<3` (obligatorio: `pydataxm` falla con pandas 3.0), `statsmodels`, `matplotlib`, `pydataxm`, `requests`, `mlflow`, `dvc`, `pyyaml`.
* Git con nombre y correo configurados (`git config user.name`, `git config user.email`).
* Acceso a internet hacia las APIs de datos (XM y Open-Meteo).
* Notebooks de referencia de las sesiones 1 y 2 (orden candidato $(p,d,q)(P,D,Q)_{24}$ y funciones de diagnóstico).

### Convenciones del taller:
* **Experimento MLflow:** `s1_taller_xm`
* **Nombre de cada run:** `sarimax_{exog|sinexog}_pdq_PDQs` (ejemplo: `sarimax_exog_210_111_24`)
* **Métricas (sobre prueba):** `rmse_test`, `mae_test`, `mape_test`; del ajuste: `aic`, `bic`
* **Tags de trazabilidad:** `dvc_data_hash` (MD5 de la versión de datos), `git_commit`, `fuente`
* **Mensajes de commit:** Describen el cambio (ej. `"v2: datos hasta T1"` o `"run: con exógena, orden B"`)

---

## 5. Uso de IA generativa
* Lo que importa es que el resultado sea **correcto, verificado y entendido**.
* Reportar solo lo que haya salido de la propia ejecución real (métricas, hashes y figuras inventadas no son evidencia).
* Las predicciones «antes de ejecutar», explicaciones escritas y la conclusión de la Parte 7 deben resolverse y redactarse conscientemente.
* Validar que las funciones de librerías invocadas existan realmente en las versiones instaladas.
* No versionar ni filtrar credenciales o rutas privadas.

---

## 6. Desarrollo detallado por partes

### Parte 1 · Arranque (15 min)
*Objetivo:* Dejar listo el andamiaje del taller.
1. Crear repositorio Git nuevo, inicializar DVC y definir remoto DVC (carpeta local fuera del repo o bucket).
2. Levantar servidor MLflow local con backend SQLite `sqlite:///mlflow.db` y verificar interfaz web. *(Nota: backend de archivos `file:./mlruns` está en modo mantenimiento en MLflow 3.x)*.
3. Crear estructura de carpetas (`data/`, `src/`, `params.yaml`, `NOTAS.md`) y realizar el commit inicial.
* **Criterio Listo cuando:** `git log` muestra el commit inicial, `dvc remote list` muestra el remoto y la interfaz de MLflow carga correctamente.

### Parte 2 · Ingesta real de XM y temperatura (30 min)
*Objetivo:* Construir el snapshot inicial (v1) de la serie ancla con su exógena.
1. Con `pydataxm`, consultar demanda comercial horaria del sistema (`ReadDB().request_data("DemaCome", "Sistema", inicio, fin)`) de los últimos 60 a 90 días cerrados. Anotar instante $T_0$ del último dato disponible.
2. Construir serie con índice horario regular: ordenada, sin duplicados y con huecos tratados/reportados.
3. Control de calidad breve: número de observaciones, rango de fechas, nulos y valores atípicos + gráfica de la serie.
4. Con Open-Meteo, obtener temperatura horaria de una ciudad representativa (ej. Bogotá o promedio) alineada al mismo índice.
5. Guardar tabla consolidada `xm_demanda.csv` con demanda (convertida a MWh) y temperatura.
* *Pistas:* XM entrega una fila por día y horas en columnas (pasar a formato largo). XM publica con rezago (~3-4 días hacia atrás). Si la API de XM falla persistentemente (+10 min), Plan B: usar serie PJM y aclararlo en `NOTAS.md`.
* **Criterio Listo cuando:** El CSV tiene índice horario regular, sin nulos en demanda ni temperatura y la gráfica está guardada.

### Parte 3 · Snapshots de datos con DVC (25 min)
*Objetivo:* Versionar dos estados de los datos y poder alternar entre ellos.
1. Versionar CSV como v1: `dvc add data/xm_demanda.csv`, commit del archivo `.dvc` en Git y `dvc push`.
2. Simular polling: consultar XM para traer datos más recientes que $T_0$ (o ampliar ventana 3 días atrás). Regenerar CSV y versionar como v2 (`dvc add`, commit en Git, `dvc push`).
3. Ejecutar `dvc diff` entre commits de v1 y v2; interpretar cambios (hash, tamaño, filas).
4. Viajar en el tiempo: volver a v1 (`git checkout` + `dvc checkout`), comprobar filas y regresar a v2.
5. Anotar en `NOTAS.md` el hash MD5 de cada versión.
* **Criterio Listo cuando:** `git log` muestra al menos dos commits de datos con hashes distintos y ambas versiones están en el remoto DVC.

### Parte 4 · Pipeline reproducible y baseline (30 min)
*Objetivo:* Encadenar preparación de datos y entrenamiento para que DVC decida qué recalcular.
1. Crear `params.yaml` con: días de entrenamiento, horas de prueba (ej. 48), orden $(p,d,q)$, orden estacional $(P,D,Q,s)$ y flag `usar_exogena`.
2. Declarar en `dvc.yaml` dos etapas:
   - `prepare`: lee CSV versionado y hace split temporal train/test sin mezclar fechas.
   - `train`: baseline naive estacional (predice valor de hace 24 horas) y genera `metrics.json`.
3. Ejecutar `dvc repro` dos veces; explicar por escrito por qué la segunda no ejecuta nada.
4. Predecir antes de ejecutar qué etapas se recalculan si cambia: (a) parámetro que solo usa `train`, (b) parámetro que usa `prepare`. Verificar con `dvc repro` y `dvc dag`.
5. Commit de `params.yaml`, `dvc.yaml` y `dvc.lock`.
* **Criterio Listo cuando:** `dvc dag` muestra las etapas, las predicciones coinciden con lo observado y existe el RMSE del baseline.

### Parte 5 · SARIMAX y runs en MLflow (60 min)
*Objetivo:* Reemplazar baseline por SARIMAX y registrar cada configuración de forma comparable.
1. Modificar etapa `train` para ajustar SARIMAX según `params.yaml` (con/sin exógena) y pronosticar 48 horas de prueba (supuesto de temperatura observada declarado).
2. Cada ejecución de `train` debe registrar en MLflow:
   - **Parámetros:** órdenes, exógena, ventana, tamaños train/test.
   - **Métricas:** `rmse_test`, `mae_test`, `mape_test`, `aic`, `bic`.
   - **Artefactos:** gráfica pronóstico vs real con intervalo de predicción, gráfica diagnóstico de residuales (`plot_diagnostics`).
   - **Tags:** `dvc_data_hash`, `git_commit`, `fuente`.
   - **Dataset:** adjunto con `mlflow.log_input`.
3. Ejecutar con `dvc repro` cambiando solo `params.yaml`, mínimo 4 configuraciones (cada una con su commit):
   - (a) Orden candidato Sesión 1 sin exógena.
   - (b) Orden candidato Sesión 1 con exógena.
   - (c) Orden alternativo sin exógena.
   - (d) Orden alternativo con exógena.
4. Extensión opcional: repetir configuración (a) sobre datos v1 y v2, comparando métricas y `dvc_data_hash`.
* **Criterio Listo cuando:** El experimento tiene al menos 4 runs comparables en MLflow con métricas, artefactos, tags y dataset registrado.

### Parte 6 · Verificación del código generado con IA (20 min)
*Objetivo:* Auditar el pipeline antes de extraer conclusiones.
1. Comprobar que no hay fuga de información (split temporal estricto sin usar test en imputación/escalado).
2. Recalcular a mano con NumPy el RMSE de un run y contrastarlo con MLflow.
3. Verificar trazabilidad (`dvc_data_hash` coincide con el archivo `.dvc` del commit y `git_commit` es el hash real de Git).
4. Verificar funciones, argumentos, supuesto de temperatura y alineación horaria de XM.
5. Anotar en `NOTAS.md` al menos 3 verificaciones con evidencia.
* **Criterio Listo cuando:** `NOTAS.md` tiene al menos 3 verificaciones detalladas con evidencia.

### Parte 7 · Comparación de runs y elección del modelo (15 min)
*Objetivo:* Decidir con evidencia la mejor configuración.
1. En MLflow UI, comparar runs ordenados por `rmse_test`.
2. Desde Python con `mlflow.search_runs`, exportar `comparacion_runs.csv`.
3. Redactar conclusión (máximo 10 líneas) en `NOTAS.md` respondiendo: ¿Mejora la exógena y cuánto frente al baseline? ¿El menor AIC/BIC coincide con el menor error de prueba? ¿Qué modelo se elige y por qué?
* **Criterio Listo cuando:** Existe `comparacion_runs.csv`, modelo elegido y conclusión en `NOTAS.md`.

### Parte 8 · Prueba de reproducibilidad (25 min)
*Objetivo:* Demostrar reconstrucción total a partir del registro.
1. Identificar `git_commit` y `dvc_data_hash` del modelo ganador.
2. Borrar datos locales y caché de DVC.
3. Recuperar código con `git checkout <commit>` y datos con `dvc pull`.
4. Reejecutar pipeline y verificar que las métricas coincidan exactamente (o con margen numérico mínimo).
5. Documentar procedimiento y evidencia en `NOTAS.md`.
* **Criterio Listo cuando:** Las métricas reproducidas coinciden con las del run original y el checklist final de entrega está completo.
