# F2.1 — Techo 5D

Especificación: `docs/protocolo_preregistrado.md` §4.3 (congelado, etiqueta `protocolo-v1`) y `configs/protocolo.json` → `techo_5d`. Formato de las predicciones: `docs/especificacion_predicciones.md`. **No se calculó ninguna métrica de prueba** (protocolo §10): las predicciones de test quedan archivadas hasta la Fase 5.

## Commits

| Momento | Commit | Fecha (−06:00) | Contenido |
|---|---|---|---|
| Scripts, antes de ejecutar | `eea2ae1` | 2026-11-03 18:40 | `f21_techo5d.py`, `f21_fragment.py`. **Es el código con el que se generaron las predicciones** (`git_commit` de los 5 manifiestos, árbol limpio) |
| Resultados | el commit que introduce este README (`[F2.1] Predicciones, tablas y cifras canónicas del techo 5D`) | 2026-11-04 | Predicciones, tablas, fragmento canónico, `canonical_numbers.json`, este README. La corrida es del 2026-11-03 (`created_at` de los manifiestos) |

Las precisiones operativas que el protocolo no fija (orden de la rejilla, `max_iter=5000` de LogReg, valores por defecto de scikit-learn, claves `ref.techo5d.*` sin `supersedes`) constan en los docstrings de `f21_techo5d.py` y `f21_fragment.py` en `eea2ae1`, anterior a la ejecución.

Antes de la corrida completa se hizo una estimación acotada (`--estimar`, una combinación por familia, sin escribir nada; 2.4 s). Esa prueba mostró F1-Val de esas cinco combinaciones, solo de validación.

## Entorno

- Windows 11 Pro 10.0.26200, solo CPU, `.venv` del repositorio. `TT_DATA_ROOT = G:\Mi unidad\tt2-insumos`.
- Python 3.13.2 · numpy 2.1.3 · pandas 2.3.3 · scikit-learn 1.6.1 · scipy 1.16.3 · joblib 1.6.0 · jsonschema 4.26.0 · pytest 8.4.2.
- Duración: `f21_techo5d.py` 31.7 s (128 ajustes); batería de pruebas 1689 s (28 min; ver `pytest.log`). Nota: la batería se estimó en ≈15 min según F2.2 y superó los 20 min de referencia de CLAUDE.md para corridas locales. Según los tiempos de `pytest.xml`, `test_C_inventario_y_alineacion` (capa C de F0.3) tomó 1153 s y `test_B_invarianza` (capa B) 525 s, de 1687 s en total.

## Insumos

| Insumo | Ruta | SHA-256 |
|---|---|---|
| `UMAP_PARAMS` (verificado contra `meta/data_manifest.csv`) | `umap_params.pkl` | `081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224` |
| Índice maestro | `meta/sample_index.csv` | `74dc778292447c66be60b35b5ec224c53b01433423923fa6d6ddef91749de0f5` |
| Manifiesto de datos | `meta/data_manifest.csv` | `3afbe63bb07fe3e6f43f45bb81edb2bff88995d45a6f05b45ad4db1c3765ec47` |
| Parámetros del protocolo (LF) | `configs/protocolo.json` | `92ef8774f10f6c7ca04abd71a6a55be6ef29852b049932fb54dcecfe7ee00e5a` (= encabezado del protocolo) |

Scripts, SHA-256 sobre los bytes con LF tal como están en git:

| Script | SHA-256 |
|---|---|
| `f21_techo5d.py` | `a2332df58ffc393d2254197eca14e795a828b737f9a890c93a6baec5ff91add4` |
| `f21_fragment.py` | `5d4e6740fa106546b673027c33c04213c87a8c03ea4d237bd42bcad628a3caa9` |
| `src/predictions/write.py` | `4a1175c5bd1349bb7ed1e9853be78b1563176e2c9bfa595bd7b9269a0870a6b1` |

## Valores que el protocolo no fija

Se decidieron antes de ejecutar y se aplican igual a todas las combinaciones:

| Punto | Valor usado |
|---|---|
| Orden de la rejilla (desempate «combinación anterior») | `itertools.product` en el orden de claves de `configs/protocolo.json`: el primer hiperparámetro varía más lento. Los valores siguen el orden del config, con `null` primero. En SVM, γ = `'scale'` va antes del logspace |
| `max_iter` de LogReg | 5000 (igual que F2.2). Ninguna combinación emitió `ConvergenceWarning` |
| Hiperparámetros no nombrados | Valores por defecto de scikit-learn 1.6.1 (p. ej. MLP `solver='adam'`, RF `n_jobs=None`) |
| Cifras canónicas | Claves nuevas `ref.techo5d.*`. Las cifras WITCOM `ref.umap5d.*` (reportadas) se conservan y no se sustituyen |

Además se aplicaron las reglas de §4.3 y de la tarea: `random_state=42` en los cinco estimadores; `StandardScaler` ajustado solo en train (dentro del pipeline) para LogReg, SVM y MLP; `SVC(probability=True)`; `HistGradientBoostingClassifier(early_stopping=False)`; `p_dep = predict_proba[:, 1]`; F1-Val de selección con `p_dep > 0.5`.

## Ejecución

Desde la raíz del repo, con `TT_DATA_ROOT` definida:

```
python results/f21_techo5d/f21_techo5d.py
python results/f21_techo5d/f21_fragment.py
python -m src.predictions.validate "results/predictions/referencia/*.csv"
CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 python -m src.canonical.build
CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 python -m src.canonical.build --check
CANON_UMAP_PARAMS=... CANON_STRICT=1 F03_UMAP_PARAMS=... F03_DATA_ROOT=<images_grayscale_classblind> \
    F03_BETO_PKL=<EMB_41> F03_GS_PATH=cpu F03_STRICT=1 python -m pytest tests/
```

## Salidas

| Archivo | Contenido |
|---|---|
| `../predictions/referencia/umap5d__{logreg,svm_rbf,rf,gbm,mlp}__none__seed42.csv` (+ manifiesto) | `p_dep` del mejor modelo de cada familia, val y test (228 filas c/u); **test sin métricas hasta la Fase 5** |
| `rejilla_completa.csv` | Las 128 combinaciones: familia, orden en la rejilla, combinación (JSON), F1-Val, `n_iter`, avisos de convergencia |
| `seleccion.csv` | Mejor combinación por familia, número de combinaciones empatadas dentro de 1e-12 y la marca `techo` |
| `canonical_fragment.json` | `ref.techo5d.{familia}.f1_macro` ×5 y `ref.techo5d.f1_macro` (fuente de `results/canonical_numbers.json`) |
| `f21_techo5d.log` | Bitácora de la corrida (F1-Val de cada combinación) |
| `canonical_check.log` | Salida de `build --check` |
| `pytest.log`, `pytest.xml` | Batería completa: 379 aprobadas, 7 omitidas (ruta GPU sin CUDA), 2 fallidas (`test_C_reproduccion_desde_u[perlin]` y `[lorenz]`, capa C de F0.3 en Windows; las mismas 2 que en F2.2) |

SHA-256 de los CSV de predicciones (`csv_sha256` del manifiesto): logreg `d92afb83…`, svm_rbf `c061b615…`, rf `12216015…`, gbm `1abd93e2…`, mlp `37d93554…`.

## Resultados (solo validación)

Fuente: `seleccion.csv` y `f21_techo5d.log`.

| Familia | Combinación seleccionada (orden) | F1-Val | Empatadas (1e-12) |
|---|---|---|---|
| LogReg | C = 0.1, `balanced` (#13) | 0.7004 | 6 |
| **SVM-RBF (techo)** | C = 1, γ = 1, `balanced` (#33) | **0.7246** | 1 |
| Random Forest | `max_depth=None`, `min_samples_leaf=1`, `balanced` (#1) | 0.6874 | 1 |
| HistGradientBoosting | lr = 0.03, `max_depth=3`, `max_iter=100` (#2) | 0.6874 | 3 |
| MLP | (16,), α = 1e-4 (#0) | 0.6933 | 2 |

- **Techo 5D: SVM-RBF, F1-Val 0.7246.** Es el único máximo, así que la elección no depende de la regla de desempate entre familias.
- **SVM seleccionado:** 3 de las 114 predicciones de val difieren entre `predict()` y `p_dep > 0.5` (F1-Val con `predict()` = 0.7185; no se usó para seleccionar). La diferencia viene de que `probability=True` calibra con Platt por validación cruzada interna, separada de la función de decisión.
- En LogReg, GBM y MLP la selección dentro de la familia sí se resolvió por el orden de la rejilla. En LogReg las 6 combinaciones empatadas son todas `balanced` con C ≥ 0.1, así que el ganador sería el mismo con el anidamiento inverso.

## Procedencia del 0.7246 de WITCOM (búsqueda de solo lectura)

- **F2.1:** 0.7246 es el **F1-Val** del SVM-RBF seleccionado (C = 1, γ = 1, `balanced`, orden #33). Fuentes: `seleccion.csv` línea 3, `rejilla_completa.csv` línea 61 y `f21_techo5d.log` líneas 74, 113 y 154. Confusión en val: tp 30, fn 14, tn 54, fp 16 (`canonical_fragment.json`).
- **WITCOM («LogReg / SVM (5D-UMAP directo)» = 0.7246, `docs/resumen_modulo3_camera_ready.md` línea 119):** en el repositorio de TT-I (`proyecto-transformacion-texto-imagen`, commit `cee79b1`, «Update: Cambios para WITCOM»), `notebooks/5_reduccion_dimensionalidad.ipynb`, celda 2, ajusta `LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000)`, sin escalador, sobre `umap_params.pkl`. Calcula el F1 con `clf.predict(X_te)` sobre **test**, y su salida guardada es «F1-Test Componentes UMAP (5d): 0.7246». En ese repositorio y en este no se encontró código de un SVM sobre 5D.
- **Hecho aritmético:** con soportes 70/44 (los mismos en val y en test), el F1-Macro toma 1936 valores distintos a 4 decimales, 250 de ellos en [0.65, 0.75]. 0.7246 solo se obtiene con tp 30, fn 14, tn 54, fp 16. Lo mismo ocurre con 0.6933 (MLP de F2.1 en val y «MLP (5D-UMAP directo)» de WITCOM): solo se obtiene con tp 27, fn 17, tn 54, fp 16. No se localizó el código del MLP de WITCOM.
- En esta búsqueda no se ejecutó nada ni se calculó ninguna métrica de test: solo se leyeron salidas ya guardadas en los notebooks de TT-I.
