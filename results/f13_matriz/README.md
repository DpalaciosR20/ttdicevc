# F1.3 — Matriz de experimentos del Módulo 3

Evidencia de la tarea **F1.3** del plan v2. El entregable es `docs/matriz_experimentos.md`, y su versión legible por código es `configs/matriz_experimentos.json`. SV hizo la tarea el 2 de noviembre de 2026 en ausencia de CE. Las decisiones de §12 de la matriz están pendientes del visto bueno de CE, y los pendientes de implementación de §11 se le transfieren.

No se entrenó ningún modelo, no se usó GPU y no se leyó el conjunto de prueba. `select_finalist.py` **no se ejecutó sobre resultados reales**: solo con casos sintéticos en las pruebas.

## Archivos

| Archivo | Contenido |
|---|---|
| `reconocimiento_ce.md` | Reconocimiento de solo lectura del código de CE (`src/training/`) y de `results/f32_primera_pasada/`, con archivo:línea sobre `main` @ `076c9a5`. Bloques A (configuración real), B (capacidades), C (costos) y E (hallazgo 8), más las contradicciones entre el borrador y el código |
| `contar_parametros.py` | Cuenta los parámetros entrenables y totales de las 6 combinaciones arquitectura × estrategia (CPU, `weights=None`) y los compara con `configs/matriz_experimentos.json` |
| `contar_parametros.log` | Salida de `contar_parametros.py` |
| `pytest.log`, `pytest.xml` | Pruebas nuevas de F1.3: `tests/test_matriz_experimentos.py` y `tests/test_select_finalist.py` |
| `pytest_suite.log` | Suite completa de `tests/` |

Archivos de F1.3 fuera de esta carpeta:

| Archivo | Contenido |
|---|---|
| `docs/matriz_experimentos.md` | Matriz (§1–§12) |
| `configs/matriz_experimentos.json` | §2–§8 en el vocabulario de F1.5 (`vit_b16`), con los conteos de parámetros del reconocimiento |
| `tests/test_matriz_experimentos.py` | Coherencia del JSON con el documento, con `configs/preprocesamiento.json`, con el reconocimiento y con `src/predictions/schema.py` |
| `src/selection/select_finalist.py` | Regla de §5 sobre predicciones de `primera_pasada` cargadas con `load_predictions` |
| `tests/test_select_finalist.py` | Casos sintéticos de la regla de §5 |

## Commits

Rama `sv/f1.3-matriz-experimentos`, creada desde `main` @ `076c9a5`.

| Commit | Fecha (−06:00) | Contenido |
|---|---|---|
| `d364e54` | 2026-11-02 15:24 | `docs/matriz_experimentos.md` y `reconocimiento_ce.md` |
| `21831c9` | 2026-11-02 15:26 | `configs/matriz_experimentos.json`, `tests/test_matriz_experimentos.py` y las aclaraciones de §5 y §12 (varios líderes → mayor DE; empate residual → error) |
| `48b6f7d` | 2026-11-02 15:30 | `src/selection/select_finalist.py` y `tests/test_select_finalist.py` |
| El de este README | 2026-11-02 | `contar_parametros.py`, logs de pytest y este README |

Los logs de esta carpeta se generaron con HEAD en `48b6f7d` y el árbol limpio, salvo los propios archivos nuevos de esta carpeta.

## Código auditado (reconocimiento)

| Archivo | Último commit | Uso |
|---|---|---|
| `src/training/build_model.py` | `f1fa300` | Pesos, cabeza y congelado |
| `src/training/data.py` | `37e5652` | Transformaciones, pesos de clase, permutación, DataLoaders |
| `src/training/train_model.py` | `37e5652` | Optimizador, early stopping, semillas y determinismo |
| `src/training/run_grid.py` | `4175be1` | CLI, compuerta de test y salidas |

El código de entrenamiento es el mismo con el que se corrió F3.2 (`ab81cca` para Gray-Scott y `607df40` para Perlin y Lorenz). Desde entonces solo cambió la lectura de `umap_params.pkl` en `run_grid.py` (`git diff 607df40 HEAD -- src/training/`). F1.3 no modificó `src/training/`.

## Entorno

- Windows 11 Pro 10.0.26200, `.venv` del repositorio, solo CPU.
- Python 3.13.2 · numpy 2.1.3 · pandas 2.3.3 · scikit-learn 1.6.1 · pytest 8.4.2.
- Conteo de parámetros: torch 2.11.0+cpu · torchvision 0.26.0+cpu (`contar_parametros.log`, línea 1).
- No usa `TT_DATA_ROOT`. Los costos del bloque C del reconocimiento se leyeron de `results/f32_primera_pasada/` (Kaggle, Tesla T4, torch 2.11.0+cu128), que está versionado.

## Resultados de la verificación

- `contar_parametros.log`: las 6 combinaciones coinciden con el bloque `parametros` del JSON.
- `pytest.log`: 37 pruebas, 37 pasan (20 de `test_matriz_experimentos.py` y 17 de `test_select_finalist.py`).
- `pytest_suite.log`: 279 pasan y 31 se omiten. Las omisiones son pruebas de F0.3 y de las cifras canónicas que requieren `F03_UMAP_PARAMS` o `CANON_UMAP_PARAMS`, que no se definieron.
- Antes del commit `48b6f7d` se comprobó, con mutaciones temporales de `select_finalist.py` que después se revirtieron, que cada paso de la regla tiene una prueba que falla si se altera: DE menor en lugar de mayor con varios líderes, tolerancia estricta en lugar de `≥`, desempate sin parámetros totales, desempate por menor media y empate residual sin error. Esas corridas no se guardaron.
- Igualmente, antes del commit `21831c9`, mutaciones en memoria del JSON (presupuesto de HPO, conteo de parámetros, preprocesamiento, lr, semillas, nombre de ViT y regla de varios líderes) hicieron fallar la prueba correspondiente. Tampoco se guardaron.

## Por qué `select_finalist.py` no se ejecutó

Además de la instrucción de no ejecutarlo sobre resultados reales: no hay predicciones de `primera_pasada` en el formato de F1.5 (`results/predictions/` solo contiene `referencia/`). Las de F3.2 v0 están en `results/f32_primera_pasada/predictions.csv`, en el formato anterior de `run_grid.py` (`vit_b_16`, un solo CSV, sin manifiesto), y están sustituidas (§5 de la matriz). La selección real se hará sobre la repetición de F3.2, cuando `run_grid.py` escriba en el formato de F1.5 (§11, punto 2).
