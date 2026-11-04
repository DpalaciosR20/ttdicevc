# F1.1 — Análisis de potencia del contraste de no inferioridad

El diseño está en `docs/especificacion_f11.md`: §1–§6 se preregistraron, §7 se completó después de ejecutar y §8 es una propuesta pendiente de revisión por SV. El código es `potencia.py`, cuyo docstring documenta el generador de S y las aclaraciones de implementación. El fragmento canónico lo escribe `f11_fragment.py`.

No se calculó ninguna métrica de prueba de modelos visuales. En prueba solo intervienen `y_test` y las predicciones de BETO (F2.2, ya conocidas); todo lo visual de prueba es simulado.

## Commits

| Momento | Commit | Fecha (−06:00) | Contenido |
|---|---|---|---|
| Preregistro | `e15d2b2` | 2026-11-02 13:19 | §1–§6 de la especificación, con §7–§8 vacías. Solo corrige la ruta de §4 respecto de `6d975c8` |
| Script | `af6abc9` | 2026-11-02 13:26 | `potencia.py` y `requirements-kaggle.txt` |
| Estimación de tiempo | `cb5f567` | 2026-11-02 13:26 | `potencia_dryrun.log`. **Es el código con el que se ejecutó la corrida** |
| Prueba de canónicas | `5369d9b` | 2026-11-02 13:52 | `tests/test_canonical_numbers.py` exime las cifras de diseño `f11.*` de la prueba de alcanzabilidad |
| Resultados | `04007be` | 2026-11-02 13:52 | Salidas, figura regenerada, fragmento y `canonical_numbers.json` |

La corrida completa se ejecutó una sola vez, con HEAD en `cb5f567` y el árbol limpio, según `potencia.log`. Antes hubo dos corridas que no se reportan:

- Una corrida de depuración con 10 réplicas y B = 200, escrita en una carpeta temporal fuera del repositorio. Sirvió para corregir errores del script antes de su commit; sus números no se usaron.
- El *dry-run*, registrado en `potencia_dryrun.log`.

Después de la corrida, `curva_potencia.png/.pdf` se regeneró con `potencia.py --solo-figura` a partir de `malla_escenarios.csv`, sin volver a simular. El cambio, incluido en `04007be`, solo mueve la leyenda, que tapaba la etiqueta de 0.80, y parte el pie de figura en dos líneas.

## Estimación de tiempo y dispositivo

- **Dry-run** (`potencia_dryrun.log`): 20 réplicas del escenario ρ = 1.00, p50, DE 0.010 tardaron 47.4 ms por réplica, más 1.2 s de calibración. Extrapolado a 54 escenarios × 1,000 réplicas, da **43.7 min en un proceso**, por encima del umbral de 20 min de `docs/ejecucion_kaggle.md`.
- Se prepararon las celdas de Kaggle, pero SV decidió ejecutar localmente con `--jobs 6`.
- **Duración real: 6.6 min.**
- Dispositivo: CPU (Intel64 Family 6 Model 140, 8 núcleos), sin GPU. Todo el cómputo es numpy en float64, salvo el producto de la matriz de permutaciones, que es float32 con conteos enteros exactos.

## Entorno

- Windows 11 Pro 10.0.26200, `.venv` del repositorio.
- Python 3.13.2 · numpy 2.1.3 · scipy 1.16.3 · pandas 2.3.3 · scikit-learn 1.6.1 · matplotlib 3.10.0 · jsonschema 4.26.0 · pytest 8.4.2.
- No usa `TT_DATA_ROOT`: todos los insumos están versionados. `canonical_numbers.json` se reconstruyó con `CANON_UMAP_PARAMS = G:\Mi unidad\tt2-insumos\umap_params.pkl` y `CANON_STRICT=1`.

## Insumos

Hay dos SHA-256 por archivo:
- **git (LF):** el del contenido en el commit `cb5f567`. Es el que usa `canonical_numbers.json` y coincide en cualquier sistema.
- **copia de trabajo:** el que registró `potencia.log`. En este checkout de Windows (`core.autocrlf`), los archivos fuera de `results/predictions/` tienen fin de línea CRLF.

| Insumo | Papel | SHA-256 git (LF) | SHA-256 copia de trabajo |
|---|---|---|---|
| `meta/sample_index.csv` | `y_val`, `y_test` (70/44) | `d2f789bb2812e55bed0ddc71ed13695fe092859722e4826a324a8569f1bd040a` | `74dc778292447c66be60b35b5ec224c53b01433423923fa6d6ddef91749de0f5` |
| `results/predictions/referencia/text__beto__finetune__seed42.csv` | BETO, val y test | `3d3587469545e592e39fa62d3b00b0acd52889bfe21dceb0a5c2816a28f7916e` | ídem |
| `results/predictions/referencia/emb768__logreg__none__seed42.csv` | LogReg-768D, solo val | `b8c3d471a860b943df962fa8e4652d2ff400e3fcbcf6b9af2e84871091cb778a` | ídem |
| `results/f32_primera_pasada_v0/predictions.csv` | F3.2 v0, solo val (54 corridas) | `46a9a4cba45681de266c4d9e9ce15c5d61aa57ca8164574959ce56b930c8c30e` | `448b457f37921ab5549218fb870de1642820305fce0df317ab5d212f2ba515ed` |
| `results/canonical_numbers.json` (antes de integrar F1.1) | F1_BETO, piso, Δ de WITCOM, F1-val de F3.2 | `7f17e3bf8c74c9578a76540ec5d55e49cd6713e6daae032b8921a16e939c62bb` | `f94c37f677dc89f262fc366d75784d77b2a83c3e09d25552b1163e0b411bf660` |

## Archivos

| Archivo | Contenido |
|---|---|
| `potencia.py` | Bloques E, S, D y P; `--dry-run`, `--jobs`, `--solo-figura` |
| `potencia.log` / `potencia_dryrun.log` | Bitácora de la corrida y estimación de tiempo |
| `empirico.csv` | E: Δ y θ (r = 0.50/0.75/0.80) frente a BETO-val por configuración de F3.2 y LogReg-768D, con IC 1−2α_h y su ancho para α_h ∈ {α/3, α/2, α} |
| `empirico_corridas.csv` | E: F1-val y discordancia con BETO por corrida (54 + LogReg-768D) |
| `malla_escenarios.csv` | S: escenario × formulación × r. Contiene P(V1..V4) con su SE, el SE de Monte Carlo del límite del IC, P(piso), los parámetros del generador (a, η, κ), la factibilidad y la retención, DE y discordancia realizadas |
| `retencion_minima.csv` | D: ρ_min y F1 equivalente por formulación × r × escenario |
| `potencia_piso.csv` | P: P(superar el piso) por escenario, con su SE |
| `curva_potencia.png` / `.pdf` | P(V1) frente a ρ por r. Paneles A y B; la línea es el escenario central y la banda, el rango sobre los escenarios |
| `f11_fragment.py` / `canonical_fragment.json` | `f11.delta_margen.*` y `f11.f1_min_detectable.*` (cifras de diseño) |

## Verificación

- El F1 vectorizado coincide con `sklearn.metrics.f1_score(average='macro')` en 100 réplicas × 6 columnas (diferencia máxima 0).
- El F1-val recalculado de las 54 corridas de F3.2 coincide con `canonical_numbers.json` (diferencia máxima 0). BETO (val y test) y LogReg-768D (val) también coinciden.
- P(V1) + P(V2) + P(V3) + P(V4) = 1 en cada celda factible (lo comprueba el script).
- En las celdas factibles, la retención, la DE y la discordancia realizadas difieren de las objetivo como máximo en 0.0086, 0.0003 y 0.0016, respectivamente. La excepción es la celda con `de_factible = False`.
- `python -m src.canonical.build --check` indica que el archivo está al día, y `pytest tests/test_canonical_numbers.py` da 205 pruebas aprobadas.
