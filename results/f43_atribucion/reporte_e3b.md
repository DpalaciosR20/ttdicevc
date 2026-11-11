# F4.3, etapa 3b: reporte

> Rama `sv/f4-atribucion` · Rol: SV · Sin interpretación (la interpretación, marcada como tal, está en `docs/bitacora/F4.3.md`)

## Estado

| Paso | Estado |
|---|---|
| 1. Recepción de la parte GPU | Hecho. 16/16 comprobaciones OK. Commit `1de5683`. |
| 2. Parte local | Prueba acotada: ≈ 43 min estimados (> 20 min). Se corrió en Kaggle CPU con el commit `9581304`, una sola vez (112.7 s). Recepción: 16/16 OK, commit `5718f69`. |
| 3. Cierre | Canónicas `f43.*` (`8611788`), notebook y figuras (`4a783ed`), bitácora, log de pytest y este reporte (commit final de la rama). Push y PR sin fusionar. |

## 1. Recepción

- **Origen.** `G:\Mi unidad\tt2-insumos\f43_resultados\results\f43_atribucion\gpu\` se copió a `results/f43_atribucion/gpu/`, sin los `desktop.ini` de Drive.
  - `.gitattributes` tiene ahora `results/f43_atribucion/gpu/** -text`, para conservar los bytes de las salidas que tienen sha256.
- **Verificación.** `results/f43_atribucion/recepcion/verificacion_local.py` aplica la «Verificación local» de `docs/correr_f43_kaggle.md`. Su salida está en `verificacion_local.json`.

| Comprobación | Resultado |
|---|---|
| `commit.txt`, `meta.git_commit` y `cascada.meta.git_commit` | `660985c36d1f118b166428c247c903670f946ca4` |
| `smoke` / `limit` / `split` | `false` / `null` / `val` |
| sha256 de las resoluciones | `8aaf1561…`, igual al del archivo versionado en `660985c` |
| `datos_verificacion.log` | `manifiesto OK` |
| `modelos_verificacion.json` (F3.5) | 15/15, sin faltantes ni discrepancias |
| `mapas_verificacion.json` (F3.7) | 30/30, sin faltantes ni discrepancias |
| sha256 de los 15 `.pt` en `meta` frente a `meta/artefactos_manifest.csv` | 15/15 iguales |
| Máximo de `p_dep_max_abs_diff` (tolerancia 1e-4) | 7.66e-06 (Gray-Scott s99); ViT ≤ 4.6e-07 |
| Máximo de `vs_npz_f37.*.max_abs_diff_normalizado` (tolerancia 1e-2) | 2.44e-04 en los 15 modelos. Se comparan `map` y, en ViT, `backbone_map` |
| Máximo de `ablacion_p_dep_original_max_abs_diff` (tolerancia 1e-4) | 7.66e-06 |
| `f43_sha256sum.txt` | las 6 salidas coinciden |
| Orden de pasos de `cascada.json` | Gray-Scott: cabeza, features.8 … features.0. Perlin y Lorenz: cabeza, encoder.ln, encoder.layers.11 … 0, embeddings |
| Pares de referencia | 10 en cada (técnica, paso, mapa, medida) |
| `ablacion.json` | 9 condiciones por técnica; `delta_f1_media = 0` en `original`; `ablacion.csv` con 135 filas |

- **Hardware.** 2 × Tesla T4 (se usó `cuda`), torch 2.11.0+cu130, Python 3.13.16 y determinismo activado (`meta.determinism`).
- **Tiempos.** 1,673 s en total: Gray-Scott 139.6 s, Perlin 757.2 s y Lorenz 776.2 s. La estimación de E3a era ≈ 45 min (35–70).
- **Observación sobre el log.** `atribucion_gpu.log` imprime en cada línea el diccionario de tiempos acumulado de la técnica, no solo el de la semilla. Es cosmético y no afecta las salidas.

## 2. Parte local: prueba acotada (P20)

`results/f43_atribucion/estimacion_local/estimar.py` midió con datos reales, sin escribir las salidas de `atribucion_local.py`:
- `find_all` sobre la copia de Drive;
- la lectura y verificación de 1 `.npz` de Gray-Scott, 1 `.npz` de ViT y 1 `.pt` de ViT, con las mismas funciones del script;
- el cálculo por técnica.

Después extrapoló al total. La salida está en `estimacion.json`.

| Parte | Medido | Extrapolado |
|---|---|---|
| `find_all` | 21.3 s | 21 s |
| Lectura de `.npz` | 76.5 MB en 26.2 s (2.8 MiB/s) | 1.31 GiB → ≈ 8 min |
| Lectura de `.pt` | 343 MB en 91.7 s (3.6 MiB/s) | 6.39 GiB → ≈ 31 min |
| Cálculo, Gray-Scott | Spearman de 114 pares en 7.38 s; SSIM 56, 0.07 s; permutación de 250 réplicas, 0.03 s | ≈ 3.7 min |
| Cálculo, por ViT | Spearman de 114 pares en 0.12 s; SSIM 56, 0.08 s; permutación de 250 réplicas, 0.02 s | ≈ 0.2 min |
| **Total** | | **≈ 43 min** |

- **El Spearman de Gray-Scott (7.38 s frente a 0.12 s en ViT)** es una sola medición. Incluye la primera llamada del proceso y mapas de Gray-Scott, posiblemente con más empates. No se investigó: no cambia la decisión, porque la lectura domina.
- **Decisión.** El total es mayor que 20 min, así que, por P20 y `CLAUDE.md`, la parte local no se ejecutó en este equipo.
- **Celdas de Kaggle CPU.** Están en `docs/correr_f43_local_kaggle.md`. Se estiman ≈ 15 min (10–25) y la salida es `f43_local_resultados.zip`.

## 3. Parte local en Kaggle CPU

- **Primer intento de la celda 2.** Sin el input `tt2-mapas`, F3.7 dio 0/30 («falta» en los 30 `.npz`). Se detuvo, se agregó el input y la celda 2 dio 15/15 (F3.5), 15/15 (F3.6) y 30/30 (F3.7). La celda 3 se ejecutó una sola vez.
- **Origen.** `G:\Mi unidad\tt2-insumos\f43_local_resultados\results\f43_atribucion\local\` se copió a `results/f43_atribucion/local/`, sin los `desktop.ini`. `.gitattributes`: `results/f43_atribucion/local/** -text`.` se copió a `results/f43_atribucion/local/`, sin los `desktop.ini`. `.gitattributes`: `results/f43_atribucion/local/** -text`.
- **Verificación.** `results/f43_atribucion/recepcion/verificacion_local_cpu.py` → `verificacion_local_cpu.json`, 16/16 OK:

| Comprobación | Resultado |
|---|---|
| `commit.txt` y `meta.git_commit` | `95813042a618ae3097bec3a80aef8683004475d4` |
| sha256 de las resoluciones | igual al del archivo versionado en `9581304` |
| sha256 de la cascada leída (`meta.cascada`) | igual a `gpu/cascada.json` versionado |
| `meta.artefactos_verificados` | 50 (30 `.npz`, 20 `.pt`), sha256 y bytes iguales al manifiesto, rutas bajo `f37_mapas/maps/`, `f35_final/` o `f36_permutacion/`, ninguna en `f37_humo/` |
| Huellas | F3.5 15/15, F3.6 15/15, F3.7 30/30; sin faltantes, discrepancias ni ambiguos |
| `f43_local_sha256sum.txt` y `meta.*_sha256` | las 3 salidas coinciden |
| `holm.familia` / `lectura_vit` / B | m = 3 / Perlin y Lorenz con `cabeza_cascada` no nulo / 5,000 |

- **Entorno.** Python 3.13.16, `nproc` = 4, numpy 2.1.3, scipy 1.16.3, torch 2.11.0+cu130.

## 4. Resultados (`val` salvo indicación)

**Permutación y Holm (m = 3, α = 0.05)** — `local/resultados.json`:

| Técnica | L2 obs. | k | p | p ajustado | Rechaza |
|---|---|---|---|---|---|
| Gray-Scott | 70.0611 | 1 | 0.0003999 | 0.0007998 | sí |
| Perlin | 5.2376 | 177 | 0.03559 | 0.03559 | sí |
| Lorenz | 41.0496 | 0 | 0.0002000 | 0.0005999 | sí |

**Veredictos de cordura (Spearman 56 × 56)** — `local/resultados.json` y `gpu/cascada.json`:

| Técnica | Datos, `map` | Datos, `backbone_map` | Cascada, cabeza (`map`) | Pasos «cambia» (`map`) | Transiciones a constante (P9) |
|---|---|---|---|---|---|
| Gray-Scott | cambia (4/5) | — | cambia | 10/10 | 5 |
| Perlin | cambia (5/5) | no cambia | cambia | 15/15 | 0 |
| Lorenz | no cambia (5/5) | no cambia | cambia | 15/15 | 0 |

`backbone_map` en la cascada: no cambia en cabeza y `encoder.ln`; 13/15 pasos «cambia», con rango de referencia [1, 1].

**Cosenos de cabezas (nivel; rango de los 10 pares)** — Perlin w1 bajo [0.4539, 0.7427], w1 − w0 bajo [0.5457, 0.8133]; Lorenz w1 alto [0.1593, 0.4173], w1 − w0 alto [0.3133, 0.5894].

**Lectura de los ViT** — Perlin `cordura_por_datos_superada`; Lorenz `dependiente_de_la_cabeza_sin_validacion_por_datos`.

**Ablación, ΔF1 medio de 5 semillas** — `gpu/ablacion.json`:

| Técnica | Bajo σ1 | σ2 | σ4 | σ8 | Alto σ1 | σ2 | σ4 | σ8 |
|---|---|---|---|---|---|---|---|---|
| Gray-Scott | −0.0243 | −0.0546 | −0.1608 | −0.2615 | −0.2102 | −0.2045 | −0.2061 | −0.1656 |
| Perlin | −0.0760 | −0.0804 | −0.1595 | −0.1540 | −0.0591 | −0.0462 | −0.0335 | −0.0288 |
| Lorenz | −0.0024 | −0.0186 | −0.0570 | −0.1739 | −0.2815 | −0.1209 | −0.1123 | −0.0977 |

**Fracción de mapas constantes (5 semillas agrupadas)** — Gray-Scott `val`: clase real 0 49/350 (0.1400), clase real 1 13/220 (0.0591); clase predicha 0 62/305 (0.2033), 1 0/265. Prueba (réplica): 60/350 (0.1714) y 9/220 (0.0409). Perlin y Lorenz: 0 en todas las celdas.

**Gray-Scott, mapas frente a `p_dep`** — Pearson entre semillas 0.8255, Spearman 0.7 (n = 5).

## 5. Cierre

- **Canónicas.** `collect_f43` en `src/canonical/build.py`: 207 claves `f43.*` (p, k, B, estadístico y Holm; veredictos de cordura y del paso cabeza como identidad; pasos «cambia» y transiciones a constante; cosenos por semilla y su rango; lectura de los ViT; F1 base y ΔF1 por banda y σ; fracción y conteos de constantes, `val` y réplica de prueba). Métrica nueva en `schema.json`: `cosine`. Pruebas nuevas en `tests/test_canonical_numbers.py` (`test_f43_*`, 8): split, Holm y p = (k + 1)/(B + 1), criterio relativo y mayoría, lectura con `lectura_vit`, ΔF1, constantes, encadenamiento de insumos y valores en su localizador. `CANON_STRICT=1` con `CANON_UMAP_PARAMS="G:/Mi unidad/tt2-insumos/umap_params.pkl"`: build escrito y `--check` «al día (27620 líneas)».
- **Pruebas.** `python -m pytest tests` con `CANON_STRICT=1`: 1261 passed, 30 skipped (`results/f43_atribucion/pytest_e3b.log`).
- **Notebook.** `notebooks/F4.3_atribucion.ipynb`, ejecutado con `nbconvert --execute`; 11 figuras embebidas (`results/f43_atribucion/figures/f43_01…11`) y tablas. Solo lee salidas versionadas y comprueba las cifras contra `f43.*`.
- **Bitácora.** `docs/bitacora/F4.3.md`, con la tabla «qué se afirma / qué no» del plan, la lectura de los ViT y la regla de P9. Queda un **[completar]**: las versiones de los *datasets* de Kaggle.

## PENDIENTES

1. Versiones de los *datasets* de Kaggle (`tt2-insumos`, `tt2-artefactos`, `tt2-mapas`) usadas en las dos sesiones: no constan en los logs (bitácora, §3.3).
2. Revisión y fusión del PR (SV).
