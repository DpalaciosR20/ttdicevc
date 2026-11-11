# F4.3, etapa 3b: reporte (detenido en el paso 2)

> Rama `sv/f4-atribucion` · Rol: SV · Sin interpretación

## Estado

| Paso | Estado |
|---|---|
| 1. Recepción de la parte GPU | Hecho. 16/16 comprobaciones OK. Commit `9867373`. |
| 2. Parte local | Prueba acotada: ≈ 43 min estimados, más de 20 min. Se prepararon las celdas de Kaggle CPU y se detuvo aquí (P20). |
| 3. Cierre (canónicas `f43.*`, bitácora, notebook, PR) | Pendiente, hasta recibir `f43_local_resultados.zip`. |

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

## PENDIENTES

1. **Correr las celdas de `docs/correr_f43_local_kaggle.md` en Kaggle (CPU)** con el `COMMIT` que entrega Claude Code junto con este reporte. Traer `f43_local_resultados.zip`.
2. **Al recibirlo:**
   - verificación local (sección del mismo documento);
   - commit de `results/f43_atribucion/local/`;
   - cierre del paso 3: canónicas `f43.*` con pruebas, `CANON_STRICT=1` y `--check`; bitácora `docs/bitacora/F4.3.md`; notebook `notebooks/F4.3_atribucion.ipynb` ejecutado; push y PR sin fusionar.
3. **Alternativa a decidir por SV:** correr la parte local en este equipo pese a superar los 20 min, fuera de la regla de `CLAUDE.md`. Solo con indicación explícita.
