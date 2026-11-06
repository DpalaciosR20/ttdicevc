# F3.4 — HPO breve de los finalistas (solo validación)

Resultados de la tarea **F3.4** del plan v2 (`plan/plan_tt2_v2.md`). Cubre un estudio de Optuna por finalista sugerido de F3.2, con el presupuesto de la matriz §6 (`configs/matriz_experimentos.json`, bloque `hpo`): 12 *trials* × semillas 42, 7 y 123, es decir **108 entrenamientos**. Bitácora: `docs/bitacora/F3.4.md`. Guía de ejecución: `docs/correr_hpo_kaggle.md`.

> **Solo validación.** `run_hpo.py` construye los *loaders* con `with_test=False` y ningún JSON de `runs/` tiene campos de prueba. El log muestra que existen las 114 imágenes de test (la verificación de insumos cuenta los archivos), pero no se cargan.

## Hiperparámetros para F3.5 y F3.6

| Técnica | Finalista | Trial adoptado | lr | weight decay | lote | F1-val trial 0 (media ± DE) | Mejor trial (media) | Ganancia | Decisión |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gray-Scott | EfficientNet-B0 FT | 0 | 3e-5 | 0 | 32 | 0.7176 ± 0.0125 | 7 (0.7255) | +0.0079 | se conserva el trial 0 (ganancia ≤ 1 DE) |
| Perlin | ViT-B/16 congelado | **10** | 0.008597392498054104 | 0.06998750099215974 | 16 | 0.7028 ± 0.0073 | 10 (0.7194) | +0.0165 | se adopta el mejor |
| Lorenz | ViT-B/16 congelado | 0 | 1e-3 | 0 | 32 | 0.7148 ± 0.0047 | 5 (0.7185) | +0.0036 | se conserva el trial 0 (ganancia ≤ 1 DE) |

Las banderas exactas para `run_grid` están en `run_grid_flags` de cada `hpo_result.json`. DE con `ddof=1`.

## Archivos (`<técnica>/`)

| Archivo | Contenido |
| --- | --- |
| `hpo_result.json` | Decisión de la regla final, `run_grid_flags`, comprobación de la selección, trial 0 frente a F3.2, commit y entorno |
| `hpo_meta.json` | Presupuesto, espacio, *sampler* (TPE, semilla 42, sin *pruner*), preprocesamiento y variante |
| `trials.csv` | Una fila por *trial*: parámetros, F1-val por semilla a precisión completa, media y DE |
| `runs/trial<NN>__seed<k>.json` (36) | Detalle por entrenamiento: historial por época, `val_probs` sin redondear, `val_preds`, `val_true`, determinismo, insumos, entorno y commit |
| `hpo.log` | Salida completa |
| `grid_meta.json` | `{"fase": "hpo", "smoke": false}` |

`errors/` quedó vacío en las tres técnicas, por eso no se versiona. `study.pkl` tampoco se versiona porque `.gitignore` excluye `*.pkl`. Si hace falta, el estudio se reconstruye con el mismo *sampler* a partir de `runs/` (`run_hpo.py --prev-results`). `entorno/` guarda `pip_freeze.txt` y `gpu.txt` de la sesión.

## Procedencia

- **Ejecución:** Kaggle, GPU Tesla T4, 5 nov 2026, con `created_at` de 22:51 UTC del 5 nov a 01:59 UTC del 6 nov. Repositorio en el commit **`b848d2c`** (rama `ce/f3.4-hpo`), registrado en los 108 JSON.
- **Entorno:** Python 3.13.16, torch 2.11.0+cu130, cuDNN 9.27 (92700) y optuna 5.0.0. Es la misma imagen de Kaggle que la repetición de F3.2.
- **Insumos:** `TT_DATA_ROOT=/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos`, con 908/114/114 imágenes por técnica.
- **Tiempo de GPU:** Gray-Scott 30.7 min, Perlin 76.3 min y Lorenz 81.5 min, en total 188.5 min.

## Verificación local (5 nov 2026)

- 36 JSON por técnica, cada uno con 12 *trials* × 3 semillas, `warning` vacío y commit `b848d2c`.
- El F1-val recalculado desde `val_probs` (umbral 0.5) es igual a `best_f1_val` en los 108 JSON.
- La regla final, recalculada de forma independiente desde `runs/`, da la misma decisión y el mismo *trial* que `hpo_result.json` en las tres técnicas.
- `selection.is_suggested_finalist` es `true` en las tres técnicas y `override` es `null`.
- **El trial 0 reproduce F3.2 bit a bit:** su `val_probs` es idéntico al `p_dep` de `results/predictions/primera_pasada/` en las 9 corridas (3 técnicas × 3 semillas).
