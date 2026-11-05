# F3.2 — Primera pasada del grid (solo validación), repetición

Resultados de la tarea **F3.2** del plan v2 (`plan/plan_tt2_v2.md`), repetida con las configuraciones de F1.3 (`configs/matriz_experimentos.json`) y F1.4 (`configs/preprocesamiento.json`) y con el formato de predicciones de F1.5. Cubre 3 técnicas × 3 arquitecturas × 2 estrategias × 3 semillas (42, 7, 123), es decir **54 corridas** sobre el split fijo 80/10/10 (908 train / 114 val). Sustituye a la primera pasada v0 (`results/f32_primera_pasada_v0/`). Bitácora: `docs/bitacora/F3.2.md`.

> **Solo validación.** Ninguna corrida cargó el conjunto de prueba. `grid_meta.json` registra `fase=primera_pasada`, y en cada JSON `f1_test` y `test_*` son `null`. Estos resultados sirven para seleccionar finalistas (F3.3) sin violar la compuerta del protocolo.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `../predictions/primera_pasada/<tec>__<arq>__<est>__seed<k>.csv` (54) | Predicciones por muestra en formato F1.5 (`docs/especificacion_predicciones.md`). Columnas: `sample_id, split, y_true, technique, arch, strategy, seed, p_dep`. `p_dep` va en float64 sin redondear, con 114 filas por archivo. Cada CSV tiene su `.manifest.json`, con variante, hiperparámetros, determinismo, `dir_digest` de los insumos, entorno y commit |
| `summary_by_config.csv` | 18 configuraciones con su F1-val (media ± DE, `ddof=1`) y la marca `seleccion`, que es informativa: la selección oficial de F3.3 la corre SV con `select_finalist.py` |
| `summary.csv` | 54 corridas. Columnas: F1-val, mejor época, épocas corridas, parámetros entrenables y totales, minutos y `warning` (vacío en las 54) |
| `runs/*.json` | Detalle por corrida: configuración, historial por época (`history`), `val_probs` sin redondear, pesos de clase, determinismo aplicado, insumos, entorno y commit |
| `grid_meta.json` | Fase del out-dir: `{"fase": "primera_pasada", "smoke": false}` |
| `determinismo.json`, `determinismo.log` | Prueba de humo de determinismo (matriz §11.6): cada configuración se entrena dos veces con la semilla 42 durante 2 épocas en Gray-Scott, y se comparan los pesos finales |
| `smoke.log` | Prueba de humo del grid: 1 época, ResNet-18 congelado, semilla 42, tres técnicas. Su salida no se versiona |
| `run_grid.log` | Salida completa de las 54 corridas |
| `entorno/pip_freeze.txt`, `entorno/gpu.txt` | Entorno de Kaggle de la corrida |
| `figures/*.png` | Figuras de `notebooks/F3.2_primera_pasada.ipynb`, en PNG a 300 dpi: F1-val por configuración, curvas de aprendizaje por técnica, v0 frente a la repetición, parada temprana, distribución de `p_dep` y matriz de confusión de los candidatos, y acuerdo entre semillas |

`errors/` quedó vacío; por eso no se versiona.

## Procedencia

- **Ejecución:** Kaggle, GPU Tesla T4, 2 nov 2026 (`created_at` de 2026-11-03T02:20 a 03:52 UTC). El repositorio estaba en el commit **`36c6398`**, registrado en los 54 manifiestos.
- **Entorno:** Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, cuDNN 9.27 (92700), transformers 5.20.2, tokenizers 0.23.3. **Es una imagen de Kaggle posterior** a la de la v0 y a la de `requirements.txt` (3.13.15, cu128, cuDNN 9.19); ver `docs/entorno_reproducibilidad.md`.
- **Insumos:** `TT_DATA_ROOT=/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos`. `build_manifest --verify` dio `manifiesto OK`, y el script verificó el `dir_digest` de las 3 carpetas de imágenes contra `meta/data_manifest.csv` antes de entrenar.
- **Comando:** `python -m src.training.run_grid --out-dir results/f32_primera_pasada`, sin banderas. Todo sale de los dos JSON de configuración:
  - preprocesamiento: Gray-Scott `px224-flip`, Perlin y Lorenz `px224-noflip`;
  - optimización: AdamW, lr 3e-5 en fine-tuning y 1e-3 en congelado, `weight_decay=0`, lote 32, hasta 15 épocas, paciencia 4;
  - pérdida: CE con pesos `[0.8180, 1.2861]`;
  - determinismo: `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` y SDPA *flash* y *memory-efficient* desactivados.
- **Tiempo total de GPU:** 95.5 min, contra 88.2 en la v0.

## Verificación local (2 nov 2026)

- `python -m src.predictions.validate "results/predictions/primera_pasada/*.csv"`: **54/54 archivos válidos**.
- `load_predictions` carga 6,156 filas (54 × 114), todas con `split=val`. `y_true` es idéntico en las 54 corridas (44 label 1, 70 label 0).
- `p_dep` de cada CSV es igual, bit a bit, a `val_probs` de su JSON. Para leerlo con pandas sin perder precisión se usa `float_precision='round_trip'`.
- El F1-val recalculado desde `p_dep` (umbral 0.5) es igual al `best_f1_val` del JSON y, redondeado, al de `summary.csv`.
- Los 54 manifiestos tienen el commit `36c6398` y la variante correcta por técnica.

## Determinismo (prueba de humo, Gray-Scott, semilla 42, 2 épocas)

| Configuración | Pesos idénticos | Tensores distintos | máx \|Δ p_dep\| | Minutos |
| --- | --- | --- | --- | --- |
| ViT-B/16 fine-tuning | sí | 0 de 152 | 0.0 | 2.66 |
| ResNet-18 fine-tuning | sí | 0 de 122 | 0.0 | 0.33 |
| EfficientNet-B0 fine-tuning | sí | 0 de 360 | 0.0 | 0.44 |

Fuente: `determinismo.json`.

## Finalistas sugeridos por la regla de la matriz §5 (pendiente de confirmar por SV en F3.3)

| Técnica | Sugerido | F1-val (media ± DE) | Empatados dentro de 1 DE del líder |
| --- | --- | --- | --- |
| Gray-Scott | EfficientNet-B0 fine-tuning | 0.7176 ± 0.0125 | ResNet-18 FT (líder, 0.7197 ± 0.0113), ViT-B/16 FT (0.7175) |
| Perlin | ViT-B/16 congelado | 0.7028 ± 0.0073 | ViT-B/16 FT (0.6999) |
| Lorenz | ViT-B/16 congelado | 0.7148 ± 0.0047 | EfficientNet-B0 FT (líder, 0.7166 ± 0.0018) |

En Lorenz el empate es muy justo: la diferencia de medias (0.001709) apenas queda por debajo de la DE del líder (0.001794), según los JSON a precisión completa. Detalle en la bitácora (H3).
