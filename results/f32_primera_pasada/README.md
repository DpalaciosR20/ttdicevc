# F3.2 — Primera pasada del grid (solo validación)

Resultados de la tarea **F3.2** del plan v2 (`plan/plan_tt2_v2.md`). Cubre 3 técnicas × 3 arquitecturas × 2 estrategias × 3 semillas (42, 7, 123), es decir **54 corridas**. Todas usan el split fijo 80/10/10 (908 train / 114 val).

> **Solo validación.** Ninguna corrida cargó ni evaluó el conjunto de prueba (`grid_meta.json`: `evaluate_test=false`; en cada JSON, `f1_test` y `test_*` son `null`). Estos resultados se pueden usar para seleccionar finalistas (F3.3) sin violar la compuerta del protocolo.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `summary_by_config.csv` | 18 configuraciones: F1-val media ± std sobre las 3 semillas y `seleccion` (`finalista_sugerido` / `empate_1std`, regla 3.6). **La selección final la confirma SV (F3.3).** |
| `summary.csv` | Una fila por corrida (54): F1-val, mejor epoch, épocas corridas, parámetros entrenables y minutos. |
| `predictions.csv` | Predicciones por muestra en formato F1.5 (6,156 filas = 54 × 114): `sample_id, split, y_true, technique, arch, strategy, seed, p_dep, run_id`. `p_dep` = P(label=1, depresivo) en el mejor epoch por F1-val; `sample_id` = `val_<idx:04d>`, con el mismo idx que `umap_params.pkl`. |
| `runs/*.json` | Detalle de cada corrida: hiperparámetros, historial por época, predicciones, class weights, GPU, versión de torch y commit del código. |
| `grid_meta.json` | Modo del out-dir (`evaluate_test=false`). |

## Procedencia

- **Ejecución:** Kaggle, notebook `palaciosroblesdiego/grid-de-18-configuraciones`, GPU Tesla T4, torch 2.11.0+cu128, 21 oct 2026. Comando: `src/training/run_grid.py` con sus defaults (224 px + flip horizontal, lr 3e-5 fine-tuning / 1e-3 frozen, batch 32, hasta 15 épocas, paciencia 4).
- **Código:**
  - Gray-Scott: commit `ab81cca`.
  - Perlin y Lorenz: commit `607df40`.
  - La única diferencia entre ambos es la validación de `--prev-results` (PR #4); el código de entrenamiento es idéntico.
- **Gray-Scott se corrió dos veces** (la segunda, por accidente). Aquí están los resultados de la **segunda** corrida, importados con `--prev-results`. Las CNN y ViT frozen dieron lo mismo en ambas; ViT fine-tuning no (ver `docs/hallazgos_metodologicos.md`, hallazgo 8).
- Tiempo total de GPU: 88.5 min.

## Verificación hecha antes de subirlos (21 oct 2026)

- Existen las 54 corridas esperadas, sin errores; ningún JSON tiene datos de test.
- `predictions.csv`:
  - solo `split=val`;
  - 114 filas por corrida con `sample_id` `val_0000`…`val_0113` en orden;
  - `y_true` idéntico en las 54 corridas (mismo orden de muestras en las 3 técnicas). En validación hay 44 muestras con label 1 y 70 con label 0.
- El F1-val recalculado desde `p_dep` (umbral 0.5) coincide exactamente con el de cada JSON y con `summary.csv`.
- La media y la std de `summary_by_config.csv` coinciden con las de los JSON.

## Finalistas sugeridos por la regla 3.6 (pendiente de confirmar por SV)

| Técnica | Sugerido | F1-val (media ± std) | Empatados dentro de 1 std |
| --- | --- | --- | --- |
| Gray-Scott | EfficientNet-B0 fine-tuning | 0.7176 ± 0.0125 | ResNet-18 FT (0.7278, mejor media), ViT-B/16 FT |
| Perlin | EfficientNet-B0 fine-tuning | 0.6977 ± 0.0040 | ViT-B/16 FT |
| Lorenz | ViT-B/16 frozen | 0.7068 ± 0.0168 | EfficientNet-B0 FT (0.7096, mejor media), EfficientNet-B0 frozen |

Pendientes para SV:
- **Lorenz:** la regla cuenta parámetros *entrenables* (ViT frozen tiene 1,538, aunque el modelo completo pesa 85.8 M). Falta confirmar ese criterio.
- **F1.4:** si se cambia el preprocesamiento de alguna técnica, hay que repetir su primera pasada antes de elegir finalista.
