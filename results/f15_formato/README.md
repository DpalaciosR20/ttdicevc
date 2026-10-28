# F1.5 — Formato de predicciones por muestra: evidencia

Tarea F1.5 del plan v2 (rol SV). Especificación: `docs/especificacion_predicciones.md`; insumos y rutas: `docs/datos.md`.
Fecha: 2026-10-28.

## Contenido

| Archivo | Qué es |
|---|---|
| `build_sample_index.log` | Construcción de `meta/sample_index.csv` y sus cinco verificaciones (§2) |
| `build_manifest.log` | Construcción de `meta/data_manifest.csv` |
| `pytest.txt` | `python -m pytest tests/test_predictions_format.py -v` |
| `validate_f32.txt` | Validador aplicado a `results/f32_primera_pasada/predictions.csv` (prueba de humo de la CLI) |

## Código

- Commit base: `e0ece05f8cc90e5b417e874e448876df695f2f8b`. Los archivos de F1.5 estaban **sin commitear** cuando se generó esta evidencia: el commit que los incorpore es el de referencia.
- `src/paths.py` resuelve las rutas desde `TT_DATA_ROOT` (traduce `gray_scott → rd_` y `val → validation.csv`) e impide escribir en la carpeta de datos.
- `src/build_manifest.py` genera `meta/data_manifest.csv`; `src/build_sample_index.py` genera `meta/sample_index.csv`.
- En `src/predictions/` están `schema.py` y `validate.py` (`validate_predictions`, `load_predictions` y la CLI).
- `tests/test_predictions_format.py` contiene un archivo válido y casos de falla para las reglas 1–7 de §5.

Reproducción (desde la raíz del repo, con `TT_DATA_ROOT` definido):

```
python -m src.build_manifest
python -m src.build_sample_index
python -m pytest tests/test_predictions_format.py -v
python -m src.predictions.validate "results/predictions/**/*.csv"
```

## Entorno

| | |
|---|---|
| Plataforma | Windows 11 Pro (10.0.26200), CPU, `.venv` local |
| Python | 3.13.2 |
| Bibliotecas | pandas 2.3.3, numpy 2.1.3, joblib 1.6.0, transformers 5.21.0, tokenizers 0.23.3, pytest 8.4.2 |
| `TT_DATA_ROOT` | `G:\Mi unidad\tt2-insumos` (Drive para escritorio) |

`transformers` y `tokenizers` se agregaron a `requirements.txt` con la versión local; la de Kaggle está marcada [completar].

## Insumos (SHA-256)

Detalle por archivo en `meta/data_manifest.csv`: 3,427 filas, 3,423 archivos y 4 carpetas. El SHA-256 de cada carpeta es el de las líneas `rel_path\tsha256\n` de sus archivos, ordenadas.

| Insumo | Ruta relativa | Archivos | SHA-256 |
|---|---|---|---|
| `CSV[train]` | `processed/train.csv` | 1 | `4c677f404c19cf8c7d99a2fbc049d2655264c2f05787b3f448e2f8d2b6c9d37b` |
| `CSV[val]` | `processed/validation.csv` | 1 | `d249b14b01149ec1e5cd933ae3349c470f71b8ba79d4745f534610d71c1d9f61` |
| `CSV[test]` | `processed/test.csv` | 1 | `e251582ae76e419e344228c3bcd3d6c8f468b481968e3cb6f06f221f0c9f6b5b` |
| `UMAP_PARAMS` | `umap_params.pkl` | 1 | `081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224` |
| `EMB_41` | `exp_4_1_final/Exp_4.1_BETO_HPO_embeddings.pkl` | 1 | `64b7682e6d6517e0e643502a96cc80949bb5b35d2a4c04a4ba345243ad1b5d44` |
| `CKPT_41` | `exp_4_1_final/checkpoint-342/` | 10 | `2969f6466ad469bb2fe21ac0d57d48614d053ac3188392cbb68695b8a5895ae5` |
| `IMAGES[perlin]` | `images_grayscale_classblind/perlin/` | 1,136 | `b8ce53c6fa339d3a4dcca2ba647c55aeaec79909f7137563efd8b5d04f7e3ccb` |
| `IMAGES[gray_scott]` | `images_grayscale_classblind/gray_scott/` | 1,136 | `c3593c9bd04b4abb45afe7958bded087b53b7812a7cb11e72bda8e382f2d31db` |
| `IMAGES[lorenz]` | `images_grayscale_classblind/lorenz/` | 1,136 | `6fc478a58477f5ba76d9105435be179c52539db2d67dbee3918a67dc5cfdde9b` |

`meta/sample_index.csv`: `d2f789bb2812e55bed0ddc71ed13695fe092859722e4826a324a8569f1bd040a`.

## Resultado

**Índice maestro: las cinco verificaciones pasan.**

| # | Verificación | Resultado |
|---|---|---|
| 1 | 1,136 filas; 908/114/114; soportes 70/44 en val y test | OK |
| 2 | Etiqueta igual en CSV, EMB_41, `umap_params.pkl` y sufijo `label{y}` de las 3 imágenes | OK, en las 1,136 filas |
| 3 | 1,136 archivos por técnica, todos referenciados | OK, después de borrar los duplicados de perlin (ver abajo) |
| 4 | `personal_key` única; ningún participante en dos splits | OK: 1,136 claves únicas, 0 cruces |
| 5 | `truncated_128` = 59.1 % (No Depresivo 62.6 %, Depresivo 53.5 %) | OK: 59.07 % / 62.59 % / 53.51 % |

Mediana de `n_tokens_beto` = 192; máximo = 2,209.

**Validador:** 27/27 pruebas pasan. Sobre el archivo de F3.2, la CLI lo rechaza por las reglas 1 (columna `run_id`), 5 (nombre `predictions.csv`) y 6 (sin manifiesto).

## Incidencias y hallazgos

1. **Duplicados en `perlin/`.** Al inicio había 1,138 archivos: `perlin_train_0838_label0 (1).png` y `perlin_train_0294_label0 (1).png`, con el mismo SHA-256 que sus originales (artefacto de descarga de Drive). Los borró SV en Drive el 2026-10-28 antes de generar el manifiesto y el índice definitivos. Si la copia de Kaggle se hizo antes de esa fecha, también los tiene: `python -m src.build_manifest --verify` lo detecta.
2. **`desktop.ini` en `checkpoint-342/`.** Drive para escritorio crea un `desktop.ini` oculto. No es un insumo y no existe en Colab ni en Kaggle, así que `paths.SYSTEM_FILES` lo excluye del manifiesto y del inventario (queda registrado en `build_manifest.log`). Con esta exclusión el checkpoint tiene los 10 archivos de `docs/datos.md` §6.
3. **F3.2 no cumple el formato de §3.** `run_grid.py` y `results/f32_primera_pasada/predictions.csv` tienen:
   - 9 columnas (`run_id`);
   - `arch = vit_b_16` en vez de `vit_b16`;
   - un solo archivo en vez de uno por corrida, sin manifiesto;
   - `p_dep` redondeado a 6 decimales.
   
   Queda como pendiente para CE: adaptar `run_grid.py` y regenerar o convertir las predicciones de la primera pasada. El redondeo no se puede deshacer sin volver a inferir.
4. **`processed/*_indices.csv` no están documentados** en `docs/datos.md` §3. Por decisión de SV quedan fuera del manifiesto. Según las dos primeras filas, parecen ser `id − 1` (posición 0-based en el corpus crudo): `train_0000` tiene `id = 554` y `train_indices` empieza en 553. [completar: confirmar y documentar.]
5. **No se corrió `python -m src.build_manifest --verify` completo** después de generar el manifiesto: volver a calcular el SHA-256 de 1.3 GB desde Drive tarda más de 10 minutos. El manifiesto se generó dos veces con resultados idénticos para todo, salvo el `desktop.ini`.
