# Especificación del formato de predicciones por muestra (F1.5)

> Tarea F1.5 del plan v2 · Rol: SV · Estado: **aprobada por CE** ([completar: fecha de aprobación])
> Desarrolla la regla 9 del proyecto. Insumos y rutas: `docs/datos.md`.

## 1. Propósito

Todos los contrastes del Objetivo V y del análisis de preservación son pareados: comparan modelos sobre las mismas muestras. El *bootstrap* de Δ (F5.2), McNemar y la Q de Cochran (F5.4), el kappa y la intersección de errores (F4.6) y el análisis por longitud (F4.5) exigen que cada predicción se identifique sin ambigüedad con una muestra, un modelo y una corrida. Esta especificación fija ese formato para **todo** modelo del proyecto, visual o de referencia, y su cumplimiento se verifica mediante un validador automático.

## 2. Índice maestro de muestras

Archivo versionado `meta/sample_index.csv`, una fila por muestra (1,136 en total).

| Columna | Tipo | Definición |
|---|---|---|
| `sample_id` | str | `{split}_{idx:04d}`; no contiene la etiqueta |
| `split` | str | `train`, `val` o `test` |
| `idx` | int | Posición dentro del split en el orden de `umap_params.pkl` |
| `id` | int | Consecutivo del corpus crudo (columna `id` del CSV) |
| `y_true` | int | `manual_classification`; 0 = No Depresivo, 1 = Depresivo |
| `text_sha256` | str | SHA-256 del texto en UTF-8 |
| `n_tokens_beto` | int | Tokens con el tokenizador de `checkpoint-342`, incluidos `[CLS]` y `[SEP]`, sin truncar |
| `truncated_128` | bool | `n_tokens_beto > 128` |
| `img_perlin`, `img_gray_scott`, `img_lorenz` | str | Nombre de archivo de la imagen correspondiente |

No incluye `personal_key`, `text` ni columnas sociodemográficas o derivadas del inventario de Beck.

**Verificaciones de construcción**

1. 1,136 filas; 908 / 114 / 114; soportes 70 / 44 en `val` y en `test`.
2. Para cada fila, la etiqueta coincide en el CSV, en `Exp_4.1_BETO_HPO_embeddings.pkl`, en `umap_params.pkl` y en el sufijo `label{y}` de las tres imágenes.
3. Cada carpeta de técnica contiene exactamente 1,136 archivos, todos referenciados por el índice.
4. `personal_key` es única por fila y ningún participante aparece en dos splits; en el log solo se registra el resultado agregado.
5. La proporción de `truncated_128` reproduce el 59.1 % registrado (No Depresivo 62.6 %, Depresivo 53.5 %).

## 3. Archivo de predicciones

### 3.1 Columnas

Exactamente estas ocho, en este orden (regla 9):

| Columna | Tipo | Dominio |
|---|---|---|
| `sample_id` | str | Debe existir en el índice maestro |
| `split` | str | `train`, `val`, `test` |
| `y_true` | int | {0, 1}; igual al índice maestro |
| `technique` | str | Fuente de entrada (§3.2) |
| `arch` | str | Modelo (§3.2) |
| `strategy` | str | Estrategia (§3.2) |
| `seed` | int | Semilla de la corrida; los modelos deterministas registran su `random_state` |
| `p_dep` | float64 | Probabilidad de la clase Depresivo, en [0, 1], sin redondear |

Sin valores faltantes. Filas ordenadas por `split` (`train`, `val`, `test`) y, dentro de cada uno, por `sample_id`.

### 3.2 Vocabulario controlado

| Columna | Valores |
|---|---|
| `technique` | `perlin`, `gray_scott`, `lorenz` (entrada visual o descriptores de imagen); `text` (BETO); `emb768` (clasificador sobre los embeddings); `umap5d` (techo 5D); `none` (pisos que no dependen de la entrada) |
| `arch` | `resnet18`, `efficientnet_b0`, `vit_b16`, `beto`, `logreg`, `svm_rbf`, `gbm`, `rf`, `mlp`, `majority`, `stratified`, `brightness`, `descriptors` |
| `strategy` | `frozen`, `finetune` (modelos preentrenados); `none` (resto) |

Correspondencia con las referencias de la Fase 2:

| Referencia | `technique` | `arch` | `strategy` |
|---|---|---|---|
| BETO, Exp 4.1 (F2.2) | `text` | `beto` | `finetune` |
| Clasificador sobre 768D (F2.2) | `emb768` | según F2.2 | `none` |
| Techo 5D (F2.1) | `umap5d` | `logreg`, `svm_rbf`, `gbm`, `rf`, `mlp` | `none` |
| Clase mayoritaria / aleatorio estratificado (F2.3) | `none` | `majority` / `stratified` | `none` |
| Brillo promedio (F2.3) | técnica | `brightness` | `none` |
| Descriptores de bajo nivel (F2.4) | técnica | `descriptors` | `none` |

### 3.3 Regla de decisión

ŷ = 1 si y solo si `p_dep > 0.5`. La desigualdad estricta coincide con `argmax` sobre el softmax de dos clases (que, en empate, devuelve la clase 0) y con `predict` de scikit-learn. El umbral es fijo y no se ajusta sobre validación. Los pisos deterministas registran `p_dep ∈ {0.0, 1.0}`; el aleatorio estratificado registra la predicción realizada con su semilla.

### 3.4 Fases y clave única

Cada archivo pertenece a una fase, indicada por su carpeta y su manifiesto:

| Fase | Tarea | Splits permitidos |
|---|---|---|
| `referencia` | F2.1–F2.4 | `val`, `test` (las métricas de prueba se calculan solo en la Fase 5) |
| `primera_pasada` | F3.2 | `val` (**`test` prohibido**) |
| `final` | F3.5 | `val`, `test` solo si existe la etiqueta `protocolo-v1` en el historial |
| `control_permutacion` | F3.6 | Ídem `final` |

`train` es opcional en todas las fases. La clave única es (`fase`, `sample_id`, `technique`, `arch`, `strategy`, `seed`). La fase forma parte de la clave porque una misma combinación puede repetirse entre fases con configuraciones distintas; por ejemplo, la semilla 99 se usa tanto en los finalistas como en el control de permutación. La función de carga añade la columna `fase` desde el manifiesto; los archivos conservan las ocho columnas de la regla 9.

## 4. Ubicación y manifiesto

```
results/predictions/{fase}/{technique}__{arch}__{strategy}__seed{seed}.csv
results/predictions/{fase}/{technique}__{arch}__{strategy}__seed{seed}.manifest.json
```

Los archivos de predicciones se versionan: no contienen texto ni datos personales.

Campos obligatorios del manifiesto:

| Campo | Contenido |
|---|---|
| `task_id` | Tarea del plan (p. ej. `F2.2`) |
| `fase` | Según §3.4 |
| `git_commit` | Commit del código que generó el archivo |
| `created_at` | Fecha y hora UTC (ISO 8601) |
| `config` | Hiperparámetros efectivos, o ruta y SHA-256 del archivo de configuración |
| `inputs` | Ruta relativa a `TT_DATA_ROOT` y SHA-256 de cada insumo (checkpoint, PKL, carpeta de imágenes) |
| `model_selection` | Criterio y punto seleccionado (p. ej. época con mejor F1-Val) |
| `environment` | Versiones de Python y bibliotecas principales, plataforma y GPU |
| `csv_sha256` | SHA-256 del CSV de predicciones |
| `n_rows` | Número de filas por split |

## 5. Validación

`src/predictions/validate.py` (función `validate_predictions(path)` y uso por línea de comandos) rechaza un archivo si:

1. Las columnas, su orden o sus tipos no coinciden con §3.1, o hay valores faltantes.
2. Algún valor está fuera del vocabulario de §3.2 o `p_dep ∉ [0, 1]`.
3. El conjunto de `sample_id` de un split entregado no es exactamente el del índice maestro, o `y_true` difiere del índice.
4. Hay claves duplicadas.
5. El nombre del archivo no coincide con su contenido (`technique`, `arch`, `strategy`, `seed`).
6. Falta el manifiesto, le falta un campo obligatorio o `csv_sha256` no coincide.
7. Hay filas de `test` en una fase que no las permite según §3.4.

`load_predictions(patrón)` valida cada archivo, añade la columna `fase` y concatena.

## 6. Fuera del alcance

La agregación entre semillas, el cálculo de métricas y las pruebas estadísticas no forman parte del formato. Se definen en `docs/protocolo_preregistrado.md` (F1.2).
