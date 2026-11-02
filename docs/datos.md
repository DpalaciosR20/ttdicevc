# Mapa de datos — Trabajo Terminal II

> Fuente de verdad sobre **dónde están** los insumos del proyecto, **qué contienen** y **cómo se usan**.
> Todo código del repositorio resuelve rutas mediante `src/paths.py` a partir de `TT_DATA_ROOT`; ninguna ruta se escribe a mano.
> Los campos **[completar]** restantes solo afectan la ejecución en Kaggle.

## 1. Reglas

1. **Los datos no se versionan.** Ningún CSV, PKL, imagen ni checkpoint entra a git. `.gitignore` excluye `data/` y `models/`.
2. **La carpeta de datos es de solo lectura.** Ningún script escribe dentro de `TT_DATA_ROOT`; toda salida va a `results/` dentro del repositorio.
3. **Datos personales.** Las columnas sociodemográficas y `personal_key` no salen de Drive ni aparecen en archivos del repositorio (sección 4).
4. **Sin variables derivadas de la etiqueta.** `beck_range` y `depression_level` nunca se usan como variables de entrada.
5. **La etiqueta se toma del índice maestro**, nunca del nombre de archivo de una imagen.
6. **Integridad.** Antes de usar un insumo se verifica su SHA-256 contra `meta/data_manifest.csv`.
7. **Metadatos versionados.** El índice maestro (`meta/sample_index.csv`) y el manifiesto (`meta/data_manifest.csv`) sí se versionan: no contienen texto ni datos personales.

## 2. Raíz de datos por entorno

| Entorno | `TT_DATA_ROOT` | Observación |
|---|---|---|
| Local (Windows, Claude Code) | `G:\Mi unidad\tt2-insumos` | Google Drive para escritorio en modo *duplicar archivos* |
| Google Colab | `/content/drive/MyDrive/tt2-insumos` | Drive montado |
| Kaggle | `/kaggle/input/datasets/santiagosoria26ss/tt2-insumos` [completar: verificar con `ls`] | Copia de Drive; debe coincidir con `meta/data_manifest.csv` |

## 3. Estructura e insumos

```
<TT_DATA_ROOT>/
├── exp_4_1_final/
│   ├── Exp_4.1_BETO_HPO_embeddings.pkl
│   └── checkpoint-342/
├── images_grayscale_classblind/
│   ├── perlin/                           ← 1,136 PNG
│   ├── gray_scott/                       ← 1,136 PNG
│   ├── lorenz/                           ← 1,136 PNG 
│   └── gray_scott_nested_cv/fold0/       ← NO USAR (ver sección 7)
├── processed/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   
└── umap_params.pkl
```

| Nombre lógico (`src/paths.py`) | Ruta relativa | Contenido |
|---|---|---|
| `CSV[split]` | `processed/{train,validation,test}.csv` | Corpus particionado 908 / 114 / 114 |
| `UMAP_PARAMS` | `umap_params.pkl` | Componentes UMAP en [0, 1] |
| `EMB_41` | `exp_4_1_final/Exp_4.1_BETO_HPO_embeddings.pkl` | Embeddings de Exp 4.1 |
| `CKPT_41` | `exp_4_1_final/checkpoint-342/` | Modelo de Exp 4.1 |
| `IMAGES[technique]` | `images_grayscale_classblind/{perlin,gray_scott,lorenz}/` | 1,136 imágenes por técnica |

## 4. Columnas de `processed/*.csv`

| Columna | Uso permitido | ¿Sale de Drive? |
|---|---|---|
| `id` | Consecutivo del corpus crudo; clave de enlace en el índice maestro | Sí |
| `personal_key` | Identificador del participante; solo para verificar unicidad (1,136) y ausencia de participantes compartidos entre splits | **No** |
| `text` | Entrada de BETO; en el repositorio solo su SHA-256, su número de tokens y la indicación de truncamiento | Solo derivados |
| `manual_classification` | Etiqueta `y_true` (0 = No Depresivo, 1 = Depresivo) | Sí |
| `beck_range`, `depression_level` | Ninguno como variable de entrada | **No** |
| `age`, `occupation`, `max_level_studies`, `residence`, `marital_status`, `Tiempo` | Ninguno sin previsión en el protocolo | **No** |

## 5. Estructura de los PKL

| Archivo | Claves |
|---|---|
| `umap_params.pkl` | `pkg['train']['params']` (908, 5), `pkg['train']['labels']`; ídem `'val'` y `'test'`. Se lee con `load_umap_arrays()` de `src/training/umap_io.py`, que verifica su SHA-256 |
| `Exp_4.1_BETO_HPO_embeddings.pkl` | `X_train` (908, 768), `y_train`, `X_val` (114, 768), `y_val`, `X_test` (114, 768), `y_test`, `model_name`, `shape`. Se carga con `joblib.load` |

## 6. Checkpoint de Exp 4.1

| Atributo | Valor | Fuente |
|---|---|---|
| Corrida | Reentrenamiento final del 11 de marzo de 2026 (notebook 3, commit `effff4a` del repositorio de TT-I); es la que generó `EMB_41` (commit `6cfc1ce`, mismo SHA-256) | Historial del repositorio de TT-I; F2.2, V3 |
| Modelo base | `dccuchile/bert-base-spanish-wwm-cased` | `3_hyperparameter_tuning.ipynb` |
| Tokenizador | Guardado en el checkpoint (`tokenizer.json`, `tokenizer_config.json`); `max_length = 128`, `padding = 'max_length'`, truncamiento activo | Ídem y contenido del checkpoint |
| Semilla | 42 (`set_seed(42)` y `TrainingArguments(seed=42)`) | Ídem |
| Época del checkpoint | 3 (114 pasos por época × 3 = 342); mejor F1-Val con `load_best_model_at_end` | Ídem |
| Precisión | `fp16 = True`: el reentrenamiento no es reproducible bit a bit; las predicciones se infieren del checkpoint, no se reentrena | Ídem |
| Archivos presentes | `config.json`, `model.safetensors` (incluye la cabeza de clasificación), `tokenizer.json`, `tokenizer_config.json`, `trainer_state.json`, `training_args.bin`, `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `rng_state.pth` | Drive |
| Historial de entrenamiento | `trainer_state.json` registra `best_model_checkpoint` y el F1-Val por época: fuente *extraída* del F1-Val de Exp 4.1 (0.7199) | — |

**Checkpoint descartado (29-oct-2026).** Hasta F2.2, `exp_4_1_final/checkpoint-342/` contenía la corrida del 10 de marzo (notebook 3, commit `6265aae`; F1-Val 0.7328). Esa corrida usó el mismo código, los mismos hiperparámetros, la misma semilla y la misma ruta de salida, pero con `fp16` sus pesos son distintos. No reproduce `EMB_41`: coseno [CLS] medio de 0.67 (F2.2, `results/f22_referencias_textuales/diagnostico_v3/`). Se retiró de `TT_DATA_ROOT` y no se usa. El checkpoint vigente es el de la corrida del 11 de marzo (`model.safetensors` con SHA-256 `c4f35147…`), que reproduce `EMB_41` con coseno 1.0000000 en las 1,136 filas.

## 7. Imágenes

| Técnica (vocabulario) | Prefijo de archivo | Generador |
|---|---|---|
| `perlin` | `perlin_` | Ruido de Perlin |
| `gray_scott` | `rd_` | Gray-Scott  |
| `lorenz` | `lorenz_` | Atractor de Lorenz |

El vocabulario coincide con los nombres de carpeta. El prefijo de Gray-Scott (`rd_`) es la única excepción y se traduce en `src/paths.py`.

- **Patrón:** `{prefijo}{split}_{NNNN}_label{y}.png`, por ejemplo `lorenz_test_0038_label0.png`.
- **`split`** toma los valores `train`, `val` y `test`. El CSV de validación se llama `validation.csv`; en todo el proyecto el split se denomina `val`.
- **`NNNN`** es el número de imagen generada dentro del split, desde `0000`, con cuatro dígitos.
- **Formato:** escala de grises, un canal, 256 × 256.
- **`gray_scott_nested_cv/fold0/`** pertenece a la validación cruzada anidada explorada en WITCOM. Por la decisión 1 de TT-II no se repite ni es confirmatoria: queda fuera del índice maestro y de todo análisis del Módulo 3.

## 8. Correspondencia entre fuentes

La fila *i* de cada split es la misma muestra en todas las fuentes:

```
processed/{split}.csv, fila i  ↔  EMB_41[X_{split}][i]  ↔  UMAP_PARAMS[split]['params'][i]  ↔  imagen {split}_{i:04d}
```

y su identificador es `sample_id = {split}_{i:04d}`.

| Verificación | Dónde |
|---|---|
| Etiqueta del CSV = etiqueta del PKL de embeddings = etiqueta de `umap_params.pkl` = sufijo `label{y}`, fila por fila | Índice maestro (F1.5) |
| Inventario y contrato de las imágenes | Capa C de F0.3 |
| Embeddings recalculados desde `CKPT_41` sobre el CSV ≈ filas de `EMB_41` (detecta permutaciones dentro de una misma clase, que la comparación de etiquetas no detecta) | F2.2 |

## 9. Pendientes

- [ ] Ruta exacta del dataset en Kaggle.

> Nota: el nombre correcto del PKL es `Exp_4.1_BETO_HPO_embeddings.pkl`; la bitácora de F0.3 lo transcribe con guion bajo.
