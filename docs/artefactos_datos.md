# Artefactos de Datos

Estos archivos NO están en este repositorio (son binarios/pesados — ver `.gitignore`). Este documento dice qué existe, qué contiene, y dónde debería estar (actualiza las rutas/enlaces reales de Drive o Kaggle Datasets conforme los suban).

## `umap_params.pkl`
Cargar con `joblib.load` (NO con `pickle.load` — el archivo usa el formato de arrays de joblib).

Estructura:
```python
{
  'reducer': <objeto UMAP ya ajustado>,      # n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42, n_components=5
  'scaler': <MinMaxScaler ya ajustado>,
  'train': {'params': ndarray (908, 5), 'labels': ndarray (908,)},
  'val':   {'params': ndarray (114, 5), 'labels': ndarray (114,)},
  'test':  {'params': ndarray (114, 5), 'labels': ndarray (114,)},
  'n_components': 5, 'metric': 'cosine', 'random_state': 42,
}
```
Los 5 componentes ya están normalizados con MinMaxScaler (train cae exactamente en [0,1]; val/test se salen ligeramente, como corresponde a "fit solo en train, transform en val/test").

**Ubicación:** https://drive.google.com/file/d/16m-yVMYYpve7ZrewtEzADiyu4hytZ91A/view?usp=drive_link

## `Exp_4_1_BETO_HPO_embeddings.pkl`
Cargar con `joblib.load`. Embeddings BETO de 768 dimensiones (vector [CLS] de la última capa oculta, modelo `BETO_HPO_finetuned`).

Estructura:
```python
{
  'X_train': ndarray (908, 768), 'y_train': ndarray (908,),
  'X_val':   ndarray (114, 768), 'y_val':   ndarray (114,),
  'X_test':  ndarray (114, 768), 'y_test':  ndarray (114,),
  'model_name': 'BETO_HPO_finetuned',
  'checkpoint': '...',
}
```
Las etiquetas coinciden en orden exacto con `umap_params.pkl` en los 3 splits — mismo pipeline, mismo orden de muestras, se pueden combinar con confianza.

**Necesario solo si se va a reajustar UMAP** (p. ej. para un k-fold anidado correcto — ver `docs/hallazgos_metodologicos.md`, punto 3).

**Ubicación:** https://drive.google.com/file/d/1nVethfLzeW7vZps71Aem0K27DSkoF38X/view?usp=drive_link

## Datasets de imágenes class-blind (3 técnicas, 1,136 imágenes c/u)

Estructura de carpetas (una carpeta plana por técnica, sin subcarpetas de split ni de clase):
```
images_grayscale_classblind/
├── gray_scott/
│   ├── rd_train_0000_label0.png
│   ├── rd_train_0001_label1.png
│   ├── ...
│   ├── rd_val_0000_label0.png
│   └── rd_test_0000_label0.png
├── perlin/
│   └── perlin_<split>_<idx:04d>_label<0o1>.png
└── lorenz/
    └── lorenz_<split>_<idx:04d>_label<0o1>.png
```

Todas son PNG de 256×256, modo `'L'` (escala de grises real, 1 canal). El índice `<idx>` corresponde a la posición de la muestra dentro de su split (`train`/`val`/`test`) en `umap_params.pkl` — mismo orden, mismo índice.

**Generadas con:** `src/generation/gray_scott_lib.py`, `perlin_lib.py`, `lorenz_lib.py` (ver `docs/hallazgos_metodologicos.md` para el historial de por qué se regeneraron).

**Ubicación:** https://drive.google.com/drive/folders/1ELsTGeDASVe1URfg1HT81mqh9pLHmyih?usp=drive_link

**En Kaggle** (Dataset `palaciosroblesdiego/data-files`, verificado con el smoke test de `tests/`):
```
/kaggle/input/datasets/palaciosroblesdiego/data-files/TT/images_grayscale_classblind/gray_scott
```

> ⚠️ **No usar `TT/images/`** del mismo Dataset (p. ej. `TT/images/gray_scott/train/no_depresivo/...`). Su estructura de subcarpetas por clase corresponde a las imágenes **originales coloreadas por clase** (fuga #1 de `docs/hallazgos_metodologicos.md`). Usar siempre `TT/images_grayscale_classblind/`.

## Nota sobre `/kaggle/input/` vs `/kaggle/working/`

Si trabajan en Kaggle: `/kaggle/input/` es **solo lectura** (ahí se monta cualquier Dataset que suban). Ojo: Kaggle ahora monta los datasets en `/kaggle/input/datasets/<usuario>/<dataset>/`, no en `/kaggle/input/<dataset>/` — si una ruta no existe, verificar con `!find /kaggle/input -name "rd_train_0000_*.png"`. Todo lo que se genere en la sesión (nuevas imágenes, checkpoints de modelos, resultados) debe ir a `/kaggle/working/` — intentar escribir en `/kaggle/input/` da `OSError: Read-only file system`.
