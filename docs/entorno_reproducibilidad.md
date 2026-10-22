# Entorno y deserialización de `umap_params.pkl` (F0.2)

Nota técnica de la tarea **F0.2** del plan v2: *fijar versiones del entorno y resolver la deserialización de `umap_params.pkl`, o documentar el re-ajuste*.

## 1. Entorno de referencia

Es el de Kaggle donde se produjeron los resultados de F3.2 (`results/f32_primera_pasada/`). Las versiones exactas están en `requirements.txt`.

| Componente | Versión |
| --- | --- |
| Python | 3.13.15 |
| GPU / CUDA / cuDNN | Tesla T4 / 12.8 / 9.19 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| numpy / scipy / scikit-learn | 2.1.3 / 1.16.3 / 1.6.1 |
| matplotlib / pillow | 3.10.0 / 12.3.0 |
| umap-learn / pynndescent / numba | 0.5.12 / 0.6.0 / 0.61.2 |

- Cada corrida de `run_grid.py` registra su propio entorno en el JSON (`env`: python, torch, cuDNN, GPU, commit). Si Kaggle actualiza su imagen, esa diferencia queda registrada.
- **Kaggle no trae `noise`** (necesario para generar Perlin). Si se regenera Perlin allí: `!pip install noise==1.2.2`.
- **Reproducibilidad de entrenamiento:** con el mismo entorno, las CNN y ViT frozen se reproducen bit a bit con la misma semilla. ViT fine-tuning no: hay que reportarlo en distribución (hallazgo 8).

**Pendiente (no se puede saber desde aquí):** con qué versiones se **generaron las imágenes** del Módulo 2 y se creó `umap_params.pkl` (al menos scikit-learn 1.7.2, ver abajo, lo que sugiere Colab y no Kaggle). La capa C de la prueba de independencia (F0.3) responde si las imágenes guardadas se reproducen bit a bit con este entorno. Lorenz es el caso más sensible, porque renderiza con matplotlib.

## 2. `umap_params.pkl`: diagnóstico y solución

**Archivo de referencia:** `umap_params.pkl`, 3,021,128 bytes, **sha256 `081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224`**. Ubicación en `docs/artefactos_datos.md`. Si tu copia tiene otro sha256, no es el mismo archivo.

**Contenido** (verificado el 21 oct 2026):
- `reducer`: objeto UMAP ajustado (`n_components=5`, `metric='cosine'`, `random_state=42`).
- `scaler`: MinMaxScaler, guardado con **scikit-learn 1.7.2**, `clip=False`, ajustado con 908 muestras.
- `train` / `val` / `test`: `params` (float32, n×5) y `labels`.
- `n_components`, `metric`, `random_state`.
- Etiquetas por split: train 353 label 1 / 555 label 0; val 44 / 70; test 44 / 70. Las de val coinciden 1 a 1 con `y_true` de `results/f32_primera_pasada/predictions.csv`.

**Por qué fallaba:**
- `joblib.load` reconstruye el objeto UMAP completo, y para eso necesita umap-learn, pynndescent y numba en versiones compatibles con las que lo crearon. Las funciones compiladas de numba son lo que rompe.
- Además, el scaler viene de sklearn 1.7.2, y Kaggle trae 1.6.1.

**Solución adoptada: leer solo los arreglos (`src/training/umap_io.py`).** En TT2 el reductor no se usa: no se transforman datos nuevos, y el re-ajuste por fold se descartó (hallazgos 3–4). `load_umap_arrays()` carga el pkl con joblib, pero sustituye por objetos vacíos las clases de `umap`, `pynndescent`, `numba`, `llvmlite` y `sklearn`. Así devuelve `params` y `labels` **sin depender de esas librerías**, solo de numpy y joblib.

Lo que se verificó:
- Con el pkl real y esos módulos **bloqueados**, los arreglos cargan bien.
- Con un pkl sintético que contiene un UMAP real ajustado, los arreglos salen idénticos a los originales.
- Una carga auditada confirmó que el pkl real solo pide clases de numpy, joblib, scipy (`csr_matrix`) y sklearn (`MinMaxScaler`), además de las de umap, que se sustituyen. No pide nada más.

Uso:
```bash
python src/training/umap_io.py umap_params.pkl                                 # resumen, sha256, rango y saturación
python src/training/umap_io.py umap_params.pkl --export umap_params_arrays.npz # arreglos planos
```
`run_grid.py --umap-params` acepta el `.pkl` o el `.npz`, y ya no necesita `umap-learn`.

**Re-ajustar UMAP no es alternativa:** aun con los mismos hiperparámetros y `random_state=42`, UMAP no da el mismo resultado entre versiones de umap-learn y numba. Los nuevos `u` no corresponderían a las imágenes ya generadas, y habría que regenerar los 3 datasets e invalidar F3.2.

## 3. Hallazgo colateral: val/test recortados a [0,1]

Ver `docs/hallazgos_metodologicos.md`, hallazgo 7, corregido con estos datos. En resumen: **no hay parámetros fuera del contrato**, porque val y test se guardaron recortados (`clip`) a [0,1]. El efecto residual es que algunas muestras quedan saturadas en el borde: 20/114 de val y 8/114 de test tienen algún componente en exactamente 0 o 1.
