# notebooks/historial/

Notebooks de trabajo anteriores a TT2 que se conservan como **evidencia de procedencia**. No se ejecutan desde el repo: sus rutas son las de la sesión de Kaggle original y dependen de archivos que no están aquí. No se editan; si hay que repetir un cálculo, se hace en un script o notebook nuevo de la tarea correspondiente.

## `camera_ready_CE.ipynb`

Notebook de trabajo de CE (Diego) para el camera-ready del Módulo 3. Es una copia sin cambios del que se descargó de Kaggle el 29 jul 2026 a las 04:45 (`notebook32c2241c10.ipynb`, SHA-256 `ce10cf8e26da67c9…`) y conserva sus salidas.

| Celdas | Contenido |
| --- | --- |
| 2–4 | Setup y baselines directos sobre los 5 componentes UMAP |
| 5–7 | Diagnóstico del dataset coloreado original y prueba de fuga por luminancia (Gray-Scott) |
| 8–18 | Regeneración class-blind de Gray-Scott, Perlin y Lorenz |
| **19–20** | **Tabla 11 recalculada: entropía de Shannon, contraste, densidad de bordes y energía LBP; prueba t de Welch con corrección de Holm (k = 12)** |
| 21 | Prueba de brillo (clasificador con solo el brillo medio) sobre las 3 técnicas |
| 22–26 | ResNet-18 en Gray-Scott, evaluación y McNemar frente a LogReg-5D |
| 27–32 | K-fold y holdout (hallazgos metodológicos 3 y 4) |

### Tabla 11 (celdas 19–20): cómo se calculó

- **Muestras:** solo el split de **train** (908: 353 depresivos y 555 no depresivos). La etiqueta se toma de `umap_params.pkl`. Si falta un archivo, la función lo salta sin avisar (`if not os.path.exists(path): continue`), y la salida no imprime los conteos por clase.
- **Imágenes:** las class-blind generadas **en esa misma sesión** (`/kaggle/working/images_grayscale_classblind`, celdas 8–18), en escala de grises de 256×256.
- **Descriptores**, por imagen:
  - entropía de Shannon del histograma de 256 niveles, en bits;
  - contraste como desviación estándar de los niveles de gris (contraste RMS, no GLCM);
  - densidad de bordes como la fracción de píxeles en que la magnitud de Sobel (`skimage.filters.sobel` sobre la imagen en [0, 1]) supera su media + 1 DE;
  - energía LBP como la suma de cuadrados del histograma normalizado de `local_binary_pattern(P=24, R=3, method='uniform')`, con 26 bins.
- **Prueba:** t de Welch (`scipy.stats.ttest_ind(equal_var=False)`) por técnica y descriptor. Holm (`statsmodels multipletests`, α = 0.05) se aplica sobre las 12 comparaciones juntas.
- **Tamaño del efecto:** d de Cohen con la DE combinada como $\sqrt{(s_1^2 + s_2^2)/2}$, donde cada DE usa `ddof=0` (`np.std`). No pondera por tamaño de grupo.

Los d de esta salida son los de `docs/resumen_modulo3_camera_ready.md` §3: Gray-Scott −1.10, −0.39, −0.66 y 1.32; Perlin 0.06, 0.26, 0.95 y −0.06; Lorenz −0.42, −0.14, −0.09 y 0.44.

**Otra ejecución, con Lorenz distinto.** Una versión posterior del mismo día (`notebook32c2241c10 (2).ipynb`, 12:55, misma celda) da en Lorenz d = −0.412 y 0.431 en entropía y LBP, en vez de −0.423 y 0.443. Gray-Scott y Perlin salen idénticos. Lo más probable es que las imágenes de Lorenz se hayan regenerado entre una ejecución y otra; Lorenz renderiza con matplotlib y es el caso más sensible al entorno (`docs/entorno_reproducibilidad.md`, F0.3 H3). **No está verificado que ninguna de las dos corresponda a las imágenes de Lorenz vigentes en `tt2-insumos`.** Si la Tabla 11 se cita en TT2 o se usa en F2.4/F4.4, conviene recalcularla sobre los insumos versionados (`TT_DATA_ROOT`, verificados con `meta/data_manifest.csv`).
