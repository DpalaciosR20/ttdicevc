# TTDICEVC — Contexto del Proyecto para Claude Code

Este archivo se carga automáticamente al inicio de cada sesión de Claude Code en este repo. Da contexto del proyecto para que cualquiera de los 3 integrantes pueda trabajar con Claude Code sin tener que re-explicar el proyecto desde cero.

## Qué es este proyecto

**TTDICEVC** (Transformación de Textos Depresivos en Imágenes mediante Visión Computacional) es un Trabajo Terminal de ESCOM-IPN. Equipo: Diego Palacios Robles, Marco Antonio Mánica Quintero, Santiago Elihu Soria Solís. Director: Dr. Abdiel Reyes Vera.

Un paper derivado de este proyecto (WITCOM 2026) **ya fue aceptado y publicado** — esa fase está cerrada. El trabajo activo ahora es **TT2** (Trabajo Terminal II, tesis formal), con defensa entre el 17 de noviembre y el 1 de diciembre de 2026, meta interna de tenerlo listo a finales de octubre.

## Qué hace el proyecto (resumen técnico)

Se transforma texto (depresivo/no depresivo, corpus SPADE-BERT en español, 1,136 muestras, split 80/10/10 = 908/114/114) en imágenes abstractas deterministas mediante 3 técnicas generativas, con el objetivo de clasificar esas imágenes por visión computacional y comparar el resultado contra un clasificador de texto.

**Pipeline (3 módulos):**
1. **Módulo 1** (completo): BETO (BERT en español) fine-tuneado + HPO con Optuna → embeddings de 768D. F1-Macro final = 0.7586. **No usó class weights** (a diferencia del clasificador clásico de referencia, que sí usa `class_weight='balanced'`).
2. **Módulo 2** (completo): UMAP reduce 768D → 5D (`n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42`, luego `MinMaxScaler`). Esos 5 valores se mapean de forma determinista a los parámetros de control de 3 algoritmos generativos independientes:
   - **Perlin Flow Fields** — `src/generation/perlin_lib.py`
   - **Atractor de Lorenz** — `src/generation/lorenz_lib.py`
   - **Reacción-Difusión Gray-Scott** — `src/generation/gray_scott_lib.py`
3. **Módulo 3** (en desarrollo para TT2): clasificación visual de las imágenes generadas con CNNs (ResNet-18, EfficientNet-B0, ViT-B-16), comparando contra clasificar directamente los 5 valores UMAP y contra BETO.

## Artefactos de datos (no están en este repo — ver `docs/artefactos_datos.md`)

- `umap_params.pkl`: reductor UMAP ya ajustado + scaler + los 5 componentes reducidos/normalizados para train/val/test, con labels.
- `Exp_4_1_BETO_HPO_embeddings.pkl`: embeddings BETO de 768D (necesarios solo si se reajusta UMAP).
- 3 datasets de imágenes class-blind (Gray-Scott, Perlin, Lorenz), 256×256, escala de grises, 1,136 c/u. Nomenclatura: `<prefijo>_<split>_<idx:04d>_label<0o1>.png` (prefijos: `rd` para Gray-Scott, `perlin`, `lorenz`).

Estos archivos son grandes/binarios — viven en Drive/Kaggle Datasets, no en git. Ver `.gitignore`.

## Hallazgos metodológicos ya resueltos — NO los repitas

Ver `docs/hallazgos_metodologicos.md` para el detalle completo. Resumen:

1. **Fuga de color por clase**: las imágenes originales del Módulo 2 estaban coloreadas por clase (paleta fría=depresivo, cálida=no depresivo). Ya se regeneraron las 3 técnicas en verdadero class-blind (escala de grises real, el renderizador nunca recibe la etiqueta).
2. **Fuga de semilla en Perlin**: el generador original usaba `seed = label*10000 + i` — corregido, ahora la semilla es un hash determinista de los 5 valores UMAP.
3. **Fuga de UMAP no reajustado en k-fold**: combinar train+val+test y re-particionar para k-fold infla los resultados porque UMAP ya vio esos puntos en su ajuste original. Se corrige reajustando UMAP por fold (k-fold anidado) — código de referencia en `docs/hallazgos_metodologicos.md`.
4. **Fuga de fine-tuning de BETO**: incluso con UMAP reajustado por fold, los resultados de k-fold/holdout siguen inflados (~0.80-0.86) porque BETO fue fine-tuneado específicamente sobre las 908 muestras del split de train original, y cualquier re-partición que no respete esa frontera hereda esa ventaja. **No se puede arreglar del todo sin reentrenar BETO por fold (no viable en el tiempo disponible) — se documenta como limitación conocida, no se persigue.**
5. **Conclusión operativa:** el split fijo 80/10/10 original es la referencia confiable para reportar resultados. Cualquier k-fold/holdout adicional se reporta como "estimado optimista" con esta limitación explicada.

## Resultado ya validado (Gray-Scott + ResNet-18, referencia para comparar nuevas arquitecturas)

Config: `lr=3e-5`, Adam, `CrossEntropyLoss` con pesos de clase, batch=32, early stopping paciencia=4 sobre F1-val, fine-tuning completo.

- **CNN Gray-Scott (media de 5 semillas: 42, 7, 123, 2024, 99): F1-Macro = 0.7142 ± 0.0095**
- Clasificar directo los 5 valores UMAP: LogReg/SVM=0.7246, RF=0.7304, MLP=0.6933
- BETO (texto): 0.7586
- McNemar (CNN vs. LogReg-5D-UMAP): consistentemente no significativo (p=1.0)

**Conclusión central del proyecto:** el CNN no supera de forma consistente ni significativa a clasificar directamente los 5 valores UMAP — no hay evidencia de que la transformación a imagen añada poder discriminativo sobre la proyección que la genera. Esta es una conclusión validada y estable; cualquier trabajo nuevo debe contrastarse contra ella, no asumir que "más arquitectura = debe ganar".

## Alcance de TT2 (lo que falta)

Clasificación visual con 3 arquitecturas (ResNet-18, EfficientNet-B0, ViT-B-16) × 2 estrategias (frozen, fine-tuning) × 3 técnicas generativas = 18 configuraciones, + Grad-CAM (interpretabilidad) + comparación estadística formal (McNemar, Holm) + análisis comparativo entre las 3 técnicas. Ver `plan/plan_tt2_por_semana.md` para el desglose semana a semana con tareas por rol.

## Estructura del repo

```
ttdicevc/
├── CLAUDE.md                          ← este archivo
├── README.md
├── .gitignore
├── src/
│   ├── generation/                    ← generadores class-blind de las 3 técnicas
│   │   ├── gray_scott_lib.py
│   │   ├── perlin_lib.py
│   │   └── lorenz_lib.py
│   └── training/                      ← entrenamiento generalizado (3 arquitecturas)
│       ├── build_model.py
│       ├── data.py                    ← preprocesamiento/loaders compartidos (3ch, 224, ImageNet)
│       ├── train_model.py
│       └── run_grid.py                ← grid 18 configs / multi-seed, reanudable (ver docs/correr_grid_kaggle.md)
├── tests/
│   └── smoke_test_training.py         ← 3 arquitecturas × frozen/fine-tune, corre en CPU
├── notebooks/                         ← notebooks de Kaggle/Colab (limpiar outputs pesados antes de commitear)
├── docs/
│   ├── artefactos_datos.md            ← dónde están los .pkl y los datasets de imágenes
│   ├── correr_grid_kaggle.md          ← cómo correr run_grid.py en Kaggle
│   ├── hallazgos_metodologicos.md     ← detalle completo de las 4 fugas y cómo se resolvieron
│   └── resumen_modulo3_camera_ready.md
└── plan/
    └── plan_tt2_por_semana.md
```

## Roles del equipo (para tareas de TT2)

- 🔵 **CE (Diego)** — código, experimentos, entrenamiento de modelos.
- 🟢 **SV** — diseño estadístico, validación, Grad-CAM, corrección de métricas.
- 🟠 **RP** — redacción del documento de TT2, formato de tesis, integración de resultados.

## Convenciones de código

- Python, PyTorch/torchvision para modelos, scikit-learn para baselines clásicos y métricas.
- Todo entrenamiento debe usar semilla fija (`torch.manual_seed` + `torch.cuda.manual_seed_all`) para reproducibilidad — ya se observó variación real de varios puntos porcentuales entre entornos sin esto.
- Reportar F1-Macro (no accuracy) por el desbalance de clases (~61/39%).
- Al seleccionar la mejor configuración entre varias, usar F1 de **validación**, nunca de test, para no contaminar la decisión.
- Cualquier resultado de un solo run debe tratarse con sospecha — preferir multi-seed (mínimo 5) antes de reportar un número como definitivo, dado que ya se observó variación de ~8pp entre semillas/entornos en corridas nominalmente idénticas.
- Generación de imágenes: usar la versión GPU-batched (`generate_batch_gpu` en `gray_scott_lib.py`) cuando haya GPU disponible — es órdenes de magnitud más rápida que la versión CPU/numpy puro.
- En Kaggle: `/kaggle/input/` es de solo lectura; todo lo que se genere debe ir a `/kaggle/working/`.
