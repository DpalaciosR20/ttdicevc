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
3. **Módulo 3** (en desarrollo para TT2): clasificación visual de las imágenes generadas (ResNet-18, EfficientNet-B0, ViT-B-16) para medir **cuánta información discriminativa se conserva** al pasar de texto a imagen, contrastando contra BETO (no inferioridad) y contra clasificar directamente los 5 valores UMAP (techo 5D).

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
6. **Rango de u en val/test (verificado con el archivo real):** val/test se guardaron **recortados a [0,1]**, así que no hay parámetros fuera del contrato. El efecto residual es saturación: 20/114 muestras de val y 8/114 de test tienen algún componente en exactamente 0 o 1. Ver `docs/hallazgos_metodologicos.md`, punto 7.
7. **`umap_params.pkl` se lee con `src/training/umap_io.py`**, sin umap-learn, numba ni sklearn. `joblib.load` directo falla por versiones (ver `docs/entorno_reproducibilidad.md`). No re-ajustar UMAP: los `u` nuevos no corresponderían a las imágenes ya generadas.

## Plan vigente: `plan/plan_tt2_v2.md` (19 oct 2026)

El v2 reencuadra TT2: la pregunta ya no es si la imagen *añade* poder discriminativo (eso lo acota la desigualdad de procesamiento de datos, `I(Y;X) ≤ I(Y;u)`), sino **cuánta información se conserva, dónde se pierde y por qué**. El Objetivo V se decide con una prueba de **no inferioridad pareada frente a BETO** (margen preregistrado δ, bootstrap pareado, Holm), con cuatro veredictos posibles (V1 conservación, V2 parcial, V3 no conservación, V4 no concluyente). McNemar pasa a ser complementario: su no significancia con n=114 no demuestra equivalencia.

Reglas del v2 que afectan cualquier trabajo nuevo:
- **El conjunto de prueba está bloqueado** hasta que `docs/protocolo_preregistrado.md` esté congelado (commit fechado, Fase 1). La primera pasada del grid (18 configs × 3 semillas: 42, 7, 123) es **solo validación**. `run_grid.py` hace cumplir esto: sin `--evaluate-test` no carga test, y con él se niega si el protocolo no está commiteado.
- Selección de finalistas por **F1-val promedio entre semillas**; empate dentro de 1 std → el modelo con menos parámetros entrenables. Test se consulta una sola vez por finalista (5 semillas: 42, 7, 123, 2024, 99).
- CE entrega a SV **predicciones por muestra** (`predictions.csv`: `sample_id, split, y_true, technique, arch, strategy, seed, p_dep`), no solo métricas.
- Aumentos y resolución de entrada se validan por técnica (F1.4) antes de fijarlos; por ahora se usan los del camera-ready (224 px + flip horizontal).

## Resultado de WITCOM (exploratorio, no confirmatorio)

Gray-Scott + ResNet-18 fine-tuning (`lr=3e-5`, Adam, CE con pesos de clase, batch=32, paciencia=4 sobre F1-val): **F1-Macro = 0.7142 ± 0.0095** (5 semillas). Referencias: 5D-UMAP LogReg/SVM=0.7246, RF=0.7304, MLP=0.6933; BETO=0.7586; McNemar CNN vs LogReg-5D no significativo.

Su conjunto de prueba ya se consultó antes de fijar el protocolo del v2, así que **se reporta como evidencia exploratoria y no participa en la selección de finalistas**. Sirve como chequeo de cordura del pipeline (si ResNet-18 en Gray-Scott da algo muy distinto en validación, revisar antes de seguir), no como conclusión del proyecto.

## Alcance de TT2 (lo que falta)

Ver `plan/plan_tt2_v2.md`: Fase 0 (reproducibilidad), Fase 1 (protocolo preregistrado), Fase 2 (techo 5D y pisos), Fase 3 (grid 18 configs × 3 semillas → finalistas → HPO → 5 semillas con test → control de permutación → Grad-CAM/attention rollout), Fase 4 (análisis de preservación, SV), Fase 5 (veredicto), Fases 6–8 (síntesis, documento, defensa). `plan/plan_tt2_por_semana.md` es el plan v1, conservado como historial.

## Estructura del repo

```
ttdicevc/
├── CLAUDE.md                          ← este archivo
├── README.md
├── .gitignore
├── requirements.txt                   ← versiones exactas del entorno de Kaggle (F0.2)
├── results/                           ← resultados versionados (p. ej. f32_primera_pasada/)
├── src/
│   ├── generation/                    ← generadores class-blind de las 3 técnicas
│   │   ├── gray_scott_lib.py
│   │   ├── perlin_lib.py
│   │   └── lorenz_lib.py
│   └── training/                      ← entrenamiento generalizado (3 arquitecturas)
│       ├── build_model.py
│       ├── data.py                    ← preprocesamiento/loaders compartidos (3ch, 224, ImageNet)
│       ├── train_model.py
│       ├── run_grid.py                ← grid 18 configs / multi-seed, reanudable (ver docs/correr_grid_kaggle.md)
│       └── umap_io.py                 ← lee umap_params.pkl sin umap/numba (F0.2)
├── tests/
│   └── smoke_test_training.py         ← 3 arquitecturas × frozen/fine-tune, corre en CPU
├── notebooks/                         ← notebooks de Kaggle/Colab (limpiar outputs pesados antes de commitear)
├── docs/
│   ├── artefactos_datos.md            ← dónde están los .pkl y los datasets de imágenes
│   ├── correr_grid_kaggle.md          ← cómo correr run_grid.py en Kaggle
│   ├── entorno_reproducibilidad.md    ← entorno fijado y lectura de umap_params.pkl (F0.2)
│   ├── hallazgos_metodologicos.md     ← detalle completo de las 4 fugas y cómo se resolvieron
│   └── resumen_modulo3_camera_ready.md
└── plan/
    ├── plan_tt2_v2.md                 ← plan vigente
    └── plan_tt2_por_semana.md         ← plan v1 (historial)
```

## Roles del equipo (para tareas de TT2)

- 🔵 **CE (Diego)** — consolidación del pipeline, entrenamiento de clasificadores visuales, HPO, controles que requieren reentrenamiento.
- 🟢 **SV (Santiago)** — protocolo preregistrado, techo y pisos, análisis de preservación (Fase 4), contraste estadístico y veredicto.
- 🟠 **RP (Marco)** — documento de TT2, corrección de capítulos heredados de TT1, coherencia de cifras, preparación de la defensa.

## Convenciones de código

- Python, PyTorch/torchvision para modelos, scikit-learn para baselines clásicos y métricas.
- Todo entrenamiento debe usar semilla fija (`torch.manual_seed` + `torch.cuda.manual_seed_all`) para reproducibilidad — ya se observó variación real de varios puntos porcentuales entre entornos sin esto.
- Reportar F1-Macro (no accuracy) por el desbalance de clases (~61/39%).
- Al seleccionar la mejor configuración entre varias, usar F1 de **validación**, nunca de test, para no contaminar la decisión.
- No evaluar ni mirar test fuera de la tarea F3.5 del plan v2 (finalistas, protocolo congelado). Si alguien lo consulta antes, se declara como desviación del protocolo.
- Cualquier resultado de un solo run debe tratarse con sospecha — preferir multi-seed (mínimo 5) antes de reportar un número como definitivo, dado que ya se observó variación de ~8pp entre semillas/entornos en corridas nominalmente idénticas.
- Generación de imágenes: usar la versión GPU-batched (`generate_batch_gpu` en `gray_scott_lib.py`) cuando haya GPU disponible — es órdenes de magnitud más rápida que la versión CPU/numpy puro.
- En Kaggle: `/kaggle/input/` es de solo lectura; todo lo que se genere debe ir a `/kaggle/working/`.

## Convención de commits y PRs

El asunto de cada commit y el título de cada PR empiezan con el ID de la tarea de `plan/plan_tt2_v2.md` que atienden, entre corchetes:

- Una tarea: `[F3.2] Primera pasada del grid solo con validación`
- Varias tareas: `[F1.5, F3.3] Predicciones por muestra y resumen por configuración`
- Cambios que no corresponden a ninguna tarea del plan (mantenimiento, herramientas, convenciones): `[repo] ...`

Las ramas pueden llevar el ID también (p. ej. `ce/f3.4-hpo`), pero no es obligatorio.
