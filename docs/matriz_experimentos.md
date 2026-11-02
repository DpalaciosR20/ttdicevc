# Matriz de experimentos del Módulo 3 (F1.3)

> Tarea F1.3 del plan v2 · Roles: SV + CE · Estado: **decisiones de SV provisionales; pendiente de visto bueno de CE (§12)**
> Se versiona antes de repetir la primera pasada (F3.2) y forma parte del protocolo preregistrado (F1.2).
> Fuentes únicas legibles por código: `configs/matriz_experimentos.json` (este documento) y `configs/preprocesamiento.json` (F1.4).
> Reconocimiento del código: `results/f13_matriz/reconocimiento_ce.md` (base `main` @ `076c9a5`).

## 1. Propósito

Fijar, antes de producir los resultados que se usarán para seleccionar y evaluar, todas las corridas del Módulo 3: qué se entrena, con qué configuración, con qué semillas, cómo se elige el finalista de cada técnica, cómo se ajustan sus hiperparámetros, qué controles se ejecutan y con qué método se interpreta cada arquitectura. Atiende las amenazas «sesgo de selección en el grid», «BETO con HPO, visuales sin ella», «variabilidad entre semillas / cuDNN» y «mapas de atribución no fiables» (sección 5 del plan).

## 2. Factores

| Factor | Niveles |
|---|---|
| Técnica | `perlin`, `gray_scott`, `lorenz` |
| Arquitectura | `resnet18`, `efficientnet_b0`, `vit_b16` |
| Estrategia | `frozen` (solo la capa de clasificación es entrenable), `finetune` (todos los parámetros entrenables) |
| Preprocesamiento | Uno por técnica, leído de `configs/preprocesamiento.json`: `perlin` = `px224-noflip`, `gray_scott` = `px224-flip`, `lorenz` = `px224-noflip` |

18 configuraciones (técnica × arquitectura × estrategia). El preprocesamiento no es un factor del grid. Los nombres siguen el vocabulario controlado de F1.5 (§3.2 de la especificación de predicciones); el código interno puede usar `vit_b_16`, pero toda salida y este JSON usan `vit_b16`.

## 3. Configuración común de entrenamiento

Fuente: `results/f13_matriz/reconocimiento_ce.md`, bloque A. Referencias `archivo:línea` sobre `main` @ `076c9a5`.

| Parámetro | Valor | Fuente |
|---|---|---|
| Pesos preentrenados | `IMAGENET1K_V1` en las tres: `ResNet18_Weights`, `EfficientNet_B0_Weights`, `ViT_B_16_Weights` | `build_model.py:37,44,51` |
| Entrada | 1 canal replicado a 3 (`Grayscale(3)`); normalización con media y DE de ImageNet; tamaño y volteo según `configs/preprocesamiento.json` | `data.py:35-52`; F1.4 |
| Capa de clasificación | `nn.Linear(·, 2)`; reemplaza `fc` (512), `classifier[-1]` (1280) y `heads.head` (768) | `build_model.py:41,48,55` |
| Modo `frozen` | *Backbone* en `eval()` (BatchNorm sin actualizar estadísticas, *dropout* apagado, incluido `classifier[0]` de EfficientNet) y cabeza en `train()` | `train_model.py:55-63,117` |
| Optimizador | AdamW con `betas` y `eps` por defecto, sin *scheduler*. Con `weight_decay = 0` es idéntico al Adam usado en la primera pasada v0 | `train_model.py:109` (Adam en v0) |
| Tasa de aprendizaje | 3e-5 (`finetune`), 1e-3 (`frozen`) | `train_model.py:40,97-98` |
| *Weight decay* | 0 | — |
| Tamaño de lote | 32 | `run_grid.py:320` |
| Épocas máximas | 15 | `run_grid.py:318` |
| *Early stopping* | F1-Macro de validación con `argmax` de los *logits* (equivalente a `p_dep > 0.5`); mejora estricta; paciencia 4; se restaura la mejor época | `train_model.py:73,130-141,149`; `run_grid.py:319` |
| Pérdida | `CrossEntropyLoss` con pesos de clase `n / (2 · n_c)` calculados sobre `train` ([0.8180, 1.2861]) | `data.py:101-104,155`; `train_model.py:110` |
| Precisión | fp32, sin AMP | `train_model.py` |
| Semillas | `random`, `numpy`, `torch`, `cuda`; `torch.Generator` del *DataLoader* re-sembrado por corrida; `worker_init_fn` por *worker*; `num_workers = 2` | `train_model.py:47-50,102-104`; `data.py:107-110,140-148`; `run_grid.py:321` |
| Determinismo | `cudnn.deterministic = True`, `cudnn.benchmark = False`. **A partir de la repetición de F3.2:** además, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` y kernels SDPA *flash* y *memory-efficient* deshabilitados (§11, punto 6) | `train_model.py:51-52`; hallazgo 8 |

**Asimetría declarada.** BETO (Exp 4.1) se entrenó sin pesos de clase y los clasificadores visuales con ellos. Se mantiene porque cada referencia usa su receta ajustada; se declara como limitación y su efecto se acota en F5.5.

## 4. Fases del Módulo 3

| Fase | Tarea | Corridas | Semillas | Splits con predicciones | Requisito |
|---|---|---|---|---|---|
| `primera_pasada` | F3.2 | 18 configuraciones | 42, 7, 123 | `val` | Este documento versionado en `main` |
| — | F3.3 | Selección de un finalista por técnica (§5) | — | — | Primera pasada completa |
| — | F3.4 | HPO de cada finalista (§6) | 42, 7, 123 | `val` (solo en registros de HPO, no en `results/predictions/`) | F3.3 |
| `final` | F3.5 | 3 finalistas con hiperparámetros de F3.4 | 42, 7, 123, 2024, 99 | `val`, `test` | `docs/protocolo_preregistrado.md` versionado sin cambios **y** etiqueta `protocolo-v1` en el historial |
| `control_permutacion` | F3.6 | 3 finalistas con `y_train` permutado (§7) | 42, 7, 123, 2024, 99 | `val`, `test` | Ídem `final` |
| — | F3.7 | Mapas de atribución de los modelos de F3.5 y F3.6 (§8) | Las 5 de F3.5 | `val`, `test` | Modelos guardados |

Total de entrenamientos: 54 (F3.2) + 3 × 12 × 3 = 108 (F3.4) + 15 (F3.5) + 15 (F3.6) = 192. Costo estimado en T4: ≈ 3.9 h si los finalistas son los sugeridos por v0 y ≈ 13.2 h en el peor caso (tres ViT `finetune`), con cotas de 6 h y 21 h (reconocimiento, bloque C).

## 5. Selección del finalista (F3.3)

1. Para cada configuración se calcula la media y la desviación estándar muestral (`ddof = 1`) del F1-Macro de validación sobre las semillas 42, 7 y 123, a partir de las predicciones de la fase `primera_pasada` cargadas con `load_predictions`.
2. El **líder** de la técnica es la configuración con mayor media.
3. Están **empatadas** con el líder las configuraciones cuya media es ≥ media del líder − DE del líder. Si varias configuraciones comparten la media máxima, se usa la mayor de sus DE.
4. Entre las empatadas, el finalista es la de **menor complejidad**, con este orden lexicográfico:
   1. menor número de parámetros entrenables;
   2. menor número de parámetros totales.
5. Si persiste el empate, la de mayor media. Si aún persiste, la selección se detiene con un error y la resuelve SV.

Fundamento del criterio de complejidad: el riesgo de sobreajuste a 908 muestras de entrenamiento y a 114 de validación depende de los parámetros que se ajustan a los datos, no de los que se heredan fijos del preentrenamiento. En consecuencia, toda configuración `frozen` es más simple que cualquier `finetune`. Orden resultante: `resnet18 frozen` (1,026) < `vit_b16 frozen` (1,538) < `efficientnet_b0 frozen` (2,562) < `efficientnet_b0 finetune` (4.01 M) < `resnet18 finetune` (11.18 M) < `vit_b16 finetune` (85.80 M).

La selección oficial la produce `src/selection/select_finalist.py`. La columna `seleccion` que escribe `run_grid.py` es informativa.

**Declaración.** La regla de §3.6 del plan se conocía y la primera pasada v0 ya se había ejecutado cuando se fijó el criterio de complejidad; los resultados v0 están sustituidos y no se usan en la selección.

## 6. Ajuste de hiperparámetros de los finalistas (F3.4)

| Parámetro | Valor | Justificación |
|---|---|---|
| Algoritmo | Optuna, TPE (`seed = 42`, parámetros por defecto), sin *pruner* | Mismo esquema que la HPO de BETO |
| Presupuesto | **12 *trials*** por finalista | Igual al de BETO; atiende la amenaza «BETO con HPO, visuales sin ella» |
| Objetivo | Media del F1-Macro de validación sobre las semillas 42, 7 y 123 | Reduce el ruido de selección respecto de una sola semilla |
| *Trial* 0 | La configuración de §3 (`enqueue_trial`): lr por defecto, `weight_decay = 0`, lote 32 | Garantiza que la HPO no empeora la configuración de partida en validación |
| Espacio (`finetune`) | `lr` ∈ [5e-6, 3e-4], logarítmico; `weight_decay` ∈ [0, 0.1], uniforme; lote ∈ {16, 32, 64} (`vit_b16`: {16, 32}) | Rango habitual de ajuste fino con Adam; lote 64 de ViT-B/16 en fp32 no cabe con margen en una T4 |
| Espacio (`frozen`) | `lr` ∈ [1e-4, 1e-2], logarítmico; `weight_decay` ∈ [0, 0.1], uniforme; lote ∈ {16, 32, 64} | Solo se entrena una capa lineal |
| Fijos | Arquitectura, estrategia, preprocesamiento, optimizador, épocas máximas, paciencia, pesos de clase | — |
| Regla final | Se adoptan los hiperparámetros del mejor *trial*, salvo que su media no supere a la del *trial* 0 en más de una DE del *trial* 0; en ese caso se conserva la configuración de partida | Simplicidad ante mejoras dentro del ruido |
| Registro | Estudio de Optuna serializado, tabla de *trials* y F1-Val por semilla | — |

**Declaración.** Con 12 *trials* y los 10 de arranque por defecto de TPE, la búsqueda es casi aleatoria; ocurrió lo mismo en la HPO de BETO, por lo que la paridad se mantiene.

## 7. Control de permutación (F3.6)

- Se permuta `y_train` con `numpy.random.default_rng(99)`; `y_val` y `y_test` no se alteran. La permutación conserva el conteo por clase, de modo que los pesos de clase no cambian.
- Cada finalista se entrena con sus hiperparámetros de F3.4 y las cinco semillas de F3.5; el *early stopping* usa el F1-Val con etiquetas reales.
- Se guardan los índices de la permutación aplicada.
- Su F1-Macro de prueba promedio entra en el piso de referencia del protocolo, junto con los pisos de F2.3.
- Sus modelos se guardan: sirven como prueba de cordura por aleatorización de datos en F4.3.

## 8. Interpretabilidad (F3.7)

| Arquitectura | Método | Capa o detalle |
|---|---|---|
| `resnet18` | Grad-CAM | Salida de `layer4` |
| `efficientnet_b0` | Grad-CAM | Salida del último bloque de `features` |
| `vit_b16` | *Attention rollout* (Abnar y Zuidema, 2020) | Fusión de cabezas por media, identidad añadida y renormalizada, sin descarte; se excluye el *token* de clase del mapa final |

- **Objetivo del mapa:** el *logit* de la clase Depresivo (1) en todas las muestras, para que el signo sea comparable entre muestras y clases.
- **Muestras:** las 114 de `val` y las 114 de `test` para las cinco semillas de cada finalista.
- **Resolución:** mapas reescalados a 224 × 224, normalizados por muestra a [0, 1]; se guarda además el mapa sin normalizar.
- **Formato:** `float16`, un archivo `.npz` comprimido por finalista y semilla, indexado por `sample_id`.
- **Modelos `frozen`:** el cálculo de gradientes requiere habilitar `requires_grad` en las activaciones de la capa objetivo, porque los parámetros del *backbone* están congelados.
- **ViT:** `EncoderBlock.forward` de torchvision llama a la atención con `need_weights=False`; los pesos se obtienen con *hooks* que recalculan la atención con `need_weights=True` y `average_attn_weights=False`. Una prueba verifica que la salida del modelo no cambia con los *hooks*.
- **Pruebas de cordura (para F4.3):** aleatorización en cascada de los parámetros (Adebayo et al., 2018) y aleatorización de datos con los modelos de F3.6.

## 9. Artefactos obligatorios

| Fase | Predicciones (F1.5) | Modelos guardados | Otros |
|---|---|---|---|
| `primera_pasada` | Sí | No | Historial por época, `summary.csv`, `summary_by_config.csv` |
| F3.4 | No | No | Estudios de Optuna |
| `final` | Sí | **Sí** | Historial por época |
| `control_permutacion` | Sí | **Sí** | Índices de la permutación |
| F3.7 | — | — | Mapas `.npz` |

Los modelos y los mapas no se versionan en git. Se publican como versiones de un *dataset* privado de Kaggle (`tt2-artefactos`), del que los consume F3.7 sin descarga intermedia, con copia en Drive en `tt2-artefactos/`, fuera de `tt2-insumos/`. Su SHA-256 se registra en `meta/artefactos_manifest.csv`, versionado. Volumen estimado: 3.5–9.6 GiB de modelos y < 0.7 GiB de mapas.

## 10. Representación legible por código

`configs/matriz_experimentos.json` contiene los factores (§2), la configuración común (§3), las semillas por fase (§4), la regla de selección (§5), el espacio y presupuesto de HPO (§6) y los métodos de interpretabilidad (§8). `run_grid.py` lee de este archivo las configuraciones y semillas; una prueba verifica que el JSON coincide con este documento.

## 11. Pendientes de implementación (CE)

Contradicciones entre esta matriz y el código vigente, en orden de necesidad:

| # | Pendiente | Necesario para |
|---|---|---|
| 1 | `run_grid.py` lee `configs/preprocesamiento.json` y `configs/matriz_experimentos.json` | Repetición de F3.2 |
| 2 | Salida en el formato de F1.5 (`vit_b16`, un CSV por corrida, manifiesto, `p_dep` sin redondear) | Repetición de F3.2 |
| 3 | Optimizador AdamW con `weight_decay` configurable; `lr` y `weight_decay` configurables desde el grid | Repetición de F3.2 (con valores de §3) y F3.4 |
| 4 | Integración de Optuna según §6 | F3.4 |
| 5 | La compuerta de `--evaluate-test` exige también la etiqueta `protocolo-v1` | F3.5, F3.6 |
| 6 | Ajustes de determinismo de §3 y prueba de humo: dos corridas de `vit_b16 finetune` (semilla 42, 2 épocas) con pesos finales idénticos. Si no lo son, se registra y se mantiene la limitación del hallazgo 8 | Repetición de F3.2 |
| 7 | Guardado de los índices de la permutación | F3.6 |
| 8 | Grad-CAM y *attention rollout* según §8 | F3.7 |
| 9 | *Dataset* `tt2-artefactos` y `meta/artefactos_manifest.csv` | F3.5 |

## 12. Decisiones provisionales de SV (pendientes de visto bueno de CE)

Tomadas en ausencia de CE el 2 de noviembre de 2026. CE las aprueba o propone alternativas en la revisión del PR; cualquier cambio se incorpora antes de repetir la primera pasada.

| Decisión | Sección |
|---|---|
| Criterio de complejidad por parámetros entrenables y, en segundo lugar, totales | §5 |
| Varios líderes con la misma media: tolerancia con la mayor DE; empate residual: error | §5 |
| AdamW con `weight_decay = 0` en la configuración común | §3 |
| Espacios, presupuesto, *trial* 0 y regla final de la HPO | §6 |
| Ajustes de determinismo desde la repetición de F3.2 | §3, §11.6 |
| Cinco semillas en el control de permutación | §7 |
| Almacenamiento de modelos y mapas | §9 |
| Mantener la asimetría de pesos de clase con BETO | §3 |
