# F1.3 — Reconocimiento del código de CE (paso 0 sustituto)

> Tarea F1.3 del plan v2 · Hecho por SV en ausencia de CE · 2 nov 2026 · **Solo lectura**: no se modificó código de CE.
> Base: `main` en `076c9a5`. Los números de línea se refieren a ese árbol.

## Código auditado

| Archivo | Último commit | ¿Igual al de F3.2? |
|---|---|---|
| `src/training/build_model.py` | `f1fa300` | Sí |
| `src/training/data.py` | `37e5652` | Sí |
| `src/training/train_model.py` | `37e5652` | Sí |
| `src/training/run_grid.py` | `4175be1` | Lo de entrenamiento sí. Desde `607df40` solo cambió la lectura de `umap_params.pkl` (`umap_io`) y el registro `labels_verified_against` (`git diff 607df40 HEAD -- src/training/`) |

Las corridas de F3.2 usaron `ab81cca` (Gray-Scott) y `607df40` (Perlin, Lorenz), en Tesla T4 con torch 2.11.0+cu128 y cuDNN 91900 (`env` de `results/f32_primera_pasada/runs/*.json`).

Notación: `archivo:línea` sin commit = `076c9a5`.

## A. Configuración real

### Modelo

| Dato | Valor | Fuente |
|---|---|---|
| Pesos preentrenados | `ResNet18_Weights.IMAGENET1K_V1`, `EfficientNet_B0_Weights.IMAGENET1K_V1`, `ViT_B_16_Weights.IMAGENET1K_V1` | `build_model.py:37,44,51` |
| Capa de clasificación | `nn.Linear(in_features, 2)` que reemplaza `fc` (512 → 2), `classifier[-1]` (1280 → 2) y `heads.head` (768 → 2) | `build_model.py:41,48,55` |
| Congelado (`frozen`) | `requires_grad=False` en todos los parámetros **antes** de reemplazar la cabeza, así que solo la cabeza nueva es entrenable | `build_model.py:38-41,45-48,52-55` |
| Modo durante el entrenamiento `frozen` | Todo el modelo en `eval()` y solo la cabeza en `train()`: BatchNorm no actualiza estadísticas y el dropout está apagado (incluido el `Dropout` de `classifier[0]` de EfficientNet, que no forma parte de la cabeza) | `train_model.py:55-63,117` |
| Nombre de la arquitectura ViT | `vit_b_16` | `build_model.py:50`, `run_grid.py:62` |

### Parámetros por combinación

Instanciados en CPU con `weights=None`, cabeza reemplazada como en `build_model.py` (torch 2.11.0+cpu, torchvision 0.26.0). Los entrenables coinciden con `n_trainable_params` de `results/f32_primera_pasada/summary.csv`.

| Arquitectura | Estrategia | Entrenables | Totales | Tamaño fp32 |
|---|---|---:|---:|---:|
| `resnet18` | `finetune` | 11 177 538 | 11 177 538 | 42.6 MiB |
| `resnet18` | `frozen` | 1 026 | 11 177 538 | 42.6 MiB |
| `efficientnet_b0` | `finetune` | 4 010 110 | 4 010 110 | 15.3 MiB |
| `efficientnet_b0` | `frozen` | 2 562 | 4 010 110 | 15.3 MiB |
| `vit_b_16` | `finetune` | 85 800 194 | 85 800 194 | 327.3 MiB |
| `vit_b_16` | `frozen` | 1 538 | 85 800 194 | 327.3 MiB |

Los seis conteos de entrenables son distintos entre sí.

### Optimización

| Dato | Valor | Fuente |
|---|---|---|
| Optimizador | `torch.optim.Adam(params entrenables, lr=lr)`. Sin `weight_decay` (vale 0, el valor por defecto) y con `betas`/`eps` por defecto | `train_model.py:109` |
| Scheduler | No existe | `train_model.py` completo |
| lr | 3e-5 (`finetune`), 1e-3 (`frozen`) | `train_model.py:40,97-98` |
| ¿lr configurable desde el grid? | No: `run_grid.py` no tiene `--lr` ni pasa `lr` a `train_model`, así que siempre usa el valor por defecto | `run_grid.py:376-378` |
| Tamaño de lote | 32 (`--batch-size`) | `run_grid.py:320` |
| Épocas máximas | 15 (`--epochs`) | `run_grid.py:318` |
| Paciencia | 4 (`--patience`) | `run_grid.py:319` |
| Métrica de early stopping | F1-Macro de validación (`f1_score(..., average='macro')`) con predicción `argmax` de los logits | `train_model.py:73,130-131` |
| Criterio de mejora | Estrictamente mayor (`>`): ante un empate se conserva la época anterior | `train_model.py:139` |
| Restauración | Se guarda `deepcopy(state_dict)` en la mejor época y se restaura al final; las predicciones de val que se reportan son las de esa época | `train_model.py:140-141,149` |
| Pérdida | `nn.CrossEntropyLoss(weight=class_weights)` | `train_model.py:110` |
| Pesos de clase | `n / (n_clases · n_c)` (equivalente a `'balanced'`), calculados solo con las etiquetas de train (después de la permutación, si la hay) | `data.py:101-104,155` |
| Valor en F3.2 | `[0.8180, 1.2861]` (clase 0, clase 1) | `class_weights` en `results/f32_primera_pasada/runs/*.json` |
| AMP / fp16 | No existe: todo en fp32 | `train_model.py` completo |

### Determinismo y semillas

| Dato | Valor | Fuente |
|---|---|---|
| `cudnn.deterministic` | `True` | `train_model.py:51` |
| `cudnn.benchmark` | `False` | `train_model.py:52` |
| `torch.use_deterministic_algorithms` | No existe | — |
| Control de kernels de atención (SDPA) | No existe | — |
| `random`, `numpy`, `torch`, `cuda` | `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` | `train_model.py:47-50` |
| Barajado del DataLoader | `torch.Generator` sembrado al crear los loaders y re-sembrado con la semilla de cada corrida, porque los loaders se reutilizan entre semillas de una misma técnica | `data.py:140-141,148`; `train_model.py:102-104`; `run_grid.py:367-374` |
| Workers | `worker_init_fn=_seed_worker` (siembra `numpy` y `random` con `torch.initial_seed()`) | `data.py:107-110,142` |
| `num_workers` | 2 (`--num-workers`) | `run_grid.py:321` |

### Transformaciones

| Dato | Valor | Fuente |
|---|---|---|
| Entrenamiento | `Grayscale(3)` → `Resize((224,224))` si `img_size != 256` → `RandomHorizontalFlip()` si `hflip` → `ToTensor` → `Normalize(ImageNet)` | `data.py:35-37,43-52` |
| Evaluación (val/test) | Igual, sin volteo | `data.py:149-150,153-154` |
| Selección del preprocesamiento | Una sola variante por invocación, igual para todas las técnicas: `--img-size` y `--no-hflip` | `run_grid.py:309-311,327-328` |
| ¿Lee `configs/preprocesamiento.json`? | No | — |
| Variante usada en F3.2 | `px224-flip` en las tres técnicas | `summary.csv`, columna `variant` |

## B. Capacidades

| Capacidad | Estado | Fuente |
|---|---|---|
| Leer configuración desde JSON | **No existe.** Todo se pasa por CLI; `json.load` solo se usa para `grid_meta.json` y los `runs/*.json` | `run_grid.py:161,214,297-325` |
| `--save-models` | Existe. Guarda `model.state_dict()` con `torch.save` en `<out-dir>/models/<run_id>.pt`, en fp32. Tamaño ≈ el de la tabla de parámetros (42.6 / 15.3 / 327.3 MiB), un poco más por los buffers de BatchNorm | `run_grid.py:322-323,411-413` |
| Optuna | **No existe** en el código. Solo está fijado `optuna==5.0.0` en `requirements.txt:22` | — |
| Hiperparámetros del espacio de §6 | `--batch-size` sí existe; `lr` y `weight_decay` **no** se pueden configurar desde el grid | `run_grid.py:320,376-378`; `train_model.py:109` |
| Grad-CAM | **No existe** | — |
| Attention rollout | **No existe.** Además, `EncoderBlock.forward` de torchvision 0.26 llama `self_attention(x, x, x, need_weights=False)`, así que los pesos de atención no se obtienen sin hooks o un parche | torchvision `models/vision_transformer.py`, `EncoderBlock.forward` |
| `--permute-labels SEED` | Existe. Permuta solo las etiquetas de **train** con `np.random.default_rng(SEED).permutation`; val y test conservan las reales. El conteo por clase no cambia, así que los pesos de clase tampoco. La variante lleva el sufijo `-perm<SEED>` y la selección de finalistas la excluye | `data.py:113-116,144-146`; `run_grid.py:77-79,241,312-313` |
| ¿Se guardan los índices permutados? | **No.** El JSON guarda `permute_labels` (la semilla), pero no la permutación ni `y_train` | `run_grid.py:386-407` |
| Qué impide evaluar test sin `--evaluate-test` | Sin la bandera, `make_loaders(..., with_test=False)` no lista ni construye el split de test, y `train_model` recibe `test_loader=None` | `data.py:137,151-154`; `run_grid.py:373`; `train_model.py:151-157` |
| Compuerta con `--evaluate-test` | `check_protocol_frozen` exige que `docs/protocolo_preregistrado.md` exista, esté rastreado por git y no tenga cambios locales; registra commit y fecha en cada corrida. **No comprueba ninguna etiqueta git** | `run_grid.py:109-125,329,394` |
| Separación de modos | `grid_meta.json` impide mezclar corridas con y sin test en un mismo `--out-dir` | `run_grid.py:157-166,338` |
| Matiz | `preflight` lista los **nombres** de archivo de test aun sin `--evaluate-test`, para verificar conteos y etiquetas por nombre; no abre imágenes ni evalúa | `run_grid.py:136-154` |

## C. Costos

### Minutos de GPU en F3.2

Datos de `results/f32_primera_pasada/summary.csv` (columna `minutes`) y `seconds`/`epochs_run` de `runs/*.json`. Tesla T4, 9 corridas por combinación (3 técnicas × 3 semillas).

| Arquitectura | Estrategia | Media min/corrida | Total (9) | Media min/época | Min/corrida a 15 épocas |
|---|---|---:|---:|---:|---:|
| `resnet18` | `finetune` | 1.41 | 12.7 | 0.205 | 3.1 |
| `resnet18` | `frozen` | 0.44 | 4.0 | 0.053 | 0.8 |
| `efficientnet_b0` | `finetune` | 0.78 | 7.0 | 0.093 | 1.4 |
| `efficientnet_b0` | `frozen` | 0.43 | 3.9 | 0.054 | 0.8 |
| `vit_b_16` | `finetune` | 5.11 | 46.0 | 0.572 | 8.6 |
| `vit_b_16` | `frozen` | 1.62 | 14.6 | 0.208 | 3.1 |
| **Total** | | | **88.2** (88.5 según el README de F3.2; diferencia por redondeo) | | |

Una corrida de `resnet18 finetune` tomó 0.61 min/época, contra ≈0.15 en las demás. Parece ser la primera corrida de una sesión (calentamiento de caché de disco/pesos); no está confirmado.

Las corridas duraron en promedio de 5.7 a 10 épocas (`epochs_run`), así que la columna «a 15 épocas» es una cota superior.

### Estimación para las 192 corridas de §4

Supuestos: misma GPU (T4), duración por corrida igual a la media de F3.2 (cota: 15 épocas). Los finalistas aún no están elegidos, así que se dan dos escenarios. La HPO puede cambiar el tamaño de lote, lo que mueve los tiempos; la estimación es aproximada.

| Fase | Corridas | Escenario A: finalistas sugeridos en F3.2 (EffNet FT, EffNet FT, ViT frozen) | Escenario B: peor caso (3 × ViT FT) |
|---|---:|---:|---:|
| F3.2 repetida | 54 | 88 min | 88 min |
| F3.4 HPO (12 trials × 3 semillas) | 108 | 2 × 36 × 0.78 + 36 × 1.62 ≈ 114 min (cota 212) | 108 × 5.11 ≈ 552 min (cota 927) |
| F3.5 final | 15 | ≈ 16 min (cota 30) | ≈ 77 min (cota 129) |
| F3.6 permutación | 15 | ≈ 16 min (cota 30) | ≈ 77 min (cota 129) |
| **Total** | **192** | **≈ 234 min ≈ 3.9 h** (cota ≈ 6 h) | **≈ 794 min ≈ 13.2 h** (cota ≈ 21 h) |

Si F1.4 solo obliga a repetir Perlin y Lorenz, F3.2 cuesta ≈ 2/3 de lo indicado.

### Almacenamiento

- **Modelos (F3.5 + F3.6, 30 archivos `.pt`):** escenario A ≈ 20 × 15.3 + 10 × 327.3 ≈ 3.5 GiB; escenario B ≈ 30 × 327.3 ≈ 9.6 GiB.
- **Mapas (F3.7):** por finalista y semilla, 228 muestras (114 val + 114 test) × 224 × 224 × 2 mapas (normalizado y sin normalizar) × 2 B (`float16`) ≈ 43.6 MiB sin comprimir; × 15 ≈ 654 MiB. Con `np.savez_compressed` debería ser menos (no medido).

## E. Hallazgo 8: no determinismo de ViT fine-tuning

Documentado en `docs/hallazgos_metodologicos.md:116-137` (commit `4175be1`, 21 oct 2026).

- **Observación:** al repetir Gray-Scott con el mismo código y las mismas semillas, ViT fine-tuning dio otro F1-val en las semillas 7 (0.7219 → 0.7137) y 42 (0.7143 → 0.7165). Las CNN y ViT frozen se repitieron al 4.º decimal.
- **Causa:** está documentada como **probable, no verificada**. La hipótesis es que el backward de la atención usa kernels SDPA (flash / memory-efficient) no deterministas, que `cudnn.deterministic` no controla. No se descarta que Kaggle haya asignado una GPU distinta. Los JSON de F3.2 registran `Tesla T4` en las tres técnicas, pero solo se conservó la segunda corrida de Gray-Scott, así que no se puede comparar la GPU de la primera.
- **Mitigación propuesta en el hallazgo:** `torch.backends.cuda.enable_flash_sdp(False)` y `enable_mem_efficient_sdp(False)`, o `torch.use_deterministic_algorithms(True)`. El código **no aplica ninguna** (`train_model.py:43-52`) y no se ha probado.
- **Consecuencia para la matriz:** si un finalista es ViT fine-tuning, F3.5 y F3.6 se reproducen en distribución, no bit a bit.

## Contradicciones entre el borrador y el código

Ordenadas por importancia. Por decisión de SV (2 nov 2026) se registran aquí y **no se cambia §4–§9** de la matriz.

1. **Compuerta de test (§4).** La matriz exige la etiqueta `protocolo-v1` para F3.5 y F3.6. El código solo exige que `docs/protocolo_preregistrado.md` esté commiteado y sin cambios (`run_grid.py:109-125`); no consulta etiquetas. Hoy el repo no tiene ninguna etiqueta (`git tag -l` vacío). **Pendiente para CE:** alinear la compuerta con la etiqueta.
2. **HPO (§6).** No hay Optuna en el código, `lr` no se puede pasar desde el grid y `weight_decay` no existe en el optimizador (`run_grid.py:376-378`, `train_model.py:109`). El *trial* 0 corresponde a `weight_decay=0`, lote 32 y los lr por defecto, que caen dentro de los espacios propuestos.
3. **Preprocesamiento por técnica y JSON de la matriz (§2, §10).** `run_grid.py` aplica una sola variante a todas las técnicas y no lee `configs/preprocesamiento.json` ni `configs/matriz_experimentos.json` (este último no existe). F3.2 se corrió con `px224-flip` en las tres técnicas; Perlin y Lorenz (`px224-noflip`) deben repetirse, como ya prevé el encabezado de la matriz.
4. **Índices de permutación (§9).** La matriz pide guardar la permutación aplicada. El código guarda solo la semilla (`run_grid.py:389`), no los índices.
5. **Interpretabilidad (§8).** No existen Grad-CAM ni attention rollout. Para ViT, torchvision no devuelve los pesos de atención (`need_weights=False`).
6. **Regla de selección (§5).** `aggregate_by_config` desempata por (entrenables, −media) y no usa parámetros totales (`run_grid.py:249`). Hoy no cambia el resultado, porque los seis conteos de entrenables son distintos.
7. **Nombre de ViT (§2, §8).** La matriz usa `vit_b16` y el código `vit_b_16` (`run_grid.py:62`). Hay que unificarlo antes de que una prueba compare `configs/matriz_experimentos.json` con el código.
8. **Determinismo (§3).** `cudnn.deterministic` no basta para ViT fine-tuning (hallazgo 8). Es una limitación declarada, no un error.
9. **Ubicación de modelos y mapas (§9).** El código guarda los modelos en `<out-dir>/models/` (en Kaggle, `/kaggle/working/`). No hay manifiesto con SHA-256 ni ubicación en Drive; el `[completar]` de §9 sigue abierto.
