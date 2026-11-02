# Referencias textuales: BETO (Exp 4.1) y clasificador sobre 768D (F2.2)

> Tarea F2.2 del plan v2 · Rol: SV · Estado: **ejecutada** (29-oct-2026; preregistro `228ee85`)
> Las secciones 1 a 5 se versionan con commit propio **antes** de ejecutar. Las secciones 6 y 7 se completan después.
> Insumos y rutas: `docs/datos.md`. Formato de salida: `docs/especificacion_predicciones.md`.

## 1. Propósito

El estadístico principal del Objetivo V es $\Delta_t = F1_{BETO} - \overline{F1}_{img,t}$, y todos los contrastes son pareados. F2.2 produce las predicciones por muestra de las dos referencias textuales:

1. **BETO (Exp 4.1)**, línea base del Objetivo V y término $F1_{BETO}$ de $\Delta_t$.
2. **Un clasificador sobre los embeddings de 768 dimensiones**, que separa el efecto de la cabeza de clasificación de BETO del efecto de la representación de la que se derivan los componentes UMAP.

La tarea resuelve además dos pendientes previos al congelamiento del protocolo (F1.2): (a) el F1-Test reportado de Exp 4.1 (0.7586) no puede obtenerse con los soportes 70/44 del conjunto de prueba, mientras que la matriz de confusión reportada, [[57, 13], [13, 31]], implica 0.7594; (b) la correspondencia fila a fila entre el corpus y `Exp_4.1_BETO_HPO_embeddings.pkl` solo se ha verificado mediante etiquetas, lo que no detecta permutaciones dentro de una misma clase (`docs/datos.md`, §8).

## 2. Decisiones

| Decisión | Valor | Justificación |
|---|---|---|
| Evaluador de BETO | **Cabeza de clasificación** de `CKPT_41` (`BertForSequenceClassification`) | La línea base del Objetivo V es el clasificador textual completo, no un extractor de características |
| Tokenización | Tokenizador de `CKPT_41`; `padding='max_length'`, `truncation=True`, `max_length=128` | Reproduce las condiciones de evaluación de `DepressionDataset` en `3_hyperparameter_tuning.ipynb` |
| Inferencia | CPU, fp32, `model.eval()`, sin gradientes; $p_{dep} = \mathrm{softmax}(\text{logits})_{[:,1]}$ convertido a `float64` | Determinista; no se reentrena, porque el entrenamiento original usó `fp16` y no es reproducible bit a bit |
| Semilla | `seed = 42` | `set_seed(42)` y `TrainingArguments(seed=42)` en el notebook 3 |
| Splits entregados | `val` y `test`; fase `referencia` | `train` no aporta a los contrastes, porque el modelo se ajustó sobre él |
| Clasificador sobre 768D | `StandardScaler` ajustado solo en `X_train` + `LogisticRegression` (L2, `lbfgs`, `max_iter=5000`, sin pesos de clase, `random_state=42`) sobre `EMB_41` | Mismo tipo de evaluador que los experimentos 2.x y 3.x; sin pesos de clase, igual que BETO |
| Hiperparámetro del clasificador | C ∈ 10^[−4, 2], 13 valores logarítmicos; selección por F1-Macro de validación; en empate, el menor C | El conjunto de validación solo se usa para selección (regla 1); el empate favorece el modelo más regularizado (§3.6 del plan) |
| Ajuste final del clasificador | Solo sobre `train`, sin reajustar con `train + val` | Regla 1 |
| Regla de decisión | ŷ = 1 si y solo si $p_{dep} > 0.5$ | Especificación de predicciones, §3.3 |

## 3. Verificaciones

| # | Verificación | Criterio | Si no se cumple |
|---|---|---|---|
| V1 | `config.json` de `CKPT_41`: dos etiquetas y arquitectura de clasificación; `trainer_state.json`: `best_model_checkpoint` termina en `checkpoint-342` | Igualdad exacta | Detener y reportar |
| V2 | F1-Macro de validación recalculado frente al `eval_f1_macro` de la época 3 en `trainer_state.json` | Coincidencia a cuatro decimales (0.7199) | Reportar el número de predicciones que cambian y su distancia a 0.5. La evaluación original usó `fp16`; no se ajusta nada |
| V3 | Embeddings [CLS] recalculados con `AutoModel` desde `CKPT_41`, con el procedimiento de `extraer_embeddings_finetuned` (padding dinámico, lotes de 16, `max_length=128`), frente a `EMB_41`, en las 1,136 filas | Coseno ≥ 0.9999 en cada fila **y** el vecino más cercano de cada fila recalculada dentro de su split es su misma fila | Detener y reportar: la correspondencia corpus ↔ embeddings ↔ u quedaría en duda |
| V4 | Validación de ambos archivos de predicciones con `src.predictions.validate` | Aceptados | Corregir el formato |

## 4. Métricas de prueba

- **BETO.** Se calculan en esta tarea el F1-Macro de prueba, la exactitud, la matriz de confusión y la tasa de falsos negativos. **Es una excepción explícita** a la regla de no calcular métricas de prueba antes de congelar el protocolo, porque BETO es un modelo fijo desde TT-I, su F1-Test ya era conocido y es el ancla del margen δ; debe resolverse antes de F1.2 para que el protocolo no se congele sobre una cifra que no puede obtenerse.
- **Clasificador sobre 768D.** Sus predicciones de prueba se archivan, pero sus métricas de prueba **no** se calculan hasta la Fase 5.

## 5. Consecuencias para las cifras canónicas

1. Las métricas de BETO se registran con estado `verificado` y evaluador `trainer_head`, y sustituyen la entrada `reportado` de Exp 4.1, mediante el mecanismo de sustitución de `src/canonical/build.py`.
2. Se recalculan como cantidades derivadas: $\delta = F1^{test}_{BETO} - F1^{test}_{Exp\,1.2}$ y el Δ exploratorio de WITCOM.
3. Si δ cambia respecto de 0.058, el valor nuevo se registra antes de F1.2, y el protocolo se redacta con él.
4. El F1-Val del clasificador sobre 768D se registra con evaluador `logreg_on_embeddings`; con ello Exp 4.1 obtiene, por primera vez, una cifra comparable con la de Exp 3.1 bajo el mismo tipo de evaluador.

## 6. Resultados

Ejecución en CPU (Windows 11, `.venv` local; versiones en `results/f22_referencias_textuales/README.md`). Salidas en `results/f22_referencias_textuales/` y `results/predictions/referencia/`.

### 6.1 Desviaciones declaradas

1. **Checkpoint de `CKPT_41`.** Con el `checkpoint-342` que había en `exp_4_1_final/` al preregistrar, V3 falló: coseno [CLS] medio de 0.67 y 0/1,136 filas ≥ 0.9999. Ninguna de 18 variantes de extracción (pooler, pooling medio, modo train, modelo base, padding) reprodujo `EMB_41` (`results/f22_referencias_textuales/diagnostico_v3/`). El historial del repositorio de TT-I muestra lo siguiente:
   - el notebook 3 se reentrenó con el mismo código, los mismos hiperparámetros, la misma semilla y la misma ruta el 10 de marzo (commit `6265aae`, F1-Val 0.7328) y el 11 de marzo (commit `effff4a`, F1-Val 0.7199);
   - el checkpoint de Drive era el del 10 de marzo;
   - `EMB_41` (SHA-256 `64b7682e…`, commit `6cfc1ce`) es el del 11 de marzo.

   Se localizó el checkpoint del 11 de marzo y se colocó en `exp_4_1_final/checkpoint-342` (`model.safetensors` `c4f35147…`). `meta/data_manifest.csv` se regeneró (commit `b14b4b9`) y solo cambian las filas de `CKPT_41`. Todo lo que sigue usa ese checkpoint. La regla de §2 no cambia, solo el artefacto que la cumple.
2. **V3, empate entre textos idénticos.** `train_0532` y `train_0584` tienen el mismo texto (mismo `text_sha256`; `id` 1116 y 1050, ambos con etiqueta 1). Sus embeddings son idénticos, y `argmax` asigna a `train_0584` el vecino `train_0532`. Se registra como empate, no como fallo del criterio, que no se editó; `f22_beto.check_v3` lo reporta aparte (commit `1e67498`). La duplicación es un hallazgo del corpus: no cruza splits, así que no afecta los contrastes.
3. **V2.** La referencia es el `eval_f1_macro` de la época 3 en `trainer_state.json`, como dice el criterio. Con el checkpoint correcto vale 0.7199, igual que la cifra entre paréntesis; con el checkpoint anterior valía 0.7328.
4. **Lectura exacta de `p_dep`.** `pd.to_numeric` y el lector por defecto de `read_csv` se desvían 1 ulp en 117 de los 228 valores de BETO. `src/predictions/validate.py` ahora lee `p_dep` con `float()`, de ida y vuelta exacta (commit `fa50f96`, con prueba). No afecta la regla de decisión: el `p_dep` de BETO más cercano a 0.5 está a 0.0038.

### 6.2 Verificaciones

| # | Resultado | Detalle |
|---|---|---|
| V1 | **Se cumple** | `BertForSequenceClassification`, `id2label` = {0: `no_depresivo`, 1: `depresivo`}, `best_model_checkpoint` termina en `checkpoint-342`, `best_metric` = 0.7199476 |
| V2 | **Se cumple** | F1-Val recalculado (CPU, fp32) = 0.7199476 = `eval_f1_macro` de la época 3 (fp16), sin diferencia en ningún decimal |
| V3 | **Se cumple** (desviación 6.1.2) | Coseno mínimo 0.99999999999752 en las 1,136 filas (\|Δ\| máx 2.4 × 10⁻⁵). Vecino más cercano = propia fila en 1,135 filas, más el empate `train_0584` ↔ `train_0532` |
| V4 | **Se cumple** | `python -m src.predictions.validate "results/predictions/referencia/*.csv"` → 2/2 archivos válidos |

La alineación previa también se cumple: la fila *i* de cada CSV coincide con el índice maestro en `id`, `text_sha256` y `y_true`, y las etiquetas de `EMB_41` coinciden con el índice en los tres splits.

### 6.3 Selección de C (LogReg sobre 768D)

| log₁₀ C | −4 | **−3.5** | −3 | −2.5 | −2 | −1.5 | −1 | −0.5 | 0 | 0.5 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1-Val | 0.6874 | **0.7305** | 0.6854 | 0.6484 | 0.6669 | 0.6530 | 0.6453 | 0.6376 | 0.6484 | 0.6406 | 0.6299 | 0.6299 | 0.6222 |

C = 10⁻³·⁵ ≈ 3.16 × 10⁻⁴, sin empate. La curva es irregular y el máximo es un punto aislado cerca del extremo regularizado de la rejilla, por lo que el F1-Val de este modelo es una cifra frágil con n = 114. Tabla completa en `results/f22_referencias_textuales/seleccion_C.csv`.

### 6.4 Métricas de validación

| Modelo | F1-Macro | Exactitud | [[TN, FP], [FN, TP]] | Tasa de FN |
|---|---|---|---|---|
| BETO (cabeza de clasificación, Exp 4.1) | 0.7199 | 0.7368 | [[56, 14], [16, 28]] | 0.364 |
| LogReg sobre 768D (C ≈ 3.16 × 10⁻⁴) | 0.7305 | 0.7456 | [[56, 14], [15, 29]] | 0.341 |

### 6.5 Métricas de prueba de BETO (excepción de §4)

| | F1-Macro | Exactitud | [[TN, FP], [FN, TP]] | Tasa de FN |
|---|---|---|---|---|
| **Recalculado** (checkpoint del 11 de marzo) | **0.7039** | 0.7193 | [[54, 16], [16, 28]] | 0.364 |
| Reportado en TT-I | 0.7586 | — | [[57, 13], [13, 31]] (implica 0.7594) | — |

**No coincide: el F1-Test de BETO es 0.7039, 0.0547 por debajo de lo reportado.** La matriz reportada tampoco corresponde a este modelo. Rastreo de la cifra reportada en el repositorio de TT-I:
- 0.7586 aparece escrito a mano en `5_reduccion_dimensionalidad.ipynb` (`f1_val_orig, f1_te_orig = 0.6886, 0.7586`), etiquetado como F1-Test de un LogReg proxy (`class_weight='balanced'`, `liblinear`) sobre los embeddings de 768D, no de la cabeza de BETO.
- Ese proxy sobre `EMB_41` da un F1-Val de 0.6484, no 0.6886, así que tampoco proviene de `EMB_41` (solo se calculó el F1-Val del proxy). El PKL anterior (3 de marzo) no tiene `X_val`.
- La matriz [[57, 13], [13, 31]] no aparece en el historial.

La procedencia de ambas cifras queda **sin resolver**. Lo verificable es que el modelo que generó `EMB_41`, y por lo tanto `u` y las imágenes, tiene F1-Test 0.7039 con su cabeza de clasificación.

Las predicciones de prueba del LogReg sobre 768D se archivaron sin calcular métricas (§4).

### 6.6 Otros hallazgos

- Al correr la batería completa en esta `.venv` de Windows, `test_C_reproduccion_desde_u[perlin]` y `[lorenz]` (capa C de F0.3) fallan: las imágenes almacenadas no se reproducen bit a bit en este entorno. En el entorno de referencia de F0.3 (Kaggle) sí se reprodujeron (`docs/bitacora/F0.3.md`). F2.2 no modifica `src/generation/` ni las imágenes, cuyos SHA-256 no cambiaron. Queda como pendiente de entorno, fuera del alcance de esta tarea.

## 7. Cifras canónicas actualizadas

Fragmento `results/f22_referencias_textuales/canonical_fragment.json`, generado por `f22_fragment.py` desde los CSV de predicciones. `results/canonical_numbers.json` se reconstruyó con `CANON_STRICT=1`. Para la sustitución y las diferencias se añadieron a `src/canonical/build.py` el campo `supersedes` y la derivación `diff` (commit `c586a02`).

| Id | Estado | Antes | Ahora |
|---|---|---|---|
| `ref.beto_exp41.f1_macro` | reportado → **verificado** (`trainer_head`); sustituye a la entrada de F0.4 | val —, test 0.7586 | val **0.7199**, test **0.7039**, con matrices de confusión |
| `ref.emb768_logreg.f1_macro` | nueva, **verificado** (`logreg_on_embeddings`) | — | val **0.7305**; test bloqueado hasta la Fase 5 |
| `ref.delta_ni.f1_macro` (δ) | nueva, derivada (`diff`), estado reportado | 0.058 (plan v2 §3.2) | test **0.0035** = 0.7039 − 0.7004 |
| `witcom.delta_beto_gs_resnet18.f1_macro` | nueva, derivada (`diff`), estado reportado | 0.0444 (= 0.7586 − 0.7142) | test **−0.0103** = 0.7039 − 0.7142 |

- **xfail.** La entrada reportada (0.7586, marcada xfail por ser inalcanzable con soportes 70/44) salió de `numbers` y se conserva intacta en `superseded`. Su caso xfail desaparece; no pasa a aprobarse. La entrada nueva no lleva marca. El fragmento de F0.4 no se editó.
- **δ cambia de 0.058 a 0.0035.** Según §5.3, este valor queda registrado antes de F1.2 y el protocolo se redacta con él. Con un margen de 0.0035, la prueba de no inferioridad pierde la lectura del plan («el clasificador visual pierde menos de lo que separa a un Transformer de un modelo léxico»): BETO y TF-IDF lematizado quedan prácticamente empatados en prueba. Hay que tratarlo en F1.2 antes de congelar el protocolo.
- **Δ exploratorio de WITCOM.** Pasa de 0.0444 a −0.0103: en prueba, Gray-Scott + ResNet-18 (0.7142, media de 5 semillas) queda por encima de la cabeza de BETO (0.7039, un solo modelo). Sigue siendo exploratorio, porque el test de WITCOM se consultó antes del protocolo.
- **Comparación bajo el mismo evaluador.** En validación, el LogReg sobre 768D (0.7305) supera a la cabeza de BETO (0.7199); su F1-Test se calcula en la Fase 5.
