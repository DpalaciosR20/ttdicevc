# Referencias textuales: BETO (Exp 4.1) y clasificador sobre 768D (F2.2)

> Tarea F2.2 del plan v2 · Rol: SV · Estado: **criterio fijado; resultados pendientes**
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

[completar: V1–V4; selección de C; métricas de validación de ambos modelos; métricas de prueba de BETO frente a las reportadas]

## 7. Cifras canónicas actualizadas

[completar: entradas sustituidas o añadidas, valor anterior y nuevo, y δ resultante]
