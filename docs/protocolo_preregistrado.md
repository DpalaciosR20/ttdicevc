# Protocolo preregistrado del Objetivo V

> Tarea F1.2 del plan v2 · Rol: SV · Versión: **v1 (congelado)**
> Congelamiento: commit `91c84dfa7e192a32403abdaf401bb8f6ae5ff179`, etiqueta anotada `protocolo-v1`, fecha `2026-11-02 18:14:54 -0600`.
> Parámetros legibles por código: `configs/protocolo.json` (SHA-256 `92ef8774f10f6c7ca04abd71a6a55be6ef29852b049932fb54dcecfe7ee00e5a`).
> Matriz de experimentos: `configs/matriz_experimentos.json` (SHA-256 `ce0e0ed6602ba0f53441eb77dbd90c7704e6d7f2b877d6f0ea9c295c689a8f15`). Huellas calculadas con fines de línea LF; `tests/test_protocolo.py` las verifica.
> Enmienda previa de la matriz: fase `sensibilidad` (F1.2).
> Toda cifra citada aquí proviene de `results/canonical_numbers.json`.

## 1. Pregunta, hipótesis y alcance

**Pregunta.** ¿Se conservan las características emocionales de los textos al transformar sus embeddings en imágenes abstractas mediante tres técnicas generativas (ruido de Perlin, reacción-difusión de Gray-Scott y atractor de Lorenz)?

**Hipótesis.** Para cada técnica *t*, el mejor clasificador visual preespecificado conserva al menos la mitad de la capacidad discriminativa de BETO por encima del azar sobre el conjunto de prueba.

**Alcance.** El protocolo rige el contraste confirmatorio del Objetivo V (Fase 5) y el control de longitud (F4.5). El resto de los análisis de la Fase 4 es explicativo (§9). Cualquier análisis no descrito aquí se reporta como exploratorio.

## 2. Datos y uso de los conjuntos

| Conjunto | n | Clases (0 / 1) | Uso permitido |
|---|---|---|---|
| Entrenamiento | 908 | 555 / 353 | Ajuste de modelos y transformadores |
| Validación | 114 | 70 / 44 | *Early stopping*, HPO y selección |
| Prueba | 114 | 70 / 44 | Una inferencia por modelo y semilla, en las fases autorizadas (§10); contraste confirmatorio |

Las muestras se identifican por `sample_id` del índice maestro (`meta/sample_index.csv`). Todas las predicciones cumplen `docs/especificacion_predicciones.md`.

## 3. Información conocida al congelar

Al congelar este protocolo ya se conocían los siguientes resultados, que pudieron influir en su diseño:

| Información | Origen |
|---|---|
| F1-Macro de BETO: validación 0.7199, prueba 0.7039 | F2.2 |
| Δ exploratorio de WITCOM (ResNet-18 sobre Gray-Scott frente a BETO): −0.0103 | F2.2; su prueba se consultó antes de este protocolo |
| F1-Val de la primera pasada v0 (18 configuraciones × 3 semillas), sustituida | F3.2 v0 |
| F1-Val del clasificador sobre 768D: 0.7305 | F2.2 |
| Análisis de potencia | F1.1 |

Ningún resultado de prueba de un clasificador visual del Módulo 3 se conocía.

## 4. Modelos

### 4.1 Clasificadores visuales

Según `docs/matriz_experimentos.md`: un finalista por técnica (F3.3), con hiperparámetros de F3.4, entrenado con las semillas 42, 7, 123, 2024 y 99 (F3.5). Para cada técnica:

$$\overline{F1}_{img,t} = \frac{1}{5}\sum_{s} F1_{t,s}$$

donde cada $F1_{t,s}$ es el F1-Macro de prueba de la semilla *s*, con ŷ = 1 si y solo si `p_dep > 0.5`.

### 4.2 Referencias

| Referencia | Definición | Tarea |
|---|---|---|
| **BETO** | Cabeza de clasificación de Exp 4.1 (checkpoint de la corrida del 11 de marzo, que generó `EMB_41`); semilla 42 | F2.2 |
| **768D** | LogReg sobre `EMB_41` con C elegido por F1-Val (C = 10^−3.5) | F2.2 |
| **Techo 5D** | Mejor modelo sobre **u** por F1-Val entre las familias de §4.3 | F2.1 |
| **Pisos** | §4.4 | F2.3, F3.6 |
| **Descriptores** | Mejor clasificador sobre los cuatro descriptores de bajo nivel, por técnica, por F1-Val | F2.4 |
| **Longitud** | LogReg sobre `log(n_tokens_beto)` y `truncated_128`, estandarizados con `train`, C por F1-Val en la rejilla de LogReg de §4.3 | F4.5 |

### 4.3 Techo 5D

Entrada: los cinco componentes de `umap_params.pkl`. `StandardScaler` ajustado en `train` para LogReg, SVM y MLP. `random_state = 42`. Selección por F1-Macro de validación; en empate (tolerancia 1e-12), la familia anterior en esta tabla y, dentro de ella, la combinación anterior en el orden de la rejilla.

| Familia | Rejilla |
|---|---|
| LogReg (L2, `lbfgs`) | C ∈ 10^[−4, 2], 13 valores; `class_weight` ∈ {None, `balanced`} |
| SVM-RBF | C ∈ 10^[−2, 3], 6 valores; γ ∈ {`scale`} ∪ 10^[−3, 1], 5 valores; `class_weight` ∈ {None, `balanced`}; `probability=True` |
| Random Forest | 500 árboles; `max_depth` ∈ {None, 4, 8}; `min_samples_leaf` ∈ {1, 5}; `class_weight` ∈ {None, `balanced`} |
| *Gradient boosting* (`HistGradientBoostingClassifier`) | `learning_rate` ∈ {0.03, 0.1}; `max_depth` ∈ {2, 3, None}; `max_iter` ∈ {100, 300} |
| MLP | Capas ocultas ∈ {(16,), (32,), (32, 16)}; `alpha` ∈ {1e-4, 1e-2}; `max_iter = 2000` |

### 4.4 Pisos

| Piso | Definición | F1-Macro de prueba |
|---|---|---|
| Mayoritaria | Predice siempre la clase 0 | 0.3804 (determinista) |
| Aleatorio estratificado | Predicción aleatoria con la proporción de `train` | Valor esperado: 0.5 |
| Brillo, por técnica | LogReg sobre la intensidad media de la imagen, con C por F1-Val en la rejilla de LogReg de §4.3 | F2.3 |
| Visual permutado, por técnica | Finalista entrenado con `y_train` permutado (F3.6); media de las 5 semillas | F3.6 |

El piso de la técnica *t* es el de mayor F1-Macro de prueba entre los cuatro. Elegir el máximo hace que el contraste sea conservador.

## 5. Contraste principal

### 5.1 Estadístico

Retención exigida **r = 0.50** y $F1_{azar} = 0.5$:

$$\theta_t = (\overline{F1}_{img,t} - 0.5) - r\,(F1_{BETO} - 0.5)$$

- **H₀:** θ_t ≤ 0 (la imagen conserva menos de la mitad de la capacidad de BETO por encima del azar).
- **H₁:** θ_t > 0.

Con $F1_{BETO} = 0.7039$, θ_t > 0 equivale a $\overline{F1}_{img,t} > 0.6019$, pero el contraste remuestrea también a BETO.

### 5.2 *Bootstrap* pareado

- B = 10,000 réplicas de los 114 índices de prueba con reemplazo, generadas con `numpy.random.default_rng(2026)`.
- En cada réplica se recalculan, sobre los mismos índices: el F1 de BETO, el de cada semilla de cada finalista, el de cada piso y el de cada referencia.
- Valor *p* unilateral: $p_t = \#\{\theta_t^* \le 0\} / B$; si es 0, se reporta *p* < 1/B.

### 5.3 Superioridad sobre el piso

Para cada técnica, $D_t = \overline{F1}_{img,t} - F1_{piso,t}$, con H₁: D_t > 0, sobre las mismas réplicas de §5.2 y con $p = \#\{D_t^* \le 0\}/B$. Si el piso es el aleatorio estratificado, se compara con su valor esperado (0.5) como constante. Si es el visual permutado, se promedian sus cinco semillas en cada réplica.

Este contraste sustituye la prueba de permutación prevista en el plan (§3.3), por coherencia con el contraste principal: ambos usan las mismas réplicas pareadas. El análisis de potencia (F1.1) usó una aproximación por permutación con un piso fijo de 0.529.

### 5.4 Multiplicidad

Dos familias de tres contrastes (una por técnica), cada una con Holm a α = 0.05 unilateral: superioridad sobre el piso (§5.3) y no inferioridad (§5.2). El intervalo de cada técnica se reporta como intervalo percentil bilateral al nivel 1 − 2α_h, donde α_h es el nivel de Holm que le corresponde por su rango en la familia. Si varias técnicas tienen el mismo valor p, todas reciben el menor α_h de los rangos que ocupan. El rechazo de Holm no se ve afectado; solo el nivel de los intervalos.

### 5.5 Veredictos

Por técnica, en este orden:

| Veredicto | Condición |
|---|---|
| **V3. No conservación** | No se rechaza H₀ de superioridad sobre el piso (Holm) |
| **V1. Conservación** | Supera el piso y se rechaza H₀ de no inferioridad (Holm) |
| **V2. Conservación parcial** | Supera el piso y el límite superior del intervalo de θ_t (§5.4) es < 0 |
| **V4. No concluyente** | Supera el piso y no se cumplen V1 ni V2 |

**Lectura de V1.** «La imagen generada con la técnica *t* conserva al menos la mitad de la capacidad discriminativa de BETO por encima del azar». V1 no implica equivalencia con BETO.

**Veredicto global.** G1 (conservación) si al menos una técnica obtiene V1; si no, G2 (conservación parcial) si al menos una obtiene V2; si no, G3 (no conservación) si las tres obtienen V3; en otro caso, G4 (no concluyente). Holm ya controla la multiplicidad entre técnicas.

### 5.6 Estimación de la retención

Como respuesta cuantitativa a la pregunta, para cada técnica:

$$\hat\rho_t = \frac{\overline{F1}_{img,t} - 0.5}{F1_{BETO} - 0.5}$$

con intervalo percentil al 95 % sobre las réplicas de §5.2. Es descriptiva: no interviene en los veredictos ni lleva corrección por multiplicidad.

## 6. Control de longitud (F4.5, confirmatorio)

**Pregunta.** ¿Lo que se conserva es contenido emocional o la longitud del texto?

- **Contraste.** Para cada técnica con V1, V2 o V4: $L_t = \overline{F1}_{img,t} - F1_{longitud}$, con H₁: L_t > 0, sobre las réplicas de §5.2 y con Holm sobre las técnicas evaluadas, α = 0.05.
- **Calificación.** Si no se rechaza H₀, el veredicto de esa técnica se reporta con la calificación «no distinguible de un clasificador de longitud».
- **Descriptivo.** F1-Macro del finalista por terciles de `n_tokens_beto`, con los puntos de corte calculados en `train`.

## 7. Contrastes secundarios

Todos son descriptivos o complementarios y ninguno modifica los veredictos.

| Contraste | Método |
|---|---|
| **Descomposición de la pérdida** | Tres etapas, con intervalo percentil al 95 % sobre las réplicas de §5.2: cabeza (F1_BETO − F1_768D), reducción (F1_768D − F1_5D) y renderizado (F1_5D − $\overline{F1}_{img,t}$). Su suma es F1_BETO − $\overline{F1}_{img,t}$ |
| **Entre técnicas** | Q de Cochran sobre el acierto por muestra de los tres finalistas; si p < 0.05, McNemar exacto por pares con Holm (3 pares) |
| **Frente a BETO y al techo 5D** | McNemar exacto por técnica, con Holm dentro de cada familia de tres. Es complementario: su no significancia no se interpreta como equivalencia |
| **Frente a descriptores de bajo nivel** | $\overline{F1}_{img,t} - F1_{descriptores,t}$ con intervalo al 95 % |
| **Concordancia** | Kappa de Cohen entre BETO, techo 5D y visual; intersección de errores (F4.6) |

**Predicción por muestra con cinco semillas.** Para los contrastes que requieren una predicción por muestra (McNemar, Cochran, kappa e intersección de errores), se usa la media de `p_dep` sobre las cinco semillas, con ŷ = 1 si la media es > 0.5. El contraste principal (§5) usa el F1 de cada semilla y su media.

## 8. Análisis de sensibilidad (F5.5)

Se reporta el veredicto de cada técnica bajo cada variante, sin modificar el veredicto principal.

| Variante | Detalle |
|---|---|
| Retención exigida | r ∈ {0.75, 0.80}. Según F1.1, con estos valores el veredicto más probable es V4 aunque la conservación sea total |
| Formulación A | Margen fijo δ = (1 − r)(0.7039 − 0.5), con $\Delta_t = F1_{BETO} - \overline{F1}_{img,t}$ |
| Por semilla | Veredicto con cada semilla por separado; se reporta cuántas de las cinco dan V1 |
| Variabilidad de BETO | F1_BETO desplazado ±0.0128, la diferencia de F1-Val entre las dos corridas nominalmente idénticas de Exp 4.1 (F2.2, H1) |
| Segundo mejor modelo | Segunda configuración de la clasificación de F3.3 en cada técnica, con hiperparámetros de la primera pasada y cinco semillas (requiere 15 entrenamientos adicionales, §12). Se excluye al finalista y se aplica de nuevo la regla completa de §5 de la matriz (líderes, tolerancia, complejidad) a las cinco configuraciones restantes de la técnica. |
| Piso sin el visual permutado | Piso = máximo de mayoritaria, estratificado y brillo |

## 9. Análisis explicativos (Fase 4)

F4.1, F4.2, F4.3, F4.4, F4.6, F4.7 y F4.8 siguen los métodos del plan v2 (§4, Fase 4) y se reportan como explicativos: no modifican los veredictos ni se usan para seleccionar modelos. Las pruebas de hipótesis que incluyan se corrigen con Holm dentro de cada análisis.

## 10. Acceso al conjunto de prueba

| Quién consulta prueba | Cuándo | Qué se calcula |
|---|---|---|
| BETO | Ya consultado (F2.2) | — |
| Finalistas (F3.5) y visual permutado (F3.6) | Con la etiqueta `protocolo-v1` en el historial | Una inferencia por modelo y semilla; predicciones archivadas |
| Techo 5D, 768D, pisos, descriptores y longitud | Predicciones generadas en sus tareas | Métricas de prueba solo en la Fase 5 |
| Segundo mejor modelo (§8) | Después de F3.5 | Una inferencia por semilla |

La compuerta la hacen cumplir `src/predictions/validate.py` (regla 7) y `run_grid.py --evaluate-test` (matriz, §11.5). Ningún modelo, hiperparámetro ni decisión se modifica después de consultar su prueba.

## 11. Fallas y desviaciones

- **Falla de una corrida** (error, divergencia o NaN): se repite una vez con la misma semilla y configuración. Si vuelve a fallar, la técnica se reporta con las semillas restantes y la falla se documenta. No se sustituye la semilla ni el finalista.
- **Enmiendas.** Todo cambio posterior al congelamiento se registra al final de este documento como «Enmienda *n* (fecha)», con su motivo e indicando si ocurre antes o después de consultar alguna prueba de los finalistas. Las enmiendas posteriores a esa consulta solo pueden añadir análisis exploratorios.

## 12. Dependencias del protocolo

| Dependencia | Responsable |
|---|---|
| Fase `sensibilidad` para el segundo mejor modelo (15 entrenamientos) en `docs/matriz_experimentos.md` §4 | SV + CE (enmienda de la matriz antes de congelar) |
| Compuerta de `run_grid.py` con la etiqueta `protocolo-v1` | CE (matriz, §11.5) |
| Plantillas de discusión por veredicto (F6.1), antes de la Fase 5 | SV + RP |

## 13. Cifras de diseño registradas

| Cifra | Valor |
|---|---|
| r (principal) | 0.50 |
| F1_azar | 0.5 |
| α | 0.05, unilateral, Holm por familia |
| B | 10,000 |
| Semilla del *bootstrap* | 2026 |
| Umbral equivalente de F1 visual para θ = 0 con F1_BETO observado | 0.6019 |
| δ de la formulación A con r = 0.50 / 0.75 / 0.80 | 0.1019 / 0.0510 / 0.0408 |

## Enmienda 1 (2026-11-02)

**Momento.** Es posterior al congelamiento (etiqueta `protocolo-v1`) y **anterior a cualquier consulta de prueba de los finalistas**: aún no se ha ejecutado F3.5.

**Alcance.** Es aclaratoria. No cambia ningún modelo, parámetro, estadístico, regla de decisión ni archivo de configuración: `configs/protocolo.json` y `configs/matriz_experimentos.json` conservan las huellas del encabezado.

**Motivo.** Una revisión independiente del protocolo contra la evidencia, hecha al cerrar F1.2, encontró tres imprecisiones en el texto congelado.

1. **Origen de las cifras (encabezado).** La frase «Toda cifra citada aquí proviene de `results/canonical_numbers.json`» se precisa así:
   - Las cifras observadas y las cifras de diseño en escala de F1-Macro provienen de `results/canonical_numbers.json`.
   - Los parámetros provienen de `configs/protocolo.json` y `configs/matriz_experimentos.json`, cuyas huellas registra el encabezado. Son parámetros r, F1_azar, α, B, la semilla del *bootstrap*, las rejillas y tolerancias, las semillas de entrenamiento, el número de entrenamientos y los niveles de los intervalos.
   - ±0.0128 (§8) es `f12.variabilidad_beto` calculada sin redondear. Restar las cifras redondeadas de F2.2 daría 0.0129.
2. **Plazo de las plantillas de discusión (§12).** Las plantillas por veredicto (F6.1) se versionan **antes de ejecutar F3.5**, no solo antes de la Fase 5. El motivo es que `run_grid.py` registra el F1 de prueba al evaluar los finalistas en F3.5.
3. **Desviaciones respecto del plan v2.** Se declaran tres:
   - **Estadístico y margen.** El plan (§3.1–§3.2) proponía Δ_t con un margen fijo δ = 0.058 («El margen definitivo lo fija SV en la Fase 1»). El protocolo usa θ_t con retención r = 0.50 (formulación B, §5.1). La cadena de motivos es:
     - el F1-Test verificado de BETO es 0.7039, no el reportado (F2.2, `docs/especificacion_f22.md` §6.5);
     - con él, el δ del plan pasa a 0.0035 y pierde su lectura (`docs/especificacion_f22.md` §7);
     - recomendación de F1.1 (`docs/especificacion_f11.md` §8).
   - **Conjunto de sensibilidad.** El plan (F5.5) preveía δ ∈ {0.03, 0.058, 0.08}. El protocolo lo sustituye por r ∈ {0.75, 0.80} y la formulación A (§8).
   - **Superioridad sobre el piso.** La prueba de permutación del plan (§3.3) se sustituye por el *bootstrap* pareado. Ya estaba declarado en §5.3.

**Nota de equivalencia.** «Hiperparámetros de la primera pasada» (§8; `segundo_mejor.hyperparams = "primera_pasada"` en `configs/protocolo.json`) designa la configuración común de la matriz (`fases.sensibilidad.hyperparams = "configuracion_comun"`, `docs/matriz_experimentos.md` §3). Es la misma con la que se ejecuta la primera pasada.
