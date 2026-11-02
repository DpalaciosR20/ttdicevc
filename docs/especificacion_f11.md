# Análisis de potencia del contraste de no inferioridad (F1.1)

> Tarea F1.1 del plan v2 · Rol: SV · Estado: **diseño fijado; resultados pendientes**
> Las secciones 1 a 6 se versionan con commit propio antes de ejecutar. Las secciones 7 y 8 se completan después.
> Insumos: `docs/datos.md`, `docs/especificacion_predicciones.md`, `results/canonical_numbers.json`.

## 1. Propósito

Estimar, antes de cualquier resultado confirmatorio, la capacidad del protocolo para decidir el Objetivo V con n = 114 y cinco semillas por finalista: el ancho esperado del intervalo del estadístico principal, la menor retención verdadera que el protocolo puede declarar con potencia de 0.80 y la probabilidad de cada veredicto (V1–V4) en escenarios plausibles. Sus resultados informan la elección de la retención exigida *r* en F1.2; no constituyen un contraste.

## 2. Estadístico y margen por retención del efecto

El margen se define por retención de la capacidad discriminativa por encima del azar, con $F1_{azar} = 0.5$, el F1-Macro esperado de un clasificador aleatorio estratificado, valor teórico independiente de los datos del proyecto.

Se evalúan dos formulaciones equivalentes en su punto, distintas en su tratamiento de la incertidumbre de BETO:

| Formulación | Estadístico | Hipótesis alternativa | Tratamiento de BETO |
|---|---|---|---|
| **A. Margen fijo** | $\Delta_t = F1_{BETO} - \overline{F1}_{img,t}$, con $\delta = (1-r)(F1^{obs}_{BETO} - 0.5)$ | $\Delta_t < \delta$ | $F1_{BETO}$ fijo en su valor observado al calcular δ |
| **B. Retención lineal** | $\theta_t = (\overline{F1}_{img,t} - 0.5) - r\,(F1_{BETO} - 0.5)$ | $\theta_t > 0$ | $F1_{BETO}$ se remuestrea junto con el visual |

En ambas, $\overline{F1}_{img,t}$ es la media del F1-Macro de prueba de las cinco semillas del finalista, cada uno calculado sobre las mismas muestras remuestreadas.

**Valores de r evaluados:** 0.50, 0.75 y 0.80. Con $F1^{obs}_{BETO} = 0.7039$ (`ref.beto_exp41.f1_macro`), δ = 0.1019, 0.0510 y 0.0408.

## 3. Regla de decisión simulada

- *Bootstrap* pareado: B réplicas de índices de prueba con reemplazo, comunes a BETO y a las cinco semillas.
- Valor *p* unilateral por *bootstrap*: proporción de réplicas con $\Delta^* \ge \delta$ (A) o con $\theta^* \le 0$ (B).
- Corrección de Holm sobre las tres técnicas, α = 0.05.
- Superioridad sobre el piso: prueba de permutación con 10,000 permutaciones frente a un piso de 0.529 (brillo promedio de WITCOM, el mayor de los pisos conocidos), con Holm.
- Veredictos (con el nivel de confianza ajustado por Holm para cada técnica): V3 si no supera el piso; V1 si supera el piso y se rechaza H₀; V2 si supera el piso y el intervalo indica pérdida mayor al margen (A: límite inferior de Δ > δ; B: límite superior de θ < 0); V4 en otro caso.

## 4. Insumos y su papel

| Insumo | Fuente | Papel |
|---|---|---|
| `y_test` y `y_val` (70/44) | `meta/sample_index.csv` | Estructura de clases |
| Predicciones de BETO, val y test | `results/predictions/referencia/text__beto__finetune__seed42.csv` | Término de BETO; patrón de aciertos |
| Predicciones de la primera pasada (F3.2, versión sustituida), solo validación | `results/f32_primera_pasada/predictions.csv` | Calibración empírica del acuerdo BETO–visual y de la variabilidad entre semillas |
| Predicciones del clasificador sobre 768D, val | `results/predictions/referencia/emb768__logreg__none__seed42.csv` | Calibración del acuerdo entre BETO y un clasificador que no ve el texto |

Las predicciones de F3.2 son de validación, usan un preprocesamiento sustituido en Perlin y Lorenz (F1.4) y tienen `p_dep` redondeado a seis decimales. Se usan **solo para planificar**: no seleccionan modelos ni intervienen en ninguna decisión confirmatoria. No se dispone de predicciones por muestra de la CNN de WITCOM.

No se calcula ninguna métrica de prueba de modelos visuales. Las métricas de prueba de BETO ya son conocidas (F2.2).

## 5. Bloques de análisis

| Bloque | Contenido |
|---|---|
| **E. Empírico** | Para cada configuración de F3.2 (18) y para el clasificador sobre 768D, sobre validación: ancho del IC bilateral de Δ y de θ al nivel de Holm, con BETO-val como referencia. Distribución de la tasa de discordancia BETO–visual y de la DE entre semillas |
| **S. Simulación** | Generador de predicciones visuales para cinco semillas con: retención verdadera $\rho \in \{0.25, 0.50, 0.75, 0.90, 1.00, 1.10\}$, acuerdo con BETO calibrado con E (percentiles 10, 50 y 90 de la discordancia) y DE entre semillas ∈ {0.005, 0.010, 0.015}. Para cada escenario y cada r: P(V1), P(V2), P(V3), P(V4), con 1,000 simulaciones |
| **D. Retención mínima detectable** | Para cada r y formulación, la menor $\rho$ con P(V1) ≥ 0.80, por interpolación sobre la malla de S |
| **P. Piso** | Potencia de la superioridad sobre el piso en los mismos escenarios |

El generador de S se documenta en el script: para cada muestra, el acierto visual se obtiene como el acierto de BETO con probabilidad *a* (acuerdo) o como un acierto independiente con probabilidad por clase ajustada para alcanzar la retención objetivo; las semillas comparten el componente de acuerdo y difieren en el independiente, calibrado para la DE objetivo. La calibración se verifica reportando la retención y la DE realizadas en cada escenario.

## 6. Parámetros de ejecución

| Parámetro | Valor |
|---|---|
| B (en E y en cada simulación de S) | 2,000 en S; 10,000 en E |
| Simulaciones por escenario | 1,000 |
| Semilla maestra | 2026; semillas derivadas con `numpy.random.SeedSequence` |
| Cálculo del F1-Macro | Vectorizado a partir de conteos de la matriz de confusión (no `sklearn` dentro del bucle), verificado contra `f1_score(..., average='macro')` en 100 réplicas |
| Ejecución | Local si la estimación es ≤ 20 min; en otro caso, Kaggle (`docs/ejecucion_kaggle.md`) |

B = 2,000 en S introduce un error de Monte Carlo en los límites del intervalo que se reporta junto con las potencias.

## 7. Resultados

[completar: tabla E; malla de S por r y formulación; retención mínima detectable; potencia frente al piso; figura de curvas de potencia]

## 8. Recomendación para F1.2

[completar: formulación (A o B) y r recomendados, con su justificación; declaración de que el Δ exploratorio de WITCOM (−0.0103) era conocido al fijarlos]
