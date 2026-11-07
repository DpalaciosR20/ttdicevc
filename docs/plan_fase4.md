# Plan de la Fase 4 — Análisis de preservación

> Rol: SV · Estado: **congelado (2026-11-06), junto con la Enmienda 4** (`docs/protocolo_preregistrado.md`), en el mismo commit · Rama `sv/f5-contraste`.
> Cubre F4.1, F4.2, F4.3, F4.4, F4.6, F4.7 y F4.8. F4.5 (control de longitud) está fijado en el protocolo §6 y en la Enmienda 4 (A7–A9), y no se repite aquí.
>
> **Fuentes:**
> - `plan/plan_tt2_v2.md` (Fase 4, tabla y párrafo final; §5, amenazas);
> - `docs/protocolo_preregistrado.md` §7, §9 y §10, y las Enmiendas 3 y 4;
> - bitácoras F0.3, F2.1, F2.4, F3.5, F3.6 y F3.7; `docs/matriz_experimentos.md` §8.
>
> Lo que esas fuentes no fijaban se resuelve aquí como **Decisión SV (2026-11-06)**. Todo cambio posterior se registra como enmienda de este plan, con fecha, indicando si es anterior o posterior al uso de prueba.

## 0. Reglas comunes

- **Carácter.** Según el protocolo §9, «F4.1, F4.2, F4.3, F4.4, F4.6, F4.7 y F4.8 siguen los métodos del plan v2 (§4, Fase 4) y se reportan como explicativos». No modifican los veredictos ni se usan para seleccionar modelos. «Las pruebas de hipótesis que incluyan se corrigen con Holm dentro de cada análisis».
- **Modelos.** Se usan los finalistas fijados en F3.3:
  - Gray-Scott: EfficientNet-B0 *fine-tuning*;
  - Perlin: ViT-B/16 congelado;
  - Lorenz: ViT-B/16 congelado.

  Cada uno con sus 5 semillas de F3.5 y, donde aplique, sus controles de F3.6. Los modelos (`.pt`) están en los *datasets* de Kaggle `tt2-artefactos` y `tt2-artefactos-f36`, y en la copia de Drive `MiUnidad/tt2-artefactos/`. Los mapas están en `tt2-mapas`.
- **Una predicción por muestra con cinco semillas.** Se usa la media de `p_dep`, con ŷ = 1 si la media es > 0.5 (§7), cuando el análisis requiera una sola predicción por muestra.
- **Réplicas *bootstrap*.** Las de §5.2 con la Enmienda 4, A1: `default_rng(2026)`, `rng.integers(0, 114, size=(10000, 114))`, filas de prueba ordenadas por `sample_id`.
- **G1. Uso de prueba** (Decisión SV, 2026-11-06).
  - `train` y `val` se usan sin restricción.
  - Las predicciones y los mapas de prueba ya archivados se usan solo **después del commit del veredicto de la Fase 5**.
  - Los modelos exploratorios nuevos de la Fase 4 (regresor de F4.1, LogReg sobre **û**) consultan prueba **una sola vez**, después de ese commit.
- **G2. Sin inferencia nueva sobre prueba** (Decisión SV, 2026-11-06). Ninguna inferencia nueva de los finalistas (F3.5) ni de los controles (F3.6) sobre imágenes de prueba, transformadas o no. La ablación y la cascada de cordura se hacen en `val`.
- **G3. Familia de Holm** (Decisión SV, 2026-11-06). Es el conjunto de pruebas de una misma pregunta dentro de un análisis, normalmente las tres técnicas. Cada análisis declara su familia abajo.
- **Ejecución.** Todo lo que requiera GPU o más de 20 min en local se ejecuta en Kaggle según `docs/ejecucion_kaggle.md` (`CLAUDE.md`).

---

## F4.1 — Recuperabilidad de parámetros (B)

**Pregunta** (plan): «¿Qué componentes de **u** preserva cada técnica?».

**Insumos.**
- Imágenes class-blind de las tres técnicas.
- **u** de `umap_params.pkl`, leído con `src/training/umap_io.py`.
- *Backbone* de cada finalista.
- `meta/sample_index.csv`.
- `results/predictions/referencia/umap5d__logreg__none__seed42.csv` (F2.1).

**Método** (plan, con Decisiones SV, 2026-11-06).
1. **Sonda lineal.**
   - Características: penúltima capa del *backbone* del finalista, congelado.
   - ViT (Perlin, Lorenz): las características son idénticas entre semillas, porque el *backbone* es el de ImageNet.
   - Gray-Scott: la semilla 42 es la principal; las 5 semillas y el *backbone* ImageNet sin ajustar son sensibilidad.
   - Regresor: Ridge multisalida, con predicciones recortadas a [0, 1].
2. **Hiperparámetros.** alpha ∈ `logspace(-3, 3, 13)`, elegido por el R² medio en `val`. Sin épocas ni parada.
3. **Splits.**
   - Ajuste en `train` y selección en `val`.
   - R² en `val` (principal) y en prueba (una vez, después de la Fase 5; G1).
   - **û** de `train` por *cross-fitting* de 5 pliegues (`KFold(shuffle=True, random_state=42)`); **û** de `val` y prueba con el modelo ajustado en todo `train`.
4. **LogReg.** Rejilla, escalado y regla de §4.3, sobre **û** y, de la misma forma, sobre **u**. La de **u** se compara con `umap5d__logreg` de F2.1.
5. **Comparación.** Sin umbral de «reproduce». ΔF1 = F1(**u**) − F1(**û**), con IC percentil bilateral al 95 % sobre las réplicas de §5.2 (A1), en prueba y después de la Fase 5.
6. **R².** `sklearn.metrics.r2_score(multioutput='raw_values')` sobre la escala [0, 1].
7. **Cómputo.** La extracción de características se hace en Kaggle; Ridge y LogReg, en local.

**Métricas.** R² por componente y técnica (val; prueba después de la Fase 5); F1-Macro de la LogReg sobre **û** y sobre **u**; ΔF1 con su IC.

**Qué se afirma con cada resultado.**

| Resultado | Se afirma | No se afirma |
|---|---|---|
| R² alto en un componente | La técnica codifica ese componente de forma **linealmente** recuperable desde la penúltima capa del *backbone* | Que el clasificador use ese componente para discriminar |
| R² bajo en un componente | Una sonda lineal sobre ese *backbone* no lo recupera | Que la información no esté en la imagen: una sonda no lineal u otro *backbone* podrían recuperarla |
| El IC de ΔF1 contiene 0 | «Compatible con ausencia de pérdida» de la dirección discriminativa al pasar por imagen y sonda | «Equivalente»; tampoco que el mecanismo quede demostrado para el clasificador entrenado, que es un modelo distinto |
| El IC de ΔF1 queda por encima de 0 | Se pierde información discriminativa en imagen + sonda | En cuál de las dos etapas se pierde |

**Familia de Holm.** Sin pruebas de hipótesis; solo IC.

---

## F4.2 — Régimen dinámico (C)

**Pregunta** (plan): «¿La separación de clases se debe a cruzar fronteras de régimen en vez de una codificación continua?».

**Insumos.**
- **u** de `train` y el mapeo determinista de cada generador:
  - Gray-Scott (`gray_scott_lib.umap_to_rd_params`): f = 0.022 + 0.024·u0, k = 0.059 + 0.007·u1;
  - Lorenz (`lorenz_lib.map_umap_to_lorenz`): ρ = 24 + 8·u1;
  - Perlin (`perlin_lib.map_umap_to_perlin`): octavas = int(lerp(u4, 1, 8)).
- Etiquetas.
- Imágenes almacenadas de Gray-Scott.
- Predicciones de prueba archivadas de Lorenz (después de la Fase 5).

**Método** (plan, con Decisiones SV, 2026-11-06).
1. **Gray-Scott.** Descriptivo, sin discretizar regiones de Pearson: dispersión de (f, k) por clase, con el diagrama de Pearson (1993) superpuesto solo como referencia visual aproximada. Sin prueba de régimen.
2. **Lorenz.**
   - Proporción por clase con ρ < 24.74 (u1 < 0.0925), con prueba exacta de Fisher.
   - Distribución de ρ por clase, con Mann–Whitney bilateral.
3. **Splits.** Solo `train` para las distribuciones (sin modelos).
4. **Perlin.** Octavas por clase con Mann–Whitney bilateral.
5. **Reintentos de Gray-Scott.**
   - Se regenera `train` (908) por la ruta de CPU en Kaggle (CPU), con `run_gray_scott_safe` instrumentado para registrar los reintentos de cada muestra.
   - Se comprueba que cada imagen regenerada es idéntica bit a bit a la almacenada, y se reporta.
   - Tasa de reintentos por clase, con prueba de Fisher en la familia de F4.2.
   - La ruta de GPU no se usa (decisión 6 de TT-II; F0.3, H2: las imágenes almacenadas corresponden a la ruta de CPU).
6. **Lorenz por estrato.** F1 del finalista por estrato ρ < 24.74 / ρ ≥ 24.74, con las predicciones de prueba archivadas, después de la Fase 5 (G1). Descriptivo.

**Métricas.**
- Dispersión de (f, k) por clase.
- Proporciones y distribuciones por clase (Lorenz, Perlin), con valores p de Fisher y de Mann–Whitney.
- Tasa de reintentos por clase.
- Número de imágenes regeneradas idénticas bit a bit.
- F1 por estrato (Lorenz).

**Familia de Holm.** Todas las pruebas de F4.2: Fisher y Mann–Whitney de Lorenz, Mann–Whitney de Perlin y Fisher de reintentos de Gray-Scott (m = 4).

**Qué se afirma con cada resultado.**

| Resultado | Se afirma | No se afirma |
|---|---|---|
| Lorenz o Perlin: diferencia significativa por clase (Holm) en la proporción de régimen o en la distribución | Las clases ocupan regiones de parámetros distintas, y en Lorenz eso puede incluir un cambio cualitativo de dinámica inducido por **u** | Que el clasificador use el régimen; que el régimen tenga contenido emocional |
| Lorenz: F1 muy distinto entre estratos | El desempeño del finalista depende del régimen de la muestra (descriptivo) | Una relación causal; el estrato ρ < 24.74 puede ser pequeño |
| Sin diferencias significativas | No hay evidencia de que la separación se deba a cruzar fronteras de régimen | Que la codificación sea continua en sentido estricto |
| Gray-Scott: las clases se concentran en zonas distintas del plano (f, k) | Descripción visual de cómo se reparten las clases respecto del diagrama de Pearson | Ninguna afirmación de régimen: no hay discretización ni prueba |
| La tasa de reintentos difiere entre clases (Holm) | El *wrapper* introduce una diferencia por clase (vía **u**) que el modelo podría aprovechar | Una fuga de etiqueta: el *wrapper* no recibe la etiqueta (F0.3) |
| Alguna imagen regenerada no coincide bit a bit | Se reporta como hallazgo de reproducibilidad, y la tasa de reintentos de esas muestras queda en duda | — |

---

## F4.3 — Atribución espacial y espectral (E)

**Pregunta** (plan): «¿Sobre qué estructuras discrimina el modelo?».

**Insumos.**
- Mapas de F3.7 (`tt2-mapas`; `results/f37_mapas/`):
  - `map` normalizado por muestra;
  - `map_raw` y `map_raw_scale`;
  - `backbone_map` en los ViT.
- Modelos de F3.5 y F3.6.
- Imágenes de `val`.
- Resumen de F3.7 (`resumen/per_run.csv`, `class_means.npz`).
- Correlaciones de `p_dep` entre real y permutado de F3.6 (bitácora F3.6, §4).

**Método** (plan, Enmienda 3, matriz §8 y Decisiones SV, 2026-11-06).

*Decisiones de método.*
1. **Coseno de las cabezas en los ViT congelados** (F3.6 H2, F3.7 H2). Coseno entre los vectores de pesos de la cabeza real y de la permutada de la misma semilla, para w1 y para w1 − w0. Sirve para distinguir entre «el método no es sensible a la cabeza» y «la cabeza permutada aprendió una dirección parecida».
2. **Aleatorización de la cabeza en los modelos reales.** Es el paso decisivo de la cascada en un ViT congelado (Enmienda 3: el mapa principal debe cambiar y la atención del *backbone* no).
3. **Gray-Scott.** Por semilla, la correlación entre los mapas real y permutado frente a la correlación de `p_dep` real y permutado (F3.6, columna «Corr. con finalista»).
4. **Grad-CAM constantes** (F3.7 H5).
   - La agregación principal se hace sobre `map_raw · map_raw_scale`, sin normalizar.
   - Se reporta la fracción de mapas constantes por clase real y por clase predicha.
   - Como sensibilidad, el promedio de los mapas normalizados sin los constantes.

*Parámetros.*
1. **Similitud.** Spearman por muestra a 56 × 56 (principal) y SSIM (secundaria).
   - Criterio relativo: el mapa «no cambia» si la similitud real–aleatorizado (mediana por muestra) cae dentro del rango de similitudes entre semillas reales del mismo finalista (10 pares).
   - «Cambia» si queda por debajo del mínimo de ese rango.
2. **Cascada por bloques**, desde la salida: cabeza → bloques en orden inverso → *embeddings*.
   - EfficientNet-B0: `features[8]` … `features[0]`.
   - ViT-B/16: `encoder.layers` 11 … 0.
   - Reinicialización con `reset_parameters()` de PyTorch y `torch.manual_seed(2026)`.
   - Las 5 semillas de cada finalista, en `val` (G2).
3. **Coseno de cabezas.** La referencia son los cosenos entre cabezas reales de semillas distintas (10 pares). Es «alto» si el coseno real–permutado está dentro del rango de la referencia, y «bajo» si está por debajo de su mínimo.
4. **Agregación espacial.**
   - Lorenz: sin marco canónico, porque la elevación varía con u4. Se agrega en coordenadas de imagen, estratificado por terciles de u4 (cortes en `train`).
   - Gray-Scott: perfil radial desde el centro de la imagen, 28 anillos de 4 px hasta 224 px.
5. **Clase y split.** Agregación por clase real (principal) y predicha (secundaria). `val` es el principal; prueba es réplica, con mapas archivados, después de la Fase 5 (G1).
6. **Ablación espectral, en `val`** (G2), en Kaggle con GPU.
   - Filtro gaussiano paso-bajo con σ ∈ {1, 2, 4, 8} px sobre la imagen de 256 px, antes del preprocesamiento de F1.4.
   - Paso-alto complementario: imagen − paso-bajo + media.
   - ΔF1 = media sobre las 5 semillas del F1 filtrado − F1 sin filtrar.
7. **Prueba.** Una por técnica: permutación de etiquetas (5,000, `default_rng(2026)`) sobre la norma L2 de la diferencia de mapas medios por clase (`map_raw · map_raw_scale`, en `val`).
8. **Mapas constantes.** Si una clase tiene menos de 10 mapas no constantes, la sensibilidad sin constantes no se reporta para esa clase; se reporta el conteo.

**Lectura de los ViT.**
- Si falla la cordura por aleatorización de datos y el **coseno es bajo**, los mapas se reportan como atención del *backbone*, no como evidencia de qué aprendió el clasificador.
- Si falla la cordura por datos y el **coseno es alto**, la prueba por datos se declara no informativa: el control permutado no es un modelo nulo (F3.6, H2). La lectura depende entonces de la aleatorización de la cabeza (parámetro 2):
  - si el mapa cambia, se reporta como «dependiente de la cabeza, sin validación por datos»;
  - si no cambia, como atención del *backbone*.

**Métricas.**
- Mapas medios por clase y su diferencia.
- Perfiles radiales (Gray-Scott) y mapas por tercil de u4 (Lorenz).
- Similitud (Spearman, SSIM) en cada paso de la cascada y en real frente a permutado.
- Cosenos de cabezas y su referencia.
- Fracción de mapas constantes.
- ΔF1 por σ y tipo de filtro.
- Valor p de permutación por técnica.

**Familia de Holm.** Las tres pruebas de permutación de la diferencia de mapas medios (una por técnica, m = 3).

**Qué se afirma con cada resultado.**

| Resultado | Se afirma | No se afirma |
|---|---|---|
| La cascada cambia el mapa, la cordura por datos lo cambia y la diferencia por clase es significativa (Holm) | El mapa depende del modelo entrenado y de las etiquetas; la diferencia describe dónde se apoya el modelo | Causalidad: un mapa no prueba que la región sea necesaria (para eso está la ablación) |
| ViT: falla la cordura por datos y el coseno es bajo | El método es poco sensible a la cabeza; el mapa se reporta como atención del *backbone* | Nada sobre qué aprendió el clasificador a partir de esos mapas |
| ViT: falla la cordura por datos, el coseno es alto y el mapa cambia al aleatorizar la cabeza | El mapa depende de la cabeza; se reporta «dependiente de la cabeza, sin validación por datos» | Que esté validado como explicación del modelo real |
| ViT: falla la cordura por datos, el coseno es alto y el mapa no cambia al aleatorizar la cabeza | El mapa se reporta como atención del *backbone* | Nada sobre qué aprendió el clasificador |
| Gray-Scott: la correlación de mapas sigue a la de `p_dep` entre semillas | La semejanza entre mapas real y permutado se explica por la semejanza de las funciones aprendidas | Que los mapas sean independientes de la etiqueta |
| ΔF1 negativo grande en una banda (val) | El modelo depende de la información de esa banda | Que esa banda tenga contenido emocional; nada sobre prueba (G2) |

---

## F4.4 — Convergencia con la Tabla 11

**Pregunta** (plan): «¿Las propiedades usadas por el modelo coinciden con los descriptores significativos?».

**Insumos.**
- `results/f24_descriptores/tabla11_recalculada.csv`, con las 8 comparaciones significativas tras Holm:
  - Gray-Scott: entropía, contraste, densidad de bordes y energía LBP;
  - Perlin: contraste y densidad de bordes;
  - Lorenz: entropía y energía LBP.
- Definiciones y parámetros de los descriptores (`configs/protocolo.json` → `descriptores`).
- Mapas de `val` (F3.7) y resultados de F4.3.
- Contraste visual − descriptores de §7 (Fase 5; Enmienda 4, A11).

**Método** (plan, con Decisiones SV, 2026-11-06).
1. **Por cada descriptor significativo.**
   - El descriptor se calcula por parche: rejilla de 7 × 7 en EfficientNet y de 14 × 14 en ViT.
   - Spearman por muestra entre el descriptor por parche y la masa del mapa por parche.
   - Media sobre las muestras de `val`, con IC percentil al 95 % por *bootstrap* sobre muestras (B = 10,000, `default_rng(2026)`).
   - La relación entre las bandas de la ablación (F4.3) y los descriptores se describe cualitativamente.
2. **Peso en la redacción.** El |d| de Cohen; la significancia solo define el conjunto de las 8 comparaciones. Sin pruebas nuevas sobre la Tabla 11.
3. **Contexto.** El contraste visual − descriptores (A11) se cita como contexto, sin combinarlo con lo anterior.

**Métricas.** Spearman medio por descriptor y técnica con su IC; |d| de Cohen de la Tabla 11.

**Familia de Holm.** Sin pruebas de hipótesis; solo IC.

**Qué se afirma con cada resultado.**

| Resultado | Se afirma | No se afirma |
|---|---|---|
| IC del Spearman medio por encima de 0 en un descriptor significativo | En `val`, el mapa pone más masa donde el descriptor es más alto: convergencia entre atribución y descriptor | Que el modelo calcule ese descriptor; nada si F4.3 declaró el mapa no interpretable (atención del *backbone*) |
| IC que contiene 0 o queda por debajo | Sin convergencia local entre mapa y descriptor | Que el descriptor sea irrelevante para la clase: es significativo a nivel global (Tabla 11) |

---

## F4.6 — Análisis de errores

**Pregunta** (plan): «¿El clasificador visual hereda los errores de **u**?».

**Insumos.** Predicciones de prueba de BETO (F2.2), del 768D (F2.2), del techo 5D (SVM-RBF, F2.1) y de los finalistas (F3.5). Todas existen. Se usan después del commit del veredicto de la Fase 5 (G1).

**Método** (protocolo §7, Enmienda 4 A10, y Decisiones SV, 2026-11-06).
- **Kappa de Cohen** entre BETO, 768D, techo 5D y visual:
  - sobre ŷ (principal) y sobre vectores de acierto (secundario);
  - IC percentil bilateral al 95 % sobre las réplicas de §5.2, cuantiles `linear`;
  - descriptivo.
- Predicción visual = media de `p_dep` de las 5 semillas > 0.5.
- **Contención.** C = |E_vis ∩ E_5D| / |E_vis|. La referencia nula es |E_5D| / n (independencia). Se reporta también la contención frente a BETO.
- **IC.** Percentil bilateral al 95 % sobre las réplicas de §5.2.
- **Por semilla**, como sensibilidad descriptiva.
- **Diagrama de intersección** de los errores de BETO, 768D, techo 5D y visual.

**Métricas.** Kappa por par y técnica con su IC; C frente al techo y frente a BETO, con su IC y su referencia nula; tamaños de las intersecciones.

**Familia de Holm.** Sin pruebas de hipótesis; solo IC.

**Qué se afirma con cada resultado.**

| Resultado | Se afirma | No se afirma |
|---|---|---|
| El límite inferior del IC de C (frente al techo) supera la referencia nula **y** el IC de ΔF1 de F4.1 contiene 0 | «Consistente con que el visual hereda los errores de **u**» | «Demostrado»; tampoco equivalencia entre visual y techo |
| El límite inferior del IC de C supera la referencia nula, pero el IC de ΔF1 de F4.1 no contiene 0 | Los errores visuales se concentran en los de **u** más de lo esperado por independencia | La frase «consistente con que el visual hereda los errores de **u**» |
| El IC de C contiene la referencia nula o queda por debajo | No hay evidencia de que los errores visuales se concentren en los de **u** | Su causa (renderizado, clasificador o semillas) |
| Kappa (BETO, visual) bajo y kappa (techo, visual) alto | Lo que se pierde respecto de BETO ocurre antes de la imagen (cabeza o reducción), coherente con F5.3 | Equivalencia entre visual y techo |

---

## F4.7 — Correspondencia geométrica (D)

Se ejecuta (Decisión SV, 2026-11-06), porque casi no tiene costo al reutilizar F4.1.

**Pregunta** (plan): «¿El generador preserva las vecindades del *embedding*?».

**Insumos.** **u** de `train` y las características de la penúltima capa extraídas en F4.1.

**Método** (plan, con Decisiones SV, 2026-11-06).
1. **Características.** Las de la penúltima capa extraídas en F4.1 (Gray-Scott: semilla 42 principal, como en F4.1).
2. **Distancias.** Euclídea en **u** (escala [0, 1]); coseno en las características.
3. **Muestras.** `train` (n = 908). Correlación de Spearman entre las matrices de distancia; prueba de Mantel con 9,999 permutaciones, `default_rng(2026)`.

**Métricas.** ρ de Spearman y valor p de Mantel por técnica.

**Familia de Holm.** Las tres pruebas de Mantel (una por técnica, m = 3).

**Qué se afirma con cada resultado.**

| Resultado | Se afirma | No se afirma |
|---|---|---|
| ρ alto y Mantel significativo (Holm) | El generador más el *backbone* conservan la geometría de vecindades de **u** en `train` | Que conserven la dirección discriminativa (eso es F4.1) |
| ρ bajo o no significativo | Las vecindades no se conservan en ese espacio de características | Que la información de **u** se pierda: puede conservarse de forma no isométrica |

---

## F4.8 — Figura ilustrativa (A)

**Pregunta** (plan): «¿Cómo se ve la dirección discriminativa?».

**Insumos.** **u** y etiquetas de `train`; la LogReg `umap5d__logreg` de F2.1; generadores de `src/generation/` (Gray-Scott por la ruta de CPU; Perlin requiere el paquete `noise`).

**Método** (plan, con Decisiones SV, 2026-11-06).
1. **Dirección.** La LogReg `umap5d__logreg` de F2.1 (C = 0.1, `balanced`).
2. **Recorridos.**
   - Centroides de clase en `train` y 7 pasos lineales entre ellos.
   - Además, un recorrido de ±1.5 DE sobre la dirección normalizada de w, desde la media global.
   - Los puntos fuera de [0, 1] se recortan y se marcan.
3. **Renderizado.** Con las mismas funciones de `src/generation/` (Gray-Scott por CPU), en escala de grises. Si se usa paleta, solo en la versión ilustrativa.

**Métricas.** Ninguna.

**Qué se afirma y qué no.** La figura solo ilustra, y no sustenta ninguna conclusión (plan: «sin peso argumentativo»).

---

## Registro de decisiones (2026-11-06)

Todos los puntos ABIERTOS del borrador quedan resueltos:

| ID | Resolución, en resumen |
|---|---|
| G1 | `train` y `val` sin restricción; prueba archivada después del commit del veredicto de la Fase 5; los modelos exploratorios nuevos consultan prueba una vez, después de ese commit |
| G2 | Ninguna inferencia nueva de F3.5 ni F3.6 sobre prueba; ablación y cascada en `val` |
| G3 | Familia de Holm = pruebas de una misma pregunta dentro del análisis; cada análisis la declara |
| F4.1-1…7 | Sonda Ridge sobre la penúltima capa; alpha por R² en `val`; *cross-fitting* 5 pliegues; LogReg de §4.3; ΔF1 con IC (sin umbral); `r2_score` por componente; Kaggle + local |
| F4.2-1…6 | Gray-Scott descriptivo; Lorenz Fisher y Mann–Whitney; solo `train`; Perlin Mann–Whitney; Holm sobre F4.2; reintentos regenerados por CPU con comprobación bit a bit; F1 de Lorenz por estrato |
| F4.3-1…9 | Spearman a 56 × 56 y SSIM con criterio relativo; cascada por bloques con `reset_parameters()` y `manual_seed(2026)`; referencia de cosenos con 10 pares; terciles de u4 y perfil radial; clase real principal y `val` principal; ablación gaussiana en `val`; permutación con Holm (m = 3); mínimo de 10 mapas no constantes; lectura de los ViT por casos |
| F4.4-1…3 | Spearman descriptor–mapa por parche con IC; \|d\| como peso; A11 como contexto |
| F4.6-1…5 | Contención con referencia nula; IC sobre las réplicas de §5.2; por semilla; incluye 768D; «consistente con» con doble condición |
| F4.7-1…5 | Características de F4.1; euclídea y coseno; `train`, Mantel 9,999; Holm (m = 3); se ejecuta |
| F4.8-1…3 | `umap5d__logreg`; centroides, 7 pasos y ±1.5 DE recortados; renderizado class-blind |
