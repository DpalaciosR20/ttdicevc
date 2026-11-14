# Plan del análisis exploratorio de píxeles (FX.1)

> **Estado:** borrador de SV para revisión. Se versiona en `docs/plan_exploratorio_pixeles.md` **antes** de calcular con datos reales; la etiqueta `plan-pixeles` la crea una persona sobre el commit de fusión del PR.
> **Responsables:** SV (diseño y revisión) · CE (implementación y ejecución).
> **Carácter:** **exploratorio**. No figura en el protocolo preregistrado (`protocolo-v1`), así que se presenta en una subsección separada titulada como exploratoria (regla 5 de `docs/plantillas_discusion.md`). No modifica veredictos, no selecciona modelos y no consulta el conjunto de prueba.
> **Fecha:** 2026-11-10.

## 0. Origen y pregunta

En una reunión de revisión, un sinodal comentó:

> «Yo esperaría ver un análisis del traslape de los píxeles para ver qué regiones de las imágenes […], por ejemplo, los cambios de pendientes son muy agudos y hacen parte de la clasificación de la CNN.»

Se descompone en tres preguntas, de menor a mayor fuerza de evidencia:

| ID | Pregunta | Nivel de evidencia |
|---|---|---|
| Q1 | ¿La atribución de los modelos se concentra donde la intensidad cambia bruscamente? | Asociación |
| Q2 | ¿Eliminar solo los cambios bruscos de intensidad altera la decisión más que eliminar un área de igual forma en otra posición? | Necesidad |
| Q3 | ¿Dónde difieren los cambios bruscos de intensidad entre clases, sin intervención de ningún modelo? | Propiedad del generador |

**Terminología.** «Pendiente» es la primera derivada espacial de la intensidad (magnitud del gradiente, que marca los bordes). «Cambio de pendiente» es la segunda derivada (curvatura, que se mide con el laplaciano). Se miden ambas.

## 1. Qué cubre ya la Fase 4 y qué falta

| Aspecto que pide el sinodal | Cubierto por | Resultado | Qué falta |
|---|---|---|---|
| Regiones en que se apoya el modelo | F4.3: mapas medios por clase, prueba de permutación con Holm, cordura por parámetros y por datos | Gray-Scott: diferencia entre clases concentrada en el centro (p ajustado = 0.0007998). Perlin: diferencia pequeña y significativa (0.03559). Lorenz: los mapas no pasan la cordura por datos y no se interpretan | Un mapa de regiones válido para Lorenz (C2) |
| Dependencia de los cambios bruscos de intensidad | F4.3: ablación espectral por bandas (paso-bajo y paso-alto, σ ∈ {1, 2, 4, 8}) | Quitar las frecuencias altas baja el F1-val hasta 0.2615 (Gray-Scott) y 0.1595 (Perlin); en Lorenz, quitar las bajas lo baja hasta 0.2815 | Una ablación **dirigida a los píxeles de pendiente alta**, con un control de igual área y forma (C1). La ablación por bandas actúa sobre toda la imagen y no localiza |
| Coincidencia entre atribución y bordes | F4.4: Spearman por parche entre la masa del mapa y la densidad de bordes (Sobel con umbral adaptativo por parche) | Gray-Scott: IC contiene 0. Perlin: contiene 0 por poco ([−0.0017, 0.0275]) | Una medida continua de pendiente, sin umbral por parche (limitación N1 de F4.4); la segunda derivada; una nula que respete la autocorrelación espacial; y el contraste con los modelos permutados (B) |
| Diferencias entre clases a nivel de píxel | F4.2 (parámetros de control) y Tabla 11 (descriptores globales) | Las clases ocupan regiones distintas del espacio de parámetros; 8 descriptores significativos | Un mapa por píxel de dónde difiere la pendiente entre clases, con control de multiplicidad (D) |

**Restricción de resolución, que se declara en el documento.** Grad-CAM en EfficientNet-B0 tiene una resolución nativa de 7 × 7, y el *rollout* de ViT-B/16, de 14 × 14. Las pruebas que usan esos mapas (B) trabajan a esa resolución; no se interpola a píxel para no atribuir una precisión espacial que los mapas no tienen. La resolución de píxel solo se usa en las pruebas sin mapas de atribución (C1 y D).

## 2. Reglas generales

1. **Splits.** B y C usan solo `val` (n = 114 por técnica). D usa solo `train` (n = 908), porque no interviene ningún modelo. **Prueba no se lee**: el código aborta si algún insumo trae `split == 'test'` (G2 de `docs/plan_fase4.md`: no hay inferencia nueva de finalistas ni de controles sobre prueba).
2. **Modelos.** Los 15 finalistas de F3.5 (Gray-Scott: EfficientNet-B0 con ajuste fino; Perlin y Lorenz: ViT-B/16 congelado; semillas 42, 7, 123, 2024 y 99) y los 15 controles permutados de F3.6. Ningún entrenamiento nuevo.
3. **Predicción.** `p_dep` = media de las 5 semillas por muestra; ŷ = 1[`p_dep` > 0.5]. F1 = `f1_score(..., average='macro')`.
4. **Preprocesamiento.** El de inferencia de F1.4 (224 px, bilineal con *antialias*), sin aumentos. Toda modificación de la imagen (C) se aplica sobre la PNG de 256 px, en uint8, **antes** de ese preprocesamiento, como en la ablación de F4.3. El resultado se recorta a [0, 255] y se redondea.
5. **Reproducción previa.** Antes de cualquier modificación, la `p_dep` de `val` de cada modelo sobre las imágenes originales debe coincidir con la archivada de F3.5 / F3.6 (diferencia absoluta máxima ≤ 1e-4). Si no coincide, se detiene la ejecución.
6. **Reutilización.** Se reutilizan, sin reescribirlas, las funciones existentes de carga de modelos, preprocesamiento, lectura de mapas de F3.7 y agregación (`src/fase4/atribucion_comun.py`, `atribucion_gpu.py`, `convergencia.py` y las de `src/generation` o `src/selection` que correspondan). Si alguna no existe con esa forma, se documenta en el reporte en lugar de reimplementar su lógica.
7. **Unidad estadística.** La muestra (imagen). IC percentil bilateral al 95 % por *bootstrap* sobre muestras: B = 10,000, `default_rng(2026)`, una matriz de índices por técnica compartida entre las medidas de esa técnica, cuantiles `linear`.
8. **Multiplicidad.** Holm con α = 0.05 dentro de cada familia de la sección 7 (`src.protocol.verdict.holm`). Todo lo demás es descriptivo (IC sin prueba).
9. **Una ejecución.** Cada corrida con datos reales se hace una sola vez, con el árbol limpio, desde un commit que descienda de `plan-pixeles`, con la salida completa redirigida a un archivo (sin tuberías ni filtros sobre el proceso).
10. **Lorenz.** Sus mapas de F3.7 son «dependientes de la cabeza, sin validación por datos» (F4.3). Lorenz **no entra en B**; sí entra en C1, C2 y D, que no usan esos mapas.

## 3. A — Mapas de pendiente (sin modelo)

Para cada imagen, sobre la intensidad en [0, 1]:

| Mapa | Definición | Uso |
|---|---|---|
| `grad` | Magnitud de Sobel, √(gx² + gy²), sobre la imagen suavizada con gaussiana σ = 1 px (`mode='reflect'`) | Pendiente |
| `log` | \|`scipy.ndimage.gaussian_laplace`(I, σ = 1 px, `mode='reflect'`)\| | Cambio de pendiente |
| `canny` | `skimage.feature.canny`(I, σ = 1), umbrales por defecto | Solo figuras; sin estadística |

- **Resolución.** Para B, los mapas se calculan sobre la imagen redimensionada a 224 px con el `Resize` de F1.4, como la ve el modelo. Para C1 y D, sobre la PNG nativa de 256 px.
- **Sensibilidad declarada.** `grad` y `log` con σ = 2 px, solo en B1 y D, descriptiva.

## 4. B — Traslape entre atribución y pendiente (Q1; `val`; Gray-Scott y Perlin)

**Mapa de atribución.** El `map` principal de F3.7, crudo (`map_raw · map_raw_scale`). Cada semilla se divide entre su media en `val` y se promedian las 5 semillas por muestra, igual que en F4.3 y F4.4. Si el mapa promediado de una muestra es constante, la muestra se excluye de B y se cuenta.

**Rejilla nativa.** Gray-Scott: 7 × 7 parches de 32 px. Perlin: 14 × 14 parches de 16 px. Atribución por parche = suma del mapa en el parche; pendiente por parche = media de `grad` (o de `log`) en el parche.

| Prueba | Estadístico por muestra | Resumen | Familia |
|---|---|---|---|
| **B1. Correlación** | Spearman (rangos promedio) entre atribución y pendiente por parche, para `grad` y para `log`. Si alguno de los dos vectores es constante, se excluye la muestra y se cuenta | Media con IC al 95 % | Descriptivo |
| **B2. Traslape del q superior** | Para q ∈ {10 %, 20 %}: k = ⌈q · n_parches⌉ (Gray-Scott: 5 y 10; Perlin: 20 y 40). Dice = \|A ∩ G\| / k entre los k parches de mayor atribución (A) y los k de mayor pendiente (G). Empates: orden estable (`argsort(kind='stable')` sobre el valor negado) | ΔDice = Dice observado − media del Dice nulo (abajo), media con IC al 95 % | Descriptivo |
| **B3. Contraste con el modelo permutado** | B1 y B2 repetidos con los mapas de F3.6 (mismo procesamiento). Diferencia por muestra: real − permutado | Media con IC al 95 % | Descriptivo |
| **B4. Diferencia por clase** | Spearman de B1 (`grad` y `log`) comparado entre clases reales con Mann–Whitney bilateral (asintótico, con corrección de continuidad); correlación biserial de rangos como efecto | p, p ajustado, biserial | **Familia H-B** |

**Nula espacial de B2.** Las 8 transformaciones diédricas de la rejilla de pendiente (rotaciones de 0°, 90°, 180° y 270°, con y sin reflejo) combinadas con todos los desplazamientos circulares (Δx, Δy) de la rejilla, excluida la identidad. Son 391 transformaciones en Gray-Scott (7 · 7 · 8 − 1) y 1,567 en Perlin (14 · 14 · 8 − 1). El Dice nulo de la muestra es la media sobre todas ellas. Esta nula conserva la autocorrelación espacial del mapa de pendiente; barajar parches no la conserva, y por eso no se usa.

## 5. C — Necesidad de la pendiente (Q2; `val`; tres técnicas; GPU)

### C1. Suavizado dirigido

1. **Máscara de pendiente alta.** Para cada imagen de 256 px y cada mapa m ∈ {`grad`, `log`} (sección 3, a 256 px), M_alta = el q superior de píxeles por m, con q ∈ {10 %, 20 %} (k = ⌈q · 65,536⌉ píxeles; empates por orden estable).
2. **Modificación.** I' = I · (1 − M) + B₄(I) · M, donde B₄ es un desenfoque gaussiano con σ = 4 px (`mode='reflect'`) de la imagen completa. Luego se recorta y redondea a uint8 (regla 2.4).
3. **Controles con la misma área**, sobre la misma imagen:
   - **Desplazada (control principal):** M_alta desplazada circularmente (`np.roll`) en (Δx, Δy), con Δx y Δy enteros uniformes en [−128, 127] y \|Δx\| + \|Δy\| ≥ 32. Son K = 20 sorteos por imagen y condición, con `default_rng(2026)` nuevo por técnica, en este orden: técnica → mapa → q → imagen → sorteo. Conserva la forma y el área de la máscara y cambia su posición.
   - **Azar (secundario):** k píxeles uniformes sin reemplazo, con K = 20 sorteos y el mismo generador, a continuación de los desplazados.
   - **Pendiente baja (secundario):** el q inferior de píxeles por m.
4. **Estadístico por muestra.** δ_alta = \|`p_dep`(I'_alta) − `p_dep`(I)\|; δ_desp = media sobre los 20 sorteos de \|`p_dep`(I'_desp,k) − `p_dep`(I)\|. Con S_i = δ_alta − δ_desp, se reporta la media de S con IC al 95 %.
5. **Prueba.** Wilcoxon de rangos con signo, bilateral, sobre los 114 S_i (`zero_method='wilcox'`), por técnica × mapa × q: **familia H-C** (m = 12).
6. **Descriptivo.** ΔF1-val = F1(modificada) − F1(original) para la máscara alta, la baja y cada sorteo de los controles. Para los controles se reporta la mediana y el rango de los 20 sorteos, y la posición de la máscara alta dentro de esa distribución.

### C2. Mapa de oclusión (independiente de Grad-CAM y del *rollout*)

1. **Rejilla.** 8 × 8 parches de 32 px sobre la imagen de 256 px, sin solapamiento.
2. **Oclusión.** Cada parche se reemplaza por el mismo parche de B₈(I) (desenfoque gaussiano con σ = 8 px de la imagen completa); recorte y redondeo a uint8.
3. **Mapa por modelo.** O[parche] = \|`p_dep`(ocluida) − `p_dep`(original)\|, para cada uno de los 15 finalistas y los 15 permutados. El mapa de la técnica es la media de las 5 semillas por muestra.
4. **Cordura por datos.** Spearman por muestra, en la rejilla de 8 × 8, entre O del finalista y O del permutado de la misma semilla. Se compara la mediana de cada semilla con el rango de los 10 pares de semillas reales; «cambia» si queda por debajo del mínimo, y el veredicto se decide por mayoría de semillas. Es el mismo criterio relativo de F4.3; si una semilla da un mapa constante, se excluye y se cuenta.
5. **Correlación con la pendiente.** Spearman por muestra entre O (8 × 8) y la media de `grad` y de `log` por parche (8 × 8); media con IC al 95 %. **Solo se interpreta en las técnicas cuyo O «cambia»** en el punto 4; en las demás se reporta con la etiqueta «sin validación por datos».
6. **Figura.** O medio por clase real y su diferencia.

### C3. Estimación de costo (antes de ejecutar)

- **Pasadas hacia adelante:** 114 imágenes × [C1: 3 técnicas × 2 mapas × 2 q × (1 + 1 + 20 + 20) máscaras × 5 modelos + C2: 64 parches × 30 modelos] = 114 × (2,520 + 1,920) ≈ 506,000.
- **Prueba acotada.** Antes de la corrida completa se hace una con 8 imágenes de `val` y se extrapola el tiempo.
- **Regla de parada.** Si la estimación supera 3 h en una T4, CE se detiene y lo reporta a SV antes de ejecutar. La reducción ya autorizada es K = 10 sorteos en los controles; cualquier otra la decide SV.

## 6. D — Diferencias de pendiente entre clases (Q3; `train`; sin modelo)

1. **Mapas.** `grad` y `log` a 256 px de las 908 imágenes de `train` de cada técnica.
2. **Gray-Scott y Lorenz (prueba por clústeres; Maris y Oostenveld, 2007).**
   - Mapa t de Welch por píxel (depresivo − no depresivo).
   - Clústeres con conectividad 8 entre los píxeles con \|t\| > 2.58, separados por signo; masa del clúster = Σ\|t\|.
   - 1,000 permutaciones de etiquetas (`default_rng(2026)` nuevo por técnica y mapa). La nula es la distribución de la máxima masa de clúster por permutación.
   - p de cada clúster = (k + 1)/(B + 1), con k = número de permutaciones cuya máxima masa es ≥ la masa observada.
   - Para la familia se toma el p del clúster de mayor masa de cada mapa. Se reportan todos los clústeres con p < 0.05, con su área, su signo y su centroide.
3. **Perlin (textura estacionaria, sin posición privilegiada).** Se compara por clase la media por imagen de `grad` y la de `log` con Mann–Whitney bilateral; correlación biserial de rangos como efecto. Los histogramas por clase son descriptivos.
4. **Familia H-D:** Gray-Scott × {`grad`, `log`} + Lorenz × {`grad`, `log`} + Perlin × {`grad`, `log`} (m = 6).
5. **Sensibilidad (descriptiva):** σ = 2 px y umbral \|t\| > 3.29, sin entrar en la familia.

## 7. Familias de Holm (α = 0.05)

| Familia | Pruebas | m |
|---|---|---|
| H-B | B4: Mann–Whitney por clase del Spearman de B1, {Gray-Scott, Perlin} × {`grad`, `log`} | 4 |
| H-C | C1: Wilcoxon de S, {Gray-Scott, Perlin, Lorenz} × {`grad`, `log`} × {10 %, 20 %} | 12 |
| H-D | D: máximo clúster (Gray-Scott y Lorenz) y Mann–Whitney (Perlin), × {`grad`, `log`} | 6 |

## 8. Qué se afirma y qué no

| Resultado | Se afirma | No se afirma |
|---|---|---|
| B1 con IC sobre 0, ΔDice de B2 con IC sobre 0 y B3 con IC sobre 0 | En `val`, la atribución se concentra donde la pendiente es alta, más de lo que explica la estructura espacial, y esa coincidencia depende del entrenamiento con etiquetas reales | Que el modelo use la pendiente de forma causal (eso es C1) |
| B1 o B2 con IC sobre 0, pero B3 con IC que contiene 0 | La coincidencia existe, pero también aparece en el modelo permutado: es un sesgo del modelo hacia los bordes, no algo aprendido de la etiqueta | Que la coincidencia se relacione con la clasificación |
| B1 o B2 con IC que contiene 0 | No se detecta coincidencia a la resolución nativa del mapa | Que la pendiente sea irrelevante (puede serlo de forma distribuida; ver C1) |
| C1: Holm rechaza con S > 0 | En `val`, eliminar los cambios bruscos de intensidad altera la predicción más que eliminar un área de igual forma en otra posición: la información de pendiente interviene en la decisión | Que esa información tenga contenido emocional; que sea la única que usa el modelo; nada sobre prueba |
| C1: Holm no rechaza | No se detecta que la pendiente sea más necesaria que otra región de igual forma | Que el modelo no use la pendiente |
| C1: Holm rechaza con S < 0 | La pendiente alta es menos necesaria que otras regiones de igual forma | Que no intervenga |
| C2: O «cambia» | El mapa de oclusión depende del entrenamiento; su diferencia por clase describe qué parches alteran la decisión | Causalidad más allá de la oclusión de cada parche |
| C2: O «no cambia» | El mapa de oclusión no se distingue del modelo permutado y no se interpreta | — |
| D: clústeres significativos | Las imágenes de las dos clases difieren en la pendiente en esas regiones; es una propiedad del mapeo de **u** a la imagen | Que el modelo use esas regiones; contenido emocional |

## 9. Limitaciones declaradas de antemano

1. Exploratorio, no preregistrado en el protocolo, y diseñado después de conocer los resultados confirmatorios y los de la Fase 4. No modifica veredictos.
2. `val` ya se usó para la parada temprana, la HPO y la selección de finalistas; los resultados de B y C pueden estar sesgados hacia esos modelos.
3. n = 114 en B y C.
4. El desenfoque como reemplazo puede producir imágenes fuera de la distribución de entrenamiento. El control desplazado sufre la misma modificación, lo que mitiga ese efecto sin eliminarlo.
5. Las máscaras de C1 se definen a 256 px y el modelo ve 224 px; el preprocesamiento mezcla ligeramente los bordes de la máscara.
6. La resolución nativa de los mapas (7 × 7 y 14 × 14) limita B.
7. Mapas de atribución de Lorenz sin validación por datos (F4.3): Lorenz solo se analiza con C y D.
8. D describe el generador, no el clasificador.

## 10. Etapas y entregables

| Etapa | Contenido | Datos reales | Entregable |
|---|---|---|---|
| X0 | Este plan, versionado sin cambios en `docs/plan_exploratorio_pixeles.md`; PR revisado por SV; etiqueta `plan-pixeles` creada por una persona | No | PR del plan |
| X1 | Código (`src/fase_x/pixeles_*.py`), resoluciones (`results/fx_pixeles/resoluciones.json`, con los valores de este plan), pruebas sintéticas (`tests/test_fx_pixeles.py`), celdas de Kaggle para C y prueba acotada | No | `results/fx_pixeles/reporte_x1.md`, `pytest_x1.log` |
| X2 | B y D (CPU; local o Kaggle CPU según la prueba acotada) | Sí | `results/fx_pixeles/local/` |
| X3 | C1 y C2 (Kaggle, GPU T4), después de la prueba acotada y la estimación de costo | Sí | `results/fx_pixeles/gpu/` con manifiesto, sha256, entorno y log |
| X4 | Recepción y verificación, claves `fx.pixeles.*` en `canonical_numbers.json`, `notebooks/FX_pixeles.ipynb` con figuras, bitácora `docs/bitacora/FX_pixeles.md` (con «Texto para el reporte técnico») y PR | — | PR final |

**Figuras mínimas:**
- ejemplos por técnica de la imagen, `grad`, `log` y el mapa de atribución (B) o de oclusión (C2);
- distribución de Spearman y de ΔDice, real frente a permutado;
- ejemplos de imágenes modificadas en C1 (alta, desplazada, azar y baja);
- S y ΔF1 con la distribución de los controles;
- mapas t con los contornos de los clústeres (D).

Los mapas de pendiente y las imágenes modificadas se muestran en escala de grises; las paletas de color solo en mapas de calor de atribución, t u oclusión, que son figuras ilustrativas.
