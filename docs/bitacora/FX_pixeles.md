# FX.1 — Análisis exploratorio de píxeles: atribución, pendiente y regiones

## 1. Identificación

| Campo | Valor |
|---|---|
| Tarea | FX.1, análisis **exploratorio** fuera del protocolo preregistrado (`docs/plan_exploratorio_pixeles.md`, etiqueta `plan-pixeles` sobre `d0eed6b`) |
| Origen | Comentario de un sinodal: «análisis del traslape de los píxeles para ver qué regiones de las imágenes […], por ejemplo, los cambios de pendientes son muy agudos y hacen parte de la clasificación de la CNN» |
| Responsable | Diseño: SV (plan del 2026-11-10). Implementación, ejecución, resolución de PENDIENTES y revisión: CE (Diego Palacios), a cargo de todo FX.1 desde el 2026-11-13 por reasignación de SV |
| Rama | `ce/fx-pixeles-x1` |
| Commit(s) | `0a109dc` (plan, X0) · `676ba9e` (código, X1) · `531acb0` (PENDIENTES resueltos; **commit de ejecución**) · commit de esta bitácora (resultados, claves canónicas, notebook) |
| Entregable | `results/fx_pixeles/local/` (B y D), `results/fx_pixeles/gpu/` (C1 y C2), claves `fx.pixeles.*` en `results/canonical_numbers.json` |
| Evidencia | `results/fx_pixeles/`, `notebooks/FX_pixeles.ipynb`, `results/fx_pixeles/figures/` |
| Fechas de ejecución | 2026-11-16: B y D a las 21:05 UTC (Kaggle CPU, 4.4 min); C1 y C2 a las 21:36 UTC (Kaggle GPU T4, 77 min) |
| Estado | **Cerrada** (exploratoria; no modifica veredictos) |

## 2. Propósito

Responder, con evidencia de fuerza creciente, si los cambios bruscos de intensidad («pendiente») intervienen en la clasificación visual:

| Pregunta | Prueba | Nivel de evidencia |
|---|---|---|
| Q1. ¿La atribución se concentra donde la pendiente es alta? | B (val; Gray-Scott y Perlin) | Asociación |
| Q2. ¿Quitar solo la pendiente alta altera la decisión más que quitar un área de igual forma en otra posición? | C1 (val; tres técnicas); C2, un mapa de regiones sin Grad-CAM ni *rollout* | Necesidad |
| Q3. ¿Dónde difiere la pendiente entre clases, sin modelo? | D (train) | Propiedad del generador |

Complementa la Fase 4 en lo que no cubría: una medida continua de pendiente sin umbral por parche, la segunda derivada, una nula espacial, el contraste con los modelos permutados, una ablación dirigida y no global, y un mapa de regiones válido para Lorenz. Es exploratorio: se diseñó después de conocer los resultados confirmatorios, y el documento lo presenta en una subsección separada.

## 3. Método

### 3.1 Diseño

Lo fija `docs/plan_exploratorio_pixeles.md`, sin cambios frente a la etiqueta (SHA-256 `2972e527…`). En resumen:

- **A, pendiente.**
  - `grad` es la magnitud de Sobel tras una gaussiana de σ = 1 px.
  - `log` es |LoG σ = 1 px|.
  - Ambos con borde `reflect`, a 224 px en B y a 256 px en C1, C2 y D.
- **B, traslape.**
  - Mapa principal de F3.7, crudo: cada semilla dividida entre su media en val y luego la media de las 5 semillas.
  - Rejilla nativa: 7 × 7 en Gray-Scott y 14 × 14 en Perlin.
  - B1: Spearman por muestra entre masa de atribución y pendiente media por parche.
  - B2: Dice del 10 % y 20 % superior frente a la nula de 8 transformaciones diédricas × todos los desplazamientos circulares, sin la identidad (391 y 1,567).
  - B3: real − permutado (F3.6).
  - B4: Mann–Whitney entre clases del Spearman de B1, familia H-B (m = 4).
  - Lorenz no entra en B, porque F4.3 encontró que sus mapas no pasan la cordura por datos.
- **C1, suavizado dirigido.**
  - Para cada imagen de 256 px, mapa ∈ {`grad`, `log`} y q ∈ {10 %, 20 %}, se reemplaza el q superior de píxeles por la imagen desenfocada (σ = 4), con recorte y redondeo a uint8.
  - Control principal: la misma máscara desplazada circularmente, con K = 20 sorteos, |Δx| + |Δy| ≥ 32 y `default_rng(2026)` nuevo por técnica.
  - S_i = |Δp_dep|_alta − media |Δp_dep|_desplazada, con p_dep la media de las 5 semillas de F3.5.
  - Wilcoxon bilateral, familia H-C (m = 12).
  - Secundarios: máscara de azar (K = 20), q inferior y ΔF1-val.
- **C2, oclusión.**
  - Cada parche de una rejilla 8 × 8 (32 px) se reemplaza por su versión desenfocada (σ = 8). O = |Δp_dep| en los 15 finalistas y los 15 permutados.
  - Cordura por datos con el criterio relativo de F4.3: mediana del Spearman real–permutado por semilla frente al rango de los 10 pares de semillas reales, decidida por mayoría.
  - Spearman de O con la pendiente por parche.
- **D, diferencias por clase en train.**
  - Gray-Scott y Lorenz: prueba por clústeres (Maris y Oostenveld, 2007). t de Welch por píxel, |t| > 2.58, conectividad 8, masa Σ|t|, 1,000 permutaciones y p = (k + 1)/(B + 1) del clúster de mayor masa.
  - Perlin: Mann–Whitney de la media por imagen.
  - Familia H-D (m = 6).
- **Estadística común.** IC percentil 95 % por *bootstrap* sobre muestras (B = 10,000, `default_rng(2026)`, una matriz de índices por técnica). Holm con α = 0.05 dentro de cada familia.

**Reglas que hicieron cumplir los scripts:**
- **Prueba:** no se leyó. Ante una fila de prueba el código aborta; los `.npz` de F3.7 se filtraron a val al cargarlos.
- **Modelos:** no se entrenó nada nuevo.
- **Compuerta:** `plan-pixeles` como ancestro, plan sin cambios, `resoluciones.json` versionado, árbol limpio y carpeta de salida vacía.
- **Reproducción previa:** de `p_dep` antes de modificar imágenes.
- **Una sola corrida** por parte, con la salida redirigida a un archivo.

### 3.2 Configuración

`results/fx_pixeles/resoluciones.json`: los valores del plan (`FIJOS`), que el código exige idénticos, más 7 puntos que el plan no fijaba. CE los resolvió el 2026-11-13, antes de calcular:

| Clave | Valor |
|---|---|
| `sobel_borde` | `reflect` |
| `c1_orden_azar` | `tras_los_desplazados_de_cada_imagen` |
| `c1_muestreo_desplazamiento` | `rechazo_dx_luego_dy` |
| `c1_muestreo_azar` | `choice_sin_reemplazo` |
| `b2_mapas` | `grad_y_log` (la menos favorable) |
| `wilcoxon_metodo` | `normal_con_correccion` (la menos favorable) |
| `d_sensibilidad` | `dos_variantes_sigma2_y_t329` |

### 3.3 Entornos y corridas

| Corrida | Plataforma | Alcance | Duración | Evidencia |
|---|---|---|---|---|
| X1 | Local (Windows, CPU) | Código y 40 pruebas sintéticas, sin datos reales | — | `results/fx_pixeles/reporte_x1.md`, `pytest_x1.log` |
| X2 | Kaggle CPU, commit `531acb0` | B y D | 262 s | `results/fx_pixeles/local/` (`run.log`, `manifiesto.csv`, `mapas_verificacion.json` 30/30) |
| X3 | Kaggle GPU Tesla T4, torch 2.11.0+cu130, commit `531acb0` | C1 y C2, K = 20 | 4,641 s (Gray-Scott 5.6 min, Perlin 36 min, Lorenz 36 min) | `results/fx_pixeles/gpu/` (`run.log`, `manifiesto.csv`, `modelos_f35.json` y `modelos_f36.json`, 15/15 cada uno) |
| X4 | Local | Verificación y recálculo independientes, claves canónicas, figuras | — | este documento; `notebooks/FX_pixeles.ipynb` |

**Verificación X4:**
- **Archivos:** los dos `manifiesto.csv` íntegros. El commit desciende de `plan-pixeles`. El SHA-256 de `resoluciones.json` registrado coincide con el blob de ese commit, y el del plan con el de la etiqueta.
- **Reproducción:** Δp_dep ≤ 2.8e-6 en los 30 modelos.
- **Imágenes:** las de val tienen la misma huella en las dos corridas.
- **Recálculo** desde `arreglos.npz` y `b_por_muestra.csv`: S, Wilcoxon, Holm H-C y ΔF1 de C1; la mediana de cordura de C2; la media de B1 y el Mann–Whitney de B4. Todo coincide con lo registrado.

## 4. Resultados

### B. Traslape (val; `results/fx_pixeles/local/resultados.json`)

| Técnica · mapa | B1 real | B1 permutado | B2 ΔDice 10 % real | B3 (B1 real − perm) | B4 p Holm (biserial) |
|---|---|---|---|---|---|
| Gray-Scott · grad | +0.025 [−0.073, +0.122] | +0.603 [+0.541, +0.659] | +0.130 [+0.075, +0.186] | −0.578 [−0.688, −0.471] | 0.00037 (+0.432) |
| Gray-Scott · log | +0.041 [−0.061, +0.140] | +0.608 [+0.546, +0.664] | +0.199 [+0.133, +0.264] | −0.567 [−0.680, −0.458] | 0.00037 (+0.436) |
| Perlin · grad | +0.414 [+0.387, +0.440] | +0.286 [+0.256, +0.314] | +0.184 [+0.165, +0.204] | +0.129 [+0.113, +0.145] | 0.218 (−0.138) |
| Perlin · log | +0.324 [+0.294, +0.353] | +0.198 [+0.166, +0.229] | +0.088 [+0.068, +0.107] | +0.126 [+0.111, +0.142] | 0.215 (−0.180) |

- **B2 permutado** en Gray-Scott: ΔDice entre +0.297 y +0.446, mayor que el real. **B3 de B2** en Gray-Scott: entre −0.281 y −0.148 (IC bajo 0).
- **Perlin, B3 de B2:** IC sobre 0 salvo `log` 10 % (+0.013 [−0.002, +0.026]).
- **Mapas constantes:** ninguno.

### C1. Suavizado dirigido (val; `results/fx_pixeles/gpu/resultados.json`)

| Técnica · mapa · q | S [IC 95 %] | p Holm (H-C) | ΔF1 alta | ΔF1 desplazada (mediana de 20) |
|---|---|---|---|---|
| Gray-Scott · grad · 10 % | +0.010 [−0.021, +0.043] | 1 | −0.053 | −0.103 |
| Gray-Scott · grad · 20 % | +0.119 [+0.086, +0.153] | 1.3e-8 | −0.242 | −0.133 |
| Gray-Scott · log · 10 % | +0.224 [+0.197, +0.252] | 4.9e-19 | −0.299 | −0.085 |
| Gray-Scott · log · 20 % | +0.103 [+0.081, +0.125] | 2.1e-12 | −0.190 | −0.143 |
| Perlin · grad · 10 % | +0.004 [−0.004, +0.013] | 1 | −0.013 | +0.001 |
| Perlin · grad · 20 % | −0.015 [−0.027, −0.002] | 0.017 | −0.021 | −0.019 |
| Perlin · log · 10 % | +0.049 [+0.036, +0.063] | 1.6e-10 | −0.021 | −0.007 |
| Perlin · log · 20 % | +0.034 [+0.017, +0.051] | 0.00059 | +0.015 | −0.006 |
| Lorenz · grad · 10 % | +0.132 [+0.110, +0.155] | 7.0e-19 | −0.077 | −0.009 |
| Lorenz · grad · 20 % | +0.063 [+0.047, +0.080] | 8.4e-12 | −0.038 | −0.012 |
| Lorenz · log · 10 % | +0.110 [+0.091, +0.131] | 1.8e-18 | −0.057 | −0.014 |
| Lorenz · log · 20 % | +0.065 [+0.050, +0.082] | 5.6e-13 | −0.038 | −0.018 |

Holm rechaza en 10 de 12 condiciones: 9 con S > 0 y una con S < 0 (Perlin · grad · 20 %). No rechaza en Gray-Scott · grad · 10 % ni en Perlin · grad · 10 %.

### C2. Oclusión (val)

| Técnica | Cordura (mediana real–perm por semilla; rango de pares reales) | Lectura | Spearman O–grad [IC] | Spearman O–log |
|---|---|---|---|---|
| Gray-Scott | cambia en 4/5 (0.62–0.72; [0.719, 0.763]) | interpretable | +0.756 [+0.715, +0.792] | +0.754 |
| Perlin | cambia en 5/5 (0.006–0.132; [0.237, 0.667]) | interpretable | +0.175 [+0.141, +0.209] | +0.134 |
| Lorenz | cambia en 5/5 (0.973–0.979; [0.983, 0.993]) | interpretable | +0.887 [+0.881, +0.893] | +0.878 |

### D. Diferencias por clase (train)

| Técnica · mapa | p (familia) | p Holm (H-D) | Efecto |
|---|---|---|---|
| Gray-Scott · grad | 0.001 | 0.004 | 28 clústeres con p < 0.05; el mayor, negativo (44,614 px), es un anillo alrededor de un cuadro central donde t > 0 |
| Gray-Scott · log | 0.001 | 0.004 | 2 clústeres: anillo negativo (42,931 px) y cuadro central positivo (9,757 px) |
| Lorenz · grad | 0.001 | 0.004 | 4 clústeres: uno positivo (5,591 px, parte superior y central del atractor) y tres negativos (lóbulos) |
| Lorenz · log | 0.001 | 0.004 | 4 clústeres, con la misma disposición |
| Perlin · grad | 6.7e-55 | 4.0e-54 | biserial +0.614 (más pendiente media en depresivos) |
| Perlin · log | 5.0e-32 | 2.5e-31 | biserial +0.463 |

p = 0.001 es el mínimo posible con 1,000 permutaciones. Las dos variantes de sensibilidad (σ = 2 con 2.58; σ = 1 con 3.29) dan p = 0.001 en los cuatro mapas.

## 5. Hallazgos

**H1. La pendiente alta interviene en la decisión de los clasificadores en las tres técnicas (Q2, C1).**
- Holm rechaza con S > 0 en las cuatro condiciones de Lorenz, en tres de Gray-Scott y en las dos de `log` en Perlin.
- Según la tabla del §8 del plan, se afirma: en val, eliminar los cambios bruscos de intensidad altera la predicción más que eliminar un área de igual forma en otra posición.
- No se afirma que esa información tenga contenido emocional, que sea la única que usa el modelo, ni nada sobre prueba.
- El efecto más grande en Gray-Scott y en Perlin es el de `log` al 10 % (S = +0.224 y +0.049). Con `grad` al 10 % no se detecta en ninguna de las dos. En Lorenz, `grad` y `log` dan efectos parecidos.

**H2. En Perlin, la pendiente (primera derivada) a 20 % es *menos* necesaria que otras regiones de igual forma.**
- S = −0.015 [−0.027, −0.002], Holm 0.017.
- Con `grad` a 10 % no se detecta diferencia. Con `log` sí, y en sentido positivo (H1).
- *Interpretación (no verificada):* la textura estacionaria de Perlin hace que el 20 % superior de `grad` cubra regiones redundantes. La información está en la curvatura y no en la magnitud del borde.

**H3. Perlin cumple el patrón de coincidencia aprendida (Q1, B).**
- B1 tiene IC sobre 0 (+0.41 con `grad`, +0.32 con `log`). El ΔDice de B2 está sobre la nula espacial.
- B3 tiene IC sobre 0: la coincidencia es mayor en el modelo real que en el permutado.
- Según el §8, se afirma: en val, la atribución se concentra donde la pendiente es alta, más de lo que explica la estructura espacial, y esa coincidencia depende del entrenamiento con etiquetas reales.
- No se afirma un uso causal. Eso lo da C1 (H1), con la salvedad de H2.

**H4. En Gray-Scott, la atribución coincide *menos* con la pendiente que la del modelo permutado (B3 negativo).**
- B1 real contiene 0. El permutado da +0.60, y la diferencia real − permutado es −0.58 [−0.69, −0.47].
- Es un caso que la tabla del §8 no prevé. Allí B3 «contiene 0» o es positivo; aquí es negativo.
- *Lo que muestran los datos:* el sesgo hacia los bordes es del modelo sin aprendizaje útil (EfficientNet con ajuste fino sobre etiquetas permutadas). El entrenamiento con etiquetas reales lo desplaza hacia el centro de la imagen, donde F4.3 ya había localizado la diferencia entre clases.
- B4 añade que en los depresivos la atribución coincide más con la pendiente que en los no depresivos: biserial +0.43, Holm 0.0004.
- **C1 sí detecta necesidad en Gray-Scott (H1).** La pendiente interviene en la decisión aunque el mapa de Grad-CAM, a 7 × 7, no se concentre en ella.

**H5. La oclusión «cambia» en las tres técnicas, pero en Lorenz por poco y casi toda ligada al atractor.**
- En Lorenz, los mapas de oclusión real y permutado se parecen mucho: mediana 0.97–0.98, justo por debajo del rango de los pares de semillas reales (0.983–0.993).
- Se correlacionan 0.89 con `grad`. La oclusión pesa donde está el atractor, que es donde hay pendiente.
- Por el criterio de F4.3 es interpretable, y la diferencia por clase (figura 3) se concentra en la parte superior y central del atractor.
- *Interpretación:* buena parte del mapa de Lorenz refleja dónde hay estructura en la imagen, y solo una parte, lo aprendido.
- **C2 es el primer mapa de regiones de Lorenz que pasa la cordura por datos.** Los de F3.7 no la pasaban.

**H6. Las clases difieren en la pendiente por regiones, sin intervención de ningún modelo (Q3, D; propiedad del mapeo de u a la imagen).**
- Gray-Scott: los depresivos tienen más pendiente en un cuadro central y menos en un anillo alrededor.
- Lorenz: más pendiente en la parte superior y central del atractor y menos en sus lóbulos.
- Perlin: más pendiente media en toda la imagen.
- Coinciden con las regiones que alteran la decisión: el centro en Gray-Scott (F4.3 y C2) y la parte superior del atractor en Lorenz (C2).
- No se afirma que el modelo use exactamente esas regiones, ni contenido emocional.

**Desviaciones respecto del procedimiento previsto.**
1. **Los PENDIENTES los resolvió CE y no SV.** SV dejó FX.1 a cargo de CE el 2026-11-13. Se resolvieron antes de cualquier cálculo con datos reales, con la opción menos favorable en los dos casos donde existía (`_pendientes_resueltos` de `resoluciones.json`).
2. **No se enviaron a revisión las salidas de las pruebas acotadas** (`--limit`) antes de las corridas completas. Las corridas completas ocurrieron una sola vez y su duración (77 min de GPU) quedó por debajo del límite de 3 h del plan.
3. **Figuras con imágenes del generador:** la de ejemplos de imagen, `grad`, `log` y modificaciones de C1 requiere `TT_DATA_ROOT`. Está en el notebook, pero no se generó en el equipo local, donde no están las imágenes (§7).

## 6. Limitaciones

1. **Diseño a posteriori.** Exploratorio, no preregistrado, y diseñado después de conocer los resultados confirmatorios y los de la Fase 4. No modifica veredictos.
2. **Sesgo posible de val.** Val ya se usó para la parada temprana, la HPO y la selección de finalistas; B y C pueden estar sesgados hacia esos modelos. Además n = 114.
3. **Imágenes fuera de distribución.** El desenfoque como reemplazo puede producir imágenes que el modelo no vio al entrenar. El control desplazado sufre la misma modificación, lo que mitiga ese efecto sin eliminarlo. La máscara alta de C1 cambia más píxeles «informativos» que la desplazada por construcción: es justo lo que se mide, pero también lo que la hace más fuera de distribución.
4. **Escala de la máscara.** Las máscaras de C1 se definen a 256 px y el modelo ve 224 px.
5. **Resolución de los mapas.** La resolución nativa de los mapas de F3.7 (7 × 7 y 14 × 14) limita B. C2 tiene una rejilla de 8 × 8.
6. **Lorenz.** Sus mapas de F3.7 no pasan la cordura por datos, así que no entra en B. Su mapa de oclusión pasa la cordura por poco (H5).
7. **D describe el generador, no el clasificador.**
8. **Resolución del p en D.** Con 1,000 permutaciones el p mínimo es 0.001; los p de D están en ese piso.

## 7. Estado

**Cerrada** (exploratoria).

| Pendiente | Responsable | Criterio de cierre |
|---|---|---|
| Figura de ejemplos (imagen, `grad`, `log`, modificaciones de C1): última celda de `notebooks/FX_pixeles.ipynb` con `TT_DATA_ROOT` | CE | `results/fx_pixeles/figures/ejemplos_pendiente_c1.png` versionada |
| Subsección exploratoria en el documento (regla 5 de `docs/plantillas_discusion.md`) | RP | Texto basado en §9, con las cifras de `fx.pixeles.*` |

## 8. Artefactos

```
docs/plan_exploratorio_pixeles.md       plan (etiqueta plan-pixeles)
docs/correr_fx_kaggle.md                celdas de X2 y X3
src/fase_x/                             pixeles_comun.py, pixeles_local.py, pixeles_gpu.py, manifiesto.py
tests/test_fx_pixeles.py                40 pruebas sintéticas
results/fx_pixeles/
├── resoluciones.json, reporte_x1.md, pytest_x1.log
├── local/   resultados.json, b_por_muestra.csv, arreglos.npz, run.log, mapas_verificacion.json, entorno/, manifiesto.csv
├── gpu/     resultados.json, arreglos.npz, run.log, modelos_f35.json, modelos_f36.json, entorno/, manifiesto.csv
└── figures/ b_spearman_dice_real_vs_permutado, c1_s_y_delta_f1, c2_oclusion_por_clase, d_mapas_t_clusteres
results/canonical_numbers.json          + 247 claves fx.pixeles.* (src/canonical/build.py: collect_fx)
notebooks/FX_pixeles.ipynb
```

## 9. Texto para el reporte técnico

> **Análisis exploratorio: pendiente de intensidad y regiones de decisión.** A partir de una observación del comité, se exploró si los cambios bruscos de intensidad de las imágenes participan en la clasificación. El análisis se diseñó después de los resultados confirmatorios, usa solo validación y entrenamiento, y no modifica el veredicto. Se midió la pendiente con la magnitud del gradiente de Sobel y su cambio con el laplaciano de la gaussiana. Para cada imagen de validación se reemplazó por su versión desenfocada el 10 % o el 20 % de píxeles de mayor pendiente, y se comparó el cambio en la probabilidad predicha con el que produce la misma máscara desplazada a otras posiciones. En las tres técnicas, eliminar los píxeles de mayor cambio de pendiente alteró la predicción más que eliminar un área de igual forma en otra posición (prueba de Wilcoxon con corrección de Holm). Con la pendiente ocurrió también en Lorenz y, al 20 %, en Gray-Scott. En Perlin, el 20 % de mayor pendiente resultó menos determinante que otras regiones. Es decir, la información de bordes y curvatura interviene en la decisión de los clasificadores, sin que ello implique que tenga contenido emocional.
>
> La coincidencia entre los mapas de atribución y la pendiente difirió por técnica. En Perlin, la atribución se concentró en las zonas de mayor pendiente más de lo que explica la estructura espacial y más que en el modelo entrenado con etiquetas permutadas. En Gray-Scott ocurrió lo contrario: el modelo permutado se alineaba más con los bordes, y el entrenamiento con etiquetas reales desplazó la atribución hacia el centro de la imagen. Un mapa de oclusión, independiente de Grad-CAM y del *rollout*, superó la prueba de cordura por datos en las tres técnicas. En Lorenz quedó muy ligado a la posición del atractor. Por último, sin intervención de ningún modelo, las imágenes de las dos clases difieren en la pendiente por regiones (prueba por clústeres en el conjunto de entrenamiento): un cuadro central y su anillo en Gray-Scott, la parte superior del atractor frente a sus lóbulos en Lorenz, y la pendiente media en Perlin. Esto describe cómo el mapeo de los componentes UMAP a la imagen codifica la clase.
