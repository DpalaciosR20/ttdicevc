# Plantillas de discusión por veredicto (F6.1)

> Tarea F6.1 del plan v2 · Roles: SV + RP · Estado: **borrador de SV para redacción de RP**
> Debe versionarse **antes de ejecutar F3.5** (protocolo, §12 y Enmienda 1).
> Fuente de las reglas: `docs/protocolo_preregistrado.md` (`protocolo-v1`). Fuente de las cifras: `results/canonical_numbers.json`.

## 0. Propósito y reglas de uso

Estas plantillas fijan, antes de conocer los resultados confirmatorios, **qué se afirmará en la discusión según el veredicto que resulte**. Su función es impedir que la interpretación se ajuste a los resultados: el texto de cada escenario está escrito de antemano, y en la Fase 6 solo se elige el escenario que corresponde y se sustituyen los marcadores por cifras.

1. **Elección del escenario.** La determina exclusivamente la salida de `src/protocol/verdict.py` y de los análisis preregistrados. Nadie elige un escenario por criterio propio.
2. **Sustitución.** Los marcadores ⟦·⟧ se sustituyen por cifras de `canonical_numbers.json`, con el formato de redondeo de cada entrada. No se escriben cifras a mano.
3. **Afirmaciones prohibidas.** Las frases de la columna «No permite afirmar» de la sección 2 no aparecen en el documento de TT-II ni en la presentación, con ninguna redacción equivalente.
4. **Cambios de redacción.** RP puede reescribir el estilo antes de ejecutar F3.5. Después de esa fecha, cualquier cambio de **contenido** se registra como enmienda de este documento, con su motivo y su fecha; los cambios de estilo quedan en el historial de git y no alteran el alcance de ninguna afirmación.
5. **Análisis no preregistrados.** Todo análisis que no figure en el protocolo se presenta en una subsección separada, titulada como exploratoria.

## 1. Marcadores

| Marcador | Significado | Origen |
|---|---|---|
| ⟦t⟧ | Nombre de la técnica (ruido de Perlin, reacción-difusión de Gray-Scott, atractor de Lorenz) | — |
| ⟦arq_t⟧, ⟦est_t⟧ | Arquitectura y estrategia del finalista | F3.3 |
| ⟦F1img_t⟧, ⟦DE_t⟧ | Media y DE del F1-Macro de prueba sobre cinco semillas | F3.5 |
| ⟦F1BETO⟧ | F1-Macro de prueba de BETO (0.7039) | F2.2 |
| ⟦θ_t⟧, ⟦IC_θ_t⟧, ⟦p_θ_t⟧ | Estadístico principal, intervalo al nivel de Holm y valor *p* ajustado | F5.2 |
| ⟦piso_t⟧, ⟦tipo_piso_t⟧, ⟦p_piso_t⟧ | F1 del piso, cuál piso fue el máximo y valor *p* ajustado de superioridad | F5.1 |
| ⟦ρ_t⟧, ⟦IC_ρ_t⟧ | Retención estimada e intervalo al 95 % | F5.2 |
| ⟦L_t⟧, ⟦IC_L_t⟧, ⟦p_L_t⟧, ⟦F1long⟧ | Diferencia frente al clasificador de longitud y su F1 | F4.5 |
| ⟦cab⟧, ⟦red⟧, ⟦rend_t⟧ (con IC) | Pérdida por etapa: cabeza, reducción a 5D, renderizado | F5.3 |
| ⟦F1_768⟧, ⟦F1_5D⟧ | F1 de prueba del clasificador sobre 768D y del techo 5D | F2.1, F2.2 |
| ⟦Q⟧, ⟦p_Q⟧ | Q de Cochran entre técnicas | F5.4 |
| ⟦n_V1_semillas_t⟧ | Semillas (de 5) con veredicto V1 en la sensibilidad por semilla | F5.5 |

## 2. Qué permite afirmar cada veredicto

| Veredicto | Permite afirmar | No permite afirmar |
|---|---|---|
| **V1. Conservación** | Que las imágenes de ⟦t⟧ conservan al menos la mitad de la capacidad discriminativa de BETO por encima del azar, con el nivel de confianza preespecificado | Que la imagen es equivalente al texto; que conserva *toda* la información; que la imagen es una alternativa clínica al texto; que el resultado se generaliza a otros corpus, idiomas o etiquetados |
| **V2. Conservación parcial** | Que las imágenes conservan capacidad discriminativa por encima del azar, pero menos de la mitad de la de BETO | Que la transformación es inútil; que la pérdida se debe a una etapa concreta sin el apoyo de la descomposición (§5) |
| **V3. No conservación** | Que no se detectó capacidad discriminativa por encima del mayor de los pisos preespecificados | Que las imágenes no contienen información emocional; que la hipótesis queda refutada en general, más allá de la técnica y del diseño evaluados |
| **V4. No concluyente** | Que las imágenes superan el piso, pero que con n = 114 no es posible decidir si la retención supera o no el umbral del 50 %, tal como anticipó el análisis de potencia | Que «no hay diferencia» con BETO; que la imagen conserva la información; que la conserva parcialmente |
| **Cualquiera** | El valor de ρ̂ con su intervalo, como estimación de cuánto se conserva | Interpretar la no significancia de McNemar como equivalencia; atribuir la conservación a contenido emocional si la técnica recibió la calificación por longitud (§4) |

## 3. Plantillas por técnica

### 3.1 V1 — Conservación

> Las imágenes generadas mediante ⟦t⟧ conservaron la capacidad discriminativa del texto en la medida exigida por el protocolo preregistrado. El clasificador ⟦arq_t⟧ (⟦est_t⟧) alcanzó un F1-Macro de prueba de ⟦F1img_t⟧ ± ⟦DE_t⟧ sobre cinco semillas, superior al mayor de los pisos de referencia (⟦tipo_piso_t⟧, ⟦piso_t⟧; *p* ajustado = ⟦p_piso_t⟧). El estadístico de retención resultó positivo (θ = ⟦θ_t⟧, IC ⟦IC_θ_t⟧; *p* ajustado = ⟦p_θ_t⟧), de modo que se rechaza la hipótesis de que la imagen conserve menos de la mitad de la capacidad de BETO por encima del azar. La retención estimada fue de ⟦ρ_t⟧ (IC 95 % ⟦IC_ρ_t⟧).

**Variante obligatoria si ρ̂_t ≥ 1 (el visual iguala o supera puntualmente a BETO):**

> Que el clasificador visual iguale o supere puntualmente a BETO no indica que la imagen contenga más información sobre la etiqueta que el texto. Puesto que cada imagen es una función determinista de los componentes UMAP, que a su vez derivan de los embeddings de BETO, la información disponible está acotada por la de la representación de origen (desigualdad de procesamiento de datos). La diferencia observada se atribuye a que los clasificadores comparados difieren en capacidad, regularización y procedimiento de entrenamiento, y a la variabilidad muestral con n = 114, reflejada en la amplitud del intervalo de ρ̂.

### 3.2 V2 — Conservación parcial

> Las imágenes generadas mediante ⟦t⟧ conservaron capacidad discriminativa por encima del azar, pero en una proporción inferior a la exigida. El clasificador ⟦arq_t⟧ (⟦est_t⟧) superó el mayor de los pisos (⟦tipo_piso_t⟧, ⟦piso_t⟧; *p* ajustado = ⟦p_piso_t⟧) con un F1-Macro de ⟦F1img_t⟧ ± ⟦DE_t⟧; sin embargo, el intervalo del estadístico de retención se situó por completo por debajo de cero (θ = ⟦θ_t⟧, IC ⟦IC_θ_t⟧), lo que indica que la imagen conserva menos de la mitad de la capacidad de BETO por encima del azar. La retención estimada fue de ⟦ρ_t⟧ (IC 95 % ⟦IC_ρ_t⟧). La sección de descomposición de la pérdida examina en qué etapa se concentra esa reducción.

### 3.3 V3 — No conservación

> Las imágenes generadas mediante ⟦t⟧ no permitieron detectar capacidad discriminativa por encima del mayor de los pisos preespecificados. El clasificador ⟦arq_t⟧ (⟦est_t⟧) alcanzó un F1-Macro de ⟦F1img_t⟧ ± ⟦DE_t⟧, frente a ⟦piso_t⟧ del piso (⟦tipo_piso_t⟧; *p* ajustado = ⟦p_piso_t⟧). Este resultado no demuestra que la imagen carezca de información sobre la etiqueta: indica que, con los clasificadores, el preprocesamiento y el tamaño muestral evaluados, esa información no fue recuperable por encima de las referencias triviales. La retención estimada fue de ⟦ρ_t⟧ (IC 95 % ⟦IC_ρ_t⟧).

**Variante obligatoria si el piso máximo es el visual permutado:**

> El piso más exigente fue el del propio clasificador entrenado con etiquetas permutadas, lo que sugiere que parte del desempeño del finalista es atribuible a regularidades de las imágenes no relacionadas con la etiqueta, como su estructura o la proporción de clases.

### 3.4 V4 — No concluyente

> Las imágenes generadas mediante ⟦t⟧ superaron el mayor de los pisos preespecificados (⟦tipo_piso_t⟧, ⟦piso_t⟧; *p* ajustado = ⟦p_piso_t⟧), con un F1-Macro de ⟦F1img_t⟧ ± ⟦DE_t⟧, de modo que conservan capacidad discriminativa por encima del azar. El intervalo del estadístico de retención, en cambio, incluyó el cero (θ = ⟦θ_t⟧, IC ⟦IC_θ_t⟧): con 114 muestras de prueba no es posible establecer si la imagen conserva más o menos de la mitad de la capacidad de BETO por encima del azar. Este resultado coincide con lo anticipado por el análisis de potencia, que estimó que la menor retención detectable con potencia de 0.80 era de aproximadamente 0.92. La retención estimada, ⟦ρ_t⟧ (IC 95 % ⟦IC_ρ_t⟧), constituye la mejor descripción disponible de cuánto se conserva.

## 4. Calificación por longitud (F4.5)

Se añade al párrafo de la técnica si el contraste con el clasificador de longitud no rechaza H₀:

> No obstante, el clasificador visual no superó de forma significativa a un clasificador que solo dispone de la longitud del texto (F1-Macro de ⟦F1long⟧; diferencia ⟦L_t⟧, IC ⟦IC_L_t⟧; *p* ajustado = ⟦p_L_t⟧). En consecuencia, el veredicto de ⟦t⟧ se califica como no distinguible de un clasificador de longitud: con este diseño no puede descartarse que lo que la imagen conserva sea, en todo o en parte, la extensión del texto y no su contenido emocional. Esta calificación es pertinente porque el 59.1 % de los textos supera los 128 tokens que procesa BETO.

Si el contraste rechaza H₀:

> El clasificador visual superó además a un clasificador basado únicamente en la longitud del texto (diferencia ⟦L_t⟧, IC ⟦IC_L_t⟧; *p* ajustado = ⟦p_L_t⟧), lo que descarta que la capacidad conservada se reduzca a esa variable.

## 5. Descomposición de la pérdida

Se elige el párrafo según la etapa con mayor pérdida puntual y se agrega la variante de ganancia cuando alguna etapa es negativa.

> **Reducción a 5D dominante.** La mayor parte de la diferencia entre BETO y el clasificador visual se concentra en la reducción de 768 a 5 dimensiones (⟦red⟧), mientras que el renderizado de la imagen añade una pérdida de ⟦rend_t⟧. Esto sitúa el cuello de botella en la representación de entrada a los generadores y no en las técnicas generativas.

> **Renderizado dominante.** La mayor parte de la diferencia se produce al transformar los cinco componentes UMAP en imagen (⟦rend_t⟧), frente a una pérdida de ⟦red⟧ en la reducción dimensional. La técnica ⟦t⟧, o su lectura por parte de los clasificadores visuales, es la etapa limitante.

> **Cabeza de clasificación dominante.** La mayor diferencia aparece entre la cabeza de clasificación de BETO y un clasificador lineal sobre sus propios embeddings (⟦cab⟧). Esta etapa no involucra imágenes y refleja diferencias entre procedimientos de clasificación sobre la misma representación.

> **Variante de ganancia.** En la etapa ⟦etapa⟧, el F1 aumenta (⟦valor⟧). Como en la sección 3.1, esto no implica una ganancia de información, sino diferencias entre los clasificadores comparados y variabilidad muestral; el intervalo correspondiente (⟦IC⟧) indica si la diferencia es distinguible de cero.

> **Cierre común.** Todas las etapas se estiman sobre las mismas 114 muestras y sus intervalos son descriptivos; la descomposición no modifica los veredictos.

## 6. Comparación entre técnicas

> **Sin diferencias.** La Q de Cochran no detectó diferencias en el acierto por muestra entre las tres técnicas (Q = ⟦Q⟧, *p* = ⟦p_Q⟧). Con este tamaño muestral, ello no establece que las técnicas sean equivalentes, sino que no puede ordenarlas.

> **Con diferencias.** La Q de Cochran detectó diferencias entre técnicas (Q = ⟦Q⟧, *p* = ⟦p_Q⟧). Las comparaciones por pares con corrección de Holm indican ⟦resultado por pares⟧. La interpretación de por qué una técnica conserva más información se apoya en los análisis de recuperabilidad de **u** (F4.1) y de régimen dinámico (F4.2), que son explicativos.

## 7. Sensibilidad

> **Robustez.** El veredicto de ⟦t⟧ se mantiene al usar la formulación de margen fijo, al desplazar el F1 de BETO en ±0.0128, al sustituir el finalista por el segundo modelo de la regla de selección y al excluir el visual permutado del piso.

> **Cambio a V4 con r ≥ 0.75 (esperado).** Con retenciones exigidas de 0.75 y 0.80 el veredicto pasa a no concluyente. El análisis de potencia anticipó este resultado: con esos márgenes, la probabilidad de declarar conservación no superaba 0.59 ni siquiera con conservación total. Por ello, este cambio refleja el límite del tamaño muestral y no debilita el veredicto principal.

> **Dependencia de la semilla.** ⟦n_V1_semillas_t⟧ de las cinco semillas, consideradas por separado, conducen a V1. ⟦Si es menor que 5: «El veredicto principal depende, por tanto, del promedio entre semillas, que es el estadístico preespecificado; la variabilidad entre semillas forma parte de la incertidumbre del resultado.»⟧

> **Dependencia del modelo o del piso.** Si el veredicto cambia al sustituir el finalista o el piso, se declara: «El veredicto de ⟦t⟧ es sensible a ⟦factor⟧ y debe leerse junto con esa dependencia.»

## 8. Relación con el resultado exploratorio de WITCOM

> El resultado exploratorio previo (ResNet-18 sobre Gray-Scott, cuya prueba se consultó antes del protocolo) se ⟦«confirma» si Gray-Scott obtiene V1 / «matiza» si obtiene V2 o V4 / «no se reproduce» si obtiene V3⟧ con el diseño confirmatorio. Dado que el conjunto de prueba es el mismo, este acuerdo no constituye una replicación independiente.

Si el F1-Test de BETO reportado en el artículo difiere del verificado: «El artículo reportó un F1-Macro de prueba de BETO de 0.7586; la verificación de TT-II, a partir del modelo que generó los embeddings, lo sitúa en ⟦F1BETO⟧, y es la cifra empleada en este contraste.»

## 9. Integración con la Fase 4

Se incluyen solo si el análisis correspondiente está completo. Son explicativos y no modifican los veredictos.

| Análisis | Plantilla condicional |
|---|---|
| F4.1 (recuperabilidad de **u**) | «Los componentes de **u** que mejor se recuperan a partir de las imágenes de ⟦t⟧ son ⟦componentes⟧ (R² = ⟦R²⟧). ⟦Si coinciden con los de mayor peso en el techo 5D: «coinciden con los más informativos para la clasificación, lo que es coherente con el veredicto».⟧ ⟦Si no: «no coinciden con los más informativos, lo que ayuda a explicar la pérdida en el renderizado».⟧» |
| F4.2 (régimen dinámico) | «La separación entre clases ⟦se asocia / no se asocia⟧ a cambios de régimen de ⟦t⟧ (⟦evidencia⟧). ⟦Si se asocia: «la imagen podría codificar la clase como una diferencia cualitativa de régimen y no como una variación gradual de **u**».⟧» |
| F4.3–F4.4 (atribución) | «Los mapas de atribución ⟦superan / no superan⟧ las pruebas de cordura. ⟦Solo si las superan: descripción de las estructuras y su relación con los descriptores significativos de la Tabla 11.⟧ ⟦Si no las superan: «no se interpretan»⟧.» |
| F4.6 (errores) | «El clasificador visual ⟦comparte / no comparte⟧ los errores de BETO (κ = ⟦κ⟧; ⟦n⟧ errores comunes), lo que ⟦indica que hereda las limitaciones de la representación textual / sugiere que la imagen introduce errores propios⟧.» |

## 10. Veredicto global y conclusión del Trabajo Terminal

### G1 — Conservación

> La hipótesis del Trabajo Terminal se confirma para ⟦técnicas con V1⟧: en ellas, las características que permiten distinguir textos depresivos de no depresivos se conservan, en al menos la mitad de la capacidad discriminativa de BETO por encima del azar, al transformar los embeddings en imágenes abstractas. ⟦Si alguna técnica no obtuvo V1: «Para ⟦otras técnicas⟧ el resultado fue ⟦veredicto⟧, lo que indica que la conservación depende de la técnica generativa.»⟧ La conclusión se limita al corpus, al idioma y al etiquetado empleados.
>
> *Trabajo futuro:* replicación en un corpus independiente; contraste con un margen más exigente con mayor tamaño muestral; estudio de qué propiedades de la técnica favorecen la conservación.

### G2 — Conservación parcial

> La hipótesis se confirma solo en forma parcial: ⟦técnicas con V2⟧ conservan capacidad discriminativa por encima del azar, pero en una proporción inferior a la mitad de la de BETO, y ninguna técnica alcanzó el umbral preespecificado. La transformación preserva una parte de la información emocional, con una pérdida que ⟦según §5⟧ se concentra en ⟦etapa⟧.
>
> *Trabajo futuro:* técnicas generativas o mapeos de **u** que reduzcan la pérdida en la etapa identificada; mayor dimensionalidad de entrada a los generadores.

### G3 — No conservación

> La hipótesis no se confirma: ninguna de las tres técnicas permitió detectar capacidad discriminativa por encima de los pisos preespecificados. Con el diseño evaluado, la transformación de embeddings en imágenes abstractas no preservó información recuperable sobre la etiqueta. Este resultado no excluye que otras técnicas, clasificadores o tamaños muestrales sí la recuperen.
>
> *Trabajo futuro:* verificar dónde se pierde la información con la descomposición por etapas; evaluar mapeos que preserven la geometría de la representación (F4.7).

### G4 — No concluyente

> El contraste no permite confirmar ni refutar la hipótesis: ⟦técnicas⟧ superaron los pisos de referencia, pero, con 114 muestras de prueba, sus intervalos no permitieron decidir respecto del umbral del 50 % de retención. Este desenlace fue anticipado por el análisis de potencia previo al protocolo. Las estimaciones de retención (⟦ρ por técnica⟧) constituyen la respuesta cuantitativa disponible a la pregunta de cuánta información se conserva.
>
> *Trabajo futuro:* ampliar el conjunto de prueba o emplear un corpus adicional para alcanzar la potencia necesaria.

## 11. Limitaciones comunes (se incluyen en todos los escenarios)

1. El conjunto de prueba contiene 114 muestras, lo que limita la potencia de todos los contrastes.
2. Se emplea un único corpus, en un solo idioma y con un único procedimiento de etiquetado; no se afirma invarianza cultural ni lingüística.
3. BETO procesa como máximo 128 tokens y el 59.1 % de los textos se trunca; la información de las imágenes está acotada por la de esa representación truncada.
4. BETO se evalúa con un único entrenamiento; dos entrenamientos nominalmente idénticos difirieron en 0.0128 de F1-Macro de validación.
5. Los clasificadores visuales se entrenan con pesos de clase y BETO sin ellos.
6. La regla de selección prefiere el modelo con menos parámetros entrenables dentro de una desviación estándar del mejor, lo que puede elegir un finalista con una media de validación ligeramente inferior.
7. El conjunto de prueba se consultó previamente en el estudio exploratorio de WITCOM para Gray-Scott.
8. El análisis de preprocesamiento (F1.4) mide la información accesible para un lector lineal de características de ImageNet.
