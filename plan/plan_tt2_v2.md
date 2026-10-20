# Plan de Trabajo TT2 (v2) — Módulo 3 y contraste de preservación

Oct 19, 2026 · @Cinco2

## Justificación del ajuste:

- **Alineación con el objetivo original.** El v1 heredaba la pregunta del artículo de WITCOM: si la imagen *añade* poder discriminativo. Esa pregunta está acotada por la desigualdad de procesamiento de datos y no es la del Trabajo Terminal. El objetivo general pregunta si la información **se conserva**, y esa sí es contrastable. El v2 reorienta todo el diseño hacia ella.
- &#32;**El v1 no podía producir un veredicto defendible.** Tenía tres fallas que comprometían el Objetivo V:
  - No definía qué resultado confirmaría o refutaría la hipótesis.
  - Se apoyaba en McNemar, cuya no significancia con n = 114 no demuestra equivalencia, solo falta de potencia.
  - Seleccionaba ganadores con una semilla, cuando la variabilidad entre semillas ya medida (rango de \~0.026) es comparable con las diferencias esperadas entre arquitecturas.
- **Robustez ante cualquier resultado.** La evidencia preliminar sugiere un veredicto de conservación parcial o no concluyente. El v2 está construido para que ese resultado sea tan sólido como uno positivo: los análisis de preservación explican el mecanismo, las plantillas preescritas evitan el sesgo narrativo y la tabla de amenazas anticipa las objeciones del jurado.
- **Lecciones del camera-ready.** Durante la revisión del artículo aparecieron discrepancias entre el código y el texto (`max_length`, configuración de HPO, tasa de falsos negativos, partición). La Fase 0 y el archivo de cifras canónicas existen para que esas inconsistencias no se repitan en el documento de tesis, donde la reproducibilidad es criterio de evaluación

### Qué se mantuvo

El plan v2 conserva el alcance experimental y la estructura de roles del v1; lo que cambia es cómo se decide y se justifica el resultado.

| Elemento del plan v1 | Estado en v2 |
| --- | --- |
| Grid de 3 arquitecturas (ResNet-18, EfficientNet-B0, ViT) × 2 estrategias (frozen, fine-tuning) × 3 técnicas = 18 configuraciones | Se mantiene íntegro |
| 5 semillas (42, 7, 123, 2024, 99) para los finalistas | Se mantiene |
| Selección por F1-Val; conjunto de prueba consultado una sola vez | Se mantiene y se formaliza como regla preregistrada |
| Grad-CAM, con *attention rollout* para ViT, y su conexión con la Tabla 11 | Se mantiene, ampliado (F4.3, F4.4) |
| Comparación entre las tres técnicas generativas | Se mantiene, con pruebas formales (Q de Cochran) |
| No repetir la validación cruzada anidada con BETO reentrenado; split fijo 80/10/10 como referencia | Se mantiene |
| Corrección de capítulos heredados de TT-I y verificación cruzada de cifras | Se mantiene y se amplía |
| Roles: CE entrena, SV valida, RP redacta | Se mantiene |
| Preparación de la defensa, presentando las fugas corregidas como fortaleza | Se mantiene |

### Qué cambió

| # | Aspecto | Plan v1 | Plan v2 |
| --- | --- | --- | --- |
| 1 | Pregunta central | Qué arquitectura clasifica mejor | Cuánta información discriminativa se conserva, dónde se pierde y por qué |
| 2 | Criterio del Objetivo V | No definido | Protocolo preregistrado: no inferioridad frente a BETO, δ = 0.058, cuatro veredictos posibles (V1–V4) |
| 3 | Contraste principal | McNemar contra el baseline 5D | No inferioridad pareada contra BETO; McNemar pasa a ser complementario |
| 4 | Primera pasada del grid | 1 semilla por configuración | 3 semillas, con inferencia solo sobre validación |
| 5 | Referencias | Baselines 5D existentes | Techo 5D ajustado (SVM-RBF, *gradient boosting*), pisos de azar, clasificador sobre descriptores de bajo nivel y control de permutación del clasificador visual |
| 6 | HPO de los modelos visuales | Mencionada, sin tarea asignada | Tarea explícita para cada finalista (F3.4) |
| 7 | Análisis de imágenes | Solo Grad-CAM | Fase 4 completa a cargo de SV: recuperabilidad de parámetros, régimen dinámico, atribución espectral, control de longitud, análisis de errores |
| 8 | Reproducibilidad | No contemplada | Fase 0: código consolidado, prueba unitaria de independencia de etiqueta, entorno fijado, archivo de cifras canónicas, paquete etiquetado |
| 9 | Preprocesamiento | Implícito | Validación de que aumentos y redimensionamiento no destruyen información de **u** (F1.4) |
| 10 | Intercambio CE → SV | Métricas agregadas | Predicciones por muestra en formato fijo, requisito de las pruebas pareadas |
| 11 | Redacción de la discusión | Después de los resultados | Plantillas por veredicto escritas antes de conocer el resultado |
| 12 | Validez | Limitaciones al final | Tabla de amenazas a la validez, cada una con su tarea, y criterios de terminado |
| 13 | Resultado de WITCOM (ResNet-18 sobre Gray-Scott) | Punto de partida | Reclasificado como exploratorio: su prueba ya se consultó |
| 14 | Calendario | Ocho semanas con fechas | Sin fechas; fases ordenadas por dependencias con una sola compuerta: el protocolo se congela antes de abrir la prueba de los finalistas |

## 1. Propósito y reencuadre

Este plan reorienta TT-II hacia el objetivo general original del proyecto: determinar si las características emocionales **se conservan** al transformar los embeddings textuales en imágenes abstractas. El artículo de WITCOM 2026 fue una actividad paralela; sus correcciones se heredan, pero su pregunta (si la imagen *añade* poder discriminativo) no es la pregunta del Trabajo Terminal.

El reencuadre se apoya en un límite teórico. Cada imagen es una función determinista de los cinco componentes UMAP, por lo que la información que contiene sobre la clase no puede exceder la de esos componentes:

```latex
\mathbf{X} = G(\mathbf{u}) \;\Rightarrow\; I(Y;\mathbf{X}) \le I(Y;\mathbf{u})
```

Este límite acota el techo de la clasificación visual, pero no impide medir cuánta información se conserva. La hipótesis de preservación es, por tanto, contrastable, y este plan está diseñado para que cualquier resultado (conservación, conservación parcial o pérdida) quede respaldado con la misma solidez.

### Cambios respecto al plan v1

| Aspecto | Plan v1 | Plan v2 |
| --- | --- | --- |
| Pregunta central | Qué arquitectura clasifica mejor | Cuánta información discriminativa se conserva y dónde se pierde |
| Criterio del Objetivo V | No definido | Protocolo preregistrado con margen de no inferioridad y cuatro veredictos posibles (sección 3) |
| Primera pasada del grid | 18 configuraciones × 1 semilla | 18 configuraciones × 3 semillas; selección por F1-Val promedio |
| Comparación contra BETO | McNemar solo contra 5D-UMAP | Prueba de no inferioridad pareada contra BETO, complementada con McNemar |
| Referencias | Baselines 5D existentes | Techo 5D ajustado, pisos de azar y control de permutación sobre el clasificador visual |
| Análisis de imágenes | Solo Grad-CAM | Rama de análisis de preservación completa a cargo de SV (Fase 4) |
| Reproducibilidad | No contemplada | Fase 0: código del Módulo 2 consolidado, prueba unitaria de independencia de etiqueta, fuente única de cifras |

El resultado obtenido en WITCOM con ResNet-18 sobre Gray-Scott se conserva como **evidencia exploratoria**: su conjunto de prueba ya fue consultado antes de fijar este protocolo, por lo que no puede contar como contraste confirmatorio.

## 2. Roles

Cada rol es responsable de entregables concretos; la rama de análisis de preservación y el contraste estadístico recaen en SV.

| Rol | Integrante | Responsabilidad principal | Objetivos que atiende |
| --- | --- | --- | --- |
| CE — Código y Experimentos | Diego | Consolidación del pipeline, entrenamiento de clasificadores visuales, HPO, controles que requieren reentrenamiento | IV |
| SV — Estadística y Validación | Santiago | Protocolo preregistrado, referencias y pisos, rama de análisis de preservación (A–F), contraste estadístico y veredicto | IV, V |
| RP — Redacción | Marco | Documento de TT-II, corrección de capítulos heredados de TT-I, coherencia de cifras, preparación de la defensa | I–V |

**Regla de intercambio entre roles:** CE entrega a SV las predicciones por muestra (probabilidades por semilla) en archivos, no solo métricas agregadas. Todas las pruebas pareadas de las Fases 4 y 5 dependen de ese formato.

## 3. Protocolo de decisión del Objetivo V (preregistrado)

El Objetivo V se resuelve con una prueba de **no inferioridad** del clasificador visual frente a BETO, con margen fijado antes de consultar el conjunto de prueba de los modelos finalistas. El protocolo se congela en el repositorio, con fecha, al cierre de la Fase 1; cualquier desviación posterior se reporta como tal.

### 3.1 Estadístico principal

Para cada técnica *t*, sobre el split fijo de prueba (n = 114):

```latex
\Delta_t = F1_{\text{BETO}} - \overline{F1}_{\text{img},t}
```

donde el segundo término es el F1-Macro promedio del modelo finalista de la técnica sobre 5 semillas (42, 7, 123, 2024, 99). El intervalo de confianza de cada Δ se obtiene por *bootstrap* pareado: se remuestrean los índices del conjunto de prueba (B = 10,000) y en cada réplica se recalculan ambos F1 sobre las mismas muestras.

### 3.2 Hipótesis y margen

- **H₀:** Δ ≥ δ (el clasificador visual pierde al menos δ respecto a BETO).
- **H₁:** Δ < δ (no inferioridad: la pérdida es menor que el margen).
- Prueba unilateral con α = 0.05, equivalente a exigir que el límite superior del IC bilateral al 90 % quede por debajo de δ. Corrección de Holm sobre las tres técnicas.

**Margen propuesto: δ = 0.058.** Corresponde a la diferencia entre BETO (0.7586) y la mejor línea base léxica clásica, TF-IDF con lematización (0.7004). La lectura es: el clasificador visual conserva la carga emocional si pierde menos de lo que separa a un Transformer ajustado de un modelo léxico superficial. El margen definitivo lo fija SV en la Fase 1 y se reporta un análisis de sensibilidad con δ ∈ {0.03, 0.058, 0.08}.

### 3.3 Piso de referencia

Antes del contraste contra BETO se verifica que el clasificador visual supere al azar. El piso es el máximo F1-Macro entre: clase mayoritaria, predictor aleatorio estratificado, brillo promedio y el mismo clasificador visual entrenado con etiquetas permutadas (Fase 2). La superioridad se prueba con una prueba de permutación sobre las etiquetas de prueba (10,000 permutaciones), con corrección de Holm.

### 3.4 Veredictos posibles

| Veredicto | Condición | Lectura para el Objetivo V |
| --- | --- | --- |
| V1. Conservación | Supera el piso y el límite superior del IC de Δ < δ | Se confirma la preservación de la carga emocional |
| V2. Conservación parcial | Supera el piso y el límite inferior del IC de Δ > δ | La información se conserva por encima del azar, pero con pérdida mayor al margen |
| V3. No conservación | No supera el piso | Se refuta la preservación |
| V4. No concluyente | Supera el piso y el IC de Δ contiene a δ | El tamaño muestral no permite decidir; se reporta la estimación y su incertidumbre |

Los cuatro veredictos son resultados válidos del Trabajo Terminal. El documento se redacta con plantillas para cada uno (Fase 6), de modo que ninguno obligue a reescribir la discusión.

### 3.5 Contrastes secundarios

- **Descomposición de la pérdida por etapa:** retención 768D → 5D (techo 5D frente a BETO) y 5D → imagen (visual frente a techo 5D), cada una con IC *bootstrap*.
- **Comparación entre técnicas:** prueba Q de Cochran sobre los aciertos por muestra de las tres técnicas y McNemar por pares con Holm.
- **McNemar contra BETO y contra el techo 5D:** como complemento, no como criterio de decisión, porque su no significancia no implica equivalencia.

### 3.6 Reglas de selección

- Todo modelo se selecciona por F1-Val promedio entre semillas; el conjunto de prueba se consulta una sola vez por finalista.
- Ante empate dentro de una desviación estándar, se elige el modelo más simple (menos parámetros entrenables).
- El resultado de ResNet-18 sobre Gray-Scott obtenido para WITCOM se reporta como exploratorio y no participa en la selección.

## 4. Fases de trabajo

El plan se organiza en nueve fases encadenadas; la única dependencia rígida es que la inferencia sobre el conjunto de prueba de los finalistas ocurre después de congelar el protocolo (Fase 1).

&#91;embedded content: dependencias entre fases · 9 fases, 1 compuerta\]

Las Fases 2 y 3 avanzan en paralelo tras el protocolo; ambas alimentan el análisis de preservación, que sostiene el veredicto.

### Fase 0 — Consolidación y reproducibilidad

Asegura que los resultados de TT-II sean reproducibles y que la premisa del límite teórico (el generador depende solo de **u**) esté verificada.

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F0.1 | Consolidar en el repositorio el código definitivo de los tres generadores class-blind, con un script único de regeneración y sumas SHA-256 de las 3,408 imágenes | CE | `src/generators/`, `data/images/checksums.csv` |
| F0.2 | Fijar versiones del entorno (umap-learn, numba, torch, timm) y resolver la deserialización de `umap_params.pkl`, o documentar el re-ajuste con los hiperparámetros del contrato | CE | `requirements.txt` y nota técnica |
| F0.3 | Prueba unitaria de independencia de etiqueta: con el mismo **u**, cada generador debe producir la misma imagen al variar etiqueta, índice, split y ruta; verificar que ninguna semilla dependa de un índice ordenado por clase | SV | `tests/test_label_independence.py` y reporte |
| F0.4 | Archivo de cifras canónicas generado por código, del que se toman todas las cifras del documento | SV | `results/canonical_numbers.json` |
| F0.5 | Corregir los capítulos heredados de TT-I: partición 80/10/10, `n_iter` 7,000, `max_length` 128 con truncamiento del 59.1 %, HPO de 12 trials y 5 hiperparámetros sin pruner, falsos negativos del 30 % en Exp 4.1, BETO sin pesos de clase | RP | Capítulos 1 y 2 corregidos |

### Fase 1 — Protocolo preregistrado

Fija las reglas de decisión antes de generar resultados confirmatorios.

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F1.1 | Análisis de potencia por *bootstrap* con las predicciones existentes (BETO y CNN Gray-Scott exploratorio): ancho esperado del IC de Δ con n = 114 y diferencia mínima detectable | SV | Nota de potencia |
| F1.2 | Fijar δ, α, B, pisos, reglas de selección y contrastes secundarios (sección 3); congelar con commit fechado | SV | `docs/protocolo_preregistrado.md` |
| F1.3 | Matriz de experimentos: 3 arquitecturas × 2 estrategias × 3 técnicas, 3 semillas en la primera pasada; método de interpretabilidad por arquitectura (Grad-CAM en `layer4` de ResNet, último bloque de `features` en EfficientNet, *attention rollout* en ViT) | SV + CE | Matriz de experimentos |
| F1.4 | Validar que el preprocesamiento preserva la información de **u** por técnica: los volteos o rotaciones pueden destruir información codificada en la orientación (p. ej., ángulo de elevación en Lorenz) y el redimensionamiento de 256 a 224 px puede borrar trazos finos en Perlin. Decidir aumentos permitidos y resolución de entrada por técnica | SV + CE | Tabla de decisiones de preprocesamiento |
| F1.5 | Formato de entrega de predicciones: `sample_id, split, y_true, technique, arch, strategy, seed, p_dep` | SV | Especificación del formato |

### Fase 2 — Referencias: techo y pisos

Establece contra qué se mide la conservación: lo máximo alcanzable desde **u** y lo mínimo atribuible al azar.

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F2.1 | Techo 5D: SVM-RBF y *gradient boosting* con selección por F1-Val, junto a LogReg, RF y MLP existentes; el techo es el mejor por F1-Val | SV | Predicciones por muestra y tabla |
| F2.2 | Predicciones por muestra de BETO (Exp 4.1) y de un clasificador sobre los embeddings de 768D, para la descomposición de la pérdida | SV | Predicciones por muestra |
| F2.3 | Pisos: clase mayoritaria, aleatorio estratificado y brillo promedio, por técnica | SV | Tabla de pisos |
| F2.4 | Clasificador sobre los cuatro descriptores de bajo nivel (entropía, contraste, densidad de bordes, energía LBP), por técnica; eslabón intermedio entre el brillo y la CNN, solicitado por un revisor | SV | Tabla y predicciones |

### Fase 3 — Clasificación visual (Objetivo IV)

Produce los tres modelos finalistas, uno por técnica, y sus predicciones sobre prueba.

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F3.1 | Función de entrenamiento generalizada (ResNet-18, EfficientNet-B0, ViT-B/16 preentrenados; frozen y fine-tuning; pesos de clase; early stopping por F1-Val) y prueba de humo | CE | Módulo de entrenamiento |
| F3.2 | Primera pasada: 18 configuraciones × 3 semillas (42, 7, 123), **solo con inferencia sobre validación** | CE | Predicciones de validación |
| F3.3 | Selección del finalista por técnica según F1-Val promedio, con su dispersión entre semillas | SV | Tabla de selección |
| F3.4 | HPO breve de cada finalista con Optuna sobre F1-Val (tasa de aprendizaje, *weight decay*, regularización), con presupuesto fijo documentado | CE | Hiperparámetros finales |
| F3.5 | Entrenamiento final de los tres finalistas con 5 semillas e inferencia única sobre prueba | CE | Predicciones de prueba |
| F3.6 | Control de permutación visual: cada finalista reentrenado con etiquetas de entrenamiento permutadas (semilla 99); debe caer al nivel de azar | CE | Predicciones del control |
| F3.7 | Mapas de Grad-CAM o *attention rollout* de los finalistas sobre validación y prueba | CE | Mapas por muestra |

### Fase 4 — Análisis de preservación (rama de SV)

Explica el veredicto: cuánta información de **u** sobrevive en cada técnica, en qué propiedades visuales reside y qué amenazas podrían estar imitando una conservación aparente. Las letras A–F corresponden a la propuesta original de la rama.

| ID | Análisis | Pregunta | Método | Entregable |
| --- | --- | --- | --- | --- |
| F4.1 (B) | Recuperabilidad de parámetros | ¿Qué componentes de **u** preserva cada técnica? | Regresor imagen → **u** con el backbone del finalista (5 salidas, MSE); R² por componente y técnica. Además, LogReg entrenada sobre las **û** recuperadas frente a la misma sobre **u** real: la diferencia mide la pérdida en la dirección discriminativa | Tabla de R² y de F1 sobre **û** |
| F4.2 (C) | Régimen dinámico | ¿La separación de clases se debe a cruzar fronteras de régimen en vez de una codificación continua? | Gray-Scott: muestras en el plano (f, k) de Pearson por clase. Lorenz: distribución de ρ por clase respecto a la bifurcación ρ ≈ 24.74. Perlin: distribución de octavas por clase. Tasa de reintentos del *wrapper* de Gray-Scott por clase | Figuras de régimen y tabla de reintentos |
| F4.3 (E) | Atribución espacial y espectral | ¿Sobre qué estructuras discrimina el modelo? | Grad-CAM agregado por clase (marco canónico en Lorenz, perfil radial en Gray-Scott); prueba de cordura por aleatorización de pesos; ablación por filtrado paso-bajo y paso-alto con medición de la caída de F1 | Mapas agregados y curva de ablación espectral |
| F4.4 | Convergencia con la Tabla 11 | ¿Las propiedades usadas por el modelo coinciden con los descriptores significativos? | Contraste cualitativo y cuantitativo entre F4.3 y las 8 comparaciones significativas | Párrafo de convergencia por técnica |
| F4.5 (F) | Control de longitud del texto | ¿Lo que se conserva es contenido emocional o longitud? | Correlación de cada componente de **u** con la longitud en tokens y con el indicador de truncamiento; evaluación estratificada por terciles de longitud del finalista y de BETO | Tabla de correlaciones y F1 por estrato |
| F4.6 | Análisis de errores | ¿El clasificador visual hereda los errores de **u**? | Kappa de Cohen entre vectores de acierto de BETO, techo 5D y finalista visual; intersección de errores | Tabla de concordancia y diagrama de intersección |
| F4.7 (D) | Correspondencia geométrica (opcional) | ¿El generador preserva las vecindades del embedding? | Correlación de Spearman entre matrices de distancia en **u** y en el espacio de características de imagen; prueba de Mantel | Coeficiente y p-valor por técnica |
| F4.8 (A) | Figura ilustrativa | ¿Cómo se ve la dirección discriminativa? | Renderizado de los centroides de clase y del recorrido sobre la dirección de la LogReg en **u**; sin peso argumentativo | Una figura |

F4.1 y F4.6 son los análisis de mayor valor probatorio: si las **û** recuperadas reproducen el F1 de **u** y los errores visuales están contenidos en los errores 5D, el mecanismo de conservación queda demostrado, no solo inferido. F4.5 protege la validez de constructo de "características emocionales".

### Fase 5 — Contraste estadístico y veredicto (Objetivo V)

Aplica el protocolo congelado sin desviaciones.

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F5.1 | Superioridad sobre el piso por técnica (sección 3.3) | SV | Tabla de pruebas de permutación |
| F5.2 | No inferioridad frente a BETO por técnica, con Holm (secciones 3.1 y 3.2) | SV | Tabla de Δ con IC y p-valores ajustados |
| F5.3 | Descomposición de la pérdida por etapa con IC (768D → 5D → imagen) | SV | Tabla de retención por etapa |
| F5.4 | Comparación entre técnicas: Q de Cochran y McNemar por pares con Holm | SV | Tabla de comparaciones |
| F5.5 | Sensibilidad del veredicto a δ ∈ {0.03, 0.058, 0.08}, al segundo mejor modelo por técnica y a la semilla | SV | Tabla de robustez |
| F5.6 | Veredicto por técnica y global (V1–V4) | SV | Tabla final del Objetivo V |

### Fase 6 — Síntesis interpretativa

Convierte el veredicto en una conclusión defendible, sin sesgo narrativo.

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F6.1 | Redactar plantillas de discusión para cada veredicto (V1–V4) **antes** de conocer los resultados de la Fase 5 | SV + RP | Cuatro plantillas |
| F6.2 | Integrar el veredicto con los análisis de la Fase 4: qué se conserva, dónde se pierde y por qué | SV + RP | Sección de discusión |
| F6.3 | Redactar la sección de amenazas a la validez a partir de la sección 5 de este plan | SV + RP | Sección de amenazas a la validez |

### Fase 7 — Documento de TT-II

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F7.1 | Metodología del Módulo 3, incluido el protocolo preregistrado | RP | Capítulo de metodología |
| F7.2 | Resultados del Módulo 3, con todas las cifras tomadas de `canonical_numbers.json` | RP | Capítulo de resultados |
| F7.3 | Discusión, limitaciones, conclusiones y trabajo futuro | RP | Capítulos finales |
| F7.4 | Verificación cruzada de cifras del documento contra el archivo canónico | Todos | Lista de verificación firmada |
| F7.5 | Paquete de reproducibilidad: código, entorno, semillas, sumas de verificación, configuración del mapeo y metadatos; restricciones de privacidad del corpus declaradas | CE + SV | Repositorio etiquetado |
| F7.6 | Formato final según los lineamientos de titulación de ESCOM-IPN | RP | Documento final |

### Fase 8 — Defensa

| ID | Tarea | Rol | Entregable |
| --- | --- | --- | --- |
| F8.1 | Banco de preguntas esperadas con respuesta: límite teórico (por qué imágenes si 5D basta), fugas encontradas y corregidas, potencia con n = 114, truncamiento a 128 tokens, fuga de BETO en validación cruzada, ViT con pocos datos, determinismo del generador | Todos | Banco de preguntas |
| F8.2 | Ensayos de la presentación | Todos | Presentación final |

## 5. Amenazas a la validez

Cada amenaza identificada tiene una tarea asignada que la atiende o una limitación que la declara; ninguna queda implícita.

| Amenaza | Tipo de validez | Cómo se atiende |
| --- | --- | --- |
| Fuga de etiqueta residual en algún generador | Interna | Prueba unitaria de independencia (F0.3) |
| Preprocesamiento que destruye información de **u** (aumentos, redimensionamiento) | Interna | Tabla de decisiones de preprocesamiento (F1.4) |
| Sesgo de selección al elegir el máximo entre 18 configuraciones | Conclusión estadística | 3 semillas desde la primera pasada, selección por F1-Val, prueba consultada una vez (F3.2, F3.3, F3.5) |
| Baja potencia con n = 114 | Conclusión estadística | Análisis de potencia previo (F1.1) y veredicto V4 explícito |
| Interpretar la no significancia como equivalencia | Conclusión estadística | Prueba de no inferioridad con margen preregistrado (F5.2) |
| Variabilidad entre semillas y no determinismo de cuDNN | Conclusión estadística | 5 semillas por finalista y análisis de sensibilidad (F3.5, F5.5) |
| Comparación asimétrica: BETO con HPO, clasificadores visuales sin ella | Interna | HPO breve por finalista (F3.4) |
| La conservación aparente es longitud del texto y no contenido emocional | Constructo | Control de longitud y evaluación estratificada (F4.5) |
| La separación se debe a fronteras de régimen dinámico | Constructo | Análisis de régimen (F4.2) |
| Mapas de atribución no fiables | Constructo | Prueba de cordura por aleatorización de pesos (F4.3) |
| Truncamiento al 59.1 % del corpus limita el techo de todo el pipeline | Constructo | Declarado como limitación; acotado parcialmente por F4.5 |
| Fuga de BETO al re-particionar para validación cruzada | Conclusión estadística | Solo el split fijo 80/10/10 es confirmatorio; cualquier validación cruzada se reporta como estimación optimista |
| Conjunto de prueba ya consultado para ResNet-18 sobre Gray-Scott | Conclusión estadística | Resultado clasificado como exploratorio (sección 3.6) |
| Un solo corpus, idioma y protocolo de etiquetado | Externa | Declarado como limitación; no se afirma invarianza cultural |
| Cifras inconsistentes entre capítulos | Confiabilidad del reporte | Archivo canónico y verificación cruzada (F0.4, F7.4) |

## 6. Criterios de terminado

El proyecto se considera concluido cuando se cumplen todos los criterios siguientes, independientemente de cuál sea el veredicto.

- [ ] El protocolo preregistrado está congelado con fecha anterior a la inferencia sobre prueba de los finalistas.
- [ ] La prueba unitaria de independencia de etiqueta pasa para las tres técnicas.
- [ ] Existen tres finalistas, uno por técnica, evaluados con 5 semillas y con predicciones por muestra archivadas.
- [ ] El Objetivo V tiene un veredicto (V1–V4) por técnica, con IC, p-valores ajustados por Holm y análisis de sensibilidad.
- [ ] La pérdida está descompuesta por etapa (768D → 5D → imagen) con sus intervalos.
- [ ] Los análisis F4.1, F4.2, F4.3, F4.5 y F4.6 están completos e integrados en la discusión.
- [ ] Todas las cifras del documento coinciden con `canonical_numbers.json`.
- [ ] El paquete de reproducibilidad está etiquetado en el repositorio.
- [ ] El documento de TT-II cumple los lineamientos de titulación de ESCOM-IPN.
