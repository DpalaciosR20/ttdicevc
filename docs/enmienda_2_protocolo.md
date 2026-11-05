## Enmienda 2 (2026-11-04)

**Motivo.** Antes de ejecutar F2.3, F2.4 y las predicciones del clasificador de longitud (F4.5) se identificaron cuatro definiciones que el protocolo no fija por completo. Esta enmienda se registra **antes de cualquier consulta de prueba de los finalistas** y antes de calcular cualquiera de estas referencias; conforme a §11, no modifica ningún criterio de decisión ya fijado.

**1. Clasificador sobre descriptores de bajo nivel (§4.2, F2.4).**
- *Descriptores:* entropía de Shannon, contraste, densidad de bordes y energía LBP, con las definiciones de la Tabla 11 del Módulo 2, tal como las implementa el código original. La implementación y sus parámetros (número de bins, vecindario y radio de LBP, umbrales del detector de bordes) se fijan en `configs/protocolo.json` → `descriptores` antes de calcularlos. Si el código original no puede localizarse, la tarea se detiene y la definición se fija en una enmienda adicional antes de continuar.
- *Imagen de entrada:* la imagen nativa de 256 × 256 en escala de grises, sin el preprocesamiento de los clasificadores visuales.
- *Modelos:* las cinco familias y rejillas de §4.3, con `StandardScaler` ajustado en `train` para LogReg, SVM y MLP, y la misma regla de selección por F1-Val y desempate.
- *Alcance:* un clasificador por técnica.

**2. Piso de brillo (§4.4, F2.3).** Intensidad media de la imagen nativa de 256 × 256 en escala de grises, dividida entre 255. Modelo: LogReg con la rejilla completa de LogReg de §4.3 (C y `class_weight`), con `StandardScaler` ajustado en `train` y la regla de selección y desempate de §4.3.

**3. Pisos mayoritaria y aleatorio estratificado (§4.4).**
- *Mayoritaria:* `p_dep = 0.0` en todas las muestras.
- *Aleatorio estratificado:* en los contrastes interviene como la constante 0.5 (§5.3). Se archiva además un archivo de predicciones con `p_dep ∈ {0.0, 1.0}`, generado con `numpy.random.default_rng(42)` y la proporción de clases de `train`, por completitud del formato; ese archivo no interviene en ningún contraste.

**4. Clasificador de longitud (§4.2, F4.5).** Se añade al vocabulario controlado de `technique` (especificación de predicciones, §3.2) el valor `text_length`. El clasificador se registra como `technique = text_length`, `arch = logreg`, `strategy = none`, `seed = 42`, fase `referencia`. Características: `log(n_tokens_beto)` y `truncated_128` de `meta/sample_index.csv`. Modelo: LogReg con la rejilla completa de LogReg de §4.3, `StandardScaler` ajustado en `train` y la regla de selección y desempate de §4.3.
