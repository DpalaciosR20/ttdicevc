# Hallazgos Metodológicos — Detalle Completo

Este documento existe para que nadie tenga que re-descubrir (bajo presión de tiempo) problemas que ya se encontraron y resolvieron durante el desarrollo del camera-ready de WITCOM. Léelo antes de tocar el pipeline de generación de imágenes o de diseñar cualquier validación cruzada.

---

## 1. Fuga de color por clase (las 3 técnicas)

**Problema:** las imágenes originales del Módulo 2 estaban coloreadas por clase — paleta fría (PuBuGn) para depresivo, cálida (YlOrRd) para no depresivo. El dataset "ciego a la clase" que la metodología prometía nunca se había generado.

**Riesgo:** un clasificador de visión podría aprender a distinguir clases por el color/brillo de la paleta en vez de por la geometría real de la imagen — invalidando cualquier resultado de clasificación.

**Verificación empírica:** se probó un clasificador trivial (solo brillo promedio) sobre las imágenes originales de Gray-Scott — dio F1-Macro=0.5077 (≈ azar), sugiriendo que el brillo simple no era explotable, pero no descartaba fugas más sutiles.

**Solución:** se regeneraron las 3 técnicas desde cero, en escala de grises real (1 canal), verificando **estructuralmente** (por firma de función) que ningún renderizador recibe la etiqueta en ningún punto del proceso — el label solo se usa para nombrar el archivo de salida, nunca para el cálculo de píxeles.

---

## 2. Fuga de semilla aleatoria en Perlin

**Problema:** el generador original de Perlin usaba `seed = label*10000 + i` como semilla para el ruido base Y para las posiciones iniciales de las partículas. Esto significaba que la **geometría misma** (no solo el color) dependía de la etiqueta — muestras `label=1` siempre con semillas ≥10,000, `label=0` con semillas <10,000.

**Por qué es grave:** a diferencia de la fuga de color (solo cosmética), esta fuga afecta directamente la estructura espacial de la imagen — un CNN podría explotarla de forma mucho más sutil y difícil de detectar visualmente.

**Solución:** nueva semilla determinista derivada de un hash MD5 de los propios 5 valores UMAP (`seed_from_umap` en `perlin_lib.py`) — depende solo del embedding, nunca del label ni del índice de muestra.

---

## 3. Fuga por UMAP no reajustado (en intentos de k-fold)

**Problema:** al combinar las 1,136 muestras (train+val+test originales) y re-particionar para hacer k-fold, ~80% de cada "fold de test" resultan ser muestras que el reductor UMAP **ya vio durante su propio ajuste** (porque el UMAP se ajustó una sola vez sobre las 908 muestras del train original). UMAP, al ajustarse, acomoda esos puntos específicos para preservar su vecindad — quedan artificialmente mejor separados que puntos genuinamente nuevos transformados después.

**Evidencia:** un primer intento de k-fold dio F1≈0.82-0.83 para los baselines UMAP y el CNN, muy por encima del ~0.72-0.73 del split fijo original — salto demasiado grande para explicarse solo por variación de muestreo.

**Solución (k-fold anidado):** reajustar UMAP + MinMaxScaler **dentro de cada fold**, usando solo el train de ese fold, y transformar el test de ese fold como genuinamente nuevo (out-of-sample). Función de referencia:

```python
def fit_umap_scaler(X_fit, random_state=42):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine',
                         n_components=5, random_state=random_state)
    coords_fit = reducer.fit_transform(X_fit)
    scaler = MinMaxScaler().fit(coords_fit)
    return reducer, scaler, scaler.transform(coords_fit)

def transform_umap_scaler(reducer, scaler, X):
    return scaler.transform(reducer.transform(X))
```

Esto requiere los embeddings BETO de 768D (`Exp_4_1_BETO_HPO_embeddings.pkl`), no el `umap_params.pkl` ya reducido.

---

## 4. Fuga por fine-tuning de BETO (no resuelta del todo — limitación documentada)

**Problema:** incluso con UMAP reajustado por fold (punto 3 resuelto), los resultados de k-fold anidado y de un holdout final **seguían dando ~0.80-0.86** — casi tan altos como con la fuga de UMAP sin resolver. La explicación: **BETO fue fine-tuneado específicamente sobre las 908 muestras del split de train original.** Cualquier re-partición nueva (para dev/holdout del k-fold anidado) que no respete esa frontera original termina incluyendo, por pura probabilidad (~80% de las veces), muestras que BETO ya vio durante su propio entrenamiento supervisado — sus embeddings para esos textos están "más limpios" (organizados para separar las clases) que para textos genuinamente nunca vistos por BETO.

**Confirmado de forma independiente dos veces:** una corrida local de un colaborador externo (GPU propia, lr=1e-4, sin seed) y una corrida en Kaggle (GPU, lr=3e-5, con seed) — ambas con implementaciones de código distintas, ambas cayeron en el mismo rango (~0.80-0.86) para nested-CV/holdout.

**Por qué no se resolvió del todo:** arreglarlo requeriría reentrenar BETO (con su propia búsqueda de hiperparámetros vía Optuna) dentro de cada fold de validación — computacionalmente muy caro (horas por fold, multiplicado por 5+ folds), no viable en el tiempo disponible ni para el camera-ready ni, probablemente, para TT2 tampoco.

**Decisión operativa (aplica a TT2 también):**
- El **split fijo original 80/10/10** es la referencia principal y confiable para reportar cualquier resultado de clasificación.
- Cualquier k-fold anidado u holdout adicional se reporta explícitamente como **"estimado optimista"**, con esta limitación explicada — no como una métrica de generalización libre de sesgo.
- No se debe repetir un k-fold *simple* (sin reajustar UMAP) como si fuera una métrica válida — hereda ambas fugas (3 y 4).

---

## 5. Bug de nomenclatura de archivos (encontrado por un colaborador externo)

**Problema:** en una primera versión del código de generación para k-fold anidado, la función que construía las rutas de archivo (`generate_gray_scott_batch`) no incluía el prefijo `rd_` que sí usaba la función real de guardado (`generate_and_save_grayscale`) — esto rompía tanto la detección de "ya generado" (el chequeo `os.path.exists` nunca encontraba coincidencias) como la carga de imágenes para entrenamiento (rutas devueltas apuntaban a archivos que no existían).

**Solución:** centralizar la construcción de nombres de archivo en una sola función (`expected_path`), usada tanto para guardar como para verificar existencia — ya aplicado en `gray_scott_lib.py`, `perlin_lib.py`, y `lorenz_lib.py` de este repo.

**Lección para nuevo código:** cualquier función nueva de generación/carga de imágenes debe usar `expected_path(...)` en vez de reconstruir el nombre de archivo de forma independiente en más de un lugar.

---

## 6. Otras correcciones menores identificadas (para el texto de TT2)

- **Partición real: 80/10/10** (908/114/114), verificado directamente desde los artefactos serializados — no 70/15/15 como decía el TT1 en algunas secciones.
- **Gray-Scott usa `n_iter` con tope de 7,000**, no 9,000 como decía el texto original — verificado en el código real.
- **BETO no usó class weights** en su fine-tuning (CrossEntropyLoss estándar, sin ponderar) — a diferencia del clasificador clásico de referencia (Regresión Logística sobre TF-IDF/embeddings congelados), que sí usa `class_weight='balanced'`. Es una asimetría real entre módulos, documentada tal cual, sin inventar una justificación retroactiva que nadie recuerda.
- **Variabilidad entre entornos de ejecución:** con semilla fija, entrenar el mismo modelo en distinto hardware (Colab, Kaggle, GPU local) da resultados distintos (~8 puntos porcentuales de F1 observados entre corridas nominalmente idénticas) — cuDNN no es determinista por defecto entre GPUs distintas. Por eso se reporta media±std de 5 semillas en vez de un solo run.

---

## 7. Rango de u en val/test: recortados a [0,1], no extrapolados (verificado con el archivo real)

**Historia:**
- **Primera versión (19 oct 2026), INCORRECTA:** con base en `docs/artefactos_datos.md` ("val/test se salen ligeramente") y en que ningún generador recorta su entrada, se supuso que algunas imágenes de val/test se habían generado con parámetros fuera del contrato (p. ej. Lorenz con 250 puntos).
- **Corrección (21 oct 2026, F0.2):** al leer el `umap_params.pkl` real (sha256 `081f5bab…f224`) con `src/training/umap_io.py`, resultó que val y test **están dentro de [0,1]**: mínimo 0 y máximo 1 en los tres splits, 0 muestras fuera de rango.

**Diagnóstico: los valores se recortaron (`clip`) al guardarlos, no se escalaron por split.**
- En train hay exactamente un valor en 0 y uno en 1 por componente: son el mínimo y el máximo del ajuste del MinMaxScaler.
- En val y test hay valores **apilados** en exactamente 0 o 1. Por ejemplo, en val hay 6 valores exactamente en 0 o 1 en el componente u2. Además, en algunos componentes el mínimo es mayor que 0 (val u0: 0.0243). Eso es lo que deja un `np.clip` después de `scaler.transform`. Un MinMax ajustado por split pondría siempre un único 0 y un único 1.
- El scaler guardado tiene `clip=False`, así que el recorte se aplicó aparte, en el código que armó el pkl. Ese código no está en el repo.

**Consecuencias:**
- **No hay parámetros fuera del contrato:** los generadores recibieron `u ∈ [0,1]`. Se descarta la amenaza de imágenes extrapoladas.
- **Efecto residual, saturación en el borde:** muestras con algún componente en exactamente 0 o 1. En train son los valores extremos propios del ajuste.

| Split | Valores en 0 o 1 por componente (u0…u4) | Muestras con alguno | label 1 | label 0 |
| --- | --- | --- | --- | --- |
| train | 2, 2, 2, 2, 2 | 9/908 | 4/353 | 5/555 |
| val | 2, 2, 6, 5, 6 | 20/114 | 5/44 | 15/70 |
| test | 1, 1, 2, 3, 1 | 8/114 | 4/44 | 4/70 |

En val y test, las muestras saturadas pierden la información de cuánto se salían del rango de train. Son un poco más frecuentes en val que en test, y en val, más en label 0. Es una pérdida de información menor y documentable. **No requiere regenerar nada.** Si SV quiere usarla, entra en F4.2 (régimen) o en el análisis de errores (F4.6).

**`docs/artefactos_datos.md` decía lo contrario** ("val/test se salen ligeramente"). Ya se corrigió.

**Detalle menor que se mantiene:** por el `int()` (truncamiento), el extremo superior de los parámetros enteros solo se alcanza con u = 1.0 exacto. En train, `n_octaves` = 8 y `seed_radius` = 18 casi nunca ocurren.

---

## 8. ViT-B/16 en fine-tuning no es reproducible con la misma semilla (observado, 21 oct 2026)

**Observación:** la primera pasada de Gray-Scott (F3.2, 18 corridas, solo validación) se corrió dos veces en Kaggle con el mismo código, las mismas semillas y `set_seed` completo (incluido `cudnn.deterministic=True`). La segunda vez fue accidental, por un borrador desincronizado del notebook.

| Configuración | ¿Mismo F1-val en ambas corridas? |
| --- | --- |
| ResNet-18 y EfficientNet-B0, frozen y fine-tuning (12 corridas) | Sí, idénticos al 4.º decimal |
| ViT-B/16 frozen (3 corridas) | Sí |
| ViT-B/16 fine-tuning, seed 123 (detenido en el epoch 2) | Sí (0.7121) |
| ViT-B/16 fine-tuning, seed 7 | **No**: 0.7219 → 0.7137 |
| ViT-B/16 fine-tuning, seed 42 | **No**: 0.7143 (best epoch 15) → 0.7165 (best epoch 8) |

**Interpretación (probable, no verificada):** el backward de la atención de ViT usa kernels de GPU no deterministas (*scaled dot-product attention* eficiente/flash), que `cudnn.deterministic` no controla. Varias cosas encajan con esto:
- en frozen, el gradiente no pasa por la atención y el resultado se repite;
- las CNN no usan esos kernels;
- la diferencia crece con la duración del entrenamiento.

No se puede descartar del todo que la GPU asignada por Kaggle haya sido distinta entre corridas (queda registrada en `env.gpu` de cada JSON en `runs/`).

**Impacto:** para la selección no cambia nada. El finalista sugerido de Gray-Scott fue el mismo en ambas corridas, y el promedio sobre semillas absorbe esta variación. Para el **paquete de reproducibilidad (F7.5)** hay que declararlo: los resultados de ViT fine-tuning se reproducen en distribución, no bit a bit.

**Si se necesita reproducibilidad exacta para ViT** (decisión de SV): forzar el kernel matemático de atención con `torch.backends.cuda.enable_flash_sdp(False)` y `torch.backends.cuda.enable_mem_efficient_sdp(False)`, o activar `torch.use_deterministic_algorithms(True)`. Cuesta más tiempo por corrida y no se ha probado en este proyecto.
