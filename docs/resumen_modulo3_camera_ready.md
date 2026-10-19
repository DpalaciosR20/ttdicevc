# Módulo 3 — Resumen Consolidado de Resultados
**Para:** redacción de la sección del Módulo 3 (camera-ready WITCOM 2026)
**De:** Diego (rol CE), con apoyo de Claude
**Alcance:** versión mínima viable — 1 CNN (Gray-Scott) + controles de circularidad + validación class-blind en las 3 técnicas

---

## 1. Resumen ejecutivo

El Módulo 3 se completó con alcance mínimo viable, respondiendo directamente a las 3 observaciones centrales de los revisores:

1. **Se identificó y corrigió una fuga real de etiqueta** en el pipeline de generación de imágenes (color por clase, y en Perlin también una semilla aleatoria dependiente del label). Se regeneraron las 3 técnicas (Gray-Scott, Perlin, Lorenz) de forma genuinamente ciega a la etiqueta.
2. **Se entrenó un clasificador de visión (CNN, ResNet-18)** sobre el dataset class-blind de Gray-Scott y se comparó contra clasificadores directos sobre los 5 componentes UMAP (controles de circularidad).
3. **Resultado central, honesto:** el CNN (F1-Macro = 0.7142 ± 0.0095, 5 semillas) **no supera** a los clasificadores directos sobre los 5 valores UMAP crudos (0.69-0.73) — no hay evidencia de que la transformación a imagen añada poder discriminativo sobre la proyección UMAP que la genera. Confirmado con prueba de McNemar (no significativa) de forma consistente en múltiples corridas.
4. Se identificaron y documentan honestamente **dos fugas metodológicas adicionales** (UMAP no reajustado; fine-tuning de BETO) que invalidan un primer intento de validación cruzada — se reporta el split fijo original como resultado principal, y los números de k-fold/holdout como "estimado optimista" con la limitación explicada.

---

## 2. Metodología

### 2.1 Corrección de la fuga de etiqueta (class-blind rendering)

Se auditaron las imágenes originales del proyecto y se confirmó que estaban coloreadas por clase (PuBuGn para depresivo, YlOrRd para no depresivo) — el dataset "ciego a la clase" que la metodología original prometía nunca se había generado. Adicionalmente, se encontró que el generador de Perlin usaba una semilla aleatoria derivada de la etiqueta (`seed = label*10000 + i`), condicionando también la geometría (no solo el color) a la clase.

Se regeneraron las 3 técnicas desde cero, en escala de grises real (1 canal), verificando estructuralmente (por firma de función) que ningún renderizador recibe la etiqueta en ningún punto:

| Técnica | Estado del código original | Acción tomada |
|---|---|---|
| Gray-Scott | Ya era ciego a la etiqueta en su motor físico (solo el color final dependía de la clase) | Se usó el código original de Marco, con el color reemplazado por render directo de `V` en gris |
| Perlin | La semilla aleatoria dependía de la etiqueta (fuga real) | Corregido: semilla determinista derivada de un hash de los 5 valores UMAP |
| Lorenz | Ya era ciego a la etiqueta (el color ya estaba comentado en el código) | Solo se reemplazó el color por intensidad de gris derivada de z(t), preservando la información |

**Nota de transparencia para el texto:** el código original completo del Módulo 2 se había perdido parcialmente; el mapeo UMAP→parámetros usado para Gray-Scott es el código original recuperado (de Marco), Lorenz es el código original (de Santiago/Marco) con una corrección menor, y Perlin es el código original de Diego con la corrección de semilla. Ningún mapeo fue inventado desde cero salvo verificaciones puntuales.

**Discrepancia de parámetro a corregir en el texto:** el código real de Gray-Scott usa `n_iter` con tope de **7,000** (no 9,000 como dice actualmente el manuscrito, sección 6.2/11.4.3) — ajustar la cifra en el texto.

**Partición de datos confirmada:** 80/10/10 (908/114/114), verificado directamente desde los artefactos serializados. El documento TT-1 reporta 70/15/15 — es una discrepancia a corregir en ese documento (el paper y el código coinciden en 80/10/10, que es lo correcto).

### 2.2 Diseño experimental del CNN

- Arquitectura: **ResNet-18** preentrenada en ImageNet, fine-tuning completo (todos los pesos se actualizan), ~11.18M parámetros.
- Entrada: imagen 256×256 en gris, replicada a 3 canales (R=G=B, sin color real agregado), redimensionada a 224×224, normalización estándar de ImageNet.
- Entrenamiento: Adam, `lr=3e-5` (seleccionado tras una búsqueda comparativa — ver sección 4), `CrossEntropyLoss` con pesos de clase (dataset desbalanceado 61/39), batch=32, hasta 15 épocas con *early stopping* (paciencia=4 sobre F1-Macro de validación), augmentación con `RandomHorizontalFlip`.
- No se realizó búsqueda exhaustiva de hiperparámetros (HPO) tipo Optuna para el CNN, a diferencia de BETO en el Módulo 1 — se documenta como limitación.

---

## 3. Tabla 11 recalculada (class-blind, las 3 técnicas)

Se recalcularon las 4 métricas de bajo nivel (entropía de Shannon, contraste, densidad de bordes, energía LBP) sobre los datasets regenerados, con prueba de Welch y **corrección de Holm** para las 12 comparaciones. Se comparó cada resultado contra la Tabla 11 original (con color) del manuscrito, para distinguir señal genuina de artefacto de paleta:

| Técnica | Métrica | ¿Se mantiene sin color? | Cohen's d (class-blind) |
|---|---|---|---|
| Gray-Scott | Entropía | ✅ Se mantiene (incluso más fuerte) | −1.10 (grande) |
| Gray-Scott | Contraste | ⚠️ Se reduce sustancialmente (parcial artefacto de color) | −0.39 |
| Gray-Scott | Bordes | ✅ Se mantiene | −0.66 |
| Gray-Scott | LBP | ✅ Se mantiene, igual de fuerte | 1.32 (grande) |
| Perlin | Entropía | ❌ Era artefacto de color | 0.06 (nulo) |
| Perlin | Contraste | 🔄 Señal nueva (antes enmascarada por el color) | 0.26 (pequeño) |
| Perlin | Bordes | ✅ Se mantiene — el efecto MÁS GRANDE de todo el estudio | **0.95 (grande)** |
| Perlin | LBP | ❌ Era artefacto de color | −0.06 (nulo) |
| Lorenz | Entropía | ✅ Se mantiene | −0.42 (medio) |
| Lorenz | Contraste | — Consistente (nunca hubo señal) | −0.14 |
| Lorenz | Bordes | ❌ Era artefacto de color | −0.09 (nulo) |
| Lorenz | LBP | ✅ Se mantiene | 0.44 (medio) |

**Conclusión clave:** la separabilidad visual **no depende únicamente del color** — sobrevive de forma robusta en la mayoría de las métricas (Gray-Scott casi por completo; bordes en Perlin con el mayor tamaño de efecto del estudio). Al mismo tiempo, se identifican honestamente las métricas que sí eran artefacto de paleta. Este análisis autocrítico responde directamente a la pregunta explícita de Reviewer 1 sobre si la Tabla 11 se calculó sobre imágenes coloreadas convertidas a gris.

### 3.1 Test complementario: fuga por brillo (post class-blind)

Sobre las imágenes ya regeneradas (genuinamente ciegas a la etiqueta), se verificó si el brillo promedio por sí solo separa las clases:

| Técnica | F1-train (solo brillo) | F1-test (solo brillo) |
|---|---|---|
| Gray-Scott | 0.629 | 0.529 |
| Perlin | 0.536 | 0.453 |
| Lorenz | 0.607 | 0.474 |

El brillo promedio no es un discriminador robusto en ninguna técnica (F1-test cercano o por debajo de 0.5) — sirve como piso de referencia adicional para contextualizar el desempeño del CNN.

---

## 4. Selección de hiperparámetros del CNN

Antes de comprometer cómputo costoso a validación cruzada, se compararon 5 configuraciones sobre el split fijo (criterio de selección: mejor F1 de **validación**, no de test):

| Configuración | F1-val | F1-test |
|---|---|---|
| **lr=3e-5, fine-tuning completo** (ganador) | **0.7449** | 0.7512 |
| lr=1e-4, fine-tuning completo | 0.7223 | 0.7032 |
| lr=1e-4 + weight_decay=1e-4 | 0.7162 | 0.7264 |
| Frozen (solo capa final), lr=1e-3 | 0.7121 | 0.7564 |
| lr=1e-3, fine-tuning completo | 0.7078 | 0.7179 |

`lr=3e-5` se adoptó como configuración final.

---

## 5. Resultado central del CNN (split fijo 80/10/10)

Dada la variabilidad observada entre corridas individuales (0.70-0.75, atribuible a inicialización aleatoria y diferencias de hardware/cuDNN entre entornos), se promediaron **5 semillas distintas** (42, 7, 123, 2024, 99) con la configuración final:

| Seed | F1-test |
|---|---|
| 42 | 0.7093 |
| 7 | 0.7224 |
| 123 | 0.7246 |
| 2024 | 0.6984 |
| 99 | 0.7165 |

**CNN Gray-Scott: F1-Macro = 0.7142 ± 0.0095** (rango [0.6984, 0.7246]).

### Tabla comparativa final (resultado principal para el paper)

| Método | F1-Test |
|---|---|
| BETO (texto, línea base del Módulo 1) | 0.7586 |
| RF (5D-UMAP directo) | 0.7304 |
| LogReg / SVM (5D-UMAP directo) | 0.7246 |
| **CNN Gray-Scott (media de 5 semillas)** | **0.7142 ± 0.0095** |
| MLP (5D-UMAP directo) | 0.6933 |
| Brillo promedio de la imagen | 0.5290 |

**Prueba de McNemar** (CNN vs. LogReg-5D-UMAP, n=114, mismo test set): consistentemente **no significativa** (p=1.0) en múltiples corridas independientes.

### Conclusión (para la discusión del paper)

El CNN opera dentro de la misma banda de desempeño que los clasificadores que usan directamente los 5 componentes UMAP — no la supera de forma consistente ni estadísticamente significativa. **No hay evidencia de que la transformación de embedding a imagen añada poder discriminativo sobre la proyección de 5 dimensiones que la genera.** Esta es la respuesta directa, y honesta, a la pregunta de circularidad planteada por los 3 revisores.

---

## 6. Limitaciones documentadas (para la sección de limitaciones/discusión)

1. **Sin HPO exhaustivo para el CNN** — a diferencia de BETO (Módulo 1, con Optuna), la configuración del CNN se seleccionó por comparación manual de 5 alternativas, no por búsqueda sistemática.
2. **Variabilidad entre entornos de ejecución** — con semilla fija, corridas en distinto hardware (Colab, Kaggle, GPU local) no son bit-idénticas (cuDNN no determinista por defecto); se mitigó promediando 5 semillas.
3. **Intento de validación cruzada (k-fold) inflado por dos fuentes de fuga no supervisada, ambas identificadas y documentadas:**
   - *Fuga por UMAP no reajustado:* al recombinar las 1,136 muestras y re-particionar para k-fold, los puntos de "test" habían sido vistos por el ajuste original de UMAP. Corregido reajustando UMAP por fold (k-fold anidado).
   - *Fuga por fine-tuning de BETO:* aun con UMAP reajustado por fold, los resultados de k-fold/holdout se mantuvieron optimistas (~0.80-0.86, confirmado en dos implementaciones independientes) porque BETO fue afinado específicamente sobre las 908 muestras del split de train original, y el nuevo particionamiento dev/holdout no respeta esa frontera. Arreglarlo del todo requeriría reentrenar BETO (con su propia HPO) dentro de cada fold — no viable en el tiempo disponible.
   - **Por esto, el split fijo original (80/10/10) se reporta como el resultado principal y confiable; los números de k-fold anidado y holdout final se reportan aparte, explícitamente etiquetados como estimados optimistas**, con esta limitación documentada como trabajo futuro.
4. **Mapeo UMAP→parámetros de Gray-Scott usa `n_iter` con tope de 7,000**, no 9,000 como indica actualmente el texto del manuscrito — corregir la cifra.
5. **Solo se evaluó una arquitectura (ResNet-18) y una técnica de generación (Gray-Scott)** para el CNN, consistente con el alcance mínimo viable acordado; EfficientNet, ViT, y CNNs sobre Perlin/Lorenz quedan como trabajo futuro (Trabajo Terminal II).

---

## 7. Cifras de referencia rápida (para tablas/redacción)

- Partición: 908 train / 114 val / 114 test (80/10/10)
- Balance de clases: 695 no-depresivo / 441 depresivo (~61/39%) sobre el total de 1,136
- BETO (línea base textual, Módulo 1): F1-Macro = 0.7586
- Mejor baseline directo 5D-UMAP: RF = 0.7304
- CNN Gray-Scott (resultado final): F1-Macro = 0.7142 ± 0.0095
- Arquitectura CNN: ResNet-18, ~11.18M parámetros, fine-tuning completo
- McNemar CNN vs. LogReg-5D-UMAP: no significativo (p=1.0)
