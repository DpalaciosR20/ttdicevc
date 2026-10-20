# TT2 — Plan Detallado de Tareas (por semana y rol)

> ⚠️ **Reemplazado por `plan/plan_tt2_v2.md` (19 oct 2026).** Este v1 se conserva como historial. Donde difieran, manda el v2: primera pasada con 3 semillas y solo validación, test bloqueado hasta congelar el protocolo, resultado de WITCOM como exploratorio.
Diseñado para que cada integrante pueda trabajar de forma independiente, en su propia conversación con Claude, sin depender del historial de los demás.

**Cómo usar este documento:** al iniciar una tarea nueva, copia el bloque de "Contexto General" (abajo) al inicio de tu conversación con Claude, y después pega la tarea específica de tu semana/rol. Eso le da a Claude suficiente información para ayudarte sin que tengas que re-explicar todo el proyecto.

---

## 📋 CONTEXTO GENERAL (pegar esto al inicio de cualquier conversación nueva)

```
Estoy trabajando en TTDICEVC (Transformación de Textos Depresivos en Imágenes
mediante Visión Computacional), un Trabajo Terminal de ESCOM-IPN. Somos 3
integrantes: Diego Palacios Robles, Marco Antonio Mánica Quintero, y Santiago
Elihu Soria Solís, dirigidos por el Dr. Abdiel Reyes Vera.

CONTEXTO DEL PROYECTO:
El proyecto transforma embeddings de texto (depresivo/no depresivo, corpus
SPADE-BERT en español, 1,136 muestras, split 80/10/10 = 908/114/114) en
imágenes abstractas deterministas mediante 3 técnicas generativas (Perlin
Flow Fields, Atractor de Lorenz, Reacción-Difusión Gray-Scott), con el
objetivo de clasificar esas imágenes mediante visión computacional y comparar
contra un clasificador de texto (BETO fine-tuneado, F1-Macro=0.7586).

PIPELINE YA CONSTRUIDO (Módulos 1 y 2, completos):
- Módulo 1: BETO fine-tuneado + HPO con Optuna → embeddings de 768D.
- Módulo 2: UMAP reduce 768D → 5D (n_neighbors=15, min_dist=0.1, métrica
  coseno, seed=42, MinMaxScaler), y esos 5 valores se mapean de forma
  determinista a los parámetros de cada técnica generativa.

ARTEFACTOS DE DATOS DISPONIBLES (todos ya generados y verificados):
- `umap_params.pkl`: reductor UMAP ajustado + scaler + los 5 componentes ya
  reducidos y normalizados para train/val/test (908/114/114), con labels.
- `Exp_4_1_BETO_HPO_embeddings.pkl`: embeddings BETO de 768D (X_train/val/test,
  908/114/114), útiles si se necesita reajustar UMAP.
- 3 datasets de imágenes CLASS-BLIND (escala de grises, 256x256, SIN fuga de
  etiqueta, ya verificados) para Gray-Scott, Perlin y Lorenz — 1,136 c/u.
  Convención de nombres: `<prefijo>_<split>_<idx:04d>_label<0o1>.png`
  (prefijos: gray-scott='rd', perlin='perlin', lorenz='lorenz').

HALLAZGOS METODOLÓGICOS IMPORTANTES YA RESUELTOS (no repetir el diagnóstico):
1. Las imágenes ORIGINALES estaban coloreadas por clase (fuga de etiqueta) —
   ya se regeneraron las 3 técnicas en verdadero class-blind.
2. El generador de Perlin tenía una semilla aleatoria que dependía del label
   (fuga real) — ya corregido (semilla determinista, hash del vector UMAP).
3. Un primer intento de k-fold combinando train+val+test dio resultados
   inflados (~0.82 vs ~0.72) por dos fugas: (a) UMAP ya había visto esos
   puntos en su ajuste original, y (b) BETO fue fine-tuneado específicamente
   sobre las 908 muestras del train original, así que cualquier re-partición
   que no respete esa frontera infla los resultados. NO se debe repetir k-fold
   simple como métrica de generalización confiable — usar el split fijo
   80/10/10 original como referencia principal, y documentar cualquier
   k-fold/holdout adicional como "estimado optimista" con esta limitación.
4. Para Gray-Scott ya se probó: ResNet-18 (fine-tuning completo, lr=3e-5,
   Adam, CrossEntropyLoss con pesos de clase, batch=32, early stopping
   paciencia=4 sobre F1-val) da F1-Macro=0.7142±0.0095 (5 semillas: 42, 7,
   123, 2024, 99) sobre el split fijo — no supera a clasificar directamente
   los 5 valores UMAP (LogReg/SVM=0.7246, RF=0.7304, MLP=0.6933). McNemar
   consistentemente no significativo.
5. La generación de imágenes en GPU (batched, PyTorch) ya está resuelta y
   probada — mucho más rápida que CPU/numpy puro (minutos en vez de horas).
6. Lección de Kaggle: `/kaggle/input/` es SOLO LECTURA, todo lo que se genere
   debe ir a `/kaggle/working/`.

ALCANCE DE TT2 (lo que falta, es lo que estamos trabajando ahora):
Clasificación visual con 3 arquitecturas (ResNet-18, EfficientNet-B0, ViT) x
2 estrategias (frozen, fine-tuning) x 3 técnicas generativas, + Grad-CAM +
comparación estadística formal + análisis comparativo entre técnicas.

Mi rol en este momento es: [CE / SV / RP — especificar]
Mi tarea específica de esta semana es (ver abajo).
```

---

## SEMANA 1 (3–9 sept) — Confirmación y generalización del pipeline

### 🔵 CE — Generalizar el código de entrenamiento a múltiples arquitecturas

**Objetivo:** tener una función de entrenamiento reutilizable que acepte la arquitectura como parámetro, en vez de tener código copiado 3 veces (uno por arquitectura).

**Punto de partida:** ya existe una función de entrenamiento para ResNet-18 (transfer learning, fine-tuning completo, con class weights, early stopping por F1-val, y soporte para semilla fija). Está probada y da resultados reproducibles sobre Gray-Scott.

**Qué hacer:**
1. Refactorizar la función de entrenamiento para que reciba un modelo ya instanciado (no crearlo adentro), de forma que puedas pasarle `models.resnet18(...)`, `models.efficientnet_b0(...)`, o un ViT (de `torchvision.models.vit_b_16` o la librería `timm`) indistintamente.
2. Agregar soporte para "frozen" (congelar todos los pesos salvo la última capa) vs. "fine-tune completo" como parámetro booleano — ya existe un ejemplo de esto en la búsqueda de hiperparámetros que se hizo para ResNet-18 (opción `freeze_backbone`).
3. Ojo con las diferencias entre arquitecturas: cada una tiene su propio tamaño de entrada esperado y su propia capa final a reemplazar (ResNet: `.fc`, EfficientNet: `.classifier[-1]`, ViT: `.heads.head` o similar según la librería — verifica la arquitectura exacta antes de asumir el nombre del atributo).
4. Probar que las 3 arquitecturas cargan, corren un forward pass, y entrenan sin error sobre un subconjunto pequeño (unas 20-30 imágenes) antes de comprometerse a correr el grid completo.

**Entregable:** un notebook o script con la función de entrenamiento generalizada, probada con las 3 arquitecturas en modo "smoke test" (pocas imágenes, pocas épocas, solo para confirmar que no truena).

---

### 🟢 SV — Diseño de la matriz de experimentos y plan de Grad-CAM

**Objetivo:** definir exactamente qué se va a correr (para que CE no tenga que decidir esto sobre la marcha) y cómo se va a interpretar Grad-CAM.

**Qué hacer:**
1. Confirmar la matriz de experimentos: 3 arquitecturas × 2 estrategias × 3 técnicas = 18 configuraciones. Decidir el orden de prioridad (sugerencia: empezar por Gray-Scott ya que es la técnica con la señal más fuerte y ya validada, así se detectan problemas de código temprano).
2. Definir el criterio de selección del "ganador" por técnica (sugerencia: mejor F1 de validación, no de test — así se hizo para Gray-Scott+ResNet-18, para evitar contaminar la decisión con el propio test set).
3. Investigar qué capa de cada arquitectura es la adecuada para Grad-CAM (típicamente la última capa convolucional antes del pooling global — en ResNet es `layer4`, en EfficientNet es la última capa de `features`, en ViT el mecanismo es distinto, normalmente se usa "attention rollout" en vez de Grad-CAM clásico — investigar la diferencia antes de prometer Grad-CAM para ViT específicamente).
4. Preparar una plantilla de tabla para ir llenando resultados conforme lleguen de CE (arquitectura, estrategia, técnica, F1-val, F1-test).

**Entregable:** documento breve con la matriz de experimentos confirmada, el criterio de selección, y la investigación de qué método de interpretabilidad usar por arquitectura.

---

### 🟠 RP — Esqueleto del documento de TT2

**Objetivo:** tener la estructura completa del documento antes de empezar a llenarlo, para no reescribir la organización a medio camino.

**Qué hacer:**
1. Revisar el formato/lineamientos de titulación de ESCOM-IPN para Trabajo Terminal (si difiere del formato de paper que se usó para WITCOM).
2. Armar el índice completo: probablemente Introducción, Marco Teórico, Estado del Arte, Metodología (Módulos 1, 2 y 3), Resultados, Discusión, Conclusiones y Trabajo Futuro, Anexos.
3. Los capítulos de Módulo 1 y 2 pueden reutilizar directamente el contenido ya escrito para el TT1 y el paper de WITCOM (con las correcciones ya identificadas: partición 80/10/10 no 70/15/15, n_iter=7000 no 9000 en Gray-Scott, aclaración de que BETO no usó class weights).
4. Dejar marcada la sección de Módulo 3 con placeholders claros de qué va en cada parte (arquitecturas evaluadas, Grad-CAM, comparación estadística, análisis entre técnicas) para que sea fácil ir llenándola conforme lleguen resultados de CE/SV.

**Entregable:** documento con el esqueleto completo (índice + capítulos 1-2 ya redactados/actualizados + placeholders claros para el Módulo 3).

---

## SEMANAS 2–3 (10–23 sept) — Grid de arquitecturas (primera pasada, 1 semilla)

### 🔵 CE — Correr las 18 configuraciones

**Objetivo:** obtener un F1-Macro (val y test) para cada una de las 18 combinaciones arquitectura×estrategia×técnica, con una sola semilla cada una (esto es una primera pasada exploratoria, no el resultado final).

**Punto de partida:** la función de entrenamiento generalizada de la Semana 1, y los 3 datasets class-blind ya generados.

**Qué hacer:**
1. Para cada técnica (Gray-Scott, Perlin, Lorenz), cargar el dataset ya generado (no hay que regenerar imágenes, ya existen).
2. Correr las 6 combinaciones arquitectura×estrategia por técnica (18 en total).
3. Usar una sola semilla fija (sugerencia: 42, para consistencia con lo ya hecho) — el multi-seed se hace después, solo para los ganadores.
4. Guardar resultados en una tabla/archivo conforme van saliendo (no esperar a que terminen las 18 para empezar a registrar) — recomendable llevar un log similar a la bitácora que se usó para el camera-ready, documentando cada corrida.
5. Si alguna arquitectura da error o resultados degenerados (ej. F1 cercano a 0 o a 1 de forma sospechosa), documentarlo y no descartar silenciosamente — puede ser una pista de un bug real (como pasó con el bug del prefijo `rd_` durante el camera-ready).

**Entregable:** tabla de 18 resultados (arquitectura, estrategia, técnica, F1-val, F1-test).

---

### 🟢 SV — Consolidar resultados conforme llegan

**Objetivo:** mantener la tabla comparativa actualizada y empezar el análisis estadístico donde ya haya suficientes datos.

**Qué hacer:**
1. Recibir los resultados de CE y consolidarlos en la tabla acordada en la Semana 1.
2. Para las técnicas que ya tengan sus 6 configuraciones completas, calcular cuál es la mejor por criterio de F1-val (no test).
3. Empezar a preparar el código de Holm/effect-size que se va a aplicar más adelante a las comparaciones finales (reutilizar el que ya se usó para la Tabla 11 class-blind, adaptado a esta nueva matriz).

**Entregable:** tabla consolidada y actualizada de las 18 configuraciones, con el ganador marcado por técnica en cuanto estén completas.

---

### 🟠 RP — Capítulo de metodología del TT2

**Objetivo:** avanzar en la redacción sin depender de que los experimentos terminen.

**Qué hacer:**
1. Redactar la metodología del Módulo 3: descripción de las 3 arquitecturas, las 2 estrategias de transfer learning, el diseño experimental (18 configuraciones), y la justificación de por qué se eligieron estas arquitecturas (mencionar que el paper original ya las proponía).
2. Redactar la sección de Grad-CAM (qué es, por qué se usa, qué se espera encontrar) — puede escribirse sin tener los resultados todavía.
3. Incluir la limitación de la fuga de BETO ya documentada en la sección correspondiente (metodología o limitaciones, según la estructura del documento).

**Entregable:** capítulo de metodología del Módulo 3, borrador completo (sin resultados, esos se agregan después).

---

## SEMANA 4 (24–30 sept) — Multi-seed de los ganadores + Grad-CAM

### 🔵 CE — Multi-seed de los 3 ganadores + generar Grad-CAM

**Objetivo:** obtener el número final y robusto (media ± std, 5 semillas) para el mejor modelo de cada técnica, y generar las visualizaciones de Grad-CAM.

**Qué hacer:**
1. Tomar la mejor combinación arquitectura+estrategia por técnica (definida en la Semana 3 por SV).
2. Correr 5 semillas (42, 7, 123, 2024, 99 — mismas que se usaron para Gray-Scott+ResNet-18, para consistencia) de cada una de las 3 configuraciones ganadoras.
3. Implementar Grad-CAM (o el método de interpretabilidad correspondiente, según lo que investigó SV en la Semana 1) sobre el mejor modelo de cada técnica — generar mapas de calor para al menos 4-6 muestras de cada clase.

**Entregable:** tabla final (media±std, 5 semillas) de los 3 modelos ganadores, y un conjunto de imágenes Grad-CAM organizadas por técnica y clase.

---

### 🟢 SV — Interpretar Grad-CAM y conectar con la Tabla 11

**Objetivo:** dar sentido a los mapas de calor, no solo generarlos.

**Qué hacer:**
1. Revisar los mapas de Grad-CAM generados por CE — ¿las zonas de mayor atención del modelo coinciden con las zonas de mayor variación estructural de la imagen (bordes, densidad)?
2. Conectar esto explícitamente con los hallazgos de la Tabla 11 class-blind (ya documentados: bordes en Perlin tiene el efecto más grande del estudio, d=0.95; LBP y entropía son fuertes en Gray-Scott). Si Grad-CAM se enciende en las regiones de alta densidad de bordes en Perlin, eso es evidencia convergente fuerte — vale la pena buscarlo explícitamente.
3. Documentar tanto los casos donde Grad-CAM "tiene sentido" como los que no — igual que con la Tabla 11, la honestidad sobre lo que no encaja es más valiosa que forzar una narrativa.

**Entregable:** análisis escrito de los mapas de Grad-CAM, con al menos 2-3 ejemplos visuales comentados por técnica.

---

### 🟠 RP — Capítulo de resultados de Módulos 1 y 2

**Objetivo:** avanzar en la redacción con lo que ya existía del TT1, sin depender de que Semana 4 termine sus experimentos.

**Qué hacer:**
1. Integrar/actualizar el contenido de resultados del Módulo 1 (BETO, F1=0.7586) y Módulo 2 (generación de imágenes) que ya existe en el TT1 — aplicar las correcciones ya identificadas (partición, n_iter, class weights de BETO).
2. Dejar la sección de resultados del Módulo 3 con la estructura lista para recibir las tablas de CE/SV en la Semana 5.

**Entregable:** capítulo de resultados de Módulos 1 y 2, completo y corregido.

---

## SEMANA 5 (1–7 oct) — Comparación estadística y análisis entre técnicas

### 🟢 SV — McNemar extendido + análisis comparativo entre técnicas

**Objetivo:** cerrar el análisis estadístico formal del Módulo 3.

**Qué hacer:**
1. Correr McNemar de cada uno de los 3 modelos ganadores contra BETO y contra su baseline 5D-UMAP correspondiente (ya existe el código de McNemar de Gray-Scott, solo hay que aplicarlo a Perlin y Lorenz con sus propios datos).
2. Escribir el análisis comparativo entre las 3 técnicas: ¿cuál preserva mejor la información discriminativa? ¿Coincide con lo que ya sugería la Tabla 11 class-blind (Gray-Scott con las métricas más robustas, Perlin con el efecto de bordes más grande)?
3. Aplicar Holm/effect-size a esta ronda final de comparaciones.

**Entregable:** tabla final de comparaciones estadísticas (las 3 técnicas vs. BETO vs. baselines UMAP) y el análisis comparativo escrito.

---

### 🔵 CE — Apoyo a corridas adicionales

**Objetivo:** estar disponible para cualquier corrida extra que el análisis de SV requiera (ej. si se necesita un control de permutación adicional, o volver a correr algo con más detalle).

**Qué hacer:** según lo que pida SV esa semana — no hay tarea fija predefinida, es semana de cierre experimental.

---

### 🟠 RP — Capítulo de resultados del Módulo 3

**Objetivo:** integrar todo lo generado por CE y SV en el documento.

**Qué hacer:**
1. Integrar la tabla de las 18 configuraciones (primera pasada) y la de los 3 ganadores multi-seed.
2. Integrar los mapas de Grad-CAM con su interpretación.
3. Integrar la comparación estadística final y el análisis entre técnicas.

**Entregable:** capítulo de resultados del Módulo 3, completo.

---

## SEMANA 6 (8–14 oct) — Redacción intensiva (discusión, conclusiones, limitaciones)

### 🟠 RP (con apoyo de CE y SV para revisar exactitud técnica)

**Qué hacer:**
1. Redactar la discusión: interpretar en conjunto los resultados de las 3 técnicas, conectar con la hipótesis central del proyecto (¿la información emocional sobrevive la transformación de modalidad?).
2. Redactar limitaciones: incluir explícitamente la fuga de BETO (ya documentada, no se resolvió del todo por costo computacional), la falta de HPO exhaustivo para los modelos de visión, y cualquier limitación nueva que haya surgido durante el grid de arquitecturas.
3. Redactar conclusiones y trabajo futuro.

**Entregable:** documento de TT2 con todos los capítulos en borrador completo.

---

## SEMANA 7 (15–21 oct) — Revisión y ajustes

### Todos

**Qué hacer:**
1. Lectura completa del documento — verificar coherencia entre capítulos.
2. **Revisar que las cifras coincidan en todo el documento** (partición, F1s, tablas) — ya tuvieron un problema de este tipo (70/15/15 vs 80/10/10) durante el camera-ready, vale la pena una pasada específica solo para esto.
3. RP: aplicar formato final según lineamientos de titulación.

---

## SEMANA 8 (22–28 oct) — Buffer y preparación de defensa

### Todos

**Qué hacer:**
1. Ensayo de la presentación de defensa.
2. Preparar respuestas a preguntas esperadas — sugerencia: las 2 fugas metodológicas que encontraron y corrigieron (color por clase, semilla de Perlin) son un punto fuerte del trabajo, no algo a minimizar — practicar explicarlas con confianza.
3. Entrega formal con margen antes de la ventana de defensa (17 nov – 1 dic).
