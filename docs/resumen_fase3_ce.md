# Fase 3, parte de CE (F3.4–F3.7): cerrada

Sant, Marco: ya terminé mi parte de la Fase 3. Cada tarea tiene su bitácora en `docs/bitacora/F3.x.md`, con hallazgos, limitaciones y un §9 con texto para el reporte. Los PRs #31 (verificación de los modelos de F3.6) y #32 (F3.7) están por fusionarse.

## F3.4. HPO de los finalistas

- Gray-Scott (EfficientNet-B0 FT) y Lorenz (ViT-B/16 congelado) conservan los hiperparámetros de la primera pasada: la mejora no superó 1 DE.
- Perlin (ViT-B/16 congelado) adopta el trial 10 (lr 8.6e-3, wd 0.07, lote 16), que es inestable (H2).
- Hay una desviación de procedencia: las corridas salieron de un commit que no desciende de `seleccion-f33`, aunque el código de entrenamiento es idéntico (H5).

## F3.5. Entrenamiento final con prueba

Cinco semillas; la prueba se consultó una sola vez. Antes de consultarla quedó registrada la Enmienda 3 y se confirmó la revisión de las plantillas de F6.1.

| Finalista | F1-val | F1-prueba |
|---|---|---|
| Gray-Scott · EfficientNet-B0 FT | 0.7170 ± 0.0093 | 0.7250 ± 0.0180 |
| Perlin · ViT-B/16 congelado | 0.7142 ± 0.0168 | 0.6719 ± 0.0151 |
| Lorenz · ViT-B/16 congelado | 0.7141 ± 0.0035 | 0.7307 ± 0.0146 |

- Las semillas 42, 7 y 123 reproducen bit a bit lo que se usó para seleccionar.
- Perlin cae de val a prueba en 5/5 semillas (H2, coherente con la inestabilidad de su HPO).
- Una corrida de Gray-Scott llegó al tope de 15 épocas (H3).
- Predicciones: `results/predictions/final/`, 15 CSV en formato F1.5.

## F3.6. Control de permutación

`y_train` permutado con semilla 99; val y prueba con etiquetas reales.

- F1-prueba medio: Gray-Scott 0.527 ± 0.119, Perlin 0.548 ± 0.063, Lorenz 0.533 ± 0.042.
- En los 15 pares de semilla, el modelo real supera a su control.
- **Ojo (H2):** la parada temprana sobre val real deja algunas corridas por encima del azar (Gray-Scott, semilla 2024: 0.7185). El piso «visual permutado» es un poco más exigente que el azar puro. Así está preregistrado y no se cambió.
- Predicciones: `results/predictions/control_permutacion/`.

## F3.7. Mapas de atribución

30 modelos × 228 muestras.

- Grad-CAM en Gray-Scott. En Perlin y Lorenz, el *rollout* ponderado por gradiente (Enmienda 3) más la atención del *backbone*.
- Cada modelo reprodujo sus predicciones archivadas antes de calcular los mapas.
- **Hallazgo clave (H2):** correlación media por muestra entre el mapa del control permutado y el del modelo real.

  | Técnica · método | r |
  |---|---|
  | Gray-Scott · Grad-CAM | 0.23 |
  | Perlin · *rollout* ponderado | 0.80 |
  | Lorenz · *rollout* ponderado | 0.97 |

  En los ViT congelados el mapa refleja sobre todo el *backbone* fijo, así que previsiblemente no pasarán la cordura por aleatorización de datos.
- El 11.5 % de los mapas de Grad-CAM de Gray-Scott son constantes (H5).
- La atención del *backbone* es idéntica entre semillas y controles, como se esperaba.

## Dónde está cada cosa

- Datasets privados de Kaggle:
  - `palaciosroblesdiego/tt2-artefactos`: modelos de F3.5;
  - `palaciosroblesdiego/tt2-artefactos-f36`: modelos de F3.6;
  - `palaciosroblesdiego/tt2-mapas`: mapas `.npz`.
- Sus huellas están en `meta/artefactos_manifest.csv`. Para comprobarlas: `python -m src.verify_artefacts --root <carpeta>`.
- En `tt2-mapas` se colaron 20 `.npz` de una prueba de humo en `f37_humo/`. Las herramientas del repo los descartan; si leen los mapas a mano, usen `ttdicevc/results/f37_mapas/maps/`.
- Notebooks de figuras: `notebooks/F3.5_final.ipynb`, `notebooks/F3.6_permutacion.ipynb` y `notebooks/F3.7_mapas.ipynb`.

## Para Sant (F4/F5)

- Ya puedes arrancar F5 con `results/predictions/final/` y `results/predictions/control_permutacion/`.
- En F4.3 hay que decidir dos cosas:
  - cómo reportar la atribución de los ViT congelados (F3.7 H2);
  - qué hacer con los mapas constantes de Grad-CAM (F3.7 H5).
- Avísame cuando quieras correr los segundos mejores de F5.5 (fase `sensibilidad`).

## Para Marco (documento)

- Limitaciones a registrar:
  - la caída de Perlin de val a prueba (F3.5 H2);
  - el piso permutado no equivale a azar puro (F3.6 H2);
  - la reserva sobre los mapas de los ViT (F3.7 H2).
- El texto base está en el §9 de cada bitácora.
- La referencia textual sigue siendo BETO 0.7039 en prueba. El contraste formal lo hace Sant en F5.

## Pendiente de mi lado

- Copia en Drive de los tres datasets.
