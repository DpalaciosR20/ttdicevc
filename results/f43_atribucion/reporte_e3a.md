# Reporte E3a: F4.3 (atribución espacial y espectral)

> Bloque E, etapa 3a · Rol: SV · Rama `sv/f4-atribucion`, creada desde `main` (`b0bd07e`)
> Especificación: `docs/plan_fase4.md`, F4.3 · Celdas: `docs/correr_f43_kaggle.md`
> Estado: **DETENIDO antes de Kaggle y de la parte local.**
> - No se usaron datos reales: ni mapas, ni modelos, ni imágenes, ni **u**.
> - No se tocó `results/f5_contraste/`.
> - No hubo inferencia de finalistas ni de controles.

## 0. Compuerta

| Comprobación | Resultado |
|---|---|
| `enmienda-5` es ancestro de la rama | sí (`git merge-base --is-ancestor enmienda-5 HEAD`) |
| `docs/plan_fase4.md` sin cambios frente a `enmienda-5` | sí (`git diff --quiet enmienda-5 HEAD -- docs/plan_fase4.md`) |

Los scripts vuelven a aplicar la compuerta al correr:
- `atribucion_local.py` usa la de `regimen.check_gate`: `enmienda-5`, el plan sin cambios, `results/f5_contraste/resultados.json` versionado (G1) y el árbol limpio.
- `atribucion_gpu.py` exige `enmienda-5`, el plan sin cambios y el archivo `--resoluciones` versionado y sin cambios. No exige el árbol limpio, porque la celda de Kaggle escribe `entorno/` antes de llamar al script.

## 1. Código

Commit `57095a4`.

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `src/fase4/atribucion_comun.py` | — | Funciones puras, sin torch ni E/S: similitud, criterio relativo, agregación, constantes, perfil radial, terciles, permutación + Holm, cosenos, lectura de los ViT y contrato de `--resoluciones` |
| `src/fase4/atribucion_local.py` | local (CPU) | Agregación, constantes, perfiles, terciles, permutación + Holm, cordura por datos, cosenos de cabezas, Gray-Scott real frente a permutado, lectura de los ViT |
| `src/fase4/atribucion_gpu.py` | Kaggle (GPU), solo val | Cascada por bloques y ablación espectral |
| `tests/test_fase4_atribucion.py` | local | 38 pruebas sintéticas |
| `docs/correr_f43_kaggle.md` | — | Celdas de Kaggle, contenido esperado, duración y verificación local |

### 1a. `atribucion_local.py`

- **Insumos.** Se leen de la copia de Drive (`--root`) y se ubican por nombre. Si hay copias, se elige la que termina en el `rel_path` del manifiesto, con las funciones de `verify_artefacts`.
  - **Mapas:** los 30 `.npz` de F3.7, siempre de `ttdicevc/results/f37_mapas/maps/`. Una ruta con `f37_humo` detiene el análisis.
    - Bytes y sha256 se comparan con `meta/artefactos_manifest.csv` (`MAPS[...]`) y con `results/f37_mapas/records/<corrida>.json`.
    - Se comprueba el orden de `sample_id` y que `y_true` coincida con `meta/sample_index.csv`.
  - **Modelos:** los 20 `.pt` de ViT (F3.5 y F3.6), solo para leer `heads.head.weight`. Su sha256 se compara con el manifiesto y con `model_file.sha256` del JSON de la corrida.
  - **u:** sale de `umap_params.pkl` (`regimen.load_u`, verificado contra el manifiesto). Solo se usa para los terciles de u4 de Lorenz, con los cortes calculados en train.
  - **Ubicación en este equipo:** la copia de Drive está en `G:\.shortcut-targets-by-id\14OfuU8hf_FZ4OstB3TJEVfzZQnNR_qz0\tt2-artefactos`, a la que apunta el acceso directo `G:\Mi unidad\tt2-artefactos.lnk`. Contiene `tt2-artefactos/`, `tt2-artefactos-f36/` y `tt2-mapas/`, y su `drive_verificacion.json` (2026-11-06) dice 60/60.
- **Mapa analizado.** Es `map_raw · map_raw_scale`, sin normalizar.
  - Por muestra, se promedia sobre las 5 semillas, con la escala que fije la resolución `escala_entre_semillas`.
  - Se agrega por clase real (principal) y por clase predicha (secundaria).
  - Split principal: val. Prueba se analiza como réplica, con los mapas archivados (G1).
- **Mapas constantes.** Un mapa es constante si su máximo es igual a su mínimo en 224 × 224.
  - Se reporta la fracción por clase real y por clase predicha, por semilla y agrupada.
  - Sensibilidad: medias de los mapas normalizados sin los constantes. Una clase con menos de 10 mapas no constantes no se reporta; se da el conteo.
- **Gray-Scott, perfil radial.** 28 anillos de 4 px desde el centro (111.5, 111.5): por clase real, por clase predicha y de la diferencia.
- **Lorenz, terciles de u4.** Cortes con `np.quantile` sobre train. Asignación: T1 si u4 ≤ c1; T2 si c1 < u4 ≤ c2; T3 si u4 > c2. Mapas medios por clase en cada tercil, en coordenadas de imagen.
- **Prueba de permutación** (solo val).
  - Estadístico: T = ‖media(y=1) − media(y=0)‖₂ sobre los 114 mapas promediados entre semillas.
  - 5,000 réplicas de `rng.permutation(y)` con `default_rng(2026)`.
  - k = número de réplicas con T ≥ T_obs, con tolerancia relativa de 1e-12.
  - Holm sobre m = 3 con `src.protocol.verdict.holm`; además se reporta el p ajustado.
- **Cordura por aleatorización de datos.** Para cada semilla, similitud por muestra entre el mapa real (F3.5) y el permutado (F3.6).
  - Medidas: Spearman a 56 × 56 (principal) y SSIM (secundaria).
  - Se comparan las medianas por semilla con el rango de las medianas de los 10 pares de semillas reales.
  - Se calcula en val (principal) y en prueba (réplica). En los ViT, también para la atención del *backbone*.
- **Cosenos de cabezas** (solo ViT, decisión 1 del plan). Para w1 y w1 − w0, sin sesgo:
  - real frente a permutada de la misma semilla (5 valores);
  - referencia: los 10 pares de cabezas reales;
  - nivel «alto» o «bajo».
- **Gray-Scott, mapas frente a `p_dep`.** Por semilla, la correlación de mapas real–permutado frente a la de `p_dep`. Se reportan todas las variantes y la principal la fija una resolución (P11, P12). Hay además la asociación entre semillas (Pearson y Spearman, n = 5).
- **Lectura de los ViT.** Combina la cordura por datos (Spearman, val), el nivel del coseno y, con `--cascada`, el paso «cabeza» de la cascada de GPU. Aplica la tabla de `docs/plan_fase4.md`.
- **Salidas** en `results/f43_atribucion/local/`:
  - `resultados.json`;
  - `medias_clase.npz`: mapas medios a 56 × 56;
  - `similitudes_por_muestra.csv`.

  No se sobrescriben.

### 1b. `atribucion_gpu.py`

- **G2: solo val.**
  - `filas_val` toma solo las filas de val de `sample_index`.
  - `nombres_val` rechaza cualquier fila que no sea de val y cualquier nombre que no tenga `_val_`.
  - `leer_imagenes_val` lee solo esos archivos.
  - No hay opción de split.
- **Verificaciones.** El script se detiene si falla alguna:
  - sha256 de cada `.pt` contra el manifiesto y contra `model_file.sha256` (`caracteristicas.verified_model`);
  - sha256 de cada imagen de val contra `data_manifest.csv`;
  - `p_dep` sin aleatorizar contra `val_probs`, con tolerancia ≤ 1e-4;
  - con `--maps-root`, el mapa normalizado sin aleatorizar contra el `.npz` de F3.7, con tolerancia ≤ 1e-2;
  - en la ablación, `p_dep` sin filtrar contra `val_probs`.
- **Cascada.** Pasos según la arquitectura:
  - EfficientNet-B0: `classifier.1` → `features.8` … `features.0`;
  - ViT-B/16: `heads.head` (más `encoder.ln`, según P14) → `encoder_layer_11` … `encoder_layer_0` → `conv_proj` (más los tokens, según P15).

  Cómo se reinicializa:
  - Es acumulativa y se hace sobre una copia en CPU: el generador de números aleatorios no depende del dispositivo.
  - Cada paso llama a `reset_parameters()` en cada submódulo que lo tiene. `nn.MultiheadAttention` no tiene uno público (P16).
  - Semilla: `torch.manual_seed(2026)` (P17).
  - El paso se detiene si no cambia ningún tensor.

  En cada paso se calculan:
  - los mapas de F3.7, con el método de `configs/matriz_experimentos.json`;
  - la similitud por muestra con el mapa sin aleatorizar: Spearman a 56 × 56, SSIM a 56 y SSIM a 224;
  - en los ViT, lo mismo para la atención del *backbone*.

  Criterio relativo: el rango de los 10 pares de semillas reales, sin aleatorizar.
- **Ablación.** Las imágenes de 256 px pasan por estos filtros:
  - paso-bajo: `scipy.ndimage.gaussian_filter`, σ ∈ {1, 2, 4, 8}, `truncate = 4.0`;
  - paso-alto: imagen − paso-bajo + media de la imagen.

  Después se convierten a uint8 (P19) y pasan por el mismo `get_transforms` de F1.4 que el JSON de la corrida. ŷ = 1[p_dep > 0.5]. Por condición se reportan el F1-val de cada semilla, el ΔF1 por semilla y el ΔF1 medio de las 5. En el paso-alto se registra también la fracción de píxeles recortados.
- **Prueba de humo.** `--limit N` escribe `smoke: true` y `estimacion_114`: el cómputo escalado a 114 muestras más el tiempo de carga.
- **Salidas** en `results/f43_atribucion/gpu/`: `cascada.json`, `cascada_similitudes.npz`, `cascada_medias.npz`, `ablacion.json`, `ablacion.csv` y `atribucion_gpu_meta.json`. No se sobrescriben.

## 2. Pruebas sintéticas

`python -m pytest tests/test_fase4_atribucion.py`: **38 passed**. No usan datos reales ni `TT_DATA_ROOT`, salvo el `TT_DATA_ROOT` temporal con imágenes sintéticas de la prueba de punta a punta.

| Grupo | Qué comprueban |
|---|---|
| Resoluciones | Las 19 claves. Se detiene si falta una clave, si un valor no está previsto o si hay una clave extra. Las claves `_*` son metadatos |
| Similitud | `pool` y mapas constantes. Spearman contra `scipy.stats.spearmanr`, por fila; da NaN con un mapa constante. SSIM contra `skimage.metrics.structural_similarity` (diferencia ≤ 1e-10). Mediana con `excluir` y con `cero` |
| Criterio relativo | 10 pares de semillas. El mínimo del rango cuenta como «dentro»; por encima del máximo, según P7. Criterios de mediana, mayoría y unanimidad. Caso sintético: un mapa distinto da «cambia» y uno idéntico da «no cambia» (o «indeterminado», según P7) |
| Agregación | La media por clase real es la misma promediando por muestra que agrupando. Clase predicha con las dos opciones de P1. Escala por semilla |
| Constantes | Fracción por clase, por semilla y agrupada. Un umbral de menos de 10 da `None`, agrupado y por semilla |
| Perfil radial | 28 anillos, simétricos. Con r como mapa, el valor del anillo i queda en [4i, 4i + 4). Los píxeles con r ≥ 112 quedan fuera |
| Terciles | Cortes de train y asignación en los cortes |
| Permutación | T contra el cálculo manual. Con efecto fuerte, k = 0 y p = 1/(B + 1). Sin efecto, p > 0.05. Es reproducible con la misma semilla. La regla `k_sobre_B`. La identidad cuenta como T ≥ T_obs. Generadores por técnica o secuenciales |
| Holm | p ajustados a mano, y rechazo coherente con p_Holm ≤ α |
| Cosenos y lectura | Cabeza igual da «alto» y cabeza aleatoria da «bajo». w1 − w0 y forma (2, d). `nivel_coseno`. Las 7 combinaciones de la tabla de lectura de los ViT |
| Local | `Ubicador`: elige la copia del manifiesto, rechaza `f37_humo` y se detiene con un sha256 distinto. `cargar_npz`: sha256 del registro y orden. `analizar` de punta a punta con datos sintéticos de las tres técnicas: Holm m = 3, p, perfil, constantes, terciles, lectura y salida JSON. No sobrescribe |
| G2 | `exigir_val`, `filas_val` y `nombres_val` rechazan prueba. Un espía sobre `open` comprueba que `leer_imagenes_val` solo abre imágenes de val aunque haya de prueba en la carpeta. `--split` no existe |
| Cascada | Pasos de EfficientNet-B0 y ViT con cada opción de P14 y P15, y nombres existentes. `reiniciar` es determinista con la semilla, solo cambia los tensores del paso y deja `in_proj` sin cambiar con `solo_reset_parameters`. Tokens. ViT pequeño: al aleatorizar la cabeza cambia el rollout ponderado y no la atención del *backbone* (Enmienda 3), y es reproducible. EfficientNet-B0 con 5 semillas: 10 pasos, resumen y 10 pares |
| Ablación | 9 condiciones. El paso-bajo conserva la media y suaviza más con σ mayor; el paso-alto conserva la media. Conversión a uint8 con las dos opciones. Un clasificador de juguete basado en la varianza da ΔF1 < 0 con paso-bajo σ = 8 |
| Punta a punta | `atribucion_gpu.main` en CPU (Gray-Scott, EfficientNet-B0 sin pesos, 2 imágenes de val sintéticas): `smoke`, `p_dep`, orden de la cascada, 9 condiciones, `.npz` de similitudes; no sobrescribe |

Suite completa (`python -m pytest tests`): **1250 passed, 31 skipped** en 320 s (`results/f43_atribucion/pytest_e3a.log`). Los omitidos:
- 30 de `tests/test_label_independence.py`, porque no se definió `F03_UMAP_PARAMS`;
- 1 de `tests/test_canonical_numbers.py`, porque en esta corrida no se definió `CANON_UMAP_PARAMS`.

## 3. Celdas de Kaggle

Detalle en `docs/correr_f43_kaggle.md`.
- **Acelerador:** GPU T4. **Internet:** activado.
- **Inputs:** `tt2-insumos`, `tt2-artefactos` y `tt2-mapas`.

| Celda | Contenido |
|---|---|
| 1 | Clonar en el `COMMIT` que se indique y aplicar la compuerta `enmienda-5` |
| 2 | `TT_DATA_ROOT`, `pip install`, `build_manifest --verify`, `verify_artefacts` F3.5 (15/15) y F3.7 (30/30), y GPU |
| 3 | Prueba de humo: 8 muestras de val en `/tmp`, con `estimacion_114` |
| 4 | En la versión guardada: verificación, entorno, cascada y ablación, `commit.txt`, `f43_sha256sum.txt` y zip |

**Requisito:** `results/f43_atribucion/resoluciones.json` debe estar versionado en el commit clonado. La rama no se ha subido (*push*).

## 4. Duración estimada

**Parte GPU (Kaggle T4).** Base: tiempos de F3.7 (`records/*.json`) y medición local de las similitudes. Detalle en `docs/correr_f43_kaggle.md`.

| Parte | Estimado |
|---|---|
| Cascada ViT (Perlin y Lorenz, 10 modelos, 15 pasadas cada uno) | ≈ 32 min |
| Cascada EfficientNet-B0 (Gray-Scott, 5 modelos, 11 pasadas) | ≈ 4 min |
| Ablación (15 modelos × 9 condiciones) | ≈ 5 min |
| Referencias, carga y huellas | ≈ 3–5 min |
| **Total** | **≈ 45 min (35–70 min)** |

- Por pasada de mapas sobre 114 muestras: ViT ≈ 6.3 s y EfficientNet ≈ 0.85 s (12.6 y 1.7 s por 228 muestras en F3.7).
- Similitudes medidas en local, en CPU, sobre 114 pares: Spearman a 56 × 56, 1.13 s; SSIM a 224, 0.54 s; SSIM a 56, 0.07 s. En la CPU de Kaggle se suponen al doble.
- La celda 3 da la cifra real.

**Parte local (sin estimar).** Lee y hashea 7.7 GiB de la copia de Drive: 1.31 GiB de `.npz` y 6.39 GiB de `.pt` de ViT, según `meta/artefactos_manifest.csv`. Su duración depende de si Drive tiene los archivos en caché local. Antes de correrla hay que estimarla con una prueba acotada (`CLAUDE.md`), y si pasa de 20 min, llevarla a Kaggle CPU (PENDIENTE P20).

## 5. PENDIENTES (sin resolver)

Son las claves de `--resoluciones` (`ac.RESOLUCIONES`). Ningún script corre sin todas.

| # | Clave | Qué no fija el plan | Opciones | Nota |
|---|---|---|---|---|
| P1 | `clase_predicha` | Qué ŷ define la «clase predicha» | `yhat_media_5_semillas` (§7, regla común «una predicción por muestra»), `yhat_por_semilla` | Con la media, la clase predicha es la misma en las 5 semillas |
| P2 | `escala_entre_semillas` | Cómo promediar entre semillas mapas sin normalizar de magnitudes distintas | `sin_reescalar` (literal del plan), `dividir_media_semilla` (cada semilla entre la media de sus mapas de val) | Sin reescalar, una semilla con Grad-CAM más intenso pesa más |
| P3 | `umbral_constantes_unidad` | Dónde se cuentan los «10 mapas no constantes» | `agrupado` (5 semillas juntas), `por_semilla` | En Gray-Scott, F3.7 H5 da 3.5–21.1 % de mapas constantes por semilla |
| P4 | `l2_resolucion` | Resolución de la norma L2 de la prueba | `224`, `56` | El p cambia poco; el estadístico, sí de escala |
| P5 | `perm_p` | Fórmula del valor p | `k_mas_1_sobre_B_mas_1` (como F4.7, E1-5), `k_sobre_B` | |
| P6 | `perm_rng` | Generador por técnica o compartido | `por_tecnica` (`default_rng(2026)` nuevo en cada una, como F4.7, E1-6), `secuencial` | |
| P7 | `sobre_maximo` | Similitud o coseno **por encima** del máximo de la referencia | `como_dentro` («no cambia» / «alto»), `indeterminado` | El plan define «dentro» y «por debajo», no «por encima». Un mapa casi idéntico (atención del *backbone*) cae ahí |
| P8 | `criterio_semillas` | Cómo combinar las 5 semillas en un veredicto por técnica | `mediana_de_semillas`, `mayoria` (≥ 3 de 5), `unanimidad` | Se aplica igual a la cordura por datos, la cascada y los cosenos |
| P9 | `similitud_constantes` | Spearman con un mapa constante (NaN) | `excluir` (se reporta el conteo), `cero` | Afecta sobre todo a Grad-CAM de Gray-Scott y a los pasos de la cascada que lo anulan (ReLU) |
| P10 | `ssim_resolucion` | Resolución de la SSIM secundaria en la parte local | `56`, `224` | La GPU calcula las dos |
| P11 | `gs_corr_pdep` | Qué correlación de `p_dep` es «la de F3.6» | `f36_prueba_media_real` (la columna «Corr. con finalista»: prueba, `p_dep` permutado frente a la **media de las 5 semillas reales**), `misma_semilla_val`, `misma_semilla_prueba` | La decisión 3 habla de la misma semilla; la columna de F3.6 usa la media real. Se calculan las tres |
| P12 | `gs_corr_mapas` | Medida de la correlación de mapas | `spearman56_mediana` (la similitud principal), `pearson_media` (`corr_con_real` de F3.7) | El split sigue al de P11 |
| P13 | `coseno_lectura` | Qué coseno decide «alto / bajo» en la lectura | `w1_menos_w0`, `w1`, `ambos` (deben coincidir) | El plan pide los dos; la lectura necesita uno |
| P14 | `vit_ln_final` | El `encoder.ln` final del ViT en la cascada | `con_cabeza`, `paso_propio`, `sin_reinicializar` | El plan enumera cabeza → bloques → *embeddings* |
| P15 | `vit_embeddings` | Qué incluye el paso «*embeddings*» | `solo_conv_proj`, `conv_proj_y_tokens` (`class_token` a cero y `pos_embedding` ~ N(0, 0.02), la inicialización de torchvision) | Los tokens son `nn.Parameter` sin `reset_parameters()` |
| P16 | `mha_in_proj` | `nn.MultiheadAttention` no tiene `reset_parameters()` público | `reset_mha_completo` (`_reset_parameters()`; `out_proj` con el suyo), `solo_reset_parameters` (`in_proj` queda entrenado) | Con la segunda, la atención del bloque no se aleatoriza |
| P17 | `semilla_cascada` | Dónde se fija `torch.manual_seed(2026)` | `una_vez_por_modelo`, `en_cada_paso` | |
| P18 | `ablacion_borde` | Borde del filtro gaussiano | `reflect` (por omisión en scipy), `nearest` | |
| P19 | `ablacion_rango` | Paso-alto fuera de [0, 255] y conversión a uint8 antes de F1.4 | `recortar_redondear`, `reescalar_min_max` | Se registra la fracción de píxeles recortados |
| P20 | — (operativo) | Dónde corre la parte local | Local, si la prueba acotada estima ≤ 20 min; si no, Kaggle CPU con `tt2-artefactos`, `tt2-artefactos-f36` y `tt2-mapas` | 7.7 GiB desde Drive. No requiere código nuevo: `--root /kaggle/input` |

**Elecciones técnicas que no son PENDIENTES** (SV puede objetarlas):
- **Similitud.**
  - Spearman: promedio en bloques de 4 × 4 (224 → 56) y rangos promedio en empates.
  - SSIM: ventana gaussiana σ = 1.5, `truncate` = 3.5, K1 = 0.01, K2 = 0.03, sobre mapas normalizados a [0, 1]. Equivale a la de skimage.
- **Perfil radial.** Centro (111.5, 111.5), anillos [4i, 4i + 4); los píxeles con r ≥ 112 (esquinas) no entran.
- **Terciles.** `np.quantile` lineal; un empate en el corte va al tercil inferior.
- **Permutación.** Sobre las 114 etiquetas de val, con `rng.permutation(y)`. Tolerancia relativa de 1e-12 en T ≥ T_obs.
- **Cosenos.** Sin sesgo y solo en los ViT, por la decisión 1 del plan.
- **Cordura por datos.** Se calcula también para la atención del *backbone* (se espera 1, F3.7 H3) y en prueba, como réplica.
- **Escala.** Con `dividir_media_semilla`, el factor de cada semilla sale de val y se aplica también a prueba.
- **Cascada.**
  - Se reinicializa en CPU y se copia al dispositivo.
  - `BatchNorm.reset_parameters()` reinicia también las estadísticas acumuladas.
  - El método de atribución y la clase objetivo (1) son los de F3.7.
- **Ablación.**
  - `truncate` = 4.0 (por omisión en scipy).
  - La media que se suma en el paso-alto es la de la imagen original.
  - Se convierte a uint8 para usar sin cambios la ruta PIL → `get_transforms` de F1.4.
  - ŷ = 1[p_dep > 0.5].
- **Tolerancias.**
  - `p_dep`: 1e-4, la misma de F3.7.
  - Mapa frente al `.npz` de F3.7: 1e-2 en el mapa normalizado, porque el `.npz` está en float16 y la GPU no es determinista.
  - Ambas se registran.

## 6. Observaciones (sin resolver; no afectan el código de F4.3)

1. **Cabeza inicializada en cero en el ViT de prueba de F3.7.** torchvision inicializa en cero la cabeza de `VisionTransformer`. Con el ViT pequeño de `tests/test_attribution.py`, el rollout ponderado es entonces un mapa constante (gradiente nulo).
   - Por eso la prueba de la Enmienda 3 (`test_cordura_cabeza_aleatoria_vit_congelado`) compara un mapa constante con uno no constante.
   - En `tests/test_fase4_atribucion.py`, la cabeza del ViT pequeño se inicializa con N(0, 1) para que el mapa no sea trivial.
   - Los modelos reales no tienen este problema: tienen la cabeza entrenada.
2. **Correlación de `p_dep` de F3.6.** La columna «Corr. con finalista» de la bitácora F3.6 correlaciona el `p_dep` permutado de una semilla con la **media** de `p_dep` de las 5 semillas reales, en prueba (`notebooks/F3.6_permutacion.ipynb`). No usa la misma semilla. Ver P11.

## 7. Siguiente paso

1. SV resuelve P1–P20 en `results/f43_atribucion/resoluciones.json` y hace commit.
2. *Push* de `sv/f4-atribucion` y Kaggle: celdas 1–4 de `docs/correr_f43_kaggle.md`, con el commit que contenga las resoluciones.
3. Recepción del zip con la verificación local de la guía.
4. Prueba acotada de la parte local (P20). Después, `python -m src.fase4.atribucion_local --root … --resoluciones … --cascada results/f43_atribucion/gpu/cascada.json`, una sola vez y con el árbol limpio.

## Adenda (2026-11-10): resoluciones P1–P20

- SV fijó P1–P19 en `results/f43_atribucion/resoluciones.json` (`_autor` SV, `_fecha` 2026-11-10), antes de cualquier cálculo con datos reales. `l2_resolucion` y `ssim_resolucion` quedan como cadenas (`"56"`), que es lo que espera el contrato. P20: prueba acotada local; si estima más de 20 min, Kaggle CPU con `--root /kaggle/input`.
- **Nota de P9, implementada en la lectura de la cascada** (`atribucion_comun.resumen_cascada_spearman` y `atribucion_gpu.resumen_cascada`).
  - Una muestra cuyo mapa pasa de no constante (sin aleatorizar) a constante (aleatorizado), en la rejilla de 56 × 56, cuenta como cambio.
  - Su Spearman, indefinido, se imputa al mínimo posible (−1) antes de la mediana.
  - Por semilla y paso se reportan `n_transiciones_a_constante` y `n_base_constante`.
  - Los demás NaN (mapa sin aleatorizar ya constante) siguen `similitud_constantes` = `excluir`.
  - La regla solo aplica a la cascada; la cordura por datos de la parte local sigue la regla general.
  - `similitudes()` guarda además `const_base` y `const_nuevo` por muestra en `cascada_similitudes.npz`.
- **Pruebas:** 2 nuevas, 40 passed en `tests/test_fase4_atribucion.py`.
  - Un paso que anula 12 de 20 mapas da mediana −1 y 12 transiciones; con la regla general, la mediana habría sido 1 («no cambia»).
  - Con 5 semillas y mayoría, un paso que anula 7 de 10 mapas se lee como «cambia».
