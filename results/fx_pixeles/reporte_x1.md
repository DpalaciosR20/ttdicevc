# FX.1 — Reporte de la etapa X1 (código y pruebas sintéticas)

> Plan: `docs/plan_exploratorio_pixeles.md`. Etiqueta `plan-pixeles` sobre `d0eed6b` (merge del PR #45), plan con SHA-256 `2972e527e84b52d2380f53df494725964f1ac470111b9c9fb7dbded277e21caa`, igual en la etiqueta y en HEAD.
> Rama: `ce/fx-pixeles-x1`, desde `main` con `plan-pixeles` como ancestro.
> **Sin datos reales:** en esta etapa no se leyó ninguna imagen, mapa ni modelo del proyecto. Sin interpretación.

## 1. Archivos creados

| Archivo | Contenido |
|---|---|
| `src/fase_x/__init__.py` | Paquete |
| `src/fase_x/pixeles_comun.py` | Funciones puras de A, B, C1, C2 y D: `FIJOS` (valores del plan), `PENDIENTES`, `check_resoluciones`, `exigir_split`, compuerta y metadatos |
| `src/fase_x/pixeles_local.py` | B y D (CPU): carga verificada, `analizar_b`, `analizar_d`, familias H-B y H-D |
| `src/fase_x/pixeles_gpu.py` | C1 y C2 (GPU): reproducción de `p_dep`, generador de sorteos, oclusión, cordura, familia H-C, prueba acotada con estimación frente al límite de 3 h |
| `src/fase_x/manifiesto.py` | `manifiesto.csv` (bytes y SHA-256) de una carpeta de salida |
| `results/fx_pixeles/resoluciones.json` | Todos los valores del plan (`FIJOS`), con `_autor` «SV», `_fecha` «2026-11-10» y `_nota`; PENDIENTES en `null` con sus opciones en `_pendientes` |
| `tests/test_fx_pixeles.py` | 40 pruebas sintéticas |
| `docs/correr_fx_kaggle.md` | Celdas de X2 (Kaggle CPU) y X3 (Kaggle GPU T4), con prueba acotada |
| `results/fx_pixeles/pytest_x1.log` | `pytest` completo del repositorio |
| `results/fx_pixeles/reporte_x1.md` | Este reporte |

## 2. Funciones reutilizadas (sin reescribirlas)

| Necesidad | Función | Ruta |
|---|---|---|
| Parada ante una comprobación fallida | `Detener` | `src/fase4/atribucion_comun.py` |
| Mapa crudo `map_raw · map_raw_scale` | `mapa_crudo` | `src/fase4/atribucion_comun.py` |
| Cada semilla entre su media en val; media de 5 semillas | `escalar_semillas('dividir_media_semilla')`, `media_entre_semillas` | `src/fase4/atribucion_comun.py` |
| Promedio por parche; mapa constante | `pool`, `es_constante` | `src/fase4/atribucion_comun.py` |
| Spearman por muestra en rejilla y criterio relativo de cordura (C2) | `spearman_filas`, `resumen_similitud`, `referencia_mapas`, `criterio_similitud`, `pares_semillas` | `src/fase4/atribucion_comun.py` |
| Holm con p ajustado | `holm` (que llama a `src.protocol.verdict.holm`) | `src/fase4/atribucion_comun.py` |
| JSON sin NaN | `limpiar`, `meta_resoluciones` | `src/fase4/atribucion_comun.py` |
| Masa de atribución por parche | `masa_por_parche` | `src/fase4/convergencia.py` |
| Spearman por muestra (rangos promedio, NaN si constante) | `spearman_por_muestra` | `src/fase4/convergencia.py` |
| *Bootstrap* (matriz de índices por técnica, IC percentil `linear`) | `indices_bootstrap`, `ic_bootstrap` | `src/fase4/convergencia.py` |
| Imagen a 224 px con el `Resize` de F1.4 (B) | `redimensionar_224` | `src/fase4/convergencia.py` |
| Preprocesamiento de F1.4 por técnica | `preprocesamiento` | `src/fase4/convergencia.py` |
| Archivos del manifiesto, verificados (`.npz` y `.pt`), sin copias de humo | `Ubicador`, `logical_maps`, `logical_model`, `run_json`, `RECORDS` | `src/fase4/atribucion_local.py` |
| `sample_index` e imágenes verificadas contra `data_manifest` | `load_index`, `image_rows`, `read_verified_images`, `load_data_manifest`, `sha256_file` | `src/fase4/caracteristicas.py` |
| Decodificar PNG 256 × 256; recorte y redondeo a uint8; F1-macro; inferencia; tensores | `decodificar`, `a_uint8('recortar_redondear')`, `f1_macro`, `predecir`, `tensores` | `src/fase4/atribucion_gpu.py` |
| Arquitectura y determinismo | `build_model(pretrained=False)`, `configure_determinism`, `config.determinism` | `src/training/` |
| Transformación de inferencia de F1.4 | `get_transforms(False, …)` | `src/training/data.py` |
| Copias repetidas por ruta del manifiesto | `find_all`, `locate`, `sha256` | `src/verify_artefacts.py` |

**Lo nuevo,** porque no existe en el repositorio con esa forma:
- **A:** `grad`, `log` y `canny`.
- **B2:** Dice con empates estables y la nula diédrica con desplazamientos.
- **C1:** máscaras, generador de sorteos en el orden del plan y suavizado dirigido.
- **C2:** oclusión 8 × 8.
- **D:** t de Welch para muchas permutaciones (`TPermutaciones`), clústeres con conectividad 8 y prueba por clústeres.
- **Estadística:** Mann–Whitney con biserial y Wilcoxon.

`src/generation` y `src/selection` no tienen funciones aplicables a estas secciones.

## 3. Cobertura de las pruebas (`tests/test_fx_pixeles.py`, 40 pruebas)

| Pedido en el prompt | Prueba(s) |
|---|---|
| grad y log sobre imágenes con bordes conocidos | `test_grad_y_log_sobre_un_borde_conocido`, `test_grad_crece_con_el_contraste_y_sigma2_lo_ensancha`, `test_canny_solo_figuras` |
| Número de transformaciones de la nula de B2 (391 y 1,567), sin la identidad | `test_nula_b2_numero_de_transformaciones_sin_identidad` (además, todas distintas y cada una es una permutación); `test_dice_nulo_es_la_media_sobre_las_transformaciones` |
| Dice y empates con orden estable | `test_dice_y_empates_con_orden_estable`, `test_k_del_plan` |
| Máscaras de C1: área exacta k, \|Δx\| + \|Δy\| ≥ 32, reproducibilidad en el orden técnica → mapa → q → imagen → sorteo | `test_mascaras_c1_area_exacta_y_desplazamiento`, `test_generador_c1_reproducible_y_en_el_orden_del_plan` (con las dos opciones del PENDIENTE 2), `test_sorteos_cambian_con_la_tecnica_solo_por_el_generador_nuevo`, `test_imagenes_c1_orden_y_regiones` |
| Recorte y redondeo a uint8 | `test_recorte_y_redondeo_uint8` |
| Oclusión de C2 en una rejilla de 8 × 8 | `test_oclusion_8x8`, `test_cordura_c2_criterio_relativo` |
| Prueba por clústeres con una señal plantada | `test_prueba_por_clusteres_con_senal_plantada`, `test_clusteres_conectividad_8_y_signos`, `test_welch_permutaciones_igual_al_directo`, `test_analizar_d_sintetico` |
| Holm por familia con los m del plan | `test_holm_por_familia_con_m_del_plan` (H-B 4, H-C 12, H-D 6; falla con otro m) |
| Aborto ante `split == 'test'` | `test_aborta_ante_split_test` |
| Además | Resoluciones (claves faltantes o no previstas, valores distintos del plan, PENDIENTES en `null`); `analizar_b` sintético; de punta a punta de `pixeles_gpu` (incluye la detención si no reproduce `p_dep` y el rechazo de K ≠ 20/10) y de `pixeles_local` con modelos e imágenes falsos |

`pytest` completo del repositorio: ver `results/fx_pixeles/pytest_x1.log` (resultado en §6).

**Hallazgo en las pruebas.** En zonas planas, `scipy.ndimage.gaussian_laplace` (σ = 1, kernel truncado) no da exactamente 0: deja un residuo proporcional a la intensidad, de 1.1e-4 a intensidad 0.78. Es tres órdenes menor que la respuesta en un borde (> 0.1) y no cambia rangos dentro de una imagen. La prueba usa tolerancia 1e-3.

## 4. PENDIENTES (el plan no los fija y cambian resultados)

Están en `results/fx_pixeles/resoluciones.json` con valor `null`. Los scripts se detienen hasta que SV los resuelva. No se eligió ninguno. «Menos favorable» se refiere a la hipótesis de cada prueba; cuando la opción no favorece ni perjudica a la hipótesis, se dice.

| # | Clave | Qué no fija el plan | Opciones | Menos favorable |
|---|---|---|---|---|
| 1 | `sobel_borde` | §3 fija `mode='reflect'` para la gaussiana y el LoG, no para el Sobel | `reflect` · `nearest` | Ninguna: solo cambia la primera y la última fila y columna |
| 2 | `c1_orden_azar` | §5.C1.3: el azar va «a continuación de los desplazados», sin decir si tras los 20 de cada imagen o tras todos los de la técnica | `tras_los_desplazados_de_cada_imagen` · `tras_todos_los_desplazados_de_la_tecnica` | Ninguna: solo cambia qué sorteos concretos salen |
| 3 | `c1_muestreo_desplazamiento` | Cómo sortear (Δx, Δy) uniformes en [−128, 127] con \|Δx\| + \|Δy\| ≥ 32 | `rechazo_dx_luego_dy` (repetir hasta cumplir) · `uniforme_sobre_pares_validos` (un índice sobre los 63,551 pares válidos) | Ninguna: ambas son uniformes sobre el mismo conjunto |
| 4 | `c1_muestreo_azar` | Cómo elegir k píxeles sin reemplazo | `choice_sin_reemplazo` (`rng.choice(65536, k, replace=False)`) · `prefijo_de_permutacion` | Ninguna (ambas uniformes) |
| 5 | `b2_mapas` | §4: B1 usa `grad` y `log`; B2 habla solo de «pendiente» | `grad_y_log` · `solo_grad` | `grad_y_log`: más comparaciones descriptivas, sin familia |
| 6 | `wilcoxon_metodo` | §5.C1.5 fija `zero_method='wilcox'` y bilateral, no el cálculo de p (con n = 114, SciPy usa la normal) | `normal_sin_correccion` · `normal_con_correccion` | `normal_con_correccion`: p algo mayor, más conservador para H-C |
| 7 | `d_sensibilidad` | §6.5: «σ = 2 px y umbral \|t\| > 3.29»: dos variantes por separado o una combinada | `dos_variantes_sigma2_y_t329` (σ = 2 con 2.58 y σ = 1 con 3.29) · `una_variante_sigma2_con_t329` | Ninguna: descriptivo, fuera de H-D |

**Decisiones derivadas del plan,** documentadas en el código y no abiertas:
- **Criterio relativo de C2:** se toman de F4.3 `criterio_semillas = mayoria`, `sobre_maximo = como_dentro` y `similitud_constantes = excluir` (`results/f43_atribucion/resoluciones.json`). El plan (§5.C2.4) dice «el mismo criterio relativo de F4.3».
- **Biserial de rangos:** r = 2·U₁/(n₁·n₀) − 1, con U₁ el del grupo depresivo (r > 0: mayor en depresivos). El signo es convención y no cambia p.
- **D sin ningún clúster sobre el umbral:** el p de familia es 1.
- **t de Welch con varianza conjunta 0:** vale 0. No entra en ningún clúster, igual que si se excluyera.
- **Pendiente para B:** se calcula sobre la imagen de 224 px en uint8 tal como la entrega el `Resize` de F1.4 (`redimensionar_224`, igual que F4.4), dividida entre 255.
- **Sensibilidad σ = 2 de B1:** se calcula también para los mapas permutados (descriptivo).
- **Lorenz no entra en B** (regla 2.10).
- **K:** el script solo admite K = 20 o la reducción autorizada a 10 (`--k-sorteos 10`, solo con confirmación de SV).

**Punto sobre la regla 2.1 que conviene que SV confirme.**
- **El problema:** los `.npz` de F3.7 traen val y prueba en los mismos arreglos.
- **Lo que hace el código:** carga cada arreglo y selecciona de inmediato las filas de val con `exigir_arreglo_split(split, 'val')`. Ningún valor de prueba pasa a ningún cálculo ni salida.
- **La alternativa** sería abortar ante cualquier insumo que *contenga* filas de prueba, lo que haría imposible usar los mapas de F3.7.
- **La interpretación adoptada:** el aborto aplica a toda fila que llegue al análisis (filas de `sample_index`, nombres de imagen, arreglos filtrados). Las pruebas cubren el aborto en esas tres vías.
- Las imágenes de prueba nunca se abren.

## 5. Costo de cómputo (§5.C3)

**C1 y C2 (GPU), pasadas hacia adelante,** como en el plan: 114 × (2,520 + 1,920) ≈ 506,000.

| Técnica | Arquitectura | C1 (5 modelos) | C2 (10 modelos) |
|---|---|---|---|
| Gray-Scott | EfficientNet-B0 | 95,760 | 72,960 |
| Perlin | ViT-B/16 | 95,760 | 72,960 |
| Lorenz | ViT-B/16 | 95,760 | 72,960 |

- **Sin GPU disponible en X1,** la duración no se pudo medir aquí. Referencia de F3.7: un ViT-B/16 hizo 228 pasadas hacia adelante **y hacia atrás** en ≈ 15 s en la T4, carga incluida.
- **Supuesto:** solo inferencia en lotes de 64, ≈ 200–300 imágenes/s para ViT-B/16 y ≈ 1,000 para EfficientNet-B0.
- **Estimación:**
  - ViT: ≈ 337,000 pasadas, 19–28 min;
  - EfficientNet: ≈ 169,000 pasadas, ≈ 3 min;
  - preparación de imágenes en CPU (≈ 80,000 imágenes modificadas): ≈ 2–5 min;
  - **total ≈ 35–60 min, por debajo del límite de 3 h.**
- La celda 3 de `docs/correr_fx_kaggle.md` (8 imágenes) mide el costo real y el script lo compara con el límite. Si lo supera, se detiene la etapa y se avisa a SV.

**B y D (CPU), medido en el equipo local con datos sintéticos del tamaño real** (Windows, CPU; `pixeles_comun` y `analizar_b`):

| Parte | Medida | Tiempo |
|---|---|---|
| Mapas de pendiente | 13.9 ms por imagen de 256 px | ≈ 76 s para 908 × 3 técnicas × 2 σ |
| B (114 muestras) | Gray-Scott · Perlin | 2 s · 37 s (la nula de 1,567 transformaciones) |
| D, prueba por clústeres | 1,000 permutaciones por mapa | ≈ 46 s por mapa; 4 mapas de la familia + 8 de sensibilidad ≈ 9 min |

- **Con datos reales** hay que sumar la lectura de los 30 `.npz` de F3.7 (1.3 GiB) y de 3,408 imágenes.
- **Desde la copia de Drive** (≈ 3 MiB/s, como midió F4.3) la lectura sola tarda ≈ 7–10 min.
- **En el equipo local** `TT_DATA_ROOT` no está configurado y los mapas no están descargados. Con la regla de los 20 min, **B y D van a Kaggle con CPU** (celdas de X2), con la misma prueba acotada (`--limit 12`) para confirmar el tiempo.

## 6. `pytest`

`results/fx_pixeles/pytest_x1.log` (suite completa, `CANON_STRICT=1`, sin `smoke_test_training.py`, equipo local): **1,325 pasan, 35 omitidas, 3 fallan y 4 errores**. Las 40 de `tests/test_fx_pixeles.py` pasan.

Las 7 fallas son del entorno local y no tocan código de FX.1:

| Pruebas | Causa (del log) |
|---|---|
| `test_f24_descriptores.py` (4 errores) | `ImportError: DLL load failed while importing _seq_dataset: An Application Control policy has blocked this file` (scikit-learn, `linear_model`; política de control de aplicaciones de Windows) |
| `test_fase4_convergencia.py::test_render_determinista[perlin]`, `test_fase4_regimen.py::test_mapeos_coinciden_con_el_plan`, `test_fase4_regimen.py::test_estrato_lorenz_equivale_a_u1` | `ModuleNotFoundError: No module named 'noise'` (`src/generation/perlin_lib.py`; el paquete no está en el venv local) |

Esta rama solo agrega archivos: `git diff origin/main` no muestra cambios en módulos ni en pruebas existentes. Esas 7 pruebas no se pudieron comprobar aquí y deberían pasar en un entorno con `noise` y sin el bloqueo de la DLL, como el de Kaggle.

## 7. Siguiente paso

1. SV resuelve los 7 PENDIENTES en `results/fx_pixeles/resoluciones.json` y confirma la interpretación de la regla 2.1 (§4).
2. Con ese commit, X2 (Kaggle CPU) y X3 (Kaggle GPU) según `docs/correr_fx_kaggle.md`, empezando por las pruebas acotadas.
