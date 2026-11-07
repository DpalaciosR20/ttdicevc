# Reporte E1b — F4.1 y F4.7 con los datos reales

> Bloque E, etapa 1b · Rol: SV · Rama `sv/f4-recuperabilidad` (desde `main` `5af17a8`) · 2026-11-07
> Reporte sin interpretación: cifras, verificaciones, pruebas y hashes. La lectura está en `docs/bitacora/F4.1.md`, `F4.7.md` y `F4.6.md`.

## a. Commits

| Commit | Contenido |
|---|---|
| `e9959df` | Recepción de los zips de Kaggle (sin `.npz`), verificación y `resoluciones.json` (E1-1 a E1-7) |
| `a5d13c0` | `recuperabilidad --val` → `val/resultados_val.json` |
| `71f9277` | `recuperabilidad --prueba` (consulta única) → `prueba/resultados_prueba.json`, `prueba/predicciones_prueba.csv` |
| `0cfdd17` | `geometria` → `results/f47_geometria/resultados.json` |
| `b0a28dc` | Cifras canónicas `f41.*` y `f47.*` (esquema, colectores y pruebas) |
| `aaddad9` | `notebooks/F4.1_F4.7.ipynb` ejecutado y 5 figuras |
| `a21c9eb` | Bitácora F4.1 (+ `e1b/replicas_delta_f1_signo.{py,json}`) |
| `451908b` | Bitácora F4.7 |
| `13dfcb7` | Cierre de F4.6 (condición 2 y frase por técnica) |
| commit de este reporte | `reporte_e1b.md` y los logs de pytest y `--check` en `e1b/` |

Todos están en `origin/sv/f4-recuperabilidad`.

## b. Recepción (punto 1)

Origen: `G:\Mi unidad\tt2-insumos\`.

| Zip | sha256 (bytes) | Destino |
|---|---|---|
| `f41_resultados.zip` (15,555 B; 11 entradas) | `39e65de7bf8a6c396c1b9b984b358d7ab5037b3a6327e79c7103659e30ea8ed1` | `results/f41_recuperabilidad/caracteristicas/` (versionado) |
| `f41_caracteristicas_npz.zip` (38,669,230 B; 8 `.npz`) | `55a074a07981e5a374d771270f16c5adca7a5715be15fa393c84944eb7353513` | `f41_caracteristicas/` (ignorado por `.gitignore:32`) |

Se listó el contenido antes de extraer: solo las rutas esperadas.

Verificación (`e1b/verificacion_recepcion.py` → `e1b/verificacion_recepcion.log`; resultado **TODO OK**):

| Comprobación | Resultado |
|---|---|
| `caracteristicas_meta.json:git_commit` | `9f3b7b016bcb24d41b58d46ffdb8af7f26267295` · `smoke` = `false`, `limit` = `null` |
| `modelos_verificacion.json` | 15/15, `passed` = `true` |
| `datos_verificacion.log` | `manifiesto OK` |
| Identidad ViT = ImageNet | Perlin y Lorenz, semillas 42, 7, 123, 2024 y 99: `identico` = `true`, 150/150 tensores |
| `dir_digest` = `meta/data_manifest.csv` | gray_scott `c3593c9b…31db`, perlin `b8ce53c6…3ccb`, lorenz `6fc478a5…de9b` |
| sha256 de los modelos del meta | los 15 están en `meta/artefactos_manifest.csv` |
| sha256 de cada `.npz` | = `caracteristicas_manifest.csv` = `npz_sha256sum.txt` (8/8) |
| Contenido de cada `.npz` | `features` float32 1136 × d (1280 o 768), finitas; `sample_id` y `split` en el orden de `meta/sample_index.csv` |

sha256 de los `.npz`:

| Archivo | sha256 |
|---|---|
| `gray_scott__efficientnet_b0_finetune_seed42.npz` | `662fffa11582d84049af338a4725521456a3fe3abb943e4993d7e602f2b524fd` |
| `gray_scott__efficientnet_b0_finetune_seed7.npz` | `fb9e0d05fede0fd8abe19da9c6ad876cc2c530d37f7c5592edd9a9f6cb644b9b` |
| `gray_scott__efficientnet_b0_finetune_seed123.npz` | `7f1a77a5d9b211460309c4119e3ad6bb1896aae0a5a70084259957187d287f4d` |
| `gray_scott__efficientnet_b0_finetune_seed2024.npz` | `72674fb5ec7f415795af01d7b27a9150528ed4aae1b445fb37d5b8d0c46d5260` |
| `gray_scott__efficientnet_b0_finetune_seed99.npz` | `92eba28bc5b8d5658da59bb05723625f3d48779cad65e46ccc088e8510043016` |
| `gray_scott__efficientnet_b0_imagenet.npz` | `95a6ceb318360636eb9677f87bab050b88ae8d55e3aa117f03b9e9e2d47d1e6a` |
| `perlin__vit_b16_imagenet.npz` | `7c73f0542589caf0287dd1963b6b6c67ee8d721614cd0774d32644bc16077816` |
| `lorenz__vit_b16_imagenet.npz` | `17a7443bbf85b82805bd58757df13b6f838cab24bb61a297f899a091961498be` |

## c. Resoluciones (punto 2)

`results/f41_recuperabilidad/resoluciones.json`, con el contenido indicado por SV, commiteado en `e9959df` antes de cualquier cálculo. sha256 (LF): `098a38ad6637fe25ad184e0549baa18309b1a9cb32657f75c7ff121595ef26af`.

## d. Ejecuciones (punto 3)

Cada paso corrió una sola vez, sin tuberías ni filtros, con la salida redirigida a un archivo del *scratchpad* y copiada después al repo.

| Paso | Comando | HEAD | Código | Duración | Salida |
|---|---|---|---|---|---|
| a | `python -m src.fase4.recuperabilidad --val --features-dir f41_caracteristicas --resoluciones results/f41_recuperabilidad/resoluciones.json` | `e9959df` | 0 | 16.0 s | `val/consola.log` |
| b | igual con `--prueba` | `a5d13c0` (árbol limpio) | 0 | 13.4 s | `prueba/consola.log` |
| c | `python -m src.fase4.geometria --features-dir f41_caracteristicas --resoluciones results/f41_recuperabilidad/resoluciones.json` | `71f9277` | 0 | 390.2 s | `results/f47_geometria/consola.log` |

- En (b), la compuerta pasó: etiqueta, plan, árbol limpio, val recalculada idéntica a la versionada, `idx_sha256` = `322edaaa5f1175bb4d4a16409d7f852fa7e67fa5daa8a504bb4976e67d0a7e37` = Fase 5.
- En (c), el árbol tenía cambios sin commitear en `src/canonical/build.py`, `src/canonical/schema.json` y `tests/test_canonical_numbers.py` (cifras canónicas en preparación). `geometria.py` no exige árbol limpio ni lee esos archivos; `git diff 71f9277 HEAD -- src/fase4` está vacío.

## e. Cifras de F4.1

### R² por componente

Validación (`resultados_val.json:…r2_val`):

| Técnica | Backbone | u0 | u1 | u2 | u3 | u4 |
|---|---|---|---|---|---|---|
| Gray-Scott | EfficientNet-B0 ajuste fino, s42 (principal) | 0.9638 | 0.8579 | 0.9413 | 0.8440 | 0.9643 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s7 | 0.9642 | 0.8444 | 0.9327 | 0.8692 | 0.9666 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s123 | 0.9661 | 0.8543 | 0.9488 | 0.8452 | 0.9711 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s2024 | 0.9665 | 0.8705 | 0.9479 | 0.8588 | 0.9627 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s99 | 0.9633 | 0.8617 | 0.9529 | 0.8424 | 0.9625 |
| Gray-Scott | EfficientNet-B0 ImageNet sin ajustar | 0.9807 | 0.8622 | 0.9435 | 0.9009 | 0.9806 |
| Perlin | ViT-B/16 ImageNet (principal) | 0.9458 | 0.7656 | 0.9534 | 0.8526 | 0.9059 |
| Lorenz | ViT-B/16 ImageNet (principal) | 0.9422 | 0.6368 | 0.9014 | 0.8010 | 0.9928 |

Prueba (`resultados_prueba.json:…r2_test`):

| Técnica | Backbone | u0 | u1 | u2 | u3 | u4 |
|---|---|---|---|---|---|---|
| Gray-Scott | EfficientNet-B0 ajuste fino, s42 (principal) | 0.9684 | 0.8609 | 0.9391 | 0.8619 | 0.9651 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s7 | 0.9684 | 0.8997 | 0.9399 | 0.8867 | 0.9664 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s123 | 0.9653 | 0.8901 | 0.9386 | 0.8784 | 0.9634 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s2024 | 0.9610 | 0.8646 | 0.9487 | 0.9051 | 0.9651 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s99 | 0.9585 | 0.8736 | 0.9368 | 0.9086 | 0.9609 |
| Gray-Scott | EfficientNet-B0 ImageNet sin ajustar | 0.9748 | 0.8848 | 0.9497 | 0.8988 | 0.9818 |
| Perlin | ViT-B/16 ImageNet (principal) | 0.9492 | 0.7556 | 0.9515 | 0.8530 | 0.9142 |
| Lorenz | ViT-B/16 ImageNet (principal) | 0.9290 | 0.6143 | 0.9253 | 0.8734 | 0.9921 |

Train por *cross-fitting* (`resultados_val.json:…r2_train_crossfit`; sin clave canónica):

| Técnica | Backbone | u0 | u1 | u2 | u3 | u4 |
|---|---|---|---|---|---|---|
| Gray-Scott | EfficientNet-B0 ajuste fino, s42 (principal) | 0.9760 | 0.8915 | 0.9568 | 0.8992 | 0.9743 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s7 | 0.9762 | 0.9062 | 0.9546 | 0.9120 | 0.9749 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s123 | 0.9746 | 0.8821 | 0.9618 | 0.9026 | 0.9739 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s2024 | 0.9801 | 0.9017 | 0.9628 | 0.9184 | 0.9767 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s99 | 0.9684 | 0.9082 | 0.9503 | 0.9074 | 0.9674 |
| Gray-Scott | EfficientNet-B0 ImageNet sin ajustar | 0.9875 | 0.9196 | 0.9657 | 0.9265 | 0.9873 |
| Perlin | ViT-B/16 ImageNet (principal) | 0.9671 | 0.7803 | 0.9577 | 0.8759 | 0.9512 |
| Lorenz | ViT-B/16 ImageNet (principal) | 0.9661 | 0.6592 | 0.9303 | 0.8662 | 0.9939 |

### alpha y LogReg (validación)

| Técnica | Backbone | alpha | LogReg sobre û: C, `class_weight` | F1-Val û |
|---|---|---|---|---|
| Gray-Scott | EfficientNet-B0 ajuste fino, s42 (principal) | 100 | 0.3162, `balanced` | 0.6800 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s7 | 316.2 | 1, `balanced` | 0.7004 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s123 | 100 | 3.162, `balanced` | 0.7004 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s2024 | 316.2 | 1, `None` | 0.6854 |
| Gray-Scott | EfficientNet-B0 ajuste fino, s99 | 316.2 | 1, `balanced` | 0.6879 |
| Gray-Scott | EfficientNet-B0 ImageNet sin ajustar | 100 | 10, `balanced` | 0.7165 |
| Perlin | ViT-B/16 ImageNet (principal) | 100 | 100, `balanced` | 0.6902 |
| Lorenz | ViT-B/16 ImageNet (principal) | 10 | 0.0001, `balanced` | 0.6800 |

LogReg sobre u: C = 0.1, `balanced`, índice 13, F1-Val 0.700358422939068, 6 combinaciones empatadas, 0 avisos de convergencia; `p_dep` de val y prueba = CSV de F2.1 (diferencia máxima 0.0). F1-Test sobre u: 0.7246376811594203 (= `ref.techo5d.logreg.f1_macro`).

### F1 sobre û, ΔF1 e IC (prueba)

ΔF1 = F1(u) − F1(û); IC percentil al 95 % sobre las réplicas de A1. Las dos últimas columnas son de `e1b/replicas_delta_f1_signo.json` (recalculadas desde `predicciones_prueba.csv`; el script comprueba que el IC coincide con `resultados_prueba.json`).

| Técnica | Backbone | F1-Test û | F1-Test u | ΔF1 | IC 95 % | IC contiene 0 | Réplicas < 0 / = 0 / > 0 | Muestras con ŷ distinta |
|---|---|---|---|---|---|---|---|---|
| Gray-Scott | s42 (principal) | 0.7348 | 0.7246 | −0.0102 | [−0.0335, 0.0000] | sí | 6290 / 3710 / 0 | 1 |
| Gray-Scott | s7 | 0.7267 | 0.7246 | −0.0020 | [−0.0410, 0.0358] | sí | 5310 / 780 / 3910 | 4 |
| Gray-Scott | s123 | 0.7063 | 0.7246 | +0.0184 | [−0.0197, 0.0592] | sí | 1487 / 830 / 7683 | 4 |
| Gray-Scott | s2024 | 0.6796 | 0.7246 | +0.0451 | [0.0009, 0.0960] | no | 208 / 31 / 9761 | 6 |
| Gray-Scott | s99 | 0.6879 | 0.7246 | +0.0367 | [−0.0067, 0.0845] | sí | 483 / 176 / 9341 | 6 |
| Gray-Scott | ImageNet sin ajustar | 0.6959 | 0.7246 | +0.0287 | [0.0000, 0.0650] | sí | 0 / 536 / 9464 | 3 |
| Perlin | ViT-B/16 ImageNet (principal) | 0.6747 | 0.7246 | +0.0499 | [0.0107, 0.0971] | no | 0 / 56 / 9944 | 5 |
| Lorenz | ViT-B/16 ImageNet (principal) | 0.7367 | 0.7246 | −0.0120 | [−0.0470, 0.0166] | sí | 7692 / 523 / 1785 | 3 |

Valores exactos de los límites que coinciden con 0: Gray-Scott s42 `ic` = [−0.0335097001763669, 0.0]; Gray-Scott ImageNet `ic` = [0.0, 0.06500557784796973]. `contiene_0` usa `ic[0] <= 0 <= ic[1]` (`src/fase4/recuperabilidad.py:304`, commit `9f3b7b0`).

### Condición 2 de F4.6 (`…condicion_2_F46`, backbone principal)

| Técnica | Condición 2 |
|---|---|
| Gray-Scott | `true` |
| Perlin | `false` |
| Lorenz | `true` |

## f. Cifras de F4.7

`results/f47_geometria/resultados.json`; n = 908 (`train`), B = 9,999, `default_rng(2026)` por técnica, cola superior, p = (k + 1)/(B + 1); sin sensibilidad.

| Técnica | Backbone | ρ de Spearman | k | p | Holm: rango, α_h, rechaza | Media bajo permutación | DE bajo permutación |
|---|---|---|---|---|---|---|---|
| Gray-Scott | `efficientnet_b0_finetune_seed42` | 0.8019337781831091 | 0 | 0.0001 | 1, 0.0167, `true` | −6.013950180537694e-06 | 0.0033511020353439863 |
| Perlin | `vit_b16_imagenet` | 0.5221038592161575 | 0 | 0.0001 | 1, 0.0167, `true` | −3.383190893109259e-06 | 0.007346918329106561 |
| Lorenz | `vit_b16_imagenet` | 0.3190933701840265 | 0 | 0.0001 | 1, 0.0167, `true` | 1.2978540364422558e-05 | 0.007027478324237247 |

Reproducción en el notebook: ρ recalculado desde u y los `.npz` = `resultados.json`, |dif| = 0.0 en las tres (`results/f47_geometria/figures/reproduccion_rho.json`).

## g. Cifras canónicas (punto 4)

- `src/canonical/build.py`: `collect_f41`, `collect_f47`, `check_f41`, `_SplitCollector`.
- `src/canonical/schema.json`: métricas `r2`, `ridge_alpha`, `spearman_rho`; `value.train` opcional (F4.7 es solo `train`).
- 179 claves nuevas (57 `f41.val.*`, 92 `f41.prueba.*`, 30 `f47.*`); ninguna clave previa cambia; los `inputs` previos no cambian y se añaden 4.
- Los F1 de prueba de F4.1 (û por *backbone* y u) son `verificado`, con matriz de confusión, recalculados desde `predicciones_prueba.csv`; `check_f41` exige que el F1 sobre u sea `ref.techo5d.logreg.f1_macro` en val y prueba.
- `tests/test_canonical_numbers.py`: 6 pruebas nuevas (`test_f41_f47_presencia`, `test_f41_f47_valores_en_su_localizador`, `test_f41_f1_prueba_desde_predicciones`, `test_f41_u_coincide_con_f21`, `test_f41_condicion_2_y_contiene_0`, `test_f47_solo_train_y_holm`) y ajuste de 2 pruebas existentes (`test_evaluator_y_null_reason`, `test_sin_cambios_previos_salvo_desbloqueo`) y de `leaf_cases` para admitir los prefijos `f41.`/`f47.`.
- La nota de `f46.contencion.<técnica>.condicion_1` sigue diciendo «la condición 2 … está PENDIENTE (P6)»: no se modificó, porque cambiaría una clave previa (`test_sin_cambios_previos_salvo_desbloqueo`). La condición 2 está en `f41.prueba.<técnica>.condicion_2_f46`.

| Verificación | Resultado | Evidencia |
|---|---|---|
| `python -m src.canonical.build` con `CANON_UMAP_PARAMS` y `CANON_STRICT=1` | escrito | — |
| `python -m src.canonical.build --check` | al día (23413 líneas) | `e1b/check_e1b.log` |
| `pytest tests/test_canonical_numbers.py` (CANON_STRICT=1) | 854 aprobadas | `e1b/pytest_canon_e1b.log` |

## h. Pruebas

`python -m pytest tests/` con `CANON_UMAP_PARAMS` y `CANON_STRICT=1`, en `13dfcb7`: **1173 aprobadas, 30 omitidas, 0 fallidas** (164.46 s). Las 30 omisiones son las de `tests/test_label_independence.py` (requieren `F03_UMAP_PARAMS`), como en bloques anteriores. Evidencia: `e1b/pytest_e1b.log`, `e1b/pytest_e1b.xml`.

## i. Notebook y figuras

`notebooks/F4.1_F4.7.ipynb`, ejecutado con `jupyter nbconvert --execute --inplace`: 19 celdas, 0 errores, 9.1 s de kernel. No reajusta modelos ni consulta prueba; verifica el sha256 de los `.npz` y de los JSON frente a `canonical_numbers.json:inputs`. PNG a 300 dpi, títulos y pies en español, paleta de `src/viz/style.py`.

| Figura | Archivo |
|---|---|
| R² por componente y técnica, val y prueba | `results/f41_recuperabilidad/figures/f41_01_r2_por_componente.png` |
| F1 sobre û frente a u y ΔF1 con IC | `results/f41_recuperabilidad/figures/f41_02_f1_u_hat_delta.png` |
| Matrices de distancia de train | `results/f47_geometria/figures/f47_01_matrices_distancia.png` |
| Distancias por par y ρ de Mantel | `results/f47_geometria/figures/f47_02_pares_mantel.png` |
| ρ observado frente a la permutación | `results/f47_geometria/figures/f47_03_rho_vs_permutacion.png` |

La primera ejecución se repitió una vez para corregir la maquetación de las figuras 2 y 3 (barras de color encimadas y pie desplazado); no cambió ninguna cifra.

## j. Bitácoras

- `docs/bitacora/F4.1.md` y `F4.7.md`: estructura de `/cierre-tarea`, tablas «qué se afirma / qué no» de `docs/plan_fase4.md`, estado **Cerrada**, sin **[completar]**.
- `docs/bitacora/F4.6.md`: condición 2 por técnica (4.4), H5 con la frase solo en Gray-Scott y Lorenz y la condición que falla en Perlin, estado **Cerrada**. Queda un **[completar]**: el diagrama de intersección con los cuatro conjuntos (BETO, 768D, techo 5D y visual); el de tres conjuntos (sin 768D) está en `83c5edd`, rama `sv/f5-resultados`, no fusionada en `main`.
- `/cierre-tarea` pide preguntar antes de redactar y no hacer *push*; se siguió el bloque E1b, que pide commit, *push* y PR. Los datos que la evidencia no cubría se resolvieron solo desde el repo; no hubo preguntas abiertas.
- Al fusionar `sv/f5-resultados` (`83c5edd` añade una sección «Figuras» al final de `F4.6.md`) puede haber un conflicto de texto con este cierre, que modifica §7–§9 del mismo archivo.

Tabla de control de las cifras de las bitácoras (fuente de cada una):

| Cifras | Fuente |
|---|---|
| R² (val, prueba, train) | `resultados_val.json:…r2_val`, `…r2_train_crossfit`; `resultados_prueba.json:…r2_test` |
| alpha, combinaciones y F1-Val de las LogReg | `resultados_val.json:resultados.tecnicas.<t>.<b>`, `resultados.u.logreg_u` |
| F1-Test, ΔF1, IC, `contiene_0`, condición 2 | `resultados_prueba.json:resultados` |
| Réplicas por signo, muestras con ŷ distinta | `e1b/replicas_delta_f1_signo.json` |
| ρ, k, p, Holm, media y DE bajo permutación | `results/f47_geometria/resultados.json` |
| C, nula, kappa (F4.6) | `results/f5_contraste/f46_resultados.json` (sin cambios) |
| Duraciones | salida de las ejecuciones (§d) |
| Versiones locales | `.venv`: Python 3.13.2, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1 |

## k. Hashes de los resultados

sha256 con fin de línea LF (= `canonical_numbers.json:inputs`):

| Archivo | sha256 |
|---|---|
| `results/f41_recuperabilidad/val/resultados_val.json` | `028a55edc71ec7ad51b1f2dc4e43c963f12c7d29747c62a32fb271704458c597` |
| `results/f41_recuperabilidad/prueba/resultados_prueba.json` | `4205b49d74361f1f397cdd4a39346979f1f6ce19227271b6e3a5bdb1a5f614df` |
| `results/f41_recuperabilidad/prueba/predicciones_prueba.csv` | `c7c03fc044482df012dff2d486cbb4b95c40f86ab95576bc937903d244fc883f` |
| `results/f47_geometria/resultados.json` | `492a408ce441dfca998c10a3335f40bbcf6e7fc132fbe45c5c3ddbab05aaa687` |
| `results/canonical_numbers.json` | `c4af96792ec7d64a9518794108bfb00c8a9d46282b86f0e2dca8a1aa494bc125` |

## l. PR

**No se abrió.** `gh auth status`: «You are not logged into any GitHub hosts». La rama está en `origin/sv/f4-recuperabilidad`. Para abrirlo: `! gh auth login` y luego `gh pr create --base main --head sv/f4-recuperabilidad`. No se fusionó nada.

## m. Pendientes fuera de este bloque

| Pendiente | Responsable |
|---|---|
| Abrir el PR de `sv/f4-recuperabilidad` (sin sesión de `gh`) | SV |
| Subir `f41_caracteristicas_npz.zip` a Drive `MiUnidad/tt2-artefactos/f41_caracteristicas/` y a un *dataset* de Kaggle, si no se hizo ya | SV |
| PR y fusión de `sv/f5-resultados` (`83c5edd`, notebook de la Fase 5) | SV |
| Diagrama de intersección de F4.6 con los cuatro conjuntos ([completar] en `F4.6.md`) | SV |
