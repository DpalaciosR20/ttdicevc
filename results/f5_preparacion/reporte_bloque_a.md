# Bloque A posterior a la Fase 3 — recepción de insumos de CE e inventario de F4.5, F4.6 y F5.1–F5.6

> Rol: SV · Fecha: 2026-11-06 · Rama `sv/f5-contraste` @ `1cbf606` (= `main` = `origin/main`).
> Solo se escribió en `results/f5_preparacion/`. No hay bitácora ni commit, y no se modificó código existente.
> **Barreras que se respetaron:**
> - No se calculó ninguna métrica de prueba de las referencias de la Fase 2 (768D, techo 5D, pisos, brillo, descriptores, longitud).
> - De `results/predictions/referencia/` solo se leyeron las columnas `sample_id`, `split` e `y_true`, y únicamente para comprobar presencia y alineación; `p_dep` no se cargó.
> - No se calculó ningún estadístico de la Fase 5 (θ, IC, piso, Holm, Q de Cochran, McNemar, veredicto).
> - `src/protocol/verdict.py` solo se leyó y se probó con sus pruebas sintéticas (`tests/test_verdict.py`).
> - El F1 de prueba se recalculó solo para las 45 corridas visuales, como verificación de integridad.

## Archivos de este directorio

| Archivo | Contenido |
|---|---|
| `verificar_45.py`, `verificar_45.log` | Punto 1, a)–g) |
| `f1_recalculado_45.csv` | F1 de val y prueba recalculados de las 45 corridas, commit, `created_at` y selección |
| `consultas_test.py`, `consultas_test.log`, `consultas_test.csv` | Punto 1 h): inventario de predicciones de prueba visuales en `results/` |
| `verificar_reproducibilidad.py`, `verificar_reproducibilidad.log` | Punto 2 |
| `validate.log` | `src.predictions.validate` sobre los 45 CSV |
| `insumos_referencia.log` | Presencia de filas val/test de cada referencia (solo `split`) y selecciones de F2.1/F2.4 |
| `git_estado.log` | Estado git, etiquetas, ancestría, `git diff protocolo-v1 HEAD` |
| `pytest.log` | `tests/test_verdict.py`, `test_protocolo.py`, `test_predictions_format.py`, `test_select_finalist.py` |

---

## 0. Estado

- **PR #31** (`6178433`, 2026-11-06 01:25 −0600) y **PR #32** (`711606f`, 01:19 −0600) están fusionados en `main`. También lo está el **#33** (`1cbf606`, F5.5 entrenamiento), que es el HEAD actual.
- **La rama `sv/f5-contraste` ya existía**, en local y en `origin`, sobre `1cbf606`, idéntica a `main`. No se creó de nuevo; se trabajó en ella.
- Etiquetas (`git_estado.log`):
  - `protocolo-v1`: anotada el 2026-11-02 18:14:54 −0600, apunta a `91c84df`;
  - `seleccion-f33`: anotada el 2026-11-05 16:46:46 −0600, apunta a `b2cf9f8`.
- **Lecturas hechas:**
  - bitácoras F3.4, F3.5, F3.6, F3.7, F5.5-entrenamiento y F2.1–F2.4 (identificación, método y hallazgos);
  - `docs/protocolo_preregistrado.md` completo, con las Enmiendas 1–3;
  - `docs/matriz_experimentos.md` §3–§10;
  - `configs/protocolo.json`;
  - `src/protocol/verdict.py` y `tests/test_verdict.py`;
  - filas F4.5, F4.6 y F5.1–F5.6 de `plan/plan_tt2_v2.md`.
- **Pruebas** (`pytest.log`): 110 aprobadas en 4.9 s.

---

## 1. Integridad de los 45 CSV

**Comando:** `python results/f5_preparacion/verificar_45.py` (rc 0, «FALLAS: 0»).

| Punto | Comprobación | Resultado |
|---|---|---|
| a | `python -m src.predictions.validate` sobre `final/`, `control_permutacion/` y `sensibilidad/` | **45/45 válidos** (`validate.log`). 15 CSV por fase; semillas {42, 7, 123, 2024, 99} en las tres técnicas de cada fase |
| b | `sample_id` de val y de prueba frente a `meta/sample_index.csv`; `y_true` frente al índice, entre corridas y frente al `y_true` de prueba de `text__beto__finetune__seed42.csv` (leído con `usecols=[sample_id, split, y_true]`) | **Idénticos en las 45**. 114 + 114 filas, 44 depresivas en cada split |
| c | Configuración: `final` y `control_permutacion` frente a `select_finalist`; `sensibilidad` frente a `select_second` (recalculados sobre `primera_pasada`) | **Coinciden.** Finalistas: GS `efficientnet_b0/finetune`, Perlin `vit_b16/frozen`, Lorenz `vit_b16/frozen`. Segundos: GS `resnet18/finetune`, Perlin `efficientnet_b0/frozen`, Lorenz `efficientnet_b0/finetune`. En los JSON: `selection.role` = `finalist` / `finalist` / `second`, `matches = True`, `override = None` |
| d | Hiperparámetros: `lr`, `weight_decay` y `batch_size` del JSON de corrida y del manifiesto frente a `decision.params` de `results/f34_hpo/<tec>/hpo_result.json` (final y permutación), o frente a `configuracion_comun` de la matriz (sensibilidad); además épocas máximas 15 y paciencia 4 | **Coinciden en las 45** |
| e | `permutation_indices` (JSON y manifiesto) = `numpy.random.default_rng(99).permutation(908)`, con `permute_labels = 99`. En `final` y `sensibilidad`, `None` | **Coinciden.** `y_true` de val y prueba sin alterar (punto b) |
| f | F1-macro de val y de prueba recalculado desde `p_dep` (ŷ = 1 ⇔ `p_dep > 0.5`) frente a `best_f1_val` y `f1_test` del JSON | **\|Δ\| = 0.0 en las 45** (tolerancia 1e-12). El `p_dep` del CSV es igual bit a bit a `val_probs` y `test_probs` del JSON, y `val_preds`/`test_preds` = `probs > 0.5`. Ningún `p_dep` es exactamente 0.5 |
| g | Ancestría y Enmienda 3 | Ver abajo |
| h | Consulta única | Ver abajo |

### f) Resúmenes y `ddof`

`summary_by_config.csv` (escrito por `src/training/run_grid.py:450-452`, `aggregate_by_config`) usa **`ddof = 1`** en val y en prueba. Coincide a 4 decimales con la DE muestral recalculada en las 9 combinaciones de fase y técnica. Por ejemplo, en `final/gray_scott` la DE de val es 0.0093 con `ddof = 1` y 0.0083 con `ddof = 0`.

En el código revisado (`run_grid.py`, `run_hpo.py`, `select_finalist.py`, `src/canonical/build.py`) **no hay `ddof = 0`**. Las únicas apariciones de `ddof = 0` están en definiciones de descriptores de F2.4 (`configs/protocolo.json` → `descriptores.params.contrast.ddof`, `edge_density` y la `d` de Cohen de la Tabla 11), que no intervienen aquí.

> **DISCREPANCIA D1 (documentación, CE).** `docs/bitacora/F3.5.md` §4 dice: «El resumen de `run_grid` en `summary_by_config.csv` usa la desviación estándar poblacional en val; las cifras de arriba usan ddof = 1». El código y los valores muestran `ddof = 1`: la nota es incorrecta. No afecta ninguna cifra.
>
> **DISCREPANCIA D2 (documentación, CE).** `docs/bitacora/F3.5.md` §3.1, paso 3, dice «predicciones = `probs ≥ 0.5`». El código (`train_model.py:100`, `DECISION_THRESHOLD`), la especificación F1.5 §3.3 y el protocolo §4.1 usan `p_dep > 0.5`. Ningún `p_dep` de las 45 corridas vale 0.5, así que no hay efecto numérico.

### g) Ancestría y Enmienda 3

| Fase | Commit de ejecución (JSON y manifiesto) | `protocolo-v1` ancestro | `seleccion-f33` ancestro | `9027167` ancestro | `protocol.commit` en JSON | `created_at` (UTC) |
|---|---|---|---|---|---|---|
| `final` (F3.5) | `ef16ab5` | sí (rc 0) | sí (rc 0) | sí (rc 0) | `9027167` | 03:58:56 – 04:21:41 del 2026-11-06 |
| `control_permutacion` (F3.6) | `a005f75` | sí | sí | sí | `9027167` | 05:27:09 – 05:46:59 |
| `sensibilidad` (F5.5) | `b170d84` | sí | sí | sí | `9027167` | 07:42:06 – 07:54:13 |

- La **Enmienda 3** es el commit `9027167` (DpalaciosR20, 2026-11-05T20:32:22−06:00, es decir, 2026-11-06 02:32:22 UTC). El `created_at` mínimo de F3.5 es 03:58:56 UTC, **1 h 26 min después**, y `9027167` es ancestro de `ef16ab5`.
  - *Nota:* `created_at` es el momento en que se escribió cada JSON, al terminar la corrida. La primera corrida de Gray-Scott duró unos 0.8 min (`summary.csv`), así que su consulta de prueba también es posterior a la enmienda.
- **Resumen del texto de la Enmienda 3:**
  - **Momento:** se registró antes de cualquier consulta de prueba de los finalistas y antes de generar mapas de F3.7, con los finalistas ya fijados por `seleccion-f33`.
  - **Alcance:** solo afecta a los análisis explicativos de F3.7 y F4.3. No modifica ningún modelo, hiperparámetro, estadístico ni criterio de decisión del Objetivo V. Cambia `configs/matriz_experimentos.json` (nueva huella `01d177bb…`); `configs/protocolo.json` no cambia.
  - **Motivo:** en los ViT-B/16 congelados (Perlin y Lorenz), el *attention rollout* sin gradientes depende solo de la imagen y de los pesos de ImageNet. Sus mapas serían idénticos entre semillas y frente al control de permutación.
  - **Cambio:**
    - el método principal de `vit_b16` pasa a ser el *rollout* ponderado por gradiente de Chefer et al. (2021), con el gradiente del *logit* depresivo a través de la cabeza entrenada;
    - el *rollout* de Abnar y Zuidema se conserva solo como descripción de la atención del *backbone*;
    - la prueba de cordura de F4.3 añade la aleatorización de la cabeza como primer paso.

### h) Consulta única de prueba

**Comando:** `python results/f5_preparacion/consultas_test.py`. Recorre todo `results/` (excepto esta carpeta) y busca:
- CSV con `split == test` y `p_dep` no nulo, de técnica visual y arquitectura visual;
- JSON con `test_probs` o `test_preds` no nulos a cualquier nivel.

Para cada registro calcula una huella del vector de prueba. *Aclaración de la barrera:* al recorrer `results/`, el script carga completos los CSV de `referencia/` que tienen columnas `split`, `p_dep`, `technique` y `arch`, pero los descarta por `arch` (brightness, descriptors, logreg…) antes de cualquier operación. Sobre esos `p_dep` no se calcula ni huella ni métrica. `p_dep` se lee con `float()`, porque `pd.read_csv` difiere en 1 ulp; es el mismo problema de F2.2 que corrigió `validate.py`.

| Resultado | Valor |
|---|---|
| Registros encontrados | 90: los 45 CSV de `results/predictions/{final, control_permutacion, sensibilidad}/` y sus 45 JSON de corrida en `results/{f35_final, f36_permutacion, f55_sensibilidad}/<tec>/runs/` |
| Claves (técnica, arquitectura, estrategia, fase, semilla) | **45, cada una con exactamente una huella**: el CSV es copia exacta del JSON de su corrida |
| Registros de otras fases (primera pasada, F3.4, *smoke*, pruebas, `artefactos_drive`) | **0** |
| Métricas de prueba fuera de las predicciones | Solo `results/{f35_final, f36_permutacion, f55_sensibilidad}/<tec>/summary.csv` (columna `f1_test`, 5 filas cada uno) y `summary_by_config.csv`, derivados de las mismas corridas |

**Otros puntos a informar:**

1. **F3.7 volvió a hacer *forward* sobre prueba** con los 30 modelos guardados (F3.5 y F3.6). `results/f37_mapas/records/*.json` no guarda predicciones, solo `p_dep_max_abs_diff` frente a las archivadas: máximo 1.21e-5 en prueba (Gray-Scott perm99, semilla 99). La matriz §8 autoriza los mapas sobre las 114 muestras de prueba («Muestras: las 114 de `val` y las 114 de `test`»). No hay una segunda predicción archivada ni ninguna decisión basada en ella. Se registra porque el protocolo §10 dice «una inferencia por modelo y semilla».
2. Fuera de `results/`, los notebooks `notebooks/F3.2_primera_pasada.ipynb` y `F3.6_permutacion.ipynb` y las pruebas `tests/{smoke_test_training, test_run_grid, test_run_hpo, test_run_maps}.py` mencionan `test_probs`. Las pruebas usan datos sintéticos; los notebooks leen los JSON ya listados. No se recorrieron sus salidas celda por celda.

---

## 2. Afirmaciones de reproducibilidad de CE

**Comando:** `python results/f5_preparacion/verificar_reproducibilidad.py` (rc 0, «FALLAS: 0»). Solo validación.

| Afirmación | Comprobación | Resultado |
|---|---|---|
| F3.5, semillas 42, 7 y 123, Gray-Scott y Lorenz: iguales a la primera pasada | `val_probs` (JSON de F3.5) frente a `val_probs` de `results/f32_primera_pasada/runs/*.json`; `p_dep` de val del CSV frente a `results/predictions/primera_pasada/`; `best_epoch`; `best_f1_val` | **6/6 iguales bit a bit**, misma mejor época |
| F3.5, Perlin: igual al trial 10 de F3.4 | `val_probs`, `best_epoch` y `best_f1_val` frente a `results/f34_hpo/perlin/runs/trial10__seed{42,7,123}.json` | **3/3 iguales bit a bit** (mejores épocas 3, 3 y 2) |
| Extra: F5.5 (sensibilidad), semillas 42, 7 y 123, frente a la primera pasada | Igual que arriba | **9/9 iguales bit a bit** |

**F3.4, H5.** Se volvieron a ejecutar los comandos de `results/f34_hpo/git_seleccion.log`:

| Comando | Log original | Ahora |
|---|---|---|
| `git rev-parse seleccion-f33^{commit}` | `b2cf9f87…` | `b2cf9f87…` |
| `git log -1 --format="%H %cI" seleccion-f33` | `… 2026-11-05T16:38:01-06:00` | igual |
| `git merge-base --is-ancestor seleccion-f33 b848d2c` | 1 | **1** |
| `git merge-base --is-ancestor seleccion-f33 HEAD` (HEAD era `a92e4b6`) | 0 | 0, ejecutado con `a92e4b6` explícito |
| `git merge-base --is-ancestor protocolo-v1 b848d2c` | 0 | 0 |
| `git diff --stat b848d2c seleccion-f33 -- src/training src/generation src/selection configs` | solo `run_hpo.py`, 403 líneas borradas | igual |
| `git diff --stat b848d2c a92e4b6 -- <archivos de entrenamiento> src/generation src/selection configs` | vacío | vacío |
| `git for-each-ref refs/tags/seleccion-f33 …` | `tag 2026-11-05 16:46:46 -0600` | igual |

Además:
- los 108 JSON de F3.4 tienen `git_commit = b848d2c` y `created_at` entre 2026-11-05T22:51:39Z y 2026-11-06T01:59:42Z;
- la primera línea de `entorno/gpu.txt` es «Thu Nov 5 22:47:47 2026».

**Las comprobaciones de H5 se reproducen.** La desviación de procedencia es la declarada: el commit de ejecución no desciende de la etiqueta, el código de entrenamiento es el mismo y la ejecución es posterior a la etiqueta.

---

## 3. Cifras canónicas (diagnóstico, sin cambios)

**Convención vigente** (`src/canonical/build.py`, docstring, y `schema.json`):
- toda métrica lleva el par `{val, test}`, y un split nulo exige `null_reason`;
- `verificado` significa que la cifra se recalcula desde predicciones por muestra;
- las referencias de la Fase 2 tienen `test = null` con el motivo «bloqueado hasta la Fase 5 (docs/protocolo_preregistrado.md §10); las predicciones de test están archivadas» (`ref.techo5d.*`, `ref.piso.*`, `ref.descriptores.*`, `ref.emb768_logreg`);
- `TEST_BLOCKED_F32` bloquea la prueba de F3.2 porque esa fase no la consulta;
- **hoy no existe ninguna clave ni colector para F3.4, F3.5, F3.6 ni F5.5.**

| Bloque | Claves propuestas | Colector y fuentes | Prueba: ¿se registra ya? |
|---|---|---|---|
| F3.4 | `f34.<tec>.trial0.f1_macro_mean` / `_std`; `f34.<tec>.mejor_trial.f1_macro_mean`; `f34.<tec>.ganancia.f1_macro` (derivada `diff`); `f34.<tec>.trial_adoptado` y `f34.<tec>.trials` (`kind: count`) | `collect_f34`: F1 de val recalculado desde `val_probs`/`val_true` de `results/f34_hpo/<tec>/runs/*.json` (`verificado`), contrastado con `hpo_result.json` → `decision` y `trials.csv`. Entrada en `inputs` con el sha256 de `hpo_result.json` | **No aplica**: `test = null` con `null_reason` «F3.4 solo validación (loaders sin test; matriz §4)», análogo a `TEST_BLOCKED_F32` |
| F3.5 | `f35.<tec>.<arch>.<strat>.seed<s>.f1_macro` (15) y `f35.<tec>.finalista.f1_macro_mean` / `_std` (derivadas `mean` y `std_ddof1`) | `collect_f35`: `load_predictions('results/predictions/final/*.csv')` → F1 desde `p_dep > 0.5` (`verificado`), contrastado con `runs/*.json` y `summary_by_config.csv`, como `_f32_derived` | **Puede registrarse ya.** El §10 autoriza esta consulta («Una inferencia por modelo y semilla; predicciones archivadas»), la Enmienda 1, punto 2, reconoce que `run_grid.py` registra el F1 de prueba en F3.5, y la bitácora de F3.5 ya lo publica. Lo que sigue bloqueado son los estadísticos de la Fase 5 (θ, D, piso, ρ̂), que no son claves de F3.5 |
| F3.6 | `f36.<tec>.<arch>.<strat>.seed<s>.f1_macro` y `f36.<tec>.permutado.f1_macro_mean` / `_std`. Su media es el «piso visual permutado» | `collect_f36`: igual sobre `results/predictions/control_permutacion/*.csv` | **AMBIGÜEDAD A16.** Por el §10, fila 2, se puede registrar (es la consulta autorizada de F3.6). Por el §4.4 es un **piso**, y el §10, fila 3, reserva las «métricas de prueba» de los pisos a la Fase 5. La bitácora de F3.6 ya publica 0.527, 0.548 y 0.533. Las dos opciones son: valor, o `null_reason` «piso: bloqueado hasta la Fase 5» en las medias, sin tocar las corridas |
| F5.5 | `f55.<tec>.segundo.<arch>.<strat>.seed<s>.f1_macro` y `f55.<tec>.segundo.f1_macro_mean` / `_std` | `collect_f55`: igual sobre `results/predictions/sensibilidad/*.csv` | **Puede registrarse ya.** El §10, fila 4, dice «Una inferencia por semilla», y no es una referencia ni un piso |

**Notas para implementarlo:**
- las claves cumplen el patrón `^[a-z0-9_]+(\.[a-z0-9_]+)*$` (las semillas como `seed42`);
- `std_ddof1` ya existe en `derive`;
- los colectores irían en `build()`, junto a `collect_f33`, con sus propias entradas en `inputs` (sha256 de cada CSV);
- `tests/test_canonical_numbers.py` necesitaría las pruebas correspondientes.

No se ejecutó `src.canonical.build`.

---

## 4. Inventario del protocolo

**Insumos comunes:** todas las referencias tienen 114 filas de val y 114 de prueba ya archivadas (`insumos_referencia.log`), excepto la de **longitud**, que no existe (`text_length*`: «NO EXISTE»). La prueba de esas referencias se archivó en sus tareas con modelos ya ajustados, así que **no hay que inferir nada** salvo para la longitud:

| Referencia | Archivo | Tarea, commit |
|---|---|---|
| BETO | `text__beto__finetune__seed42.csv` | F2.2, `fa50f96` |
| 768D | `emb768__logreg__none__seed42.csv` | F2.2, `fa50f96` |
| Techo 5D (SVM-RBF; `results/f21_techo5d/seleccion.csv`: `techo=True`, C = 1, γ = 1, `balanced`, F1-val 0.7246) | `umap5d__svm_rbf__none__seed42.csv` | F2.1, `eea2ae1` |
| Mayoritaria y estratificado | `none__{majority,stratified}__none__seed42.csv` | F2.3, `53f6e28` |
| Brillo | `<tec>__brightness__none__seed42.csv` | F2.3, `53f6e28` |
| Descriptores (Perlin LogReg, Gray-Scott RF y Lorenz según `results/f24_descriptores/seleccion.csv`) | `<tec>__descriptors__none__seed42.csv` | F2.4, `1d68a4a` |
| Visual permutado | `control_permutacion/*` | F3.6 |
| Finalistas | `final/*` | F3.5 |
| Segundos | `sensibilidad/*` | F5.5 |

**Bootstrap común (§5.2).**
- B = 10,000 réplicas de los 114 índices de prueba con reemplazo, `numpy.random.default_rng(2026)`, pareadas: los mismos índices para todo.
- La forma exacta de la llamada no está fijada (AMBIGÜEDAD A1). Se comprobó con numpy 2.1.3 que `rng.integers(0, 114, (B, 114))`, `rng.choice(114, (B, 114))` y un bucle por réplica de cualquiera de las dos dan **las mismas réplicas**. `rng.integers(0, 114, (114, B)).T` **no** da las mismas. El precedente es `results/f11_potencia/potencia.py:129`, que usa `rng.integers(0, n, size=(B, n))`.
- Las 5 semillas se combinan dentro de cada réplica así: F1 de cada semilla sobre los índices de la réplica y luego su media (§4.1, §5.2). El piso permutado sigue la misma regla (§5.3).

**`verdict.py`** implementa solo §5.2 (p), §5.4 (Holm e IC de θ) y §5.5 (V1–V4 y G1–G4), a partir de réplicas θ\* y D\* ya calculadas. No genera réplicas, no calcula F1, no elige el piso y no implementa §5.6, §6, §7 ni §8.

### F5.2 — No inferioridad frente a BETO (§5.1, §5.2, §5.4)

| Definición (cita) | Insumos | Procedimiento | Implementación | Marca |
|---|---|---|---|---|
| §5.1: «$\theta_t = (\overline{F1}_{img,t} - 0.5) - r\,(F1_{BETO} - 0.5)$ … **H₀:** θ_t ≤ 0 … **H₁:** θ_t > 0». §5.2: «Valor *p* unilateral: $p_t = \#\{\theta_t^* \le 0\} / B$; si es 0, se reporta *p* < 1/B». §5.4: «Dos familias de tres contrastes … Holm a α = 0.05 unilateral … intervalo percentil bilateral al nivel 1 − 2α_h … Si varias técnicas tienen el mismo valor p, todas reciben el menor α_h» | BETO y finalistas: existen. No requieren inferencia | Unidad: índice de prueba (n = 114). F1_img\* = media de los 5 F1 por semilla; F1_BETO\* sobre los mismos índices. B = 10,000, semilla 2026. p unilateral. Holm escalonado en la familia «no_inferioridad» (m = 3), α = 0.05. IC percentil bilateral de θ al nivel 1 − 2α_h, cuantiles `linear` (`configs/protocolo.json`) | `verdict.p_value`, `holm`, `percentile_interval` y `verdicts`. Pruebas: `test_p_value_counts_le_zero`, `test_percentile_interval_level`, `test_holm_*` (7 pruebas), `test_v2_depends_on_holm_level`, `test_three_zero_p_in_both_families` | Sin discrepancia con el texto. A1 (llamada del *bootstrap*). Nota sin marca: el texto no dice que el θ puntual que se reporta sea el observado, aunque es lo convencional |

### F5.1 — Superioridad sobre el piso (§4.4, §5.3, §5.4)

| Definición (cita) | Insumos | Procedimiento | Implementación | Marca |
|---|---|---|---|---|
| §4.4: «El piso de la técnica *t* es el de mayor F1-Macro de prueba entre los cuatro». §5.3: «$D_t = \overline{F1}_{img,t} - F1_{piso,t}$, con H₁: D_t > 0, sobre las mismas réplicas de §5.2 y con $p = \#\{D_t^* \le 0\}/B$. Si el piso es el aleatorio estratificado, se compara con su valor esperado (0.5) como constante. Si es el visual permutado, se promedian sus cinco semillas en cada réplica». §5.2: «En cada réplica se recalculan … el de cada piso» | Mayoritaria, brillo (×3) y permutado (×15): existen. Estratificado: constante 0.5 (Enmienda 2, punto 3; su archivo no interviene). No requieren inferencia | Igual que F5.2, en la familia «superioridad_piso» (m = 3), α = 0.05, unilateral. `configs/protocolo.json`: `pisos.rule = "max_f1_test"`, `order = [majority, stratified, brightness, visual_permutado]`, `superioridad_piso.visual_permutado = "mean_over_seeds_per_replicate"` | `verdict.holm` y `verdicts` (D\* como entrada). La elección del piso **no está implementada**. Pruebas: `test_floor_family_uses_holm`, `test_v3_takes_precedence_over_v1` | **A2 AMBIGÜEDAD** (revisión a): máximo fijado sobre el observado o recalculado en cada réplica. **A3 AMBIGÜEDAD**: empate en el máximo observado. **A4 AMBIGÜEDAD**: IC de D_t (¿se reporta?, nivel). **A16 AMBIGÜEDAD**: estatus de prueba del piso permutado (§10) |

### F5.6 — Veredicto por técnica y global (§5.5) y estimación de la retención (§5.6)

| Definición (cita) | Insumos | Procedimiento | Implementación | Marca |
|---|---|---|---|---|
| §5.5: «V3 … No se rechaza H₀ de superioridad sobre el piso (Holm) · V1 … Supera el piso y se rechaza H₀ de no inferioridad (Holm) · V2 … Supera el piso y el límite superior del intervalo de θ_t (§5.4) es < 0 · V4 … en otro caso». Global: «G1 si al menos una técnica obtiene V1; si no, G2 si al menos una obtiene V2; si no, G3 si las tres obtienen V3; en otro caso, G4» | Salidas de F5.1 y F5.2 | Orden V3 → V1 → V2 → V4; global G1 → G4 | `technique_verdict` y `global_verdict`. Pruebas: `test_v1_v2_v3_v4`, `test_global_verdicts` (parametrizada), `test_step_down_failure_gives_v4` | Coincide con el texto. Asignación: el plan llama F5.6 al veredicto; el protocolo §5.6 es la retención ρ̂. Aquí se tratan juntos |
| §5.6: «$\hat\rho_t = (\overline{F1}_{img,t} - 0.5)/(F1_{BETO} - 0.5)$ con intervalo percentil al 95 % sobre las réplicas de §5.2. Es descriptiva» | BETO y finalistas | Réplicas de §5.2; percentil al 95 % | **No implementado** (`retencion` solo existe en `configs/protocolo.json`) | **A5 AMBIGÜEDAD**: lado y cuantiles del 95 %. **A6 AMBIGÜEDAD**: réplicas con F1_BETO\* ≤ 0.5 |

### F4.5 — Control de longitud (§6, confirmatorio)

| Definición (cita) | Insumos | Procedimiento | Implementación | Marca |
|---|---|---|---|---|
| §6: «Para cada técnica con V1, V2 o V4: $L_t = \overline{F1}_{img,t} - F1_{longitud}$, con H₁: L_t > 0, sobre las réplicas de §5.2 y con Holm sobre las técnicas evaluadas, α = 0.05». «Si no se rechaza H₀, el veredicto … se reporta con la calificación "no distinguible de un clasificador de longitud"». Enmienda 2, punto 4: LogReg sobre `log(n_tokens_beto)` y `truncated_128`, rejilla completa de LogReg de §4.3, `StandardScaler` en `train`, selección de §4.3; `technique = text_length`, `arch = logreg`, `strategy = none`, `seed = 42`, fase `referencia` | **No existe** `text_length__logreg__none__seed42.csv`. Hay que **ajustar** el modelo (aún no está ajustado ni congelado). Características en `meta/sample_index.csv` (sí existen). La rejilla, el escalador, la regla de selección y la semilla 42 están fijados. No están fijados `max_iter` (F2.1 y F2.3 usaron 5000, declarado «no fijado por el protocolo»), la codificación de `truncated_128` (booleano → {0, 1}) ni la base del logaritmo (irrelevante tras `StandardScaler`) | Réplicas de §5.2; L\* = F1_img\* − F1_longitud\*. Holm sobre m = número de técnicas con V1, V2 o V4 (de 0 a 3). α = 0.05, unilateral por H₁. Por analogía, p = #{L\* ≤ 0}/B. No se pide IC. Solo añade una calificación y no cambia V ni G | **No implementado** (`verdict.holm` es reutilizable) | **A7 AMBIGÜEDAD**: fórmula de p (no escrita), m variable y si aplica a las variantes de §8. **A9 AMBIGÜEDAD**: parámetros no fijados del ajuste (`max_iter`) |
| §6: «F1-Macro del finalista por terciles de `n_tokens_beto`, con los puntos de corte calculados en `train`». Plan F4.5: además «Correlación de cada componente de **u** con la longitud en tokens y con el indicador de truncamiento; evaluación estratificada por terciles de longitud del finalista y de BETO» | `sample_index.csv`, finalistas y BETO | Terciles de `train` (n = 908). Con `np.quantile` lineal, los cortes son 86.33 y 381.0, y en `train` hay 3 muestras con exactamente 381 | No implementado | **A8 AMBIGÜEDAD**: método de cuantiles y asignación en el corte; F1 por tercil con 5 semillas (media de F1 o media de `p_dep`); si las correlaciones con **u** y los terciles de BETO del plan siguen vigentes (§9 no lista F4.5) |

### F4.6 — Análisis de errores (§7, «Concordancia»; plan F4.6)

| Definición (cita) | Insumos | Procedimiento | Implementación | Marca |
|---|---|---|---|---|
| §7: «Kappa de Cohen entre BETO, techo 5D y visual; intersección de errores (F4.6)». «Para los contrastes que requieren una predicción por muestra (McNemar, Cochran, kappa e intersección de errores), se usa la media de `p_dep` sobre las cinco semillas, con ŷ = 1 si la media es > 0.5». Plan: «Kappa de Cohen entre vectores de acierto de BETO, techo 5D y finalista visual; intersección de errores». §9: explicativo; «pruebas de hipótesis … con Holm dentro de cada análisis» | BETO, techo 5D (SVM-RBF) y finalistas: existen | Predicción visual por muestra = 1[media de `p_dep` de 5 semillas > 0.5]; una media de exactamente 0.5 da ŷ = 0. BETO y techo: su propio `p_dep` > 0.5. Kappa por pares (3 pares por técnica) | No implementado | **A10 AMBIGÜEDAD**: kappa sobre ŷ (protocolo, sin objeto explícito) o sobre vectores de acierto (plan); IC o prueba de kappa sin fijar |

### F5.3 — Descomposición de la pérdida (§7) y contraste frente a descriptores (§7)

| Definición (cita) | Insumos | Procedimiento | Implementación | Marca |
|---|---|---|---|---|
| §7: «Tres etapas, con intervalo percentil al 95 % sobre las réplicas de §5.2: cabeza (F1_BETO − F1_768D), reducción (F1_768D − F1_5D) y renderizado (F1_5D − $\overline{F1}_{img,t}$). Su suma es F1_BETO − $\overline{F1}_{img,t}$» | BETO, 768D, techo 5D (SVM-RBF) y finalistas: existen | **Mismas réplicas de §5.2** (revisión d: sí, explícito). F1_img\* = media de 5 semillas. Percentil al 95 % | No implementado | **A5**: lado y cuantiles. F1_5D = techo 5D (§4.2), no ambiguo |
| §7: «$\overline{F1}_{img,t} - F1_{descriptores,t}$ con intervalo al 95 %» | Descriptores ×3: existen | Nivel 95 %; ni tipo de intervalo ni réplicas escritos (aunque el §5.2 incluye «el de cada referencia» en cada réplica) | No implementado | **A11 AMBIGÜEDAD**: mismas réplicas y percentil, o no. Asignación: el plan no le da tarea; aquí va con F5.3 |

### F5.4 — Comparación entre técnicas y frente a BETO y al techo (§7)

| Definición (cita) | Insumos | Procedimiento | Implementación | Marca |
|---|---|---|---|---|
| §7: «Q de Cochran sobre el acierto por muestra de los tres finalistas; si p < 0.05, McNemar exacto por pares con Holm (3 pares)» | Finalistas | Acierto por muestra con la media de `p_dep` de las 5 semillas > 0.5 (revisión c). Cochran con 3 grupos y 114 bloques; condicional en p < 0.05 | No implementado | **A12 AMBIGÜEDAD**: McNemar exacto bilateral o unilateral; Cochran asintótico (χ², gl = 2) o exacto |
| §7: «McNemar exacto por técnica, con Holm dentro de cada familia de tres. Es complementario: su no significancia no se interpreta como equivalencia» | Finalistas, BETO y techo 5D | Dos familias (frente a BETO y frente al techo), con m = 3 cada una | No implementado | A12 |

### F5.5 — Análisis de sensibilidad (§8), uno por variante

En todas, «Se reporta el veredicto de cada técnica bajo cada variante, sin modificar el veredicto principal». Ninguna variante está implementada; `verdict.verdicts` acepta θ\* y D\* arbitrarios, así que sirve de núcleo si se le entregan réplicas modificadas. Revisión (f), computabilidad:

| Variante (cita §8) | Insumos | Procedimiento | ¿Computable tal como está escrito? | Marca |
|---|---|---|---|---|
| «Retención exigida r ∈ {0.75, 0.80}» | Los de F5.1 y F5.2 | §5 completo con r sustituido en θ. Superioridad sobre el piso sin cambio. IC al nivel 1 − 2α_h | **Sí** | — |
| «Formulación A: margen fijo δ = (1 − r)(0.7039 − 0.5), con $\Delta_t = F1_{BETO} - \overline{F1}_{img,t}$» | Los de F5.2. δ canónicos: `f11.delta_margen.r0_50`, `r0_75` y `r0_80` (0.1019, 0.0510 y 0.0408) | δ fijo, con F1_BETO observado. H₀: Δ ≥ δ por implicación. Falta escribir p y la regla equivalente a V2 | **Parcial.** Si se pasa δ − Δ\* como «θ\*» a `verdicts`, queda p = #{Δ\* ≥ δ}/B y V2 ⇔ límite inferior de Δ > δ. Es una lectura, no texto | **A13 AMBIGÜEDAD**: hipótesis, p y regla V2 sin escribir; ¿qué r? (`canonical_delta` lista 0.5, 0.75 y 0.8) |
| «Por semilla: veredicto con cada semilla por separado; se reporta cuántas de las cinco dan V1» | Los de F5.1 y F5.2 | F1_img\* = F1 de la semilla s. ¿Holm entre técnicas por semilla? | **Parcial** | **A14 AMBIGÜEDAD**: piso permutado (media de 5 o la semilla emparejada); Holm por semilla |
| «Variabilidad de BETO: F1_BETO desplazado ±0.0128» (`f12.variabilidad_beto`, sin redondear; Enmienda 1) | Los de F5.2 | ¿Se suma ±0.0128 a F1_BETO observado y a cada F1_BETO\*? Si es así, θ\* se traslada ∓r·0.0128; D no cambia | **Parcial** | **A15 AMBIGÜEDAD**: desplazamiento en las réplicas o solo en el punto |
| «Segundo mejor modelo … cinco semillas … se aplica de nuevo la regla completa de §5 de la matriz» | `sensibilidad/*` (existen); BETO; pisos | §5 completo con F1_img del segundo | **Parcial**: no hay control permutado del segundo (F3.6 solo entrenó finalistas; matriz §7) | **A17 AMBIGÜEDAD**: qué piso permutado usar |
| «Piso sin el visual permutado: Piso = máximo de mayoritaria, estratificado y brillo» | Mayoritaria, brillo y estratificado = 0.5 | §5 completo con el piso reducido | **Sí**, salvo lo que herede de A2 y A3 | Hereda A2 y A3 |

### Revisiones explícitas

- **(a) Piso.** El texto no fija si el máximo se recalcula en cada réplica.
  - §4.4 define el piso como el de mayor F1 de prueba entre los cuatro, que es un valor observado.
  - §5.3 habla de «el piso» en singular y da reglas según cuál sea («Si el piso es el aleatorio estratificado… Si es el visual permutado…»). Eso presupone una identidad única, fijada sobre el observado y remuestreada después.
  - §5.2 dice que «se recalculan … el de cada piso», lo que permite cualquiera de las dos lecturas.
  - `configs/protocolo.json` solo dice `rule: "max_f1_test"`.
  - `verdict.py` recibe D\* ya calculado y no decide. → **A2**.
- **(b) Tipo de IC y lado.**
  - θ: percentil bilateral al nivel 1 − 2α_h, con cuantiles `linear` (fijado en texto, config y código).
  - D_t: no se fija si lleva intervalo (A4).
  - ρ̂, descomposición y descriptores: «percentil al 95 %» sin lado ni método de cuantiles; para descriptores ni siquiera «percentil» (A5, A11).
  - Las pruebas son unilaterales (H₁ > 0). BCa no aparece en ninguna parte.
- **(c) Cochran y McNemar con 5 semillas.** Fijado en §7: media de `p_dep` sobre las cinco semillas, ŷ = 1 si la media es > 0.5. Las referencias de una sola semilla usan su `p_dep`. No es ambiguo.
- **(d) ¿Los IC de F5.3 usan el mismo *bootstrap*?** La descomposición sí, de forma explícita («sobre las réplicas de §5.2»). El contraste frente a descriptores no lo dice (A11).
- **(e) F4.5.**
  - Estadístico: L_t = F1_img − F1_longitud, unilateral, Holm sobre las técnicas con V1, V2 o V4, α = 0.05.
  - Regla: si no se rechaza H₀, se añade la calificación al veredicto.
  - Efecto: no cambia V1–V4 ni G1–G4; solo califica el texto del veredicto de cada técnica evaluada.
  - Las técnicas con V3 no se evalúan.
  - Pendiente: A7 y A9, y la referencia de longitud, que hay que ajustar.
- **(f) Computabilidad de §8.** Computables tal como están escritas: r ∈ {0.75, 0.80} y el piso sin permutado. Las otras cuatro necesitan una lectura (A13, A14, A15, A17).
- **(g) `git diff protocolo-v1 HEAD -- src/protocol/`:** **vacío**. Tampoco cambió `tests/test_verdict.py`.
  - `configs/protocolo.json` sí cambió (Enmienda 2, `41dd200`, y F2.4, `8d7fbb2`). Por bloque: `contraste`, `longitud`, `secundarios`, `sensibilidad`, `pisos`, `modelos`, `techo_5d` y `cifras_diseno` están **iguales**; `descriptores`, `piso_brillo` y `pisos_triviales` son **nuevos**; `referencias.longitud` está **ampliado** (vocabulario `text_length` y rejilla completa).
  - `verdict.load_contrast` solo lee `contraste`, que no cambió. Por tanto `verdict.py` en `protocolo-v1` y en HEAD aplican la misma regla.
  - El texto de §1–§13 del protocolo no cambió desde la etiqueta: solo el encabezado y las enmiendas añadidas al final (`git_estado.log`).

**Discrepancias entre el código y el texto del protocolo: ninguna** en lo que `verdict.py` implementa. La regla de empates de Holm, el rechazo escalonado, el orden V3 → V1 → V2 → V4, la fórmula de p y el nivel del IC coinciden con §5.2–§5.5. Lo único que no se exige es B = 10,000: `verdicts()` solo comprueba B si se le pasa (`B=None` por omisión). No es una discrepancia, pero quien lo llame debe pasar `B=10000`.

---

## Comandos ejecutados

```text
git fetch -q; git log/branch/tag/merge-base/diff …            → git_estado.log
python results/f5_preparacion/verificar_45.py                 → verificar_45.log, f1_recalculado_45.csv
python results/f5_preparacion/consultas_test.py               → consultas_test.log, consultas_test.csv
python results/f5_preparacion/verificar_reproducibilidad.py   → verificar_reproducibilidad.log
python -m src.predictions.validate "results/predictions/final/*.csv" \
    "results/predictions/control_permutacion/*.csv" "results/predictions/sensibilidad/*.csv"   → validate.log
python -m pytest tests/test_verdict.py tests/test_protocolo.py tests/test_predictions_format.py \
    tests/test_select_finalist.py -q                          → pytest.log (110 passed)
(lectura de split de referencia/ y selecciones de F2.1/F2.4)  → insumos_referencia.log
Comprobación de numpy 2.1.3: equivalencia de integers/choice/bucle con default_rng(2026) (A1)
```

Todo con `.venv/Scripts/python.exe` y `PYTHONIOENCODING=utf-8`, en CPU local y en segundos.

---

## Discrepancias

| ID | Dónde | Detalle | Efecto |
|---|---|---|---|
| D1 | `docs/bitacora/F3.5.md` §4 | Dice que `summary_by_config.csv` usa la DE poblacional en val; el código y los valores muestran `ddof = 1` | Ninguno en cifras; corregir la bitácora (CE) |
| D2 | `docs/bitacora/F3.5.md` §3.1, paso 3 | «predicciones = `probs ≥ 0.5`» frente a `p_dep > 0.5` en código, F1.5 y protocolo | Ninguno: no hay `p_dep` = 0.5 en las 45 corridas |
| — | Plan v2, F5.1 («pruebas de permutación») y F5.5 («δ ∈ {0.03, 0.058, 0.08}») frente al protocolo | Ya declaradas en la Enmienda 1, punto 3 | Ninguno; se citan para trazabilidad |

## Ambigüedades y la opción que dicta cada regla

(1) texto del protocolo · (2) `verdict.py` en `protocolo-v1` (idéntico en HEAD) · (3) opción menos favorable a la hipótesis de conservación. No se elige ninguna.

| ID | Ambigüedad | (1) Texto del protocolo | (2) `verdict.py` en `protocolo-v1` | (3) Menos favorable a la conservación |
|---|---|---|---|---|
| A1 | Forma exacta de generar las réplicas con `default_rng(2026)` y orden de los 114 índices | «B = 10,000 réplicas de los 114 índices de prueba con reemplazo, generadas con `numpy.random.default_rng(2026)`»; no fija la llamada ni el orden. `integers(0,114,(B,114))`, `choice` y el bucle coinciden; `(114,B).T` no | No genera réplicas | No aplica, porque no hay dirección conocida sin consultar prueba. Precedente: F1.1, `rng.integers(0, n, size=(B, n))`, con los índices en orden de `sample_id` |
| A2 | Piso: máximo fijado sobre el observado o recalculado en cada réplica | §4.4 («el de mayor F1-Macro de prueba») y §5.3 (singular, «Si el piso es …») → identidad fijada sobre el observado y remuestreada. §5.2 no lo excluye | No lo decide: recibe D\* | **Máximo recalculado en cada réplica**: D\*_máx ≤ D\*_fijo en toda réplica |
| A3 | Empate en el máximo observado entre pisos | No lo fija. `configs/protocolo.json` → `pisos.order` lista un orden, pero no como desempate | No lo decide | Entre los empatados, el que dé el D\* menor; equivale a A2 (3) |
| A4 | IC de D_t: si se reporta y a qué nivel | §5.4: «El intervalo de cada técnica…» sin estadístico; config: `alpha_h_from = no_inferioridad` → solo θ | Solo calcula el IC de θ; ninguno para D | No afecta al veredicto (D usa solo p). Si se reporta: el más ancho, 1 − 2α_h con α_h de la familia de superioridad |
| A5 | Lado y cuantiles de los IC al 95 % (ρ̂ §5.6, descomposición §7) | «intervalo percentil al 95 %»; lado y método no escritos | No implementado; `percentile_interval(reps, 0.025)` daría bilateral con cuantiles `linear` | Descriptivo: no afecta al veredicto. El más conservador para afirmar retención es el bilateral 2.5–97.5 |
| A6 | ρ̂ con F1_BETO\* ≤ 0.5 en alguna réplica (denominador ≤ 0) | No lo fija | No implementado | Tratar esas réplicas como ρ̂\* mínima (−∞), lo que baja el límite inferior |
| A7 | F4.5: fórmula de p, tamaño de la familia de Holm y si aplica a las variantes de §8 | «H₁: L_t > 0, sobre las réplicas de §5.2 y con Holm sobre las técnicas evaluadas, α = 0.05» → p = #{L\* ≤ 0}/B por analogía; m = número de técnicas con V1, V2 o V4; solo veredicto principal | No implementado (`holm` reutilizable) | Holm con m = 3 siempre, y calificación aplicada también a cada variante de §8 |
| A8 | F4.5 descriptivo: cuantiles y empates en el corte (3 muestras de `train` en 381); F1 por tercil con 5 semillas; vigencia de las correlaciones con **u** y de los terciles de BETO del plan | «F1-Macro del finalista por terciles de `n_tokens_beto`, con los puntos de corte calculados en `train`»; sin método, sin regla de semillas, y sin BETO ni **u** (el plan sí los pide; §9 no lista F4.5) | No implementado | No aplica (descriptivo) |
| A9 | Ajuste del clasificador de longitud: `max_iter` y codificación de `truncated_128` | Enmienda 2, punto 4 y config fijan la rejilla, el escalador, la regla y `seed = 42`, pero no `max_iter`. La base del log es irrelevante tras `StandardScaler` | No aplica | Sin dirección conocida sin consultar prueba. Precedente: F2.1/F2.3, `max_iter = 5000`, booleano → {0, 1} |
| A10 | F4.6: kappa sobre ŷ o sobre vectores de acierto; IC o prueba | §7: «Kappa de Cohen entre BETO, techo 5D y visual» (objeto no escrito). Plan: «vectores de acierto» | No implementado | No aplica (explicativo) |
| A11 | IC frente a descriptores: mismas réplicas y percentil, o no | «con intervalo al 95 %»; §5.2 incluye «el de cada referencia» en cada réplica | No implementado | No aplica al veredicto. Si se reporta, bilateral 2.5–97.5 sobre las réplicas de §5.2 |
| A12 | McNemar exacto bilateral o unilateral; Cochran asintótico o exacto | «McNemar exacto», «Q de Cochran», sin lado ni variante | No implementado | No modifica veredictos. Frente a BETO, el unilateral (visual < BETO) detecta antes la pérdida |
| A13 | Formulación A: H₀, p y regla equivalente a V2; valores de r | δ = (1 − r)(0.7039 − 0.5) fijo, Δ_t = F1_BETO − F1_img; ni p ni V2 escritos. Config: δ para r ∈ {0.5, 0.75, 0.8} | Reutilizable pasando δ − Δ\* como θ\* (V2 ⇔ límite inferior de Δ > δ) | Evaluarla con r ∈ {0.5, 0.75, 0.8} y con δ fijo, que no reduce la variabilidad de BETO |
| A14 | Por semilla: piso permutado (media de 5 o la semilla emparejada) y Holm por semilla | «Veredicto con cada semilla por separado»; §4.4 define el piso permutado como «media de las 5 semillas» → media; Holm como en §5 | No lo decide | En cada réplica, el mayor de los dos pisos (p. ej., Gray-Scott s2024 permutado = 0.7185 frente a una media de 0.527) |
| A15 | BETO ±0.0128: desplazamiento en las réplicas o solo en el punto | «F1_BETO desplazado ±0.0128» sin más; la coherencia con §5.2 sugiere trasladar también las réplicas | No lo decide | +0.0128 aplicado al observado y a cada réplica |
| A16 | ¿Las métricas de prueba del piso visual permutado quedan reservadas a la Fase 5? (F3.6 ya las publicó) | §10, fila 2: F3.6 consulta prueba con «una inferencia por modelo y semilla»; fila 3: «pisos → Métricas de prueba solo en la Fase 5» | No aplica | Tratar la publicación de F3.6 como desviación declarada. No cambia ninguna decisión: nada se modificó después |
| A17 | Segundo mejor modelo: qué piso permutado usar si no existe control del segundo | No lo fija. El §4.4 define el piso permutado como «Finalista entrenado con `y_train` permutado» → el del finalista de la técnica, o sin piso permutado | No lo decide | Incluir el piso permutado del finalista de la técnica, porque el máximo no puede bajar |

**Notas sin marca:** la asignación entre el plan y el protocolo es solo de nombres (F5.6 del plan = §5.5; §5.6 = ρ̂; el contraste frente a descriptores y McNemar frente a BETO y al techo no tienen tarea propia en el plan). Tampoco tienen marca el estimador puntual de θ (fila de F5.2) ni la nueva pasada de F3.7 sobre prueba para los mapas, sin predicciones archivadas (punto 1h).
