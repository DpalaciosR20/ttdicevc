# Bloque D, etapa 1 — Pipeline de la Fase 5 y pruebas con datos sintéticos

> Rol: SV · Fecha: 2026-11-07 · Rama `sv/f5-ejecucion`, creada desde `main` @ `1020830` (merge del PR #35, bloque C).
> **No se ejecutó ningún cálculo sobre las predicciones reales de prueba.** Con datos reales solo se comprobó presencia y alineación (`--verificar-insumos`), sin cargar `p_dep`.
> `src/protocol/verdict.py` no se modificó.

## 0. Compuerta

| Comprobación | Resultado |
|---|---|
| `git merge-base --is-ancestor enmienda-4 HEAD` | 0 |
| `git diff --stat enmienda-4 HEAD -- docs/protocolo_preregistrado.md configs/protocolo.json docs/plan_fase4.md` | vacío |
| Lo mismo frente al árbol de trabajo | vacío |

## 1. Mapa de funciones → protocolo y Enmienda 4

Todos los parámetros se leen de `configs/protocolo.json` (`contraste`, `pisos`, `longitud`, `secundarios`, `sensibilidad`, `enmienda_4`). Cada función comprueba que el bloque que lee pide lo que implementa y, si no, falla.

### `src/protocol/bootstrap.py`

| Función | Protocolo / Enmienda 4 |
|---|---|
| `bootstrap_spec`, `indices` | §5.2; **A1**. Reconstruye la llamada textual `rng.integers(0, 114, size=(10000, 114))` y exige que sea igual a `enmienda_4.bootstrap.call`; además seed = `contraste.bootstrap.seed`, B = `contraste.bootstrap.B` y `row_order = sample_id` |
| `weights` | Matriz de multiplicidades B × n (conteos por réplica) |
| `f1_macro`, `f1_macro_from_counts` | F1-Macro desde conteos ponderados; `labels=[0, 1]`, `zero_division=0`, como sklearn |
| `f1_macro_seeds_mean` | §4.1, §5.2: media de los F1 por semilla en cada réplica |
| `predict` | §4.1: ŷ = 1 ⇔ `p_dep > modelos.decision_threshold` |
| `percentile_interval` | A4, A5, A10, A11 (cuantiles `linear`). Sin −inf es exactamente `np.quantile(reps, [lo, hi])`, como `verdict.percentile_interval`. Con −inf (**A6**), el cuantil es −inf si el estadístico de orden inferior que interpola es −inf |

### `src/protocol/contraste.py`

| Función | Protocolo / Enmienda 4 |
|---|---|
| `f1_table`, `img`, `ref` | F1 observado y B réplicas por clasificador |
| `theta_reps` | §5.1–§5.2 (θ) |
| `formulation_a_reps` | §8; **A13**. Formulación A como δ − Δ\*: así `verdict.verdicts` da p = #{Δ\* ≥ δ}/B, y V2 ⇔ límite inferior de Δ > δ (comprobado en las pruebas) |
| `floor_candidates` | §4.4, §5.3. Mayoritaria; estratificado = `stratified_as_constant` (0.5) constante; brillo; permutado = media de 5 semillas por réplica. Orden `pisos.order` |
| `d_reps` | §5.3. **A2**: identidad del máximo observado, remuestreada; sensibilidad `max_por_replica`. **A3**: empate exacto en el máximo observado → el menor D\* por réplica |
| `veredicto`, `contraste` | §5.2–§5.5 vía `verdict.verdicts(…, B=10000)`. IC de D (**A4**): percentil bilateral 1 − 2α_h con α_h de la familia de superioridad |
| `retencion` | §5.6; **A5** y **A6** (ρ̂\* = −inf si F1_BETO\* ≤ 0.5, con conteo) |
| `descomposicion` | §7 (cabeza, reducción, renderizado); **A5** |
| `frente_a_descriptores` | §7; **A11** |
| `contraste_longitud` | §6; **A7**. Solo técnicas con V1, V2 o V4; m = su número; `m3` como sensibilidad |
| `yhat_mean` | §7: media de `p_dep` de las 5 semillas > 0.5 |
| `cochran_q`, `mcnemar_bilateral`, `mcnemar_unilateral_more_errors`, `comparaciones_mcnemar` | §7; **A12**. Cochran χ² con gl = 2; si p < α, McNemar exacto bilateral por pares con Holm (3 pares); unilateral frente a BETO y al techo, con Holm en cada familia de tres; sensibilidad bilateral |
| `kappa_weighted`, `kappa` | §7, F4.6; **A10** (ŷ y aciertos, IC 95 % bilateral `linear`) |
| `contencion` | F4.6 (`docs/plan_fase4.md`): C, referencia nula \|E_ref\|/n, IC 95 %; frente al techo, a BETO y al 768D |
| `tercile_cuts`, `assign_terciles`, `terciles_longitud` | §6 descriptivo; **A8** (cortes en train con `linear`; T1 ≤ c1 < T2 ≤ c2 < T3; F1 por semilla y luego la media; también BETO) |
| `spearman_u_longitud` | **A8**: Spearman de u0–u4 con ln(`n_tokens_beto`) y `truncated_128` en train |

### `src/protocol/sensibilidad.py`

| Función | Protocolo / Enmienda 4 |
|---|---|
| `retencion_exigida` | §8: r ∈ {0.75, 0.80} |
| `formulacion_a` | §8; **A13**: r ∈ {0.5, 0.75, 0.8}, δ fijo de `f11.delta_margen.*` |
| `por_semilla` | §8; **A14**: piso permutado = media de 5; Holm entre técnicas dentro de cada semilla; `n_semillas_V1` |
| `variabilidad_beto` | §8; **A15**: ± `f12.variabilidad_beto` en el observado y en cada réplica |
| `segundo_mejor` | §8; **A17**: fase `sensibilidad` con el piso permutado del finalista |
| `piso_sin_permutado` | §8 |
| `robustez` | Enmienda 4, relación con el §11. Alternativas: A2 máximo por réplica; A3; A7 con m = 3 y en cada variante; A12 bilateral; A14 semilla emparejada |

### `src/protocol/insumos.py` y `src/protocol/run_fase5.py`

| Función | Qué hace |
|---|---|
| `insumos.verificar_insumos` | Alineación sin `p_dep`: `usecols` = `sample_id`, `split`, `y_true` e identidad; `csv_sha256` frente al manifiesto; u de train alineada por `idx` |
| `insumos.cargar` | Carga `p_dep` con `validate_predictions`. Solo se llama desde `run_fase5 --ejecutar-prueba` / `--analisis-f46` |
| `run_fase5.main` | Sin bandera, solo `--verificar-insumos`. `--ejecutar-prueba` exige: compuerta, árbol limpio, ningún PENDIENTE en el config y que no exista `resultados.json`. `--analisis-f46` exige además que `resultados.json` esté versionado y sin cambios (G1 de `docs/plan_fase4.md`) y que no exista `f46_resultados.json` |
| `run_fase5.ejecutar` / `ejecutar_f46` | Cálculo completo sobre insumos ya cargados, probado con insumos sintéticos |
| `run_fase5.write_outputs` | `resultados.json` en JSON estricto (−inf como `"-inf"`), CSV por análisis y `run_fase5.log` con los hashes de los insumos, el commit, la etiqueta y la huella de los índices |

## 2. Pruebas con datos sintéticos

`tests/fase5_sintetico.py` genera insumos sintéticos del tamaño real:
- n = 114 (70/44);
- 3 técnicas × 3 fases × 5 semillas;
- referencias;
- train de 908 muestras con `n_tokens_beto`, `truncated_128` y u.

Ninguna prueba lee predicciones reales. Las de `run_fase5` sustituyen `cargar()` y git por dobles controlados.

**44 aprobadas** (`pytest_fase5_d1.log`):

| Archivo | Prueba | Qué comprueba |
|---|---|---|
| `test_fase5_bootstrap.py` | `test_f1_vectorizado_igual_a_sklearn_replica_a_replica` | 250 réplicas × 5 clasificadores, incluidos uno que nunca predice 1, uno que nunca predice 0 y uno perfecto, frente a `sklearn.f1_score(average='macro', labels=[0,1], zero_division=0)` |
| | `test_f1_replica_sin_una_clase_verdadera` | Réplica degenerada |
| | `test_f1_semillas_media_de_f1_por_semilla` | Media de 5 semillas |
| | `test_indices_forma_y_llamada_del_config` | (10000, 114); igual a `default_rng(2026).integers(…)`; igual a `rng.choice`; reproducible |
| | `test_indices_rechaza_config_distinto` | Llamada, semilla, tamaño u orden distintos ⇒ error |
| | `test_pesos`, `test_umbral_estricto`, `test_percentil_con_menos_infinito` | — |
| `test_fase5_contraste.py` | `test_clasificador_perfecto_V1` | — |
| | `test_identico_a_beto_V1_si_beto_supera_el_umbral` | θ = (1 − r)(F1_BETO − 0.5) |
| | `test_azar_V3` | — |
| | `test_V2_V3_V4` (30/40/48 errores con BETO casi perfecto) | V4, V2 y V3 |
| | `test_theta_y_D_por_replica` | — |
| | `test_piso_identidad_fija_frente_a_maximo_por_replica` | — |
| | `test_empate_A3_menor_D` | — |
| | `test_estratificado_constante_y_permutado_promediado` | — |
| | `test_piso_estratificado_como_identidad` | — |
| | `test_ic_de_D_con_alpha_h_de_superioridad` | — |
| | `test_retencion_menos_infinito_y_conteo` | 300 réplicas con F1_BETO\* ≤ 0.5 ⇒ conteo 300 y límite −inf |
| | `test_descomposicion_suma_y_descriptores` | — |
| | `test_longitud_solo_V1_V2_V4_y_m` | — |
| | `test_mcnemar_y_cochran_contra_statsmodels_y_scipy` | 30 casos frente a `statsmodels.mcnemar(exact=True)`, `scipy.binomtest(alternative='greater')` y `statsmodels.cochrans_q` |
| | `test_comparaciones_mcnemar_lados_y_holm`, `test_yhat_media_de_p_dep` | — |
| | `test_kappa_contra_sklearn` | Observado y 200 réplicas frente a `cohen_kappa_score` sobre filas remuestreadas |
| | `test_kappa_y_contencion_f46` | — |
| | `test_terciles_cortes_y_asignacion`, `test_spearman_u_longitud` | — |
| | `test_formulacion_A_p_y_V2` | p = #{Δ\* ≥ δ}/B; límite inferior de Δ = cuantil α_h; V2 ⇒ límite inferior > δ |
| | `test_beto_desplazado_traslada_theta` | Δθ = ∓r·0.0128 |
| | `test_por_semilla_y_segundo_mejor_y_piso_sin_permutado` | — |
| | `test_A14_emparejada_es_pendiente_sin_comparacion` | — |
| | `test_tabla_de_robustez` | — |
| `test_fase5_run.py` | `test_sin_bandera_solo_verifica` | No llama a `cargar()` ni escribe nada |
| | `test_se_niega_sin_compuerta_o_con_arbol_sucio` (×3) | Sin la etiqueta como ancestro, con archivos congelados cambiados o con árbol sucio |
| | `test_se_niega_con_pendientes` | — |
| | `test_config_real_tiene_pendientes` | — |
| | `test_ejecucion_sintetica_y_no_sobrescribe` | JSON estricto, 9 CSV y log; la segunda ejecución se niega y no cambia el archivo |
| | `test_f46_exige_veredicto_commiteado` | — |
| | `test_rendimiento_B10000_tamano_real` | — |

**Suite completa** (sin `test_label_independence.py`, con `CANON_*`): **1058 aprobadas** en 138 s (`pytest_bloque_d1.log`).

## 3. Tiempo de ejecución sintético

Con B = 10,000 sobre insumos sintéticos del tamaño real, en local con CPU (`pytest_fase5_d1.log`, `test_rendimiento_B10000_tamano_real`):

| Cálculo | Tiempo |
|---|---|
| Fase 5 completa (principal, descriptivos, longitud, McNemar/Cochran, las 6 familias de sensibilidad, robustez, terciles y Spearman) | **0.2 s** |
| F4.6 (kappa y contención) | 0.1 s |

Los índices de A1 y la carga de los insumos no entran en el tiempo; cada uno tarda menos de un segundo.

## 4. `--verificar-insumos` sobre los datos reales

Archivo: `verificar_insumos_d1.log`.

```text
run_fase5 — solo --verificar-insumos (sin --ejecutar-prueba no se carga p_dep)
— Corridas visuales (45)
  final: perlin vit_b16/frozen, gray_scott efficientnet_b0/finetune, lorenz vit_b16/frozen
  control_permutacion: perlin vit_b16/frozen, gray_scott efficientnet_b0/finetune, lorenz vit_b16/frozen
  sensibilidad: perlin efficientnet_b0/frozen, gray_scott resnet18/finetune, lorenz efficientnet_b0/finetune
  OK … (56 archivos: 45 visuales + BETO, 768D, techo 5D (umap5d__svm_rbf), mayoritaria, longitud, brillo ×3, descriptores ×3)
— Prueba: n = 114, soportes 0/1 = 70/44; filas ordenadas por sample_id (test_0000 … test_0113)
— u de train (umap_params.pkl): (908, 5), etiquetas alineadas con el índice por idx
```

Además se probó `python -m src.protocol.run_fase5 --ejecutar-prueba` con el árbol sucio. Se negó en la comprobación de árbol limpio, antes de cargar ningún insumo; no existe `results/f5_contraste/`.

## 5. Puntos todavía no computables (PENDIENTE, no resueltos)

| ID | Punto | Comportamiento del código |
|---|---|---|
| **P1** | **A14, sensibilidad.** «Piso permutado de la semilla emparejada, si es mayor» no fija frente a qué se compara «mayor»: el F1 observado (y luego se remuestrea esa identidad, como en A2) o cada réplica (máximo por réplica) | `run_fase5 --ejecutar-prueba` **se niega** mientras falte `enmienda_4.por_semilla.sensibilidad.comparacion` ∈ {`observado`, `por_replica`}. El config congelado no la tiene (`test_config_real_tiene_pendientes`). Se ejecuta en cuanto una enmienda la fije |
| **P2** | **A3, alternativa** de la tabla de robustez. La Enmienda 4 no declara alternativa para A3 | Si no hay empate en el máximo observado, la fila dice «A3 no aplica». Si lo hay, `PendienteError` |
| **P3** | **Kappa indefinida** (1 − p_e = 0, ambos vectores constantes) en el observado o en alguna réplica (A10 no lo trata) | `PendienteError` si ocurre. Con clasificadores no constantes es prácticamente imposible |
| **P4** | **Contención indefinida** (\|E_vis\|\* = 0 en el observado o en una réplica) | `PendienteError` si ocurre |
| **P5** | **Q de Cochran indefinida** (0/0 si todos los bloques son concordantes) | `PendienteError` si ocurre |
| **P6** | **F4.6, frase «consistente con que el visual hereda los errores de u».** La segunda condición (IC de ΔF1 de F4.1 contiene 0) depende de F4.1, que no existe | Se reporta la condición 1 y la 2 como `'PENDIENTE: requiere el IC de ΔF1 de F4.1'` |

## 6. Decisiones de implementación (para revisión de SV)

No son ambigüedades del protocolo, pero conviene que SV las confirme antes de la etapa 2:

1. **Momento de F4.6.** El §7 incluye la kappa y la intersección de errores entre los secundarios, pero las etiqueta como F4.6. `docs/plan_fase4.md` (G1) solo permite usar prueba en la Fase 4 después del commit del veredicto. Por eso kappa y contención van en `--analisis-f46`, que exige `resultados.json` versionado. Todo lo demás del §7 (descomposición, descriptores, Cochran y McNemar) va en `--ejecutar-prueba`.
2. **Empate de A3.** Se considera empate la igualdad exacta del F1 observado (la Enmienda 4 no da tolerancia).
3. **α de McNemar frente a BETO y al techo.** `contraste.multiplicidad.alpha` (0.05; §13: «α 0.05, unilateral, Holm por familia»). Entre técnicas se usa `secundarios.entre_tecnicas.alpha`.
4. **Pares de kappa.** Todos los pares entre {BETO, 768D, techo 5D} y cada visual, más los pares entre referencias (6 por técnica; plan_fase4 F4.6 incluye el 768D). No se calculan pares entre técnicas visuales.
5. **Niveles de los IC al 95 %.** Se calculan con `Decimal` (0.025 y 0.975 exactos). En coma flotante, (1 − 0.95)/2 = 0.025000000000000022.
6. **Cuantil con −inf (A6).** Si el estadístico de orden inferior que interpola el cuantil es −inf, el límite es −inf. Es el límite de la interpolación lineal; numpy devolvería nan.
7. **Robustez de A12.** El §7 dice que McNemar no modifica veredictos, así que la fila de A12 informa si cambia el rechazo de McNemar, no el veredicto.

## 7. Commit y PR

Ver el reporte de cierre en el PR. El commit `[F5] Pipeline de la Fase 5 y pruebas sintéticas` incluye `src/protocol/{bootstrap,contraste,insumos,sensibilidad,run_fase5}.py`, `tests/fase5_sintetico.py`, `tests/test_fase5_{bootstrap,contraste,run}.py` y este reporte con sus logs.
