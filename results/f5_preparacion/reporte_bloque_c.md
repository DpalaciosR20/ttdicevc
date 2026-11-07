# Bloque C de la Fase 5 — Referencia de longitud (F4.5) y colectores canónicos de la Fase 3

> Rol: SV · Fecha: 2026-11-07 · Rama `sv/f5-contraste` (avanzada por *fast-forward* desde `main` @ `5c17047`, merge del PR #34).
> **No se calculó ninguna métrica de prueba de referencias**, incluida la de longitud, ni ningún estadístico de la Fase 5 (θ, D, L, ρ̂, IC, Holm, piso, veredicto, Cochran, McNemar, kappa). No se fusionó nada ni se creó ninguna etiqueta.

## 0. Compuerta

| Comprobación | Resultado |
|---|---|
| `git merge-base --is-ancestor enmienda-4 HEAD` | **0** |
| Etiqueta `enmienda-4` | Anotada el 2026-11-07 00:30:09 −0600; apunta a `5c17047` |
| `git diff --stat enmienda-4 -- docs/protocolo_preregistrado.md configs/protocolo.json docs/plan_fase4.md` | **Vacío** |

**Compuerta superada.**

## 1. Referencia de longitud

**Código.** `results/f45_longitud/f45_longitud.py` reutiliza `logreg_grid`, `p_dep_of`, `f1_val_of`, `select`, `n_iter_of`, `load_index`, `Log` y `text_sha256` de `results/f23_pisos/f23_pisos.py`, que son las mismas funciones de F2.1. Las carga con `importlib` y no reimplementa nada.

**Parámetros.** Todos se leen de `configs/protocolo.json`:
- `referencias.longitud`: características, escalador, identidad, rejilla `techo_5d.familias.logreg` y regla `techo_5d.selection`;
- `enmienda_4.longitud.clasificador`: `max_iter`, logaritmo, codificación de `truncated_128`, splits.

El script se detiene si el config pide algo distinto de lo implementado; por ejemplo, si `max_iter` difiriera del de F2.1/F2.3.

**Resultado** (`results/f45_longitud/f45_longitud.log`, `resultado.json`):

| Campo | Valor |
|---|---|
| Configuración elegida | #15 de 26: `{"C": 0.31622776601683794, "class_weight": "balanced"}`; fijos `penalty=l2`, `solver=lbfgs`, `max_iter=5000`, `random_state=42` |
| F1-Val | **0.5525902668759812** (única combinación dentro de la tolerancia 1e-12) |
| Avisos de convergencia | 0 |
| Splits evaluados | solo `val` |

**Validación del CSV.**
- `python -m src.predictions.validate results/predictions/referencia/text_length__logreg__none__seed42.csv` → **1/1 válido** (`results/f45_longitud/validate.log`). Con todo `referencia/`, 16/16.
- Contenido: 114 filas de val y 114 de test; `technique=text_length`, `arch=logreg`, `strategy=none`, `seed=42`, fase `referencia`.
- Manifiesto: `task_id=F4.5`, `git_commit=ef9bae0…` sin `-dirty`.

**Prueba sin métricas de test** (`tests/test_f45_longitud.py`, 7 pruebas). Comprueba que:
- `resultado.json`, el manifiesto (`model_selection`, `config`) y `rejilla_longitud.csv` no tienen ninguna clave con «test» con valor;
- el log no contiene `F1-Test`, `f1_test` ni `test = <número>`, y toda línea con «F1» es de Val;
- el código fuente no pasa `y['test']` a ninguna métrica;
- además: CSV válido, F1-Val reproducible desde el CSV, configuración igual a la del protocolo y características sobre datos sintéticos.

**Evidencia y bitácora.** `results/f45_longitud/` (con `README.md`) y `docs/bitacora/F4.5-referencia.md`, con la estructura de `docs/bitacora/_plantilla.md` y las reglas de `.claude/skills/cierre-tarea`.

**Tabla de control de cifras de la bitácora:**

| Cifra | Fuente |
|---|---|
| #15, C = 0.31622776601683794, `balanced`, F1-Val 0.5525902668759812, 1 dentro de la tolerancia | `f45_longitud.log` (línea «seleccionada»), `resultado.json` |
| Fijos `l2`, `lbfgs`, 5000, 42 | `f45_longitud.log`, línea 6 |
| Rangos 0.533649–0.552590 (`balanced`) y 0.377049–0.423769 (`None`); 0 avisos; `n_iter` máximo 7 | `rejilla_longitud.csv` |
| 0.380435, mayoritaria en val | `results/f23_pisos/f23_pisos.log`, línea 10 |
| 0.714–0.717 | `docs/bitacora/F3.5.md`, líneas 96–98 |
| 58.6 % de `train` truncado | `meta/sample_index.csv` (media de `truncated_128` en `train` = 0.5859) |
| SHA-256 de insumos y del CSV; `created_at` | `f45_longitud.log`; manifiesto |
| 1.9 s | `f45_longitud.log`, última línea |
| 7 aprobadas | `results/f45_longitud/pytest.log` |

No quedaron `[completar]`.

**Desviación de la estructura de commits.** La referencia de longitud se cerró en dos commits, no en uno:
1. `ef9bae0`: script y prueba, antes de ejecutar;
2. `3e38793`: resultados y bitácora.

Así el manifiesto registra un commit limpio, como en F2.3. Con un solo commit, `write_predictions` habría marcado `git_commit` como `-dirty`. La desviación está declarada en la bitácora, §5.

## 2. Colectores canónicos

**Implementación** (`src/canonical/build.py`, enganchados en `build()` después de `collect_f33`):

| Colector | Claves | Fuente y contrastes |
|---|---|---|
| `collect_f34` | Por técnica: `f34.<tec>.trial0.seed{42,7,123}.f1_macro` y `f34.<tec>.mejor_trial.seed{…}.f1_macro` (hojas verificadas desde `val_probs`/`val_true`); `trial0.f1_macro_mean/_std`, `mejor_trial.f1_macro_mean` y `ganancia.f1_macro` (`diff`) como derivadas; `trial_adoptado` (count, verificado: regla de la matriz §6 sobre las medias recalculadas) y `trials` (count, extraído). En total 12 por técnica | `results/f34_hpo/<tec>/runs/*.json` (los 36, sin campos de test). Contrastado con `trials.csv` (media y DE de los 12 trials) y `hpo_result.json` → `decision`. Test nulo, con `null_reason` «F3.4 solo validación (matriz §4)» |
| `collect_f3x('f35')` | `f35.<tec>.finalista.<arch>.<strat>.seed<s>.f1_macro` (15, val y test) y `f35.<tec>.finalista.f1_macro_mean/_std` | `results/predictions/final/*.csv` vía `load_predictions`; JSON de cada corrida (probabilidades bit a bit, `evaluate_test`, selección sin `override`); `summary.csv` y `summary_by_config.csv` (val y test, 4 decimales) |
| `collect_f3x('f36')` | `f36.<tec>.permutado.*`, igual estructura | `results/predictions/control_permutacion/*.csv`; `permute_labels = 99`. Test registrado, con nota que cita la **Enmienda 4, A16** en las 21 claves |
| `collect_f3x('f55')` | `f55.<tec>.segundo.*`, igual estructura | `results/predictions/sensibilidad/*.csv` |
| `collect_ref_longitud` | `ref.longitud.f1_macro` | F1 de val verificado desde el CSV (solo filas de val). Test nulo, con el mismo `null_reason` que `ref.techo5d` y las demás referencias. Contrastado con `resultado.json` |

**Configuraciones.** `build()` calcula una sola vez `select_second(config_table(f1_by_run(primera_pasada)))`. `collect_f3x` exige que cada técnica tenga en su fase exactamente la configuración del finalista (`f35`, `f36`) o del segundo (`f55`). Si se intercambian las columnas `finalist` y `second`, el colector falla con «`f35.gray_scott: configuraciones [('efficientnet_b0', 'finetune')] != selección ('resnet18', 'finetune')`»; lo comprobé a mano.

**Insumos.** Se añadieron a `inputs`, con SHA-256 LF, todos los CSV, manifiestos y JSON de corrida usados, además de `summary.csv`, `summary_by_config.csv`, `hpo_result.json`, `trials.csv` y `resultado.json`.

**Pruebas nuevas** (`tests/test_canonical_numbers.py`):
- `test_f34_presencia_y_conteos`;
- `test_f34_recalculo_y_hpo_result` (×3);
- `test_f3x_presencia_y_conteos`;
- `test_f3x_configuracion_de_seleccion` (×3, contra `select_finalists`/`select_second`);
- `test_f3x_recalculo_desde_predicciones` (×3, val y test desde el CSV, y frente a `summary_by_config.csv`);
- `test_f36_nota_enmienda_4_a16`;
- `test_ref_longitud`;
- `test_sin_metricas_de_test_de_referencias`.

Las pruebas genéricas ya existentes (`test_derivadas`, `test_alcanzabilidad_f1`, `test_matriz_confusion`, `test_verificado_desde_predicciones`…) cubren automáticamente las 100 claves nuevas.

**Regeneración.** `CANON_UMAP_PARAMS="G:/Mi unidad/tt2-insumos/umap_params.pkl" CANON_STRICT=1 python -m src.canonical.build`. `--check` con las mismas variables: «al día (10734 líneas)».
- *Nota:* sin `CANON_UMAP_PARAMS`, `--check` informa una diferencia. Los conteos estructurales cambian de estado (`extraido` en vez de `verificado`), igual que antes de este bloque; el archivo commiteado es el del build estricto.

### Diff de claves canónicas (`diff_claves_canonicas.log`)

| Sección | Antes → después | Nuevas | Eliminadas | Cambiadas |
|---|---|---|---|---|
| `numbers` | 276 → 376 | 100 (f34: 36; f35: 21; f36: 21; f55: 21; ref.longitud: 1) | 0 | **0** |
| `inputs` | 233 → 413 | 180 | 0 | **0** |
| `superseded`, `schema_version` | iguales | — | — | — |

Ninguna clave nueva queda fuera de `f34`/`f35`/`f36`/`f55`/`ref.longitud`. La lista completa está en el log.

**Desviación de nombres respecto de la petición.** Para que las medias y DE sean derivadas, como exige la convención (las hojas `verificado` llevan matriz de confusión), F3.4 tiene además 6 hojas por semilla y técnica (`trial0.seed*` y `mejor_trial.seed*`). En f35, f36 y f55, las claves por semilla incluyen el rol y la configuración (`<pref>.<tec>.<rol>.<arch>.<strat>.seed<s>.f1_macro`), como en `f33.*`. Los agregados se llaman `f35.<tec>.finalista.f1_macro_mean/_std`, `f36.<tec>.permutado.…` y `f55.<tec>.segundo.…`.

## 3. Pruebas

```text
$ CANON_UMAP_PARAMS=… CANON_STRICT=1 python -m pytest tests -q --ignore=tests/test_label_independence.py
1014 passed in 139.64s
$ CANON_UMAP_PARAMS=… CANON_STRICT=1 python -m src.canonical.build --check
results/canonical_numbers.json: al día (10734 líneas)
```

Salida en `pytest_bloque_c.log`; `results/f45_longitud/pytest.log` tiene 7 aprobadas. No se corrió `test_label_independence.py`.

## 4. Commits y PR

| Hash | Asunto |
|---|---|
| `ef9bae0` | `[F4.5] Script de la referencia de longitud y prueba sin métricas de test (antes de ejecutar)` |
| `3e38793` | `[F4.5] Referencia de longitud: predicciones archivadas (solo F1-val) y bitácora` |
| `06eb9c3` | `[F0.4, F5] Colectores canónicos f34/f35/f36/f55 y ref.longitud` |
| (este reporte) | `[F5] Reporte del bloque C` |

**PR:** pendiente. `gh` no tiene sesión iniciada en este equipo; el push por git sí funcionó.
