# Reporte D3 — Cierre de la Fase 5

Rama `sv/f5-resultados`. No se volvió a ejecutar `--ejecutar-prueba` ni `--analisis-f46`, y ningún archivo de `results/f5_contraste/` ya commiteado se modificó. Este bloque solo añade `reporte_d3.md` y `d3/`.

## Commits

| Commit | Asunto |
|---|---|
| `e02104f` | [F5] Resultados de la ejecución confirmatoria (run_fase5 --ejecutar-prueba) |
| `43d933c` | [F4.6] Kappa y contención de errores (después del veredicto) |
| `2abd60e` | [F5, F4.6] Reporte D2 de la ejecución confirmatoria (sin interpretación) |
| `4837dcf` | [F5.5] Erratum: params.r de la Formulación A registra el r de la variante |
| `5e4a71b` | [F5, F4.6] Cifras canónicas de la Fase 5 y F4.6; desbloqueo de prueba de ref.* |
| `21c3362` | [F5, F4.5, F4.6] Bitácoras de cierre |
| (este commit) | [F5] Reporte D3 |

## 1. Erratum de metadatos (Formulación A), `4837dcf`

- **Síntoma:** en `resultados.json`, `sensibilidad.formulacion_A["r=0.75"].params.r` y `["r=0.8"].params.r` registran 0.5. Su `params.delta` sí es el de cada variante: 0.05097402597402598 y 0.04077922077922077.
- **Causa:** `src/protocol/sensibilidad.py::formulacion_a` no pasaba `r` a `contraste()`, que registra su valor por omisión. Esto no afecta el cálculo: `formulation_a_reps` solo usa δ.
- **Prueba sintética:** `tests/test_fase5_contraste.py::test_formulacion_A_solo_usa_delta_y_registra_r[0, 20, 40]`.
  - Comprueba que, con δ fijo, `contraste(formulation="A")` da el mismo resultado, salvo `params`, para r ∈ {0.5, 0.75, 0.8, 0.1}.
  - Comprueba también que `formulacion_a` registra el r de cada variante.
  - Sin la corrección falla en el registro; con ella pasa (40 pruebas de `test_fase5_contraste.py` y `test_f55_sensibilidad.py`, 3.55 s).
- **Corrección:** `formulacion_a` pasa `r=r`. No se reejecutó sobre datos reales, así que `resultados.json` conserva el valor erróneo y el erratum está en la bitácora F5, H6.

## 2. Cifras canónicas, `5e4a71b`

### Diseño

- **`collect_f5` / `collect_f46`** (`src/canonical/build.py`). Leen `results/f5_contraste/resultados.json` y `f46_resultados.json`, con su sha256 en `inputs`.
  - Todas las entradas tienen `status` = `extraido` y `value` = {val: null, test: x}; `null_reason.val` = «solo prueba».
  - Los veredictos (V1–V4, G1–G4) y el piso aplicado van como **identidad en la clave**, como en `f33.*`: conteo con valor 1, por ejemplo `f5.principal.perlin.veredicto.v1` o `f5.principal.gray_scott.piso.visual_permutado`.
  - Los booleanos (rechazo, empate, cambio, condición 1) van como conteos 0/1.
  - El localizador es la ruta JSON. Una clave con punto va entre comillas: `sensibilidad.formulacion_A["r=0.75"]`.
- **Esquema.** `metric` admite, además de `f1_macro`: `p_value`, `alpha`, `confidence_level`, `ratio`, `kappa`, `proportion` y `statistic`.
  - θ, D, la descomposición, visual − descriptores y L quedan como `f1_macro`, porque están en su escala.
- **Sin codificar** (no son números):
  - `calificacion` de longitud: null en las tres técnicas; `collect_f5` falla si deja de serlo;
  - `mcnemar.pares`: null; falla si deja de serlo;
  - `condicion_2_F41`: PENDIENTE (P6); va en la nota de `f46.contencion.<t>.condicion_1`.
- **Sin duplicar.** Los F1 observados no se duplican: `check_f5` comprueba que los de `resultados.json` coinciden con `ref.*`, `f35.*`, `f36.*` y `f55.*`.
  - Por semilla, la igualdad es exacta.
  - En `f1_img_obs`, `f1_beto_obs` y cada piso, la tolerancia es 1e-12.
  - El piso estratificado constante (0.5) entra como `f5.piso.stratified.f1_macro`. No es `ref.piso.stratified`, que es una corrida.
- **Desbloqueo de `ref.*`** (`unlock_ref_test`, después de la fusión). A cada `ref.*` con `null_reason.test` «bloqueado hasta la Fase 5»:
  - se le calcula el F1 de prueba desde su CSV (ŷ = 1[p_dep > 0.5]) con su matriz de confusión;
  - se le retira `null_reason.test`;
  - se le amplía el localizador a val y test y se le añade una nota.
  - Si la referencia entra en la Fase 5, debe coincidir **exactamente** con `resultados.json:observados`, o el build falla.
  - Los fragmentos de F2.1–F2.4 no se tocaron.

### Diff de claves frente a `dd2e462` (archivo previo a la Fase 5)

- Añadidas: **574**. Retiradas: **0**. Modificadas: **16**, todas `ref.*` desbloqueadas.
- `superseded` sin cambios: sí.
- Ningún hash previo de `inputs` cambió: sí.

Añadidas por grupo:

| Grupo | Claves |
|---|---|
| `f46.contencion.*` | 75 |
| `f46.contencion_por_semilla.*` | 120 |
| `f46.kappa.*` | 96 |
| `f5.cochran.*` | 4 |
| `f5.descomposicion.*` | 30 |
| `f5.frente_a_descriptores.*` | 9 |
| `f5.longitud.*` | 16 |
| `f5.mcnemar.*` | 36 |
| `f5.piso.*` | 1 |
| `f5.principal.*` | 58 |
| `f5.retencion.*` | 12 |
| `f5.robustez.*` | 58 |
| `f5.sensibilidad.*` | 59 |

Modificadas (prueba desbloqueada). El valor de val no cambia en ninguna:

| Clave | F1-Val | F1-Test (nuevo) | Coincide con `observados` | Campos cambiados |
|---|---|---|---|---|
| `ref.descriptores.gray_scott.f1_macro` | 0.7280131633072809 | 0.6795634920634921 | `('descriptors', 'gray_scott')`: sí | confusion, note, null_reason, source, value |
| `ref.descriptores.lorenz.f1_macro` | 0.6610313482286976 | 0.5853174603174602 | `('descriptors', 'lorenz')`: sí | confusion, note, null_reason, source, value |
| `ref.descriptores.perlin.f1_macro` | 0.6959017056018106 | 0.6420722135007849 | `('descriptors', 'perlin')`: sí | confusion, note, null_reason, source, value |
| `ref.emb768_logreg.f1_macro` | 0.7304964539007093 | 0.6984126984126984 | `emb768`: sí | confusion, note, null_reason, source, value |
| `ref.longitud.f1_macro` | 0.5525902668759812 | 0.5142045454545454 | `longitud`: sí | confusion, note, null_reason, source, value |
| `ref.piso.brightness.gray_scott.f1_macro` | 0.4461717839098329 | 0.44503042596348885 | `('brightness', 'gray_scott')`: sí | confusion, note, null_reason, source, value |
| `ref.piso.brightness.lorenz.f1_macro` | 0.4519230769230769 | 0.4735221674876847 | `('brightness', 'lorenz')`: sí | confusion, note, null_reason, source, value |
| `ref.piso.brightness.perlin.f1_macro` | 0.6451914098972922 | 0.45344880915558305 | `('brightness', 'perlin')`: sí | confusion, note, null_reason, source, value |
| `ref.piso.majority.f1_macro` | 0.3804347826086957 | 0.3804347826086957 | `majority`: sí | confusion, note, null_reason, source, value |
| `ref.piso.stratified.f1_macro` | 0.5558441558441558 | 0.5410628019323671 | sin contraparte en la Fase 5 | confusion, note, null_reason, source, value |
| `ref.techo5d.f1_macro` | 0.7246376811594203 | 0.6933235509904623 | `techo5d`: sí | confusion, note, null_reason, source, value |
| `ref.techo5d.gbm.f1_macro` | 0.6874117158288326 | 0.6717400246812011 | sin contraparte en la Fase 5 | confusion, note, null_reason, source, value |
| `ref.techo5d.logreg.f1_macro` | 0.700358422939068 | 0.7246376811594203 | sin contraparte en la Fase 5 | confusion, note, null_reason, source, value |
| `ref.techo5d.mlp.f1_macro` | 0.6933235509904623 | 0.6795634920634921 | sin contraparte en la Fase 5 | confusion, note, null_reason, source, value |
| `ref.techo5d.rf.f1_macro` | 0.6874117158288326 | 0.7191402599609209 | sin contraparte en la Fase 5 | confusion, note, null_reason, source, value |
| `ref.techo5d.svm_rbf.f1_macro` | 0.7246376811594203 | 0.6933235509904623 | `techo5d`: sí | confusion, note, null_reason, source, value |

Insumos añadidos a `inputs`:

- `results/f5_contraste/f46_resultados.json`: `f77c5ea2d628ccb9137a6a9b64ba89077de9dec5e88abcfe0e081887fb7109ce`
- `results/f5_contraste/resultados.json`: `42c0ed12ab271bea2b8ddbd2b1f6dfad31c8396ff6ae720751a632d79c752261`
- `results/predictions/referencia/emb768__logreg__none__seed42.csv`: `b8c3d471a860b943df962fa8e4652d2ff400e3fcbcf6b9af2e84871091cb778a`
- `results/predictions/referencia/gray_scott__brightness__none__seed42.csv`: `f5183e70a6327deaa7d6c1ca0f7e28957900f4244621808aaa81d7e47bee37fb`
- `results/predictions/referencia/gray_scott__descriptors__none__seed42.csv`: `a464b321e40d9601067352e349f40a8d2f9baee608a5b2f31276a185bedbb51b`
- `results/predictions/referencia/lorenz__brightness__none__seed42.csv`: `de67b6e5a7531b5cd854a016c6e2c520d7759bb9b8c74fbc09d929a246f1a43b`
- `results/predictions/referencia/lorenz__descriptors__none__seed42.csv`: `98017425fe294d932d5082353e1bcfb369e8db988f3dba674bb60333cae8e4f0`
- `results/predictions/referencia/none__majority__none__seed42.csv`: `14c4b6efbcfcaaedb6112eab13b9b06edf49150101d5f38b68e9602beb6ef819`
- `results/predictions/referencia/none__stratified__none__seed42.csv`: `2c5a94075833659f446f3aaa8f81a77d1396c8870ffc87427c7f8c7b93657a3b`
- `results/predictions/referencia/perlin__brightness__none__seed42.csv`: `93e6a60e61b98bad816c46b9916110b15b6e7726e3ccc204571158021c8d3369`
- `results/predictions/referencia/perlin__descriptors__none__seed42.csv`: `92f45fd2252a2545843d9adab1398f6b0a4ba8f233a8867ab0af510588342538`
- `results/predictions/referencia/umap5d__gbm__none__seed42.csv`: `1abd93e2e1002f8ee7bb62413fc3f80c9f994d324879ccd958c825b6736ba68b`
- `results/predictions/referencia/umap5d__logreg__none__seed42.csv`: `d92afb8350ca7750f41e31c0f58b0ff4f1d7b205c9341c87b8c494a49b1425db`
- `results/predictions/referencia/umap5d__mlp__none__seed42.csv`: `37d935541d09a03869bdd487a7cf44e0cf28aa348fbb2a34afc94f21b026e472`
- `results/predictions/referencia/umap5d__rf__none__seed42.csv`: `1221601564e07c4b4a3e7ca7b361469bb23963497446490028a6e001edef9935`
- `results/predictions/referencia/umap5d__svm_rbf__none__seed42.csv`: `c061b61580e3132b4b5ad0093cb8a5b032d7065b8ef9fb18f2cafa2e03cf426f`

<details><summary>Lista completa de claves añadidas</summary>

```
f46.contencion.gray_scott.beto.c
f46.contencion.gray_scott.beto.c_ic_inf
f46.contencion.gray_scott.beto.c_ic_sup
f46.contencion.gray_scott.beto.n_errores_ref
f46.contencion.gray_scott.beto.n_errores_vis
f46.contencion.gray_scott.beto.n_interseccion
f46.contencion.gray_scott.beto.n_replicas_indefinidas
f46.contencion.gray_scott.beto.nula
f46.contencion.gray_scott.condicion_1
f46.contencion.gray_scott.emb768.c
f46.contencion.gray_scott.emb768.c_ic_inf
f46.contencion.gray_scott.emb768.c_ic_sup
f46.contencion.gray_scott.emb768.n_errores_ref
f46.contencion.gray_scott.emb768.n_errores_vis
f46.contencion.gray_scott.emb768.n_interseccion
f46.contencion.gray_scott.emb768.n_replicas_indefinidas
f46.contencion.gray_scott.emb768.nula
f46.contencion.gray_scott.techo5d.c
f46.contencion.gray_scott.techo5d.c_ic_inf
f46.contencion.gray_scott.techo5d.c_ic_sup
f46.contencion.gray_scott.techo5d.n_errores_ref
f46.contencion.gray_scott.techo5d.n_errores_vis
f46.contencion.gray_scott.techo5d.n_interseccion
f46.contencion.gray_scott.techo5d.n_replicas_indefinidas
f46.contencion.gray_scott.techo5d.nula
f46.contencion.lorenz.beto.c
f46.contencion.lorenz.beto.c_ic_inf
f46.contencion.lorenz.beto.c_ic_sup
f46.contencion.lorenz.beto.n_errores_ref
f46.contencion.lorenz.beto.n_errores_vis
f46.contencion.lorenz.beto.n_interseccion
f46.contencion.lorenz.beto.n_replicas_indefinidas
f46.contencion.lorenz.beto.nula
f46.contencion.lorenz.condicion_1
f46.contencion.lorenz.emb768.c
f46.contencion.lorenz.emb768.c_ic_inf
f46.contencion.lorenz.emb768.c_ic_sup
f46.contencion.lorenz.emb768.n_errores_ref
f46.contencion.lorenz.emb768.n_errores_vis
f46.contencion.lorenz.emb768.n_interseccion
f46.contencion.lorenz.emb768.n_replicas_indefinidas
f46.contencion.lorenz.emb768.nula
f46.contencion.lorenz.techo5d.c
f46.contencion.lorenz.techo5d.c_ic_inf
f46.contencion.lorenz.techo5d.c_ic_sup
f46.contencion.lorenz.techo5d.n_errores_ref
f46.contencion.lorenz.techo5d.n_errores_vis
f46.contencion.lorenz.techo5d.n_interseccion
f46.contencion.lorenz.techo5d.n_replicas_indefinidas
f46.contencion.lorenz.techo5d.nula
f46.contencion.perlin.beto.c
f46.contencion.perlin.beto.c_ic_inf
f46.contencion.perlin.beto.c_ic_sup
f46.contencion.perlin.beto.n_errores_ref
f46.contencion.perlin.beto.n_errores_vis
f46.contencion.perlin.beto.n_interseccion
f46.contencion.perlin.beto.n_replicas_indefinidas
f46.contencion.perlin.beto.nula
f46.contencion.perlin.condicion_1
f46.contencion.perlin.emb768.c
f46.contencion.perlin.emb768.c_ic_inf
f46.contencion.perlin.emb768.c_ic_sup
f46.contencion.perlin.emb768.n_errores_ref
f46.contencion.perlin.emb768.n_errores_vis
f46.contencion.perlin.emb768.n_interseccion
f46.contencion.perlin.emb768.n_replicas_indefinidas
f46.contencion.perlin.emb768.nula
f46.contencion.perlin.techo5d.c
f46.contencion.perlin.techo5d.c_ic_inf
f46.contencion.perlin.techo5d.c_ic_sup
f46.contencion.perlin.techo5d.n_errores_ref
f46.contencion.perlin.techo5d.n_errores_vis
f46.contencion.perlin.techo5d.n_interseccion
f46.contencion.perlin.techo5d.n_replicas_indefinidas
f46.contencion.perlin.techo5d.nula
f46.contencion_por_semilla.gray_scott.seed123.techo5d.c
f46.contencion_por_semilla.gray_scott.seed123.techo5d.c_ic_inf
f46.contencion_por_semilla.gray_scott.seed123.techo5d.c_ic_sup
f46.contencion_por_semilla.gray_scott.seed123.techo5d.n_errores_ref
f46.contencion_por_semilla.gray_scott.seed123.techo5d.n_errores_vis
f46.contencion_por_semilla.gray_scott.seed123.techo5d.n_interseccion
f46.contencion_por_semilla.gray_scott.seed123.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.gray_scott.seed123.techo5d.nula
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.c
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.c_ic_inf
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.c_ic_sup
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.n_errores_ref
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.n_errores_vis
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.n_interseccion
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.gray_scott.seed2024.techo5d.nula
f46.contencion_por_semilla.gray_scott.seed42.techo5d.c
f46.contencion_por_semilla.gray_scott.seed42.techo5d.c_ic_inf
f46.contencion_por_semilla.gray_scott.seed42.techo5d.c_ic_sup
f46.contencion_por_semilla.gray_scott.seed42.techo5d.n_errores_ref
f46.contencion_por_semilla.gray_scott.seed42.techo5d.n_errores_vis
f46.contencion_por_semilla.gray_scott.seed42.techo5d.n_interseccion
f46.contencion_por_semilla.gray_scott.seed42.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.gray_scott.seed42.techo5d.nula
f46.contencion_por_semilla.gray_scott.seed7.techo5d.c
f46.contencion_por_semilla.gray_scott.seed7.techo5d.c_ic_inf
f46.contencion_por_semilla.gray_scott.seed7.techo5d.c_ic_sup
f46.contencion_por_semilla.gray_scott.seed7.techo5d.n_errores_ref
f46.contencion_por_semilla.gray_scott.seed7.techo5d.n_errores_vis
f46.contencion_por_semilla.gray_scott.seed7.techo5d.n_interseccion
f46.contencion_por_semilla.gray_scott.seed7.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.gray_scott.seed7.techo5d.nula
f46.contencion_por_semilla.gray_scott.seed99.techo5d.c
f46.contencion_por_semilla.gray_scott.seed99.techo5d.c_ic_inf
f46.contencion_por_semilla.gray_scott.seed99.techo5d.c_ic_sup
f46.contencion_por_semilla.gray_scott.seed99.techo5d.n_errores_ref
f46.contencion_por_semilla.gray_scott.seed99.techo5d.n_errores_vis
f46.contencion_por_semilla.gray_scott.seed99.techo5d.n_interseccion
f46.contencion_por_semilla.gray_scott.seed99.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.gray_scott.seed99.techo5d.nula
f46.contencion_por_semilla.lorenz.seed123.techo5d.c
f46.contencion_por_semilla.lorenz.seed123.techo5d.c_ic_inf
f46.contencion_por_semilla.lorenz.seed123.techo5d.c_ic_sup
f46.contencion_por_semilla.lorenz.seed123.techo5d.n_errores_ref
f46.contencion_por_semilla.lorenz.seed123.techo5d.n_errores_vis
f46.contencion_por_semilla.lorenz.seed123.techo5d.n_interseccion
f46.contencion_por_semilla.lorenz.seed123.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.lorenz.seed123.techo5d.nula
f46.contencion_por_semilla.lorenz.seed2024.techo5d.c
f46.contencion_por_semilla.lorenz.seed2024.techo5d.c_ic_inf
f46.contencion_por_semilla.lorenz.seed2024.techo5d.c_ic_sup
f46.contencion_por_semilla.lorenz.seed2024.techo5d.n_errores_ref
f46.contencion_por_semilla.lorenz.seed2024.techo5d.n_errores_vis
f46.contencion_por_semilla.lorenz.seed2024.techo5d.n_interseccion
f46.contencion_por_semilla.lorenz.seed2024.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.lorenz.seed2024.techo5d.nula
f46.contencion_por_semilla.lorenz.seed42.techo5d.c
f46.contencion_por_semilla.lorenz.seed42.techo5d.c_ic_inf
f46.contencion_por_semilla.lorenz.seed42.techo5d.c_ic_sup
f46.contencion_por_semilla.lorenz.seed42.techo5d.n_errores_ref
f46.contencion_por_semilla.lorenz.seed42.techo5d.n_errores_vis
f46.contencion_por_semilla.lorenz.seed42.techo5d.n_interseccion
f46.contencion_por_semilla.lorenz.seed42.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.lorenz.seed42.techo5d.nula
f46.contencion_por_semilla.lorenz.seed7.techo5d.c
f46.contencion_por_semilla.lorenz.seed7.techo5d.c_ic_inf
f46.contencion_por_semilla.lorenz.seed7.techo5d.c_ic_sup
f46.contencion_por_semilla.lorenz.seed7.techo5d.n_errores_ref
f46.contencion_por_semilla.lorenz.seed7.techo5d.n_errores_vis
f46.contencion_por_semilla.lorenz.seed7.techo5d.n_interseccion
f46.contencion_por_semilla.lorenz.seed7.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.lorenz.seed7.techo5d.nula
f46.contencion_por_semilla.lorenz.seed99.techo5d.c
f46.contencion_por_semilla.lorenz.seed99.techo5d.c_ic_inf
f46.contencion_por_semilla.lorenz.seed99.techo5d.c_ic_sup
f46.contencion_por_semilla.lorenz.seed99.techo5d.n_errores_ref
f46.contencion_por_semilla.lorenz.seed99.techo5d.n_errores_vis
f46.contencion_por_semilla.lorenz.seed99.techo5d.n_interseccion
f46.contencion_por_semilla.lorenz.seed99.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.lorenz.seed99.techo5d.nula
f46.contencion_por_semilla.perlin.seed123.techo5d.c
f46.contencion_por_semilla.perlin.seed123.techo5d.c_ic_inf
f46.contencion_por_semilla.perlin.seed123.techo5d.c_ic_sup
f46.contencion_por_semilla.perlin.seed123.techo5d.n_errores_ref
f46.contencion_por_semilla.perlin.seed123.techo5d.n_errores_vis
f46.contencion_por_semilla.perlin.seed123.techo5d.n_interseccion
f46.contencion_por_semilla.perlin.seed123.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.perlin.seed123.techo5d.nula
f46.contencion_por_semilla.perlin.seed2024.techo5d.c
f46.contencion_por_semilla.perlin.seed2024.techo5d.c_ic_inf
f46.contencion_por_semilla.perlin.seed2024.techo5d.c_ic_sup
f46.contencion_por_semilla.perlin.seed2024.techo5d.n_errores_ref
f46.contencion_por_semilla.perlin.seed2024.techo5d.n_errores_vis
f46.contencion_por_semilla.perlin.seed2024.techo5d.n_interseccion
f46.contencion_por_semilla.perlin.seed2024.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.perlin.seed2024.techo5d.nula
f46.contencion_por_semilla.perlin.seed42.techo5d.c
f46.contencion_por_semilla.perlin.seed42.techo5d.c_ic_inf
f46.contencion_por_semilla.perlin.seed42.techo5d.c_ic_sup
f46.contencion_por_semilla.perlin.seed42.techo5d.n_errores_ref
f46.contencion_por_semilla.perlin.seed42.techo5d.n_errores_vis
f46.contencion_por_semilla.perlin.seed42.techo5d.n_interseccion
f46.contencion_por_semilla.perlin.seed42.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.perlin.seed42.techo5d.nula
f46.contencion_por_semilla.perlin.seed7.techo5d.c
f46.contencion_por_semilla.perlin.seed7.techo5d.c_ic_inf
f46.contencion_por_semilla.perlin.seed7.techo5d.c_ic_sup
f46.contencion_por_semilla.perlin.seed7.techo5d.n_errores_ref
f46.contencion_por_semilla.perlin.seed7.techo5d.n_errores_vis
f46.contencion_por_semilla.perlin.seed7.techo5d.n_interseccion
f46.contencion_por_semilla.perlin.seed7.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.perlin.seed7.techo5d.nula
f46.contencion_por_semilla.perlin.seed99.techo5d.c
f46.contencion_por_semilla.perlin.seed99.techo5d.c_ic_inf
f46.contencion_por_semilla.perlin.seed99.techo5d.c_ic_sup
f46.contencion_por_semilla.perlin.seed99.techo5d.n_errores_ref
f46.contencion_por_semilla.perlin.seed99.techo5d.n_errores_vis
f46.contencion_por_semilla.perlin.seed99.techo5d.n_interseccion
f46.contencion_por_semilla.perlin.seed99.techo5d.n_replicas_indefinidas
f46.contencion_por_semilla.perlin.seed99.techo5d.nula
f46.kappa.vector_acierto.beto__emb768.kappa
f46.kappa.vector_acierto.beto__emb768.kappa_ic_inf
f46.kappa.vector_acierto.beto__emb768.kappa_ic_sup
f46.kappa.vector_acierto.beto__emb768.n_replicas_indefinidas
f46.kappa.vector_acierto.beto__gray_scott.kappa
f46.kappa.vector_acierto.beto__gray_scott.kappa_ic_inf
f46.kappa.vector_acierto.beto__gray_scott.kappa_ic_sup
f46.kappa.vector_acierto.beto__gray_scott.n_replicas_indefinidas
f46.kappa.vector_acierto.beto__lorenz.kappa
f46.kappa.vector_acierto.beto__lorenz.kappa_ic_inf
f46.kappa.vector_acierto.beto__lorenz.kappa_ic_sup
f46.kappa.vector_acierto.beto__lorenz.n_replicas_indefinidas
f46.kappa.vector_acierto.beto__perlin.kappa
f46.kappa.vector_acierto.beto__perlin.kappa_ic_inf
f46.kappa.vector_acierto.beto__perlin.kappa_ic_sup
f46.kappa.vector_acierto.beto__perlin.n_replicas_indefinidas
f46.kappa.vector_acierto.beto__techo5d.kappa
f46.kappa.vector_acierto.beto__techo5d.kappa_ic_inf
f46.kappa.vector_acierto.beto__techo5d.kappa_ic_sup
f46.kappa.vector_acierto.beto__techo5d.n_replicas_indefinidas
f46.kappa.vector_acierto.emb768__gray_scott.kappa
f46.kappa.vector_acierto.emb768__gray_scott.kappa_ic_inf
f46.kappa.vector_acierto.emb768__gray_scott.kappa_ic_sup
f46.kappa.vector_acierto.emb768__gray_scott.n_replicas_indefinidas
f46.kappa.vector_acierto.emb768__lorenz.kappa
f46.kappa.vector_acierto.emb768__lorenz.kappa_ic_inf
f46.kappa.vector_acierto.emb768__lorenz.kappa_ic_sup
f46.kappa.vector_acierto.emb768__lorenz.n_replicas_indefinidas
f46.kappa.vector_acierto.emb768__perlin.kappa
f46.kappa.vector_acierto.emb768__perlin.kappa_ic_inf
f46.kappa.vector_acierto.emb768__perlin.kappa_ic_sup
f46.kappa.vector_acierto.emb768__perlin.n_replicas_indefinidas
f46.kappa.vector_acierto.emb768__techo5d.kappa
f46.kappa.vector_acierto.emb768__techo5d.kappa_ic_inf
f46.kappa.vector_acierto.emb768__techo5d.kappa_ic_sup
f46.kappa.vector_acierto.emb768__techo5d.n_replicas_indefinidas
f46.kappa.vector_acierto.techo5d__gray_scott.kappa
f46.kappa.vector_acierto.techo5d__gray_scott.kappa_ic_inf
f46.kappa.vector_acierto.techo5d__gray_scott.kappa_ic_sup
f46.kappa.vector_acierto.techo5d__gray_scott.n_replicas_indefinidas
f46.kappa.vector_acierto.techo5d__lorenz.kappa
f46.kappa.vector_acierto.techo5d__lorenz.kappa_ic_inf
f46.kappa.vector_acierto.techo5d__lorenz.kappa_ic_sup
f46.kappa.vector_acierto.techo5d__lorenz.n_replicas_indefinidas
f46.kappa.vector_acierto.techo5d__perlin.kappa
f46.kappa.vector_acierto.techo5d__perlin.kappa_ic_inf
f46.kappa.vector_acierto.techo5d__perlin.kappa_ic_sup
f46.kappa.vector_acierto.techo5d__perlin.n_replicas_indefinidas
f46.kappa.y_hat.beto__emb768.kappa
f46.kappa.y_hat.beto__emb768.kappa_ic_inf
f46.kappa.y_hat.beto__emb768.kappa_ic_sup
f46.kappa.y_hat.beto__emb768.n_replicas_indefinidas
f46.kappa.y_hat.beto__gray_scott.kappa
f46.kappa.y_hat.beto__gray_scott.kappa_ic_inf
f46.kappa.y_hat.beto__gray_scott.kappa_ic_sup
f46.kappa.y_hat.beto__gray_scott.n_replicas_indefinidas
f46.kappa.y_hat.beto__lorenz.kappa
f46.kappa.y_hat.beto__lorenz.kappa_ic_inf
f46.kappa.y_hat.beto__lorenz.kappa_ic_sup
f46.kappa.y_hat.beto__lorenz.n_replicas_indefinidas
f46.kappa.y_hat.beto__perlin.kappa
f46.kappa.y_hat.beto__perlin.kappa_ic_inf
f46.kappa.y_hat.beto__perlin.kappa_ic_sup
f46.kappa.y_hat.beto__perlin.n_replicas_indefinidas
f46.kappa.y_hat.beto__techo5d.kappa
f46.kappa.y_hat.beto__techo5d.kappa_ic_inf
f46.kappa.y_hat.beto__techo5d.kappa_ic_sup
f46.kappa.y_hat.beto__techo5d.n_replicas_indefinidas
f46.kappa.y_hat.emb768__gray_scott.kappa
f46.kappa.y_hat.emb768__gray_scott.kappa_ic_inf
f46.kappa.y_hat.emb768__gray_scott.kappa_ic_sup
f46.kappa.y_hat.emb768__gray_scott.n_replicas_indefinidas
f46.kappa.y_hat.emb768__lorenz.kappa
f46.kappa.y_hat.emb768__lorenz.kappa_ic_inf
f46.kappa.y_hat.emb768__lorenz.kappa_ic_sup
f46.kappa.y_hat.emb768__lorenz.n_replicas_indefinidas
f46.kappa.y_hat.emb768__perlin.kappa
f46.kappa.y_hat.emb768__perlin.kappa_ic_inf
f46.kappa.y_hat.emb768__perlin.kappa_ic_sup
f46.kappa.y_hat.emb768__perlin.n_replicas_indefinidas
f46.kappa.y_hat.emb768__techo5d.kappa
f46.kappa.y_hat.emb768__techo5d.kappa_ic_inf
f46.kappa.y_hat.emb768__techo5d.kappa_ic_sup
f46.kappa.y_hat.emb768__techo5d.n_replicas_indefinidas
f46.kappa.y_hat.techo5d__gray_scott.kappa
f46.kappa.y_hat.techo5d__gray_scott.kappa_ic_inf
f46.kappa.y_hat.techo5d__gray_scott.kappa_ic_sup
f46.kappa.y_hat.techo5d__gray_scott.n_replicas_indefinidas
f46.kappa.y_hat.techo5d__lorenz.kappa
f46.kappa.y_hat.techo5d__lorenz.kappa_ic_inf
f46.kappa.y_hat.techo5d__lorenz.kappa_ic_sup
f46.kappa.y_hat.techo5d__lorenz.n_replicas_indefinidas
f46.kappa.y_hat.techo5d__perlin.kappa
f46.kappa.y_hat.techo5d__perlin.kappa_ic_inf
f46.kappa.y_hat.techo5d__perlin.kappa_ic_sup
f46.kappa.y_hat.techo5d__perlin.n_replicas_indefinidas
f5.cochran.gl
f5.cochran.p
f5.cochran.q
f5.cochran.todos_concordantes
f5.descomposicion.gray_scott.cabeza
f5.descomposicion.gray_scott.cabeza_ic_inf
f5.descomposicion.gray_scott.cabeza_ic_sup
f5.descomposicion.gray_scott.reduccion
f5.descomposicion.gray_scott.reduccion_ic_inf
f5.descomposicion.gray_scott.reduccion_ic_sup
f5.descomposicion.gray_scott.renderizado
f5.descomposicion.gray_scott.renderizado_ic_inf
f5.descomposicion.gray_scott.renderizado_ic_sup
f5.descomposicion.gray_scott.total
f5.descomposicion.lorenz.cabeza
f5.descomposicion.lorenz.cabeza_ic_inf
f5.descomposicion.lorenz.cabeza_ic_sup
f5.descomposicion.lorenz.reduccion
f5.descomposicion.lorenz.reduccion_ic_inf
f5.descomposicion.lorenz.reduccion_ic_sup
f5.descomposicion.lorenz.renderizado
f5.descomposicion.lorenz.renderizado_ic_inf
f5.descomposicion.lorenz.renderizado_ic_sup
f5.descomposicion.lorenz.total
f5.descomposicion.perlin.cabeza
f5.descomposicion.perlin.cabeza_ic_inf
f5.descomposicion.perlin.cabeza_ic_sup
f5.descomposicion.perlin.reduccion
f5.descomposicion.perlin.reduccion_ic_inf
f5.descomposicion.perlin.reduccion_ic_sup
f5.descomposicion.perlin.renderizado
f5.descomposicion.perlin.renderizado_ic_inf
f5.descomposicion.perlin.renderizado_ic_sup
f5.descomposicion.perlin.total
f5.frente_a_descriptores.gray_scott
f5.frente_a_descriptores.gray_scott_ic_inf
f5.frente_a_descriptores.gray_scott_ic_sup
f5.frente_a_descriptores.lorenz
f5.frente_a_descriptores.lorenz_ic_inf
f5.frente_a_descriptores.lorenz_ic_sup
f5.frente_a_descriptores.perlin
f5.frente_a_descriptores.perlin_ic_inf
f5.frente_a_descriptores.perlin_ic_sup
f5.longitud.gray_scott.alpha_h
f5.longitud.gray_scott.l
f5.longitud.gray_scott.p
f5.longitud.gray_scott.rango_holm
f5.longitud.gray_scott.rechaza
f5.longitud.lorenz.alpha_h
f5.longitud.lorenz.l
f5.longitud.lorenz.p
f5.longitud.lorenz.rango_holm
f5.longitud.lorenz.rechaza
f5.longitud.m
f5.longitud.perlin.alpha_h
f5.longitud.perlin.l
f5.longitud.perlin.p
f5.longitud.perlin.rango_holm
f5.longitud.perlin.rechaza
f5.mcnemar.beto.unilateral.gray_scott.alpha_h
f5.mcnemar.beto.unilateral.gray_scott.b
f5.mcnemar.beto.unilateral.gray_scott.c
f5.mcnemar.beto.unilateral.gray_scott.p
f5.mcnemar.beto.unilateral.gray_scott.rango_holm
f5.mcnemar.beto.unilateral.gray_scott.rechaza
f5.mcnemar.beto.unilateral.lorenz.alpha_h
f5.mcnemar.beto.unilateral.lorenz.b
f5.mcnemar.beto.unilateral.lorenz.c
f5.mcnemar.beto.unilateral.lorenz.p
f5.mcnemar.beto.unilateral.lorenz.rango_holm
f5.mcnemar.beto.unilateral.lorenz.rechaza
f5.mcnemar.beto.unilateral.perlin.alpha_h
f5.mcnemar.beto.unilateral.perlin.b
f5.mcnemar.beto.unilateral.perlin.c
f5.mcnemar.beto.unilateral.perlin.p
f5.mcnemar.beto.unilateral.perlin.rango_holm
f5.mcnemar.beto.unilateral.perlin.rechaza
f5.mcnemar.techo5d.unilateral.gray_scott.alpha_h
f5.mcnemar.techo5d.unilateral.gray_scott.b
f5.mcnemar.techo5d.unilateral.gray_scott.c
f5.mcnemar.techo5d.unilateral.gray_scott.p
f5.mcnemar.techo5d.unilateral.gray_scott.rango_holm
f5.mcnemar.techo5d.unilateral.gray_scott.rechaza
f5.mcnemar.techo5d.unilateral.lorenz.alpha_h
f5.mcnemar.techo5d.unilateral.lorenz.b
f5.mcnemar.techo5d.unilateral.lorenz.c
f5.mcnemar.techo5d.unilateral.lorenz.p
f5.mcnemar.techo5d.unilateral.lorenz.rango_holm
f5.mcnemar.techo5d.unilateral.lorenz.rechaza
f5.mcnemar.techo5d.unilateral.perlin.alpha_h
f5.mcnemar.techo5d.unilateral.perlin.b
f5.mcnemar.techo5d.unilateral.perlin.c
f5.mcnemar.techo5d.unilateral.perlin.p
f5.mcnemar.techo5d.unilateral.perlin.rango_holm
f5.mcnemar.techo5d.unilateral.perlin.rechaza
f5.piso.stratified.f1_macro
f5.principal.global.g1
f5.principal.gray_scott.d
f5.principal.gray_scott.d_ic_inf
f5.principal.gray_scott.d_ic_nivel
f5.principal.gray_scott.d_ic_sup
f5.principal.gray_scott.no_inferioridad.alpha_h
f5.principal.gray_scott.no_inferioridad.p
f5.principal.gray_scott.no_inferioridad.rango_holm
f5.principal.gray_scott.no_inferioridad.rechaza
f5.principal.gray_scott.piso.visual_permutado
f5.principal.gray_scott.piso_empate
f5.principal.gray_scott.superioridad_piso.alpha_h
f5.principal.gray_scott.superioridad_piso.p
f5.principal.gray_scott.superioridad_piso.rango_holm
f5.principal.gray_scott.superioridad_piso.rechaza
f5.principal.gray_scott.theta
f5.principal.gray_scott.theta_ic_inf
f5.principal.gray_scott.theta_ic_nivel
f5.principal.gray_scott.theta_ic_sup
f5.principal.gray_scott.veredicto.v1
f5.principal.lorenz.d
f5.principal.lorenz.d_ic_inf
f5.principal.lorenz.d_ic_nivel
f5.principal.lorenz.d_ic_sup
f5.principal.lorenz.no_inferioridad.alpha_h
f5.principal.lorenz.no_inferioridad.p
f5.principal.lorenz.no_inferioridad.rango_holm
f5.principal.lorenz.no_inferioridad.rechaza
f5.principal.lorenz.piso.visual_permutado
f5.principal.lorenz.piso_empate
f5.principal.lorenz.superioridad_piso.alpha_h
f5.principal.lorenz.superioridad_piso.p
f5.principal.lorenz.superioridad_piso.rango_holm
f5.principal.lorenz.superioridad_piso.rechaza
f5.principal.lorenz.theta
f5.principal.lorenz.theta_ic_inf
f5.principal.lorenz.theta_ic_nivel
f5.principal.lorenz.theta_ic_sup
f5.principal.lorenz.veredicto.v1
f5.principal.perlin.d
f5.principal.perlin.d_ic_inf
f5.principal.perlin.d_ic_nivel
f5.principal.perlin.d_ic_sup
f5.principal.perlin.no_inferioridad.alpha_h
f5.principal.perlin.no_inferioridad.p
f5.principal.perlin.no_inferioridad.rango_holm
f5.principal.perlin.no_inferioridad.rechaza
f5.principal.perlin.piso.visual_permutado
f5.principal.perlin.piso_empate
f5.principal.perlin.superioridad_piso.alpha_h
f5.principal.perlin.superioridad_piso.p
f5.principal.perlin.superioridad_piso.rango_holm
f5.principal.perlin.superioridad_piso.rechaza
f5.principal.perlin.theta
f5.principal.perlin.theta_ic_inf
f5.principal.perlin.theta_ic_nivel
f5.principal.perlin.theta_ic_sup
f5.principal.perlin.veredicto.v1
f5.retencion.gray_scott.n_denominador_no_positivo
f5.retencion.gray_scott.rho
f5.retencion.gray_scott.rho_ic_inf
f5.retencion.gray_scott.rho_ic_sup
f5.retencion.lorenz.n_denominador_no_positivo
f5.retencion.lorenz.rho
f5.retencion.lorenz.rho_ic_inf
f5.retencion.lorenz.rho_ic_sup
f5.retencion.perlin.n_denominador_no_positivo
f5.retencion.perlin.rho
f5.retencion.perlin.rho_ic_inf
f5.retencion.perlin.rho_ic_sup
f5.robustez.a12.cambia.global
f5.robustez.a12.cambia.gray_scott
f5.robustez.a12.cambia.lorenz
f5.robustez.a12.cambia.perlin
f5.robustez.a12.global.g1
f5.robustez.a12.gray_scott.veredicto.v1
f5.robustez.a12.lorenz.veredicto.v1
f5.robustez.a12.perlin.veredicto.v1
f5.robustez.a14.cambia_n_semillas_v1.gray_scott
f5.robustez.a14.cambia_n_semillas_v1.lorenz
f5.robustez.a14.cambia_n_semillas_v1.perlin
f5.robustez.a14.n_semillas_v1.gray_scott
f5.robustez.a14.n_semillas_v1.lorenz
f5.robustez.a14.n_semillas_v1.perlin
f5.robustez.a14.seed123.global.g1
f5.robustez.a14.seed123.gray_scott.veredicto.v1
f5.robustez.a14.seed123.lorenz.veredicto.v1
f5.robustez.a14.seed123.perlin.veredicto.v1
f5.robustez.a14.seed2024.global.g1
f5.robustez.a14.seed2024.gray_scott.veredicto.v3
f5.robustez.a14.seed2024.lorenz.veredicto.v1
f5.robustez.a14.seed2024.perlin.veredicto.v4
f5.robustez.a14.seed42.global.g1
f5.robustez.a14.seed42.gray_scott.veredicto.v1
f5.robustez.a14.seed42.lorenz.veredicto.v1
f5.robustez.a14.seed42.perlin.veredicto.v4
f5.robustez.a14.seed7.global.g1
f5.robustez.a14.seed7.gray_scott.veredicto.v1
f5.robustez.a14.seed7.lorenz.veredicto.v1
f5.robustez.a14.seed7.perlin.veredicto.v3
f5.robustez.a14.seed99.global.g1
f5.robustez.a14.seed99.gray_scott.veredicto.v1
f5.robustez.a14.seed99.lorenz.veredicto.v1
f5.robustez.a14.seed99.perlin.veredicto.v1
f5.robustez.a2.cambia.global
f5.robustez.a2.cambia.gray_scott
f5.robustez.a2.cambia.lorenz
f5.robustez.a2.cambia.perlin
f5.robustez.a2.global.g1
f5.robustez.a2.gray_scott.veredicto.v1
f5.robustez.a2.lorenz.veredicto.v1
f5.robustez.a2.perlin.veredicto.v1
f5.robustez.a3.cambia.global
f5.robustez.a3.cambia.gray_scott
f5.robustez.a3.cambia.lorenz
f5.robustez.a3.cambia.perlin
f5.robustez.a3.global.g1
f5.robustez.a3.gray_scott.veredicto.v1
f5.robustez.a3.lorenz.veredicto.v1
f5.robustez.a3.perlin.veredicto.v1
f5.robustez.a7.cambia.global
f5.robustez.a7.cambia.gray_scott
f5.robustez.a7.cambia.lorenz
f5.robustez.a7.cambia.perlin
f5.robustez.a7.global.g1
f5.robustez.a7.gray_scott.veredicto.v1
f5.robustez.a7.lorenz.veredicto.v1
f5.robustez.a7.perlin.veredicto.v1
f5.sensibilidad.formulacion_a.r0_50.global.g1
f5.sensibilidad.formulacion_a.r0_50.gray_scott.veredicto.v1
f5.sensibilidad.formulacion_a.r0_50.lorenz.veredicto.v1
f5.sensibilidad.formulacion_a.r0_50.perlin.veredicto.v1
f5.sensibilidad.formulacion_a.r0_75.global.g1
f5.sensibilidad.formulacion_a.r0_75.gray_scott.veredicto.v1
f5.sensibilidad.formulacion_a.r0_75.lorenz.veredicto.v1
f5.sensibilidad.formulacion_a.r0_75.perlin.veredicto.v4
f5.sensibilidad.formulacion_a.r0_80.global.g1
f5.sensibilidad.formulacion_a.r0_80.gray_scott.veredicto.v1
f5.sensibilidad.formulacion_a.r0_80.lorenz.veredicto.v1
f5.sensibilidad.formulacion_a.r0_80.perlin.veredicto.v4
f5.sensibilidad.piso_sin_permutado.global.g1
f5.sensibilidad.piso_sin_permutado.gray_scott.veredicto.v1
f5.sensibilidad.piso_sin_permutado.lorenz.veredicto.v1
f5.sensibilidad.piso_sin_permutado.perlin.veredicto.v1
f5.sensibilidad.por_semilla.n_semillas_v1.gray_scott
f5.sensibilidad.por_semilla.n_semillas_v1.lorenz
f5.sensibilidad.por_semilla.n_semillas_v1.perlin
f5.sensibilidad.por_semilla.seed123.global.g1
f5.sensibilidad.por_semilla.seed123.gray_scott.veredicto.v1
f5.sensibilidad.por_semilla.seed123.lorenz.veredicto.v1
f5.sensibilidad.por_semilla.seed123.perlin.veredicto.v1
f5.sensibilidad.por_semilla.seed2024.global.g1
f5.sensibilidad.por_semilla.seed2024.gray_scott.veredicto.v1
f5.sensibilidad.por_semilla.seed2024.lorenz.veredicto.v1
f5.sensibilidad.por_semilla.seed2024.perlin.veredicto.v4
f5.sensibilidad.por_semilla.seed42.global.g1
f5.sensibilidad.por_semilla.seed42.gray_scott.veredicto.v1
f5.sensibilidad.por_semilla.seed42.lorenz.veredicto.v1
f5.sensibilidad.por_semilla.seed42.perlin.veredicto.v4
f5.sensibilidad.por_semilla.seed7.global.g1
f5.sensibilidad.por_semilla.seed7.gray_scott.veredicto.v1
f5.sensibilidad.por_semilla.seed7.lorenz.veredicto.v1
f5.sensibilidad.por_semilla.seed7.perlin.veredicto.v1
f5.sensibilidad.por_semilla.seed99.global.g1
f5.sensibilidad.por_semilla.seed99.gray_scott.veredicto.v1
f5.sensibilidad.por_semilla.seed99.lorenz.veredicto.v1
f5.sensibilidad.por_semilla.seed99.perlin.veredicto.v1
f5.sensibilidad.retencion_exigida.r0_75.global.g1
f5.sensibilidad.retencion_exigida.r0_75.gray_scott.veredicto.v1
f5.sensibilidad.retencion_exigida.r0_75.lorenz.veredicto.v1
f5.sensibilidad.retencion_exigida.r0_75.perlin.veredicto.v4
f5.sensibilidad.retencion_exigida.r0_80.global.g1
f5.sensibilidad.retencion_exigida.r0_80.gray_scott.veredicto.v1
f5.sensibilidad.retencion_exigida.r0_80.lorenz.veredicto.v1
f5.sensibilidad.retencion_exigida.r0_80.perlin.veredicto.v4
f5.sensibilidad.segundo_mejor.global.g1
f5.sensibilidad.segundo_mejor.gray_scott.veredicto.v1
f5.sensibilidad.segundo_mejor.lorenz.veredicto.v1
f5.sensibilidad.segundo_mejor.perlin.veredicto.v1
f5.sensibilidad.variabilidad_beto.mas.global.g1
f5.sensibilidad.variabilidad_beto.mas.gray_scott.veredicto.v1
f5.sensibilidad.variabilidad_beto.mas.lorenz.veredicto.v1
f5.sensibilidad.variabilidad_beto.mas.perlin.veredicto.v1
f5.sensibilidad.variabilidad_beto.menos.global.g1
f5.sensibilidad.variabilidad_beto.menos.gray_scott.veredicto.v1
f5.sensibilidad.variabilidad_beto.menos.lorenz.veredicto.v1
f5.sensibilidad.variabilidad_beto.menos.perlin.veredicto.v1
```

</details>

### Pruebas nuevas o modificadas (`tests/test_canonical_numbers.py`)

| Prueba | Qué comprueba |
|---|---|
| `test_referencias_desbloqueadas_tras_fase5` | Las 16 `ref.*` tienen prueba verificada desde su CSV, sin motivo «bloqueado», con hash en `inputs` y la nota de desbloqueo |
| `test_referencias_coinciden_con_observados[...]` | Igualdad exacta con `resultados.json:observados` (12 casos) |
| `test_desbloqueo_se_niega_si_no_coincide_con_observados` | `unlock_ref_test` falla si el CSV y `observados` difieren en 1e-9 |
| `test_f5_observados_desde_predicciones` | Los 56 insumos tienen el sha256 de `meta.inputs`; cada F1 observado (45 por semilla y 11 referencias) se recalcula desde su CSV; la media de semillas da `f1_img_obs` y el piso permutado |
| `test_f5_insumos_con_hash` | `resultados.json` y `f46_resultados.json` están en `inputs` con su hash |
| `test_f5_f46_valores_en_su_localizador[f5, f46]` | Cada entrada f5/f46 es exactamente el valor de su localizador; las identidades coinciden con la clave |
| `test_f5_veredictos_y_conteos` | Una sola identidad de veredicto por técnica y variante; `n_semillas_V1` principal y de A14 |
| `test_f46_condicion_1` | Condición 1 por técnica y nota PENDIENTE (P6) |
| `test_sin_cambios_previos_salvo_desbloqueo` | Frente a `dd2e462`: solo se añaden f5/f46; solo cambian las 16 `ref.*`, sin cambio en val ni en la matriz de val; `superseded` e `inputs` previos intactos |
| `test_ref_longitud` (modificada) | Ahora exige prueba presente y el localizador `split=val, split=test` |
| `test_sin_metricas_de_test_de_referencias` (retirada) | La sustituye `test_referencias_desbloqueadas_tras_fase5` |
| `leaf_cases` (modificada) | Excluye `f5.`/`f46.` de alcanzabilidad y matriz de confusión: no son F1 de una matriz |

## 3. Bitácoras, `21c3362`

- `docs/bitacora/F5.md`: **Cerrada con hallazgo transferido a F7 (RP)**.
- `docs/bitacora/F4.5.md`: **Cerrada**.
- `docs/bitacora/F4.6.md`: **Abierta** hasta F4.1 (condición 2, P6).

Las bitácoras siguen el skill `cierre-tarea`, aplicado a mano porque tiene `disable-model-invocation`.

- **Control de cifras.** Un script extrajo cada cifra decimal (≥ 3 decimales) de las tres bitácoras y la buscó, redondeada a sus decimales, en `resultados.json`, `f46_resultados.json`, `canonical_numbers.json` (val, test y test − val) y en las dos referencias citadas (F3.3 y protocolo §13). F5: 295 de 295 con fuente; F4.5: 45 de 45; F4.6: 116 de 116.
  - El control comprueba que cada valor existe en una fuente, no que esté en la celda correcta.
  - Las tablas se generaron por script desde los mismos JSON.
- **Archivos citados en «Artefactos»:** todos existen (`ls`).
- **[completar] pendientes:** uno, en F4.6 (desviación 3): el diagrama de intersección de errores que prevé `docs/plan_fase4.md` y que `run_fase5 --analisis-f46` no produce.
- **Paso 1.4 del skill:** no se hizo la ronda de preguntas, porque las instrucciones del bloque D3 cubren los datos de la plantilla.

## 4. Verificación

| Paso | Resultado | Evidencia |
|---|---|---|
| `pytest tests/` (con `CANON_UMAP_PARAMS` y `CANON_STRICT=1`; sin `F03_*`), código de `5e4a71b` | **1127 aprobadas, 30 omitidas, 0 fallidas** (165.22 s). Las omisiones son las de `tests/test_label_independence.py`, que requieren `F03_UMAP_PARAMS`, como en los bloques anteriores | `d3/pytest.log`, `d3/pytest.xml` |
| `python -m src.canonical.build --check` (`CANON_STRICT=1`) en `21c3362` | «al día (19698 líneas)» | `d3/canonical_check.log` |

## 5. Hashes (sha256)

| Archivo | sha256 (bytes) | sha256 (LF) |
|---|---|---|
| `results/canonical_numbers.json` | `269702af912b449edebdfb0b85b7907df639ecf0ecc856c67ac379b1eee39227` | `269702af912b449edebdfb0b85b7907df639ecf0ecc856c67ac379b1eee39227` |
| `results/f5_contraste/resultados.json` | `42c0ed12ab271bea2b8ddbd2b1f6dfad31c8396ff6ae720751a632d79c752261` | `42c0ed12ab271bea2b8ddbd2b1f6dfad31c8396ff6ae720751a632d79c752261` |
| `results/f5_contraste/f46_resultados.json` | `f77c5ea2d628ccb9137a6a9b64ba89077de9dec5e88abcfe0e081887fb7109ce` | `f77c5ea2d628ccb9137a6a9b64ba89077de9dec5e88abcfe0e081887fb7109ce` |
| `src/canonical/build.py` | `942e5f71d2d3d4cccb0dba81b410011dd0326053c15197b4885e0a47217e4117` | `3b09f16c0e8aff7c31e6ca7fa09fab37cb166affcc06dc2ff3498e0aad52177f` |
| `src/canonical/schema.json` | `0aa1b817da51df43cc2f63e0704f75745892ec4e5339e17b992c97bbaa54235f` | `bfdc0f2c6ddad7b2133686a43f12f575189b49049a53bdd3bf09c072f9db3b48` |
| `src/protocol/sensibilidad.py` | `bf8bc6340a38fd76a50d92f2ec308d368df1046a884c355109779e58f683c81c` | `496ef7fee9c3c2c1570639e7c4b584621c3137f8e0b46271c6f8f3b023755dae` |
| `tests/test_canonical_numbers.py` | `75c67cbd539c90849f9245c491c36dda60d77069a460f010b447cdaf8108a05b` | `cacc4f17f8b02a95b369237659f6c2262c66a13fc3bca28678ab5a3ea0d1c9a3` |
| `docs/bitacora/F5.md` | `f8cd65298cd48cce35d020ca34ded0218d67adc9cc6d6273a4f30d5ff336edd7` | `f8cd65298cd48cce35d020ca34ded0218d67adc9cc6d6273a4f30d5ff336edd7` |
| `docs/bitacora/F4.5.md` | `8f6a52410cdc4bc52faec2b3c0b16d51551117c5b2d059d5b1b2300fe5506c6d` | `8f6a52410cdc4bc52faec2b3c0b16d51551117c5b2d059d5b1b2300fe5506c6d` |
| `docs/bitacora/F4.6.md` | `c4548fbaeb6bb05b348713ca3029e428e631e9c97e91c7d7a3b1dcb36986e31e` | `c4548fbaeb6bb05b348713ca3029e428e631e9c97e91c7d7a3b1dcb36986e31e` |

`results/canonical_numbers.json:inputs` registra los dos JSON de la Fase 5 con su sha256 LF:
- `resultados.json`: `42c0ed12ab271bea2b8ddbd2b1f6dfad31c8396ff6ae720751a632d79c752261`
- `f46_resultados.json`: `f77c5ea2d628ccb9137a6a9b64ba89077de9dec5e88abcfe0e081887fb7109ce`

## 6. PR

**Pendiente.** `gh auth status`: «You are not logged into any GitHub hosts». La rama está en `origin/sv/f5-resultados`; el PR se abre cuando haya sesión de `gh`. No se fusiona.
