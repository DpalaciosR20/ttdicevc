# F3.3 — Etapa B: migración del colector F3.2 y claves f33.*

> Responsable: SV. Fecha: 2026-11-05. Rama `sv/f3.3-seleccion-finalista` (base `06b95c4`).
> Atiende F0.4 y la H6 de la bitácora de F3.2. **Sin commit**: los cambios quedan en el árbol para revisión. No se escribió bitácora ni se reabrieron D1, D2, D3 y D5.
> Acceso a test: ningún código nuevo carga filas de test. La única excepción, autorizada por SV antes de ejecutarla, es la reejecución de `potencia.py` (punto 3), que lee el test de BETO como siempre: ya se consultó en F2.2 y lo permite el protocolo §10. No se usó ningún test visual.

Archivos modificados (sin commit):

| Archivo | Cambio |
|---|---|
| `src/canonical/build.py` | Colector F3.2 migrado a la repetición; colectores nuevos `collect_f32v0` y `collect_f33`; `collect_structural` toma val de la repetición |
| `tests/test_canonical_numbers.py` | `test_f32_estructura` reescrita para la repetición; pruebas nuevas `test_f32v0_estructura`, `test_f33_presencia` y `test_f33_consistencia` (×3) |
| `results/f11_potencia/potencia.py` | Prefijo de las claves leídas: `f32.` → `f32v0.` (2 líneas) |
| `results/canonical_numbers.json` | Regenerado con `CANON_STRICT=1` |

Evidencia nueva en esta carpeta: `diff_canonico.log`, `pruebas_negativas.log`, `f33_vs_etapa_a.log` y `pytest_etapa_b.log`.

---

## 0. Comprobación previa (bloqueante)

```
git diff --stat protocolo-v1 HEAD -- src/selection/      → (vacío)
git ls-tree -r --name-only protocolo-v1 -- src/selection/ → src/selection/__init__.py, src/selection/select_finalist.py
```

**`src/selection/` es idéntico, byte a byte, en `protocolo-v1` y en `HEAD`.** `select_finalist` y `select_second` ya existían en la etiqueta con la misma lógica, sin ningún cambio, ni siquiera de formato. La copia de trabajo no tenía cambios en esa carpeta. No hubo que detenerse: D2 se aplica tal cual.

---

## 1. `build.py`: colector F3.2 de la repetición

### a) Lectura

- `load_f32_predictions()` llama a `load_predictions("results/predictions/primera_pasada/*.csv")`, es decir, al validador F1.5 completo.
- `collect_f32` agrupa por `(technique, arch, strategy, seed)` y recorre los 54 CSV.
- El JSON de cada corrida se toma de `config.run_json` del manifiesto, dentro de `results/f32_primera_pasada/runs/`. Se comprueba que el JSON corresponda a ese CSV: el nombre del CSV debe coincidir con la técnica, arquitectura, estrategia y semilla del JSON.

### b) Integridad por corrida

Sustituye a la comprobación de `grid_meta.evaluate_test`. Cada punto lanza `CanonicalError` si falla:

- exactamente 54 CSV y 54 grupos;
- solo `split == 'val'` y fase `primera_pasada` en el DataFrame;
- `fase == "primera_pasada"` en el manifiesto y en el JSON;
- `evaluate_test is False`;
- `f1_test`, `acc_test`, `n_test`, `test_preds`, `test_probs` y `test_true` nulos;
- `csv_sha256` del manifiesto igual al sha256 del archivo;
- variante del JSON igual a la del manifiesto;
- las 114 filas de val en orden;
- `ŷ = p_dep > 0.5` y `y_true` iguales a `val_preds` y `val_true` del JSON;
- `y_true` igual en las 54 corridas;
- F1 igual al de sklearn y a `best_f1_val` (tolerancia 1e-12) y a `summary.csv` (4 decimales).

**Pruebas negativas** (`pruebas_negativas.log`, sobre copias en un directorio temporal, sin tocar el repo). Las 8 alteraciones hacen fallar el colector con un mensaje claro:

- `evaluate_test=true`;
- `f1_test` no nulo;
- fase distinta en el JSON;
- fase distinta en el manifiesto;
- `csv_sha256` distinto;
- un CSV faltante (53);
- `val_preds` alterado;
- una fila marcada como `test` (inyectada en memoria; no es una fila real de test).

### c) Claves `f32.*`

Se generan 54 claves por corrida, `f32.<tec>.<arq>.<est>.<variante>.seed<k>.f1_macro`, con:

- `status` verificado y matriz de confusión de val;
- `source` igual al CSV F1.5 (`results/predictions/primera_pasada/<…>.csv`, `split=val`). Así, `test_verificado_desde_predicciones` las recalcula desde el archivo.

Las derivadas son las mismas que antes: 18 `f1_macro_mean` (`mean`) y 18 `f1_macro_std` (`std_ddof1`), con `_check_csv` contra `results/f32_primera_pasada/summary_by_config.csv`.

Los identificadores son los de la repetición: `vit_b16`, `px224_flip` en Gray-Scott y `px224_noflip` en Perlin y Lorenz. El motivo de test nulo es nuevo y está en `TEST_BLOCKED_F32`.

### d) v0 → `f32v0.*`

`collect_f32v0` es el colector anterior sin cambios de lógica, con estas diferencias:

- lee de `F32V0_DIR = results/f32_primera_pasada_v0/`;
- usa el prefijo `f32v0.`;
- añade `note = F32V0_NOTE` en las 90 entradas: «Primera pasada v0 de F3.2 (21 oct 2026), sustituida por la repetición (claves f32.*): preprocesamiento anterior a F1.4 (volteo en las tres técnicas) y formato anterior a F1.5. Histórica: no participa en la selección de F3.3 (matriz §5).»

Las 90 entradas son idénticas a las 90 `f32.*` anteriores, salvo el prefijo en `depends_on` y la nota (ver §4).

### e) `collect_structural`

- **Sin pkl:** `n.val` y `support.val.label{0,1}` salen ahora de la repetición (`y_true` del primer CSV F1.5, igual en los 54); antes salían de `predictions.csv` de la v0.
- **Con pkl (`CANON_STRICT=1`):** como antes, salen de `umap_params.pkl`. Se añadió en ambas vías una comprobación que exige que coincidan con la repetición.
- Resultado: 114; 70/44, igual que los valores actuales. `test_conteos_sin_pkl_coinciden` lo confirma entre ambas vías.

*Nota:* en el archivo commiteado (vía pkl), la fuente de `n.val` sigue siendo `umap_params.pkl`, como antes. La repetición entra como comprobación, no como fuente, para que esas tres cifras no cambien de procedencia.

### f) Insumos con hash

Se sigue la convención existente (`text_sha256`, con fines de línea normalizados a LF). Se añadieron 167 insumos y no se eliminó ni cambió ninguno:

- 54 CSV y 54 manifiestos de `results/predictions/primera_pasada/`;
- 54 `runs/*.json` de la repetición;
- `summary.csv`, `summary_by_config.csv` y `grid_meta.json` de la repetición;
- `configs/matriz_experimentos.json`, que lee `select_finalist`;
- `results/f33_seleccion/fragilidad_volteos.csv`.

Los insumos de la v0 conservan su ruta y su hash.

### Resumen del diff de código

`git diff --numstat`: `build.py` +291/−50, `test_canonical_numbers.py` +118/−8, `potencia.py` +2/−2. En `build.py`:

- **Constantes:** `F32_DIR`, `F32_PRED_DIR`, `F32_FASE`, `F32_RUNS`, `F32V0_DIR`, `F33_VOLTEOS`, `TEST_BLOCKED_F32`, `F32V0_NOTE` y `RUN_TEST_FIELDS`.
- **Funciones auxiliares extraídas** para compartirlas entre v0 y repetición: `_f32_cfg`, `_read_csv_dict`, `_check_f1` y `_f32_derived`. La lógica de comprobación del F1 es la misma de antes.
- **Colectores:** `load_f32_predictions`, `collect_f32` (nuevo), `collect_f32v0` (el anterior, con prefijo y nota) y `collect_f33`.
- **`collect_structural(umap_path, val_labels, inputs)`:** recibe `y_true` de la repetición en lugar de las filas de la v0.
- **`build()`:** cuatro colectores en código.
- Se añadió `REPO` a `sys.path` para importar `src.predictions` y `src.selection` también cuando build.py se importa desde fuera del paquete.

No se tocaron la fusión, las derivadas, el esquema ni `dumps`.

---

## 2. Claves `f33.*`

`collect_f33` aplica `sf.select_second(sf.config_table(sf.f1_by_run(preds, matrix), matrix))` a las mismas predicciones de `collect_f32`; es el código de `src/selection/select_finalist.py`, sin reimplementarlo. Son 10 claves por técnica, 30 en total.

El esquema solo admite valores numéricos, así que **la identidad (arquitectura y estrategia) va en la clave**:

| Clave | Tipo / derivación | Valor y comprobación |
|---|---|---|
| `f33.<tec>.finalista.<arq>.<est>.f1_macro_mean` / `_std` | métrica, `mean` / `std_ddof1` sobre las 3 claves `f32.*` por semilla de esa configuración | Debe coincidir con la tabla de `select_finalist` (tolerancia 1e-12); si no, error |
| `f33.<tec>.segundo.<arq>.<est>.f1_macro_mean` / `_std` | ídem, configuración de `select_second` | ídem |
| `f33.<tec>.lider.<arq>.<est>.f1_macro_mean` / `_std` | ídem, líder por media | ídem; error si hay más de un líder (no representable) |
| `f33.<tec>.limite_empate.f1_macro` | `diff` [media del líder, DE del líder] | = `f1_val_mean − tolerance` del líder en `select_finalist` |
| `f33.<tec>.holgura_finalista.f1_macro` | `diff` [media del finalista, límite] | ≥ 0 (lo exige la prueba) |
| `f33.<tec>.volteos_min.una_corrida` | conteo, `extraido` | mínimo de `min_volteos` de las filas «una corrida» de `fragilidad_volteos.csv` |
| `f33.<tec>.volteos_min.repartido` | conteo, `extraido` | `min_volteos` de la fila «repartida» |

Todas las métricas son `verificado`, porque heredan el status de las claves `f32.*`, y tienen test nulo con el motivo heredado. Cada entrada lleva `source` (`src/selection/select_finalist.py` o el CSV de volteos), un `evaluator` y una `note` que dice qué es.

Para los volteos, el colector exige además que todos los `min_volteos` sean enteros y que el CSV se refiera al mismo finalista que `select_finalist` (aparece en `corrida` y no en `nuevo_finalista`).

| Técnica | Finalista | Segundo | Líder | Límite | Holgura | Volteos (una / repartido) |
|---|---|---|---|---|---|---|
| gray_scott | efficientnet_b0/finetune 0.7175762313434474 ± 0.012496044907801315 | resnet18/finetune | resnet18/finetune 0.7197389205827752 ± 0.011332519182255764 | 0.7084064014005195 | 0.00916982994292792 | 2 / 2 |
| perlin | vit_b16/frozen 0.7028471978891604 ± 0.007259371318553099 | efficientnet_b0/frozen | vit_b16/frozen (= finalista) | 0.6955878265706072 | 0.007259371318553143 | 7 / 6 |
| lorenz | vit_b16/frozen 0.7148477253740412 ± 0.004722459481409753 | efficientnet_b0/finetune | efficientnet_b0/finetune 0.7165572154787841 ± 0.0017937274622905953 | 0.7147634880164936 | 8.423735754758965e-05 | 1 / 1 |

**Comparación con `reporte_etapa_a.md` / `tabla_seleccion.csv`** (`f33_vs_etapa_a.log`):

- Coinciden los finalistas, los segundos, los líderes y los volteos.
- La diferencia numérica máxima es 2.220446049250313e-16 (1 ulp), en la media del finalista y en la holgura de Gray-Scott. Se debe a que la Etapa A usó `statistics.fmean` y el build usa `np.mean`.

**Ninguna DISCREPANCIA.**

---

## 3. `potencia.py`

**Cambio:** solo el prefijo de las claves que lee, en las líneas 222 (`canonical_values`) y 294 (`block_e`): `f32.` → `f32v0.`. No hay otras lecturas de claves `f32`. `F32_PRED` ya apuntaba a la v0.

**Reejecución:**

```
python results/f11_potencia/potencia.py --out <scratchpad>/f11_rerun --jobs 6
```

Se usaron los mismos `jobs` que la corrida versionada. Tardó 5.6 min y terminó con código 0. Se escribió en el scratchpad para no sobrescribir lo versionado.

| Salida | Comparación con `results/f11_potencia/` |
|---|---|
| `empirico.csv`, `empirico_corridas.csv`, `malla_escenarios.csv`, `retencion_minima.csv`, `potencia_piso.csv` | **idénticas** (sin `\r`) |
| `curva_potencia.png`, `curva_potencia.pdf` | **idénticas byte a byte** |
| `potencia.log` | **idéntico** tras enmascarar solo tiempos, fecha, commit, ruta de la v0 y sha de `canonical_numbers.json` (ver abajo) |

Diferencias del log, todas de rutas, fechas o hashes de insumo:

- `Inicio:` cambia la fecha y el commit (`06b95c4`, «árbol con cambios»).
- La duración por escenario (`; NN s`) y la de E (`0.3 s` frente a `0.1 s`) cambian; la total, 6.6 min frente a 5.6 min.
- La línea de hash de las predicciones v0 cambia de etiqueta, de `results/f32_primera_pasada/predictions.csv` a `results/f32_primera_pasada_v0/predictions.csv`, con el **mismo sha256** `448b457f…15ed`. La corrida versionada es anterior al cambio de nombre de la carpeta.
- `sha256 results/canonical_numbers.json` cambia de `f94c37f6…` a `bfb77c09…`, porque el archivo se regeneró (se esperaba).

La línea `E: F1-val recalculado de las 54 corridas vs canonical_numbers.json, máx |dif| = …` es idéntica: las claves `f32v0.*` dan exactamente las mismas cifras que las `f32.*` anteriores. **No cambió ninguna cifra; no hubo que revertir.**

*Pendiente fuera de alcance (no modificado):* el README de F1.1 y los insumos que lista siguen citando el sha de `canonical_numbers.json` previo a F1.1. Es historial de esa corrida y sigue siendo correcto como tal.

---

## 4. Regeneración y pruebas

```
CANON_UMAP_PARAMS="G:/Mi unidad/tt2-insumos/umap_params.pkl" CANON_STRICT=1 python -m src.canonical.build
python -m src.canonical.build --check   → al día (7620 líneas)
```

El sha256 del pkl coincide con la referencia `081f5bab…`.

### Diff de claves (`diff_canonico.log`)

| | Antes | Después |
|---|---|---|
| Claves en `numbers` | 156 | 276 |
| Añadidas | — | 190: `f32` 70, `f32v0` 90, `f33` 30 |
| Eliminadas | 70 | todas `f32.*` de la v0 con nombre que ya no existe (`vit_b_16`, y Perlin/Lorenz `px224_flip`) |
| Comunes | 86 | 20 de ellas son `f32.gray_scott.{resnet18,efficientnet_b0}.*.px224_flip.*`: mismo nombre en v0 y repetición |
| Comunes con valor cambiado | — | 4, todas `f32.*`: Gray-Scott ResNet-18 FT `seed123`, `seed7`, `mean` y `std` |
| Comunes con otros campos cambiados | — | 16 `f32.gray_scott.*`: solo `source` (ahora el CSV F1.5 / summary de la repetición) y `null_reason` |
| `inputs` | 66 | 233 (167 añadidos; 0 eliminados; 0 hashes cambiados) |
| `superseded` | — | igual |

- **Fuera de `f32.*`, `f32v0.*` y `f33.*`, ningún valor cambió ni se eliminó.** Lo mismo vale para `n.*`, `support.*`, `saturation.*`, `ref.*`, `floor.*`, `witcom.*`, `m2.*`, `f11.*` y `f12.*`.
- **v0 → `f32v0.*`:** las 90 claves `f32.*` anteriores (incluidas las 20 que conservan nombre) son idénticas, campo por campo, a su `f32v0.*`, salvo el prefijo en `depends_on` y la nota.
- **Los 4 valores cambiados** son los de F3.2 H5 (Gray-Scott, ResNet-18 FT): `mean` 0.7278422523103375 → 0.7197389205827752, `std` 0.01457631918003135 → 0.011332519182255764, `seed123` 0.7304964539007093 → 0.71433344470431, `seed7` 0.740909090909091 → 0.7327621049228032. Coinciden con F3.2 §4.5 y H5: 2 corridas de ResNet-18 FT difieren de la v0 y la media baja de 0.7278 a 0.7197. La tercera corrida distinta de H5 (ViT-B/16 FT) cambió también de nombre de clave (`vit_b_16` → `vit_b16`), así que no aparece entre las comunes.

### Pruebas

`tests/test_canonical_numbers.py`:

- `test_f32_estructura` (reescrita): 54 CSV de `primera_pasada`, solo val, mismo `y_true` (70/44); 54 + 18 + 18 claves `f32.*`, test nulo; arquitecturas del vocabulario F1.5; variante igual a `configs/preprocesamiento.json` por técnica; fuente bajo `results/predictions/primera_pasada/`.
- `test_f32v0_estructura` (nueva): 54 corridas de la v0 con el mismo `y_true`; 54 + 18 + 18 claves `f32v0.*`, todas con `note == F32V0_NOTE`, `px224_flip`, test nulo, dependencias dentro de `f32v0.*` y fuente en la v0.
- `test_f33_presencia` (nueva): 30 claves; por técnica, exactamente una configuración por rol (finalista, segundo, líder) y las claves de límite, holgura y volteos.
- `test_f33_consistencia[perlin|gray_scott|lorenz]` (nueva): vuelve a correr `select_finalists` + `select_second` sobre los 54 CSV y exige:
  - misma identidad por rol;
  - cifras f33 iguales a la media/DE `f32.*` de esa configuración, con las mismas dependencias, y a la tabla de `select_finalist` (1e-12);
  - `depends_on` del límite y de la holgura en el orden correcto, y límite = media − tolerancia del líder;
  - holgura ≥ 0, test nulo;
  - volteos iguales al CSV, con 1 ≤ repartido ≤ una corrida;
  - hash del CSV de volteos en `inputs`.
- Las pruebas genéricas cubren automáticamente lo nuevo: alcanzabilidad, matriz de confusión, recálculo desde `results/predictions/` (ahora también las 54 `f32.*`), derivadas y coincidencia con el archivo commiteado.

**Suite completa** (`pytest_etapa_b.log`), con `CANON_UMAP_PARAMS` y `CANON_STRICT=1`:

```
636 passed, 30 skipped, 14 warnings in 66.00s
```

Antes de los cambios eran 409 aprobadas y 30 omitidas. Las +227 son casos parametrizados nuevos: 54 × 3 de las `f32.*` por corrida (alcanzabilidad, confusión y recálculo desde predicciones), 60 derivadas (36 `f32.*` + 24 `f33.*`) y 5 pruebas nuevas.

Las **30 omitidas son las mismas que antes**: `tests/test_label_independence.py`, por «Define F03_UMAP_PARAMS para correr esta prueba». No dependen de esta tarea. `tests/test_canonical_numbers.py` sola: 479 aprobadas, 0 omitidas.

---

## Resumen

- **Comprobación 0:** `src/selection/` es idéntico en `protocolo-v1` y `HEAD`.
- **DISCREPANCIAS con `reporte_etapa_a.md`: ninguna.** Finalistas, segundos, líderes, límites, holguras y volteos coinciden; la diferencia máxima es 1 ulp.
- **`potencia.py`:** las 7 salidas son idénticas; el log solo difiere en rutas, fechas, tiempos y el hash del archivo canónico.
- **Diff de claves:** fuera de `f32/f32v0/f33` no cambia ningún valor. Los 4 valores `f32.*` que cambian son los esperados por F3.2 H5.
- **Decisión de diseño que pongo a revisión:** como el esquema solo admite números, la identidad del finalista, el segundo y el líder va codificada en el nombre de la clave (`f33.<tec>.<rol>.<arq>.<est>.*`). La alternativa sería ampliar `schema.json` con un tipo de entrada «etiqueta», que no hice para no tocar el esquema.
