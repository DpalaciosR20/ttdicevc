# F3.3 — Etapa C: ajustes finales, entregable para CE, bitácora y commit

> Responsable: SV. Fecha: 2026-11-05. Rama `sv/f3.3-seleccion-finalista`.
> No se cargaron ni usaron filas de test en esta etapa. No se reabrieron D1–D5 ni la codificación de la identidad en el nombre de las claves `f33.*`; `schema.json` no se modificó.

---

## 1a. Referencias a claves `f32.` fuera de `build.py`, `potencia.py` y `test_canonical_numbers.py`

**Búsqueda.** Primero, en todo el árbol (`*.md`, `*.ipynb`, `*.py`, `*.json`, `*.csv`, `*.txt`, `*.log`, sin `.venv/` ni `.git/`):

```
grep -rnI -E "f32\.[a-z_]"
```

Después, sobre los archivos versionados:

```
git grep -n -I -E "f32[.](\*|<|\{|[a-z_]+[.])" -- . ':!src/canonical/build.py' ':!results/f11_potencia/potencia.py' ':!tests/test_canonical_numbers.py' ':!results/canonical_numbers.json'
git grep -n -I -E "'f32'|\"f32\"|startswith\(.f32" -- '*.py' '*.ipynb'
```

Se revisaron además los 23 archivos que leen `canonical_numbers.json` (docs, notebooks y scripts). Ninguno, salvo `potencia.py`, construye o lee claves `f32`.

| Archivo | Qué cita | Tipo de clave | Acción |
|---|---|---|---|
| `docs/bitacora/F3.2.md:162` (H6) | «Las cifras canónicas `f32.*` todavía son las de la v0» (prefijo genérico) | Las 90 de la v0: 70 eliminadas y **20 que cambiaron de significado** (Gray-Scott ResNet-18/EfficientNet-B0, `px224_flip`) | Bitácora cerrada: solo se informa. Su H6 queda cerrada por F3.3 H7 |
| `results/f32_primera_pasada_v0/README.md:3` | «Siguen leyéndola `src/canonical/build.py` (cifras `f32.*` de `canonical_numbers.json`, hasta que SV migre el colector)» | Ídem; además la frase quedó desactualizada: ahora son `f32v0.*` | No es script; no se editó. Se informa en F3.3 H7. Corrección sugerida: «(cifras `f32v0.*`)» |
| `results/f04_cifras_canonicas/pytest.log`, `junit.xml` y los `pytest.xml` de `f11_potencia`, `f12_protocolo`, `f14_preprocesamiento` (`f14_pytest.xml`), `f21_techo5d`, `f22_referencias_textuales` y `f23_pisos` | Identificadores de casos de prueba con las 90 claves de la v0 (144 apariciones por archivo) | **Eliminadas:** `vit_b_16` en las 3 técnicas y `px224_flip` en Perlin y Lorenz. **Cambiaron de significado:** las 20 `f32.gray_scott.{resnet18,efficientnet_b0}.*.px224_flip.*` | Evidencia histórica de ejecuciones ya hechas: no se edita |
| `results/f24_descriptores/pytest_1_resto.xml` | Ídem | Ídem | Ídem |
| `notebooks/F1.1_potencia.ipynb:573` | Salida de texto con la ruta `results/f32_primera_pasada/predictions.csv` | No es una clave (ruta antigua de la v0) | Salida histórica: no se edita |
| `docs/bitacora/F1.5.md`, `results/f15_formato/README.md` | `validate_f32.txt` y la ruta `results/f32_primera_pasada/predictions.csv` | No son claves | — |

**Scripts activos que lean claves de la v0:** ninguno, aparte de `potencia.py`, ya pasado a `f32v0.*` en la Etapa B. No se editó ningún archivo en este punto.

**Por qué 20 claves cambiaron de significado.** Son las `f32.*` de Gray-Scott con ResNet-18 y EfficientNet-B0: la variante (`px224_flip`) y la arquitectura no cambiaron de nombre entre la v0 y la repetición. Hoy designan la repetición. 16 conservan el mismo valor y 4 cambiaron (ResNet-18 FT `seed7`, `seed123`, `mean` y `std`; `diff_canonico.log`). Una lectura antigua de cualquiera de esas 20 claves obtendría hoy la cifra de la repetición.

## 1b. Convención de `superseded`

**Uso actual.** Hay una sola entrada: `ref.beto_exp41.f1_macro`, con `by = results/f22_referencias_textuales/canonical_fragment.json:ref.beto_exp41.f1_macro` y `origin = results/f04_cifras_canonicas/canonical_fragment.json`. Es la cifra reportada de TT-I (0.7586, inalcanzable con los soportes 70/44) sustituida por la verificada en F2.2.

**Convención, según el código y la documentación:**

- `build.py`, docstring y `merge()`: una entrada que declara `supersedes=<id>` retira a `<id>` de `numbers`, y la sustituida se conserva intacta en `superseded`.
- `docs/especificacion_f22.md` §5: «sustituyen la entrada `reportado` de Exp 4.1, mediante el mecanismo de sustitución».
- F2.1 y F2.3 conservan las cifras WITCOM «sin `supersedes`» cuando son otra medición.
- `test_superseded_se_conserva_intacta` exige que `origin` sea un fragmento (`results/<tarea>/canonical_fragment.json`) cuya entrada sea idéntica a la guardada.

Es decir, `superseded` registra **una corrección de la misma cifra**: una entrada errónea o transcrita sustituida por la correcta, con el origen en un fragmento.

**No se registraron las 4 claves `f32.gray_scott.resnet18.finetune.*`**, por tres razones:

1. **Son otra medición, no una corrección.** La v0 y la repetición son dos ejecuciones distintas del grid (otro entorno de Kaggle, determinismo y optimizador nominal; F3.2 H5). Ambas son correctas para su ejecución.
2. **La v0 ya está conservada íntegra en `f32v0.*`**, y esa conservación forma parte de D5.
3. **El mecanismo no admite el caso sin efectos indeseados.**
   - Si las `f32.*` nuevas declararan `supersedes=f32v0.…`, `merge()` retiraría esas 4 entradas de `f32v0.*`, que dejaría de estar completa (contra D5).
   - Si se declarara `supersedes` con la misma clave, habría que emitir una segunda vez la entrada v0 con nombre `f32.*` desde un colector. `test_superseded_se_conserva_intacta` fallaría, porque su `origin` no sería un fragmento.

La relación entre esas 4 claves y su `f32v0.*` queda documentada en F3.3 H7, en `diff_canonico.log` y en `reporte_etapa_b.md` §4.

## 1c. Reejecución de `fragilidad.py`

Fuente: `etapa_c_reproduccion.log`.

| Comprobación | Resultado |
|---|---|
| `python results/f33_seleccion/fragilidad.py` | código 0 |
| `fragilidad_volteos.csv` frente a la versión de la etapa A | **idéntico byte a byte** |
| `fragilidad_volteo_unico.csv`, `epocas_finalista_segundo.csv` | idénticos |
| `fragilidad.log` | idéntico salvo las 4 líneas de `time` que tenía la versión de la etapa A |
| sha256 de `fragilidad_volteos.csv` (`text_sha256`) | `c40695d91a0e7a1488e1ab7a6e18e8c12b16541d15e7ffb6ae129ff5405a6d80` = `inputs` de `canonical_numbers.json` |

## 2. Configuraciones de las fases posteriores (entregable para CE)

| Fase | Cómo obtiene las configuraciones | Fuente |
|---|---|---|
| `hpo` (F3.4) | `src/training/run_hpo.py`, rama **`ce/f3.4-hpo` (commit `2c1a8c2`, sin fusionar)**: `--config TEC:ARQ:EST` obligatorio; `check_finalist()` llama a `select_finalists()` sobre `results/predictions/primera_pasada/*.csv` y se niega si no es el finalista sugerido, salvo `--override-selection "<motivo>"` | `git show origin/ce/f3.4-hpo:src/training/run_hpo.py`, líneas 19–20 y 153–166 |
| `final` (F3.5) | `run_grid.py --evaluate-test --configs TEC:ARQ:EST …`, más `--lr/--weight-decay/--batch-size` de F3.4 (`hpo_result.json` de `run_hpo.py` da «banderas para run_grid»). **No** se contrasta con `select_finalist` | `src/training/run_grid.py`: `parse_configs`, `main` (`--configs`: «lista explícita de configuraciones (finalistas, segundo mejor)») |
| `control_permutacion` (F3.6) | Ídem, con `--permute-labels 99` | ídem; matriz §7 |
| `sensibilidad` (F5.5) | `run_grid.py --evaluate-test --sensitivity --configs …`, con los hiperparámetros de la matriz (no se pueden cambiar). **No** se contrasta con `select_second` | ídem; `resolve_hparams` |

La matriz no define ningún archivo de selección. §4 da las fases y las semillas; §6 dice «HPO de cada finalista»; §11.10 pide que `run_grid.py` admita la fase `sensibilidad` «(configuraciones de `select_second`, compuerta de `--evaluate-test` igual a `final`)». `run_grid.py` ya admite la fase y la compuerta, pero no toma las configuraciones de `select_second`.

**Conclusión:**

- `hpo` las calcula con `select_finalist`: solo se documenta.
- Para `final`, `control_permutacion` y `sensibilidad`, ningún código espera un archivo y no hay un formato definido. **No se generó ningún archivo** ni se inventó un formato.
- Queda como pendiente para CE (F3.3 H10): que `run_grid.py` contraste `--configs` con `select_finalist` en `final` y `control_permutacion`, y con `select_second` en `sensibilidad`, como hace `run_hpo.py`.

Valores que corresponden hoy a `--configs` (informativo, de `select_finalist.log`):

- `final` / `control_permutacion`: `gray_scott:efficientnet_b0:finetune perlin:vit_b16:frozen lorenz:vit_b16:frozen`
- `sensibilidad`: `gray_scott:resnet18:finetune perlin:efficientnet_b0:frozen lorenz:efficientnet_b0:finetune`

**Tabla de selección:** `results/f33_seleccion/tabla_seleccion.csv` quedó versionada en `58f0fa2` como entregable del plan («Tabla de selección», fila F3.3 de `plan/plan_tt2_v2.md`).

## 3. Bitácora

`docs/bitacora/F3.3.md`, con la estructura de `docs/bitacora/F3.2.md` y de la plantilla. Contiene los hallazgos H1–H9 indicados y uno nuevo, **H10**, que sale del punto 2: `run_grid.py` no contrasta las configuraciones. H7 incorpora los resultados de 1a y 1b. Las desviaciones son tres:

1. no existe F3.1.md;
2. la lectura autorizada del test de BETO;
3. `fragilidad_volteos.csv` va en el commit de cifras canónicas.

La tabla de pendientes incluye todos los indicados, más H10 para CE. No quedó ningún **[completar]**.

**Tabla de control de cifras de la bitácora** (cifra → archivo):

| Cifra | Fuente |
|---|---|
| 54/54 válidos | `validate.log` (última línea) |
| 6,156 filas `val` | `reporte_etapa_a.md` §1b (54 × 114; `verificar_insumos.log` sin problemas) |
| 44/70 por corrida | `verificar_insumos.log` («positivos por corrida [44]»); `f1_recalculado.csv` (`n_pos_true`) |
| commit `36c6398d…` | `verificar_insumos.py` (constante comparada) y `verificar_insumos.log` (0 problemas) |
| diferencias 0.0 y 4.864864864873031e-05 | `verificar_insumos.log` |
| 0 valores cerca de 0.5; mínimo 0.00011577456982653223 | `verificar_insumos.log` |
| etiqueta 00:14:54Z, commit 01:52:27Z | `git_protocolo.log` (−06:00 convertido a UTC) |
| primera corrida 02:20:35Z | `verificar_insumos.log` («min 2026-11-03T02:20:35+00:00») |
| 1 h 38 min, 2 h 06 min | diferencias de las tres horas anteriores (1:37:33 y 2:05:41, redondeadas) |
| medias, DE, holguras, entrenables y roles (4.2) | `tabla_seleccion.csv` |
| límites 0.7084 / 0.6956 / 0.7148 | `tabla_seleccion.csv` (`limite_empate`); claves `f33.<tec>.limite_empate.f1_macro` |
| 2.220446049250313e-16, 9.8879238130678e-17 | `regla_independiente.log` |
| tabla de tolerancias (4.3) | `fragilidad_tolerancias.csv` |
| volteos 2/2, 7/6, 1/1 y nuevos finalistas | `fragilidad_volteos.csv` |
| \|ΔF1\| 0.007977–0.011134 | `fragilidad_volteo_unico.csv` (mín. de `abs_delta_min`, máx. de `abs_delta_max`) |
| épocas y F1 de las últimas 4 (4.4) | `epocas_finalista_segundo.csv` |
| Gray-Scott ResNet-18 cong. s42: 0.7063 y 0.6958, 0.6742, 0.6761 | `results/f32_primera_pasada/runs/gray_scott__resnet18__frozen__px224-flip__seed42.json` (`history.f1_val`) |
| 156 → 276 claves; 90/90; 4 valores; 0.7278 → 0.7197; 66 → 233 insumos | `diff_canonico.log` |
| 8/8 pruebas negativas | `pruebas_negativas.log` |
| 5.6 min; versiones del entorno | `potencia_reejecucion.log` |
| 66 s (pytest, etapa B) | `pytest_etapa_b.log` |
| 636 / 30 / 0 y «al día (7620 líneas)» | `pytest_etapa_c.log`, `check_etapa_c.log` |
| §9: 0.7176 ± 0.0125, 0.7028 ± 0.0073, 0.7148 ± 0.0047 | `tabla_seleccion.csv` |

Todos los archivos citados en «Artefactos» existen (comprobado con `ls` antes del commit; ver §4).

## 4. Commits, pruebas y push

**Suite completa**, antes del último commit (`pytest_etapa_c.log`), con `CANON_UMAP_PARAMS="G:/Mi unidad/tt2-insumos/umap_params.pkl"` y `CANON_STRICT=1`:

```
python -m pytest -q tests -rs
636 passed, 30 skipped, 14 warnings in 63.45s (0:01:03)
```

Las 30 omitidas son `tests/test_label_independence.py`: «Define F03_UMAP_PARAMS para correr esta prueba». Son las mismas que antes de F3.3.

**`--check`** (`check_etapa_c.log`):

```
python -m src.canonical.build --check
...\results\canonical_numbers.json: al día (7620 líneas)
```

**Commits:**

| Hash | Mensaje |
|---|---|
| `60b3e99` | `[F3.3, F0.4] Colector F3.2 de la repetición, f32v0.* histórica y claves f33.*` (build.py, pruebas, potencia.py, canonical_numbers.json y `fragilidad_volteos.csv`, que es insumo del build) |
| `58f0fa2` | `[F3.3] Tabla de selección, scripts de verificación y evidencia de las etapas A y B` |
| (este commit) | `[F3.3] Bitácora de cierre y reporte de la etapa C` (bitácora, este reporte, `pr_descripcion.md` y la evidencia de la etapa C). Su hash no puede figurar en el propio archivo |

Push de la rama a `origin/sv/f3.3-seleccion-finalista`, sin fusionar y sin crear etiquetas.

## DISCREPANCIAS

Ninguna. Las 3 salidas de `fragilidad.py` reproducen las de la etapa A byte a byte, el hash de insumo coincide y la suite está en verde.

Observaciones que no son discrepancias:

- las 20 claves de Gray-Scott que cambiaron de significado (1a);
- el README desactualizado de la v0 (1a);
- la falta de comprobación de configuraciones en `run_grid.py` (2, H10).
