# F3.3 — Etapa A: verificación y cálculo de la selección del finalista

> Responsable: SV. Fecha: 2026-11-05. Rama `sv/f3.3-seleccion-finalista` (= `main` @ `06b95c4`, árbol limpio salvo esta carpeta).
> Alcance: solo verificación y cálculo. No se tomó ninguna decisión metodológica, no se escribió bitácora, no se hizo commit, no se modificó código existente y no se cargó ni usó ninguna fila del split de prueba (ver §0.4).

Archivos de esta carpeta:

| Archivo | Contenido |
|---|---|
| `verificar_insumos.py` → `verificar_insumos.log`, `f1_recalculado.csv` | Puntos 1b–1d, 1f y 2 |
| `regla_independiente.py` → `regla_independiente.log`, `tabla_seleccion.csv`, `fragilidad_tolerancias.csv` | Puntos 3 y 4a |
| `fragilidad.py` → `fragilidad.log`, `fragilidad_volteo_unico.csv`, `fragilidad_volteos.csv`, `epocas_finalista_segundo.csv` | Puntos 4b y 5 |
| `validate.log` | `src.predictions.validate` sobre los 54 CSV |
| `select_finalist.log`, `seleccion.csv`, `seleccion_corridas.csv` | `src.selection.select_finalist` (incluye `select_second`) |
| `git_protocolo.log` | Punto 1e |

Comandos (desde la raíz, `.venv` local, Python 3.13.2, CPU, `PYTHONIOENCODING=utf-8`; todo tarda < 10 s):

```
python -m src.predictions.validate "results/predictions/primera_pasada/*.csv"      > validate.log
python -m src.selection.select_finalist --out results/f33_seleccion/seleccion.csv   > select_finalist.log
python results/f33_seleccion/verificar_insumos.py                                    > verificar_insumos.log
python results/f33_seleccion/regla_independiente.py                                  > regla_independiente.log
python results/f33_seleccion/fragilidad.py                                           > fragilidad.log
git merge-base --is-ancestor protocolo-v1 36c6398; echo $?                           > git_protocolo.log
```

---

## 0. Preparación

### 0.1 PR #18

**Ya está fusionado en `main`**: `0b18fa2 Merge pull request #18 from DpalaciosR20/ce/f3.2-formato-predicciones` (contiene `940d14f`, `36c6398` y `9b24b3e`). La única diferencia de `origin/ce/f3.2-formato-predicciones` que no está en `main` es al revés: `main` tiene trabajo posterior (F2.x, PR #23, #27). Se trabajó sobre el árbol actual; no se fusionó nada. (`gh` no está instalado; el estado se comprobó con `git log origin/main`.)

### 0.2 Bitácoras

- **`docs/bitacora/F3.1.md` no existe**, ni en `main` ni en ninguna rama remota ni en el historial (`git log --all -- docs/bitacora/F3.1.md` vacío). F3.1 solo aparece en `plan/plan_tt2_v2.md` línea 186 («Función de entrenamiento generalizada … y prueba de humo», CE). No hay nada de F3.1 que resumir.
- **F3.2** (`docs/bitacora/F3.2.md`), lo que afecta a la selección:
  1. Las 54 corridas (18 configs × semillas 42, 7, 123) se hicieron con `36c6398`, solo validación; la v0 está sustituida y no participa (matriz §5, «Declaración»).
  2. H1: para leer `p_dep` sin perder bits hace falta `float_precision='round_trip'`; `load_predictions` lee como texto.
  3. H2: con `px224-noflip`, el finalista sugerido de Perlin cambió de EfficientNet-B0 FT (v0) a ViT-B/16 congelado.
  4. H3 (transferido a F3.3): en Lorenz, ViT-B/16 congelado entra en la zona de empate por 0.00008; la DE del líder (0.00179) fija el tamaño de la zona. Sugiere que el segundo (EfficientNet-B0 FT) cubra esa decisión en F5.5.
  5. H8 (transferido a F3.3): 4 de 54 corridas terminaron por tope de épocas, entre ellas el finalista de Perlin (s42) y el líder de Lorenz (s7).
  6. H9: los candidatos son estables entre semillas (80–96 % de acuerdo); los ViT congelados, los más estables.
  7. Limitación 2: el F1-val del finalista está sesgado al alza (se eligió época y configuración con las mismas 114 muestras).
  8. H6: `build.py` y `potencia.py` siguen leyendo la v0 (punto 6c–d).

### 0.3 Documentos leídos

`docs/matriz_experimentos.md` (§3, §4, §5, §6, §8, §11), `configs/matriz_experimentos.json`, `configs/preprocesamiento.json`, `docs/protocolo_preregistrado.md` (§2, §8, §10, §12 y nota de equivalencia), `src/selection/select_finalist.py` (incluida `select_second`), `src/predictions/validate.py`, `src/canonical/build.py`, `results/f11_potencia/potencia.py` y `README.md`, `docs/especificacion_f11.md`, `results/f32_primera_pasada/README.md`.

### 0.4 Acceso a test (declaración)

- Mis tres scripts leen `meta/sample_index.csv` con `usecols=['sample_id','split','y_true']` y **filtran `split == 'val'` en la línea siguiente**; las filas de test no se usan en ningún cálculo.
- **Observación:** `src.predictions.validate` y `load_predictions` (código existente, no modificado) leen el índice maestro completo, que incluye `y_true` de test, para validar el formato. No calculan nada sobre test.
- Ningún CSV ni JSON contiene datos de test (punto 1b y 1f).

---

## 1. Integridad de los insumos

### a) Validador F1.5

`54/54 archivos válidos`, código de salida 0 (`validate.log`).

### b) Solo validación

Las 54 × 114 = 6,156 filas tienen `split == 'val'`. Ninguna fila de test.

### c) `sample_id` e `y_true`

En los 54 CSV: el conjunto de `sample_id` es exactamente el de las 114 filas de val de `meta/sample_index.csv`; `y_true` coincide con el índice en todas las filas (44 positivos, 70 negativos en cada corrida). Las columnas `technique`, `arch`, `strategy` y `seed` coinciden con el nombre del archivo.

### d) Manifiestos

Sin discrepancias en los 54:

| Campo | Valor en los 54 manifiestos | Referencia |
|---|---|---|
| `git_commit` | `36c6398d894bdb4dfedc5252b6e5957d4620cd9c` | bitácora F3.2 §4.1 |
| `fase` | `primera_pasada` | — |
| `variant` / `preprocessing` | Gray-Scott `px224-flip`; Perlin y Lorenz `px224-noflip`; bloque completo idéntico al de `configs/preprocesamiento.json` | F1.4 |
| `optimizer` | `AdamW` | matriz §3 |
| `lr` | 3e-05 en `finetune`, 0.001 en `frozen` | matriz §3 |
| `weight_decay` | 0.0 | matriz §3 |
| `batch_size`, `max_epochs`, `patience` | 32, 15, 4 | matriz §3 |
| `loss` / `class_weights` | `CrossEntropyLoss` / `[0.8180180191993713, 1.286118984222412]` (un solo valor) | matriz §3 ([0.8180, 1.2861]) |
| Semillas | {42, 7, 123}, cada configuración con las tres | matriz §4 |
| `permute_labels`, `smoke` | `null`, `false` | — |
| Determinismo | un solo valor: `cudnn_deterministic`, `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, SDPA *flash* y *mem-efficient* off, `applied` coincide | matriz §3, §11.6 |
| `preprocessing_file.sha256` | `7a13a3b2…a451` = sha256 de `configs/preprocesamiento.json` en `36c6398` | — |
| `matrix_file.sha256` | `ce0e0ed6…8f15` = sha256 de `configs/matriz_experimentos.json` en `36c6398` | — |
| `csv_sha256` | igual al sha256 del CSV en disco | — |
| `n_rows` | `{"val": 114}` | — |
| Entorno | un solo valor: Python 3.13.16, torch 2.11.0+cu130, cuDNN 92700, Tesla T4 | bitácora F3.2 §3.3 |

**Nota (no es discrepancia):** el sha256 de los dos JSON de configuración *en el disco local* (`53f838ea…`, `5fa4298a…`) no coincide con el de los manifiestos porque la copia de trabajo tiene CRLF (`core.autocrlf=true`). Sin `\r` los hashes son idénticos a los de `36c6398` y `HEAD`; `git diff 36c6398 HEAD -- configs/preprocesamiento.json configs/matriz_experimentos.json` está vacío.

### e) Etiqueta del protocolo

| Comprobación | Resultado |
|---|---|
| `git merge-base --is-ancestor protocolo-v1 36c6398` | código 0: **`protocolo-v1` es ancestro de `36c6398`** |
| Etiqueta `protocolo-v1` (anotada) | `taggerdate` 2026-11-02T18:14:54-06:00 = 2026-11-03T00:14:54Z; apunta a `91c84df` (2026-11-02T18:07:46-06:00) |
| Commit `36c6398` | 2026-11-02T19:52:27-06:00 = 2026-11-03T01:52:27Z |
| `created_at` de los manifiestos | mín. 2026-11-03T02:20:35Z, máx. 2026-11-03T03:52:10Z |

La etiqueta precede al commit de ejecución (1 h 38 min) y a la primera corrida (2 h 06 min).

### f) `--evaluate-test`

- `results/f32_primera_pasada/grid_meta.json`: `{"fase": "primera_pasada", "smoke": false}` (no tiene la clave `evaluate_test`; ver 6d).
- `run_grid.log` línea 1: `Fase: primera_pasada | SOLO VALIDACION (test no se carga)`; cada corrida termina en `(test no evaluado)`.
- Los 54 `runs/*.json`: `evaluate_test=false`; `f1_test`, `acc_test`, `n_test`, `test_preds`, `test_probs`, `test_true` = `null`.
- **Observación:** `run_grid.log` líneas 10, 13 y 16 dicen `<tec> test 114 imagenes OK`. Es la verificación previa de `run_grid.py` (`36c6398`, líneas 296–303): compara los **nombres** de archivo de la carpeta con `img_<tec>` del índice para cada split, incluido test, y el `dir_digest` de la carpeta completa. No abre las imágenes de test ni entrena ni evalúa con ellas. Lo informo para que conste; no lo juzgo.

---

## 2. F1 independiente

`verificar_insumos.py` lee cada CSV con `pd.read_csv(..., float_precision='round_trip')`, calcula `ŷ = p_dep > 0.5` y `f1_score(average='macro', labels=[0,1], zero_division=0)`. Detalle por corrida: `f1_recalculado.csv`.

| Comparación | Diferencia máxima sobre las 54 corridas |
|---|---|
| F1 recalculado vs `best_f1_val` de `runs/*.json` | **0.0** (idénticos) |
| F1 recalculado vs `model_selection.best_f1_val` del manifiesto | **0.0** |
| F1 recalculado vs `summary.csv` (redondeado a 4 decimales) | 4.864864864873031e-05 (< 5e-5: solo redondeo) |
| Media y DE por configuración: mi reimplementación vs `select_finalist` | 2.220446049250313e-16 y 9.8879238130678e-17 (ulp de punto flotante) |

Además:

- `p_dep` del CSV = `val_probs` del JSON **bit a bit** en 54/54; `y_true` = `val_true` en 54/54; `ŷ` = `val_preds` en 54/54.
- Con el lector por omisión de pandas, 3,291 de los 6,156 `p_dep` difieren en el último bit (en las 54 corridas), sin cambiar ningún `ŷ`. Confirma F3.2 H1.
- **`|p_dep − 0.5| < 1e-6`: 0 valores en las 54 corridas.** El mínimo global de `|p_dep − 0.5|` es 0.00011577456982653223. Ninguna predicción depende del umbral exacto.

---

## 3. Selección

### 3.1 Resultado

`select_finalist`/`select_second` (oficial) y `regla_independiente.py` (reimplementación sin importar `src.selection`) **coinciden en las tres técnicas**:

| Técnica | Finalista | Segundo (`select_second`) |
|---|---|---|
| Gray-Scott | EfficientNet-B0 *fine-tuning* | ResNet-18 *fine-tuning* |
| Perlin | ViT-B/16 congelado | EfficientNet-B0 congelado |
| Lorenz | ViT-B/16 congelado | EfficientNet-B0 *fine-tuning* |

Los finalistas coinciden con los de la bitácora de F3.2 (§4.3 y §7).

### 3.2 Tabla (`tabla_seleccion.csv`)

Límite = media del líder − DE del líder (`ddof=1`); holgura = media − límite. En ninguna técnica hay varios líderes con la misma media.

| Técnica | Configuración | Media | DE (ddof=1) | Límite | Holgura | Empate | Entrenables | Totales | Rol |
|---|---|---|---|---|---|---|---|---|---|
| gray_scott | resnet18/finetune (líder) | 0.71973892058277522 | 0.011332519182255764 | 0.70840640140051947 | 0.011332519182255751 | sí | 11,177,538 | 11,177,538 | segundo |
| gray_scott | efficientnet_b0/finetune | 0.71757623134344761 | 0.012496044907801315 | 0.70840640140051947 | 0.0091698299429281427 | sí | 4,010,110 | 4,010,110 | **finalista** |
| gray_scott | vit_b16/finetune | 0.71748046233797058 | 0.0079888230626044016 | 0.70840640140051947 | 0.0090740609374511116 | sí | 85,800,194 | 85,800,194 | — |
| gray_scott | vit_b16/frozen | 0.70305802662945516 | 0.010426668516944881 | 0.70840640140051947 | −0.0053483747710643126 | no | 1,538 | 85,800,194 | — |
| gray_scott | efficientnet_b0/frozen | 0.69651988345190896 | 0.0021369873472722352 | 0.70840640140051947 | −0.011886517948610509 | no | 2,562 | 4,010,110 | — |
| gray_scott | resnet18/frozen | 0.69549145599799889 | 0.0093433205234833804 | 0.70840640140051947 | −0.012914945402520583 | no | 1,026 | 11,177,538 | — |
| perlin | vit_b16/frozen (líder) | 0.70284719788916039 | 0.0072593713185530985 | 0.69558782657060725 | 0.0072593713185531428 | sí | 1,538 | 85,800,194 | **finalista** |
| perlin | vit_b16/finetune | 0.6999042077684462 | 0.020324025911263855 | 0.69558782657060725 | 0.0043163811978389477 | sí | 85,800,194 | 85,800,194 | — |
| perlin | efficientnet_b0/finetune | 0.69334212333648138 | 0.013744357837957874 | 0.69558782657060725 | −0.0022457032341258731 | no | 4,010,110 | 4,010,110 | — |
| perlin | efficientnet_b0/frozen | 0.69181877499337807 | 0.010072692735191041 | 0.69558782657060725 | −0.0037690515772291766 | no | 2,562 | 4,010,110 | segundo |
| perlin | resnet18/finetune | 0.67068770419381385 | 0.014572785411289736 | 0.69558782657060725 | −0.024900122376793399 | no | 11,177,538 | 11,177,538 | — |
| perlin | resnet18/frozen | 0.66728728487183397 | 0.0080349894045902565 | 0.69558782657060725 | −0.028300541698773274 | no | 1,026 | 11,177,538 | — |
| lorenz | efficientnet_b0/finetune (líder) | 0.71655721547878415 | 0.0017937274622905953 | 0.71476348801649359 | 0.0017937274622905619 | sí | 4,010,110 | 4,010,110 | segundo |
| lorenz | vit_b16/frozen | 0.71484772537404118 | 0.0047224594814097532 | 0.71476348801649359 | **8.4237357547589653e-05** | sí | 1,538 | 85,800,194 | **finalista** |
| lorenz | efficientnet_b0/frozen | 0.7070058659192785 | 0.011513708982141484 | 0.71476348801649359 | −0.0077576220972150889 | no | 2,562 | 4,010,110 | — |
| lorenz | vit_b16/finetune | 0.70102555175229597 | 0.025214673174921064 | 0.71476348801649359 | −0.013737936264197614 | no | 85,800,194 | 85,800,194 | — |
| lorenz | resnet18/finetune | 0.69720931543325781 | 0.0067225121411059801 | 0.71476348801649359 | −0.017554172583235772 | no | 11,177,538 | 11,177,538 | — |
| lorenz | resnet18/frozen | 0.68998537981548458 | 0.003980053072742927 | 0.71476348801649359 | −0.024778108201009008 | no | 1,026 | 11,177,538 | — |

Comprobación de F3.2 §4.4 (Lorenz): media del líder 0.7165572154787841, DE 0.0017937274622905953, media de ViT-B/16 congelado 0.7148477253740412, diferencia 0.0017094901047429722: **se reproducen** (la holgura 8.42e-05 = DE − diferencia; la bitácora la redondea a «0.00008»). Las 18 medias y DE de F3.2 §4.3 se reproducen a 4 decimales. Los conteos de parámetros coinciden con `summary.csv` y con el bloque `parametros` del JSON.

### 3.3 Observaciones sobre el segundo

- **Perlin:** el segundo (EfficientNet-B0 congelado) **no estaba en la zona de empate del finalista** (holgura −0.0038). Sale de volver a aplicar la regla sin ViT-B/16 congelado: el nuevo líder es ViT-B/16 FT (0.69990, DE 0.02032), cuya DE grande abre la zona hasta 0.67958; entran EfficientNet-B0 FT y congelado, y gana el congelado por complejidad. Es la regla tal como está preregistrada (protocolo §8, matriz §5); no es el segundo por media (que sería ViT-B/16 FT).
- **Gray-Scott:** el segundo es el líder por media (ResNet-18 FT). ViT-B/16 congelado (0.70306) no entra en la nueva zona (límite 0.70841).
- **Lorenz:** el segundo es el líder por media (EfficientNet-B0 FT), como anticipaba F3.2 H3.

---

## 4. Fragilidad de la decisión (descriptivo; no cambia la selección)

### a) Otras medidas de la zona de empate (`fragilidad_tolerancias.csv`)

Definiciones: «DE ddof=0» y «EE» sustituyen la DE del líder en la regla (EE = DE ddof=1 / √3). «DE del retador»: cada configuración empata si su media ≥ media del líder − su propia DE (ddof=1). «DE combinada»: empata si su media ≥ media del líder − √(DE_líder² + DE_retador²). El resto de la regla (complejidad, etc.) no cambia.

| Técnica | §5 (DE líder, ddof=1) | DE líder ddof=0 | EE líder = DE/√3 | DE del retador | DE combinada |
|---|---|---|---|---|---|
| Gray-Scott | EffNet-B0 FT | EffNet-B0 FT | EffNet-B0 FT | EffNet-B0 FT | EffNet-B0 FT |
| Perlin | ViT-B/16 cong. | ViT-B/16 cong. | ViT-B/16 cong. | ViT-B/16 cong. | ViT-B/16 cong. |
| Lorenz | ViT-B/16 cong. | **EffNet-B0 FT** | **EffNet-B0 FT** | ViT-B/16 cong. | ViT-B/16 cong. |

Conjuntos de empatadas: en Gray-Scott son las mismas tres FT en las cinco variantes. En Perlin, ViT cong. + ViT FT con las tres primeras; la DE del retador añade EffNet-B0 FT y la combinada además EffNet-B0 cong. En Lorenz, con ddof=0 (DE 0.0014646) o EE (0.0010356) la zona queda por debajo de la diferencia 0.0017095 y solo queda el líder; con la DE del retador o la combinada entran también EffNet-B0 cong. y ViT-B/16 FT, y el finalista sigue siendo ViT-B/16 cong. (1,538 entrenables).

### b) Granularidad (n = 114, 44 positivos) (`fragilidad_volteo_unico.csv`, `fragilidad_volteos.csv`)

**Un solo volteo.** En las 15 corridas analizadas (finalista y líder por media; en Perlin son la misma configuración), voltear la predicción de una muestra cambia el F1-Macro entre |ΔF1| = 0.007977 y 0.011134. Por corrida:

| Técnica | Configuración (rol) | Semilla | F1 | VP FP FN VN | ΔF1 mín | ΔF1 máx | \|ΔF1\| mín | \|ΔF1\| máx |
|---|---|---|---|---|---|---|---|---|
| gray_scott | effnet_b0/finetune (finalista) | 42 | 0.720301 | 32 19 12 51 | −0.009917 | 0.009813 | 0.008180 | 0.009917 |
| gray_scott | effnet_b0/finetune (finalista) | 7 | 0.703943 | 32 21 12 49 | −0.009814 | 0.009715 | 0.008179 | 0.009814 |
| gray_scott | effnet_b0/finetune (finalista) | 123 | 0.728485 | 32 18 12 52 | −0.009967 | 0.009861 | 0.008184 | 0.009967 |
| gray_scott | resnet18/finetune (líder) | 42 | 0.712121 | 32 20 12 50 | −0.009866 | 0.009764 | 0.008179 | 0.009866 |
| gray_scott | resnet18/finetune (líder) | 7 | 0.732762 | 30 15 14 55 | −0.010360 | 0.010233 | 0.008124 | 0.010360 |
| gray_scott | resnet18/finetune (líder) | 123 | 0.714333 | 26 12 18 58 | −0.011134 | 0.010968 | 0.007977 | 0.011134 |
| perlin | vit_b16/frozen (finalista y líder) | 42 | 0.710384 | 31 19 13 51 | −0.010026 | 0.009917 | 0.008129 | 0.010026 |
| perlin | vit_b16/frozen (finalista y líder) | 7 | 0.702256 | 31 20 13 50 | −0.009972 | 0.009866 | 0.008127 | 0.009972 |
| perlin | vit_b16/frozen (finalista y líder) | 123 | 0.695902 | 28 17 16 53 | −0.010512 | 0.010378 | 0.007979 | 0.010512 |
| lorenz | vit_b16/frozen (finalista) | 42 | 0.712121 | 32 20 12 50 | −0.009866 | 0.009764 | 0.008179 | 0.009866 |
| lorenz | vit_b16/frozen (finalista) | 7 | 0.712121 | 32 20 12 50 | −0.009866 | 0.009764 | 0.008179 | 0.009866 |
| lorenz | vit_b16/frozen (finalista) | 123 | 0.720301 | 32 19 12 51 | −0.009917 | 0.009813 | 0.008180 | 0.009917 |
| lorenz | effnet_b0/finetune (líder) | 42 | 0.718519 | 31 18 13 52 | −0.010079 | 0.009967 | 0.008134 | 0.010079 |
| lorenz | effnet_b0/finetune (líder) | 7 | 0.716153 | 35 23 9 47 | −0.009441 | 0.009361 | 0.008334 | 0.009441 |
| lorenz | effnet_b0/finetune (líder) | 123 | 0.715000 | 34 22 10 48 | −0.009574 | 0.009487 | 0.008279 | 0.009574 |

(Cifras a 6 decimales; el CSV las tiene a precisión completa.) Un volteo mueve la media de 3 semillas en ≈ 0.0027–0.0037.

**Mínimo número de volteos que cambia el finalista.** Búsqueda exacta: un volteo solo depende de su tipo (VP→FN, FN→VP, VN→FP, FP→VN), así que se enumeran todas las combinaciones de conteos por tipo, respetando cuántas muestras de cada tipo hay; se aplica la regla §5 completa con las otras configuraciones fijas. (i) en una sola corrida, hasta 40 volteos; (ii) repartidos entre las corridas del finalista y del líder, hasta 6.

| Técnica | Mínimo en una corrida (mejor corrida) | Mínimo repartido | Ejemplo y nuevo finalista |
|---|---|---|---|
| Gray-Scott | **2** (EffNet-B0 FT s123) | **2** | 2 FP→VN (o 2 FN→VP) en EffNet-B0 FT s123: sube su media por encima de ResNet-18 FT, pasa a ser líder con DE mayor y la zona abarca a ViT-B/16 cong. → finalista ViT-B/16 cong. Con 3 volteos en EffNet-B0 FT s42 o s7 (2 VP→FN + 1 VN→FP) sale de la zona → ResNet-18 FT |
| Perlin | 7 (cualquier semilla de ViT-B/16 cong.) | **6** | 3 FN→VP en s42 y 3 VP→FN en s123 → ResNet-18 cong.; con 7 VP→FN en una corrida → EffNet-B0 cong. |
| Lorenz | **1** (cualquier semilla de ViT-B/16 cong., o EffNet-B0 FT s123) | **1** | 1 VN→FP o 1 VP→FN en una corrida de ViT-B/16 cong. lo saca de la zona → EffNet-B0 FT. 1 FP→VN en EffNet-B0 FT s123 también (reduce la DE del líder) |

Por corrida (en una sola corrida): Gray-Scott EffNet-B0 FT s42/s7/s123 = 3/3/2, ResNet-18 FT s42/s7/s123 = 10/3/8; Perlin ViT-B/16 cong. 7/7/7; Lorenz ViT-B/16 cong. 1/1/1, EffNet-B0 FT 2/2/1.

---

## 5. Tope de épocas (H8) (`epocas_finalista_segundo.csv`)

Causa: «paciencia» si `épocas − best_epoch ≥ 4`; «tope» si terminó en la 15 sin haber agotado la paciencia; «paciencia (coincide con el tope)» si ambas a la vez. En los 18 casos `len(history.f1_val) = épocas` y `history[best_epoch−1] = best_f1_val = max(history)`.

| Técnica | Rol | Configuración | Semilla | best_epoch | Épocas | Causa | F1-val últimas 4 épocas |
|---|---|---|---|---|---|---|---|
| gray_scott | finalista | effnet_b0/finetune | 42 | 3 | 7 | paciencia | 0.7022556390977444, 0.7022556390977444, 0.6922832645284218, 0.700358422939068 |
| gray_scott | finalista | effnet_b0/finetune | 7 | 5 | 9 | paciencia | 0.6842105263157894, 0.7022556390977444, 0.671994884910486, 0.6941287878787878 |
| gray_scott | finalista | effnet_b0/finetune | 123 | 10 | 14 | paciencia | 0.7185185185185186, 0.700358422939068, 0.7039427087432124, 0.7084398976982097 |
| gray_scott | segundo | resnet18/finetune | 42 | 2 | 6 | paciencia | 0.657200133645172, 0.6747370995353388, 0.6649029982363315, 0.6717400246812011 |
| gray_scott | segundo | resnet18/finetune | 7 | 8 | 12 | paciencia | 0.6561506480802152, 0.6687179487179487, 0.6799550814149354, 0.7012774320340649 |
| gray_scott | segundo | resnet18/finetune | 123 | 8 | 12 | paciencia | 0.6984126984126984, 0.6853896103896104, 0.6840513983371126, 0.6684669713336102 |
| perlin | finalista | vit_b16/frozen | 42 | 13 | 15 | **tope** | 0.6649029982363315, **0.7103842489679263**, 0.6933235509904623, 0.6826072715361939 |
| perlin | finalista | vit_b16/frozen | 7 | 2 | 6 | paciencia | 0.6842105263157895, 0.6717400246812011, 0.6799550814149354, 0.6840513983371126 |
| perlin | finalista | vit_b16/frozen | 123 | 8 | 12 | paciencia | 0.6853896103896104, 0.6726351351351352, 0.6959017056018106, 0.6922832645284218 |
| perlin | segundo | effnet_b0/frozen | 42 | 12 | 15 | **tope** | **0.6882051282051282**, 0.6640382317801673, 0.6561506480802152, 0.6668831168831169 |
| perlin | segundo | effnet_b0/frozen | 7 | 3 | 7 | paciencia | 0.6795634920634921, 0.6795634920634921, 0.6762445706648847, 0.6762445706648847 |
| perlin | segundo | effnet_b0/frozen | 123 | 6 | 10 | paciencia | 0.6874117158288326, 0.6795634920634921, 0.6874117158288326, 0.7031997984378937 |
| lorenz | finalista | vit_b16/frozen | 42 | 7 | 11 | paciencia | 0.6875734085036411, 0.6889373976767756, 0.7084398976982097, 0.6533783783783784 |
| lorenz | finalista | vit_b16/frozen | 7 | 2 | 6 | paciencia | 0.6457545981355506, 0.649522226838388, 0.6799550814149354, 0.7039427087432124 |
| lorenz | finalista | vit_b16/frozen | 123 | 6 | 10 | paciencia | 0.663937111038323, 0.6922832645284218, 0.6529823118058412, 0.6607142857142857 |
| lorenz | segundo | effnet_b0/finetune | 42 | 11 | 15 | paciencia (coincide con el tope) | 0.6774715059413143, 0.671994884910486, 0.659041306280818, 0.6607142857142857 |
| lorenz | segundo | effnet_b0/finetune | 7 | 12 | 15 | **tope** | **0.7161531279178338**, 0.6902173913043479, 0.6640382317801673, 0.6561506480802152 |
| lorenz | segundo | effnet_b0/finetune | 123 | 9 | 13 | paciencia | 0.6971875000000001, 0.6941287878787878, 0.6761363636363636, 0.6342245989304813 |

Sobre las 54 corridas, las que terminaron por tope son exactamente las 4 de F3.2 H8 (Gray-Scott ResNet-18 cong. s42, best 12; Perlin EffNet-B0 cong. s42, 12; Perlin ViT-B/16 cong. s42, 13; Lorenz EffNet-B0 FT s7, 12): **coincide con la bitácora**. Además hay 2 en que la paciencia se agota justo en la época 15 (Lorenz EffNet-B0 FT s42, best 11; Perlin ViT-B/16 FT s123, best 11); F3.2 no las cuenta como tope, y es coherente con su criterio. Nuevo respecto de H8: el **segundo de Perlin** (EffNet-B0 cong.) también tiene su semilla 42 detenida por tope (es la corrida de H8 «Perlin EfficientNet-B0 congelado, semilla 42»; H8 no la marcaba como candidata porque entonces no se conocía el segundo).

En las 4 corridas por tope, la mejor época no es la última: las épocas posteriores al máximo bajan (ver columna).

---

## 6. Consultas a documentos

### a) Entrenamientos del segundo mejor modelo y total de 207

- **HPO: no.** Protocolo §8, tabla: «Segundo mejor modelo | Segunda configuración de la clasificación de F3.3 en cada técnica, **con hiperparámetros de la primera pasada y cinco semillas** (requiere 15 entrenamientos adicionales, §12).» Nota de equivalencia del protocolo: «“Hiperparámetros de la primera pasada” (§8; `segundo_mejor.hyperparams = "primera_pasada"` en `configs/protocolo.json`) designa la configuración común de la matriz (`fases.sensibilidad.hyperparams = "configuracion_comun"`, `docs/matriz_experimentos.md` §3).» Matriz §4: «`sensibilidad` | F5.5 | Segundo mejor modelo por técnica según F3.3 (`select_second`, §5), con los hiperparámetros de la primera pasada (§3) | 42, 7, 123, 2024, 99 | `val`, `test`». F3.4 (§6) solo ajusta a «cada finalista».
- **Semillas: 5** (42, 7, 123, 2024, 99), 3 técnicas × 5 = 15 entrenamientos. Test: protocolo §10, «Segundo mejor modelo (§8) | Después de F3.5 | Una inferencia por semilla».
- **207**, matriz §4: «Total de entrenamientos: 54 (F3.2) + 3 × 12 × 3 = 108 (F3.4) + 15 (F3.5) + 15 (F3.6) + 15 (F5.5) = 207.» (`fases.total_entrenamientos: 207` en el JSON). La misma sección advierte: «Estas estimaciones [de costo] son anteriores a la fase `sensibilidad` y no la incluyen.»

### b) Interpretabilidad de ViT

- Matriz §8: «`vit_b16` | *Attention rollout* (Abnar y Zuidema, 2020) | Fusión de cabezas por media, identidad añadida y renormalizada, sin descarte; se excluye el *token* de clase del mapa final». JSON: `"method": "attention_rollout", "head_fusion": "mean", "add_identity": true, "renormalize": true, "discard_ratio": 0.0, "exclude_cls_token": true`. **Es *attention rollout* puro, sin gradientes**: ni el texto ni el JSON mencionan ponderación por gradiente.
- Lo que dice sobre modelos congelados: «**Modelos `frozen`:** el cálculo de gradientes requiere habilitar `requires_grad` en las activaciones de la capa objetivo, porque los parámetros del *backbone* están congelados.» Esto se refiere a Grad-CAM. **La matriz no dice nada específico sobre ViT congelado.**
- **Observación (solo informo):** dos finalistas son ViT-B/16 congelado (Perlin y Lorenz). En modo `frozen` el *backbone* no se entrena (matriz §3, «solo la capa de clasificación es entrenable») y el *rollout* se calcula solo con las atenciones del *encoder*. En consecuencia, para esos dos finalistas los mapas dependerían solo de la imagen y de los pesos de ImageNet: serían los mismos para las 5 semillas, no dependerían de la cabeza entrenada y serían idénticos a los del control de permutación de F3.6. Además, el «objetivo del mapa: el *logit* de la clase Depresivo (1)» de §8 no interviene en un *rollout* sin gradientes. Esto afecta a la prueba de cordura por aleatorización de datos (§8, F4.3).

### c) `results/f11_potencia/potencia.py` y cifras de F1.1

- Lee (líneas 85–88): `results/canonical_numbers.json`; `results/predictions/referencia/text__beto__finetune__seed42.csv` (BETO, val y test: el test de BETO ya consultado en F2.2); `results/predictions/referencia/emb768__logreg__none__seed42.csv` (solo val); `meta/sample_index.csv`; y `F32_PRED = results/f32_primera_pasada_v0/predictions.csv` (formato v0, con `run_id`, solo val).
- Historial de `F32_PRED`: en `af6abc9` ([F1.1] Script…) apuntaba a `results/f32_primera_pasada/predictions.csv`, que entonces contenía la v0; `9b24b3e` (F3.2) lo cambió a `results/f32_primera_pasada_v0/predictions.csv` al renombrar la carpeta.
- **Las cifras de F1.1 salieron de las predicciones de la primera pasada v0.** `potencia.log` registra `sha256 results/f32_primera_pasada/predictions.csv 448b457f37921ab5549218fb870de1642820305fce0df317ab5d212f2ba515ed`, que es el sha256 actual de `results/f32_primera_pasada_v0/predictions.csv`. El resultado de F1.1 (`04007be`, 2026-11-02 13:52 -0600) es anterior a la etiqueta del protocolo y a la repetición de F3.2. `docs/especificacion_f11.md` §2 lo declara: «Predicciones de la primera pasada (F3.2, versión sustituida), solo validación | `results/f32_primera_pasada_v0/predictions.csv` | Calibración empírica del acuerdo BETO–visual y de la variabilidad entre semillas», y «Se usan **solo para planificar**»; su limitación 1: «La discordancia y la DE se calibran con predicciones de validación de la primera pasada v0 … 3 semillas (no 5), un preprocesamiento sustituido en Perlin y Lorenz y `p_dep` redondeado». El protocolo §2 lista «Análisis de potencia | F1.1» y «F1-Val de la primera pasada v0 … sustituida» entre lo que se conocía al congelar.
- Además, `block_e` (línea 294) busca en `canonical_numbers.json` la clave fija `f32.<tec>.<arq>.<est>.px224_flip.seed<k>.f1_macro`, es decir, también asume la v0.

### d) `src/canonical/build.py`: claves `f32.*` y migración (diagnóstico; no se cambió)

**Claves que genera hoy** (`collect_f32`, sobre `F32_DIR = results/f32_primera_pasada_v0`): 90 claves `f32.*`.
- 54 por corrida: `f32.<tecnica>.<arch>.<strategy>.<variant>.seed<k>.f1_macro` (`value.val` = F1, `value.test = null`, matriz de confusión de val);
- 18 + 18 derivadas por configuración: `f32.<…>.<variant>.f1_macro_mean` (`derivation: mean`) y `.f1_macro_std` (`std_ddof1`), con `_check_csv` contra `summary_by_config.csv`.
- Con la v0, las claves usan `arch = vit_b_16` y `variant = px224_flip` en las tres técnicas (en `canonical_numbers.json` hay `efficientnet_b0`, `resnet18`, `vit_b_16`, todas `px224_flip`).
- `collect_structural` también usa `F32_DIR/predictions.csv` para `n.val` y `support.val.label{0,1}` cuando no se pasa `umap_params.pkl`.

**Qué tendría que cambiar para leer `results/predictions/primera_pasada/`:**
1. **Fuente de predicciones:** `_read_predictions()` lee un único `predictions.csv` con `run_id`. Habría que leer los 54 CSV F1.5 (p. ej. con `load_predictions`) y agrupar por `(technique, arch, strategy, seed)`; el orden de `sample_id` ya cumple `val_0000…val_0113`.
2. **JSON de cada corrida:** hoy `runs/{run_id}.json`. En la repetición, el nombre es `<tec>__<arq>__<est>__<variant>__seed<k>.json` (también está en `config.run_json` del manifiesto) dentro de `results/f32_primera_pasada/runs/`.
3. **`grid_meta.json`:** `collect_f32` exige `evaluate_test is False`, pero el nuevo `grid_meta.json` es `{"fase": "primera_pasada", "smoke": false}`, sin esa clave: **lanzaría `CanonicalError`** tal como está. Habría que comprobar `fase == primera_pasada` o `evaluate_test=false` en los JSON de las corridas.
4. **Claves:** cambiarían de nombre: `vit_b_16` → `vit_b16` y `px224_flip` → `px224_noflip` en Perlin y Lorenz. Afecta a todo lo que cite claves `f32.*` (por ejemplo `potencia.py` `block_e`, que fija `px224_flip`) y a `tests/test_canonical_numbers.py` (líneas 208–209 solo cuentan 54 y 18, así que no dependen del nombre).
5. **Insumos con hash:** sustituir `predictions.csv` por los 54 CSV (y sus manifiestos) en `inputs`; `summary.csv` y `summary_by_config.csv` existen con las mismas columnas en `results/f32_primera_pasada/`.
6. La comprobación `round(f1, 4) == summary.csv` sigue siendo válida (punto 2: diferencia < 5e-5).
7. `collect_structural` tendría que obtener `n.val` y los soportes de val de otra fuente (cualquier CSV de la repetición: 114, 70/44).

---

## Resumen de discrepancias y anomalías

**DISCREPANCIAS frente a la bitácora de F3.2: ninguna.** Se reproducen los 54 F1, las 18 medias y DE (§4.3), los tres finalistas, el margen de Lorenz (§4.4), las 4 corridas por tope (H8), el commit y la variante de los manifiestos (§4.1) y el comportamiento del lector de pandas (H1).

Anomalías y observaciones, solo informadas:

1. `docs/bitacora/F3.1.md` no existe (0.2).
2. El validador existente lee el índice maestro completo, con `y_true` de test (0.4). La verificación previa de `run_grid.py` listó los nombres de las imágenes de test (1f).
3. El sha256 local de los JSON de configuración difiere del de los manifiestos solo por CRLF (1d).
4. El segundo de Perlin (EffNet-B0 cong.) no estaba en la zona de empate del finalista; sale de reaplicar la regla con ViT-B/16 FT como líder (3.3). También tiene su semilla 42 detenida por tope (5).
5. Lorenz: el finalista cambia a EffNet-B0 FT con DE ddof=0 o con EE (4a), y con un solo volteo en cualquier corrida de ViT-B/16 cong. (4b). Gray-Scott cambia con 2 volteos; Perlin con 6.
6. Para los dos finalistas ViT-B/16 congelados, el *attention rollout* sin gradientes no dependería de la cabeza entrenada ni de la semilla. La matriz no trata este caso (6b).
7. `potencia.py` y `build.py` siguen leyendo la v0; `build.py` fallaría con el nuevo `grid_meta.json` sin cambios (6c–d).
