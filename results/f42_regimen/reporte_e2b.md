# Reporte E2b: F4.2 (régimen dinámico). Recepción, resoluciones, cálculo y cierre

> Bloque E, etapa 2b · Rol: SV · Rama `sv/f4-regimen`
> Sin interpretación: cifras y verificaciones. La interpretación está en `docs/bitacora/F4.2.md`.
> Fuera de alcance:
> - La capa C de la adenda de F0.3 (Lorenz pospuesto; sus resultados no se registran).
> - No se regeneraron imágenes de Lorenz.
> - No se tocaron modelos ni `results/f5_contraste/`.

## 1. Commits

| Commit | Asunto |
|---|---|
| `b1873e0` | [F4.2, F0.3] Recepción de Kaggle |
| `43ba00a` | [F4.2] Resoluciones de los PENDIENTES P1–P7 y su implementación |
| `dccaf5f` | [F4.2] Cálculo del régimen dinámico |
| `f838579` | [F4.2, F0.4] Cifras canónicas f42.* |
| `dce4ff9` | [F4.2] Notebook del régimen dinámico, ejecutado con salidas |
| `e467930` | [F4.2] Bitácora de cierre |
| `ecdb393` | [F4.2] Log de pytest de E2b |
| (este) | [F4.2] Reporte E2b |

## 2. Recepción (`results/f42_regimen/recepcion/verificacion_local.json`)

- **Origen:** `G:\Mi unidad\tt2-insumos\f42_resultados` (zip ya extraído).
- **Copiado al repo:** solo `results/f42_regimen/reintentos/`, sin los `desktop.ini`.
- **Bytes intactos:** `.gitattributes` marca `results/f42_regimen/reintentos/** -text`, para conservarlos byte a byte (`core.autocrlf = true`).

| Comprobación | Resultado |
|---|---|
| Commit de ejecución (`commit.txt` = `manifest.git_commit`) | `c121ab62f2718d7b9c299fbcde9f84bc26507a30` |
| `smoke` / `limit` / `split` / `ruta` | `false` / `null` / `train` / `cpu` |
| Filas del CSV / `csv.filas` | 908 / 908 |
| sha256 del CSV = `csv.sha256` = `f42_sha256sum.txt` | sí |
| sha256 del manifiesto = `f42_sha256sum.txt` | sí |
| `u_dtype` del manifiesto / del pkl | float32 / float32 |
| `umap_params.pkl`: `verify_manifest` y sha256 = manifiesto de Kaggle | OK / sí |
| Orden de `sample_id`, `idx` e `y_true` = `sample_index.csv` | sí / sí / sí |
| `y_true` = etiquetas del pkl; máx \|u − pkl\| | sí; 0.0 |
| Reintentos en [0, 3]; respaldo ⇔ 3; `n_llamadas` coherente | sí; sí; sí |
| Varianzas: una por llamada; la final > 1e-4; las previas ≤ 1e-4 | sí; sí; sí |
| `sha256_coincide` / `pixeles_coinciden` / `almacenada_integra` | 908 / 908 / 908 |
| sha256 de las PNG almacenadas (local) = columna del manifiesto | 908/908 |
| Nombres `rd_train_<idx>_label<y>.png` | coherentes |
| Distribución de reintentos 0/1/2/3 | 876 / 32 / 0 / 0 |
| No coincidentes (sha256, píxeles) | [] , [] |

Observación: las columnas `f`, `k` y `Dv` del CSV están escritas como `np.float32(…)`. `regimen.py` no las lee.

## 3. Resoluciones (`results/f42_regimen/resoluciones.json`)

- El JSON se versionó tal como se pidió.
- Claves que exigieron cambios de código (`43ba00a`, con pruebas):
  - `no_identicas` y `pearson`: se agregaron al contrato de `--resoluciones`. Con un valor distinto, `regimen.py` se detiene.
  - `no_identicas`: además se reporta `no_identicas_por_clase`.
  - Claves `_*`: se aceptan como metadatos y se copian a `resultados.json:meta.resoluciones`, junto con el sha256 del archivo.
  - Cualquier otra clave no prevista detiene el cálculo.
- Pie de `f42_01`: ya no dice «PENDIENTE».
- `tests/test_fase4_regimen.py`: 32 passed (había 31; se agregó una prueba sobre el JSON del repositorio).

## 4. Cálculo

- **Comando:** `python -m src.fase4.regimen --resoluciones results/f42_regimen/resoluciones.json`.
- **Condiciones:** una sola vez, con el árbol limpio en `43ba00a`.
- **Salida:** redirigida al scratchpad y copiada a `results/f42_regimen/regimen_run.log`. Código de salida 0.

### 4.1 `train` (n = 908; 353 depresivas, 555 no depresivas)

| Cifra | Depresivo | No depresivo |
|---|---|---|
| Lorenz: n con ρ < 24.74 | 14 | 5 |
| Lorenz: proporción con ρ < 24.74 | 0.039660056657223795 | 0.009009009009009009 |
| Gray-Scott: n con reintentos > 0 | 1 | 31 |
| Gray-Scott: tasa de reintentos | 0.0028328611898017 | 0.055855855855855854 |
| Gray-Scott: muestras no idénticas (píxeles) | 0 | 0 |

- Respaldo: 0.
- Idénticas: 908/908.
- Discordancias entre ρ < 24.74 y u1 < 0.0925: 0.

### 4.2 Familia de Holm (m = 4, α = 0.05)

| Prueba | p | p ajustado de Holm | Rango | α_h | Rechazo |
|---|---|---|---|---|---|
| `lorenz_fisher_regimen` | 0.0030614811511880963 | 0.0030614811511880963 | 4 | 0.05 | true |
| `lorenz_mw_rho` | 5.495419641406489e-14 | 1.6486258924219466e-13 | 2 | 0.016666666666666666 | true |
| `perlin_mw_octavas` | 2.6262777461042087e-107 | 1.0505110984416835e-106 | 1 | 0.0125 | true |
| `gray_scott_fisher_reintentos` | 3.0376130850982917e-06 | 6.075226170196583e-06 | 3 | 0.025 | true |

### 4.3 F1 por estrato de Lorenz (prueba, n = 114)

| Estrato | n | Depresivas | No depresivas | Errores | F1-Macro | Confusión (tp, fp, fn, tn) |
|---|---|---|---|---|---|---|
| ρ < 24.74 | 3 | 1 | 2 | 1 | 0.6666666666666666 | 1, 1, 0, 1 |
| ρ ≥ 24.74 | 111 | 43 | 68 | 27 | 0.7468108473430768 | 31, 15, 12, 53 |

- Discordancias en prueba: 0.
- La confusión sale de `canonical_numbers.json`. El build la recalcula desde las predicciones de F3.5 y desde **u** de prueba.

## 5. Cifras canónicas

- **Colector:** `collect_f42` en `src/canonical/build.py`, con 48 claves `f42.*`.
- **Contenido en train (`value.train`):** conteos, proporciones, tasas, p, p_holm, α_h, rango y rechazo.
- **F1 por estrato (`value.test`):** con status `verificado`, confusión y soportes propios del estrato (`f42.lorenz.estrato_prueba.<estrato>.support.label{0,1}`).
- **Pruebas nuevas en `tests/test_canonical_numbers.py`:**
  - presencia y split;
  - Holm recalculado y p ajustado;
  - proporciones y tablas 2×2 frente a los conteos;
  - reintentos desde el CSV;
  - suma de estratos igual a prueba;
  - resoluciones versionadas.
- **Build estricto:** `CANON_STRICT=1` y `CANON_UMAP_PARAMS="G:/Mi unidad/tt2-insumos/umap_params.pkl"`. `python -m src.canonical.build` escribió el archivo y `--check` dio «al día (24175 líneas)».
- **`--check` que falló una vez:** una ejecución intermedia dio «difiere del build actual». Ocurrió justo después de un intento de *patch* fallido sobre el archivo de pruebas, que no es insumo del build. Un build a un archivo aparte fue idéntico al commiteado, y los `--check` posteriores (tres) dieron «al día». No se identificó la causa.

## 6. Pytest

`results/f42_regimen/pytest_e2b.log`. Se corrió `python -m pytest tests` con `CANON_STRICT=1` y `CANON_UMAP_PARAMS`.

- **Resultado:** 1213 passed, 30 skipped, 14 warnings en 184.66 s.
- **Los 30 omitidos** son de `tests/test_label_independence.py`, que pide «Define F03_UMAP_PARAMS» (capa C de F0.3, no corrida).

## 7. Notebook y bitácora

- **`notebooks/F4.2_regimen.ipynb`:** ejecutado con `nbconvert --execute`. Tiene 5 figuras embebidas y tablas.
  - Antes de dibujar, comprueba que los conteos recalculados desde **u** coinciden con `resultados.json` y que los hashes coinciden con `meta.inputs` y `canonical_numbers.json:inputs`.
  - `lorenz_lib` fija el backend Agg al importarse; el notebook restaura `%matplotlib inline` después de `mapear`.
- **`docs/bitacora/F4.2.md`:** incluye
  - las tablas «qué se afirma / qué no»;
  - la limitación de que Gray-Scott es solo dispersión, sin fronteras de Pearson;
  - la nota de que separar por régimen no implica uso por el clasificador.

  Es la estructura de `/cierre-tarea`. Esa *skill* no se pudo invocar desde Claude Code (`disable-model-invocation`), así que se siguió a mano; su paso de aprobación previa al commit quedó cubierto por la instrucción explícita de esta etapa.

## 8. Hashes (sha256 de los archivos de trabajo, LF)

```
ee4fa94e91c514e0f0161dee060ac6c61cc3c76279bb4e6eae58344b6fddcbf4  results/f42_regimen/resoluciones.json
e4b0df5619f15538e15f1399f028ec97c684295f23f0adbdeb7404b3ba7768b2  results/f42_regimen/resultados.json
347bba0274532a919e6c05e78650d87e222f80c2a5ec62a3c25ca293a998e6cc  results/f42_regimen/regimen_run.log
45a27867a7a0453a84c44794ff92bcfc7204be8c428d262951eb78d7cbf92a2c  results/f42_regimen/reintentos/reintentos_gs_train.csv
eda83c8e43b99c3200d68123abe48285e41fd60673bd90a2f388728bbede2047  results/f42_regimen/reintentos/reintentos_gs_manifest.json
de78979e243481152da2e9abd2b21891e9516ec3962269be0e2ad558bc5d2169  results/f42_regimen/recepcion/verificacion_local.json
791eaf80b337ed3d72b13ff7b6a6495f3407e47b6d5f8faa874cf31b6cacb656  results/canonical_numbers.json
3bfb71cd44816bf5634649ef946b46073f58e358b6b94b596f8985a90426f431  notebooks/F4.2_regimen.ipynb
8dec9ad0f011132e3c477eafc46e10f4e0276b138b7f7b1ad35c69dd64b2fab8  results/f42_regimen/figures/f42_01_gray_scott_fk.png
0cd4766e66f4cafee37850012d82bb5a9200a33b4ebe82860a8ac8a248e00f55  results/f42_regimen/figures/f42_02_lorenz_rho.png
19d994033c7c0e1920e82b0c58c3f011a6557a0eb294b1f45448d963cc8e9cf3  results/f42_regimen/figures/f42_03_perlin_octavas.png
ff3d7141f83726672474c7c87c6108a1b0acd84fba9d4ca0d38d0b3994798f31  results/f42_regimen/figures/f42_04_gray_scott_fk_por_clase.png
6f1ee53d94e302b55497fcd6bcf6fad6a9201febe562971c3fc0722b2c611db6  results/f42_regimen/figures/f42_05_lorenz_rho_por_clase.png
99f8467288ae0d3722c9c4d39fa6d65838cb0ce75f480503e3a7b7c8085ce8d4  results/f42_regimen/figures/f42_06_perlin_octavas_por_clase.png
d2e6b515baa85630da09dd534ddaac8540c8a060e3852e14204eee42e79f95b6  results/f42_regimen/figures/f42_07_gray_scott_reintentos_por_clase.png
8c8c97a2be4b4a84098748796223a9242f1baff1c846f6e812cf136cff8ed4f8  results/f42_regimen/figures/f42_08_lorenz_f1_por_estrato.png
72416893c0f4b82fa9457001a789a3cdecfb0ecb048d9573ccec769270084755  docs/bitacora/F4.2.md
```

## 9. PR

- **Push:** hecho. `origin/sv/f4-regimen` está creada.
- **PR: no creado.** `gh pr create` falló con «To get started with GitHub CLI, please run: gh auth login» (código 4).
  - Hay que autenticar `gh` y crear el PR contra `main`, sin fusionarlo.
  - Enlace para crearlo desde el navegador: https://github.com/DpalaciosR20/ttdicevc/pull/new/sv/f4-regimen
  - URL del PR: **[completar]**
