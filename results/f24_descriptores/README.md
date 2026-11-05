# F2.4 — Clasificador sobre descriptores de bajo nivel

Especificación: `docs/protocolo_preregistrado.md` §4.2, §4.3 y Enmienda 2 (punto 1); `configs/protocolo.json` → `descriptores`. Formato de las predicciones: `docs/especificacion_predicciones.md`. **No se calculó ninguna métrica de prueba** (protocolo §10). Las predicciones de test quedan archivadas hasta la Fase 5.

## Commits

| Momento | Commit | Fecha (−06:00) | Contenido |
|---|---|---|---|
| Parámetros, antes de calcular | `8d7fbb2` | 2026-11-04 21:21 | `configs/protocolo.json` → `descriptores`: implementación, versión de scikit-image, parámetros de los 4 descriptores y de la Tabla 11. En `tests/test_protocolo.py`, la huella vigente del config y la verificación de los parámetros |
| Restauración de F2.1 | `c22e54f` | 2026-11-04 21:28 | `[F2.1]`: los 23 archivos de F2.1 que borró `a91a4db`, tomados sin cambios de `origin/sv/f2.1-techo-5d`. F2.4 reutiliza su código |
| Script, antes de ejecutar | `3f40273` | 2026-11-04 21:32 | `f24_descriptores.py`, `tests/test_f24_descriptores.py` |
| Script del fragmento | `1d68a4a` | 2026-11-04 21:54 | `f24_fragment.py`. **Es el árbol con el que se generaron las predicciones definitivas** (`git_commit` de los 3 manifiestos, sin `-dirty`) |
| Resultados | el commit que introduce este README | 2026-11-04 | Salidas, fragmento canónico, `canonical_numbers.json`, registros |

**Primera corrida descartada por el indicador `-dirty`.** La primera corrida completa (sobre `3f40273`) registró `git_commit = 3f40273…-dirty` porque `f24_fragment.py`, aún sin commitear, se escribió mientras corría. Se hizo commit de ese script (`1d68a4a`) y se repitió la corrida. Los 4 CSV de resultados y los 3 CSV de predicciones salieron **idénticos byte a byte**; solo cambiaron `git_commit` y `created_at` en los manifiestos.

Antes de ejecutar se hizo una estimación acotada (`--estimar`: Perlin, 20 imágenes por split, una combinación por familia, sin escribir nada), que tomó 8.7 s. Solo mostró F1-Val de esa submuestra.

## Entorno

- Windows 11 Pro 10.0.26200, solo CPU, `.venv` del repositorio. `TT_DATA_ROOT = G:\Mi unidad\tt2-insumos`.
- Python 3.13.2 · numpy 2.1.3 · pandas 2.3.3 · scikit-learn 1.6.1 · **scikit-image 0.25.2** · scipy 1.16.3 · statsmodels 0.15.0 · Pillow 12.3.0 (`f24_descriptores.log`, línea 1).
- La versión de scikit-image de la ejecución de Kaggle (29 jul 2026) no consta en el notebook, que solo registra Python 3.12.13 en sus metadatos. Queda en `null` en `configs/protocolo.json`.
- Duración de la corrida definitiva: 229.3 s; la primera tomó 514.3 s, sobre todo por la lectura desde Google Drive. Batería sin las pruebas lentas de F0.3, en tres tandas en primer plano: 63.2 s + 528.3 s + 223.3 s = 814.8 s (13.6 min).

## Descriptores (Enmienda 2, punto 1)

El código original es `notebooks/historial/camera_ready_CE.ipynb`, celda 20 (commit `f126052`). Sus funciones `shannon_entropy`, `edge_density` y `lbp_energy` se copiaron literalmente en `f24_descriptores.py`. `tests/test_f24_descriptores.py` comprueba que el AST de cada función es idéntico al de la celda. El contraste es `img.std()` (línea 42 de la celda) y la imagen se lee igual que en la celda: `np.array(Image.open(path).convert('L'))`, sobre la imagen nativa de 256 × 256.

| Descriptor | Definición |
|---|---|
| Entropía de Shannon | histograma de 256 bins en [0, 256), `density=True`, bins vacíos descartados, `−Σ h·log2 h` (bits) |
| Contraste | DE de los niveles de gris 0–255, `ddof=0` (RMS, no GLCM) |
| Densidad de bordes | fracción de los 65 536 píxeles cuya magnitud de `skimage.filters.sobel(img/255)` es `>` media + 1 DE (`ddof=0`) de esa magnitud |
| Energía LBP | `local_binary_pattern(img, P=24, R=3, method='uniform')` sobre uint8, histograma de 26 bins normalizado, `Σ h²` |

**Equivalencia con el código original.** `test_equivalencia_10_imagenes` ejecuta los imports y las tres funciones de la celda 20 tal como están en el `.ipynb` versionado. Los compara con la reimplementación en 10 imágenes fijas que cubren las tres técnicas, los tres splits y las dos clases (`rtol=1e-12`). `test_descriptores_csv_coincide` compara `descriptores.csv` con la celda en las mismas 10 imágenes. **Esta prueba solo demuestra equivalencia en este entorno, no con la ejecución de Kaggle.** Esa evidencia es la Tabla 11 recalculada (abajo).

**Robustez (diferencias con el original).** El original saltaba sin aviso las imágenes que faltaban. Aquí la ejecución se detiene si falta una imagen, si su SHA-256 no coincide con `meta/data_manifest.csv`, si no es modo L de 256 × 256, o si el sufijo `_label<y>` del archivo no coincide con `y_true`. También exige 908/114/114 por técnica (`f24_descriptores.log`, líneas 11–13).

## Clasificador (§4.3)

Se reutiliza el código de F2.1 (`results/f21_techo5d/f21_techo5d.py`, importado; SHA-256 LF `a2332df5…`) para:
- las rejillas y su orden;
- los hiperparámetros fijos (LogReg con `max_iter=5000`, SVC con `probability=True`, HistGB con `early_stopping=False`, `random_state=42`);
- `StandardScaler` ajustado en train para logreg, svm_rbf y mlp;
- `p_dep = predict_proba[:, 1]` y el F1-Val con `p_dep > 0.5`;
- el desempate con tolerancia 1e-12: familia anterior en `family_order` y, dentro de ella, combinación anterior en la rejilla.

La lectura del índice y del manifiesto viene de F2.3 (`f23_pisos.py`). Hay 128 combinaciones por técnica (384 en total) y ninguna emitió `ConvergenceWarning`.

## Resultados (solo validación)

Fuente: `seleccion.csv`, `f24_descriptores.log` (líneas 21, 27 y 33) y `canonical_fragment.json`.

| Técnica | Familia seleccionada | Combinación (orden en la rejilla) | F1-Val |
|---|---|---|---|
| Perlin | LogReg | C = 10⁻³, `balanced` (#5) | 0.6959 |
| Gray-Scott | RF | `max_depth=None`, `min_samples_leaf=1`, `balanced` (#1) | 0.7280 |
| Lorenz | SVM-RBF | C = 100, γ = 1, `balanced` (#57) | 0.6610 |

- **Gray-Scott:** RF y GBM empatan exactamente (0.728013). Gana RF por ir antes en `family_order` (regla de §4.3).
- F1-Test: **bloqueado hasta la Fase 5**.

## Tabla 11 recalculada

Fuente: `tabla11_recalculada.csv` y `f24_descriptores.log` (líneas 35–53). Se calculó solo con train: depresivo 353 y no depresivo 555 en cada técnica, los mismos conteos en las tres. Se usó t de Welch (depresivo − no depresivo), d de Cohen no ponderado con DE `ddof=0` y Holm (α = 0.05) sobre las 12 comparaciones juntas, igual que en la celda 20. Se comparó con dos ejecuciones originales:
- **A:** la salida guardada de la celda 20, leída del notebook por el script;
- **B:** la ejecución posterior documentada en `notebooks/historial/README.md`. Solo constan los d de Lorenz en entropía (−0.412) y LBP (0.431); Gray-Scott y Perlin son idénticos a A, y Lorenz contraste y bordes no constan (`NaN`).

| Técnica | Descriptor | d recalculado | d A | Δd (A) | d B | Δd (B) | Sig. Holm (recalc. / A) |
|---|---|---|---|---|---|---|---|
| Gray-Scott | entropía | −1.0977 | −1.0977 | < 1e-6 | = A | — | sí / sí |
| Gray-Scott | contraste | −0.3950 | −0.3950 | < 1e-6 | = A | — | sí / sí |
| Gray-Scott | bordes | −0.6566 | −0.6566 | < 1e-6 | = A | — | sí / sí |
| Gray-Scott | LBP | 1.3196 | 1.3196 | < 1e-6 | = A | — | sí / sí |
| Perlin | entropía | 0.0628 | 0.0628 | < 1e-6 | = A | — | no / no |
| Perlin | contraste | 0.2580 | 0.2580 | < 1e-6 | = A | — | sí / sí |
| Perlin | bordes | 0.9451 | 0.9451 | < 1e-6 | = A | — | sí / sí |
| Perlin | LBP | −0.0610 | −0.0610 | < 1e-6 | = A | — | no / no |
| Lorenz | entropía | −0.4227 | −0.4227 | < 1e-6 | −0.412 | −0.0107 | sí / sí |
| Lorenz | contraste | −0.1360 | −0.1360 | < 1e-6 | no consta | — | no / no |
| Lorenz | bordes | −0.0872 | −0.0872 | < 1e-6 | no consta | — | no / no |
| Lorenz | LBP | 0.4432 | 0.4432 | < 1e-6 | 0.431 | +0.0122 | sí / sí |

Hechos (se registran sin ajustar nada):
- **8/12 comparaciones significativas** tras Holm, igual que el original. Ninguna cambia de significancia respecto de A.
- **Las 12 filas reproducen la ejecución A.** La mayor diferencia es |Δd| = 4.8 × 10⁻⁷, que corresponde al redondeo a 6 decimales de la salida impresa. Gray-Scott y Perlin reproducen d a 2 decimales (8/8 filas), así que no hay hallazgo de imágenes distintas en esas técnicas.
- **Lorenz coincide con A y no con B** (diferencias de −0.011 y +0.012 frente a B). Las imágenes de Lorenz vigentes en `tt2-insumos` dan los d de la ejecución A.

## Cifras canónicas

El esquema solo admite F1-Macro y conteos, así que t, p y d quedan solo en `tabla11_recalculada.csv` (como en F1.2).

| Clave | Estado | Valor |
|---|---|---|
| `ref.descriptores.perlin.f1_macro` | verificado | val 0.695902; test `null` (bloqueado hasta la Fase 5) |
| `ref.descriptores.gray_scott.f1_macro` | verificado | val 0.728013; test `null` |
| `ref.descriptores.lorenz.f1_macro` | verificado | val 0.661031; test `null` |
| `m2.tabla11.recalculada.n_sig_holm` | verificado (count) | 8 |
| `m2.tabla11.original.n_sig_holm` | reportado (count) | 8; nota «imágenes de sesión de Kaggle, no versionadas» |

Las dos entradas de la Tabla 11 tienen claves distintas y no llevan `supersedes`, porque se refieren a conjuntos de imágenes distintos.

## Ejecución

Desde la raíz del repo, con `TT_DATA_ROOT` definida y `PYTHONIOENCODING=utf-8` (la consola de Windows no imprime `ŷ`):

```
python results/f24_descriptores/f24_descriptores.py --estimar
python results/f24_descriptores/f24_descriptores.py
python results/f24_descriptores/f24_fragment.py
python -m src.predictions.validate "results/predictions/referencia/*.csv"     > validate.log
CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 python -m src.canonical.build
CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 python -m src.canonical.build --check   > canonical_check.log
CANON_UMAP_PARAMS=... CANON_STRICT=1 F03_UMAP_PARAMS=... F03_DATA_ROOT=<images_grayscale_classblind> \
    F03_BETO_PKL=<EMB_41> F03_GS_PATH=cpu F03_STRICT=1 \
    python -m pytest tests/ --ignore=tests/test_label_independence.py --junitxml=pytest_1_resto.xml
# mismas variables:
python -m pytest tests/test_label_independence.py     -k "not test_C_inventario_y_alineacion and not test_B_invarianza and not test_B_contrato_salida[lorenz]"     --junitxml=pytest_2a_f03.xml
python -m pytest "tests/test_label_independence.py::test_B_contrato_salida[lorenz]" --junitxml=pytest_2b_f03_lorenz.xml
```

**Batería en tres tandas.** La batería completa (unos 12 min) se corrió una vez en segundo plano y dio 415 aprobadas, 3 omitidas, 19 deseleccionadas y 2 fallidas. Esa corrida es anterior a la corrida definitiva, y su registro se sobrescribió. Al repetirla sobre el estado final, Claude Code la detuvo porque el equipo se quedó sin memoria. Se volvió a correr en primer plano, en tres tandas de menos de 10 min cada una. La unión de las tres equivale a `-k "not test_C_inventario_y_alineacion and not test_B_invarianza"` sobre `tests/`.

## Salidas

| Archivo | Contenido |
|---|---|
| `../predictions/referencia/{perlin,gray_scott,lorenz}__descriptors__none__seed42.csv` (+ manifiesto) | `p_dep` del modelo seleccionado, val y test (228 filas c/u). La familia y la combinación ganadoras están en `config` y `model_selection`. **Test sin métricas hasta la Fase 5** |
| `descriptores.csv` | `sample_id`, técnica, split, `y_true` y los 4 descriptores (3408 filas, precisión completa) |
| `rejilla_descriptores.csv` | 3 técnicas × 128 combinaciones: familia, orden, combinación (JSON), F1-Val, `n_iter`, avisos de convergencia |
| `seleccion.csv` | Mejor combinación de cada familia por técnica, con su F1-Val y `seleccionada` |
| `tabla11_recalculada.csv` | 12 filas: conteos, medias y DE por clase, t, p, p de Holm, significancia, d, y comparación con A y B |
| `canonical_fragment.json` | Fuente de las 5 cifras de arriba en `results/canonical_numbers.json` |
| `f24_descriptores.log` | Bitácora de la corrida definitiva |
| `validate.log` | Validador F1.5: 15/15 archivos válidos en `results/predictions/referencia/` |
| `canonical_check.log` | Salida de `build --check`: al día |
| `pytest_1_resto.log`/`.xml`, `pytest_2a_f03.log`/`.xml`, `pytest_2b_f03_lorenz.log`/`.xml` | Batería sin las pruebas lentas de F0.3, en tres tandas: 406 aprobadas; 8 aprobadas, 3 omitidas, 20 deseleccionadas (las 19 lentas más `test_B_contrato_salida[lorenz]`, que corre aparte) y 2 fallidas; 1 aprobada. Total: **415 aprobadas, 3 omitidas, 19 deseleccionadas, 2 fallidas** (`test_C_reproduccion_desde_u[perlin]` y `[lorenz]`, capa C de F0.3 en Windows; las mismas 2 que en F2.1, F2.2 y F2.3) |

SHA-256 de los CSV de predicciones (`csv_sha256` del manifiesto): perlin `92f45fd2…`, gray_scott `a464b321…`, lorenz `98017425…`. Scripts (SHA-256 LF): `f24_descriptores.py` `49c1bd90…`, `f24_fragment.py` `14de8a54…`.
