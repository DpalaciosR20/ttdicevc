# F0.4 — Archivo de cifras canónicas

Resultados de la tarea **F0.4** del plan v2 (`plan/plan_tt2_v2.md`): archivo de cifras canónicas generado por código, del que se toman todas las cifras del documento de TT-II.

- **Archivo canónico:** `results/canonical_numbers.json` (125 cifras), generado por `src/canonical/build.py` y validado contra `src/canonical/schema.json`.
- **Pruebas:** `tests/test_canonical_numbers.py`: 185 pasan y 1 da XFAIL esperado (Exp 4.1, ver abajo).

## Archivos

| Archivo | Contenido |
| --- | --- |
| `canonical_fragment.json` | Fragmento de F0.4 con 20 cifras transcritas (`reportado` / `exploratorio`). Cada una indica su fuente y en qué parte del documento está. |
| `pytest.log` | Salida de `pytest -v -rxs` de la corrida estricta. |
| `junit.xml` | La misma corrida en formato JUnit. |

## Estados

| Estado | Significado | Cifras |
| --- | --- | --- |
| `verificado` | `build.py` lo recalcula desde datos por muestra (`predictions.csv`, `umap_params.pkl`) | 105 |
| `extraido` | `build.py` lo lee de un artefacto del repo sin recalcularlo por muestra (solo aparece si se construye sin el pkl) | 0 |
| `reportado` | Transcrito de un documento; en el repo no hay datos para recalcularlo | 4 |
| `exploratorio` | Usa un conjunto de prueba consultado antes de congelar el protocolo (WITCOM) | 16 |

Una derivada (`depends_on`) toma el estado más débil de sus dependencias. Los fragmentos de otras tareas se agregan como `results/<tarea>/canonical_fragment.json`. Si una clave se repite, el build falla.

## Contenido

- **Estructurales:** `n.{train,val,test,total}`, `support.{split}.label{0,1}` y `saturation.{split}.n_samples` (hallazgo 7). Todos `verificado` desde el pkl. `n.total` y `support.total.*` son derivadas (`sum`).
- **F3.2:** 54 F1-val por corrida, recalculados desde `p_dep` y acompañados de su matriz de confusión. Se cruzan con `best_f1_val` de cada JSON, con sklearn y con `summary.csv`. Además, 18 medias y 18 std (`ddof=1`) por configuración, derivadas y cruzadas con `summary_by_config.csv`. El test va en `null` con su motivo (bloqueado hasta F3.5).
- **Referencias transcritas:**
  - BETO Exp 4.1.
  - TF-IDF + lematización.
  - Baselines 5D (LogReg/SVM, RF, MLP).
  - CNN Gray-Scott WITCOM: 5 semillas, media y std.
  - Piso de brillo por técnica.
  - Tabla de HPO del camera-ready.

## Hallazgos

1. **BETO Exp 4.1, F1-Test = 0.7586 (`reportado`): no se puede alcanzar con los soportes de test 70/44.** El F1-Macro alcanzable más cercano es 0.758985 (TP=36, TN=51), que redondea a 0.7590. La prueba se marca `xfail(strict=True)`, así que si la cifra se corrige el XPASS la hace fallar y obliga a quitar la marca. Para reconciliarla hacen falta las predicciones por muestra de BETO (F2.2).
2. **La std de WITCOM (0.0095) es la poblacional (`ddof=0`).** Con la convención de `run_grid.py` (`ddof=1`) sería 0.0107. Se registra tal cual, con una nota, y la prueba de agregados verifica que sea `ddof=0`. Al compararla con las std de F3.2 hay que usar una sola convención.
3. Todas las demás cifras transcritas (19) se pueden alcanzar con los soportes reales.
4. La ruta sin pkl (tabla del hallazgo 7 + `y_true` de `predictions.csv`) da exactamente los mismos conteos que el pkl (`test_conteos_sin_pkl_coinciden`).

## Procedencia

- **Ejecución:** local, 24 oct 2026, Windows 11.
  - Python 3.13.2, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, joblib 1.6.0, jsonschema 4.26.0, pytest 8.4.2.
  - Se agrega `jsonschema==4.26.0` a `requirements.txt`; esa versión no se ha verificado en Kaggle.
- **Código:** sobre el commit `2e82727`. El build y las pruebas entran en el commit que agrega esta carpeta.
- **Insumos (sha256, con fines de línea normalizados a LF):**

| Insumo | sha256 |
| --- | --- |
| `umap_params.pkl` (externo, `tt2-insumos`) | `081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224` |
| `results/f32_primera_pasada/predictions.csv` | `46a9a4cba45681de266c4d9e9ce15c5d61aa57ca8164574959ce56b930c8c30e` |
| `results/f32_primera_pasada/summary.csv` | `5cf0524a60e5b983b7551e9ea85be5eb4523424e59676b238cc95efcb621e70b` |
| `results/f32_primera_pasada/summary_by_config.csv` | `197529356fa2128e897e375a800c3162b6017fb3cbc2e5f18de3367b9056fdda` |
| `results/f32_primera_pasada/grid_meta.json` | `903976351f665577ed020efcf1bd949ea5e11480b08557541dd08c44515f320e` |
| `results/f04_cifras_canonicas/canonical_fragment.json` | `6ce7c079b3a1e2d845fece8d6db097fe643b6d5afe1de8687ebf768e4066ae52` |
| `results/f32_primera_pasada/runs/*.json` (54) | listados uno por uno en `inputs` del archivo canónico |

## Reproducir

```bash
pip install jsonschema==4.26.0
export CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1
python -m src.canonical.build            # escribe results/canonical_numbers.json
python -m src.canonical.build --check    # falla si el archivo no está al día
python -m pytest tests/test_canonical_numbers.py -v -rxs \
    --junitxml=results/f04_cifras_canonicas/junit.xml | tee results/f04_cifras_canonicas/pytest.log
```

Sin `CANON_UMAP_PARAMS`, la suite corre igual. En ese caso solo se omite la comparación con el archivo commiteado; con `CANON_STRICT=1`, esa omisión cuenta como falla.
