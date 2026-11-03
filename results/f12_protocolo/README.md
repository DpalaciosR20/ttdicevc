# F1.2: congelamiento del protocolo preregistrado

Evidencia de la tarea F1.2 del plan v2 (rol SV). El protocolo es `docs/protocolo_preregistrado.md`, versión v1 (congelado) con la etiqueta anotada `protocolo-v1` sobre `91c84df` (2026-11-02 18:14:54 −0600). Tiene una enmienda aclaratoria, la Enmienda 1 (`0373069`, `diff_enmienda_1.log`), anterior a toda consulta de prueba de los finalistas.

## Artefactos

| Archivo | Contenido |
|---|---|
| `configs/protocolo.json` | Parámetros del protocolo legibles por código (§2, §4–§8, §13) |
| `tests/test_protocolo.py` | Coherencia del JSON con el documento, con `canonical_numbers.json`, con la matriz y con el esquema de F1.5; verifica también las huellas SHA-256 del encabezado |
| `src/protocol/verdict.py` | Regla de decisión (§5.2–§5.5): p por *bootstrap*, Holm por familia con la regla de empates, intervalo percentil a 1 − 2α_h, veredictos V1–V4 y G1–G4. No calcula réplicas: la Fase 5 no está implementada |
| `tests/test_verdict.py` | Pruebas sintéticas de `verdict.py` |
| `f12_fragment.py` | Genera `canonical_fragment.json`; ninguna cifra se escribe a mano |
| `canonical_fragment.json` | `f12.umbral_f1_visual`, `f12.piso_mayoritaria.f1_macro`, `f12.variabilidad_beto` |
| `canonical_check.log` | Salida de `python -m src.canonical.build --check` (estricto) |
| `pytest_suite.log`, `pytest.xml` | Suite completa |

Enmienda de la matriz (fase `sensibilidad`, F5.5): `docs/matriz_experimentos.md` §4, §5 y §11; `configs/matriz_experimentos.json`; `src/predictions/schema.py` y la regla 7 del validador; `docs/especificacion_predicciones.md` §3.4; `select_second` en `src/selection/select_finalist.py`.

## Verificación previa (Paso 0)

Se compararon las cifras de §2, §3, §4.4, §5.1, §5.3, §8 y §13 del protocolo con `results/canonical_numbers.json`, y los conteos por clase con `meta/sample_index.csv`.

- **Sin discrepancias de valor.**
  - `meta/sample_index.csv`: 908 / 114 / 114; train 555 / 353, val 70 / 44, test 70 / 44. Coincide con §2 y con `n.*` y `support.*`.
  - §3 coincide con `ref.beto_exp41.f1_macro` (val 0.7199, test 0.7039), `witcom.delta_beto_gs_resnet18.f1_macro` (−0.0103) y `ref.emb768_logreg.f1_macro` (val 0.7305).
  - Los δ de §8 y §13 (0.1019 / 0.0510 / 0.0408) coinciden con `f11.delta_margen.*`.
  - El piso de 0.529 de §5.3 coincide con `floor.brightness.gray_scott.f1_macro`.
- **Cifras que no tenían respaldo canónico.** Tres cifras eran aritméticamente correctas pero no estaban en el archivo: 0.6019 (§5.1, §13), 0.3804 (§4.4) y ±0.0128 (§8). Ahora están registradas en este fragmento.
- **Observaciones que no cambian el protocolo:**
  - El «valor esperado 0.5» del aleatorio estratificado es una aproximación de primer orden. Con la proporción de train (353/908) y la de test (44/114) da 0.49999.
  - El orden «después de F3.5» del segundo mejor modelo (§10) no lo hace cumplir ningún código. La compuerta técnica es la etiqueta `protocolo-v1`.

## Cifras del fragmento

| Id | Valor | Cálculo |
|---|---|---|
| `f12.umbral_f1_visual` | test 0.601948… | 0.5 + r (F1_BETO − 0.5), con r de `configs/protocolo.json` y F1_BETO de `ref.beto_exp41.f1_macro` (test) |
| `f12.piso_mayoritaria.f1_macro` | test 0.380435… | ŷ = 0 sobre los soportes de test de `meta/sample_index.csv`; confusión {tp 0, fp 0, fn 44, tn 70} |
| `f12.variabilidad_beto` | val 0.012815… | `best_metric` de `f22_referencias_textuales/diagnostico_v3/f22_beto_ckpt_10marzo.log:27` − el de `f22_referencias_textuales/f22_beto.log:27` |

Las tres tienen estado `extraido`.

- El umbral no puede declararse como derivada formal (`depends_on`), porque el esquema solo admite `sum`, `mean`, `std_ddof1` y `diff`. Por eso `tests/test_protocolo.py` lo recalcula.
- Los δ de §13 se citan por su id de F1.1, para no duplicar claves.
- r, F1_azar, α, B y la semilla del *bootstrap* solo están en `configs/protocolo.json`, porque el esquema solo admite métricas F1 y conteos.
- `tests/test_canonical_numbers.py` exime el prefijo `f12.` de la prueba de alcanzabilidad.

## Decisiones de implementación aprobadas por SV durante la tarea

1. **Segundo mejor modelo.** Se excluye al finalista y se aplica de nuevo la regla completa de §5 de la matriz. El texto se añadió al protocolo, §8.
2. **Empates de p en Holm.** Todas las técnicas empatadas reciben el menor α_h de los rangos que ocupan. El texto se añadió al protocolo, §5.4.
3. **Registro de las cifras de §13.** Se registran como se describe arriba.
4. **«§0».** Es el encabezado del documento: versión, huellas SHA-256 y enmienda de la matriz.

Hay además una elección declarada que no cambia ninguna decisión: el intervalo percentil se calcula con `numpy.quantile(method='linear')` (`contraste.multiplicidad.interval.quantile_method`).

## Congelamiento y pruebas por mutación (posteriores al merge)

- **Etiqueta.** `protocolo-v1` es una etiqueta anotada sobre `91c84df` (merge del PR #16), con fecha 2026-11-02 18:14:54 −0600. El commit `d0d6a24` solo completa commit y fecha en el encabezado del protocolo; `diff_encabezado_post_etiqueta.log` contiene `git diff protocolo-v1 HEAD` sobre el protocolo.
- **Mutaciones.** `mutaciones.py` introduce 19 defectos, uno a la vez, en `src/protocol/verdict.py`, `select_second`, `configs/protocolo.json`, `configs/matriz_experimentos.json` y el protocolo. Corre las pruebas correspondientes y restaura cada archivo. La salida está en `mutaciones.log`. La suite completa después de restaurar está en `pytest_suite_post_mutaciones.log`.

## Reproducción

```powershell
.venv\Scripts\python.exe results/f12_protocolo/f12_fragment.py
$env:CANON_UMAP_PARAMS="G:\Mi unidad\tt2-insumos\umap_params.pkl"; $env:CANON_STRICT="1"
.venv\Scripts\python.exe -m src.canonical.build
.venv\Scripts\python.exe -m src.canonical.build --check
.venv\Scripts\python.exe -m pytest tests -q -rs --junitxml=results/f12_protocolo/pytest.xml
```

## Entorno

Windows 11, `.venv` local. Versiones: Python 3.13.2, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, jsonschema 4.26.0 y pytest 8.4.2. No usa GPU ni `TT_DATA_ROOT`. Lo único externo es `umap_params.pkl`, que se usa para la reconstrucción estricta de las cifras canónicas.

## Resultado de la suite

`pytest_suite.log`: **339 passed, 30 skipped**. Las 30 omitidas son de `tests/test_label_independence.py` (capas B y C de F0.3), que requieren `F03_UMAP_PARAMS`. No se definió en esta corrida, igual que en F1.3. Por eso las capas B y C de F0.3 no se ejecutaron en esta tarea.

Además:
- `python -m src.canonical.build --check` (estricto) reporta que `canonical_numbers.json` está al día.
- `python -m src.predictions.validate "results/predictions/**/*.csv"` da 2/2 archivos válidos.
