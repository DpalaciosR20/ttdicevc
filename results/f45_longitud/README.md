# F4.5 — Referencia de longitud

**Especificación:**
- `docs/protocolo_preregistrado.md` §4.2 y §6;
- Enmienda 2, punto 4, y Enmienda 4, A9;
- `configs/protocolo.json` → `referencias.longitud` y `enmienda_4.longitud.clasificador`.

**Formato de las predicciones:** `docs/especificacion_predicciones.md`.

**No se calculó ninguna métrica de prueba** (protocolo §10). Las predicciones de test se archivaron sin evaluarlas; el contraste L_t es de la Fase 5. `tests/test_f45_longitud.py` comprueba que el log, `resultado.json`, la rejilla y el manifiesto no registran métricas de test.

## Commits

| Momento | Commit | Contenido |
|---|---|---|
| Script, antes de ejecutar | `ef9bae0` (2026-11-07 00:37 −0600) | `f45_longitud.py` y `tests/test_f45_longitud.py`. **Es el código con el que se generaron las predicciones** (`git_commit` del manifiesto, árbol limpio) |
| Resultados | el commit que introduce este README | Predicciones, rejilla, `resultado.json`, logs y bitácora `docs/bitacora/F4.5-referencia.md` |

## Ejecución

```text
python results/f45_longitud/f45_longitud.py
python -m src.predictions.validate results/predictions/referencia/text_length__logreg__none__seed42.csv
```

- Entorno: Windows 11 Pro 10.0.26200, CPU, `.venv` del repositorio.
- Versiones: Python 3.13.2, scikit-learn 1.6.1, numpy 2.1.3, pandas 2.3.3.
- Duración: 1.9 s.

## Archivos

| Archivo | Contenido |
|---|---|
| `f45_longitud.py` | Ajuste de la LogReg. Reutiliza `logreg_grid`, `p_dep_of`, `f1_val_of`, `select`, `n_iter_of` y `load_index` de `results/f23_pisos/f23_pisos.py` |
| `f45_longitud.log` | Salida completa: insumos con SHA-256, 26 combinaciones con su F1-Val y la selección |
| `rejilla_longitud.csv` | F1-Val, `n_iter` y avisos de convergencia por combinación |
| `resultado.json` | Selección, F1-Val y parámetros del estimador (sin test) |
| `validate.log` | `src.predictions.validate`: 1/1 válido |
| `pytest.log` | `tests/test_f45_longitud.py -v`: 7 aprobadas |
| `../predictions/referencia/text_length__logreg__none__seed42.csv` (+ `.manifest.json`) | 114 filas de val y 114 de test, fase `referencia` |
