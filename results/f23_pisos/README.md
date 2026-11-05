# F2.3 — Pisos

Especificación: `docs/protocolo_preregistrado.md` §4.4 y Enmienda 2 (puntos 2 y 3); `configs/protocolo.json` → `pisos`, `pisos_triviales`, `piso_brillo`. Formato de las predicciones: `docs/especificacion_predicciones.md`. **No se calculó ninguna métrica de prueba** (protocolo §10), salvo una comprobación: el F1-Test de la mayoritaria coincide con su cifra de diseño `f12.piso_mayoritaria.f1_macro`, y el log solo registra si coincide. Las demás predicciones de test quedan archivadas hasta la Fase 5.

## Commits

| Momento | Commit | Fecha (−06:00) | Contenido |
|---|---|---|---|
| Enmienda 2, antes de calcular | `41dd200` | 2026-11-04 19:29 | Enmienda 2 al final del protocolo; `text_length` en el vocabulario (`schema.py`, especificación §3.2); `pisos_triviales`, `piso_brillo`, `descriptores` (solo estructura) y `referencias.longitud` en `configs/protocolo.json`; pruebas |
| Scripts, antes de ejecutar | `53f6e28` | 2026-11-04 19:29 | `f23_pisos.py`, `f23_fragment.py`, `diff_enmienda_2.log`. **Es el código con el que se generaron las predicciones** (`git_commit` de los 5 manifiestos, árbol limpio) |
| Resultados | el commit que introduce este README (`[F2.3] Predicciones, tablas y cifras canónicas de los pisos`) | 2026-11-04 | Predicciones, rejilla, fragmento canónico, `canonical_numbers.json`, registros de validación y de pruebas, este README. La corrida es del 2026-11-04 19:30 (`created_at` de los manifiestos: 2026-11-05T01:30 UTC) |

**Enmienda 2 contra la etiqueta.** `diff_enmienda_2.log` es `git diff protocolo-v1 HEAD -- docs/protocolo_preregistrado.md`, tomado en `41dd200`. Solo contiene tres cambios:
- la línea de congelamiento del encabezado (`d0d6a24`, ya registrada en F1.2);
- la Enmienda 1 (`0373069`);
- la Enmienda 2.

El encabezado conserva la huella de `configs/protocolo.json` en `protocolo-v1` (`92ef8774…`). `tests/test_protocolo.py` la verifica con `git show protocolo-v1:configs/protocolo.json` y fija la huella vigente en `CONFIG_SHA256_ENMIENDA_2` (`8538487a…`).

Antes de la corrida completa se hizo una estimación acotada (`--estimar`: una técnica y una combinación, sin escribir nada), que tomó 33.5 s. Mostró los F1-Val de la mayoritaria y del estratificado y el de la primera combinación de Perlin, solo de validación, además de la comprobación de la mayoritaria.

## Entorno

- Windows 11 Pro 10.0.26200, solo CPU, `.venv` del repositorio. `TT_DATA_ROOT = G:\Mi unidad\tt2-insumos`.
- Python 3.13.2 · numpy 2.1.3 · pandas 2.3.3 · scikit-learn 1.6.1 · Pillow 12.3.0 · scipy 1.16.3 · joblib 1.6.0 · jsonschema 4.26.0 · pytest 8.4.2.
- Duración: `f23_pisos.py` 84.9 s (lectura y verificación de 3408 imágenes y 78 ajustes). Batería sin las pruebas lentas de F0.3 (`-k`, 19 deseleccionadas): 750.8 s (12.5 min; `pytest.log`).

## Insumos

| Insumo | Ruta | SHA-256 |
|---|---|---|
| Imágenes Perlin (cada una verificada contra `meta/data_manifest.csv`) | `images_grayscale_classblind/perlin` (1136) | agregado `b8ce53c6fa339d3a4dcca2ba647c55aeaec79909f7137563efd8b5d04f7e3ccb` |
| Imágenes Gray-Scott (ídem) | `images_grayscale_classblind/gray_scott` (1136) | agregado `c3593c9bd04b4abb45afe7958bded087b53b7812a7cb11e72bda8e382f2d31db` |
| Imágenes Lorenz (ídem) | `images_grayscale_classblind/lorenz` (1136) | agregado `6fc478a58477f5ba76d9105435be179c52539db2d67dbee3918a67dc5cfdde9b` |
| Índice maestro | `meta/sample_index.csv` | `74dc778292447c66be60b35b5ec224c53b01433423923fa6d6ddef91749de0f5` (bytes en disco, CRLF; el valor de los manifiestos) · LF `d2f789bb2812e55bed0ddc71ed13695fe092859722e4826a324a8569f1bd040a` |
| Manifiesto de datos | `meta/data_manifest.csv` | `a4fb841cddb57e93328d9866b3cce1713dceab0275e643d1627e3fd8d7aa1e20` (CRLF, en disco) · LF `3afbe63bb07fe3e6f43f45bb81edb2bff88995d45a6f05b45ad4db1c3765ec47` |
| Parámetros del protocolo (LF) | `configs/protocolo.json` | `8538487a8911e2ef2c906a335337727ee4ba84ae26f6d0f042655640720829b0` (vigente desde la Enmienda 2) |

Scripts, SHA-256 sobre los bytes con LF tal como están en git:

| Script | SHA-256 |
|---|---|
| `f23_pisos.py` | `31d90b960d51cb52e18f631dd89ee5b975cef159ded6d6b556dd6231783316cb` |
| `f23_fragment.py` | `e91b48bad9fcea7503c5857d294c8eb60362d02a68e070eb649b4b5e087efde6` |
| `src/predictions/write.py` | `b56e111bdd57f24716c9033c7b6d692794e04a97348c24d7b272e4d812696c3e` |

## Valores que el protocolo no fija

Se fijaron en el docstring de `f23_pisos.py` en `53f6e28`, antes de ejecutar:

| Punto | Valor usado |
|---|---|
| Extracción del estratificado | `default_rng(42).random(228) < 353/908`, una sola extracción sobre val y luego test, cada uno ordenado por `sample_id` |
| Lectura de la imagen | `PIL.Image.open`; se exige modo `L` y tamaño 256 × 256; `np.asarray(img, float64).mean() / 255` |
| Orden de la rejilla de LogReg | C (13 valores, de menor a mayor) varía más lento; `class_weight` con `null` primero. Es el mismo orden que en F2.1 |
| `max_iter` de LogReg | 5000 (igual que en F2.1 y F2.2). Ninguna combinación emitió `ConvergenceWarning` |
| Hiperparámetros no nombrados | Valores por defecto de scikit-learn 1.6.1 |
| Cifras canónicas | Claves nuevas `ref.piso.*`. Las cifras WITCOM `floor.brightness.*` se conservan y no se sustituyen (sin `supersedes`) |

## Ejecución

Desde la raíz del repo, con `TT_DATA_ROOT` definida:

```
python results/f23_pisos/f23_pisos.py --estimar
python results/f23_pisos/f23_pisos.py
python results/f23_pisos/f23_fragment.py
python -m src.predictions.validate "results/predictions/referencia/*.csv"     > validate.log
CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 python -m src.canonical.build
CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 python -m src.canonical.build --check   > canonical_check.log
CANON_UMAP_PARAMS=... CANON_STRICT=1 F03_UMAP_PARAMS=... F03_DATA_ROOT=<images_grayscale_classblind> \
    F03_BETO_PKL=<EMB_41> F03_GS_PATH=cpu F03_STRICT=1 \
    python -m pytest tests/ -k "not test_C_inventario_y_alineacion and not test_B_invarianza" --junitxml=pytest.xml
```

## Salidas

| Archivo | Contenido |
|---|---|
| `../predictions/referencia/none__majority__none__seed42.csv` (+ manifiesto) | `p_dep = 0.0`, val y test (228 filas) |
| `../predictions/referencia/none__stratified__none__seed42.csv` (+ manifiesto) | `p_dep ∈ {0.0, 1.0}` (228 filas). Solo completa el formato: en los contrastes interviene la constante 0.5 |
| `../predictions/referencia/{perlin,gray_scott,lorenz}__brightness__none__seed42.csv` (+ manifiesto) | `p_dep` del LogReg de brillo seleccionado, val y test (228 filas c/u); **test sin métricas hasta la Fase 5** |
| `rejilla_brillo.csv` | 3 técnicas × 26 combinaciones: orden en la rejilla, combinación (JSON), F1-Val, `n_iter`, avisos de convergencia |
| `canonical_fragment.json` | `ref.piso.{majority,stratified}.f1_macro` y `ref.piso.brightness.{técnica}.f1_macro` (fuente de `results/canonical_numbers.json`) |
| `f23_pisos.log` | Bitácora de la corrida (F1-Val de cada combinación) |
| `diff_enmienda_2.log` | Diff del protocolo contra `protocolo-v1` |
| `validate.log` | Validador F1.5: 12/12 archivos válidos en `results/predictions/referencia/` |
| `canonical_check.log` | Salida de `build --check`: al día |
| `pytest.log`, `pytest.xml` | Batería sin las pruebas lentas de F0.3: 400 aprobadas, 3 omitidas, 19 deseleccionadas, 2 fallidas (`test_C_reproduccion_desde_u[perlin]` y `[lorenz]`, capa C de F0.3 en Windows; las mismas 2 que en F2.1 y F2.2) |

SHA-256 de los CSV de predicciones (`csv_sha256` del manifiesto): majority `14c4b6ef…`, stratified `2c5a9407…`, perlin `93e6a60e…`, gray_scott `f5183e70…`, lorenz `de67b6e5…`.

## Resultados (solo validación)

Fuente: `f23_pisos.log`, `rejilla_brillo.csv` y `canonical_fragment.json`.

| Piso | Selección (orden en la rejilla) | F1-Val | Empatadas (1e-12) |
|---|---|---|---|
| Mayoritaria | — | 0.3804 | — |
| Aleatorio estratificado | — (`default_rng(42)`) | 0.5558 | — |
| Brillo, Perlin | C = 10^−3, `balanced` (#5) | 0.6452 | 11 |
| Brillo, Gray-Scott | C = 10^−1, `class_weight=None` (#12) | 0.4462 | 1 |
| Brillo, Lorenz | C = 10^−3.5, `balanced` (#3) | 0.4519 | 3 |

- **Mayoritaria:** su F1-Test coincide con la cifra de diseño `f12.piso_mayoritaria.f1_macro` (`f23_pisos.log`, línea 11).
- El F1-Val del estratificado es el de una sola extracción. En los contrastes interviene la constante 0.5 (Enmienda 2, punto 3).
- Las cifras WITCOM `floor.brightness.*` son de prueba, se obtuvieron con otro evaluador no documentado y tienen estado exploratorio. No son comparables con estos F1-Val y no se sustituyen.
