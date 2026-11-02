# F2.2 — Referencias textuales: BETO (Exp 4.1) y LogReg sobre 768D

Criterio, verificaciones V1–V4 y resultados en `docs/especificacion_f22.md`: §1–§5 preregistradas, §6–§7 completadas después de ejecutar. Formato de las predicciones en `docs/especificacion_predicciones.md`.

## Commits

| Momento | Commit | Fecha (−06:00) | Contenido |
|---|---|---|---|
| Preregistro | `228ee85` | 2026-10-29 18:49 | §1–§5 de la especificación; §6–§7 vacías |
| Infraestructura y scripts | `c586a02` | 2026-10-29 18:57 | `supersedes` y `diff` en `src/canonical/`, `src/predictions/write.py`, scripts de F2.2 |
| Diagnóstico de V3 | `1e67498` | 2026-10-29 20:41 | `diagnostico_v3/`: V3 falla con el checkpoint del 10 de marzo; `EMB_41` proviene de la corrida del 11 de marzo; manejo del empate entre textos idénticos |
| Cambio de checkpoint | `b14b4b9` | 2026-10-29 21:04 | `meta/data_manifest.csv` y `docs/datos.md` §6 con el checkpoint del 11 de marzo |
| Lectura exacta de `p_dep` | `fa50f96` | 2026-10-29 21:10 | `validate.py` lee `p_dep` con `float()`; **código con el que se generaron las predicciones** (`git_commit` de los manifiestos) |
| Ejecución | `18bda5a` | 2026-10-29 21:35 | Predicciones, métricas, fragmento canónico, `canonical_numbers.json`, §6–§7 |
| Fin de línea | `56c3ab4` | 2026-10-29 21:36 | `.gitattributes`: LF en `results/predictions/` para que `csv_sha256` coincida en checkouts de Windows |

No se ejecutó inferencia antes del preregistro.
- **Corrida con el checkpoint del 10 de marzo:** se detuvo en V3 y no escribió predicciones. Su evidencia está en `diagnostico_v3/f22_beto_ckpt_10marzo.log` y `diagnostico_v3/v3_ckpt_10marzo.csv`.
- **Primera corrida con el checkpoint correcto:** se detuvo en la verificación de ida y vuelta de `p_dep` (1 ulp, ver §6.1.4), y su CSV se borró.
- **Segunda corrida:** es la que se reporta, con HEAD en `fa50f96` y el árbol limpio.
- **`f22_fragment.py`:** se ajustó después de `fa50f96` para declarar el valor de las derivadas, y el código ejecutado es el de `18bda5a`.

## Entorno

- Windows 11 Pro 10.0.26200, solo CPU, `.venv` del repositorio. `TT_DATA_ROOT = G:\Mi unidad\tt2-insumos` (Drive para escritorio).
- Python 3.13.2 · torch 2.11.0+cpu · transformers 5.21.0 · tokenizers 0.23.3 · numpy 2.1.3 · pandas 2.3.3 · scikit-learn 1.6.1 · joblib 1.6.0 · scipy 1.16.3 · jsonschema 4.26.0 · pytest 8.4.2.
- Duraciones: `f22_beto.py` 298 s (incluye V3 sobre las 1,136 filas); `f22_logreg768.py` 8 s; batería de pruebas 15.5 min.

## Insumos

SHA-256 verificados por los scripts contra `meta/data_manifest.csv` (commit `b14b4b9`), con rutas relativas a `TT_DATA_ROOT`.

| Insumo | Ruta | SHA-256 |
|---|---|---|
| `CSV[train]` | `processed/train.csv` | `4c677f404c19cf8c7d99a2fbc049d2655264c2f05787b3f448e2f8d2b6c9d37b` |
| `CSV[val]` | `processed/validation.csv` | `d249b14b01149ec1e5cd933ae3349c470f71b8ba79d4745f534610d71c1d9f61` |
| `CSV[test]` | `processed/test.csv` | `e251582ae76e419e344228c3bcd3d6c8f468b481968e3cb6f06f221f0c9f6b5b` |
| `EMB_41` | `exp_4_1_final/Exp_4.1_BETO_HPO_embeddings.pkl` | `64b7682e6d6517e0e643502a96cc80949bb5b35d2a4c04a4ba345243ad1b5d44` |
| `CKPT_41` (carpeta, SHA agregado) | `exp_4_1_final/checkpoint-342/` (corrida del 11 de marzo) | `9084df928a97e2fc5ab49fb7376bb1d565a646d0d5160bd096a2fc205972c074` |
| `CKPT_41` / `model.safetensors` | ídem | `c4f35147c660045ef8a950e83ee73fa7d4ff915c56504317322b09458c147d7b` |
| `UMAP_PARAMS` (cifras canónicas y pruebas) | `umap_params.pkl` | `081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224` |
| Índice maestro | `meta/sample_index.csv` | `74dc778292447c66be60b35b5ec224c53b01433423923fa6d6ddef91749de0f5` |
| Manifiesto | `meta/data_manifest.csv` | `3afbe63bb07fe3e6f43f45bb81edb2bff88995d45a6f05b45ad4db1c3765ec47` |

SHA-256 de los scripts, sobre los bytes con LF tal como están en git:

| Script | SHA-256 |
|---|---|
| `common.py` | `eb8101c2f4a8ae93aa25da4c3dd59667c10ec7165de7caa75c8d3618d477607f` |
| `f22_beto.py` | `386622c90f1b9bbe3c3b22522bfa1717436e91935b7c558bfe40e0b219bf0452` |
| `f22_logreg768.py` | `3715d0a45fff4d3fbdb8b6cb60304ee31178d2273b8fcf4609aca66c16bf634b` |
| `f22_fragment.py` | `9e6870174dec700da9fa2e635711a760cf37be3e6e122f1c17213ae5f8c3ee87` |
| `src/predictions/write.py` | `4a1175c5bd1349bb7ed1e9853be78b1563176e2c9bfa595bd7b9269a0870a6b1` |

## Ejecución

Desde la raíz del repo, con `TT_DATA_ROOT` definida:

```
python results/f22_referencias_textuales/f22_beto.py
python results/f22_referencias_textuales/f22_logreg768.py
python -m src.predictions.validate "results/predictions/referencia/*.csv"
python results/f22_referencias_textuales/f22_fragment.py
CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 python -m src.canonical.build
CANON_UMAP_PARAMS=... CANON_STRICT=1 F03_UMAP_PARAMS=... F03_DATA_ROOT=<images_grayscale_classblind> \
    F03_BETO_PKL=<EMB_41> F03_GS_PATH=cpu F03_STRICT=1 python -m pytest tests/
```

## Salidas

| Archivo | Contenido |
|---|---|
| `../predictions/referencia/text__beto__finetune__seed42.csv` (+ manifiesto) | `p_dep` de la cabeza de BETO, val y test (228 filas) |
| `../predictions/referencia/emb768__logreg__none__seed42.csv` (+ manifiesto) | `p_dep` de LogReg sobre `EMB_41`, val y test; **test sin métricas hasta la Fase 5** |
| `beto_metricas.json` | V1–V3, historial de `eval_f1_macro`, métricas de val y test de BETO y comparación con lo reportado |
| `v3_embeddings.csv` | V3 por fila: coseno, vecino = propia fila, empate con texto idéntico, rango |
| `seleccion_C.csv` | F1-Val por C (13 valores) y el seleccionado |
| `canonical_fragment.json` | Cifras de §7 (fuente de `results/canonical_numbers.json`) |
| `f22_beto.log`, `f22_logreg768.log` | Bitácoras de ejecución |
| `pytest.log`, `pytest.xml` | Batería completa: 263 aprobadas, 7 omitidas (ruta GPU sin CUDA), 2 fallidas (capa C de F0.3 en Windows; §6.6) |
| `diagnostico_v3/` | Diagnóstico de la falla de V3: 18 variantes de extracción, V3 con cada checkpoint, LogReg proxy sobre `EMB_41` (solo F1-Val) |

Las salidas no contienen texto del corpus ni datos personales: solo `sample_id`, etiquetas, probabilidades, métricas y hashes.

## Resultados principales

- **BETO:** F1-Val 0.7199 (= `trainer_state.json`), **F1-Test 0.7039**, [[54, 16], [16, 28]]. Lo reportado en TT-I era 0.7586 con [[57, 13], [13, 31]]; la procedencia de esas cifras queda sin resolver (§6.5).
- **LogReg sobre 768D:** C = 10⁻³·⁵, F1-Val 0.7305.
- **δ = 0.0035** (antes 0.058) y **Δ WITCOM = −0.0103** (antes 0.0444). Hay que tratarlos en F1.2 antes de congelar el protocolo (§7).
