# Correr F4.2 (reintentos de Gray-Scott) y la adenda de F0.3 (capa C) en Kaggle

> Tareas: F4.2 del plan v2, punto 5 (reintentos de Gray-Scott), y adenda de F0.3 (capa C de Perlin y Lorenz) · Rol: SV
> Convención general: `docs/ejecucion_kaggle.md` · Especificación: `docs/plan_fase4.md`, F4.2

En una sola sesión de Kaggle con CPU se hacen dos cosas:

1. **Reintentos de Gray-Scott** (`src/fase4/reintentos_gs.py`).
   - Regenera las 908 imágenes de `train` por la ruta CPU de producción (`generate_and_save_grayscale` → `run_gray_scott_safe`).
   - Registra por muestra los reintentos del *wrapper* y si se usó el patrón de respaldo. La lógica del generador no se modifica: el registro envuelve `run_gray_scott` en tiempo de ejecución.
   - Compara cada PNG regenerado con el almacenado: sha256 contra `meta/data_manifest.csv` y, además, píxel a píxel.
   - La ruta GPU no se usa (F0.3, H2).
2. **Adenda de F0.3, capa C de Perlin y Lorenz.**
   - Se corre `tests/test_label_independence.py` sin cambios, sobre `tt2-insumos`, con la misma muestra estratificada que en F0.3 (`F03_N_PER_CELL=3`: 18 muestras por técnica, semilla 2026).
   - La capa C de Lorenz sobre los artefactos regenerados (bitácora F0.3, H3) no tenía un log versionado. La de Perlin se repite en el mismo entorno.

El análisis estadístico de F4.2 (Fisher, Mann–Whitney, Holm, figuras y F1 por estrato de Lorenz) **no** se corre aquí. Se corre después en local con `src/fase4/regimen.py`, a partir del CSV que produce esta sesión.

**G2.** No se carga ningún modelo ni se hace inferencia: solo regeneración de imágenes y comparación.

## Insumos del notebook

- **Acelerador:** ninguno (CPU).
- **Internet:** activado. Se necesita para `git clone` y para `pip install noise==1.2.2`, que usa Perlin en la capa C.
- **Inputs:** `santiagosoria26ss/tt2-insumos` (imágenes, `umap_params.pkl` y CSV).

## Celdas

**Celda 1: repositorio.** Igual que en `docs/correr_final_kaggle.md`, con el `COMMIT` que indique Claude Code. La última línea se cambia por la compuerta de la Fase 4:
```python
run(["git", "merge-base", "--is-ancestor", "enmienda-5", "HEAD"])   # falla si el commit no desciende de la Enmienda 5
```

**Celda 2: insumos y entorno.** `TT_DATA_ROOT` es la ruta anidada de `docs/datos.md` §2 (la misma de F4.1).
```python
import os
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt noise==1.2.2
!python -m src.build_manifest --verify
!nproc && python -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
```
- Debe decir `manifiesto OK`.
- Anota `nproc`: es el número de procesos de la celda 4.

**Celda 3: prueba de humo y estimación** (≈ 2–5 min). Usa 16 muestras de `train`, espaciadas de forma uniforme, y escribe en `/tmp`, fuera de la salida del notebook (lección de F3.7).
```python
!python -m src.fase4.reintentos_gs --out-dir /tmp/f42_humo/out --regen-dir /tmp/f42_humo/png --limit 16 2>&1 | tail -40
```
- Debe imprimir `resumen` con `n = 16`. Lo esperable es `n_sha256_coincide = 16` y `n_pixeles_coinciden = 16`.
  - Si el sha256 no coincide pero los píxeles sí, la diferencia viene del codificador PNG (versión de Pillow). No se detiene: es el PENDIENTE `bit_a_bit` del reporte. Anótalo.
  - Si **los píxeles** no coinciden en alguna muestra, detente y avísale a Claude Code.
- `tiempo.estimacion_908_segundos` es la duración estimada de la regeneración completa con `nproc` procesos. A eso se le suman ≈ 15 min de la capa C.
  - La capa C de Lorenz tardó 703.52 s en Colab con la misma muestra (F0.3, corrida 4).
- Si la estimación total pasa de 4 h, detente y avísale a Claude Code.

**Celda 4: verificación, regeneración, capa C, huellas y zip, en una sola celda.** Se ejecuta en la versión guardada (*Save Version → Save & Run All*), no en la sesión interactiva (lección de F3.5, H5).

Duración estimada: 1.5–2.5 h (ver «Duración estimada»). El valor de la celda 3 manda.
```python
import os
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
D = os.environ["TT_DATA_ROOT"]
OUT = "results/f42_regimen/reintentos"
C = "results/f03_independencia_etiqueta/adenda_capa_C"
PNG = "/kaggle/working/f42_gs_train_png"
!mkdir -p {OUT}/entorno {C}
!python -m src.build_manifest --verify 2>&1 | tee {OUT}/datos_verificacion.log
!pip freeze > {OUT}/entorno/pip_freeze.txt && python --version > {OUT}/entorno/python.txt && nproc > {OUT}/entorno/nproc.txt && lscpu > {OUT}/entorno/lscpu.txt
!python -m src.fase4.reintentos_gs --out-dir {OUT} --regen-dir {PNG} 2>&1 | tee {OUT}/reintentos_gs.log
!F03_UMAP_PARAMS={D}/umap_params.pkl F03_DATA_ROOT={D}/images_grayscale_classblind F03_N_PER_CELL=3 F03_STRICT=1 \
  python -m pytest tests/test_label_independence.py -k "test_C_ and (perlin or lorenz)" -v -rs -s \
  --junitxml={C}/kaggle_capa_C_perlin_lorenz.xml 2>&1 | tee {C}/kaggle_capa_C_perlin_lorenz.log
!cp {OUT}/entorno/pip_freeze.txt {OUT}/entorno/python.txt {C}/ 2>/dev/null; git rev-parse HEAD | tee {OUT}/commit.txt {C}/commit.txt
!sha256sum {OUT}/reintentos_gs_train.csv {OUT}/reintentos_gs_manifest.json {C}/*.xml {C}/*.log | tee /kaggle/working/f42_sha256sum.txt
!cp /kaggle/working/f42_sha256sum.txt {OUT}/
!zip -qr /kaggle/working/f42_resultados.zip {OUT} {C}
!ls -la /kaggle/working/f42_resultados.zip && unzip -l /kaggle/working/f42_resultados.zip
```
- Los dos scripts se niegan a sobrescribir. Si la celda se repite en la misma sesión, primero hay que borrar `{PNG}`, `{OUT}` y `{C}`.
- Las 908 imágenes regeneradas (`{PNG}`) **no** van en el zip: son reproducibles y ya están verificadas contra el manifiesto.

## Contenido esperado

`f42_resultados.zip` (se versiona):
```
results/f42_regimen/reintentos/
├── entorno/{pip_freeze.txt, python.txt, nproc.txt, lscpu.txt}
├── datos_verificacion.log             manifiesto OK
├── reintentos_gs.log                  avance cada 50 muestras y resumen
├── reintentos_gs_train.csv            908 filas (sample_id, u0–u4, f, k, Dv, radio, n_iter, reintentos,
│                                      usa_respaldo, n_llamadas, var_por_llamada, sha256 regenerado/manifiesto,
│                                      sha256_coincide, pixeles_coinciden, almacenada_integra, segundos)
├── reintentos_gs_manifest.json        commit, smoke = false, entorno, resumen por clase, tiempos, sha256 del CSV
├── commit.txt
└── f42_sha256sum.txt
results/f03_independencia_etiqueta/adenda_capa_C/
├── kaggle_capa_C_perlin_lorenz.log    4 casos: inventario y reproducción × (Perlin, Lorenz)
├── kaggle_capa_C_perlin_lorenz.xml
├── pip_freeze.txt, python.txt
└── commit.txt
```

## Duración estimada

- **Regeneración.** Medida en local (Windows, 1 núcleo), la ruta CPU tarda ≈ 3.8 ms por iteración en 256²: 7.5 s con `n_iter = 2000`, 17.8 s con 4500 y 26.6 s con 7000.
  - Con `n_iter` medio de ≈ 4500 (`u4` ≈ 0.5), las 908 muestras suman ≈ 4.5 h de un núcleo, más los reintentos.
  - Con los 4 vCPU de Kaggle (`--workers` = `nproc`), son ≈ 1.1 h si escala de forma lineal, y ≈ 2 h si los vCPU son hilos de 2 núcleos físicos.
- **Capa C** (Perlin y Lorenz, 18 muestras × 5 variantes cada una): ≈ 12–15 min.
- **Total:** ≈ 1.5–2.5 h. La celda 3 da la cifra con el hardware real.

## Verificación local (Claude Code, al recibir el zip)

- `reintentos_gs_manifest.json`:
  - `git_commit` es el entregado y `smoke` es `false`;
  - `csv.filas = 908` y el sha256 del CSV coincide con `csv.sha256` y con `f42_sha256sum.txt`;
  - `u_dtype` es el del pkl.
- En el CSV, `n_almacenada_integra = 908`. Se reportan `n_sha256_coincide` y `n_pixeles_coinciden`, con la lista de las que no coinciden.
- En el log de la capa C, 4 casos aprobados, 0 omitidos y 0 fallidos (`F03_STRICT=1`).
- Después, en local: se resuelven los PENDIENTES de `results/f42_regimen/reporte_e2a.md` y se corre `python -m src.fase4.regimen --resoluciones <json>` con el árbol limpio.
