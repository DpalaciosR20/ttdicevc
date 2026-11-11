# Correr F4.3 (cascada de cordura y ablación espectral) en Kaggle

> Tarea F4.3 del plan v2 · Rol: SV · Convención general: `docs/ejecucion_kaggle.md` · Especificación: `docs/plan_fase4.md`, F4.3

En una sola sesión de Kaggle con GPU se hacen las dos partes de F4.3 que requieren inferencia (`src/fase4/atribucion_gpu.py`):

1. **Cascada de cordura por parámetros** con las 5 semillas de cada finalista de F3.5:
   - EfficientNet-B0: cabeza → `features[8]` … `features[0]`;
   - ViT-B/16: cabeza → `encoder.layers` 11 … 0 → *embeddings*.

   Se aplica `reset_parameters()` con `torch.manual_seed(2026)`, de forma acumulativa y sobre una copia en CPU. En cada paso se recalculan los mapas de F3.7 y se comparan con los del modelo sin aleatorizar.
2. **Ablación espectral.** Paso-bajo gaussiano con σ ∈ {1, 2, 4, 8} px sobre la imagen de 256 px, antes del preprocesamiento de F1.4, y paso-alto complementario. Se mide el ΔF1-val medio de las 5 semillas.

**G2 (sin inferencia nueva sobre prueba).**
- Todo es en `val`. El script se niega a cargar cualquier otro split: `filas_val`, `nombres_val` y `leer_imagenes_val` se detienen ante una fila o un nombre de archivo que no sea de val.
- Los modelos de F3.6 no se usan aquí.

La parte local de F4.3 (agregación, constantes, perfiles, terciles, permutación y Holm, cordura por datos, cosenos de cabezas, Gray-Scott real frente a permutado) **no** se corre aquí. Se corre en local con `src/fase4/atribucion_local.py` sobre los `.npz` de F3.7 y los `.pt` de la copia de Drive.

## Requisito previo

`results/f43_atribucion/resoluciones.json` está versionado en el commit que se clona: el script lo verifica con `git ls-files` y `git diff`. Sin él no corre (PENDIENTES de `results/f43_atribucion/reporte_e3a.md`).

## Insumos del notebook

- **Acelerador:** GPU T4.
- **Internet:** activado, para `git clone` y `pip install`. No se descargan pesos de ImageNet: los modelos se construyen con `pretrained=False` y se cargan desde los `.pt`.
- **Inputs:**
  - `santiagosoria26ss/tt2-insumos`: imágenes y CSV;
  - `palaciosroblesdiego/tt2-artefactos`: modelos de F3.5;
  - `palaciosroblesdiego/tt2-mapas`: `.npz` de F3.7. Sirven para comprobar que el mapa sin aleatorizar reproduce el de F3.7; el script usa solo la copia de `ttdicevc/results/f37_mapas/maps/` y rechaza `f37_humo/`.

## Celdas

**Celda 1: repositorio.** Igual que en `docs/correr_final_kaggle.md`, con el `COMMIT` que indique Claude Code. La última línea se cambia por la compuerta de la Fase 4:
```python
run(["git", "merge-base", "--is-ancestor", "enmienda-5", "HEAD"])   # falla si el commit no desciende de la Enmienda 5
```

**Celda 2: insumos y entorno.** `TT_DATA_ROOT` es la ruta anidada de `docs/datos.md` §2.
```python
import os
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!python -m src.verify_artefacts --root /kaggle/input --task F3.5 --out /tmp/f43_modelos.json
!python -m src.verify_artefacts --root /kaggle/input --task F3.7 --out /tmp/f43_mapas.json
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
Debe decir:
- `manifiesto OK`;
- `15/15 modelos coinciden con el manifiesto -> OK` en F3.5;
- `30/30 … -> OK` en F3.7. Las copias de `f37_humo/` aparecen como descartadas, y eso es lo esperado (F3.7, H4).

**Celda 3: prueba de humo y estimación** (≈ 5–10 min, sobre todo por las huellas y la carga de los 10 ViT). Usa 8 muestras de val y escribe en `/tmp`, fuera de la salida del notebook (lección de F3.7).
```python
!python -m src.fase4.atribucion_gpu --models-root /kaggle/input --maps-root /kaggle/input \
  --resoluciones results/f43_atribucion/resoluciones.json --out-dir /tmp/f43_humo --limit 8 2>&1 | tail -25
```
- Debe imprimir una línea por (técnica, semilla) y, al final, `estimacion_114` por técnica y su `total`, en segundos.
  - La estimación escala a 114 muestras el tiempo de cómputo de cascada y ablación, y suma la carga de los modelos.
- No debe aparecer `DETENIDO`. Las comprobaciones de `p_dep` (≤ 1e-4 frente a `val_probs`) y del mapa frente al `.npz` de F3.7 (≤ 1e-2 en el mapa normalizado) se hacen también en la prueba de humo.
  - Si se detiene por alguna de ellas, avísale a Claude Code.
- Si `total` pasa de 2.5 h, detente y avísale a Claude Code.

**Celda 4: verificación, cascada, ablación, huellas y zip, en una sola celda.** Se ejecuta en la versión guardada (*Save Version → Save & Run All*), no en la sesión interactiva (lección de F3.5, H5).

Duración estimada: 35–70 min de T4 (ver «Duración estimada»). El valor de la celda 3 manda.
```python
import os
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
OUT = "results/f43_atribucion/gpu"
!mkdir -p {OUT}/entorno
!python -m src.build_manifest --verify 2>&1 | tee {OUT}/datos_verificacion.log
!python -m src.verify_artefacts --root /kaggle/input --task F3.5 --out {OUT}/modelos_verificacion.json
!python -m src.verify_artefacts --root /kaggle/input --task F3.7 --out {OUT}/mapas_verificacion.json
!pip freeze > {OUT}/entorno/pip_freeze.txt && nvidia-smi > {OUT}/entorno/gpu.txt && python --version > {OUT}/entorno/python.txt
!python -m src.fase4.atribucion_gpu --models-root /kaggle/input --maps-root /kaggle/input \
  --resoluciones results/f43_atribucion/resoluciones.json --out-dir {OUT} 2>&1 | tee {OUT}/atribucion_gpu.log
!git rev-parse HEAD | tee {OUT}/commit.txt
!cd {OUT} && sha256sum cascada.json cascada_similitudes.npz cascada_medias.npz ablacion.json ablacion.csv atribucion_gpu_meta.json | tee /kaggle/working/f43_sha256sum.txt
!cp /kaggle/working/f43_sha256sum.txt {OUT}/
!zip -qr /kaggle/working/f43_resultados.zip {OUT}
!ls -la /kaggle/working/f43_resultados.zip && unzip -l /kaggle/working/f43_resultados.zip
```
El script se niega a sobrescribir: si la celda se repite en la misma sesión, primero hay que borrar `{OUT}`.

## Contenido esperado

`f43_resultados.zip` (se versiona; unos 5–10 MB):
```
results/f43_atribucion/gpu/
├── entorno/{pip_freeze.txt, gpu.txt, python.txt}
├── datos_verificacion.log          manifiesto OK
├── modelos_verificacion.json       15/15 (F3.5)
├── mapas_verificacion.json         30/30 (F3.7)
├── atribucion_gpu.log
├── atribucion_gpu_meta.json        commit, smoke = false, resoluciones (sha256), modelos (sha256, Δp_dep, comparación con F3.7), huella de las imágenes de val, tiempos
├── cascada.json                    por técnica y paso: similitud por semilla (mediana, n usado, n NaN), referencia de 10 pares y clasificación «cambia / no cambia»
├── cascada_similitudes.npz         similitud por muestra: '<técnica>|<semilla>|<paso>|<mapa>|<medida>'
├── cascada_medias.npz              mapas medios normalizados por clase (56 × 56) en cada paso
├── ablacion.json                   F1-val y ΔF1 por condición, por semilla y media
├── ablacion.csv                    una fila por (técnica, semilla, condición)
├── commit.txt
└── f43_sha256sum.txt
```

## Duración estimada

Base: los tiempos de F3.7 en T4 (`results/f37_mapas/records/*.json`, `seconds_compute`). En 228 muestras son 12.6 s en ViT y 1.7 s en EfficientNet; es decir, una pasada de mapas sobre las 114 de val tarda ≈ 6.3 s en ViT y ≈ 0.85 s en EfficientNet. Las similitudes por paso se midieron en local, en CPU: Spearman a 56 × 56, 1.13 s; SSIM a 224, 0.54 s; SSIM a 56, 0.07 s, cada una sobre 114 pares. En la CPU de Kaggle se supone el doble.

| Parte | Cálculo | Estimado |
|---|---|---|
| Cascada ViT (Perlin y Lorenz) | 15 pasadas × 6.3 s + 14 pasos × 2 mapas × ≈ 3.5 s, por semilla; × 10 | ≈ 32 min |
| Cascada EfficientNet (Gray-Scott) | 11 × 0.85 s + 10 × ≈ 3.5 s, por semilla; × 5 | ≈ 4 min |
| Ablación | 9 condiciones de 114 imágenes por modelo; × 15 | ≈ 5 min |
| Referencias, carga y huellas de modelos y `.npz` | — | ≈ 3–5 min |
| **Total** | | **≈ 45 min (rango 35–70 min)** |

Es más de 20 min y requiere GPU, así que va a Kaggle (`CLAUDE.md`). La celda 3 da la cifra con el hardware real.

## Verificación local (Claude Code, al recibir el zip)

- `atribucion_gpu_meta.json`:
  - `git_commit` es el entregado y `smoke` es `false`;
  - `split` es `val`;
  - el sha256 de las resoluciones es el del archivo versionado;
  - en las 15 semillas, `p_dep_max_abs_diff` ≤ 1e-4, `vs_npz_f37.*.max_abs_diff_normalizado` ≤ 1e-2 y `ablacion_p_dep_original_max_abs_diff` ≤ 1e-4.
- `modelos_verificacion.json` dice 15/15 y `mapas_verificacion.json` dice 30/30.
- El sha256 de cada salida coincide con `f43_sha256sum.txt`.
- `cascada.json` tiene, por técnica, los pasos en el orden de `docs/plan_fase4.md` y 10 pares de referencia en cada medida.
- `ablacion.json` tiene 9 condiciones por técnica, y `delta_f1_media = 0` en `original`.
- Después, en local: `python -m src.fase4.atribucion_local --root <copia de Drive> --resoluciones results/f43_atribucion/resoluciones.json --cascada results/f43_atribucion/gpu/cascada.json`, con el árbol limpio.
