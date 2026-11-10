# Correr F4.1 (extracción de características) en Kaggle

> Tarea F4.1 del plan v2 (y F4.7, que reutiliza sus características) · Rol: SV · Convención general: `docs/ejecucion_kaggle.md` · Especificación: `docs/plan_fase4.md`, F4.1 y F4.7

La extracción calcula la **penúltima capa** del *backbone* de cada finalista para las 1,136 imágenes de cada técnica. El código está en `src/fase4/caracteristicas.py`. El ajuste de la sonda Ridge, las LogReg y el Mantel se hacen después en local, con `src/fase4/recuperabilidad.py` y `src/fase4/geometria.py`.

**G2 (sin inferencia nueva sobre prueba).**
- La cabeza de clasificación se reemplaza por `nn.Identity` antes de la pasada.
- No se calculan logits, `p_dep` ni predicciones de clase de ningún finalista ni control, en ningún split.
- Solo se extraen características: 768 en ViT y 1280 en EfficientNet.

| Técnica | *Backbones* | Archivos |
|---|---|---|
| Perlin | ViT-B/16 con pesos ImageNet (`IMAGENET1K_V1`); se comprueba que el de cada finalista congelado (5 semillas de F3.5) es idéntico tensor por tensor | `perlin__vit_b16_imagenet.npz` |
| Lorenz | Igual que Perlin | `lorenz__vit_b16_imagenet.npz` |
| Gray-Scott | EfficientNet-B0 ajustado de las 5 semillas del finalista (la 42 es la principal) y EfficientNet-B0 ImageNet sin ajustar (sensibilidad) | `gray_scott__efficientnet_b0_finetune_seed{42,7,123,2024,99}.npz`, `gray_scott__efficientnet_b0_imagenet.npz` |

Preprocesamiento: el de inferencia de F1.4 que registra el JSON de cada corrida (224 px, bilineal con *antialias*), **sin volteo ni ningún aumento**.

## Insumos del notebook

- **Acelerador:** GPU T4.
- **Internet:** activado. Hace falta para descargar los pesos ImageNet de torchvision; su sha256 queda registrado.
- **Inputs:**
  - `santiagosoria26ss/tt2-insumos`: imágenes, `umap_params.pkl` y CSV;
  - `palaciosroblesdiego/tt2-artefactos`: modelos de F3.5;
  - `palaciosroblesdiego/tt2-artefactos-f36`: modelos de F3.6. La extracción no los usa; si se agregan, no interfieren, porque los modelos se buscan por nombre y ruta.

## Celdas

**Celda 1: repositorio.** Igual que en `docs/correr_final_kaggle.md`, con el `COMMIT` que indique Claude Code. Cambia la última línea por la compuerta de la Fase 4:
```python
run(["git", "merge-base", "--is-ancestor", "enmienda-5", "HEAD"])   # falla si el commit no desciende de la Enmienda 5
```

**Celda 2: insumos y entorno.** `TT_DATA_ROOT` es la ruta anidada de `docs/datos.md` §2.
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!python -m src.verify_artefacts --root /kaggle/input --task F3.5 --out /tmp/f41_modelos_verificacion.json
```
Debe decir `manifiesto OK` y `15/15 modelos coinciden con el manifiesto -> OK`.

**Celda 3: prueba de humo y estimación** (≈ 2–4 min, sobre todo por las huellas de los modelos ViT). Usa 16 imágenes por técnica y escribe en `/tmp`, fuera de la salida del notebook (lección de F3.7).
```python
!python -m src.fase4.caracteristicas --models-root /kaggle/input --features-dir /tmp/f41_humo/npz --out-dir /tmp/f41_humo/meta --limit 16 2>&1 | tail -15
```
- Debe imprimir 8 líneas, una por (técnica, *backbone*), con formas `(16, 768)` o `(16, 1280)`.
- Para estimar la corrida completa, multiplica por 1136/16 ≈ 71 el tiempo por línea (`… s`) y súmalo. Ese es el tiempo de la extracción; las huellas no escalan con el número de imágenes.
- Si la estimación pasa de 30 min, detente y avísale a Claude Code.

**Celda 4: verificación, extracción, huellas y zip, en una sola celda.** Estimado: 5–15 min de T4. Debe ejecutarse en la versión guardada (*Save Version → Save & Run All*), no en la sesión interactiva (lección de F3.5, H5).
```python
import os
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
OUT = "results/f41_recuperabilidad/caracteristicas"
NPZ = "/kaggle/working/f41_caracteristicas"
!mkdir -p {OUT}/entorno
!python -m src.build_manifest --verify 2>&1 | tee {OUT}/datos_verificacion.log
!python -m src.verify_artefacts --root /kaggle/input --task F3.5 --out {OUT}/modelos_verificacion.json
!pip freeze > {OUT}/entorno/pip_freeze.txt && nvidia-smi > {OUT}/entorno/gpu.txt && python --version > {OUT}/entorno/python.txt
!python -m src.fase4.caracteristicas --models-root /kaggle/input --features-dir {NPZ} --out-dir {OUT} 2>&1 | tee {OUT}/caracteristicas.log
!cd {NPZ} && sha256sum *.npz | tee /kaggle/working/npz_sha256sum.txt
!cp /kaggle/working/npz_sha256sum.txt {OUT}/npz_sha256sum.txt
!zip -qr /kaggle/working/f41_resultados.zip {OUT}
!cd /kaggle/working && zip -qr f41_caracteristicas_npz.zip f41_caracteristicas
!ls -la /kaggle/working/f41_resultados.zip /kaggle/working/f41_caracteristicas_npz.zip
```
El script se niega a sobrescribir: si la celda se repite en la misma sesión, primero hay que borrar `{NPZ}` y `{OUT}`.

## Contenido esperado

`f41_resultados.zip` (se versiona):
```
results/f41_recuperabilidad/caracteristicas/
├── entorno/{pip_freeze.txt, gpu.txt, python.txt}
├── datos_verificacion.log            manifiesto OK
├── modelos_verificacion.json         15/15
├── caracteristicas_manifest.csv      8 filas: técnica, backbone, archivo, n = 1136, d, sha256, modelo y su sha256, dispositivo
├── caracteristicas_meta.json         identidad ViT/ImageNet por semilla, dir_digest de imágenes, pesos ImageNet (sha256), commit
├── caracteristicas.log
└── npz_sha256sum.txt                 sha256sum de los 8 .npz (debe coincidir con el manifiesto)
```

`f41_caracteristicas_npz.zip` (NO se versiona; unos 30–40 MB) contiene los 8 `.npz`. Va a:
- Drive: `MiUnidad/tt2-artefactos/f41_caracteristicas/`;
- un *dataset* privado de Kaggle (sugerencia: `tt2-f41-caracteristicas`), creado desde la salida de la versión.

## Verificación local (Claude Code, al recibir los zips)

- `caracteristicas_meta.json`:
  - `git_commit` es el entregado y `smoke` es `false`;
  - en Perlin y Lorenz, `backbone_vs_imagenet.identico = true` en las 5 semillas;
  - en las tres técnicas, `dir_digest` es igual al de `meta/data_manifest.csv`.
- `modelos_verificacion.json` dice 15/15.
- El sha256 de cada `.npz` descargado coincide con `caracteristicas_manifest.csv` y con `npz_sha256sum.txt`.
- Cada `.npz` tiene `features` float32 de 1136 × d, sin NaN, con `sample_id` y `split` en el orden de `meta/sample_index.csv`.
- Después: `python -m src.fase4.recuperabilidad --val …`, el commit de val, y solo entonces `--prueba` (G1).
