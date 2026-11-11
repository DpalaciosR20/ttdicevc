# Correr la parte local de F4.3 en Kaggle (CPU)

> Tarea F4.3 del plan v2 · Rol: SV · Convención general: `docs/ejecucion_kaggle.md` · Parte GPU: `docs/correr_f43_kaggle.md`

La parte «local» de F4.3 (`src/fase4/atribucion_local.py`) no hace inferencia. Comprende:
- agregación por clase y fracción de constantes;
- perfil radial de Gray-Scott y terciles de u4 en Lorenz;
- permutación de etiquetas y Holm;
- cordura por aleatorización de datos;
- cosenos de las cabezas ViT;
- Gray-Scott real frente a permutado;
- lectura de los ViT con `cascada.json`.

**Por qué va a Kaggle (P20).** La prueba acotada en el equipo local (`results/f43_atribucion/estimacion_local/estimacion.json`) estimó ≈ 43 min. Casi todo es lectura de los 7.7 GiB desde Drive, a ≈ 3 MiB/s. El cálculo es ≈ 4 min. Pasa de 20 min, así que se corre en Kaggle con CPU y `--root /kaggle/input`, como prevé la resolución P20.

**G1/G2.** El script no hace inferencia: `p_dep` sale de los `.npz` archivados de F3.7. Los mapas de prueba se usan solo como réplica descriptiva. La prueba de permutación y la cordura por datos que deciden son en val.

## Insumos del notebook

- **Acelerador:** ninguno (CPU).
- **Internet:** activado, para `git clone` y `pip install`.
- **Inputs:**
  - `santiagosoria26ss/tt2-insumos`: `umap_params.pkl` y CSV;
  - `palaciosroblesdiego/tt2-artefactos`: modelos de F3.5;
  - `palaciosroblesdiego/tt2-mapas`: `.npz` de F3.7.
  - En la sesión de la parte GPU, `tt2-artefactos` ya traía los `.pt` de F3.6, que aparecieron como `extra`. Si la celda 2 no da 15/15 en F3.6, agrega también `palaciosroblesdiego/tt2-artefactos-f36`.
  - Si da `ambiguous` no vacío, quita ese input: el script se detiene ante dos copias con la misma ruta.

## Celdas

**Celda 1: repositorio.** La de `docs/correr_final_kaggle.md`, con el `COMMIT` que indique Claude Code y la compuerta de la Fase 4 como última línea:
```python
run(["git", "merge-base", "--is-ancestor", "enmienda-5", "HEAD"])
```

**Celda 2: insumos y huellas.**
```python
import os
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!python -m src.verify_artefacts --root /kaggle/input --task F3.5 --out /tmp/f43l_modelos_f35.json
!python -m src.verify_artefacts --root /kaggle/input --task F3.6 --out /tmp/f43l_modelos_f36.json
!python -m src.verify_artefacts --root /kaggle/input --task F3.7 --out /tmp/f43l_mapas.json
!git status --porcelain
```
Debe decir:
- `manifiesto OK`;
- 15/15 en F3.5, 15/15 en F3.6 y 30/30 en F3.7, sin `ambiguous`;
- `git status --porcelain` vacío. El script exige el árbol limpio, por eso todo lo de esta celda va a `/tmp`.

**Celda 3: corrida, huellas y zip, en una sola celda.** Se ejecuta en la versión guardada (*Save Version → Save & Run All*).

Duración estimada: 10–25 min de CPU (ver abajo).

El log y las huellas se escriben primero en `/tmp`: un archivo nuevo dentro del repo antes de la corrida ensuciaría el árbol y la compuerta se detendría.
```python
import os
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
OUT = "results/f43_atribucion/local"
!python -m src.build_manifest --verify 2>&1 | tee /tmp/datos_verificacion.log
!python -m src.verify_artefacts --root /kaggle/input --task F3.5 --out /tmp/modelos_f35_verificacion.json
!python -m src.verify_artefacts --root /kaggle/input --task F3.6 --out /tmp/modelos_f36_verificacion.json
!python -m src.verify_artefacts --root /kaggle/input --task F3.7 --out /tmp/mapas_verificacion.json
!mkdir -p /tmp/entorno && pip freeze > /tmp/entorno/pip_freeze.txt && python --version > /tmp/entorno/python.txt && nproc > /tmp/entorno/nproc.txt
!python -m src.fase4.atribucion_local --root /kaggle/input \
  --resoluciones results/f43_atribucion/resoluciones.json \
  --cascada results/f43_atribucion/gpu/cascada.json 2>&1 | tee /tmp/atribucion_local.log
!cp -r /tmp/entorno /tmp/datos_verificacion.log /tmp/modelos_f35_verificacion.json /tmp/modelos_f36_verificacion.json /tmp/mapas_verificacion.json /tmp/atribucion_local.log {OUT}/
!git rev-parse HEAD | tee {OUT}/commit.txt
!cd {OUT} && sha256sum resultados.json medias_clase.npz similitudes_por_muestra.csv | tee f43_local_sha256sum.txt
!zip -qr /kaggle/working/f43_local_resultados.zip {OUT}
!ls -la /kaggle/working/f43_local_resultados.zip && unzip -l /kaggle/working/f43_local_resultados.zip
```
- Si aparece `DETENIDO`, detente y avísale a Claude Code.
- El script se niega a sobrescribir. Si la celda se repite en la misma sesión, primero borra `{OUT}` y `/tmp/*.log`.

## Contenido esperado

`f43_local_resultados.zip` (unos 5–10 MB):
```
results/f43_atribucion/local/
├── entorno/{pip_freeze.txt, python.txt, nproc.txt}
├── datos_verificacion.log            manifiesto OK
├── modelos_f35_verificacion.json     15/15
├── modelos_f36_verificacion.json     15/15
├── mapas_verificacion.json           30/30
├── atribucion_local.log              holm y lectura_vit al final
├── resultados.json                   meta (commit, resoluciones y cascada con sha256, artefactos verificados, segundos) y resultados
├── medias_clase.npz                  mapas medios por clase (56 × 56): real, predicha, sin constantes, backbone, terciles de u4
├── similitudes_por_muestra.csv       una fila por (técnica, mapa, split, medida, comparación, semillas, muestra)
├── commit.txt
└── f43_local_sha256sum.txt
```

## Duración estimada

Base: `results/f43_atribucion/estimacion_local/estimacion.json`, medido en el equipo local.

| Parte | Local (medido y extrapolado) | Kaggle CPU (supuesto) |
|---|---|---|
| Recorrido de `--root` | 21 s | < 1 min |
| Lectura y sha256 de 30 `.npz` (1.3 GiB) y 20 `.pt` ViT (6.4 GiB) | ≈ 39 min (≈ 3 MiB/s desde Drive) | ≈ 2–5 min (disco local de Kaggle) |
| Cálculo (similitudes, permutación 5,000, agregación) | ≈ 4 min | ≈ 8 min (el doble) |
| Huellas de la celda 3 (`verify_artefacts`, 60 archivos) | — | ≈ 2–5 min |
| **Total** | **≈ 43 min** | **≈ 15 min (10–25)** |

Memoria: los 30 `.npz` cargados ocupan ≈ 7 GB, dentro de los 30 GB de la CPU de Kaggle.

## Verificación local (Claude Code, al recibir el zip)

- `commit.txt` y `resultados.json → meta.git_commit` son el commit entregado.
- `meta.resoluciones.sha256` es el del archivo versionado. `meta.cascada.sha256` es el de `results/f43_atribucion/gpu/cascada.json` (`cd1e61b1…`).
- `meta.artefactos_verificados` tiene 50 entradas: 30 `.npz` y 20 `.pt`, todas bajo `ttdicevc/results/f37_mapas/maps/`, `f35_final/` o `f36_permutacion/`, y ninguna en `f37_humo/`. Sus sha256 coinciden con `meta/artefactos_manifest.csv`.
- Las huellas son 15/15, 15/15 y 30/30, y el sha256 de las tres salidas coincide con `f43_local_sha256sum.txt`.
- `resultados.holm.familia` tiene m = 3. `lectura_vit` tiene Perlin y Lorenz, con `cabeza_cascada` no nulo.
