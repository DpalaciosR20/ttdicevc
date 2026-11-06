# Correr F3.7 (mapas de atribución) en Kaggle

> Tarea F3.7 del plan v2 · Rol: CE · Convención general: `docs/ejecucion_kaggle.md` · Especificación: matriz §8 y Enmienda 3

F3.7 calcula los mapas de atribución de los **30 modelos guardados**: 15 de F3.5 (finalistas) y 15 de F3.6 (control de permutación), sobre las 114 muestras de val y las 114 de prueba. El código está en `src/interpretability/`.

| Arquitectura | Mapa principal (`map`) | Mapa secundario (`backbone_map`) |
|---|---|---|
| EfficientNet-B0 (Gray-Scott) | Grad-CAM sobre `features[-1]` | — |
| ViT-B/16 (Perlin, Lorenz) | *Rollout* ponderado por gradiente (Chefer, Gur y Wolf, 2021) | *Attention rollout* (Abnar y Zuidema, 2020): atención del *backbone* |

El objetivo es el *logit* de la clase depresiva en todas las muestras. Los mapas se reescalan a 224 × 224 y se guardan normalizados por muestra y sin normalizar (formato en el docstring de `run_maps.py`).

**Prueba y protocolo.** Los mapas son análisis explicativos (protocolo §9) y la matriz §8 los pide también para prueba. `run_maps` recalcula `p_dep` solo para comprobar que el modelo cargado reproduce las predicciones archivadas de F3.5 y F3.6 (tolerancia 1e-4). No calcula ninguna métrica ni genera predicciones nuevas, así que la consulta de prueba de §10 sigue siendo una por modelo y semilla.

## Insumos del notebook

**Acelerador:** GPU T4. **Internet:** activado. **Inputs**, los tres:
- `santiagosoria26ss/tt2-insumos` (imágenes);
- `palaciosroblesdiego/tt2-artefactos` (modelos de F3.5);
- `palaciosroblesdiego/tt2-artefactos-f36` (modelos de F3.6).

## Celdas

**Celda 1: repositorio** (como en `docs/correr_final_kaggle.md`, con `COMMIT` el que indique Claude Code y sin la línea de `seleccion-f33`).

**Celda 2: insumos y entorno**
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!python -m src.verify_artefacts --root /kaggle/input --out results/f37_mapas/modelos_verificacion.json
!mkdir -p results/f37_mapas/entorno && pip freeze > results/f37_mapas/entorno/pip_freeze.txt && nvidia-smi > results/f37_mapas/entorno/gpu.txt
```
`verify_artefacts` debe decir `30/30 modelos coinciden con el manifiesto -> OK`.

**Celda 3: prueba de humo y estimación** (≈ 1–2 min). Corre 8 muestras por split en Perlin (ViT) y Gray-Scott (EfficientNet), en una carpeta aparte.
```python
!python -m src.interpretability.run_maps --models-root /kaggle/input --out-dir /kaggle/working/f37_humo --limit 8 --only perlin gray_scott 2>&1 | tail -25
```
Cada línea debe mostrar `Δp_dep` ≤ 1e-4 y `Δlogit` del orden de 1e-5 o menos. Para estimar la corrida completa, multiplica el tiempo por corrida (`…s`) por 228/16 ≈ 14 y súmalo sobre las 30 corridas. Si la estimación pasa de 2 h, detente y avísale a Claude Code.

**Celda 4: mapas, manifiesto y zip, en una sola celda** (estimado: 20–40 min de T4; *Save Version → Save & Run All*)
```python
!python -m src.interpretability.run_maps --models-root /kaggle/input --out-dir results/f37_mapas 2>&1 | tee results/f37_mapas/run_maps.log
!du -sh results/f37_mapas/maps
!zip -qr /kaggle/working/f37_resultados.zip results/f37_mapas -x "results/f37_mapas/maps/*"
!ls -la /kaggle/working/f37_resultados.zip
```
Si se corta, la misma celda reanuda: salta las corridas cuyo `.npz` ya coincide con su registro.

## Contenido esperado de `f37_resultados.zip`

```
results/f37_mapas/
├── entorno/{pip_freeze.txt, gpu.txt}
├── modelos_verificacion.json         30/30
├── maps_meta.json                    {"task": "F3.7", "smoke": false, "limit": null}
├── maps_manifest.csv                 30 filas: corrida, n = 228, métodos, npz, bytes, sha256
├── records/*.json                    30: Δp_dep por split, Δlogit, huella del modelo, entorno
└── run_maps.log
```

Los 30 `.npz` (`results/f37_mapas/maps/`; estimado 1–2.5 GiB) se quedan en la salida de la versión. Publícalos como *dataset* privado `tt2-mapas` (*New Dataset* desde la salida). Después se verifican como los modelos:

```python
!python -m src.verify_artefacts --root /kaggle/input --task F3.7 --out /kaggle/working/mapas_verificacion.json
```

## Verificación local (Claude Code, al recibir el zip)

- 30 registros con `n = 228`, Δp_dep ≤ 1e-4 y la huella del modelo igual a la del manifiesto;
- el commit y la huella de la matriz son los entregados;
- `maps_manifest.csv` se incorpora a `meta/artefactos_manifest.csv` como filas `task = F3.7`.
