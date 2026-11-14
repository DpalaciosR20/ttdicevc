# Correr FX.1 (análisis exploratorio de píxeles) en Kaggle

> Análisis exploratorio FX.1 · Plan: `docs/plan_exploratorio_pixeles.md` (etiqueta `plan-pixeles`) · Rol: CE (ejecución), SV (revisión) · Convención general: `docs/ejecucion_kaggle.md`

| Etapa | Partes | Notebook | Código |
|---|---|---|---|
| X2 | B (traslape) y D (diferencias por clase) | Kaggle **CPU** | `src/fase_x/pixeles_local.py` |
| X3 | C1 (suavizado dirigido) y C2 (oclusión) | Kaggle **GPU T4** | `src/fase_x/pixeles_gpu.py` |

**Antes de cualquier corrida:**
1. SV resuelve los PENDIENTES en `results/fx_pixeles/resoluciones.json` (claves en `null`; opciones en `_pendientes` y en `results/fx_pixeles/reporte_x1.md`). Con alguno en `null`, los scripts se detienen.
2. Ese cambio se commitea y se empuja. El `COMMIT` de las celdas es ese commit o uno posterior, y debe descender de `plan-pixeles`.

**Reglas del plan que hacen cumplir los scripts:**
- **Prueba:** no se lee. Ante una fila de prueba, el script aborta.
- **Compuerta:**
  - `plan-pixeles` es ancestro de HEAD;
  - `docs/plan_exploratorio_pixeles.md` no cambió frente a la etiqueta;
  - `resoluciones.json` está versionado;
  - el árbol está limpio;
  - el `--out-dir` está vacío.
- **Reproducción previa (C):** cada modelo debe reproducir su `p_dep` de val (≤ 1e-4) antes de modificar ninguna imagen.
- **Una sola ejecución con datos reales,** con la salida completa redirigida a un archivo, sin tuberías ni filtros (regla 2.9). La prueba acotada (`--limit`) escribe fuera de `results/fx_pixeles/`; el script lo exige.

Por la regla del árbol limpio, **nada se escribe dentro del repositorio antes de la corrida**: el entorno y los logs van a `/kaggle/working/` y se copian a la carpeta de salida al terminar.

## Celda 1 (ambos notebooks): repositorio

La de `docs/correr_final_kaggle.md`, con el `COMMIT` que indique Claude Code y esta última línea:
```python
run(["git", "merge-base", "--is-ancestor", "plan-pixeles", "HEAD"])
```

## X2: B y D (Kaggle CPU)

**Acelerador:** ninguno. **Internet:** activado.

**Inputs:**
- `santiagosoria26ss/tt2-insumos`: imágenes;
- `palaciosroblesdiego/tt2-mapas`: mapas de F3.7, reales y permutados.

**Celda 2: insumos y entorno**
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!python -m src.verify_artefacts --root /kaggle/input --task F3.7 --out /kaggle/working/mapas_verificacion.json > /kaggle/working/verificacion.log 2>&1
!cat /kaggle/working/verificacion.log
!mkdir -p /kaggle/working/entorno && pip freeze > /kaggle/working/entorno/pip_freeze.txt
!git status --porcelain
```
- `verify_artefacts` debe terminar con `30/30 … -> OK`.
- `git status --porcelain` no debe imprimir nada.

**Celda 3: prueba acotada** (12 muestras de val y 12 de train; fuera del repositorio)
```python
!python -m src.fase_x.pixeles_local --root /kaggle/input --out-dir /kaggle/working/fx_local_humo --limit 12 > /kaggle/working/fx_local_humo.log 2>&1
!cat /kaggle/working/fx_local_humo.log
```
- La última línea con `estimación` da los segundos escalados a 114 y 908.
- La estimación de X1 es ≈ 15–25 min en total. Casi todo es la prueba por clústeres: unos 46 s por mapa con 1,000 permutaciones en el equipo local, sobre 12 mapas.
- Si la estimación pasa de 2 h, detente y avisa a SV.

**Celda 4: corrida única** (*Save Version → Save & Run All*)
```python
!python -m src.fase_x.pixeles_local --root /kaggle/input --out-dir results/fx_pixeles/local > /kaggle/working/fx_local.log 2>&1
!cp /kaggle/working/fx_local.log results/fx_pixeles/local/run.log
!cp -r /kaggle/working/entorno /kaggle/working/mapas_verificacion.json results/fx_pixeles/local/
!python -m src.fase_x.manifiesto results/fx_pixeles/local
!cat results/fx_pixeles/local/run.log
!zip -qr /kaggle/working/fx_local.zip results/fx_pixeles/local
```

**Contenido esperado de `fx_local.zip`:**

```
results/fx_pixeles/local/
├── resultados.json      B (B1–B4, H-B) y D (clústeres, Mann–Whitney, H-D), con meta: commit, sha256 de
│                        resoluciones, plan, imágenes y .npz, entorno y segundos
├── b_por_muestra.csv    Spearman y ΔDice por muestra
├── arreglos.npz         atribución y pendiente por parche; mapas t y nulas de D
├── run.log, mapas_verificacion.json, entorno/pip_freeze.txt
└── manifiesto.csv
```

## X3: C1 y C2 (Kaggle GPU T4)

**Acelerador:** GPU T4. **Internet:** activado.

**Inputs:**
- `santiagosoria26ss/tt2-insumos`;
- `palaciosroblesdiego/tt2-artefactos`: modelos de F3.5;
- `palaciosroblesdiego/tt2-artefactos-f36`: modelos de F3.6.

**Celda 2: insumos y entorno**
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!python -m src.verify_artefacts --root /kaggle/input --task F3.5 --out /kaggle/working/modelos_f35.json > /kaggle/working/verif35.log 2>&1
!python -m src.verify_artefacts --root /kaggle/input --task F3.6 --out /kaggle/working/modelos_f36.json > /kaggle/working/verif36.log 2>&1
!cat /kaggle/working/verif35.log /kaggle/working/verif36.log
!mkdir -p /kaggle/working/entorno && pip freeze > /kaggle/working/entorno/pip_freeze.txt && nvidia-smi > /kaggle/working/entorno/gpu.txt
!git status --porcelain
```
- Las dos verificaciones deben dar `15/15 … -> OK`.
- `git status --porcelain` no debe imprimir nada.

**Celda 3: prueba acotada con 8 imágenes de val y estimación** (§5.C3)
```python
!python -m src.fase_x.pixeles_gpu --models-root /kaggle/input --out-dir /kaggle/working/fx_gpu_humo --limit 8 > /kaggle/working/fx_gpu_humo.log 2>&1
!cat /kaggle/working/fx_gpu_humo.log
```
- La línea `estimación para 114 imágenes` compara la extrapolación con el límite de 3 h del plan.
- **Si dice `SUPERA`, detente y avisa a SV.** La reducción autorizada (`--k-sorteos 10`) solo se usa si SV la confirma.
- La estimación de X1, sin GPU disponible para medirla, es de ≈ 35–60 min (sección «Costo» de `reporte_x1.md`).

**Celda 4: corrida única** (*Save Version → Save & Run All*)
```python
!python -m src.fase_x.pixeles_gpu --models-root /kaggle/input --out-dir results/fx_pixeles/gpu > /kaggle/working/fx_gpu.log 2>&1
!cp /kaggle/working/fx_gpu.log results/fx_pixeles/gpu/run.log
!cp -r /kaggle/working/entorno /kaggle/working/modelos_f35.json /kaggle/working/modelos_f36.json results/fx_pixeles/gpu/
!python -m src.fase_x.manifiesto results/fx_pixeles/gpu
!cat results/fx_pixeles/gpu/run.log
!zip -qr /kaggle/working/fx_gpu.zip results/fx_pixeles/gpu
```

**Contenido esperado de `fx_gpu.zip`:**

```
results/fx_pixeles/gpu/
├── resultados.json      C1 (S, Wilcoxon, secundarios, ΔF1, H-C) y C2 (cordura, Spearman con la pendiente,
│                        lectura), con meta: commit, sha256, reproducción de p_dep, entorno y segundos
├── arreglos.npz         p_dep por semilla y condición, desplazamientos sorteados, S; mapas O por semilla
├── run.log, modelos_f35.json, modelos_f36.json, entorno/{pip_freeze.txt, gpu.txt}
└── manifiesto.csv
```

## Verificación local (Claude Code, etapa X4)

- La compuerta: commit, sha256 del plan y de `resoluciones.json`.
- La reproducción de `p_dep` (≤ 1e-4 en los 30 modelos).
- `manifiesto.csv` contra los archivos.
- El recálculo de los estadísticos desde `arreglos.npz`.
- Que ningún archivo contenga filas de prueba.
