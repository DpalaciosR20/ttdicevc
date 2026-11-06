# Correr F3.5 (entrenamiento final con prueba) en Kaggle

> Tarea F3.5 del plan v2 · Rol: CE · Convención general: `docs/ejecucion_kaggle.md`

F3.5 entrena los tres finalistas de F3.3 (etiqueta `seleccion-f33`) con los hiperparámetros de F3.4 y las 5 semillas de la fase `final` (42, 7, 123, 2024 y 99). **Es la única consulta de prueba de los finalistas** (protocolo §10). Ningún modelo, hiperparámetro ni decisión se cambia después.

## Compuertas (todas antes de la celda 4)

| Compuerta | Quién la cumple | Cómo se comprueba |
|---|---|---|
| Protocolo congelado, con `protocolo-v1` ancestro de HEAD | `run_grid.py --evaluate-test` | La celda 3 imprime `CON TEST (protocolo …, etiqueta protocolo-v1)` |
| Configuraciones = finalistas de `select_finalist` (F3.3 H10) | `run_grid.py` | La celda 3 imprime `Seleccion: {… 'matches': True …}` |
| Hiperparámetros = `run_grid_flags` de F3.4 | La celda 4 los lee de `results/f34_hpo/<técnica>/hpo_result.json` | Línea `Hiperparametros:` del log |
| Plantillas de F6.1 versionadas y revisadas por RP **antes** de F3.5 (protocolo §12, Enmienda 1) | SV + RP | Confirmación de SV en el chat del equipo |
| Enmienda 3 commiteada (atribución de ViT, antes de cualquier consulta de prueba) | CE + SV | Commit en la historia del commit de ejecución |

**No ejecutes la celda 4 mientras falte la confirmación de las plantillas de F6.1.**

## Celdas

**Acelerador:** GPU T4. **Internet:** activado. **Input:** `santiagosoria26ss/tt2-insumos`.

**Celda 1: repositorio en el commit indicado** (el error no muestra el token)
```python
from kaggle_secrets import UserSecretsClient
import os, subprocess

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
COMMIT = "<sha>"   # lo indica Claude Code
REPO = "/kaggle/working/ttdicevc"

def run(cmd, secreto=None):
    r = subprocess.run(cmd, capture_output=True, text=True)
    salida = r.stdout + r.stderr
    if secreto:
        salida = salida.replace(secreto, "***")
    print(salida, end="")
    if r.returncode != 0:
        raise RuntimeError(f"falló (código {r.returncode}): {cmd[0]} {cmd[1]}")

os.chdir("/kaggle/working")
run(["rm", "-rf", REPO])
run(["git", "clone", f"https://{token}@github.com/DpalaciosR20/ttdicevc.git", REPO], secreto=token)
os.chdir(REPO)
run(["git", "remote", "set-url", "origin", "https://github.com/DpalaciosR20/ttdicevc.git"])
run(["git", "fetch", "-q", "--tags"])
run(["git", "checkout", "-q", COMMIT])
run(["git", "log", "--oneline", "-1"])
run(["git", "merge-base", "--is-ancestor", "seleccion-f33", "HEAD"])   # falla si el commit no desciende de F3.3
```

**Celda 2: insumos y entorno**
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"   # docs/datos.md §2
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!mkdir -p results/f35_final/entorno && pip freeze > results/f35_final/entorno/pip_freeze.txt && nvidia-smi > results/f35_final/entorno/gpu.txt
```
`build_manifest --verify` debe decir `manifiesto OK`.

**Celda 3: comprobación sin entrenar** (segundos; no carga test)
```python
import json
FINALISTAS = ["gray_scott:efficientnet_b0:finetune", "perlin:vit_b16:frozen", "lorenz:vit_b16:frozen"]
FLAGS = {c: json.load(open(f"results/f34_hpo/{c.split(':')[0]}/hpo_result.json"))["run_grid_flags"] for c in FINALISTAS}
for cfg in FINALISTAS:
    tech = cfg.split(":")[0]
    print(cfg, "->", FLAGS[cfg])
    !python -m src.training.run_grid --out-dir /kaggle/working/f35_dry/{tech} --evaluate-test --configs {cfg} {FLAGS[cfg]} --dry-run 2>&1 | head -6
```
Por técnica deben aparecer `'matches': True`, `CON TEST (protocolo …, etiqueta protocolo-v1)`, los hiperparámetros de F3.4 y `5 corridas, 0 ya hechas, 5 pendientes`. Si algo no cuadra, **detente** y no corras la celda 4.

**Celda 4: entrenamiento final con prueba** (≈ 35–50 min de T4; usa *Save Version → Save & Run All* para conservar los modelos)
```python
for cfg in FINALISTAS:
    tech = cfg.split(":")[0]
    out = f"results/f35_final/{tech}"
    os.makedirs(out, exist_ok=True)
    !python -m src.training.run_grid --out-dir {out} --evaluate-test --configs {cfg} {FLAGS[cfg]} --save-models 2>&1 | tee {out}/run_grid.log
```

| Finalista | Hiperparámetros (F3.4) | Estimado (5 semillas; tiempos de F3.4) |
|---|---|---|
| Gray-Scott EfficientNet-B0 FT | lr 3e-5, wd 0, lote 32 (trial 0) | 5–10 min |
| Perlin ViT-B/16 congelado | lr 8.6e-3, wd 0.070, lote 16 (trial 10) | 8–12 min |
| Lorenz ViT-B/16 congelado | lr 1e-3, wd 0, lote 32 (trial 0) | 8–15 min |

A eso se suma guardar 10 modelos ViT (≈ 330 MB cada uno) y 5 EfficientNet.

**Celda 5: huellas de los modelos y empaquetado** (los modelos no van en el zip)
```python
import glob, hashlib, csv
with open("results/f35_final/models_sha256.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "bytes", "sha256"])
    for p in sorted(glob.glob("results/f35_final/*/models/*.pt")):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        w.writerow([p, os.path.getsize(p), h.hexdigest()])
!cat results/f35_final/models_sha256.csv
!cd /kaggle/working/ttdicevc && zip -qr /kaggle/working/f35_resultados.zip results/f35_final results/predictions/final -x "results/f35_final/*/models/*"
!ls -la /kaggle/working/f35_resultados.zip
```

Si se corta a medias: *Add Input → Notebook Output* de la versión anterior y agrega a esa técnica `--prev-results /kaggle/input/<output>/ttdicevc/results/f35_final/<técnica>`. Las corridas terminadas se saltan; no se repiten semillas que ya consultaron la prueba.

## Contenido esperado de `f35_resultados.zip`

```
results/f35_final/
├── entorno/{pip_freeze.txt, gpu.txt}
├── models_sha256.csv                 15 filas
└── <técnica>/                        gray_scott, perlin, lorenz
    ├── runs/*.json                   5 por técnica, con f1_test, test_probs y model_file
    ├── summary.csv, summary_by_config.csv, grid_meta.json ({"fase": "final"})
    ├── run_grid.log
    └── errors/                       vacío
results/predictions/final/*.csv       15 CSV F1.5 (val y test) con su .manifest.json
```

Los modelos (`results/f35_final/<técnica>/models/*.pt`) se quedan en la salida de la versión del notebook. De ahí se publican como el *dataset* privado `tt2-artefactos` que usa F3.7 (matriz §9), y sus huellas se registran en `meta/artefactos_manifest.csv`.

## Verificación local (Claude Code, al recibir el zip)

```bash
python -m src.predictions.validate "results/predictions/final/*.csv"
```

Además:
- 5 JSON por técnica, `errors/` vacío y `warning` vacío.
- El commit de los manifiestos es el entregado, y `seleccion-f33` y `protocolo-v1` son sus ancestros.
- `selection.matches` es `true`.
- Los hiperparámetros coinciden con `run_grid_flags`.
- El F1 de val y de test, recalculado desde `val_probs` y `test_probs`, coincide con el registrado.
- `model_file.sha256` coincide con `models_sha256.csv`.
- Por finalista: cuántas semillas terminan por el tope de 15 épocas (F3.3 H4) y la dispersión de Perlin (F3.4 H2).
