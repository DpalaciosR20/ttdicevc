# Correr F3.6 (control de permutación) en Kaggle

> Tarea F3.6 del plan v2 · Rol: CE · Convención general: `docs/ejecucion_kaggle.md` · Especificación: matriz §7

F3.6 reentrena los tres finalistas de F3.3 con las etiquetas de **entrenamiento** permutadas (`numpy.random.default_rng(99)`). Usa los hiperparámetros de F3.4 y las cinco semillas de F3.5. `y_val` y `y_test` no se tocan, y la parada temprana usa el F1-val con etiquetas reales. El F1 de prueba medio de cada técnica es el piso «visual permutado» del protocolo (§4.4), y los modelos sirven como prueba de cordura por aleatorización de datos en F4.3. Si la imagen no codifica la etiqueta por un atajo, el F1 de prueba debe caer al nivel de azar.

La permutación de la semilla 99 conserva el conteo por clase (353 de 908 depresivos), así que los pesos de clase no cambian. Coincide con la etiqueta real en el 51.3 % de las muestras de entrenamiento (52.5 % esperado al azar), con φ = −0.024.

## Compuertas

| Compuerta | Cómo se comprueba |
|---|---|
| `protocolo-v1` ancestro de HEAD y protocolo limpio | La celda 3 imprime `Fase: control_permutacion \| CON TEST (protocolo …, etiqueta protocolo-v1)` |
| Configuraciones = finalistas de `select_finalist` | La celda 3 imprime `'matches': True` |
| Hiperparámetros = `run_grid_flags` de F3.4 | Línea `Hiperparametros:` |
| Variante con sufijo `-perm99` (no choca con F3.5) | La celda 3 lista `…__px224-*-perm99__seed…` como pendientes |

## Celdas

**Acelerador:** GPU T4. **Internet:** activado. **Input:** `santiagosoria26ss/tt2-insumos`.

**Celda 1: repositorio** (igual que en `docs/correr_final_kaggle.md`)
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
run(["git", "merge-base", "--is-ancestor", "seleccion-f33", "HEAD"])
```

**Celda 2: insumos y entorno**
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!mkdir -p results/f36_permutacion/entorno && pip freeze > results/f36_permutacion/entorno/pip_freeze.txt && nvidia-smi > results/f36_permutacion/entorno/gpu.txt
```

**Celda 3: comprobación sin entrenar** (segundos; no carga test)
```python
import json
FINALISTAS = ["gray_scott:efficientnet_b0:finetune", "perlin:vit_b16:frozen", "lorenz:vit_b16:frozen"]
FLAGS = {c: json.load(open(f"results/f34_hpo/{c.split(':')[0]}/hpo_result.json"))["run_grid_flags"] for c in FINALISTAS}
for cfg in FINALISTAS:
    tech = cfg.split(":")[0]
    print(cfg, "->", FLAGS[cfg])
    !python -m src.training.run_grid --out-dir /kaggle/working/f36_dry/{tech} --evaluate-test --permute-labels 99 --configs {cfg} {FLAGS[cfg]} --dry-run 2>&1 | head -7
```
Si algo no cuadra con la tabla de compuertas, **detente**.

**Celda 4: entrenamiento, huellas y zip, en una sola celda** (≈ 25–40 min de T4; *Save Version → Save & Run All*)

Van juntos para que el empaquetado ocurra en la misma versión que entrena (F3.5, H5).
```python
for cfg in FINALISTAS:
    tech = cfg.split(":")[0]
    out = f"results/f36_permutacion/{tech}"
    os.makedirs(out, exist_ok=True)
    !python -m src.training.run_grid --out-dir {out} --evaluate-test --permute-labels 99 --configs {cfg} {FLAGS[cfg]} --save-models 2>&1 | tee {out}/run_grid.log
!python -m src.hash_models results/f36_permutacion --expected 15
!zip -qr /kaggle/working/f36_resultados.zip results/f36_permutacion results/predictions/control_permutacion -x "results/f36_permutacion/*/models/*"
!ls -la /kaggle/working/f36_resultados.zip
```

## Contenido esperado de `f36_resultados.zip`

```
results/f36_permutacion/
├── entorno/{pip_freeze.txt, gpu.txt}
├── models_sha256.csv                 15 filas
└── <técnica>/                        gray_scott, perlin, lorenz
    ├── runs/*-perm99__seed*.json     5 por técnica, con permute_labels = 99 y permutation_indices
    ├── summary.csv, summary_by_config.csv, grid_meta.json ({"fase": "control_permutacion"})
    └── run_grid.log
results/predictions/control_permutacion/*.csv   15 CSV F1.5 (val y test) con su .manifest.json
```

## Modelos

Los 15 `.pt` (≈ 3.3 GiB) se quedan en la salida de la versión. Publícalos en `tt2-artefactos`. Lo preferible es una versión nueva del *dataset* que conserve los de F3.5. Si Kaggle no permite versionarlo desde esta salida, publícalos como *dataset* aparte, `tt2-artefactos-f36`. Eso fue lo que se hizo: `palaciosroblesdiego/tt2-artefactos-f36`, registrado en la columna `dataset` del manifiesto. `src/verify_artefacts.py` busca por nombre de archivo, así que la verificación funciona en ambos casos:

```python
!python -m src.verify_artefacts --root /kaggle/input --task F3.6 --out /kaggle/working/artefactos_verificacion_f36.json
```

## Verificación local (Claude Code, al recibir el zip)

```bash
python -m src.predictions.validate "results/predictions/control_permutacion/*.csv"
```

Además, por corrida:
- fase `control_permutacion`, `permute_labels = 99` y `permutation_indices` igual a `default_rng(99).permutation(908)`;
- `y_val` y `y_test` reales (iguales a los de F3.5);
- commit, protocolo y selección;
- hiperparámetros iguales a F3.4;
- F1 recalculado desde las probabilidades;
- huellas de los modelos.
