# Correr la HPO de los finalistas en Kaggle (CE, plan v2 — F3.4)

Script: `src/training/run_hpo.py`. Ajusta los hiperparámetros de un finalista por técnica según la matriz §6. Todo sale de `configs/matriz_experimentos.json` (bloque `hpo`):

| Qué | Valor |
| --- | --- |
| Búsqueda | Optuna, TPE con `seed=42` y parámetros por defecto, sin *pruner* |
| Presupuesto | 12 *trials* por finalista; cada *trial* entrena las semillas 42, 7 y 123 (36 entrenamientos) |
| Objetivo | Media del F1-Macro de validación sobre las tres semillas |
| *Trial* 0 | La configuración de §3: lr por defecto, `weight_decay=0`, lote 32 |
| Espacio `finetune` | `lr` ∈ [5e-6, 3e-4] logarítmico; `weight_decay` ∈ [0, 0.1]; lote ∈ {16, 32, 64} (`vit_b16`: {16, 32}) |
| Espacio `frozen` | `lr` ∈ [1e-4, 1e-2] logarítmico; `weight_decay` ∈ [0, 0.1]; lote ∈ {16, 32, 64} |
| Fijos | Arquitectura, estrategia, preprocesamiento (F1.4), AdamW, 15 épocas, paciencia 4, pesos de clase |
| Regla final | Se adopta el mejor *trial*, salvo que su media no supere a la del *trial* 0 en más de una DE (`ddof=1`) del *trial* 0 |

## Reglas que el script hace cumplir

- **Solo validación.** El loader de test nunca se construye. Las predicciones de validación quedan en los JSON de cada corrida y no en `results/predictions/` (matriz §4).
- **Solo el finalista.** La configuración debe ser el finalista sugerido por `select_finalist` sobre `results/predictions/primera_pasada/`. Si SV elige otra en F3.3, por ejemplo al resolver el empate de Lorenz, se pasa `--override-selection "<motivo>"`, y el motivo queda en `hpo_meta.json` y `hpo_result.json`.
- **Presupuesto fijo.** `--n-trials`, `--seeds` y `--epochs` solo se aceptan con `--smoke`.
- **Un `--out-dir` por estudio.** Si el `hpo_meta.json` existente describe otro estudio, el script se detiene.
- **Reanudable sin perder la reproducibilidad del muestreo.** Cada entrenamiento queda en `runs/trial<NN>__seed<k>.json`. Al volver a correr, el estudio se reconstruye desde el *trial* 0 con el mismo *sampler*. Las corridas que ya existen se leen de disco y solo se entrena lo que falta. Si los parámetros propuestos no coinciden con los guardados, se detiene (`ReplayMismatch`).

## Salidas (`results/f34_hpo/<técnica>/`)

| Archivo | Contenido |
| --- | --- |
| `hpo_meta.json` | Configuración, espacio, *trial* 0, semillas y verificación del finalista |
| `runs/trial<NN>__seed<k>.json` | Hiperparámetros, historial por época, `val_probs` sin redondear, mejor época, entorno y commit |
| `trials.csv` | Una fila por *trial*: parámetros, F1-val por semilla (precisión completa), media y DE |
| `study.pkl` | Estudio de Optuna serializado |
| `hpo_result.json` | Regla final aplicada, hiperparámetros adoptados, banderas para `run_grid` (F3.5/F3.6) y el *trial* 0 frente a F3.2 |
| `hpo.log` | Salida completa |

El *trial* 0 repite la configuración de F3.2. `hpo_result.json` compara su F1-val por semilla con `results/f32_primera_pasada/runs/`. Si la imagen de Kaggle es la misma, debería coincidir; si no coincide, se documenta en la bitácora y no invalida la HPO.

## Notebook (F3.4)

Requisitos de la cuenta: `docs/ejecucion_kaggle.md` §3 (internet, *secret* `GITHUB_TOKEN`, dataset `tt2-insumos` como *Input*, **GPU T4**).

**Celda 1: repositorio en el commit indicado**
```python
from kaggle_secrets import UserSecretsClient
import os, subprocess
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
COMMIT = "<sha>"   # lo indica Claude Code
subprocess.run(["rm", "-rf", "/kaggle/working/ttdicevc"], check=True)
subprocess.run(["git", "clone", f"https://{token}@github.com/DpalaciosR20/ttdicevc.git", "/kaggle/working/ttdicevc"], check=True)
os.chdir("/kaggle/working/ttdicevc")
subprocess.run(["git", "checkout", COMMIT], check=True)
subprocess.run(["git", "log", "--oneline", "-1"], check=True)
```

**Celda 2: insumos y entorno**
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"   # docs/datos.md §2
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!mkdir -p results/f34_hpo/entorno && pip freeze > results/f34_hpo/entorno/pip_freeze.txt && nvidia-smi > results/f34_hpo/entorno/gpu.txt
```
`build_manifest --verify` debe decir `manifiesto OK`.

**Celda 3: prueba de humo** (≈ 5 min)
```python
!python -m src.training.run_hpo --config lorenz:vit_b16:frozen --out-dir /kaggle/working/hpo_smoke \
    --smoke --n-trials 2 --seeds 42 --epochs 1 2>&1 | tail -20
```
Debe terminar con la tabla de 2 *trials* y la línea `Para F3.5/F3.6 (run_grid): --lr ...`. Su salida no se versiona.

**Celda 4: HPO de los tres finalistas** (≈ 3.5–5.5 h de T4; usa *Save Version → Save & Run All*)

Las configuraciones son las que confirme SV en F3.3. Si en Lorenz elige EfficientNet-B0 FT, se cambia esa línea por `--config lorenz:efficientnet_b0:finetune --override-selection "F3.3: <motivo>"`.
```python
for cfg in ["gray_scott:efficientnet_b0:finetune", "perlin:vit_b16:frozen", "lorenz:vit_b16:frozen"]:
    tech = cfg.split(":")[0]
    out = f"results/f34_hpo/{tech}"
    os.makedirs(out, exist_ok=True)
    !python -m src.training.run_hpo --config {cfg} --out-dir {out} 2>&1 | tee {out}/hpo.log
```

| Finalista | Estimado (36 entrenamientos, tiempos de F3.2) |
| --- | --- |
| Gray-Scott EfficientNet-B0 FT | 40–60 min |
| Perlin ViT-B/16 congelado | 1.5–2.3 h |
| Lorenz ViT-B/16 congelado | 1.5–2.3 h |

**Celda 5: empaquetado**
```python
!cd /kaggle/working/ttdicevc && zip -qr /kaggle/working/f34_resultados.zip results/f34_hpo
```

Si se corta a medias: *Add Input → Notebook Output* de la versión anterior y agrega `--prev-results /kaggle/input/<output>/ttdicevc/results/f34_hpo/<técnica>` a esa técnica. El estudio se reconstruye igual y solo se entrena lo que falta.

## Al terminar

1. Descarga el `.zip` y descomprímelo en la raíz del repo.
2. Verificación local: 36 JSON por técnica, `errors/` vacío, el commit de `hpo_result.json` igual al entregado, `selection.is_suggested_finalist` (o el motivo de la excepción) y el *trial* 0 frente a F3.2.
3. Los hiperparámetros adoptados (`run_grid_flags`) son los que usa F3.5 (`docs/correr_grid_kaggle.md`, «Fases con test»), un comando por técnica.
