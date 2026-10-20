# Correr el grid en Kaggle (CE, plan v2 — Fase 3)

Script: `src/training/run_grid.py`. Cubre 3 técnicas × 3 arquitecturas × 2 estrategias = 18 configuraciones. El orden es Gray-Scott → Perlin → Lorenz.

| Paso del plan v2 | Comando (resumen) | ¿Toca test? |
| --- | --- | --- |
| **F3.2** Primera pasada | por defecto: 18 configs × semillas 42, 7, 123 | **No**: test ni se carga |
| **F3.5** Finalistas | `--evaluate-test --seeds 42 7 123 2024 99` + filtros del finalista | Sí. Requiere el protocolo congelado |
| **F3.6** Control de permutación | finalista + `--permute-labels 99` | Solo si se combina con `--evaluate-test` |
| **F1.4** Preprocesamiento | `--img-size 224\|256`, `--no-hflip` | — |

## Reglas que el script hace cumplir

- **Compuerta de test:** sin `--evaluate-test` no se construye el loader de test. Con `--evaluate-test`, el script se niega si `docs/protocolo_preregistrado.md` no existe, no está commiteado o tiene cambios sin commitear. El commit y la fecha del protocolo quedan dentro de cada JSON.
- **Un `--out-dir` es de un solo modo.** No se mezclan corridas solo-val con corridas con test; para la fase con test usa otro `--out-dir`.
- **Verificación previa** antes de entrenar:
  - las 3 carpetas existen y no faltan idx;
  - con `--umap-params`, los labels coinciden con `umap_params.pkl`;
  - ViT no se pide con una resolución distinta de 224 (el ViT-B/16 preentrenado de torchvision solo acepta 224).
- **Reanudable:** un JSON por corrida en `runs/`; las que ya existen se saltan. Los errores van a `errors/` y se reintentan en la siguiente ejecución.

## Salidas (se regeneran después de cada corrida)

| Archivo | Contenido | Para quién |
| --- | --- | --- |
| `summary.csv` | una fila por corrida: F1-val (y F1-test si aplica), épocas, parámetros entrenables, minutos, `warning` | CE (bitácora) |
| `summary_by_config.csv` | media ± std de F1-val sobre semillas. Columna `seleccion`: `finalista_sugerido` es el mejor promedio, y ante empate dentro de 1 std, el de menos parámetros entrenables (regla 3.6). `empate_1std` marca las configuraciones empatadas | SV confirma el finalista (F3.3) |
| `predictions.csv` | `sample_id, split, y_true, technique, arch, strategy, seed, p_dep, run_id` (formato F1.5). `p_dep` = P(label=1 = depresivo); `sample_id` = `<split>_<idx:04d>` (mismo idx que `umap_params.pkl`); `run_id` distingue variantes de preprocesamiento y controles de permutación | SV (pruebas pareadas) |

Los controles de permutación (variantes `-perm<semilla>`) aparecen en los resúmenes, pero nunca se eligen como finalistas.

## Notebook

**Celda 1: código**. El `%cd /kaggle/working` del inicio evita el error `getcwd` si la carpeta ya existía.
```python
%cd /kaggle/working
!rm -rf /kaggle/working/ttdicevc
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone -q https://{token}@github.com/DpalaciosR20/ttdicevc.git /kaggle/working/ttdicevc
%cd /kaggle/working/ttdicevc
!git remote set-url origin https://github.com/DpalaciosR20/ttdicevc.git
!git log --oneline -1
```

**Celda 2: rutas y dry-run** (no entrena nada)
```python
DATA = "/kaggle/input/datasets/palaciosroblesdiego/data-files/TT/images_grayscale_classblind"
UMAP = "/kaggle/input/datasets/palaciosroblesdiego/data-files/<ruta>/umap_params.pkl"  # !find /kaggle/input -name umap_params.pkl
OUT  = "/kaggle/working/f32_primera_pasada"
!python src/training/run_grid.py --data-root {DATA} --umap-params {UMAP} --out-dir {OUT} --dry-run
```
Debe decir `Modo: SOLO VALIDACION` y `54 corridas ... 54 pendientes` (18 × 3 semillas).

**Celda 3: primera pasada (F3.2)**
```python
!python src/training/run_grid.py --data-root {DATA} --umap-params {UMAP} --out-dir {OUT}
```

**Celda 4: ver resultados**
```python
import pandas as pd
pd.read_csv(f"{OUT}/summary_by_config.csv")
```

## Recomendaciones

- **Empieza con una sola técnica para medir tiempos:** `--techniques gray_scott`. ResNet-18 fine-tuning en Gray-Scott debería quedar en un F1-val cercano a lo que se vio en el camera-ready. Si sale muy distinto, revisa antes de lanzar el resto. La columna `minutes` de `summary.csv` sirve para estimar lo que falta, que el v2 multiplica (3 semillas, HPO, 5 semillas por finalista, controles).
- **Corre con *Save Version → Save & Run All (Commit)*.** Así sigue en segundo plano (hasta 12 h con GPU) y `/kaggle/working/` queda como output de la versión.
- **Si se corta a medias:** en una versión nueva, haz *Add Input → Notebook Output* de la anterior y agrega `--prev-results /kaggle/input/<output-anterior>/f32_primera_pasada`. Solo se entrena lo que faltaba.
- **Si `umap_params.pkl` no carga** (`ModuleNotFoundError: umap` o error de numba): primero `!pip install -q umap-learn`. Si aun así falla (pendiente F0.2), corre sin `--umap-params`: los labels se leen del nombre de archivo igual, solo pierdes la verificación cruzada.
- **Modelos:** por defecto no se guardan (ViT ~330 MB c/u). Para Grad-CAM y *attention rollout* (F3.7), entrena los finalistas con `--save-models`.
- **Preprocesamiento (F1.4):** hasta que SV decida, usa el default (224 + flip). Si se decide otra cosa para una técnica, córrela con su propio comando, por ejemplo `--techniques perlin --img-size 256 --no-hflip --archs resnet18 efficientnet_b0`, y ViT aparte con 224. La variante queda en el nombre de cada corrida (`px256-noflip`), así que no se pisa con las demás.

## Fase de finalistas (después de congelar el protocolo)

Ejemplo, si SV elige ResNet-18 fine-tuning para Gray-Scott:
```python
OUT_TEST = "/kaggle/working/f35_finalistas"
!python src/training/run_grid.py --data-root {DATA} --umap-params {UMAP} --out-dir {OUT_TEST} \
    --evaluate-test --techniques gray_scott --archs resnet18 --strategies finetune \
    --seeds 42 7 123 2024 99 --save-models
# Control de permutación (F3.6), mismo finalista:
!python src/training/run_grid.py --data-root {DATA} --umap-params {UMAP} --out-dir {OUT_TEST} \
    --evaluate-test --techniques gray_scott --archs resnet18 --strategies finetune \
    --seeds 99 --permute-labels 99
```
Si los hiperparámetros cambian por la HPO (F3.4), habrá que pasarlos al script. Todavía no están expuestos: se agregan cuando exista la HPO.

## Al terminar

1. Descarga la carpeta de salida (CSVs y `runs/`) desde el output de la versión.
2. Pasa `summary_by_config.csv` y `predictions.csv` a SV.
3. Si aparece algo en `errors/` o en la columna `warning`, documéntalo en la bitácora. No lo descartes.
