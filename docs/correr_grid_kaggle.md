# Correr el grid de 18 configuraciones en Kaggle (CE, Semanas 2–3)

Script: `src/training/run_grid.py`. Lo que corre: 3 técnicas × 3 arquitecturas × 2 estrategias, con `seed=42`. El orden es Gray-Scott → Perlin → Lorenz.

## Qué hace el script

- **Verificación previa:** antes de entrenar, revisa que existan las 3 carpetas y que no falten idx. Si se pasa `--umap-params`, también revisa que el label de cada imagen coincida con `umap_params.pkl`. Si algo falla, se detiene sin haber entrenado nada.
- **Guarda cada corrida al terminar** en `runs/<tecnica>__<arq>__<estrategia>__seed42.json`. Ese JSON incluye F1-val, F1-test, el historial por época y `test_preds`/`test_probs`/`test_true` para McNemar.
- **Reanudable:** las corridas que ya tienen JSON se saltan.
- **Errores:** si una corrida truena, el traceback queda en `errors/` y el grid sigue con la siguiente. Al volver a correrlo, se reintentan.
- **`summary.csv`** se regenera después de cada corrida. La columna `mejor_val` marca con `*` la mejor configuración por técnica según **F1-val** (nunca test). La columna `warning` marca resultados sospechosos (predice una sola clase, o F1-val ≥ 0.95). Esos no se descartan: hay que revisarlos.

## Notebook

**Celda 1: código** (igual que en el smoke test, pero desde `main`)
```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone -q https://{token}@github.com/DpalaciosR20/ttdicevc.git /kaggle/working/ttdicevc
%cd /kaggle/working/ttdicevc
!git remote set-url origin https://github.com/DpalaciosR20/ttdicevc.git
!git log --oneline -1
```

**Celda 2: ver qué se va a correr** (no entrena nada)
```python
DATA = "/kaggle/input/datasets/palaciosroblesdiego/data-files/TT/images_grayscale_classblind"
UMAP = "/kaggle/input/datasets/palaciosroblesdiego/data-files/<ruta>/umap_params.pkl"  # verificar con !find /kaggle/input -name umap_params.pkl
OUT  = "/kaggle/working/grid_seed42"
!python src/training/run_grid.py --data-root {DATA} --umap-params {UMAP} --out-dir {OUT} --dry-run
```

**Celda 3: correr el grid**
```python
!python src/training/run_grid.py --data-root {DATA} --umap-params {UMAP} --out-dir {OUT}
```

**Celda 4: ver resultados**
```python
import pandas as pd
pd.read_csv(f"{OUT}/summary.csv")
```

## Recomendaciones

- **Córrelo con *Save Version → Save & Run All (Commit)*, no en la sesión interactiva.** La sesión interactiva se corta si queda inactiva. En modo commit corre en segundo plano (hasta 12 h con GPU), y `/kaggle/working/` queda guardado como *output* de esa versión.
- **Si se corta a medias:** en un notebook nuevo, haz *Add Input* → *Notebook Output* de la versión anterior y agrega
  `--prev-results /kaggle/input/<output-anterior>/grid_seed42`.
  Las corridas que ya estaban se copian y solo se entrena lo que faltaba.
- **Para correr solo una parte**, por ejemplo Gray-Scott primero para detectar problemas temprano: `--techniques gray_scott`.
- **Si `umap_params.pkl` no carga** por un `ModuleNotFoundError: umap`, agrega una celda `!pip install -q umap-learn` antes. El reductor UMAP guardado necesita esa librería para deserializarse.
- **Modelos:** por defecto no se guardan, porque cada ViT pesa ~330 MB. Para Grad-CAM (Semana 4) se reentrenan los ganadores con `--save-models`.
- **Multi-seed de los ganadores (Semana 4):** es el mismo script con, por ejemplo, `--techniques gray_scott --archs resnet18 --strategies finetune --seeds 42 7 123 2024 99`.

## Al terminar

1. Descarga `grid_seed42/` (el `summary.csv` y los JSON) desde el output de la versión.
2. Pasa `summary.csv` a SV para la tabla consolidada.
3. Si aparece algo en `errors/` o en la columna `warning`, documéntalo en la bitácora. No lo descartes.
