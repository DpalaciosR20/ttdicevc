# Correr el grid en Kaggle (CE, plan v2 — Fase 3)

Script: `src/training/run_grid.py`. Cubre 3 técnicas × 3 arquitecturas × 2 estrategias = 18 configuraciones (`docs/matriz_experimentos.md`). Se ejecuta en Gray-Scott → Perlin → Lorenz. Convención general de Kaggle: `docs/ejecucion_kaggle.md`.

| Paso | Comando (resumen) | Fase (`results/predictions/<fase>/`) | ¿Toca test? |
| --- | --- | --- | --- |
| **F3.2** Primera pasada | sin banderas: 18 configs × semillas de la matriz (42, 7, 123) | `primera_pasada` | **No**: test ni se carga |
| **F3.5** Finalistas | `--evaluate-test --configs <tec:arq:est> ... --save-models` | `final` | Sí |
| **F3.6** Control de permutación | `--evaluate-test --permute-labels 99 --configs ... --save-models` | `control_permutacion` | Sí |
| **F5.5** Sensibilidad | `--evaluate-test --sensitivity --configs ...` (segundo mejor, `select_second`) | `sensibilidad` | Sí |
| Prueba de humo | `--smoke --epochs 1 ...` | `<out-dir>/predictions/primera_pasada` | No |

## De dónde sale cada valor

| Qué | Fuente única | Qué hace el script si no coincide |
| --- | --- | --- |
| Tamaño, interpolación, antialias y volteo por técnica | `configs/preprocesamiento.json` (F1.4): `perlin` y `lorenz` = `px224-noflip`, `gray_scott` = `px224-flip` | Falla si falta la técnica o si un valor no está implementado. `--img-size` y `--no-hflip` se aceptan solo si coinciden con el JSON |
| Arquitecturas, estrategias, AdamW, `lr`, `weight_decay`, lote, épocas, paciencia, semillas por fase, determinismo | `configs/matriz_experimentos.json` (F1.3) | En `primera_pasada` y `sensibilidad` no se aceptan `--lr`, `--weight-decay`, `--batch-size`, `--epochs` ni `--patience` (solo `--smoke` acorta épocas). En `final` y `control_permutacion` se aceptan los de la HPO (F3.4) y quedan en el manifiesto |
| Insumos | `TT_DATA_ROOT` (`docs/datos.md` §2, `src/paths.py`) | Antes de entrenar compara el SHA-256 agregado de cada carpeta de imágenes con `meta/data_manifest.csv` y cada nombre y etiqueta con `meta/sample_index.csv`. Si algo difiere, no entrena nada |

Los nombres de salida siguen el vocabulario de F1.5: `vit_b16` (internamente torchvision usa `vit_b_16`).

**Determinismo** (matriz §3, §11.6): además de `cudnn.deterministic`, el script activa `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` y desactiva los kernels SDPA *flash* y *memory-efficient*. Si una operación no tiene versión determinista en CUDA, PyTorch lanza `RuntimeError` y la corrida cae en `errors/`. Por eso la prueba de humo va antes del grid. `--deterministic-warn-only` relaja esto a un aviso, pero solo se usa si SV lo aprueba, y queda registrado en cada manifiesto.

## Reglas que el script hace cumplir

- **Compuerta de test:** sin `--evaluate-test` no se construye el loader de test. Con `--evaluate-test`, el script se niega si `docs/protocolo_preregistrado.md` no está commiteado sin cambios **o** si la etiqueta `protocolo-v1` no existe o no es ancestro de `HEAD`. El commit del protocolo y la etiqueta quedan en cada JSON.
- **Un `--out-dir` es de una sola fase** (y de humo o no). No se mezclan.
- **No se sobrescriben predicciones de otro experimento:** si ya existe un CSV con la misma clave (fase, técnica, arquitectura, estrategia, semilla) y su manifiesto registra otra variante de preprocesamiento o de permutación, el script se detiene.
- **Reanudable:** un JSON por corrida en `runs/`; las que ya existen se saltan. Los errores van a `errors/` y se reintentan en la siguiente ejecución. Los JSON de una versión anterior del script (sin `fase`) se rechazan.

## Salidas (se regeneran después de cada corrida)

| Archivo | Contenido | Para quién |
| --- | --- | --- |
| `results/predictions/<fase>/<tec>__<arq>__<est>__seed<k>.csv` | Formato F1.5 (`docs/especificacion_predicciones.md`): `sample_id, split, y_true, technique, arch, strategy, seed, p_dep`, con `p_dep` en float64 sin redondear y filas ordenadas por split y `sample_id` | SV (pruebas pareadas, `load_predictions`, `select_finalist`) |
| `... .manifest.json` | `task_id`, `fase`, `git_commit`, `created_at`, `config` (variante y preprocesamiento, hiperparámetros, determinismo aplicado, SHA-256 de los dos JSON de configuración, índices de permutación), `inputs` (carpeta de imágenes con `dir_digest` e índice maestro), `model_selection`, `environment`, `csv_sha256`, `n_rows` | Trazabilidad del protocolo |
| `<out-dir>/runs/*.json` | Todo lo anterior más el historial por época (`history`: `train_loss`, `f1_val`) para las curvas de aprendizaje | CE (bitácora, figuras) |
| `<out-dir>/summary.csv` | Una fila por corrida: F1-val (y F1-test si aplica), épocas, parámetros entrenables y totales, minutos, `warning` | CE |
| `<out-dir>/summary_by_config.csv` | Media ± DE de F1-val por configuración. La columna `seleccion` aplica `src/selection/select_finalist.py` sobre las técnicas completas; es informativa, la selección oficial de F3.3 la corre SV | SV |

## Notebook (F3.2)

Requisitos de la cuenta: `docs/ejecucion_kaggle.md` §3 (internet, *secret* `GITHUB_TOKEN`, dataset `tt2-insumos` como *Input*, GPU T4).

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
os.environ["TT_DATA_ROOT"] = "/kaggle/input/<ruta de tt2-insumos verificada con ls>"   # docs/datos.md §2
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!mkdir -p results/_entorno && pip freeze > results/_entorno/pip_freeze.txt && nvidia-smi > results/_entorno/gpu.txt
```
`build_manifest --verify` debe decir `manifiesto OK`. Si reporta archivos no registrados en `perlin/` (los duplicados `... (1).png` de F1.5), la copia de Kaggle es anterior al 28 oct: no sigas.

**Celda 3: prueba de humo** (≈ 10 min, solo validación)

> **Ya ejecutada (2 nov 2026, commit `36c6398`):** los resultados están en `results/f32_primera_pasada/` y `results/predictions/primera_pasada/`, y la v0 está en `results/f32_primera_pasada_v0/` (bitácora `docs/bitacora/F3.2.md`). Las celdas 3–5 quedan como registro de lo que se corrió. En un commit posterior a la integración, `run_grid` con el mismo `--out-dir` encontraría las 54 corridas hechas y no entrenaría nada; para repetirla se usa otro `--out-dir` y otra `--predictions-root`, porque el script no sobrescribe predicciones de otra variante.

En `36c6398`, `results/f32_primera_pasada/` del repo todavía tenía la primera pasada v0 (formato anterior), y el script se niega a mezclarla con la nueva. Por eso en el clon de Kaggle se apartó primero:
```python
!mv results/f32_primera_pasada /kaggle/working/f32_primera_pasada_v0 && mkdir -p results/f32_primera_pasada
!python -m src.training.check_determinism --technique gray_scott --epochs 2 \
    --configs vit_b16:finetune resnet18:finetune efficientnet_b0:finetune \
    --out results/f32_primera_pasada/determinismo.json 2>&1 | tee results/f32_primera_pasada/determinismo.log
!python -m src.training.run_grid --smoke --epochs 1 --out-dir /kaggle/working/smoke \
    --archs resnet18 --strategies frozen --seeds 42 2>&1 | tee results/f32_primera_pasada/smoke.log
!python -m src.predictions.validate "/kaggle/working/smoke/predictions/primera_pasada/*.csv"
!grep -h '"variant"' /kaggle/working/smoke/predictions/primera_pasada/*.manifest.json
```
- `check_determinism`: cada configuración reporta `identical_weights`. Una `error` con `RuntimeError ... does not have a deterministic implementation` significa que ese kernel no es determinista en CUDA: **detente y avisa a SV** antes del grid.
- `validate`: `3/3 archivos válidos`. Las variantes: `px224-flip` en Gray-Scott y `px224-noflip` en Perlin y Lorenz.

**Celda 4: primera pasada completa** (≈ 90 min de T4 en v0; puede subir con el modo determinista)
```python
OUT = "results/f32_primera_pasada"
!python -m src.training.run_grid --out-dir {OUT} --dry-run
!python -m src.training.run_grid --out-dir {OUT} 2>&1 | tee {OUT}/run_grid.log
!python -m src.predictions.validate "results/predictions/primera_pasada/*.csv"
```
El dry-run debe decir `Fase: primera_pasada`, `SOLO VALIDACION` y `54 corridas ... 54 pendientes`. Al final, `54/54 archivos válidos`.

**Celda 5: empaquetado**
```python
!cd /kaggle/working/ttdicevc && zip -qr /kaggle/working/f32_resultados.zip results/f32_primera_pasada results/predictions/primera_pasada results/_entorno
```

Si la sesión interactiva no aguanta las ~90 min, corre el mismo notebook con *Save Version → Save & Run All* y descarga el `.zip` del output de la versión. Si se corta a medias: *Add Input → Notebook Output* de la versión anterior y agrega `--prev-results /kaggle/input/<output>/ttdicevc/results/f32_primera_pasada` a la celda 4; solo se entrena lo que falta.

## Fases con test (después de F3.3 y F3.4)

Para F3.5, las celdas completas, con las compuertas, la comprobación sin entrenar, las huellas de los modelos y el empaquetado, están en `docs/correr_final_kaggle.md`. Abajo queda la forma general del comando.

```python
FIN = "--configs gray_scott:<arq>:<est> perlin:<arq>:<est> lorenz:<arq>:<est>"   # de select_finalist
!python -m src.training.run_grid --out-dir results/f35_final --evaluate-test {FIN} <run_grid_flags de results/f34_hpo/<tec>/hpo_result.json> --save-models
!python -m src.training.run_grid --out-dir results/f36_permutacion --evaluate-test --permute-labels 99 {FIN} <run_grid_flags de results/f34_hpo/<tec>/hpo_result.json> --save-models
```
Las semillas por defecto son las de la matriz para cada fase (42, 7, 123, 2024, 99). Si cada finalista tiene hiperparámetros distintos, se corre un comando por técnica (y un `--out-dir` por comando).

`final` y `control_permutacion` solo aceptan los finalistas de `select_finalist`, y `sensibilidad` los segundos de `select_second`; los dos se calculan sobre `results/predictions/primera_pasada` (F3.3 H10). Cualquier otra configuración se rechaza, salvo que se pase `--override-selection "<motivo>"`. Pasa siempre las tres banderas de `run_grid_flags` (`--lr --weight-decay --batch-size`): la que falte toma el valor común de la matriz.

## Al terminar

1. Descarga el `.zip` y descomprímelo en la raíz del repo.
2. Verificación local: `python -m src.predictions.validate "results/predictions/primera_pasada/*.csv"`, que el commit de los manifiestos sea el entregado y que `errors/` esté vacío.
3. Si aparece algo en `errors/` o en la columna `warning`, documéntalo en la bitácora. No lo descartes.
