# results/f02_entorno/

Evidencia de la tarea F0.2. Ver `docs/bitacora/F0.2.md`.

La evidencia original de CE no está en `main`: son `kaggle_versiones.txt`, `local_test_umap_io.log`, `local_test_umap_io_junit.xml` y `tests/test_umap_io.py`. Esos archivos viven en el commit `ea0467e`, que solo está en la rama remota `origin/ce/f0.2-bitacora`, y se leen con `git show ea0467e:<ruta>`. Esta carpeta, en `main`, solo contiene la verificación retrospectiva.

## verificacion_retrospectiva/

Ejecutada el **2026-11-05** como parte de la reconstrucción retrospectiva de la bitácora. **No** se ejecutó al cierre de la tarea. Es de solo lectura: no modifica código, insumos ni resultados de la tarea.

| Archivo | Contenido |
|---|---|
| `umap_io_cli.log` | `python src/training/umap_io.py <TT_DATA_ROOT>/umap_params.pkl`: sha256, tamaños, etiquetas, rango y saturación de **u** |
| `test_umap_io.log`, `test_umap_io_junit.xml` | `pytest tests/test_umap_io.py -v -s`, ejecutado en un *worktree* desacoplado en `ea0467e` (el archivo de prueba no existe en `main`), con `UMAP_PKL=<TT_DATA_ROOT>/umap_params.pkl`. El *worktree* se eliminó después de la corrida |
| `requirements_vs_entornos.txt` | Comparación de cada versión fijada en `requirements.txt` (main `b2cf9f8`) con el `.venv` local y con `results/f32_primera_pasada/entorno/pip_freeze.txt` (Kaggle, repetición de F3.2) |

**Entorno:**
- Windows 11 (10.0.26200), CPU.
- Python 3.13.2 y pytest 8.4.2, del `.venv` del repositorio.
- `TT_DATA_ROOT = G:\Mi unidad\tt2-insumos`.
- Commit de `main`: `b2cf9f8`.

**Reproducir (desde la raíz del repo):**

```
python src/training/umap_io.py "G:/Mi unidad/tt2-insumos/umap_params.pkl"
git worktree add --detach <dir> ea0467e
cd <dir>
UMAP_PKL="G:/Mi unidad/tt2-insumos/umap_params.pkl" python -m pytest tests/test_umap_io.py -v -s
```
