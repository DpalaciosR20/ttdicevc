# results/f01_checksums/

Evidencia de la tarea F0.1. Ver `docs/bitacora/F0.1.md`.

El entregable `data/images/checksums.csv` que pide `plan/plan_tt2_v2.md` (fila F0.1) no existe. No está en el repositorio, en su historial ni en `TT_DATA_ROOT`. Esta carpeta solo contiene la verificación retrospectiva.

## verificacion_retrospectiva/

Ejecutada el **2026-11-05** como parte de la reconstrucción retrospectiva de la bitácora. **No** se ejecutó al cierre de la tarea. Solo lectura: no modifica imágenes, código ni manifiesto.

| Archivo | Contenido |
|---|---|
| `verificar_imagenes_vs_manifiesto.py` | Recalcula el SHA-256 de cada PNG de `images_grayscale_classblind/{perlin,gray_scott,lorenz}` y lo compara, por técnica, con las filas `IMAGES[...]` de `meta/data_manifest.csv`. Reutiliza `src.build_manifest.dir_digest` y `src.training.umap_io.sha256` |
| `verificacion_imagenes.log` | Salida literal del script anterior |
| `verificacion_por_tecnica.csv` | Resumen por técnica: coinciden, difieren, faltantes, sobrantes y `dir_digest` |
| `build_manifest_verify.log` | Salida literal de `python -m src.build_manifest --verify` (todos los insumos del manifiesto) |

**Entorno:**
- Windows 11 (10.0.26200), Python 3.13.2 (`.venv` del repositorio).
- `TT_DATA_ROOT = G:\Mi unidad\tt2-insumos`.
- Commit del repositorio: `b2cf9f8`.
- Duración del script de imágenes: 41.2 s.

**Reproducir (desde la raíz del repo):**

```
TT_DATA_ROOT="G:\Mi unidad\tt2-insumos" python results/f01_checksums/verificacion_retrospectiva/verificar_imagenes_vs_manifiesto.py
TT_DATA_ROOT="G:\Mi unidad\tt2-insumos" python -m src.build_manifest --verify
```

No se ejecutó `src/generation/compute_checksums.py`: escribe en `data/images/`, y sus nombres de carpeta (`reaction_diffusion`, `attractor`) no corresponden a los de `TT_DATA_ROOT` (`gray_scott`, `lorenz`).
