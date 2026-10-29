# F1.4 — Recuperabilidad de u bajo el preprocesamiento

Pruebas P1–P4 de `docs/decisiones_preprocesamiento.md`. Las conclusiones y la decisión están en ese documento, en §6.2, §6.3 y §7, y la decisión se materializa en `configs/preprocesamiento.json`.

## Commits

| Momento | Commit | Fecha | Contenido |
|---|---|---|---|
| Preregistro | `5b9831c` | 2026-10-29 17:01 (−06:00) | §1–§5 (criterio y regla) y §6.1 (preprocesamiento vigente y precisiones operativas); §6.2–§7 vacías |
| Ejecución | `6d83c27` | 2026-10-29 17:17 (−06:00) | Script, salidas, §6.2–§7 completadas, `configs/preprocesamiento.json` y su prueba |

El script se ejecutó una sola vez, después del preregistro y con HEAD en `5b9831c`. El código ejecutado es el que quedó registrado en `6d83c27` (SHA-256 del script más abajo).

## Entorno

- Windows 11 Pro 10.0.26200, solo CPU, `.venv` del repositorio.
- Python 3.13.2 · torch 2.11.0+cpu · torchvision 0.26.0+cpu · numpy 2.1.3 · scipy 1.16.3 · scikit-learn 1.6.1 · statsmodels 0.15.0 · pandas 2.3.3 · pillow 12.3.0.
- Duración: 11.2 min.

## Insumos

| Insumo | Ruta | SHA-256 |
|---|---|---|
| `UMAP_PARAMS` | `umap_params.pkl` (relativa a `TT_DATA_ROOT`) | `081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224` |
| `IMAGES[perlin]` | `images_grayscale_classblind/perlin` (SHA de la carpeta según el manifiesto) | `b8ce53c6fa339d3a4dcca2ba647c55aeaec79909f7137563efd8b5d04f7e3ccb` |
| `IMAGES[gray_scott]` | `images_grayscale_classblind/gray_scott` | `c3593c9bd04b4abb45afe7958bded087b53b7812a7cb11e72bda8e382f2d31db` |
| `IMAGES[lorenz]` | `images_grayscale_classblind/lorenz` | `6fc478a58477f5ba76d9105435be179c52539db2d67dbee3918a67dc5cfdde9b` |
| Índice maestro | `meta/sample_index.csv` | `74dc778292447c66be60b35b5ec224c53b01433423923fa6d6ddef91749de0f5` |
| Manifiesto | `meta/data_manifest.csv` | `5ea6f81034cec340d19b841414d60b4d44a2334f317b46a49b8f639564b5ea4a` |
| Pesos ResNet-18 `IMAGENET1K_V1` | `resnet18-f37072fd.pth` (caché de torch hub) | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Script | `f14_recuperabilidad.py` | `97d98cb1135e0f9299e4d92643554451e70eba1e9b75f01a74a1f5cfd1f03ab2` |

El script verifica el SHA-256 de `umap_params.pkl` y el de **cada una** de las 908 imágenes de train de cada técnica contra las filas por archivo de `meta/data_manifest.csv`. No calcula el SHA agregado de las carpetas, porque para eso tendría que leer los archivos de val y de test. Del índice maestro solo lee `sample_id`, `split`, `idx` e `img_*` (`y_true` no se lee); de `umap_params.pkl` solo usa `['train']['params']`. Ninguna imagen de val ni de test se abre (aserción sobre los nombres de archivo).

## Salidas

| Archivo | Contenido |
|---|---|
| `r2_por_componente.csv` | P1: technique, variant (V0/V1/V2), component, R2, dR2 = R²(V0) − R²(V), IC 95 % *bootstrap* y si cumple ΔR² ≤ 0.02 |
| `sensibilidad_volteo.csv` | P2: R²(V), R²(F(V)), dR2_F e IC, con la partición retenida volteada completa |
| `espectro.csv` | P3: fracción de energía > 112 ciclos por imagen (media, mediana, máximo) y ρ y p de Spearman con cada u_j |
| `orientacion.csv` | P4: rho, p y p_holm (30 contrastes) del centroide horizontal y de la *skewness* ponderada de x |
| `decision.csv` | Regla de §5 aplicada por el script: variantes admitidas, tamaño, volteo y ViT |
| `f14.log` | Registro de la ejecución: entorno, verificaciones, α de `RidgeCV` y R² por variante |

## Reproducir

```
TT_DATA_ROOT="G:\Mi unidad\tt2-insumos"   # ver docs/datos.md §2
.venv/Scripts/python.exe results/f14_preprocesamiento/f14_recuperabilidad.py
python -m pytest tests/test_preprocesamiento_config.py -v
```

Todo es determinista en CPU: KFold y *bootstrap* usan la semilla 2026 y la extracción de características se hace en `eval()` sin gradientes.
