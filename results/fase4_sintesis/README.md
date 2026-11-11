# Fase 4 — Síntesis

Generado por `notebooks/Fase4_sintesis.ipynb` (ejecutado con `jupyter nbconvert --execute`, `TT_DATA_ROOT` definido). No hay cálculos estadísticos nuevos.
- Toda cifra sale de `results/canonical_numbers.json` o de un `resultados.json` registrado en sus `inputs`, con el sha256 verificado.
- Las comprobaciones están en `comprobaciones.csv`.

## Figuras (PNG, 300 dpi, paleta de `src/viz/style.py`)

| Archivo | Figura | Fuente |
|---|---|---|
| `figures/fase4_01_cadena_por_etapa.png` | Cadena texto → u → imagen → clasificador: F1 de prueba de BETO, 768D, techo 5D (SVM-RBF), LogReg sobre u, LogReg sobre û y visual (media de 5 semillas), con los IC por etapa y el umbral equivalente | `ref.beto_exp41`, `ref.emb768_logreg`, `ref.techo5d`, `f41.prueba.*`, `f35.<t>.finalista.f1_macro_mean`, `f5.descomposicion.*`, `f12.umbral_f1_visual` |
| `figures/fase4_02_ficha_gray_scott.png` | Ficha de Gray-Scott | Ver la fila de abajo |
| `figures/fase4_03_ficha_perlin.png` | Ficha de Perlin | Ver la fila de abajo |
| `figures/fase4_04_ficha_lorenz.png` | Ficha de Lorenz (con la advertencia de F4.3 en el pie) | Ver la fila de abajo |
| `figures/fase4_05_direccion_discriminativa.png` | Dirección discriminativa (F4.8) con ŵ por componente y separación por clase de u0 (f) y u4 (octavas) en F4.2; sin peso argumentativo | `results/f48_figura/recorridos.json` y `f48_direccion_grises.png` (blob de HEAD); `results/f42_regimen/resultados.json` |

**Paneles de las fichas y su fuente:**
- imágenes de val, 3 y 3, verificadas contra `meta/data_manifest.csv`;
- R² por componente: `f41.val.*` y `f41.prueba.*`, del *backbone* principal;
- diferencia de mapas medios: `results/f43_atribucion/local/medias_clase.npz`, con su sha256 en `resultados.json` de F4.3;
- ΔF1 de la ablación: `f43.<t>.ablacion.*`;
- parámetro de F4.2, recalculado desde u y comprobado contra `results/f42_regimen/resultados.json`: f en Gray-Scott, octavas en Perlin, ρ en Lorenz;
- Spearman de convergencia: `f44.*`.

## Otros archivos

| Archivo | Contenido |
|---|---|
| `tabla_evidencia.csv` | Tabla técnica × análisis (F5, F4.5, F4.1, F4.7, F4.2, F4.3, F4.4, F4.6), con valor y estado textual |
| `figuras.csv` | Figuras generadas y sus fuentes |
| `comprobaciones.csv` | Las 35 comprobaciones del notebook (sha256 de insumos, cifras graficadas frente a claves canónicas) |
| `reporte.md` | Reporte del cierre |
