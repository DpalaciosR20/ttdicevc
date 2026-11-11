# Fase 4 — Síntesis: reporte

> Rama `sv/f4-sintesis` (desde `main` en `fb59946`, con `sv/f4-cierre` fusionado, PR #43) · Rol: SV · Sin interpretación

## Entregables

- `notebooks/Fase4_sintesis.ipynb`. Ejecutado con `jupyter nbconvert --to notebook --execute --inplace`: sin errores ni salidas a stderr, con 8 figuras embebidas y tablas.
- `results/fase4_sintesis/`: `figures/` (8 PNG), `tabla_evidencia.csv`, `figuras.csv`, `comprobaciones.csv`, `README.md` y este reporte.

## Figuras

| Archivo | Contenido |
|---|---|
| `fase4_01_cadena_por_etapa.png` | Figura central: F1 de prueba por etapa (BETO, 768D, techo 5D SVM-RBF, LogReg sobre u, LogReg sobre û, visual). Diferencias con IC: cabeza, reducción y renderizado (F5.3), y ΔF1 u − û (F4.1). Línea del umbral equivalente (`f12.umbral_f1_visual` = 0.6019) |
| `fase4_02_ficha_gray_scott.png` | Ficha de Gray-Scott: imágenes de val (3 depresivas y 3 no depresivas), R² por componente (val y prueba), diferencia de mapas medios, ΔF1 de la ablación, f por clase (train) y Spearman de convergencia |
| `fase4_03_ficha_perlin.png` | Ficha de Perlin (octavas por clase) |
| `fase4_04_ficha_lorenz.png` | Ficha de Lorenz (ρ por clase, con la línea de 24.74); advertencia de F4.3 en el pie |
| `fase4_05_direccion_gray_scott.png` | Rejilla de F4.8 con Gray-Scott: fila de centroides (t = 0 … 1) y fila de ±1.5 DE sobre ŵ, con p_dep de la LogReg en cada punto y borde rojo si hubo recorte (no hubo); «ilustrativa» y «sin peso argumentativo» en el título y el pie |
| `fase4_06_direccion_perlin.png` | Ídem, Perlin |
| `fase4_07_direccion_lorenz.png` | Ídem, Lorenz |
| `fase4_08_direccion_componentes.png` | ŵ por componente, media de f por clase (u0) y mediana de octavas por clase (u4); «sin peso argumentativo» en el título y el pie |

Las fuentes de cada figura están en `results/fase4_sintesis/README.md` y `figuras.csv`.

## Comprobaciones contra las canónicas (38, todas OK; `comprobaciones.csv`)

| Grupo | N.º | Qué se comprueba |
|---|---|---|
| Insumos registrados | 6 | sha256 (LF) de los `resultados.json` de F4.1 (val y prueba), F4.2, F4.3 (local), F4.4 y F5 = `canonical_numbers.json:inputs` |
| *Backbone* principal de F4.1 | 1 | El marcado `principal` en `resultados_val.json` coincide con el de la nota canónica de `f41.prueba.gray_scott.condicion_2_f46` |
| Techo 5D | 1 | `ref.techo5d.f1_macro` = `ref.techo5d.svm_rbf.f1_macro` |
| Cadena por etapa | 12 | Para cada técnica: cabeza, reducción, renderizado (`f5.descomposicion.*`) y ΔF1 de F4.1 iguales (≤ 1e-9) a la diferencia de los F1 graficados |
| ΔF1 de F4.1 | 3 | `ic_contiene_0` canónico coincide con el IC graficado |
| Mapas medios | 1 | sha256 de `medias_clase.npz` = `meta.medias_clase_sha256` de F4.3 |
| Imágenes de val | 3 | Huella de las 114 imágenes por técnica (sha256 contra `data_manifest.csv`) = la registrada en F4.4 |
| F4.2, parámetros recalculados desde u | 7 | Media de f por clase y mediana de ρ por clase = `results/f42_regimen/resultados.json`. n con ρ < 24.74 por clase = `f42.lorenz.rho_menor_24_74.n.*`. Conteo de octavas por clase y nivel = `resultados.json` |
| F4.8 | 4 | `results/f48_figura/recorridos.json` = blob versionado en HEAD; las 14 imágenes de cada técnica (`imagenes/<t>_*.png`) = blobs de HEAD, byte a byte |

**Además, dentro de las celdas:**
- `ident()` exige que cada clave de identidad usada sea única y valga 1: veredictos, niveles y lecturas.
- La tabla de evidencia lee cada valor directamente de la clave canónica.

**Excepciones a «toda cifra de las canónicas», verificadas por otra vía:**
1. `results/f48_figura/recorridos.json` e `imagenes/*.png`: F4.8 no tiene claves canónicas por diseño. Se comparan con los blobs de HEAD.
2. `medias_clase.npz`: su sha256 está en el `resultados.json` de F4.3, que sí está registrado.
3. u de `umap_params.pkl`: se verifica contra el manifiesto con `regimen.load_u` y se usa solo para dibujar las distribuciones de F4.2; los resúmenes se comprueban contra `resultados.json`.
4. Imágenes de val: verificadas contra `data_manifest.csv`.

Prueba solo aparece a través de cifras archivadas (F5, F4.1 y F4.6). No hay inferencia nueva.

## Limitaciones citadas (sección 5 del notebook, textuales)

| Tema | Fuente |
|---|---|
| n = 114 | `docs/bitacora/F5.md`, «5. Tamaño de prueba» |
| Fragilidad de la selección | `docs/bitacora/F5.md`, «2. Fragilidad de la selección (F3.3, H2)» |
| Piso permutado | `docs/bitacora/F3.6.md`, H2, «Implicación» |
| Perlin con r ≥ 0.75 | `docs/bitacora/F5.md`, H3 |
| Mapas de Lorenz | `docs/bitacora/F4.3.md`, limitación 4 |
| Descriptores por parche | `docs/bitacora/F4.4.md`, limitaciones 1–4 (P1, N1–N3) |
| Figura ilustrativa | `docs/bitacora/F4.8.md`, nota |

## Tiempo de ejecución

| Medida | Valor |
|---|---|
| `nbconvert --execute`, ejecución final (reloj) | 28.8 s |
| Cuerpo del notebook (`T0` hasta la última celda) | 23.4 s |

- El aumento frente a la versión anterior (13.0 s) se debe a la lectura y verificación contra HEAD de las 42 imágenes de F4.8. La lectura de las 342 imágenes de val desde Drive estaba en caché.
- Durante el desarrollo hubo varias ejecuciones; ninguna escribe insumos.
  - Corregí la incrustación de figuras: `lorenz_lib` fuerza el backend Agg, y se resolvió con `%matplotlib inline` después de importarlo, igual que en F4.2.
  - Ajusté el diseño de las fichas y dos textos: la lectura de «rechaza» en F4.5 y una comprobación duplicada.
- **Corrección posterior, a pedido de SV.** La figura de la sección 4 (`fase4_05_direccion_discriminativa.png`) incrustaba la rejilla completa de F4.8 y no se distinguía. Se sustituyó por una figura por técnica (`fase4_05`–`07`) y una figura final con los tres gráficos de barras (`fase4_08`).

## PENDIENTES

1. **PR de `sv/f4-sintesis` hacia `main`** (no fusionar): ver el estado de `gh` en el mensaje de entrega.
