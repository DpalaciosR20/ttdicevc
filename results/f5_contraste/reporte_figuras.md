# Reporte de figuras — Fase 5, F4.5 y F4.6

Notebook: `notebooks/F5_contraste.ipynb`, ejecutado completo con `jupyter nbconvert --to notebook --execute --inplace`. Las salidas quedan guardadas en el notebook.
- La rama `sv/f5-resultados` se adelantó a `origin/main` (`5af17a8`, merge de PR #37) antes de empezar.
- No se ejecutó `run_fase5` en ningún modo, ni ningún archivo commiteado de `results/f5_contraste/` cambió. Solo se añadieron `figures/` y este reporte.

## Figuras

PNG a 300 dpi en `results/f5_contraste/figures/`. Los colores y marcadores por técnica vienen de `src/viz/style.py`: Gray-Scott naranja ■, Perlin azul ●, Lorenz verde ▲.
- La paleta categórica se validó con `validate_palette.js` del skill dataviz, en modo claro y con `--pairs all`.
- Pasa la separación CVD (ΔE 9.2) y el piso de visión normal (ΔE 24.0).
- El verde queda bajo 3:1 de contraste, así que toda serie lleva también etiqueta de texto y marcador propio.

| # | Archivo | Sección del notebook | Contenido |
|---|---|---|---|
| 1 | `f5_01_f1_prueba_modelos.png` | 1 | F1-Macro de prueba de todos los clasificadores: los visuales con sus 5 semillas y su media, más las líneas del umbral `f12.umbral_f1_visual` y de BETO |
| 2 | `f5_02_theta_forest.png` | 2 (F5.2) | θ con IC de Holm (nivel 1 − 2α_h) por técnica, y θ o δ − Δ en cada variante de sensibilidad |
| 3 | `f5_03_superioridad_piso.png` | 3 (F5.1) | Finalista frente a cada piso, con el piso aplicado rayado, y D con IC de Holm |
| 4 | `f5_04_distribuciones_bootstrap.png` | 4 | Distribuciones de θ\* y D\* por técnica (réplicas reconstruidas y verificadas) |
| 5 | `f5_05_descomposicion_descriptores.png` | 5 (F5.3) | Cascada BETO → 768D → 5D → visual con el IC 95 % de cada etapa, y visual − descriptores con IC |
| 6 | `f5_06_retencion.png` | 6 (F5.6) | ρ̂ con IC 95 % y líneas en 1 y en r = 0.50 |
| 7 | `f5_07_veredictos_robustez.png` | 7 (F5.5) | Mapa de veredictos: variantes de §8, alternativas A2, A3, A7 y A12, y A14 por semilla. La tabla de robustez completa está en la salida de la celda |
| — | (sin figura) | 8 (F5.4) | Tablas de Cochran y McNemar (b, c, p, Holm) en la salida de la celda |
| 8 | `f45_01_longitud.png` | 9 (F4.5) | L por técnica con p; F1 por tercil (5 semillas, media y BETO); ρ de Spearman de u0–u4 con la longitud y el truncamiento |
| 9 | `f46_01_kappa.png` | 10 (F4.6) | Mapa de calor de κ sobre ŷ y sobre vectores de acierto, con IC 95 %. Los pares entre técnicas no se calcularon en F4.6 y aparecen como «n/c» |
| 10 | `f46_02_contencion.png` | 10 (F4.6) | C frente al techo 5D, BETO y 768D, con IC 95 % y la referencia nula; C por semilla |
| 11 | `f46_03_interseccion_errores.png` | 10 (F4.6) | Diagrama de intersección de errores (BETO, techo 5D y finalista) por técnica, con áreas no proporcionales |
| 12 | `f46_04_interseccion_errores_cuatro.png` | 10 (F4.6) | Diagrama de intersección de errores de los cuatro conjuntos (BETO, 768D, techo 5D y finalista) por técnica, en cuatro elipses con las 15 regiones y áreas no proporcionales; los tamaños por par frente a 768D también se verifican contra `f46_resultados.json` |

Todas las figuras llevan:
- un título descriptivo;
- un pie con n = 114 (70/44), las unidades y la fuente.

La lectura está en la celda Markdown que sigue a cada figura y aplica la regla de la cabecera («no se detecta pérdida»; V1 no es equivalencia).

## Comprobación de reproducción de las réplicas

Detalle: `results/f5_contraste/figures/reproduccion_replicas.json` y la tabla de la sección 4 del notebook.

| Comprobación | Resultado |
|---|---|
| `idx_sha256` de `bootstrap.indices(cfg)` | `322edaaa5f1175bb4d4a16409d7f852fa7e67fa5daa8a504bb4976e67d0a7e37`, igual a `resultados.json:meta.bootstrap.idx_sha256` |
| sha256 de los 56 insumos (`insumos.cargar`) | Iguales a `resultados.json:meta.inputs` |
| θ_obs, p de no inferioridad, IC de θ, D_obs, p de superioridad, IC de D, F1 visual, F1 BETO y piso aplicado, por técnica | Coinciden con una tolerancia de 1e-12 |
| `contraste.contraste()` completo: θ_obs, D_obs, p, rechazos, veredictos y global | Coinciden |
| **Total** | **51 de 51 comprobaciones coinciden; la diferencia absoluta máxima es 0.0** |

Para el diagrama de intersección, los conteos se calcularon con las predicciones de prueba ya archivadas. Antes se comprobó que reproducen `f46_resultados.json:contencion` en los 6 pares técnica × {techo 5D, BETO}: número de errores visuales, de la referencia y de la intersección. Los 6 coinciden.

Estas reconstrucciones solo sirven para graficar. No se reportan estadísticos nuevos.

## Tiempo de ejecución

| Medida | Valor |
|---|---|
| `jupyter nbconvert --execute`, ejecución final completa (local, Windows 11, Python 3.13.2, CPU) | 25.2 s de reloj |
| Tiempo del kernel, desde la primera celda hasta la última (`figures/figuras.json`) | 20.9 s |
| Errores en celdas | 0 |

## Notas

- **Ejecuciones previas.** El notebook se ejecutó varias veces mientras se ajustaba el diseño de las figuras. Hubo un error de código propio (`verdict.p_value` devuelve una tupla) y varias correcciones de solapamiento de etiquetas. Ninguna ejecución tocó `run_fase5` ni archivos commiteados; las figuras y los JSON de `figures/` corresponden a la última ejecución.
- **Diagrama de F4.6.** El diagrama de intersección figuraba como **[completar]** en `docs/bitacora/F4.6.md` (desviación 3 y tabla de pendientes). Ahora existe (`f46_03_interseccion_errores.png`), pero esas líneas no se cambiaron porque la consigna limitaba el cambio a añadir la sección «Figuras». SV decide si las actualiza.
- **Actualización (rama `sv/f4-recuperabilidad`, tras integrar `main`).**
  - Se añadió la figura 12 (`f46_04`, cuatro conjuntos). Con ella se cerraron la desviación 3 y la tabla de pendientes de `docs/bitacora/F4.6.md`.
  - Los textos de la sección 10 que decían «condición 2 pendiente de F4.1» ahora dan su resultado: cumplida en Gray-Scott y Lorenz, no en Perlin. Por eso cambiaron los pies de `f46_02` y `f46_03`.
  - El notebook se reejecutó completo. Las demás PNG no cambiaron en bytes, y la tabla de tiempos de arriba corresponde a la ejecución anterior.
- **Tamaño del notebook.** Con las figuras embebidas en sus salidas, `notebooks/F5_contraste.ipynb` pesa unos 1.4 MB. `notebooks/README.md` pide limpiar salidas pesadas, pero aquí se conservan porque la consigna pide guardarlas.
