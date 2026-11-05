## [F3.3, F0.4] Selección del finalista por técnica y migración de las cifras canónicas de F3.2

### Resumen

- Se aplica la regla preregistrada de la matriz §5 (`src/selection/select_finalist.py`, idéntico al de `protocolo-v1`) a las 54 corridas de validación de la primera pasada de F3.2. Una reimplementación independiente da el mismo resultado.
- No se consultó ninguna prueba de los modelos visuales.
- `src/canonical/build.py` lee ahora la repetición de F3.2 (`f32.*`), conserva la v0 como `f32v0.*` (histórica) y añade 30 claves `f33.*` con la selección, el límite de la zona de empate, la holgura y la fragilidad. Cierra F3.2 H6 (F0.4).
- `potencia.py` lee `f32v0.*` y reproduce F1.1 sin cambiar ninguna cifra.
- Suite completa: 636 aprobadas, 30 omitidas (`F03_UMAP_PARAMS`), 0 fallidas. `canonical_numbers.json` al día.
- Bitácora: `docs/bitacora/F3.3.md`. Entregable: `results/f33_seleccion/tabla_seleccion.csv`.

### Finalistas y segundos (F1-Macro de validación, media ± DE sobre las semillas 42, 7, 123)

| Técnica | Finalista | Segundo mejor (F5.5) |
|---|---|---|
| Gray-Scott | EfficientNet-B0 FT (0.7176 ± 0.0125) | ResNet-18 FT (0.7197 ± 0.0113) |
| Perlin | ViT-B/16 congelado (0.7028 ± 0.0073) | EfficientNet-B0 congelado (0.6918 ± 0.0101) |
| Lorenz | ViT-B/16 congelado (0.7148 ± 0.0047) | EfficientNet-B0 FT (0.7166 ± 0.0018) |

La selección es frágil (H2, descriptivo). En Lorenz, ViT-B/16 congelado entra en la zona de empate por 8.4e-05, y un solo volteo de predicción cambia el finalista; en Gray-Scott bastan dos. Es un procedimiento preregistrado, no una conclusión sobre qué arquitectura es mejor. El segundo de Perlin no es el segundo por media: sale de reaplicar la regla sin el finalista (H3).

### Para CE

- ⚠️ **No empezar F3.4 hasta que exista la etiqueta `seleccion-f33`.** Este PR no la crea. La HPO de los tres finalistas debe correr en un commit posterior a la etiqueta.
- **H4, tope de épocas:** en F3.5, registrar por finalista cuántas de las 5 semillas terminan por el tope de 15 épocas. En la primera pasada, 4 de 54 corridas terminaron así, entre ellas el finalista de Perlin (semilla 42); en las cuatro, el F1-val bajó después de su máximo. No se propone cambiar el tope.
- **H5, interpretabilidad:** la matriz §8 fija *attention rollout* sin gradientes para ViT. Con el *backbone* congelado, los mapas de los finalistas de Perlin y Lorenz dependerían solo de la imagen y de los pesos de ImageNet: serían iguales entre semillas y respecto del control de F3.6. Se propone la Enmienda 3: *rollout* ponderado por gradiente (Chefer, Gur y Wolf, 2021) como método principal, conservando el *rollout* puro como atención del *backbone*. Hay que acordarla antes de generar cualquier mapa de F3.7.
- **H10:** `run_grid.py` recibe las configuraciones de `final`, `control_permutacion` y `sensibilidad` por `--configs` y no las contrasta con `select_finalist`/`select_second`; `run_hpo.py` (rama `ce/f3.4-hpo`) sí lo hace. Propuesta: la misma comprobación en `run_grid.py` (matriz §11.10).
- **F3.1:** no existe `docs/bitacora/F3.1.md`; falta su bitácora de cierre.

### Commits

- `60b3e99` [F3.3, F0.4] Colector F3.2 de la repetición, f32v0.* histórica y claves f33.*
- `58f0fa2` [F3.3] Tabla de selección, scripts de verificación y evidencia de las etapas A y B
- [F3.3] Bitácora de cierre y reporte de la etapa C

🤖 Generated with [Claude Code](https://claude.com/claude-code)
