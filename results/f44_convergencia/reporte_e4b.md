# F4.4 y F4.8, etapa 4b: reporte (datos reales)

> Rama `sv/f4-cierre` · Rol: SV · Sin interpretación. La interpretación, marcada como tal, está en `docs/bitacora/F4.4.md` y `docs/bitacora/F4.8.md`.

## 1. Resoluciones (`ff1654c`, antes de calcular)

**F4.4: `results/f44_convergencia/resoluciones.json`.**

| Clave | Valor |
|---|---|
| `geometria_parche` | `redimensionada_224` |
| `spearman_indefinido` | `excluir_y_contar` |
| `mapa` | `map` |
| `bootstrap_rng` | `por_tecnica_compartido` |
| `ic_cuantiles` | `linear` |
| `naturaleza_descriptores` | `propuesta` |
| Metadatos | `_autor`, `_fecha` y `_nota` |

**F4.8: `results/f48_figura/resoluciones.json`.**

| Clave | Valor |
|---|---|
| `direccion_espacio` | `u` |
| `de_recorrido` | `proyeccion_train` |
| `n_pasos_recorrido` | 7 |
| `tol_pdep` | 1e-9 |
| `paleta` | `viridis` |
| `marca_recorte` | `borde_y_lista` |
| Metadatos | `_autor`, `_fecha` y `_nota` |

**Cambios de código en el mismo commit:**
- `NATURALEZA['contrast']` pasa a «dispersión global de intensidades (sin banda específica)», según la nota de SV. Prueba: `test_naturaleza_del_contraste_sin_banda`.
- El contrato de F4.8 acepta `n_pasos_recorrido` (7, 5, 9) y `tol_pdep` (1e-9, 1e-6) como números, que es como vienen en el archivo; en E4a se esperaban cadenas. Rechaza `'7'`, `7.0`, `True` y `'1e-9'`. Prueba: `test_check_resoluciones_f48`.

## 2. Cálculo

| Corrida | Comando | HEAD | Resultado | Duración | Commit de la salida |
|---|---|---|---|---|---|
| F4.4 | `python -m src.fase4.convergencia --root G:\.shortcut-targets-by-id\…\tt2-artefactos --resoluciones results/f44_convergencia/resoluciones.json` | `ff1654c`, árbol limpio | código 0 | 106.4 s | `03c10a4` |
| F4.8, intento 1 | `python -m src.fase4.figura_ilustrativa --resoluciones results/f48_figura/resoluciones.json` | `ff1654c`, árbol limpio | **falló al dibujar:** `ModuleNotFoundError: No module named 'viz'`, después de renderizar las 42 imágenes. La salida parcial (`imagenes/`) se borró sin versionarla | — | — |
| F4.8, corrección | `src` agregado a `sys.path`; prueba nueva `test_figura_se_dibuja_y_marca_recortes` | — | 33 passed | — | `c578878` |
| F4.8, intento 2 | mismo comando | `c578878`, árbol limpio | código 0 | 164.3 s | `a27c440` |

### 2.1 F4.4 (`results/f44_convergencia/resultados.json`)

| Técnica | Descriptor | Spearman medio | IC al 95 % | n | Indefinidos |
|---|---|---|---|---|---|
| Gray-Scott | Entropía | 0.0228 | [−0.0740, 0.1182] | 114 | 0 |
| Gray-Scott | Contraste | −0.0376 | [−0.1213, 0.0450] | 114 | 0 |
| Gray-Scott | Densidad de bordes | 0.0389 | [−0.0608, 0.1353] | 114 | 0 |
| Gray-Scott | Energía LBP | −0.0158 | [−0.0994, 0.0694] | 114 | 0 |
| Perlin | Contraste | 0.4123 | [0.3858, 0.4374] | 114 | 0 |
| Perlin | Densidad de bordes | 0.0125 | [−0.0017, 0.0275] | 114 | 0 |
| Lorenz | Entropía | 0.6482 | [0.6331, 0.6626] | 114 | 0 |
| Lorenz | Energía LBP | −0.6473 | [−0.6616, −0.6324] | 114 | 0 |

**Verificación de insumos:** 15 `.npz` de F3.5 contra el manifiesto, y 342 imágenes de val contra `data_manifest.csv`.

**Contexto citado.**
- A11 (`f5.frente_a_descriptores.*`, prueba): Gray-Scott 0.0454 [−0.0165, 0.1091]; Perlin 0.0298 [−0.0487, 0.1090]; Lorenz 0.1454 [0.0485, 0.2434].
- Lectura de F4.3: Perlin `cordura_por_datos_superada`; Lorenz `dependiente_de_la_cabeza_sin_validacion_por_datos`.

### 2.2 F4.8 (`results/f48_figura/recorridos.json`)

- **LogReg reproducida:** C = 0.1, `balanced`. La diferencia máxima de `p_dep` de val frente al CSV de F2.1 es 0.0.
- **Dirección:** ŵ = [0.5719, 0.1335, 0.0970, 0.1071, −0.7964], en el espacio de u. La DE del recorrido es 0.3779.
- **Recorte:** ningún punto de los dos recorridos tiene componentes fuera de [0, 1].
- **Salidas:** 42 imágenes; `f48_direccion_grises.png` y `f48_direccion_paleta_ilustrativa.png`.

## 3. Cierre

- **Canónicas (`9089cb0`).** `collect_f44` genera 56 claves `f44.*`, todas en `value.val`. Por técnica y descriptor:
  - Spearman medio, IC inferior y superior, y nivel;
  - n y n de indefinidos;
  - si el IC contiene 0.

  Pruebas nuevas en `tests/test_canonical_numbers.py`:
  - localizador y split;
  - **recálculo** de media e IC desde `spearman_por_muestra.csv`, con `default_rng(2026)` y cuantiles `linear` (tolerancia 1e-12);
  - resoluciones versionadas.

  Con `CANON_STRICT=1` y `CANON_UMAP_PARAMS`, el build se escribió y `--check` da «al día (28565 líneas)».
- **Pruebas.** `python -m pytest tests` con `CANON_STRICT=1`: 1297 passed, 30 skipped (`results/f44_convergencia/pytest_e4b.log`).
- **Notebooks (`498dd03`),** ejecutados con `nbconvert --execute`:
  - `notebooks/F4.4_convergencia.ipynb`: Spearman con IC, distribución por muestra, tabla cualitativa, A11, lectura de F4.3 y ejemplos de parches, con 3 figuras en `results/f44_convergencia/figures/`;
  - `notebooks/F4.8_figura.ipynb`: tabla de recorridos y las dos figuras.
- **Bitácoras:** `docs/bitacora/F4.4.md` y `docs/bitacora/F4.8.md`, sin ningún [completar]. No se hizo la ronda de preguntas previa de `/cierre-tarea`: la instrucción de E4b ya fijaba el contenido.
- **Push y PR:** `gh` no tiene sesión iniciada, así que el PR no se creó (comando abajo).

## PENDIENTES

1. **Crear el PR** de `sv/f4-cierre` hacia `main`; no fusionar. `gh` necesita `gh auth login`.
