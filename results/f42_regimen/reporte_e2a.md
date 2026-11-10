# Reporte E2a: F4.2 (régimen dinámico) y adenda de F0.3 (capa C de Perlin y Lorenz)

> Bloque E, etapa 2a · Rol: SV · Rama `sv/f4-regimen`, creada desde `main` (`3a0730f`)
> Especificación: `docs/plan_fase4.md`, F4.2 · Celdas: `docs/correr_f42_kaggle.md`
> Estado: **DETENIDO antes de Kaggle.** No se usaron datos reales. No se tocó `results/f5_contraste/`. No hubo inferencia de finalistas ni de controles.

## 0. Compuerta

| Comprobación | Resultado |
|---|---|
| `enmienda-5` es ancestro de la rama | sí (`git merge-base --is-ancestor enmienda-5 origin/main`) |
| `docs/plan_fase4.md` sin cambios frente a `enmienda-5` | sí (`git diff --quiet enmienda-5 origin/main -- docs/plan_fase4.md`) |
| `src/generation/` y `tests/test_label_independence.py` sin cambios frente a `main` | sí (`git diff --stat` vacío) |

Los dos scripts vuelven a aplicar la compuerta al correr:
- `reintentos_gs.py` exige `enmienda-5` como ancestro.
- `regimen.py` exige además el plan sin cambios, `results/f5_contraste/resultados.json` versionado (G1) y el árbol limpio.

## 1. Código

Commit `2ec7d16`.

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `src/fase4/__init__.py` | — | Idéntico, blob por blob, al de `sv/f4-recuperabilidad` (`5811b85`). `git merge-tree` con esa rama no da conflictos |
| `src/fase4/regimen.py` | local | Análisis completo de F4.2: mapeos, pruebas, Holm, estratos de Lorenz, figuras y `resultados.json` |
| `src/fase4/reintentos_gs.py` | Kaggle (CPU) | Regenera las 908 imágenes de train de Gray-Scott con registro de reintentos y las verifica contra lo almacenado |
| `tests/test_fase4_regimen.py` | local | 31 pruebas sintéticas |
| `docs/correr_f42_kaggle.md` | — | Celdas de Kaggle |

### 1a. `regimen.py`

- **Mapeos.** Se toman de `src/generation`, sin reescribirlos:
  - `gray_scott_lib.umap_to_rd_params` → (f, k);
  - `lorenz_lib.map_umap_to_lorenz` → ρ;
  - `perlin_lib.map_umap_to_perlin` → `n_octaves`.
  - `u` conserva el dtype almacenado en el pkl, igual que al generar. Con `u` en float32, f y k difieren de la fórmula del plan en ~1e-8 por redondeo; la igualdad con las funciones del generador es exacta (prueba 1 y 2).
- **Datos.** Solo train para las distribuciones (908). Prueba se usa únicamente para el F1 por estrato de Lorenz, con las predicciones archivadas de F3.5 (`results/predictions/final/lorenz__vit_b16__frozen__seed{42,7,123,2024,99}.csv`).
  - ŷ es la media de `p_dep` de las 5 semillas > 0.5, con `src.protocol.contraste.yhat_mean` (§7).
  - Se registra el sha256 de cada CSV.
  - G1 se cumple (veredicto de la Fase 5 ya commiteado). No hay inferencia nueva (G2).
- **Lorenz.**
  - Estrato ρ < 24.74. Se cuentan las discordancias con u1 < 0.0925, en train y en prueba, y se reportan.
  - Fisher exacto sobre la tabla 2×2: filas depresivo / no depresivo, columnas evento / no evento.
  - Mann–Whitney bilateral de ρ, con U de la clase depresiva, la correlación biserial de rangos y las medianas.
- **Perlin.** Mann–Whitney bilateral de las octavas, más la tabla de octavas por clase.
- **Gray-Scott.**
  - Dispersión de (f, k) por clase, con medias y rangos. Sin prueba.
  - `pearson_superpuesto: false`: la figura `f42_01` va **sin** el diagrama de Pearson (PENDIENTE P7).
- **Reintentos.** Fisher sobre el evento de reintento por clase, leído del CSV de `reintentos_gs.py`. El CSV debe tener las 908 muestras en el orden de `sample_id` y las mismas etiquetas que el índice; si no, se detiene.
- **Holm.** La familia fija es m = 4: `lorenz_fisher_regimen`, `lorenz_mw_rho`, `perlin_mw_octavas` y `gray_scott_fisher_reintentos`, con α = 0.05.
  - El rechazo usa `src.protocol.verdict.holm`, el mismo de la Fase 5.
  - Además se reporta el p ajustado de Holm.
  - Una familia distinta de esos cuatro nombres detiene el análisis.
- **Salidas.** `results/f42_regimen/resultados.json` (meta con commit, compuerta y sha256 de los insumos) y `figures/f42_0{1,2,3}_*.png`. No se sobrescriben: si ya existen `resultados.json` o `figures/`, se detiene.
- **PENDIENTES.** La CLI exige `--resoluciones` con los cinco puntos de §5. Si falta alguno o trae un valor no previsto, se detiene.

### 1b. `reintentos_gs.py`

- **Ruta de cómputo.** La de producción, por CPU: `gray_scott_lib.generate_and_save_grayscale` → `run_gray_scott_safe` → `run_gray_scott`. La GPU no se usa.
- **Instrumentación sin cambiar la lógica.**
  - Durante cada muestra, el atributo de módulo `gray_scott_lib.run_gray_scott` se sustituye por un envoltorio. El envoltorio llama a la original con los mismos argumentos, devuelve su mismo resultado y anota los parámetros y `V.var()`. Al terminar se restaura.
  - Esto funciona porque `run_gray_scott_safe` busca `run_gray_scott` como global del módulo en cada llamada.
  - `reintentos` es el valor que ya devuelve `run_gray_scott_safe`, propagado hasta `generate_and_save_grayscale`. `usa_respaldo` es `reintentos == max_retries`; `max_retries` se lee de la firma, y vale 3.
- **Coherencia.** Se comprueba en cada muestra; si falla, se detiene:
  - número de llamadas = reintentos + 1, o 4 con respaldo;
  - con respaldo, la última llamada tiene los parámetros fijos (f = 0.035, k = 0.061, Dv = 0.06, radio 10, 5000 iteraciones);
  - sin respaldo, la última llamada tiene varianza > 1e-4.
- **u.** Se toma de `umap_params.pkl` (verificado contra el manifiesto) con el dtype almacenado.
- **Verificación por muestra.**
  - `sha256_coincide`: sha256 del PNG regenerado frente a `meta/data_manifest.csv`.
  - `pixeles_coinciden`: píxeles del regenerado frente al PNG almacenado.
  - `almacenada_integra`: sha256 del almacenado frente al manifiesto.
- **Paralelismo.** `ProcessPoolExecutor` con `--workers` (por omisión `os.cpu_count()`). Cada muestra es independiente y determinista: la semilla de ruido de `run_gray_scott` es fija.
- **Salida.** `reintentos_gs_train.csv` (una fila por muestra) y `reintentos_gs_manifest.json`.
  - El manifiesto guarda: commit, `smoke`, entorno (Python, numpy, Pillow), sha256 del pkl, `u_dtype`, resumen por clase, distribución de reintentos, listas de no coincidentes, tiempos con estimación para 908, y sha256 del CSV.
- **No sobrescribe.** Se detiene si el CSV o el manifiesto ya existen, o si `--regen-dir` no está vacío.
- **Datos.** Tampoco escribe dentro de `TT_DATA_ROOT` (`paths.assert_outside_data_root`).

### 1c. Adenda de F0.3

`tests/test_label_independence.py` se usa **sin cambios**: `-k "test_C_ and (perlin or lorenz)"` selecciona exactamente 4 casos (comprobado con `--collect-only`).
- `test_C_inventario_y_alineacion[perlin|lorenz]`;
- `test_C_reproduccion_desde_u[perlin|lorenz]`.

Configuración:
- `F03_N_PER_CELL=3`: 18 muestras por técnica, la misma muestra que en F0.3 con semilla 2026.
- `F03_STRICT=1`.

Los logs van a `results/f03_independencia_etiqueta/adenda_capa_C/`. `test_C_alineacion_con_pkl_41` no entra, porque no es de una técnica y necesita el PKL de Exp 4.1.

## 2. Pruebas sintéticas

`python -m pytest tests/test_fase4_regimen.py -v`: **31 passed en 66 s** (`results/f42_regimen/pytest_e2a.log`). No usan datos reales ni `TT_DATA_ROOT`.

| Grupo | Pruebas |
|---|---|
| Mapeos | Coinciden con las fórmulas del plan (tolerancia de float32) y son **idénticos** a las funciones de `src/generation`. Rangos de ρ (24–32) y octavas (1–8), y extremos u = 0 y u = 1 |
| Estratos de Lorenz | ρ < 24.74 equivale a u1 < 0.0925 lejos de la frontera. F1 por estrato contra `sklearn`. Clase ausente con `cero` / `no_definido` |
| Fisher | Contra `scipy.stats.fisher_exact` (las tres alternativas) y contra una suma hipergeométrica manual, en 4 tablas. Orientación de la tabla 2×2 |
| Mann–Whitney | Contra `scipy.stats.mannwhitneyu` y contra un cálculo manual: U por pares y p asintótico con corrección por empates, con y sin continuidad. Casos sin diferencia y con separación total |
| Holm | p ajustados a mano. Rechazo coherente con p_Holm ≤ α. Contra `statsmodels` (50 familias al azar). Detención si la familia no es la de F4.2 |
| Análisis completo | `analizar` con datos sintéticos: familia m = 4, conteos de reintentos y respaldo, muestras no idénticas, estratos. Detención si el registro no cuadra en filas o etiquetas |
| Instrumentación | Misma V e intento con y sin registro. **Mismo PNG (sha256)** por la ruta de producción con y sin registro (256², 2000 iteraciones). Detección de 3 reintentos y del respaldo con Dv y radio deterministas (0.08 → 0.068 → 0.0578; 4 → 8 → 12). Coherencia que detiene. Parámetros del respaldo iguales a los del código fuente |
| Verificación por muestra | `_trabajo` contra una «almacenada» idéntica (todo coincide) y contra otra con 1 píxel cambiado (no coincide y lo reporta) |
| No sobrescribir | `reintentos_gs.check_dirs` (CSV existente, carpeta de PNG no vacía) y `regimen.main` con `resultados.json` existente |
| Otros | Columnas del CSV, resumen por clase, muestreo de humo (16 espaciadas), figuras sintéticas (3 PNG, sin Pearson) |

**No se corrió con datos reales:** `regimen.main`, `reintentos_gs.main` y la capa C. La primera ejecución real es la celda 3 de Kaggle (prueba de humo).

## 3. Celdas de Kaggle

Detalle en `docs/correr_f42_kaggle.md`. Acelerador: ninguno (CPU). Internet activado. Input: `tt2-insumos`.

| Celda | Contenido |
|---|---|
| 1 | Clonar en `COMMIT = "2ec7d16"` (o el HEAD de `sv/f4-regimen` tras este reporte; el código es el mismo) y aplicar la compuerta `enmienda-5` |
| 2 | `TT_DATA_ROOT`, `pip install -r requirements-kaggle.txt noise==1.2.2`, `build_manifest --verify`, `nproc` y versiones de Pillow y numpy |
| 3 | Prueba de humo: 16 muestras en `/tmp`, con estimación de duración (`tiempo.estimacion_908_segundos`) |
| 4 | En la versión guardada: verificación, entorno, regeneración de las 908, capa C (Perlin y Lorenz) con junit y log, huellas y zip |

`f42_resultados.zip` contiene `results/f42_regimen/reintentos/` y `results/f03_independencia_etiqueta/adenda_capa_C/`. Las 908 PNG regeneradas no se incluyen.

**Rama:** hay que hacer *push* de `sv/f4-regimen` antes de la celda 1.

## 4. Duración estimada

- **Medición local** (Windows, 1 núcleo, `u` sintético): `run_gray_scott` en 256² tarda ≈ 3.8 ms por iteración.

  | n_iter | Duración |
  |---|---|
  | 2000 | 7.5 s |
  | 4500 | 17.8 s |
  | 7000 | 26.6 s |

- **Regeneración de 908** con n_iter medio de ≈ 4500 (u4 ≈ 0.5): ≈ 4.5 h de un núcleo, más los reintentos.
  - En local, con 8 núcleos, serían ≈ 35 min. Eso supera los 20 min, así que va a Kaggle por `CLAUDE.md`.
  - En Kaggle CPU, con 4 vCPU: ≈ 1.1 h si escala de forma lineal, ≈ 2 h si son 2 núcleos físicos con hilos.
- **Capa C:** ≈ 12–15 min. Lorenz tardó 703.52 s en Colab con la misma muestra (F0.3, corrida 4).
- **Total: ≈ 1.5–2.5 h.** La celda 3 da la cifra real. Si pasa de 4 h, hay que detenerse.

## 5. PENDIENTES (sin resolver)

Los puntos P1 a P5 son las claves de `--resoluciones`. `regimen.py` no corre sin ellos.

| # | Clave | Qué no fija el plan | Opciones | Nota |
|---|---|---|---|---|
| P1 | `fisher_alternativa` | Lateralidad de los dos Fisher (régimen de Lorenz y reintentos). El plan dice «bilateral» solo para Mann–Whitney | `two-sided`, `greater`, `less` | F0.3 usó Fisher bilateral (bitácora F0.3 §4.4) |
| P2 | `mw_continuidad` | Corrección de continuidad del Mann–Whitney asintótico | `true` (por omisión en scipy), `false` | El método se fija como asintótico porque con n = 908 y empates (octavas enteras) el exacto no aplica. Si SV prefiere otro método, es otro PENDIENTE |
| P3 | `reintento_evento` | Qué es «reintento» en la tabla 2×2 del Fisher | `reintentos_mayor_0` (≥ 1 reintento, incluye el respaldo), `usa_respaldo` | El CSV guarda los dos, más la distribución 0/1/2/3 |
| P4 | `f1_estrato_sin_clase` | F1-Macro de un estrato de Lorenz en el que una clase no aparece ni en y ni en ŷ (0/0) | `cero` (convención de `src/protocol/bootstrap.py`), `no_definido` | El estrato ρ < 24.74 de prueba puede ser pequeño (el plan lo advierte). Se reportan siempre n, n por clase y errores |
| P5 | `bit_a_bit` | Criterio de «idéntica bit a bit» | `sha256_archivo` (PNG frente al manifiesto, lo pedido), `pixeles` | Si la versión de Pillow de Kaggle codifica distinto, el sha256 puede diferir con los píxeles iguales. Se registran ambos, y la celda 3 lo muestra antes de la corrida larga |
| P6 | — | Qué hacer en el Fisher de reintentos con las muestras no idénticas | Incluirlas y listarlas (lo implementado), o excluirlas | El plan dice que «la tasa de reintentos de esas muestras queda en duda». Hoy se incluyen y se listan en `bit_a_bit.no_identicas`. Cambiarlo requiere una resolución y código |
| P7 | — | Fronteras del diagrama de Pearson (1993) | Versionar una fuente de las fronteras (tabla digitalizada con cita) y superponerlas; o dejar la figura sin superposición | En el repositorio no hay fuente versionada (búsqueda de «Pearson» en todo el repo: solo menciones en planes y reportes). **No se dibujaron fronteras de memoria.** `f42_01` lleva la nota en el pie y `pearson_superpuesto: false` |

Elecciones técnicas que no son PENDIENTES. SV puede objetarlas:
- α = 0.05 para Holm (protocolo).
- El estrato de Lorenz se define con ρ calculado por `map_umap_to_lorenz`, y la discordancia con u1 < 0.0925 se reporta en lugar de imponerse.
- La prueba de humo usa 16 muestras espaciadas de forma uniforme en el orden de `sample_id`.

## 6. Siguiente paso (después de Kaggle)

1. Integrar `f42_resultados.zip`. Verificar con `docs/correr_f42_kaggle.md`, «Verificación local», y hacer commit.
2. SV resuelve P1 a P7. Con el JSON de resoluciones versionado y el árbol limpio, correr `python -m src.fase4.regimen --resoluciones <json>`.
3. Bitácoras: F4.2, y la adenda de F0.3 con el log de la capa C.
