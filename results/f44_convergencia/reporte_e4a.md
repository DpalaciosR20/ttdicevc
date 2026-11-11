# F4.4 y F4.8, etapa 4a: reporte (código y pruebas sintéticas; detenido)

> Rama `sv/f4-cierre` (desde `main` en `c2e183b`, que incluye F4.3) · Rol: SV · Sin interpretación

## 0. Compuerta

| Comprobación | Resultado |
|---|---|
| `enmienda-5` es ancestro de HEAD | sí |
| `docs/plan_fase4.md` sin cambios frente a `enmienda-5` | sí |

La primera vez la rama salió de `b0bd07e`, que no incluía F4.3. Por indicación de SV, se esperó a la fusión del PR #42 y la rama se recreó desde `c2e183b`.

## 1. Código

### `src/fase4/convergencia.py` (F4.4)

**Descriptores.**
- Los 8 descriptores significativos se leen de `results/f24_descriptores/tabla11_recalculada.csv` (`sig_holm = True`). El script se detiene si no son exactamente los del plan:
  - Gray-Scott: entropía, contraste, densidad de bordes y energía LBP;
  - Perlin: contraste y densidad de bordes;
  - Lorenz: entropía y energía LBP.
- El descriptor por parche usa **las funciones de F2.4 sin reescribirlas.** `f24()` carga `results/f24_descriptores/f24_descriptores.py`, copia literal de la celda 20, y `funcion_descriptor` devuelve sus funciones `shannon_entropy`, `contrast`, `edge_density` y `lbp_energy`. Se aplican a cada recorte uint8.
- Rejilla: 7 × 7 en Gray-Scott y 14 × 14 en Perlin y Lorenz.
  - Fronteras: `round(i·n/g)`, exactas si g divide n.
  - La imagen del parche depende de P1: la nativa de 256 px, o la redimensionada a 224 con el `Resize` de F1.4 (`redimensionar_224`).

**Masa del mapa.** Sale de los `.npz` de F3.7, con el mismo código que F4.3:
- `Ubicador` verifica bytes y sha256 contra el manifiesto, contra el registro de F3.7 y la exclusión de `f37_humo/`;
- `cargar_npz` comprueba el orden de `sample_id` y el `y_true`. Solo se usa el split `val`.
- Se toma `map_raw · map_raw_scale`, se aplica `escalar_semillas(…, 'dividir_media_semilla')` y luego `media_entre_semillas`. La masa es la suma por parche (`masa_por_parche`).

**Imágenes.** Solo de val, con `filas_val`, `nombres_val` y `leer_imagenes_val` de `atribucion_gpu` (G2). El sha256 de cada imagen se verifica contra `data_manifest.csv`, y el orden de muestras de imágenes y mapas tiene que coincidir.

**Estadística.**
- Spearman por muestra con rangos promedio (`spearman_por_muestra`); da NaN si el descriptor o la masa es constante.
- Media sobre val e IC percentil al 95 % por *bootstrap* sobre muestras: B = 10,000, `default_rng(2026)` (`ic_bootstrap`).
- Los NaN se tratan según P2. Las réplicas se generan según P4 y los cuantiles según P5.

**Sin cálculos nuevos:**
- `tabla_cualitativa`: ΔF1 de la ablación por banda y σ (`f43.<t>.ablacion.*`), F1 base y descriptores significativos con su d y |d| de Cohen, ordenados por |d|. La naturaleza de cada descriptor depende de P6.
- `contexto_a11`: `f5.frente_a_descriptores.<t>` con su IC, solo citado.
- `advertencias`: `f43.<t>.lectura_vit.*`. Lorenz queda como `dependiente_de_la_cabeza_sin_validacion_por_datos`, y `resultados.json` lleva esa advertencia.

**Ejecución.**
- `main`: `--root --resoluciones [--out] [--limit]`.
- Sin `--limit`, exige la compuerta de `regimen.check_gate` (árbol limpio, F5 versionado).
- No sobrescribe.
- Salidas: `resultados.json`, `spearman_por_muestra.csv` y `parches.npz` (descriptor y masa por parche).

### `src/fase4/figura_ilustrativa.py` (F4.8)

- **LogReg `umap5d__logreg`.** Se reproduce con el código de F2.1 (`fixed_params`, `make_model` y `p_dep_of` de `results/f21_techo5d/f21_techo5d.py`) y con el config `techo_5d`:
  - la combinación se lee de `results/f21_techo5d/seleccion.csv` y el script se detiene si no es C = 0.1, `balanced`;
  - lleva StandardScaler ajustado en train;
  - sus `p_dep` de **val** se verifican contra `results/predictions/referencia/umap5d__logreg__none__seed42.csv`.
  - Prueba no se usa: `load_u` se llama solo con train y val.
- **Recorridos.**
  - Centroides de clase en train y 7 puntos lineales entre ellos, extremos incluidos.
  - ±1.5 DE sobre la dirección unitaria de w desde la media de train (P7–P9).
  - Recorte a [0, 1], con la máscara de las componentes recortadas.
- **Renderizado** con `src/generation`:
  - Gray-Scott: `generate_and_save_grayscale`, ruta CPU con `run_gray_scott_safe`;
  - Perlin: `generate_perlin_image_classblind`;
  - Lorenz: `map_umap_to_lorenz`, `solve_attractor` y `render_attractor_enriched`.
  - Todas en escala de grises. La etiqueta nunca entra en el renderizado: en Gray-Scott el argumento `label` solo nombra el archivo temporal.
- **Figuras.** Una en grises y otra con paleta (P11), solo ilustrativa; los puntos recortados se marcan según P12.
- **Salidas** en `results/f48_figura/`: `recorridos.json`, `imagenes/*.png`, `f48_direccion_grises.png` y `f48_direccion_paleta_ilustrativa.png`. Sin métricas.

## 2. Pruebas

`tests/test_fase4_convergencia.py`: 31 pruebas, sintéticas salvo las que leen archivos versionados del repositorio (Tabla 11, `canonical_numbers.json`, scripts de F2.1 y F2.4).

| Grupo | Qué comprueba |
|---|---|
| Parches | `limites` exactos en 224/7 y 224/14, y proporcionales en 256 (36–37 y 18–19 px). **El descriptor por parche es la función de F2.4 aplicada al recorte**: 4 descriptores × rejillas 7 y 14, y `funcion_descriptor` es el objeto de F2.4. Exige uint8. `redimensionar_224` da 224 × 224 uint8 |
| Masa | La suma por parche conserva la masa total y coincide con el bloque |
| Tabla 11 | Los 8 significativos coinciden con el plan, y el script se detiene si el CSV cambia |
| Spearman | Igual a `scipy.stats.spearmanr` por muestra, con empates; NaN si una entrada es constante; ±1 con a y −a |
| IC | Constante → IC degenerado; determinista con `default_rng(2026)`; percentil `linear`; los NaN se excluyen en cada réplica |
| `analizar` | Signos (+1, −1, ~0), conteo de indefinidos, modo `cero`. Con réplicas compartidas, dos descriptores idénticos dan el mismo IC |
| Contexto | La tabla cualitativa toma exactamente `f43.*`; A11 coincide con `f5.frente_a_descriptores.*`; la lectura de Lorenz es la de F4.3 |
| Recorridos | Centroides, 7 pasos lineales con extremos, ±1.5 DE simétrico desde la media, DE por proyección y media de componentes, recorte y máscara |
| LogReg | C = 0.1, `balanced`; la dirección en el espacio de u es w/σ unitaria; `p_dep` crece a lo largo del recorrido; la verificación de `p_dep` ignora prueba y se detiene fuera de tolerancia |
| Renderizado | Las 3 técnicas son deterministas: dos llamadas dan la misma imagen uint8 de 256 × 256, no constante |
| Resoluciones | Faltantes, sobrantes y valores no previstos detienen |

**Suite completa.** `python -m pytest tests` con `CANON_STRICT=1` y `CANON_UMAP_PARAMS`: 1292 passed, 30 skipped (`results/f44_convergencia/pytest_e4a.log`). Antes eran 1261 passed (E3b); la diferencia son las 31 nuevas.

## 3. Duración estimada (prueba acotada en local)

**F4.4.**
- `python -m src.fase4.convergencia --limit 8` con resoluciones temporales: la primera opción de cada pendiente, solo para medir.
- Se ejecutó sin compuerta (`git_commit = prueba_acotada`) y con salida en el *scratchpad*: no se versiona ni se consultaron sus valores.

| Parte | Medido (8 muestras) | Extrapolado (114) |
|---|---|---|
| Lectura y verificación de los 15 `.npz` de F3.7 desde Drive (≈ 680 MB; la lectura es completa aunque se use solo val) | 64.2 s | 64 s con caché; ≈ 4 min en frío (≈ 3 MiB/s, como en F4.3) |
| Imágenes de val (sha256) y descriptores por parche | 9.1 s | ≈ 2.2 min |
| Bootstrap (8 combinaciones, B = 10,000) | < 0.1 s | < 10 s |
| `find_all` sobre la copia de Drive | — | ≈ 21 s (medido en F4.3) |
| **Total** | 73.9 s | **≈ 4–7 min** |

**F4.8.** Cuatro puntos caros medidos (un renderizado cada uno):

| Técnica | u | Segundos |
|---|---|---|
| Gray-Scott | u4 = 0.97 | 8.7 |
| Gray-Scott | u4 = 0.07 | 2.4 |
| Lorenz | — | 5.9–6.1 |
| Perlin | — | 0.2 |

Con 7 + 7 imágenes por técnica (o 7 + 9 según P9), el total es **≈ 3–4 min**.

**Las dos partes quedan bajo 20 min y no requieren GPU,** así que se ejecutan en local. No hacen falta celdas de Kaggle.

**Verificación ya hecha con datos reales (sin resultados de F4.8).** Durante el desarrollo se reprodujo la LogReg de F2.1 sobre u de train. La diferencia máxima de `p_dep` en val frente al CSV de F2.1 es **0.0**. Con las opciones `u` y `proyeccion_train`, ninguna componente del recorrido de ±1.5 DE sale de [0, 1].

## PENDIENTES (sin resolver)

`convergencia.py` y `figura_ilustrativa.py` no corren sin un archivo de resoluciones con todas sus claves. Se propone versionar los dos archivos antes de ejecutar:
- `results/f44_convergencia/resoluciones.json` (P1–P6);
- `results/f48_figura/resoluciones.json` (P7–P12).

La primera opción de cada lista es la que el código trae primero; no es una recomendación.

### F4.4

| ID | Clave | Opciones | Qué decide |
|---|---|---|---|
| P1 | `geometria_parche` | `nativa_256_proporcional` · `redimensionada_224` | **Nativa:** el descriptor se calcula como en F2.4, sobre la imagen de 256 px, pero los parches miden 36–37 px (7 × 7) o 18–19 px (14 × 14) y su frontera no cae exactamente en la del parche de 32 o 16 px del mapa (desfase ≤ 0.5 px nativo). **224:** los parches coinciden exactamente con los del mapa, pero el descriptor se calcula sobre una imagen redimensionada, distinta de la que define F2.4 |
| P2 | `spearman_indefinido` | `excluir_y_contar` · `cero` | Muestras con masa constante (mapas constantes en las 5 semillas, posible en Gray-Scott) o con el descriptor constante en todos los parches (posible en el fondo de Lorenz). Se excluyen y se cuentan, o cuentan como 0 |
| P3 | `mapa` | `map` · `map_y_backbone` | Solo el mapa principal, o también la atención del *backbone* en los ViT (idéntica entre semillas; F4.3 la reporta aparte) |
| P4 | `bootstrap_rng` | `por_tecnica_compartido` · `por_tecnica_descriptor` | Una matriz de índices (B × 114) de `default_rng(2026)` por técnica, común a sus descriptores, o un generador nuevo por (técnica, mapa, descriptor). En esta implementación las dos opciones producen las mismas réplicas: los índices siempre se toman sobre las N = 114 muestras, y los NaN se excluyen dentro de cada réplica. La clave solo deja constancia |
| P5 | `ic_cuantiles` | `linear` · `inverted_cdf` | Método de `np.quantile` del IC percentil |
| P6 | `naturaleza_descriptores` | `propuesta` · `sin_clasificar` | Columna de la tabla cualitativa. La propuesta es la de `NATURALEZA` en el código: entropía, histograma sin estructura espacial; contraste, dispersión global dominada por frecuencias bajas; bordes, frecuencias altas; LBP, textura local de radio 3 (frecuencias altas y medias). Con `sin_clasificar` solo van el d y la ablación |

**Consecuencias de «la función de F2.4 aplicada al recorte»** (no son opciones; se piden para dejar constancia):
- **N1.** `edge_density` usa el umbral adaptativo del propio parche (media + 1 DE de su Sobel), no el de la imagen completa. Sobel se calcula sobre el recorte, así que hay efecto de borde en el perímetro de cada parche.
- **N2.** `lbp_energy` (P = 24, R = 3) en parches de 16–19 px: una parte importante de los píxeles está a menos de 3 px del borde del recorte.
- **N3.** La entropía usa 256 bins sobre 256–1,369 píxeles por parche, según la rejilla.

### F4.8

| ID | Clave | Opciones | Qué decide |
|---|---|---|---|
| P7 | `direccion_espacio` | `u` · `estandarizado` | La LogReg se ajusta sobre u estandarizado. `u`: el recorrido usa w_std/σ normalizado, que es la normal de la frontera en el espacio de u. `estandarizado`: w_std normalizado, aplicado directamente en u |
| P8 | `de_recorrido` | `proyeccion_train` · `media_de_componentes` | La DE de ±1.5 DE: la de la proyección de u de train sobre ŵ, o la media de las DE por componente |
| P9 | `n_pasos_recorrido` | `7` · `5` · `9` | Puntos del recorrido de ±1.5 DE. El plan fija 7 solo para el de centroides |
| P10 | `tol_pdep` | `1e-9` · `1e-6` | Tolerancia de la verificación contra el CSV de F2.1 (la diferencia observada es 0.0) |
| P11 | `paleta` | `viridis` · `magma` · `ninguna` | Paleta de la versión ilustrativa |
| P12 | `marca_recorte` | `borde_y_lista` · `solo_lista` | Borde rojo en la figura, o solo la máscara en `recorridos.json` |

### Del alcance

- **P13. Lorenz en local.** La adenda de F0.3 (capa C) registró imágenes de Lorenz que no se reproducen entre entornos (11/18), y su investigación está pospuesta.
  - F4.8 renderiza Lorenz en este equipo, así que sus imágenes pueden no coincidir píxel a píxel con las de la ruta original.
  - La figura no tiene peso argumentativo. Hay que decidir si se acepta con una nota en la bitácora.
- **P14. Cifras canónicas de F4.4.** Falta decidir si el cierre agrega claves `f44.*`: Spearman medio e IC por técnica, descriptor y mapa, y n de indefinidos. F4.8 no tiene métricas.
- **P15. Lectura de los ViT en la redacción.**
  - Lorenz: el plan dice «nada si F4.3 declaró el mapa no interpretable (atención del *backbone*)». F4.3 no lo declaró atención del *backbone*, sino «dependiente de la cabeza, sin validación por datos». La convergencia de Lorenz se reportaría con esa advertencia, según la instrucción de E4a. Falta confirmar que la fila del plan no excluye Lorenz.
  - Perlin: sin advertencia (`cordura_por_datos_superada`).

## Estado

- Código, pruebas y este reporte están en la rama `sv/f4-cierre`, con commit local y **sin push**.
- No se ejecutó F4.4 ni F4.8 con datos reales más allá de la prueba acotada (8 muestras, no versionada) y la verificación de la LogReg.
- Siguiente paso: SV fija P1–P15. Después se versionan las resoluciones, se corre en local con el árbol limpio y se cierra.
