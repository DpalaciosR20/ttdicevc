# Reporte E1a — F4.1 (recuperabilidad de u) y F4.7 (correspondencia geométrica): código, pruebas y celdas de Kaggle

Rama `sv/f4-recuperabilidad`, creada desde `main` (`5af17a8`). Las celdas de Kaggle se ejecutan con el commit **`9f3b7b016bcb24d41b58d46ffdb8af7f26267295`**, que contiene el código, las pruebas y `docs/correr_f41_kaggle.md`.

En esta etapa:
- no se usaron datos reales de prueba;
- no se ejecutó `run_fase5`;
- no se modificó `results/f5_contraste/`;
- no hubo inferencia de ningún finalista ni control.

## 0. Compuerta

| Comprobación | Resultado |
|---|---|
| `git merge-base --is-ancestor enmienda-5 HEAD` | 0 |
| `git diff --quiet enmienda-5 HEAD -- docs/plan_fase4.md` | 0 (sin cambios) |
| `results/f5_contraste/resultados.json` versionado | sí (último commit `e02104f`) |
| `results/f5_contraste/resultados.json` sin cambios frente a HEAD | sí |

Nota: `main` no contiene el commit del notebook de figuras de la Fase 5 (`83c5edd`), que sigue solo en `sv/f5-resultados`, sin PR. Esta rama no lo necesita.

## 1. Código

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `src/fase4/caracteristicas.py` | Kaggle (GPU) | Penúltima capa de las 1,136 imágenes de cada técnica, detallada abajo |
| `src/fase4/recuperabilidad.py` | Local | F4.1, detallada abajo |
| `src/fase4/geometria.py` | Local | F4.7, detallada abajo |
| `tests/test_fase4_recuperabilidad.py` | Local | 31 pruebas sintéticas |
| `docs/correr_f41_kaggle.md` | — | Celdas de Kaggle |
| `.gitignore` | — | Excluye `f41_caracteristicas/` y los `.npz` bajo `results/f41_recuperabilidad/` |

**`caracteristicas.py`**
- Penúltima capa de las 1,136 imágenes de cada técnica, con la cabeza reemplazada por `nn.Identity`: sin logits ni `p_dep` (G2).
- Preprocesamiento de inferencia de F1.4 tomado del JSON de la corrida, sin aumentos.
- Perlin y Lorenz: ViT-B/16 ImageNet. Comprueba con `torch.equal`, tensor por tensor salvo `heads.head.*`, que el *backbone* de cada finalista (las 5 semillas; la 42 es la que exige el plan) es idéntico al de ImageNet. Si no lo es, se detiene.
- Gray-Scott: las 5 semillas ajustadas y el *backbone* ImageNet sin ajustar.
- Verifica el sha256 de cada `.pt` contra `meta/artefactos_manifest.csv` y contra `model_file.sha256` del JSON.
- Verifica el sha256 de cada imagen y el `dir_digest` contra `meta/data_manifest.csv`. Los bytes hasheados son los mismos que se decodifican.
- Escribe un `.npz` por (técnica, *backbone*) con `features` (float32), `sample_id` y `split`, más un manifiesto CSV con sha256 y un JSON de metadatos.
- Nunca sobrescribe.

**`recuperabilidad.py`**
- Ridge multisalida con alpha ∈ `logspace(-3, 3, 13)`, elegido por el R² medio en val, con predicciones recortadas a [0, 1].
- `r2_score(raw_values)` en val y en prueba.
- **û** de train por *cross-fitting* (`KFold(5, shuffle=True, random_state=42)`); **û** de val y prueba con el modelo de todo train.
- LogReg de §4.3, tomada de `configs/protocolo.json:techo_5d`: 26 combinaciones, `StandardScaler` en train, F1 de val, empate a 1e-12 → la anterior en la rejilla, `max_iter = 5000` como en F2.1. Se aplica sobre **û** y sobre **u**.
- La LogReg sobre **u** se contrasta con `umap5d__logreg` de F2.1 (diferencia máxima de `p_dep`).
- ΔF1 = F1(**u**) − F1(**û**) con IC percentil bilateral al 95 % sobre las réplicas de A1. Antes verifica `idx_sha256` contra `resultados.json:meta` de la Fase 5.
- Condición 2 de F4.6 por técnica, con el *backbone* principal.
- Dos etapas:
  - `--val`: train y val.
  - `--prueba`: una sola vez. Exige la compuerta `enmienda-5`, el plan sin cambios, `resultados.json` y `resultados_val.json` versionados y sin cambios, y el árbol limpio. Recalcula val y exige que coincida con la versión commiteada. Se niega si existe `resultados_prueba.json`.

**`geometria.py`**
- F4.7 sobre train (n = 908): euclídea en **u** y coseno en las características.
- ρ de Spearman entre los pares i < j y Mantel con 9,999 permutaciones, `default_rng(2026)`.
- Holm (m = 3) sobre los *backbones* principales.
- El Mantel es exacto y vectorizado: los rangos se calculan una vez y cada permutación reordena la matriz de rangos.
- Nunca sobrescribe.

### Pruebas sintéticas (`tests/test_fase4_recuperabilidad.py`, 31)

| Requisito | Pruebas |
|---|---|
| Ridge recupera u conocida (R² ≈ 1) y R² ≈ 0 con ruido | `test_ridge_recupera_u_conocida[ninguno, standard_train]` (R² mínimo > 0.999); `test_ridge_ruido_r2_cercano_a_cero` (\|R²\| < 0.1); `test_alphas_del_plan` |
| *Cross-fitting* sin fuga | `test_crossfit_sin_fuga`: un estimador espía registra las filas de ajuste y falla si predice una que vio. Cada modelo vio exactamente los otros 4 pliegues, que son los de `KFold(5, shuffle=True, random_state=42)` |
| Recorte a [0, 1] | `test_recorte_a_0_1`, en la predicción directa y en el *cross-fitting* |
| ΔF1 e IC con los índices de A1 | `test_delta_f1_e_ic_con_indices_de_A1`: observado y 5 réplicas iguales a `sklearn` sobre las muestras remuestreadas; IC igual a `np.quantile(..., [0.025, 0.975], 'linear')`. `test_idx_sha256_igual_al_de_la_fase_5`: los índices reproducen el `idx_sha256` de la Fase 5 y una permutación de ellos falla |
| Mantel contra una implementación de referencia | `test_mantel_igual_a_implementacion_de_referencia[0, 1, 2]` y `test_mantel_con_empates_en_distancias`: ρ, conteo k y p iguales a una implementación directa con `scipy.stats.spearmanr` y las mismas permutaciones. No hay scikit-bio instalado |
| p ≈ uniforme bajo H0 | `test_mantel_p_uniforme_bajo_H0`: 300 simulaciones; KS frente a uniforme, media de p en (0.42, 0.58) y tasa de p ≤ 0.05 en (0.02, 0.10). `test_mantel_estructura_perfecta`: ρ = 1, p = 1/(B + 1) |
| El script de prueba se niega a sobrescribir | `test_prueba_se_niega_a_sobrescribir`, `test_val_se_niega_a_sobrescribir`, `test_write_json_nunca_sobrescribe`, `test_npz_nunca_sobrescribe`, `test_compuerta_de_prueba_exige_val_versionado` |
| Extracción | `test_penultima_capa_efficientnet` (1280; igual a `avgpool(features)`); `test_penultima_capa_vit` (768; igual al token de clase del encoder), ambos con pesos aleatorios; `test_compare_backbone_identico_salvo_cabeza`; `test_dir_digest_igual_a_build_manifest`; `test_dir_digest_real_de_data_manifest` (las tres filas `kind=dir` se reproducen); `test_imagenes_con_hash_distinto_se_detienen`; `test_modelos_y_backbones_del_plan`; `test_load_index_orden_y_tamanos` |
| PENDIENTES | `test_resoluciones_pendientes_detienen`, `test_empate_de_alpha_no_se_resuelve_solo` |

### pytest

| Ejecución | Resultado | Evidencia |
|---|---|---|
| `tests/test_fase4_recuperabilidad.py` | 31 aprobadas (9.1 s) | — |
| `pytest tests/` (con `CANON_UMAP_PARAMS` y `CANON_STRICT=1`; sin `F03_*`) | **1158 aprobadas, 30 omitidas, 0 fallidas** (175.06 s). Las 30 omitidas son las de `test_label_independence.py`, que requieren `F03_UMAP_PARAMS`, como en los bloques anteriores | `results/f41_recuperabilidad/pytest_e1a.log`, `pytest_e1a.xml` |

## 2. Celdas de Kaggle

Están en `docs/correr_f41_kaggle.md`.

**Configuración del notebook:**
- **Acelerador:** GPU T4.
- **Internet:** activado, para los pesos ImageNet de torchvision; su sha256 se registra.
- **Inputs:** `tt2-insumos`, `tt2-artefactos` y `tt2-artefactos-f36`. Los modelos de F3.6 no se usan.
- **`COMMIT`:** `9f3b7b016bcb24d41b58d46ffdb8af7f26267295`.

**Celdas:**

| Celda | Contenido |
|---|---|
| 1 | Clonado; la última línea comprueba `enmienda-5` |
| 2 | `TT_DATA_ROOT` anidado, `build_manifest --verify`, `verify_artefacts --task F3.5` |
| 3 | Prueba de humo con `--limit 16` en `/tmp` y estimación de duración |
| 4 | **Una sola celda, en la versión guardada:** verificación de datos y modelos, entorno, extracción, `sha256sum` de los `.npz` y los dos zips |

**Duración y salida:**
- **Duración estimada:** 5–15 min de T4 para la celda 4. La celda 3 da la estimación real; si pasa de 30 min, hay que detenerse.
- **Contenido esperado:**
  - `f41_resultados.zip` (se versiona): manifiesto con 8 filas, metadatos con la identidad ViT/ImageNet y los `dir_digest`, `modelos_verificacion.json` (15/15), entorno, log y `npz_sha256sum.txt`.
  - `f41_caracteristicas_npz.zip` (no se versiona; unos 30–40 MB): los 8 `.npz`. Va a Drive, en `MiUnidad/tt2-artefactos/f41_caracteristicas/`, y a un *dataset* de Kaggle.
- **Verificación local al recibirlos:** la del final de `docs/correr_f41_kaggle.md`.

## 3. Puntos del plan que no resultan computables — PENDIENTES (sin resolver)

`docs/plan_fase4.md` no fija estos detalles y el código no los decide. `recuperabilidad.py` y `geometria.py` exigen `--resoluciones <json>` con todos ellos; si falta alguno, se detienen con «PENDIENTE sin resolver». Las pruebas usan valores explícitos solo como datos sintéticos.

| ID | Análisis | Detalle no fijado | Opciones implementadas |
|---|---|---|---|
| **PENDIENTE E1-1** | F4.1 | Escalado de las características antes de Ridge: el plan dice «Ridge multisalida», sin escalado. Con 768 o 1280 dimensiones, el escalado cambia el significado de alpha | `ridge_escalado`: `ninguno` (Ridge directo) · `standard_train` (StandardScaler ajustado en train + Ridge) |
| **PENDIENTE E1-2** | F4.1 | Desempate de alpha cuando dos valores empatan en R² medio de val (tolerancia 1e-12) | `alpha_desempate`: `menor` · `mayor` · `detener` |
| **PENDIENTE E1-3** | F4.1 | alpha de los 5 pliegues del *cross-fitting*: el plan elige alpha en val y pide el *cross-fitting*, sin decir si se reutiliza el elegido | `alpha_crossfit`: `seleccionado` (el único implementado; otra regla requeriría especificarla) |
| **PENDIENTE E1-4** | F4.7 | Lado del Mantel: el plan dice «prueba de Mantel», sin hipótesis alternativa | `mantel_lado`: `mayor` (H₁: ρ > 0) · `bilateral` |
| **PENDIENTE E1-5** | F4.7 | Fórmula del valor p | `mantel_p`: `k_mas_1_sobre_B_mas_1`, es decir (k + 1)/(9,999 + 1) · `k_sobre_B` |
| **PENDIENTE E1-6** | F4.7 | Uso de `default_rng(2026)` entre las tres técnicas | `mantel_rng`: `por_tecnica` (generador nuevo con semilla 2026 en cada técnica) · `continuo` (un solo generador, técnicas en el orden gray_scott, perlin, lorenz) |
| **PENDIENTE E1-7** | F4.7 | Mantel en los *backbones* de sensibilidad de Gray-Scott: el plan dice «semilla 42 principal, como en F4.1». Duración local medida: unos 3.4 min por *backbone* (50 permutaciones en 1.02 s con n = 908). Con `true` son 8 *backbones*, unos 27 min: por la regla de los 20 min iría a Kaggle (CPU). Con `false` son 3, unos 10 min, en local | `mantel_sensibilidad`: `true` (descriptivo, fuera de Holm) · `false` |

Plantilla de `--resoluciones` (valores `null` = PENDIENTE). Puede ser un solo archivo para los dos scripts:
```json
{
  "ridge_escalado": null,
  "alpha_desempate": null,
  "alpha_crossfit": null,
  "mantel_lado": null,
  "mantel_p": null,
  "mantel_rng": null,
  "mantel_sensibilidad": null
}
```

### Detalles fijados por el texto o por convenciones ya vigentes

No son PENDIENTES; se declaran para que SV los revise.
- Recorte a [0, 1] también en la selección de alpha y en el *cross-fitting*. El plan dice «predicciones recortadas», sin excepciones.
- La LogReg sobre **û** se ajusta con **û** de train por *cross-fitting* y se selecciona con **û** de val, del modelo de todo train.
- La LogReg sobre **u** usa `max_iter = 5000`, que §4.3 no fija; es el mismo valor de F2.1, F2.2 y F4.5. Con él se busca reproducir `umap5d__logreg` y se reporta la diferencia máxima de `p_dep` en val y en prueba.
- IC de ΔF1: percentil bilateral 2.5–97.5 con cuantiles `linear` (`enmienda_4.ic.descriptivo`, la convención de A5). Las réplicas son las de A1 (`idx_sha256` verificado).
- La condición 2 de F4.6 se evalúa con el *backbone* principal de cada técnica. Los de sensibilidad de Gray-Scott también reciben ΔF1 e IC, como descriptivo.
- Para el *backbone* ImageNet de Gray-Scott (sensibilidad) se usa el preprocesamiento de inferencia del finalista de Gray-Scott.
- La extracción en GPU puede diferir de CPU en el orden de 1e-6 (F0.3, H2). El manifiesto registra el dispositivo y los análisis locales usan siempre los `.npz` publicados, con su sha256 verificado.

## 4. Estado

**DETENIDO aquí, como se pidió.** Falta:
1. Que SV ejecute Kaggle y entregue `f41_resultados.zip` y los `.npz`.
2. Que SV resuelva las PENDIENTES E1-1 a E1-7 en un archivo de resoluciones.
3. Después vienen la verificación local, `--val` y su commit, `--prueba` (una vez), `geometria`, las bitácoras de F4.1 y F4.7, y el cierre de F4.6.
