# Cierre del bloque D1 — Enmienda 5 (P1–P5 y tolerancia de A3) y compuerta `enmienda-5`

> Rol: SV · Fecha: 2026-11-07 · Rama `sv/f5-ejecucion`.
> **No se ejecutó `--ejecutar-prueba` ni `--analisis-f46`.** Con datos reales solo se corrió `--verificar-insumos`, que comprueba la alineación sin cargar `p_dep`.
> No se fusionó nada ni se creó ninguna etiqueta. `src/protocol/verdict.py` no cambió.

## Commits

| # | Hash | Asunto |
|---|---|---|
| 1 | `270d5ac` | `[F5, Enmienda 5] Resolución de P1–P5 y tolerancia de A3` (protocolo, `configs/protocolo.json`, `tests/test_protocolo.py`) |
| 2 | (el commit que añade este reporte) | `[F5] Pipeline ajustado a la Enmienda 5 y pruebas sintéticas` |

**PR:** ver la sección final.

## 1. Enmienda 5 (`docs/protocolo_preregistrado.md`)

Tiene el formato de la Enmienda 4: autor, momento, motivo, regla de resolución, declaración de exposición, alcance, relación con el §11 y cambios.
- La tabla de cambios contiene P1, P2, P3, P4, P5, A3 y P6, con el texto pedido.
- **Declaración de exposición.** Repite la de la Enmienda 4 y añade un hecho posterior a esa enmienda: el F1 de **validación** del clasificador de longitud (F4.5) ya se conoce. Ninguna métrica de prueba de las referencias se ha calculado.
- **Relación con el §11.** Es la misma desviación declarada; P1 y P2 entran en el compromiso de robustez.

## 2. `configs/protocolo.json` → `enmienda_5`

Va como último bloque, con:
- `_seccion`, `_autor`, `_fuente`, `_momento`;
- `relacion_s11 = {desviacion_declarada, misma_que: "enmienda_4", compromiso_robustez, filas_nuevas_robustez: ["P1", "P2"]}`;
- `resoluciones` (P1…P6 → clave);
- `por_semilla.sensibilidad.comparacion = "por_replica"`;
- `piso.empate_alternativa = "primero_en_pisos_order"` y `piso.empate_tolerancia = 1e-12`;
- `kappa.indefinida` y `contencion.indefinida = {"replicas": "excluir_y_contar", "observado": "no_aplica"}`;
- `cochran.todos_concordantes = {"Q": 0, "p": 1, "mcnemar_pares": false}`;
- `pendiente.P6`.

No se modificó `enmienda_4` ni ningún bloque anterior (comprobación por igualdad del JSON y `test_bloques_previos_intactos_frente_a_enmienda_4`). Huella LF nueva: `068f3352…6351544` (`CONFIG_SHA256_ENMIENDA_5`).

## 3. Ajustes del código

| Punto | Cambio |
|---|---|
| P1 | `sensibilidad._por_semilla_emparejada` lee `enmienda_5.por_semilla.sensibilidad.comparacion` (solo admite `por_replica`): piso permutado = `max(semilla emparejada, media de 5)` en el observado y en cada réplica. Se eliminó el `PendienteError` |
| P2 | `contraste.d_reps(ties='primero')`: el primero de los empatados según `pisos.order` (orden de `floor_candidates`). La fila A3 de la tabla de robustez se calcula siempre: con empate, informa en qué técnicas lo hay; sin él, coincide con el principal |
| P3 | `kappa_weighted` devuelve nan donde 1 − p_e = 0; `kappa` excluye esas réplicas del IC y las cuenta (`n_replicas_indefinidas`); si el observado es indefinido, `obs` e `interval` = `"no_aplica"` |
| P4 | `contencion`: lo mismo con \|E_vis\|\* = 0. La condición 1 de F4.6 vale `"no_aplica"` si el IC no aplica |
| P5 | `cochran_q(correct, cfg)` devuelve (Q, p, gl, todos_concordantes); con todos los bloques concordantes, Q = 0 y p = 1 del config, y no se calculan los McNemar por pares |
| A3 | `d_reps(…, tol)`: empate si \|F1_a − F1_max\| ≤ `enmienda_5.piso.empate_tolerancia` (1e-12) en el observado. Se pasa en el contraste principal, en todas las variantes y en la semilla emparejada |
| P6 | Sin cambios: `ejecutar_f46` sigue marcando la condición 2 como `PENDIENTE (P6)`, y `run_fase5` lo imprime como «AVISO (no bloquea)» |
| Compuerta | `run_fase5.GATE_TAG = 'enmienda-5'` es la única constante: la literal aparece una sola vez en el módulo. `check_gate` exige esa etiqueta como ancestro y compara con ella los tres archivos congelados. `pendientes()` ahora solo comprueba que estén las resoluciones de `enmienda_5`; con el config actual no devuelve nada |

**IC de θ.** `test_ic_de_theta_sale_de_verdict_percentile_interval` confirma que el intervalo de θ es exactamente `verdict.percentile_interval(θ*, α_h)`, sin la variante con `Decimal`. `Decimal` solo se usa en los IC descriptivos al 95 % (A5, A10, A11).

## 4. Pruebas

**Protocolo** (`tests/test_protocolo.py`, 38):
- huella `CONFIG_SHA256_ENMIENDA_5`;
- `test_enmienda_5_texto`: campos en orden, «**Relación con el §11.**» y tabla P1–P6;
- `test_enmienda_5_bloque`: último bloque, tipos y valores;
- `test_bloques_previos_intactos_frente_a_enmienda_4`: config de la etiqueta `enmienda-4`, con su huella;
- ajustes en las pruebas de la Enmienda 4 (sección delimitada por la Enmienda 5; `enmienda_4` va justo antes de `enmienda_5`).

**Sintéticas nuevas o ajustadas** (`tests/test_fase5_*.py`, 52 en total):

| Prueba | Qué comprueba |
|---|---|
| `test_P1_A14_emparejada_por_replica` | Piso = máximo por réplica de la semilla emparejada y la media de 5; p de superioridad recalculado; `comparacion ≠ por_replica` ⇒ error |
| `test_P2_robustez_con_empate` | Brillo y permutado empatados en el máximo: la fila A3 usa el brillo (primero en `pisos.order`) y su p coincide con D = img − brillo |
| `test_empate_A3_menor_D_y_alternativa_P2` | — |
| `test_tolerancia_A3_dentro_y_fuera_de_1e12` | 0.5e-12 ⇒ empate; 2e-12 ⇒ sin empate |
| `test_P3_kappa_indefinida` | 7 réplicas degeneradas excluidas y contadas; vectores constantes ⇒ «no aplica» |
| `test_P4_contencion_indefinida` | Visual perfecto ⇒ «no aplica»; un solo error visual ⇒ réplicas sin esa muestra excluidas y contadas |
| `test_P5_cochran_todos_concordantes` | (0, 1, 2, True); sin McNemar por pares |
| `test_ic_de_theta_sale_de_verdict_percentile_interval` | — |
| `test_config_real_sin_pendientes_que_bloqueen` | Sustituye a `test_config_real_tiene_pendientes`: `pendientes()` vacío; solo el aviso de P6 |
| `test_se_niega_con_pendientes` | Config sin `enmienda_5` ⇒ negativa |
| `test_compuerta_en_una_sola_constante_y_se_niega_sin_la_etiqueta` | Git real con una etiqueta inexistente ⇒ «no es ancestro» |
| `test_aviso_P6_en_el_log` | — |

**Salida de pytest:**

```text
$ python -m pytest tests/test_protocolo.py tests/test_fase5_*.py -v   →  90 passed (pytest_fase5_cierre_d1.log)
$ CANON_*… python -m pytest tests -q --ignore=tests/test_label_independence.py   →  1069 passed in 134.77s (pytest_cierre_d1.log)
```

El rendimiento con B = 10,000 sigue igual: Fase 5 sintética 0.1 s y F4.6 0.1 s.

## 5. `--verificar-insumos` sobre los datos reales

Archivo: `verificar_insumos_cierre_d1.log`.

```text
run_fase5 — solo --verificar-insumos (sin --ejecutar-prueba no se carga p_dep)
… 56 archivos OK (45 visuales + 11 referencias)
— Prueba: n = 114, soportes 0/1 = 70/44; filas ordenadas por sample_id (test_0000 … test_0113)
— u de train (umap_params.pkl): (908, 5), etiquetas alineadas con el índice por idx
```

`results/f5_contraste/` no existe.

## 6. Para la etapa 2

- **La ejecución real exige la etiqueta `enmienda-5`, que no existe todavía** (no se crean etiquetas en este bloque). Hasta entonces, `run_fase5 --ejecutar-prueba` se niega con «la etiqueta enmienda-5 no es ancestro de HEAD».
- La etiqueta debe apuntar a un commit que contenga la Enmienda 5, y los tres archivos congelados deben quedar sin cambios después de ella.

## Diff (`git diff d66e43f HEAD`, protocolo, config y código)

```text
 configs/protocolo.json          |  55 +++++++++++++++
 docs/protocolo_preregistrado.md |  38 +++++++++++
 src/protocol/contraste.py       | 107 +++++++++++++++++++----------
 src/protocol/run_fase5.py       |  39 +++++++----
 src/protocol/sensibilidad.py    |  66 +++++++++---------
 tests/fase5_sintetico.py        |   7 +-
 tests/test_fase5_contraste.py   | 146 ++++++++++++++++++++++++++++++++++------
 tests/test_fase5_run.py         |  29 ++++++--
 tests/test_protocolo.py         |  78 ++++++++++++++++++---
 9 files changed, 447 insertions(+), 118 deletions(-)
```

### `docs/protocolo_preregistrado.md`

```diff
diff --git a/docs/protocolo_preregistrado.md b/docs/protocolo_preregistrado.md
index fafedf7..378549d 100644
--- a/docs/protocolo_preregistrado.md
+++ b/docs/protocolo_preregistrado.md
@@ -325,3 +325,41 @@ Los demás puntos de la matriz §8 (objetivo, muestras, resolución, normalizaci
 - La desviación de procedencia de F3.4 (bitácora, H5) se acepta como desviación formal.
 - La pasada de F3.7 sobre prueba solo verificó el `p_dep` de los modelos cargados y no generó predicciones archivadas.
 - D1 y D2 del bloque A son errores de redacción de la bitácora de F3.5 (CE) y no afectan ninguna cifra.
+
+## Enmienda 5 (2026-11-07)
+
+**Autor.** SV (Santiago Elihu Soria Solís).
+
+**Momento.**
+- Es **posterior** a la Enmienda 4 (etiqueta `enmienda-4`) y a la implementación de la Fase 5 con datos sintéticos (`results/f5_preparacion/reporte_bloque_d1.md`).
+- Es **anterior** a todo cálculo de la Fase 5 sobre datos reales.
+
+**Motivo.** Al implementar la Fase 5 aparecieron dos tipos de problemas:
+- cinco puntos que no eran computables (P1–P5 del reporte del bloque D, etapa 1);
+- una tolerancia de empate que no estaba fijada (A3).
+
+**Regla de resolución.** La misma de la Enmienda 4.
+
+**Declaración de exposición.**
+- La misma de la Enmienda 4. Ningún estadístico de la Fase 5 se conoce.
+- Desde la Enmienda 4 solo se añadió el F1 de **validación** del clasificador de longitud (F4.5, `results/f45_longitud/`). Ninguna métrica de prueba de las referencias se ha calculado.
+
+**Alcance.** No cambia ningún modelo, estadístico principal, margen, α, B ni la regla de veredicto. Solo fija detalles de cálculo. La forma legible por código está en `configs/protocolo.json` → `enmienda_5`; los bloques anteriores del archivo no cambian.
+
+**Relación con el §11.**
+- Es la misma desviación declarada en la Enmienda 4: fija detalles de cálculo de análisis confirmatorios (A3) y de sensibilidad (P1) después de la consulta de prueba de los finalistas.
+- Las filas nuevas se incluyen en el compromiso de robustez de la Enmienda 4:
+  - la alternativa de A3 (P2);
+  - la comparación por réplica de la sensibilidad de A14 (P1).
+
+**Cambios.**
+
+| Punto | Resolución |
+|---|---|
+| P1 | A14, sensibilidad: comparación por réplica. En cada réplica se usa el mayor entre el piso permutado de la semilla emparejada y la media de las 5 semillas. |
+| P2 | A3, alternativa de robustez: entre los pisos empatados, el primero según `pisos.order`. |
+| P3 | Kappa indefinida: las réplicas con kappa indefinida se excluyen del IC y se reporta su número; si es indefinida en el observado, «no aplica». |
+| P4 | Contención indefinida (\|E_vis\| = 0): igual que P3. |
+| P5 | Q de Cochran con todos los bloques concordantes: Q = 0, p = 1, sin McNemar por pares. |
+| A3 | Tolerancia: hay empate si \|F1_a − F1_b\| ≤ 1e-12 en el observado. |
+| P6 | La condición 2 de la frase de F4.6 queda pendiente hasta F4.1 (bloque E); no se cambia. |
```

### `configs/protocolo.json`

```diff
diff --git a/configs/protocolo.json b/configs/protocolo.json
index 89c9ce2..02d5a6c 100644
--- a/configs/protocolo.json
+++ b/configs/protocolo.json
@@ -520,5 +520,60 @@
       "F3.7: la pasada sobre prueba no generó predicciones archivadas",
       "D1 y D2 (bitácora F3.5, CE): errores de redacción"
     ]
+  },
+  "enmienda_5": {
+    "_seccion": "Enmienda 5 (2026-11-07)",
+    "_autor": "SV",
+    "_fuente": "results/f5_preparacion/reporte_bloque_d1.md",
+    "_momento": "posterior a la Enmienda 4 y a la implementación con datos sintéticos; anterior a todo cálculo de la Fase 5 sobre datos reales",
+    "relacion_s11": {
+      "desviacion_declarada": true,
+      "misma_que": "enmienda_4",
+      "compromiso_robustez": true,
+      "filas_nuevas_robustez": [
+        "P1",
+        "P2"
+      ]
+    },
+    "resoluciones": {
+      "P1": "por_semilla.sensibilidad",
+      "P2": "piso",
+      "P3": "kappa",
+      "P4": "contencion",
+      "P5": "cochran",
+      "A3": "piso",
+      "P6": "pendiente_F4.1"
+    },
+    "por_semilla": {
+      "sensibilidad": {
+        "comparacion": "por_replica"
+      }
+    },
+    "piso": {
+      "empate_alternativa": "primero_en_pisos_order",
+      "empate_tolerancia": 1e-12
+    },
+    "kappa": {
+      "indefinida": {
+        "replicas": "excluir_y_contar",
+        "observado": "no_aplica"
+      }
+    },
+    "contencion": {
+      "indefinida": {
+        "replicas": "excluir_y_contar",
+        "observado": "no_aplica"
+      }
+    },
+    "cochran": {
+      "todos_concordantes": {
+        "Q": 0,
+        "p": 1,
+        "mcnemar_pares": false
+      }
+    },
+    "pendiente": {
+      "P6": "condición 2 de F4.6 (IC de ΔF1 de F4.1), hasta el bloque E"
+    }
   }
 }
```

### `src/protocol/contraste.py`

```diff
diff --git a/src/protocol/contraste.py b/src/protocol/contraste.py
index bb429b6..1308c10 100644
--- a/src/protocol/contraste.py
+++ b/src/protocol/contraste.py
@@ -2,7 +2,7 @@
 contraste.py
 ────────────
 Cálculos de la Fase 5 sobre réplicas pareadas (docs/protocolo_preregistrado.md
-§5–§7 y Enmienda 4), con la regla de decisión de src/protocol/verdict.py (sin
+§5–§7 y Enmiendas 4 y 5), con la regla de decisión de src/protocol/verdict.py (sin
 modificarla). Todo parámetro sale de configs/protocolo.json.
 
 Entrada común: un «F1-tabla» con, para cada clasificador, su F1-Macro observado
@@ -19,13 +19,14 @@ Funciones → protocolo:
                               semillas por réplica (§5.3)
     d_reps                    D* (§5.3) con A2 (identidad del máximo observado,
                               o máximo por réplica como sensibilidad) y A3
+                              (empate con la tolerancia de la Enmienda 5; P2)
     veredicto                 verdict.verdicts con B del protocolo; IC de D (A4)
     retencion                 ρ̂ (§5.6; A5, A6)
     descomposicion            §7 (A5)
     frente_a_descriptores     §7 (A11)
     contraste_longitud        §6 (A7)
-    cochran_q, mcnemar_*      §7 (A12)
-    kappa, contencion         F4.6 (A10; docs/plan_fase4.md)
+    cochran_q, mcnemar_*      §7 (A12; Enmienda 5, P5)
+    kappa, contencion         F4.6 (A10; docs/plan_fase4.md; Enmienda 5, P3 y P4)
     terciles_longitud,
     spearman_u_longitud       §6 descriptivo (A8)
 """
@@ -42,8 +43,12 @@ from src.protocol import verdict
 TECHNIQUES = ('gray_scott', 'perlin', 'lorenz')
 
 
-class PendienteError(RuntimeError):
-    """El cálculo cae en un caso que el protocolo y la Enmienda 4 no fijan (PENDIENTE)."""
+NO_APLICA = 'no_aplica'
+
+
+def empate_tolerancia(cfg):
+    """Tolerancia del empate de A3 (Enmienda 5): |F1_a − F1_b| ≤ tol en el observado."""
+    return float(cfg['enmienda_5']['piso']['empate_tolerancia'])
 
 
 # ── Tabla de F1 ───────────────────────────────────────────────────────────
@@ -105,26 +110,34 @@ def floor_candidates(F, tech, cfg, include=None, permutado_fase='control_permuta
     return out
 
 
-def d_reps(img_obs, img_rep, cands, mode='identidad', ties='min'):
+def d_reps(img_obs, img_rep, cands, mode='identidad', ties='min', tol=0.0):
     """D = F1_img − F1_piso (§5.3).
 
     mode='identidad' (A2): el piso es la referencia con el mayor F1 observado y
-    esa identidad se remuestrea. Empate exacto en el máximo observado (A3,
-    ties='min'): en cada réplica, el menor D* entre los empatados. ties='max' es
-    la opción opuesta, solo para la tabla de robustez.
+    esa identidad se remuestrea. Hay empate si |F1_a − F1_max| ≤ tol en el
+    observado (A3; tolerancia de la Enmienda 5).
+      ties='min' (A3): en cada réplica, el menor D* entre los empatados.
+      ties='primero' (Enmienda 5, P2; alternativa de robustez): el primero de
+      los empatados según el orden de cands (= pisos.order).
     mode='max_por_replica' (sensibilidad de A2): en cada réplica, el máximo de los pisos.
     Devuelve (D_obs, D_rep, info)."""
     obs = {k: float(v[0]) for k, v in cands.items()}
     best = max(obs.values())
-    tied = [k for k, v in obs.items() if v == best]
+    tied = [k for k, v in obs.items() if abs(v - best) <= tol]
     if mode == 'identidad':
-        D = np.stack([img_rep - cands[k][1] for k in tied])
-        D_rep = D.min(axis=0) if ties == 'min' else D.max(axis=0)
+        if ties == 'min':
+            D_obs = min(img_obs - obs[k] for k in tied)
+            D_rep = np.stack([img_rep - cands[k][1] for k in tied]).min(axis=0)
+        elif ties == 'primero':
+            D_obs, D_rep = img_obs - obs[tied[0]], img_rep - cands[tied[0]][1]
+        else:
+            raise ValueError(ties)
     elif mode == 'max_por_replica':
+        D_obs = img_obs - best
         D_rep = img_rep - np.stack([v[1] for v in cands.values()]).max(axis=0)
     else:
         raise ValueError(mode)
-    return img_obs - best, D_rep, {'piso': tied, 'piso_f1_obs': obs, 'empate': len(tied) > 1, 'mode': mode}
+    return D_obs, D_rep, {'piso': tied, 'piso_f1_obs': obs, 'empate': len(tied) > 1, 'mode': mode, 'ties': ties}
 
 
 # ── Veredicto (§5.2–§5.5) ─────────────────────────────────────────────────
@@ -165,7 +178,7 @@ def contraste(F, cfg, fase='final', r=None, formulation='B', delta=None, beto_sh
             raise ValueError(formulation)
         ps = permutado_seed(t) if callable(permutado_seed) else permutado_seed
         cands = floor_candidates(F, t, cfg, include=floors, permutado_seed=ps)
-        d_obs, d_rep, info = d_reps(i_obs, i_rep, cands, floor_mode, ties)
+        d_obs, d_rep, info = d_reps(i_obs, i_rep, cands, floor_mode, ties, empate_tolerancia(cfg))
         D[t] = (d_obs, d_rep)
         extras[t] = {'f1_img_obs': i_obs, 'f1_beto_obs': b_obs, **info}
     res = veredicto(theta, D, cfg, extras)
@@ -281,16 +294,20 @@ def yhat_mean(p_seeds, cfg):
     return (p.mean(axis=1) > s['threshold']).astype(np.int64)
 
 
-def cochran_q(correct):
-    """Q de Cochran asintótica (χ², gl = k − 1) sobre una matriz n × k de aciertos 0/1."""
+def cochran_q(correct, cfg):
+    """Q de Cochran asintótica (χ², gl = k − 1) sobre una matriz n × k de aciertos 0/1.
+
+    Si todos los bloques son concordantes (0/0), Q y p son los de la Enmienda 5,
+    P5 (Q = 0, p = 1). Devuelve (Q, p, gl, todos_concordantes)."""
     X = np.asarray(correct, dtype=np.float64)
     k = X.shape[1]
     C, R, N = X.sum(axis=0), X.sum(axis=1), X.sum()
     den = k * N - (R ** 2).sum()
     if den == 0:
-        raise PendienteError('Q de Cochran indefinida: todos los bloques son concordantes (0/0)')
+        tc = cfg['enmienda_5']['cochran']['todos_concordantes']
+        return float(tc['Q']), float(tc['p']), k - 1, True
     q = (k - 1) * (k * (C ** 2).sum() - N ** 2) / den
-    return float(q), float(stats.chi2.sf(q, k - 1)), k - 1
+    return float(q), float(stats.chi2.sf(q, k - 1)), k - 1, False
 
 
 def _discordant(correct_a, correct_b):
@@ -327,11 +344,13 @@ def comparaciones_mcnemar(y, yhat, cfg, lado_vs_ref=None):
     alpha_ct = cfg['secundarios']['entre_tecnicas']['alpha']
     alpha = cfg['contraste']['multiplicidad']['alpha']          # §13: α = 0.05, Holm por familia
     ok = {k: (np.asarray(v) == y).astype(int) for k, v in yhat.items()}
-    q, p_q, gl = cochran_q(np.column_stack([ok[t] for t in TECHNIQUES]))
+    q, p_q, gl, concordantes = cochran_q(np.column_stack([ok[t] for t in TECHNIQUES]), cfg)
     if gl != e['cochran']['gl']:
         raise ValueError('gl de Cochran')
-    out = {'cochran': {'Q': q, 'p': p_q, 'gl': gl, 'alpha': alpha_ct}, 'pares': None, 'frente_a': {}}
-    if p_q < alpha_ct:
+    out = {'cochran': {'Q': q, 'p': p_q, 'gl': gl, 'alpha': alpha_ct, 'todos_concordantes': concordantes},
+           'pares': None, 'frente_a': {}}
+    pares_ok = not concordantes or cfg['enmienda_5']['cochran']['todos_concordantes']['mcnemar_pares']
+    if pares_ok and p_q < alpha_ct:
         pares = {f'{a}__{b}': mcnemar_bilateral(ok[a], ok[b]) for a, b in combinations(TECHNIQUES, 2)}
         h = verdict.holm({k: v['p'] for k, v in pares.items()}, alpha)
         out['pares'] = {k: {**v, **h[k], 'lado': m['lado_entre_tecnicas']} for k, v in pares.items()}
@@ -360,15 +379,33 @@ def _joint(W, a, b):
 
 
 def kappa_weighted(W, a, b):
-    """Kappa de Cohen de dos vectores 0/1 bajo cada fila de W. 1 − p_e = 0 → PENDIENTE."""
+    """Kappa de Cohen de dos vectores 0/1 bajo cada fila de W; nan donde 1 − p_e = 0 (indefinida)."""
     n11, n10, n01, n00 = _joint(W, a, b)
     N = n11 + n10 + n01 + n00
     po = (n11 + n00) / N
     pa, pb = (n11 + n10) / N, (n11 + n01) / N
     pe = pa * pb + (1 - pa) * (1 - pb)
-    if np.any(1 - pe == 0):
-        raise PendienteError(f'kappa indefinida (1 − p_e = 0) en {int(np.sum(1 - pe == 0))} réplica(s)')
-    return (po - pe) / (1 - pe)
+    den = 1 - pe
+    return np.divide(po - pe, den, out=np.full(den.shape, np.nan), where=den != 0)
+
+
+def _indefinida_spec(cfg, name):
+    s = cfg['enmienda_5'][name]['indefinida']
+    if s != {'replicas': 'excluir_y_contar', 'observado': NO_APLICA}:
+        raise ValueError(f'enmienda_5.{name}.indefinida inesperado {s}')
+    return s
+
+
+def _summary_excluding(obs, reps, lo, hi):
+    """Enmienda 5, P3/P4: observado indefinido → «no aplica»; réplicas indefinidas se excluyen
+    del IC y se cuentan."""
+    reps = np.asarray(reps, dtype=np.float64)
+    bad = np.isnan(reps)
+    if np.isnan(obs):
+        return {'obs': NO_APLICA, 'interval': NO_APLICA, 'n_replicas_indefinidas': int(bad.sum())}
+    ok = reps[~bad]
+    return {'obs': float(obs), 'interval': bs.percentile_interval(ok, lo, hi) if ok.size else NO_APLICA,
+            'n_replicas_indefinidas': int(bad.sum())}
 
 
 def kappa(W, y, yhat, cfg):
@@ -378,6 +415,7 @@ def kappa(W, y, yhat, cfg):
     lo, hi = _ci95(k['ic'])
     if (k['principal'], k['secundario']) != ('y_hat', 'vector_acierto'):
         raise ValueError(k)
+    _indefinida_spec(cfg, 'kappa')
     W_obs = bs.observed_weights(len(y))
     names = ['beto', 'emb768', 'techo5d'] + list(TECHNIQUES)
     out = {}
@@ -387,25 +425,26 @@ def kappa(W, y, yhat, cfg):
         for a, b in combinations(names, 2):
             if a in TECHNIQUES and b in TECHNIQUES:
                 continue
-            res[f'{a}__{b}'] = {'obs': float(kappa_weighted(W_obs, vec[a], vec[b])[0]),
-                                'interval': bs.percentile_interval(kappa_weighted(W, vec[a], vec[b]), lo, hi)}
+            res[f'{a}__{b}'] = _summary_excluding(kappa_weighted(W_obs, vec[a], vec[b])[0],
+                                                  kappa_weighted(W, vec[a], vec[b]), lo, hi)
         out[base] = res
     return out
 
 
 def contencion(W, y, yhat_vis, yhat_ref, cfg):
     """C = |E_vis ∩ E_ref| / |E_vis|, referencia nula |E_ref| / n, IC percentil bilateral 95 %
-    sobre las réplicas de §5.2 (docs/plan_fase4.md, F4.6). |E_vis|* = 0 → PENDIENTE."""
+    sobre las réplicas de §5.2 (docs/plan_fase4.md, F4.6). |E_vis| = 0 → Enmienda 5, P4
+    (observado «no aplica»; réplicas excluidas y contadas)."""
+    _indefinida_spec(cfg, 'contencion')
     lo, hi = _ci95(cfg['enmienda_4']['ic']['descriptivo'])
     ev = (np.asarray(yhat_vis) != y).astype(np.float64)
     er = (np.asarray(yhat_ref) != y).astype(np.float64)
-    if ev.sum() == 0:
-        raise PendienteError('contención indefinida: el visual no comete errores')
     num, den = W @ (ev * er), W @ ev
-    if np.any(den == 0):
-        raise PendienteError(f'contención indefinida en {int(np.sum(den == 0))} réplica(s) sin errores visuales')
-    return {'C_obs': float((ev * er).sum() / ev.sum()), 'nula_obs': float(er.mean()),
-            'interval': bs.percentile_interval(num / den, lo, hi),
+    reps = np.divide(num, den, out=np.full(den.shape, np.nan), where=den > 0)
+    obs = (ev * er).sum() / ev.sum() if ev.sum() > 0 else np.nan
+    s = _summary_excluding(obs, reps, lo, hi)
+    return {'C_obs': s['obs'], 'nula_obs': float(er.mean()), 'interval': s['interval'],
+            'n_replicas_indefinidas': s['n_replicas_indefinidas'],
             'n_errores_vis': int(ev.sum()), 'n_errores_ref': int(er.sum()), 'n_interseccion': int((ev * er).sum())}
 
 
```

### `src/protocol/sensibilidad.py`

```diff
diff --git a/src/protocol/sensibilidad.py b/src/protocol/sensibilidad.py
index fa9cc9b..0a21747 100644
--- a/src/protocol/sensibilidad.py
+++ b/src/protocol/sensibilidad.py
@@ -1,9 +1,9 @@
 """
 sensibilidad.py
 ───────────────
-Análisis de sensibilidad (§8) con las resoluciones de la Enmienda 4, y la tabla
-de robustez que exige la relación con el §11 de esa enmienda. Ninguna variante
-modifica el veredicto principal.
+Análisis de sensibilidad (§8) con las resoluciones de las Enmiendas 4 y 5, y la
+tabla de robustez que exige la relación con el §11 de la Enmienda 4. Ninguna
+variante modifica el veredicto principal.
 
 Variantes de §8:
   retencion_exigida     r ∈ sensibilidad.retencion_exigida.r
@@ -16,22 +16,18 @@ Variantes de §8:
   segundo_mejor         fase sensibilidad con el piso permutado del finalista (A17)
   piso_sin_permutado    piso = máximo de sensibilidad.piso_sin_permutado
 
-Tabla de robustez (alternativas declaradas): A2 máximo por réplica; A3 (solo
-aplica si hay empate en el máximo observado); A7 con m = 3 y en cada variante;
-A12 bilateral frente a BETO y al techo; A14 semilla emparejada.
-
-PENDIENTE (no resuelto): la sensibilidad de A14, «piso permutado de la semilla
-emparejada, si es mayor», no fija frente a qué se compara (el observado o cada
-réplica). Se exige enmienda_4.por_semilla.sensibilidad.comparacion ∈
-{"observado", "por_replica"}; sin ella, la variante lanza PendienteError.
-Del mismo modo, la alternativa de A3 no está declarada: si hay empate en el
-máximo observado, la fila de A3 lanza PendienteError.
+Tabla de robustez (alternativas declaradas): A2 máximo por réplica; A3 con la
+alternativa de la Enmienda 5, P2 (el primero de los empatados según
+pisos.order); A7 con m = 3 y en cada variante; A12 bilateral frente a BETO y al
+techo; A14 semilla emparejada con la comparación por réplica de la Enmienda 5,
+P1 (en cada réplica, el mayor entre el permutado de la semilla emparejada y la
+media de las 5 semillas).
 """
 
 import numpy as np
 
 from src.protocol import contraste as cx
-from src.protocol.contraste import PendienteError, TECHNIQUES
+from src.protocol.contraste import TECHNIQUES
 
 
 def _verdicts(res):
@@ -62,11 +58,11 @@ def formulacion_a(F, cfg, canon):
     return out
 
 
-def por_semilla(F, cfg, permutado='media', comparacion=None):
+def por_semilla(F, cfg, permutado='media'):
     """Veredicto con cada semilla por separado (A14). Holm entre técnicas dentro de cada semilla
     (cada llamada a contraste es una familia de tres). Piso permutado = media de 5 (principal) o,
-    como sensibilidad, el de la semilla emparejada si es mayor (comparacion = 'observado' o
-    'por_replica', PENDIENTE de fijar)."""
+    como sensibilidad, el de la semilla emparejada si es mayor, comparado en cada réplica
+    (Enmienda 5, P1)."""
     e = cfg['enmienda_4']['por_semilla']
     if (e['piso_permutado'], e['holm'], e['reporte']) != \
             ('media_5_semillas', 'entre_tecnicas_dentro_de_cada_semilla', 'n_semillas_V1'):
@@ -77,17 +73,17 @@ def por_semilla(F, cfg, permutado='media', comparacion=None):
         if permutado == 'media':
             out[s] = cx.contraste(F, cfg, seed_index=i)
         elif permutado == 'emparejada':
-            out[s] = _por_semilla_emparejada(F, cfg, i, comparacion)
+            out[s] = _por_semilla_emparejada(F, cfg, i)
         else:
             raise ValueError(permutado)
     n_v1 = {t: sum(out[s]['techniques'][t]['verdict'] == 'V1' for s in seeds) for t in TECHNIQUES}
     return {'por_semilla': out, 'n_semillas_V1': n_v1}
 
 
-def _por_semilla_emparejada(F, cfg, i, comparacion):
-    if comparacion not in ('observado', 'por_replica'):
-        raise PendienteError('A14 (sensibilidad): «si es mayor» no fija la comparación (observado o por '
-                             'réplica); falta enmienda_4.por_semilla.sensibilidad.comparacion')
+def _por_semilla_emparejada(F, cfg, i):
+    comparacion = cfg['enmienda_5']['por_semilla']['sensibilidad']['comparacion']
+    if comparacion != 'por_replica':
+        raise ValueError(f'enmienda_5.por_semilla.sensibilidad.comparacion inesperado {comparacion!r}')
     seeds = cfg['modelos']['seeds']
     floors = [f for f in cfg['pisos']['order'] if f != 'visual_permutado']
     c = cfg['contraste']
@@ -99,11 +95,9 @@ def _por_semilla_emparejada(F, cfg, i, comparacion):
         cands = cx.floor_candidates(F, t, cfg, include=floors)
         m_obs, m_rep = cx.img(F, 'control_permutacion', t)
         s_obs, s_rep = cx.img(F, 'control_permutacion', t, i)
-        if comparacion == 'observado':
-            cands['visual_permutado'] = (s_obs, s_rep) if s_obs > m_obs else (m_obs, m_rep)
-        else:
-            cands['visual_permutado'] = (max(s_obs, m_obs), np.maximum(s_rep, m_rep))
-        d_obs, d_rep, info = cx.d_reps(i_obs, i_rep, cands)
+        # P1: en cada réplica (y en el observado), el mayor entre la semilla emparejada y la media de 5
+        cands['visual_permutado'] = (max(s_obs, m_obs), np.maximum(s_rep, m_rep))
+        d_obs, d_rep, info = cx.d_reps(i_obs, i_rep, cands, tol=cx.empate_tolerancia(cfg))
         D[t] = (d_obs, d_rep)
         extras[t] = {'f1_img_obs': i_obs, **info, 'semilla': seeds[i]}
     return cx.veredicto(theta, D, cfg, extras)
@@ -149,16 +143,18 @@ def _row(alt, principal, res, nota=None):
             'cambia_global': _global(res) != _global(principal), 'nota': nota}
 
 
-def robustez(F, cfg, principal, variantes, mcnemar_principal=None, mcnemar_bilateral=None, comparacion_a14=None):
+def robustez(F, cfg, principal, variantes, mcnemar_principal=None, mcnemar_bilateral=None):
     """Tabla de robustez (Enmienda 4, relación con el §11): veredicto por técnica y global bajo
     cada alternativa declarada y si cambia respecto del principal."""
     rows = []
     rows.append(_row('A2: máximo por réplica', principal, cx.contraste(F, cfg, floor_mode='max_por_replica')))
-    if any(principal['techniques'][t]['empate'] for t in TECHNIQUES):
-        raise PendienteError('A3: hay empate en el máximo observado y la Enmienda 4 no declara alternativa')
-    rows.append({'alternativa': 'A3: empate en el máximo observado', 'veredictos': _verdicts(principal),
-                 'global': _global(principal), 'cambia_tecnica': {t: False for t in TECHNIQUES},
-                 'cambia_global': False, 'nota': 'sin empate en el máximo observado: A3 no aplica'})
+    if cfg['enmienda_5']['piso']['empate_alternativa'] != 'primero_en_pisos_order':
+        raise ValueError(cfg['enmienda_5']['piso'])
+    empate = {t: principal['techniques'][t]['empate'] for t in TECHNIQUES}
+    rows.append(_row('A3: primero de los empatados según pisos.order (Enmienda 5, P2)', principal,
+                     cx.contraste(F, cfg, ties='primero'),
+                     nota=('empate en el máximo observado en: ' + ', '.join(t for t in TECHNIQUES if empate[t]))
+                     if any(empate.values()) else 'sin empate en el máximo observado: igual al principal'))
     # A7: m = 3 y calificación en cada variante (no cambia V ni G; solo la calificación)
     cal = {'principal': cx.contraste_longitud(F, cfg, _verdicts(principal), m_mode='m3')}
     for name, group in variantes.items():
@@ -181,9 +177,9 @@ def robustez(F, cfg, principal, variantes, mcnemar_principal=None, mcnemar_bilat
         rows.append({'alternativa': 'A12: McNemar bilateral frente a BETO y al techo', 'veredictos': _verdicts(principal),
                      'global': _global(principal), 'cambia_tecnica': {t: False for t in TECHNIQUES},
                      'cambia_global': False, 'nota': '§7: McNemar no modifica veredictos', 'cambia_rechazo_mcnemar': ch})
-    ps = por_semilla(F, cfg, permutado='emparejada', comparacion=comparacion_a14)
+    ps = por_semilla(F, cfg, permutado='emparejada')
     base = variantes['por_semilla']
-    rows.append({'alternativa': 'A14: piso permutado de la semilla emparejada, si es mayor',
+    rows.append({'alternativa': 'A14: piso permutado de la semilla emparejada, si es mayor (por réplica; Enmienda 5, P1)',
                  'n_semillas_V1': ps['n_semillas_V1'],
                  'cambia_n_semillas_V1': {t: ps['n_semillas_V1'][t] != base['n_semillas_V1'][t] for t in TECHNIQUES},
                  'por_semilla': {s: {'veredictos': _verdicts(r), 'global': _global(r)} for s, r in ps['por_semilla'].items()}})
```

### `src/protocol/run_fase5.py`

```diff
diff --git a/src/protocol/run_fase5.py b/src/protocol/run_fase5.py
index 286fc1f..720d759 100644
--- a/src/protocol/run_fase5.py
+++ b/src/protocol/run_fase5.py
@@ -10,9 +10,9 @@ Modos:
         identidad y csv_sha256). No carga p_dep. No escribe resultados.
   --ejecutar-prueba
         Ejecuta la Fase 5 sobre las predicciones de prueba. Exige:
-          · la etiqueta enmienda-4 en la historia de HEAD;
+          · la etiqueta GATE_TAG (enmienda-5) en la historia de HEAD;
           · docs/protocolo_preregistrado.md, configs/protocolo.json y
-            docs/plan_fase4.md sin cambios frente a enmienda-4;
+            docs/plan_fase4.md sin cambios frente a esa etiqueta;
           · el árbol de trabajo limpio (git status --porcelain vacío);
           · que no quede ningún punto PENDIENTE en configs/protocolo.json;
           · que no exista <out>/resultados.json (nunca sobrescribe).
@@ -49,7 +49,7 @@ from src.protocol.contraste import TECHNIQUES
 
 REPO = paths.REPO
 OUT_DEFAULT = paths.RESULTS_DIR / 'f5_contraste'
-GATE_TAG = 'enmienda-4'                     # etiqueta de la Enmienda 4 (bloque C, compuerta)
+GATE_TAG = 'enmienda-5'                     # única constante de la compuerta: etiqueta de la Enmienda 5
 FROZEN = ('docs/protocolo_preregistrado.md', 'configs/protocolo.json', 'docs/plan_fase4.md')
 CANON_PATH = paths.RESULTS_DIR / 'canonical_numbers.json'
 
@@ -78,15 +78,26 @@ def check_clean_tree():
 
 
 def pendientes(cfg):
-    """Puntos que la Enmienda 4 no deja computables y que este código no resuelve."""
+    """Puntos que bloquean --ejecutar-prueba: resoluciones de la Enmienda 5 ausentes del config."""
+    e5 = cfg.get('enmienda_5', {})
+    need = {'P1': ('por_semilla', 'sensibilidad', 'comparacion'), 'P2': ('piso', 'empate_alternativa'),
+            'A3': ('piso', 'empate_tolerancia'), 'P3': ('kappa', 'indefinida'),
+            'P4': ('contencion', 'indefinida'), 'P5': ('cochran', 'todos_concordantes')}
     out = []
-    comp = cfg['enmienda_4']['por_semilla'].get('sensibilidad', {}).get('comparacion')
-    if comp not in ('observado', 'por_replica'):
-        out.append('A14 (sensibilidad): «piso permutado de la semilla emparejada, si es mayor» no fija la '
-                   'comparación (observado o por réplica): enmienda_4.por_semilla.sensibilidad.comparacion')
+    for pid, path in need.items():
+        d = e5
+        for k in path:
+            d = d.get(k) if isinstance(d, dict) else None
+        if d is None:
+            out.append(f'{pid}: falta enmienda_5.{".".join(path)}')
     return out
 
 
+def avisos(cfg):
+    """Pendientes que no bloquean la Fase 5 (se reportan)."""
+    return [f'P6: {v}' for k, v in cfg.get('enmienda_5', {}).get('pendiente', {}).items() if k == 'P6']
+
+
 def check_results_absent(out_dir, name='resultados.json'):
     if (Path(out_dir) / name).exists():
         raise Negativa(f'{Path(out_dir) / name} ya existe: run_fase5 nunca sobrescribe una ejecución previa')
@@ -128,7 +139,6 @@ def ejecutar(ins, cfg, canon, idx, log=print):
     mcn = cx.comparaciones_mcnemar(y, yhat, cfg)
     mcn_bil = cx.comparaciones_mcnemar(y, yhat, cfg, lado_vs_ref={'beto': 'bilateral', 'techo5d': 'bilateral'})
     variantes = sens.todas(F, cfg, canon)
-    comp = cfg['enmienda_4']['por_semilla'].get('sensibilidad', {}).get('comparacion')
     res = {
         'principal': principal,
         'observados': {str(k): v['obs'] for k, v in F.items()},
@@ -138,7 +148,7 @@ def ejecutar(ins, cfg, canon, idx, log=print):
         'longitud': cx.contraste_longitud(F, cfg, v),
         'mcnemar': mcn,
         'sensibilidad': variantes,
-        'robustez': sens.robustez(F, cfg, principal, variantes, mcn, mcn_bil, comparacion_a14=comp),
+        'robustez': sens.robustez(F, cfg, principal, variantes, mcn, mcn_bil),
         'terciles_longitud': cx.terciles_longitud(
             y, ins.test_index['n_tokens_beto'].to_numpy(), ins.train['n_tokens_beto'].to_numpy(),
             {t: bs.predict(ins.visual[('final', t)]['p'], cfg['modelos']['decision_threshold']) for t in TECHNIQUES},
@@ -162,8 +172,9 @@ def ejecutar_f46(ins, cfg, idx):
     for t in TECHNIQUES:
         out['contencion'][t] = {r: cx.contencion(W, y, yhat[t], yhat[r], cfg) for r in ('techo5d', 'beto', 'emb768')}
         c = out['contencion'][t]['techo5d']
-        out['contencion'][t]['condicion_1'] = c['interval'][0] > c['nula_obs']
-        out['contencion'][t]['condicion_2_F41'] = 'PENDIENTE: requiere el IC de ΔF1 de F4.1'
+        out['contencion'][t]['condicion_1'] = (cx.NO_APLICA if c['interval'] == cx.NO_APLICA
+                                               else c['interval'][0] > c['nula_obs'])
+        out['contencion'][t]['condicion_2_F41'] = 'PENDIENTE (P6): requiere el IC de ΔF1 de F4.1'
         P = bs.predict(ins.visual[('final', t)]['p'], thr)
         out['contencion_por_semilla'][t] = {
             str(s): cx.contencion(W, y, P[:, i], yhat['techo5d'], cfg)
@@ -282,7 +293,9 @@ def main(argv=None, out_dir=OUT_DEFAULT, log=print):
         check_clean_tree()
         p = pendientes(cfg)
         if p:
-            raise Negativa('quedan puntos PENDIENTES en la Enmienda 4:\n  - ' + '\n  - '.join(p))
+            raise Negativa('quedan puntos PENDIENTES (Enmienda 5):\n  - ' + '\n  - '.join(p))
+        for a in avisos(cfg):
+            L(f'AVISO (no bloquea): {a}')
         if args.ejecutar_prueba:
             check_results_absent(out_dir)
         else:
```

Las diferencias de las pruebas (`tests/test_protocolo.py`, `tests/fase5_sintetico.py`, `tests/test_fase5_contraste.py`, `tests/test_fase5_run.py`) están en los commits `270d5ac` y siguiente.

## PR

(pendiente de crear; ver la respuesta del bloque)
