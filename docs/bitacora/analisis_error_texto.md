# Análisis de error por texto — Análisis de error II / Anexo A18

## 1. Identificación

| Campo | Valor |
|---|---|
| Tarea | Análisis descriptivo para la presentación de TT-II (diapositiva «Análisis de error II», Anexo A18); fuera de las tareas del plan v2 |
| Responsable | SV — Santiago |
| Rama | `analisis-error-por-texto` |
| Commit base | `d0eed6b` (main) |
| Entregables | `results/analisis_error_texto/` (script, `grupos_por_texto.csv`, `resumen_grupos.csv`, `resumen.md`, `insumos.json`) |
| Fecha de ejecución | 16 nov 2026 |
| Estado | **Cerrada** |

## 2. Propósito

Cruzar texto por texto los aciertos de BETO (Exp 4.1, F2.2) con los del finalista visual de cada técnica (media de p_dep de 5 semillas, fase `final`, F3.5), clasificar los 114 textos de prueba en Conservado / Perdido en la imagen / Recuperado / Error heredado y caracterizar cada grupo por longitud, truncamiento, clase, FN/FP y términos de pandemia. Es descriptivo: no entrena, no corre inferencia y no modifica `results/predictions/` ni `results/f5_contraste/`. Como se consultan predicciones de prueba ya entregadas en F3.5, no hay consulta nueva de test.

## 3. Método

- Predicciones leídas con `src.predictions.load_predictions` (valida formato F1.5 y manifiestos; p_dep con `float()`, ida y vuelta exacta). Se filtra `split == "test"`.
- Regla de decisión p_dep > 0.5 para el visual (sobre la media de las 5 semillas) y para BETO.
- `n_tokens_beto` y `truncated_128` se toman de `meta/sample_index.csv`, con la misma conversión de `results/f45_longitud/f45_longitud.py::features`; no se recalculan.
- Pandemia: `PANDEMIC_TERMS` copiada literalmente del repo del Módulo 1 (`proyecto-transformacion-texto-imagen`, `src/features/pandemic_stopwords.py`, commit `8076797048e20d22b5f9a67051c642f04ca3e4f8`, SHA-256 `af23f8094329f37b5e64ae59204d287cd883a772315a79145256fd4af41ff460`), que no está en ttdicevc. Se usa el mismo patrón `(...)` con IGNORECASE de `count_pandemic_terms`, aplicado al texto original de `processed/test.csv` en minúsculas. Antes de buscar, el texto se verifica contra `text_sha256` del índice. Decisión del usuario: usar esta lista con límites de palabra. La lista incluye términos genéricos («positivo», «negativo», «hospital», «fase», «brote»), así que el conteo es una cota superior de menciones de pandemia.
- Textos representativos: los 2 sample_id más cercanos a la mediana de tokens del grupo, con desempate por sample_id.
- Entorno: Python 3.13.2, pandas 2.3.3, scikit-learn 1.6.1, Windows 11 local, CPU, < 10 s.

Reproducir: `python results/analisis_error_texto/analisis_error_texto.py` (requiere TT_DATA_ROOT).

## 4. Insumos (SHA-256)

| Ruta | SHA-256 |
|---|---|
| `results/predictions/referencia/text__beto__finetune__seed42.csv` | `3d3587469545e592e39fa62d3b00b0acd52889bfe21dceb0a5c2816a28f7916e` |
| `meta/sample_index.csv` | `74dc778292447c66be60b35b5ec224c53b01433423923fa6d6ddef91749de0f5` |
| `TT_DATA_ROOT/processed/test.csv` | `e251582ae76e419e344228c3bcd3d6c8f468b481968e3cb6f06f221f0c9f6b5b` |
| `results/canonical_numbers.json` | `98ce1377b37b84a510c8940eeaed7069fd9953aa504f80daae5bb1b5990de8cb` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed123.csv` | `b51a2d0227a2bb9110b076fa9cb6ede73497fde3a67a26754707ca795f80bf41` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed2024.csv` | `8c5c9b633612e41facc8cd8e46d1af804da9b46faaec191dc966ae15edcc374a` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed42.csv` | `60e78e5c2f3d4cbde2584a7622ad5ff148d9e83257858e50142889a8556fc6bb` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed7.csv` | `3ead6bb773fbf2e586282a76f761c529ca24198f3d88005de0740835016acaf1` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed99.csv` | `e72dc2ae668f54ed1581ad172abc802cb73b99a405656a0f430b11e3490799d7` |
| `results/predictions/final/perlin__vit_b16__frozen__seed123.csv` | `de5d1a7c55bb629174cccacaa6da27e7acece57930262147b19f0248ca93108d` |
| `results/predictions/final/perlin__vit_b16__frozen__seed2024.csv` | `52588352631a8a3e7e8108e071a50a07e4e5c5f0a423b9781f880934d2788920` |
| `results/predictions/final/perlin__vit_b16__frozen__seed42.csv` | `2807eec87fca0391d8ce06538921571854c8b77ec6744a3145685981f5688f0d` |
| `results/predictions/final/perlin__vit_b16__frozen__seed7.csv` | `f2ec421680245dadfaad2eea27b0f26a5416eba09ac291aae207abe70ececec2` |
| `results/predictions/final/perlin__vit_b16__frozen__seed99.csv` | `27f74169fca562b1711d10ccb26411c74c5d295b2743f5ebaa284a2d3f39f199` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed123.csv` | `0b66f60777549bcf2181f931bc10e85d2dd77ffc02d71d321c74359b9484b065` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed2024.csv` | `5e2e09af39e19ec568d24ab765ee340eb8d697314c4a72138f450e89a76c640e` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed42.csv` | `fcac7c6b35a380637b0462ee4d541ec69d911278eed02d5f634cd38b67f91b76` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed7.csv` | `67b61ae0124831f7f7011be3d30d68a01fd8d89079783f5a320614f8a6387e38` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed99.csv` | `2ef17e032c1d861cc3b161a1ec6a47dc41b75f7cf8ecdddac6346bde30ce6d30` |
| `proyecto-transformacion-texto-imagen/src/features/pandemic_stopwords.py` @ `8076797` | `af23f8094329f37b5e64ae59204d287cd883a772315a79145256fd4af41ff460` |

## 5. Verificaciones

- **OK**: Matriz y F1-Macro de BETO. matriz [[54, 16], [16, 28]] (esperada [[54, 16], [16, 28]]); F1-Macro 0.7039 (ref.beto_exp41.f1_macro = 0.7039).
- **OK**: Gray-Scott: Perdido + Heredado = errores visuales (Tabla 84). 1 + 29 = 30 (esperado 30).
- **OK**: Gray-Scott: Recuperado + Heredado = errores de BETO. 3 + 29 = 32 (esperado 32).
- **OK** (informativa): Gray-Scott: F1-Macro de la media de 5 semillas frente a f35. media de p_dep → F1 0.7267; f35.gray_scott.finalista.f1_macro_mean (media de los F1 por semilla) = 0.7250.
- **OK**: Gray-Scott: Heredado ≥ errores comunes a los 4 modelos. 29 ≥ 27.
- **OK**: Gray-Scott: los cuatro grupos suman 114. 114.
- **OK**: Perlin: Perdido + Heredado = errores visuales (Tabla 84). 7 + 27 = 34 (esperado 34).
- **OK**: Perlin: Recuperado + Heredado = errores de BETO. 5 + 27 = 32 (esperado 32).
- **OK** (informativa): Perlin: F1-Macro de la media de 5 semillas frente a f35. media de p_dep → F1 0.6941; f35.perlin.finalista.f1_macro_mean (media de los F1 por semilla) = 0.6719.
- **OK**: Perlin: Heredado ≥ errores comunes a los 4 modelos. 27 ≥ 25.
- **OK**: Perlin: los cuatro grupos suman 114. 114.
- **OK**: Lorenz: Perdido + Heredado = errores visuales (Tabla 84). 3 + 25 = 28 (esperado 28).
- **OK**: Lorenz: Recuperado + Heredado = errores de BETO. 7 + 25 = 32 (esperado 32).
- **OK** (informativa): Lorenz: F1-Macro de la media de 5 semillas frente a f35. media de p_dep → F1 0.7449; f35.lorenz.finalista.f1_macro_mean (media de los F1 por semilla) = 0.7307.
- **OK**: Lorenz: Heredado ≥ errores comunes a los 4 modelos. 25 ≥ 23.
- **OK**: Lorenz: los cuatro grupos suman 114. 114.
- **OK** (informativa): % truncados en el global de prueba (informativo). 61.4 % en prueba frente a 59.1 % del corpus completo.

## 6. Notas

- Errores comunes a los 4 modelos (27/25/23) y errores visuales de la Tabla 84 (30/34/28): cifras de control dadas en la tarea, no recalculadas aquí.
- El F1 de la media de p_dep difiere del canónico `f35.<técnica>.finalista.f1_macro_mean` porque este último es la media de los F1 por semilla; ambos se reportan y el canónico no cambia.
- El clon de trabajo original tenía objetos de git faltantes (`git fsck`), así que el trabajo se hizo en un clon nuevo.
