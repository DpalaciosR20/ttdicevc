# Análisis de error por texto (prueba, n = 114)

Análisis descriptivo sobre predicciones ya existentes. Visual = media de p_dep de las 5 semillas del finalista (fase `final`, F3.5): Gray-Scott efficientnet_b0 finetune, Perlin vit_b16 frozen y Lorenz vit_b16 frozen. BETO = Exp 4.1 (F2.2). Regla de decisión: p_dep > 0.5.

## Matriz 2×2 por técnica (GS / Perlin / Lorenz)

| | Visual acierta | Visual falla |
|---|---|---|
| **BETO acierta** | Conservado: 81 / 75 / 79 | Perdido en la imagen: 1 / 7 / 3 |
| **BETO falla** | Recuperado: 3 / 5 / 7 | Error heredado: 29 / 27 / 25 |

## Caracterización por grupo

FN/FP: entre los textos mal clasificados del grupo, cuántos son depresivos (FN) y cuántos no depresivos (FP). En Perdido, Heredado y Global se cuentan los errores del visual; en Recuperado, los de BETO. Pandemia: textos con al menos un término de la lista del Módulo 1 (regex con límites de palabra sobre el texto en minúsculas).

| Técnica | Grupo | n | Mediana de tokens BETO | % truncados (>128) | % depresivo | FN / FP | Pandemia: n (%) |
|---|---|---|---|---|---|---|---|
| Gray-Scott | Conservado | 81 | 208.0 | 59.3 | 34.6 | — | 17 (21.0) |
| Gray-Scott | Perdido en la imagen | 1 | 164.0 | 100.0 | 0.0 | 0 / 1 | 0 (0.0) |
| Gray-Scott | Recuperado | 3 | 223.0 | 66.7 | 100.0 | 3 / 0 | 1 (33.3) |
| Gray-Scott | Error heredado | 29 | 297.0 | 65.5 | 44.8 | 13 / 16 | 5 (17.2) |
| Gray-Scott | Global (prueba) | 114 | 231.5 | 61.4 | 38.6 | 13 / 17 | 23 (20.2) |
| Perlin | Conservado | 75 | 192.0 | 58.7 | 36.0 | — | 15 (20.0) |
| Perlin | Perdido en la imagen | 7 | 370.0 | 71.4 | 14.3 | 1 / 6 | 2 (28.6) |
| Perlin | Recuperado | 5 | 75.0 | 40.0 | 80.0 | 4 / 1 | 1 (20.0) |
| Perlin | Error heredado | 27 | 301.0 | 70.4 | 44.4 | 12 / 15 | 5 (18.5) |
| Perlin | Global (prueba) | 114 | 231.5 | 61.4 | 38.6 | 13 / 21 | 23 (20.2) |
| Lorenz | Conservado | 79 | 208.0 | 59.5 | 35.4 | — | 16 (20.3) |
| Lorenz | Perdido en la imagen | 3 | 164.0 | 66.7 | 0.0 | 0 / 3 | 1 (33.3) |
| Lorenz | Recuperado | 7 | 370.0 | 57.1 | 57.1 | 4 / 3 | 3 (42.9) |
| Lorenz | Error heredado | 25 | 273.0 | 68.0 | 48.0 | 12 / 13 | 3 (12.0) |
| Lorenz | Global (prueba) | 114 | 231.5 | 61.4 | 38.6 | 12 / 16 | 23 (20.2) |

## Verificaciones

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

## Textos representativos de Gray-Scott

Los 2 sample_id más cercanos a la mediana de tokens del grupo (desempate por sample_id). No se copian los textos.

- Error heredado: `test_0062`, `test_0026`
- Perdido en la imagen: `test_0000`
- Recuperado: `test_0057`, `test_0039`
