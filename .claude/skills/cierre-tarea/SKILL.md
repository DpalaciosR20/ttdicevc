---
name: cierre-tarea
description: Genera el registro de cierre de una tarea del plan v2 en docs/bitacora/<ID>.md, a partir de la evidencia versionada, para el reporte técnico de TT2
disable-model-invocation: true
---
# Cierre de tarea: $ARGUMENTS

Genera `docs/bitacora/$ARGUMENTS.md`, el registro de cierre de la tarea $ARGUMENTS
de `plan/plan_tt2_v2.md`, en español formal académico. Sigue la estructura de
`docs/bitacora/_plantilla.md`. Ejemplo de referencia: `docs/bitacora/F0.3.md`.

## Paso 1. Reunir la evidencia (antes de escribir)

1. Lee la fila de $ARGUMENTS en `plan/plan_tt2_v2.md` (rol, entregable) y la
   amenaza a la validez que atiende (sección 5 del plan).
2. Lee `results/$ARGUMENTS/` (o la carpeta de resultados que indique el usuario):
   logs, CSV, JSON, XML de pytest y su README.
3. Obtén rama y commits con `git log --oneline` y `git branch --show-current`.
4. Lista lo que falta para llenar la plantilla y **pregúntamelo en un solo
   mensaje** antes de redactar: motivos de decisiones, fechas, entornos no
   registrados en los logs, decisiones pendientes del equipo.

## Paso 2. Redactar

Reglas obligatorias:

- **Toda cifra proviene de un archivo del repositorio.** Cita el archivo junto a
  la cifra o en el encabezado de la tabla. No uses cifras de memoria, de la
  conversación ni de `CLAUDE.md` si existe un archivo de resultados que las
  respalde.
- **Reporta los resultados tal como salieron.** Una prueba fallida se reporta como
  fallida, aunque se haya resuelto después. No reformules un resultado negativo
  como positivo ni omitas casos fallidos.
- **Separa hechos de interpretación.** Lo que se ejecutó y su salida literal va en
  «Resultados». Toda explicación causal va en «Hallazgos», marcada como
  *Interpretación*, y como *no verificado* si ninguna salida la respalda.
- **No inventes.** Si un dato no consta en la evidencia ni te lo di, escribe
  **[completar]** en su lugar. Nunca escribas un resultado de una corrida que no
  existe.
- **Conjunto de prueba.** No reportes métricas de test salvo en F3.5 o posteriores
  y con `docs/protocolo_preregistrado.md` congelado. Métricas siempre en par:
  `Exp X.X | F1-Val: 0.XXXX | F1-Test: 0.XXXX | Δ: ±0.XXXX`.
- **Desviaciones.** Todo cambio respecto del plan o del procedimiento previsto
  (corridas repetidas, errores de configuración, insumos distintos) se declara en
  «Hallazgos» como desviación.
- **Estado honesto.** Usa solo: *Cerrada*, *Cerrada con hallazgo transferido*
  (indica a qué tarea y responsable) o *Abierta* (con criterios pendientes).
- Respeta los roles (CE, SV, RP): los pendientes que no son del autor se asignan a
  su rol; no los resuelvas tú.

## Paso 3. Verificar antes de entregar

1. Haz una tabla de control (no va en la bitácora) con cada cifra del documento y
   el archivo y la línea de donde sale. Muéstramela. Si alguna cifra no tiene
   fuente, corrígela o cámbiala por **[completar]**.
2. Confirma que todo archivo citado en «Artefactos» existe con `ls`.
3. Lista los **[completar]** que quedaron.

## Paso 4. Commit

Solo cuando yo apruebe: commit de `docs/bitacora/$ARGUMENTS.md` (y de la evidencia
en `results/$ARGUMENTS/` si no estaba versionada) con el asunto
`[$ARGUMENTS] Bitácora de cierre`. No hagas push.
