# Cómo arrancar con Claude Code en este repo

## 1. Descomprime esta carpeta e inicializa git

```bash
cd ttdicevc
git init
git add .
git commit -m "Setup inicial: estructura del proyecto, codigo de generacion y entrenamiento, documentacion"
```

Sube el repo a GitHub (o el proveedor que usen) y que Marco y Santiago lo clonen — con esto los tres arrancan con exactamente el mismo contexto.

## 2. Abre Claude Code en la carpeta del proyecto

```bash
cd ttdicevc
claude
```

(Si no tienes Claude Code instalado: `npm install -g @anthropic-ai/claude-code`, o revisa las opciones de VS Code/JetBrains.)

## 3. Primer mensaje para pegarle a Claude Code

```
Acabo de clonar este repo. Lee CLAUDE.md, docs/hallazgos_metodologicos.md y
docs/artefactos_datos.md para entender el proyecto completo.

Después de leerlos, confírmame:
1. Que entendiste el pipeline (Módulos 1, 2, 3) y los 4 hallazgos
   metodológicos ya resueltos (no deben repetirse).
2. Revisa src/generation/ y src/training/ — dime si encuentras algún
   problema, inconsistencia, o algo que no esté probado.
3. Instala las dependencias necesarias (ver README.md) en un entorno virtual
   si no existe uno ya.
4. Confirma la estructura de carpetas contra lo que describe CLAUDE.md.

Mi rol en este proyecto es: [CE / SV / RP -- especifica el tuyo]
Mi tarea de esta semana es: [pega la tarea correspondiente de
plan/plan_tt2_por_semana.md]
```

## 4. Para las siguientes sesiones

No hace falta repetir el contexto — Claude Code lee `CLAUDE.md` automáticamente cada vez que lo abres en esta carpeta. Solo dile en qué tarea de `plan/plan_tt2_por_semana.md` estás trabajando.

## Nota sobre entrenamiento con GPU

Claude Code no tiene GPU propia. El flujo esperado es: usar Claude Code para desarrollar y mantener el código en `src/` (limpio, versionado, probado), y de ahí copiar/subir ese código ya validado a Kaggle o Colab para el entrenamiento real. Si en algún momento quieren automatizar la subida a Kaggle (API de Kaggle), es un paso posterior — no es necesario para arrancar.
