# TTDICEVC

Transformación de Textos Depresivos en Imágenes mediante Visión Computacional — Trabajo Terminal, ESCOM-IPN.

Equipo: Diego Palacios Robles, Marco Antonio Mánica Quintero, Santiago Elihu Soria Solís. Director: Dr. Abdiel Reyes Vera.

**Estado:** paper aceptado en WITCOM 2026. Trabajando en TT2 (Trabajo Terminal II), defensa 17 nov–1 dic 2026.

## Para arrancar con Claude Code

Este repo tiene un `CLAUDE.md` en la raíz que Claude Code lee automáticamente — da el contexto completo del proyecto sin que tengas que reexplicarlo. Solo abre Claude Code en esta carpeta y empieza a trabajar.

## Estructura

- `src/generation/` — generadores class-blind de las 3 técnicas (Gray-Scott, Perlin, Lorenz).
- `src/training/` — entrenamiento generalizado a 3 arquitecturas (ResNet-18, EfficientNet-B0, ViT-B-16).
- `notebooks/` — notebooks de Kaggle/Colab (limpiar outputs pesados antes de commitear).
- `docs/` — hallazgos metodológicos, ubicación de artefactos de datos, resumen del camera-ready.
- `plan/` — plan de trabajo de TT2 por semana y rol.

## Datos

Los datasets (imágenes generadas, embeddings) NO están en este repo por tamaño — ver `docs/artefactos_datos.md` para su ubicación y estructura.

## Requisitos

```
torch
torchvision
numpy
scipy
scikit-learn
scikit-image
statsmodels
umap-learn
joblib
Pillow
matplotlib
noise          # solo para generación de Perlin -- puede requerir Build Tools en Windows, usar Colab/Kaggle si falla
```
