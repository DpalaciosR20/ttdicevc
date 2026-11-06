# results/f31_entrenamiento/

Evidencia de la tarea F3.1. Ver `docs/bitacora/F3.1.md`.

No hay evidencia versionada de la ejecución original de `tests/smoke_test_training.py`. Esta carpeta solo contiene la verificación retrospectiva.

## verificacion_retrospectiva/

Ejecutada el **2026-11-05** como parte de la reconstrucción retrospectiva de la bitácora. **No** se ejecutó al cierre de la tarea. Es de solo lectura: no modifica código, insumos ni resultados.

| Archivo | Contenido |
|---|---|
| `smoke_test_training.log` | Salida literal de `python tests/smoke_test_training.py` con los valores por omisión |

**Datos de la corrida.** Usa PNG sintéticos de ruido (24 de train, 6 de val y 6 de test), escritos en un directorio temporal, con 2 épocas. **No** usa las imágenes del proyecto. Por eso las cifras «F1-val» y «F1-test» del log se refieren a esos datos sintéticos, no a los splits del corpus.

**Entorno:**
- Windows 11 (10.0.26200), CPU.
- `.venv` del repositorio: Python 3.13.2, torch 2.11.0+cpu, torchvision 0.26.0+cpu.
- Commit de `main`: `b2cf9f8`.
- Duración: 118 s, incluida la descarga de los pesos ImageNet de EfficientNet-B0 y ViT-B/16 al caché local de torch.

**Reproducir (desde la raíz del repo):** `python tests/smoke_test_training.py`
