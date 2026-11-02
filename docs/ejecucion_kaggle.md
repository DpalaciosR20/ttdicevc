# Ejecución de cómputo pesado en Kaggle

> Convención del equipo para las tareas cuyo cómputo excede el equipo local (Windows, CPU).
> Claude Code prepara el código y las celdas; el integrante las ejecuta en Kaggle; Claude Code verifica e integra los resultados.

## 1. Cuándo se usa Kaggle

Antes de ejecutar cualquier corrida, Claude Code **estima su duración** con una corrida de prueba acotada (un subconjunto de datos, de réplicas o de configuraciones) y extrapola. La corrida se envía a Kaggle si se cumple alguna condición:

| Condición | Ejemplo |
|---|---|
| Requiere GPU | Entrenamiento o inferencia de CNN o ViT; BETO sobre muchas muestras |
| Duración local estimada > 20 min | Simulaciones de potencia, *bootstrap* masivo, extracción de características de las 3,408 imágenes |
| Memoria local insuficiente | — |

La estimación (tiempo de la prueba, factor de extrapolación y resultado) se registra en el log de la tarea. Si la corrida es local, se ejecuta sin preguntar; si va a Kaggle, Claude Code se detiene y entrega las celdas.

## 2. Reglas

1. **El código vive en el repositorio, no en las celdas.** Las celdas solo preparan el entorno y llaman a un módulo (`python -m ...`). Ninguna celda contiene lógica de análisis.
2. **Se ejecuta un commit publicado.** Antes de entregar las celdas, Claude Code hace commit y el integrante hace `git push`. La celda de clonado fija ese commit (`git checkout <sha>`), y los manifiestos registran ese mismo commit.
3. **Mismos insumos.** La primera celda verifica el dataset de Kaggle contra `meta/data_manifest.csv` (`python -m src.build_manifest --verify`). Si hay discrepancias, la ejecución se detiene.
4. **Las salidas se escriben en `/kaggle/working/results/...`**, con la misma ruta relativa que tendrán en el repositorio, y se empaquetan en un único `.zip`.
5. **El entorno queda registrado.** Una celda guarda `pip freeze`, la versión de Python, la GPU (`nvidia-smi`) y la duración total.
6. **Integración local verificada.** El integrante descarga el `.zip` y lo descomprime en la raíz del repositorio. Claude Code verifica que el commit registrado coincide con el entregado, que los SHA-256 de los manifiestos coinciden, que los archivos de predicciones pasan `src.predictions.validate` y que los logs no tienen errores. Solo entonces hace commit de los resultados.

## 3. Requisitos de la cuenta de Kaggle (una sola vez)

| Requisito | Detalle |
|---|---|
| Acceso a internet en el notebook | Requiere la cuenta verificada por teléfono; se activa en *Settings → Internet* |
| *Secret* `GITHUB_TOKEN` | Token *fine-grained* de GitHub con permiso de solo lectura (*Contents: read*) sobre el repositorio; se registra en *Add-ons → Secrets* |
| Dataset de insumos | `santiagosoria26ss/tt2-insumos`, agregado como *Input* del notebook |
| Acelerador | GPU T4 cuando la tarea la requiere; CPU en otro caso |

## 4. Plantilla de celdas

```python
# Celda 1 — Repositorio en el commit indicado
from kaggle_secrets import UserSecretsClient
import os, subprocess
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
REPO = "OWNER/REPO"            # [completar una vez]
COMMIT = "<sha>"               # lo indica Claude Code
subprocess.run(["git", "clone", "-q", f"https://{token}@github.com/{REPO}.git", "/kaggle/working/repo"], check=True)
os.chdir("/kaggle/working/repo")
subprocess.run(["git", "checkout", "-q", COMMIT], check=True)
```

```python
# Celda 2 — Entorno e insumos
os.environ["TT_DATA_ROOT"] = "/kaggle/input/<ruta verificada en docs/datos.md §2>"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!mkdir -p results/_entorno && pip freeze > results/_entorno/pip_freeze.txt && nvidia-smi > results/_entorno/gpu.txt || true
```

```python
# Celda 3 — Ejecución (la indica Claude Code)
!python -m <módulo> <argumentos> 2>&1 | tee results/<tarea>/<tarea>.log
```

```python
# Celda 4 — Empaquetado
!cd /kaggle/working/repo && zip -qr /kaggle/working/<tarea>_resultados.zip results/<tarea> results/predictions/<fase> results/_entorno
```

Junto con las celdas, Claude Code indica: el acelerador, la duración estimada, el commit, los archivos que deben aparecer en el `.zip` y el comando local de verificación que ejecutará al recibirlo.

## 5. Determinismo

Los resultados en GPU pueden diferir de los obtenidos en CPU (F0.3, hallazgo H2; F3.2, hallazgo 8). Toda salida registra el dispositivo en su manifiesto o log, y las tareas cuyo resultado debe ser bit a bit reproducible declaran el dispositivo de referencia en su especificación.
