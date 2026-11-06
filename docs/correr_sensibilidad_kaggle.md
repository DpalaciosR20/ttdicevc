# Correr la fase `sensibilidad` (F5.5, segundo mejor modelo) en Kaggle

> Insumo de la tarea F5.5 (SV) del plan v2 · Entrenamiento: CE · Convención general: `docs/ejecucion_kaggle.md` · Especificación: protocolo §8 y §12, matriz §4 y §5

El análisis de sensibilidad de F5.5 incluye el veredicto con el **segundo mejor modelo** de cada técnica (protocolo §8). Para eso se entrenan 15 modelos: los segundos de la clasificación de F3.3, con los hiperparámetros de la primera pasada (sin HPO) y las cinco semillas de la fase `final`, con inferencia sobre val y prueba.

| Técnica | Segundo modelo (`select_second`, F3.3) | Hiperparámetros (configuración común, matriz §3) |
|---|---|---|
| Gray-Scott | ResNet-18 *fine-tuning* | lr 3e-5, wd 0, lote 32 |
| Perlin | EfficientNet-B0 congelado | lr 1e-3, wd 0, lote 32 |
| Lorenz | EfficientNet-B0 *fine-tuning* | lr 3e-5, wd 0, lote 32 |

`run_grid.py --sensitivity` aplica la compuerta de prueba de `final` (protocolo limpio con `protocolo-v1` en el historial) y rechaza cualquier configuración distinta de las de `select_second`. Los modelos no se guardan, porque la matriz §9 no los pide para esta fase.

## Celdas

**Acelerador:** GPU T4. **Internet:** activado. **Input:** `santiagosoria26ss/tt2-insumos`.

**Celda 1: repositorio** (como en `docs/correr_final_kaggle.md`, con el `COMMIT` que indique Claude Code; conserva la línea de `seleccion-f33`).

**Celda 2: insumos y entorno**
```python
os.environ["TT_DATA_ROOT"] = "/kaggle/input/datasets/santiagosoria26ss/tt2-insumos/tt2-insumos"
!pip install -q -r requirements-kaggle.txt
!python -m src.build_manifest --verify
!mkdir -p results/f55_sensibilidad/entorno && pip freeze > results/f55_sensibilidad/entorno/pip_freeze.txt && nvidia-smi > results/f55_sensibilidad/entorno/gpu.txt
```

**Celda 3: comprobación sin entrenar** (segundos; no carga test)
```python
SEGUNDOS = ["gray_scott:resnet18:finetune", "perlin:efficientnet_b0:frozen", "lorenz:efficientnet_b0:finetune"]
for cfg in SEGUNDOS:
    tech = cfg.split(":")[0]
    !python -m src.training.run_grid --out-dir /kaggle/working/f55_dry/{tech} --evaluate-test --sensitivity --configs {cfg} --dry-run 2>&1 | head -6
```
Por técnica deben aparecer:
- `'role': 'second'` y `'matches': True`;
- `Fase: sensibilidad | CON TEST (protocolo …, etiqueta protocolo-v1)`;
- los hiperparámetros de la tabla;
- `5 corridas, 0 ya hechas, 5 pendientes`.

Si algo no cuadra, **detente**.

**Celda 4: entrenamiento y zip, en una sola celda** (≈ 15–20 min de T4; *Save Version → Save & Run All*)
```python
for cfg in SEGUNDOS:
    tech = cfg.split(":")[0]
    out = f"results/f55_sensibilidad/{tech}"
    os.makedirs(out, exist_ok=True)
    !python -m src.training.run_grid --out-dir {out} --evaluate-test --sensitivity --configs {cfg} 2>&1 | tee {out}/run_grid.log
!zip -qr /kaggle/working/f55_resultados.zip results/f55_sensibilidad results/predictions/sensibilidad
!ls -la /kaggle/working/f55_resultados.zip
```

## Contenido esperado de `f55_resultados.zip`

```
results/f55_sensibilidad/
├── entorno/{pip_freeze.txt, gpu.txt}
└── <técnica>/                        gray_scott, perlin, lorenz
    ├── runs/*.json                   5 por técnica, con f1_test y test_probs
    ├── summary.csv, summary_by_config.csv, grid_meta.json ({"fase": "sensibilidad"})
    └── run_grid.log
results/predictions/sensibilidad/*.csv   15 CSV F1.5 (val y test) con su .manifest.json
```

## Verificación local (Claude Code, al recibir el zip)

```bash
python -m src.predictions.validate "results/predictions/sensibilidad/*.csv"
```

Además, por corrida:
- fase `sensibilidad`, `selection.role = second` con `matches`;
- protocolo y commit correctos;
- hiperparámetros de la configuración común;
- F1 recalculado desde las probabilidades;
- etiquetas de val y prueba iguales a las de F3.5.

Las semillas 42, 7 y 123 deben reproducir bit a bit su corrida de F3.2, que tiene la misma configuración y los mismos hiperparámetros.
