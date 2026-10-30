"""
write.py
────────
Escritura de archivos de predicciones por muestra y su manifiesto
(docs/especificacion_predicciones.md §3-§4). La validación sigue en
validate.py; este módulo produce archivos que la pasan.

Uso:
    from src.predictions.write import write_predictions
    csv_path = write_predictions(df, fase='referencia', task_id='F2.2',
                                 config={...}, inputs={...}, model_selection={...})

df: DataFrame con sample_id, split, y_true, technique, arch, strategy, seed,
p_dep (una sola combinación technique/arch/strategy/seed). p_dep se escribe
con repr de float64 (ida y vuelta exacta, sin redondeo).
"""

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone

import pandas as pd

from src import paths
from src.predictions import schema
from src.training.umap_io import sha256

PREDICTIONS_DIR = paths.RESULTS_DIR / 'predictions'
LIBS = ('numpy', 'pandas', 'scikit-learn', 'torch', 'transformers', 'tokenizers', 'joblib')


def git_commit():
    """HEAD del repositorio; termina en '-dirty' si hay cambios sin commitear en src/ o results/."""
    head = subprocess.check_output(['git', '-C', str(paths.REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(paths.REPO), 'status', '--porcelain', '--',
                                     'src', 'results/*/*.py'], text=True).strip()
    return head + ('-dirty' if dirty else '')


def environment():
    versions = {}
    for lib in LIBS:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            pass
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {'python': platform.python_version(), 'libraries': versions,
            'platform': platform.platform(), 'gpu': gpu}


def write_predictions(df, fase, task_id, config, inputs, model_selection, extra=None):
    """Escribe results/predictions/{fase}/{technique}__{arch}__{strategy}__seed{seed}.csv
    y su manifiesto. Devuelve la ruta del CSV."""
    if fase not in schema.FASES:
        raise ValueError(f'fase {fase!r} no está en {sorted(schema.FASES)}')
    if list(df.columns) != schema.COLUMNS:
        raise ValueError(f'columnas {list(df.columns)}; se esperan {schema.COLUMNS}')
    ident = {c: df[c].unique() for c in ('technique', 'arch', 'strategy', 'seed')}
    if any(len(v) != 1 for v in ident.values()):
        raise ValueError(f'un archivo lleva una sola combinación technique/arch/strategy/seed: {ident}')

    df = df.astype(schema.DTYPES)
    order = df['split'].map({s: i for i, s in enumerate(schema.SPLIT_ORDER)})
    df = df.assign(_o=order).sort_values(['_o', 'sample_id'], kind='stable').drop(columns='_o')

    out_dir = PREDICTIONS_DIR / fase
    name = schema.file_name(*(ident[c][0] for c in ('technique', 'arch', 'strategy', 'seed')))
    csv_path = out_dir / name
    paths.assert_outside_data_root(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, lineterminator='\n', float_format=None)

    manifest = {
        'task_id': task_id,
        'fase': fase,
        'git_commit': git_commit(),
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'config': config,
        'inputs': inputs,
        'model_selection': model_selection,
        'environment': environment(),
        'csv_sha256': sha256(csv_path),
        'n_rows': {s: int((df['split'] == s).sum()) for s in schema.SPLIT_ORDER if (df['split'] == s).any()},
    }
    if extra:
        manifest.update(extra)
    missing = [f for f in schema.MANIFEST_FIELDS if f not in manifest]
    assert not missing, missing
    mpath = csv_path.with_name(schema.manifest_name(name))
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    return csv_path
