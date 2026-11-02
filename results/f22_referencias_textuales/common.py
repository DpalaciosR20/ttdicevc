"""
common.py
─────────
Utilidades compartidas por los scripts de F2.2 (docs/especificacion_f22.md):
bitácora, verificación de SHA-256 contra meta/data_manifest.csv y lectura
del índice maestro.
"""

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import paths  # noqa: E402
from src.build_manifest import dir_digest  # noqa: E402
from src.training.umap_io import sha256  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TASK_ID = 'F2.2'
FASE = 'referencia'
SEED = 42
PRED_SPLITS = ('val', 'test')


class Log:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, msg=''):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def write(self):
        self.path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8', newline='\n')


class Stop(RuntimeError):
    """Verificación que obliga a detener (especificacion_f22 §3)."""


def manifest_rows():
    with open(paths.DATA_MANIFEST, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def verify_file(rel, log):
    """SHA-256 de un archivo de TT_DATA_ROOT contra su fila del manifiesto."""
    row = next(r for r in manifest_rows() if r['rel_path'] == rel and r['kind'] == 'file')
    digest = sha256(paths.resolve(rel))
    if digest != row['sha256']:
        raise Stop(f'{rel}: sha256 {digest} != manifiesto {row["sha256"]}')
    log(f'  sha256 OK  {rel}  {digest}')
    return digest


def verify_dir(rel, log):
    """SHA-256 de cada archivo de una carpeta contra el manifiesto y su SHA agregado."""
    rows = manifest_rows()
    files = [r for r in rows if r['kind'] == 'file' and r['rel_path'].startswith(rel + '/')]
    present, _ = paths.list_files(paths.resolve(rel))
    if sorted(f'{rel}/{p.name}' for p in present) != sorted(r['rel_path'] for r in files):
        raise Stop(f'{rel}: los archivos presentes no coinciden con el manifiesto')
    for r in files:
        verify_file(r['rel_path'], log)
    agg = next(r for r in rows if r['rel_path'] == rel and r['kind'] == 'dir')['sha256']
    if dir_digest(files) != agg:
        raise Stop(f'{rel}: sha256 agregado no coincide con el manifiesto')
    log(f'  sha256 OK  {rel}/ (agregado)  {agg}')
    return agg


def load_index():
    import pandas as pd
    idx = pd.read_csv(paths.SAMPLE_INDEX, dtype={'sample_id': str, 'split': str})
    return {s: idx[idx['split'] == s].sort_values('idx').reset_index(drop=True) for s in paths.SPLITS}
