"""
paths.py
────────
Rutas de todos los insumos a partir de TT_DATA_ROOT (docs/datos.md, regla 1
y sección 3). Ningún script del repositorio escribe una ruta de datos a mano.

TT_DATA_ROOT por entorno (docs/datos.md §2):
    Local (Windows)  G:\\Mi unidad\\tt2-insumos
    Colab            /content/drive/MyDrive/tt2-insumos
    Kaggle           /kaggle/input/datasets/santiagosoria26ss/tt2-insumos  [completar]

La carpeta de datos es de solo lectura (regla 2): toda salida va al
repositorio; assert_outside_data_root() lo hace cumplir antes de escribir.

Vocabulario de técnicas = nombres de carpeta; el prefijo de archivo de
Gray-Scott (rd_) es la única excepción y se traduce aquí (datos.md §7). Mismo
mapeo que PREFIXES en src/training/data.py, que no se importa para no
arrastrar torch.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
META_DIR = REPO / 'meta'
RESULTS_DIR = REPO / 'results'
SAMPLE_INDEX = META_DIR / 'sample_index.csv'
DATA_MANIFEST = META_DIR / 'data_manifest.csv'

SPLITS = ('train', 'val', 'test')
EXPECTED_SIZES = {'train': 908, 'val': 114, 'test': 114}
TECHNIQUES = ('perlin', 'gray_scott', 'lorenz')
PREFIX = {'perlin': 'perlin', 'gray_scott': 'rd', 'lorenz': 'lorenz'}
# El CSV de validación se llama validation.csv; en el proyecto el split es 'val'
CSV_FILENAME = {'train': 'train.csv', 'val': 'validation.csv', 'test': 'test.csv'}

# Rutas relativas a TT_DATA_ROOT (datos.md §3), con '/' para que el manifiesto
# sea idéntico en Windows, Colab y Kaggle
REL = {
    'CSV': {s: f'processed/{CSV_FILENAME[s]}' for s in SPLITS},
    'UMAP_PARAMS': 'umap_params.pkl',
    'EMB_41': 'exp_4_1_final/Exp_4.1_BETO_HPO_embeddings.pkl',
    'CKPT_41': 'exp_4_1_final/checkpoint-342',
    'IMAGES': {t: f'images_grayscale_classblind/{t}' for t in TECHNIQUES},
}


# Archivos que el sistema operativo o Google Drive para escritorio crean en
# las carpetas (p. ej. desktop.ini oculto). No son insumos: si se contaran, el
# SHA-256 agregado de una carpeta cambiaría entre Windows, Colab y Kaggle
SYSTEM_FILES = {'desktop.ini', 'thumbs.db', '.ds_store'}


def list_files(folder):
    """(archivos, ignorados) de una carpeta, ordenados por nombre; excluye SYSTEM_FILES."""
    files, ignored = [], []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file():
            (ignored if p.name.lower() in SYSTEM_FILES else files).append(p)
    return files, ignored


def _user_env_windows(name):
    """Variable de entorno de usuario en Windows (si la sesión se abrió antes de definirla)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
            return winreg.QueryValueEx(k, name)[0]
    except Exception:
        return None


def data_root():
    root = os.environ.get('TT_DATA_ROOT')
    if not root and sys.platform == 'win32':
        root = _user_env_windows('TT_DATA_ROOT')
    if not root:
        raise RuntimeError('TT_DATA_ROOT no está definida (ver docs/datos.md §2)')
    path = Path(root)
    if not path.is_dir():
        raise RuntimeError(f'TT_DATA_ROOT={root} no existe o no es una carpeta')
    return path


def resolve(rel):
    return data_root() / rel


def csv_path(split):
    return resolve(REL['CSV'][split])


def umap_params_path():
    return resolve(REL['UMAP_PARAMS'])


def emb_41_path():
    return resolve(REL['EMB_41'])


def ckpt_41_path():
    return resolve(REL['CKPT_41'])


def images_dir(technique):
    return resolve(REL['IMAGES'][technique])


def image_name(technique, split, idx, y):
    """{prefijo}_{split}_{NNNN}_label{y}.png (datos.md §7)."""
    return f'{PREFIX[technique]}_{split}_{idx:04d}_label{int(y)}.png'


def sample_id(split, idx):
    return f'{split}_{idx:04d}'


def assert_outside_data_root(path):
    """Regla 2 de datos.md: nada se escribe dentro de TT_DATA_ROOT."""
    try:
        root = data_root().resolve()
    except RuntimeError:
        return
    target = Path(path).resolve()
    if target == root or root in target.parents:
        raise PermissionError(f'{target} está dentro de TT_DATA_ROOT ({root}), que es de solo lectura')
