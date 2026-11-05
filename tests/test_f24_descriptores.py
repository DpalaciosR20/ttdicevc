"""
F2.4 — la reimplementación de los descriptores (results/f24_descriptores/f24_descriptores.py)
coincide con el código original de la Tabla 11 (notebooks/historial/camera_ready_CE.ipynb,
celda 20), ejecutado en el mismo entorno sobre 10 imágenes de los insumos versionados.

Esta prueba muestra equivalencia de código en este entorno. No prueba equivalencia con la
ejecución de Kaggle: esa evidencia es la comparación de la Tabla 11 (tabla11_recalculada.csv).

Se omite si TT_DATA_ROOT no tiene las imágenes.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / 'notebooks' / 'historial' / 'camera_ready_CE.ipynb'
CELL = 20
ORIGINAL_FUNCS = ('shannon_entropy', 'edge_density', 'lbp_energy')
# 10 imágenes fijas: las tres técnicas, los tres splits y las dos clases
CASES = [('perlin', 'train_0000'), ('perlin', 'train_0007'), ('perlin', 'val_0003'), ('perlin', 'test_0010'),
         ('gray_scott', 'train_0001'), ('gray_scott', 'val_0000'), ('gray_scott', 'test_0005'),
         ('lorenz', 'train_0002'), ('lorenz', 'val_0001'), ('lorenz', 'test_0020')]


def load_f24():
    spec = importlib.util.spec_from_file_location('f24_descriptores',
                                                  REPO / 'results/f24_descriptores/f24_descriptores.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def original_namespace():
    """Ejecuta, sin modificarlos, los imports y las tres funciones de la celda 20."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    src = ''.join(nb['cells'][CELL]['source'])
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name in ORIGINAL_FUNCS)]
    assert {n.name for n in keep if isinstance(n, ast.FunctionDef)} == set(ORIGINAL_FUNCS)
    ns = {'np': np}  # la celda usa np, importado en la celda 2
    exec(compile(ast.Module(body=keep, type_ignores=[]), f'{NOTEBOOK.name}[celda {CELL}]', 'exec'), ns)
    return ns


@pytest.fixture(scope='module')
def f24():
    return load_f24()


@pytest.fixture(scope='module')
def images(f24):
    paths = f24.paths
    try:
        folders = {t: paths.images_dir(t) for t in paths.TECHNIQUES}
    except Exception as e:  # TT_DATA_ROOT no definida
        pytest.skip(f'insumos no disponibles: {e}')
    if not all(d.is_dir() for d in folders.values()):
        pytest.skip('TT_DATA_ROOT no tiene las imágenes')
    idx = pd.read_csv(paths.SAMPLE_INDEX, dtype={'sample_id': str}).set_index('sample_id')
    manifest = pd.read_csv(paths.DATA_MANIFEST)
    sha = dict(zip(manifest['rel_path'], manifest['sha256']))
    out = []
    for t, sid in CASES:
        name = idx.loc[sid, f'img_{t}']
        path = folders[t] / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha[f'{paths.REL["IMAGES"][t]}/{name}']
        out.append((t, sid, path))
    return out


def test_casos_cubren_tecnicas_splits_y_clases(f24):
    idx = pd.read_csv(f24.paths.SAMPLE_INDEX, dtype={'sample_id': str}).set_index('sample_id')
    assert len(CASES) == 10 and len(set(CASES)) == 10
    assert {t for t, _ in CASES} == set(f24.paths.TECHNIQUES)
    assert {s.split('_')[0] for _, s in CASES} == {'train', 'val', 'test'}
    assert set(idx.loc[[s for _, s in CASES], 'y_true']) == {0, 1}


def test_funciones_copiadas_literalmente(f24):
    """El cuerpo de cada función de F2.4 es el de la celda 20 (mismo AST)."""
    import inspect
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    orig = {n.name: ast.dump(n) for n in ast.parse(''.join(nb['cells'][CELL]['source'])).body
            if isinstance(n, ast.FunctionDef) and n.name in ORIGINAL_FUNCS}
    for name in ORIGINAL_FUNCS:
        mine = ast.parse(inspect.getsource(getattr(f24, name))).body[0]
        assert ast.dump(mine) == orig[name], name


def test_equivalencia_10_imagenes(f24, images):
    ns = original_namespace()
    for t, sid, path in images:
        img = np.array(Image.open(path).convert('L'))  # lectura de la celda 20, línea 40
        expected = {'shannon_entropy': ns['shannon_entropy'](img), 'contrast': img.std(),
                    'edge_density': ns['edge_density'](img), 'lbp_energy': ns['lbp_energy'](img)}
        mine_img, mode, size = f24.read_image(path.read_bytes())
        assert (mode, size) == ('L', [256, 256])
        assert np.array_equal(mine_img, img)
        got = f24.descriptors_of(mine_img)
        for k, v in expected.items():
            assert np.isclose(got[k], v, rtol=1e-12, atol=0), (t, sid, k, got[k], v)


def test_descriptores_csv_coincide(f24, images):
    """Si ya existe descriptores.csv, sus valores para estas 10 imágenes son los del código original."""
    csv = REPO / 'results/f24_descriptores/descriptores.csv'
    if not csv.is_file():
        pytest.skip('descriptores.csv aún no generado')
    df = pd.read_csv(csv, dtype={'sample_id': str}, float_precision='round_trip').set_index(['technique',
                                                                                              'sample_id'])
    ns = original_namespace()
    for t, sid, path in images:
        img = np.array(Image.open(path).convert('L'))
        row = df.loc[(t, sid)]
        assert np.isclose(row['shannon_entropy'], ns['shannon_entropy'](img), rtol=1e-12, atol=0)
        assert np.isclose(row['contrast'], img.std(), rtol=1e-12, atol=0)
        assert np.isclose(row['edge_density'], ns['edge_density'](img), rtol=1e-12, atol=0)
        assert np.isclose(row['lbp_energy'], ns['lbp_energy'](img), rtol=1e-12, atol=0)
