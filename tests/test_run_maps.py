"""F3.7: run_maps de punta a punta en CPU con imágenes sintéticas y un
EfficientNet-B0 sin pesos de ImageNet (tamaño real, unos segundos)."""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'training'))
torch = pytest.importorskip('torch')
from PIL import Image  # noqa: E402

from src.interpretability import run_maps  # noqa: E402

LIMIT = 3


def _setup(tmp_path, monkeypatch):
    from build_model import build_model
    from data import get_transforms
    from src.training.run_grid import load_sample_index
    from src import paths

    index = load_sample_index(str(paths.SAMPLE_INDEX))
    root = tmp_path / 'datos'
    img_dir = root / paths.REL['IMAGES']['gray_scott']
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split in ('val', 'test'):
        for idx, y, names in index[split]:
            Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8), 'L').save(img_dir / names['gray_scott'])
    monkeypatch.setenv('TT_DATA_ROOT', str(root))

    torch.manual_seed(0)
    model = build_model('efficientnet_b0', freeze_backbone=False, pretrained=False).eval()
    models_root = tmp_path / 'modelos'
    models_root.mkdir()
    mpath = models_root / 'gs_test__seed42.pt'
    torch.save(model.state_dict(), mpath)
    from umap_io import sha256
    prep = {'size': 224, 'resize_mode': 'resize', 'interpolation': 'bilinear', 'antialias': True,
            'hflip': True, 'vit': 'same', 'variant': 'px224-flip'}
    tf = get_transforms(False, img_size=224, hflip=False)
    probs = {}
    for split in ('val', 'test'):
        xs = torch.stack([tf(Image.open(img_dir / names['gray_scott']).convert('L'))
                          for _, _, names in index[split][:LIMIT]])
        with torch.no_grad():
            probs[split] = torch.softmax(model(xs).double(), 1)[:, 1].tolist() + [0.5] * (114 - LIMIT)
    run = {'technique': 'gray_scott', 'arch': 'efficientnet_b0', 'strategy': 'finetune', 'seed': 42,
           'fase': 'final', 'preprocessing': prep, 'permute_labels': None,
           'val_true': [r[1] for r in index['val']], 'test_true': [r[1] for r in index['test']],
           'val_probs': probs['val'], 'test_probs': probs['test'],
           'model_file': {'path': mpath.name, 'sha256': sha256(str(mpath))}}
    src_dir = tmp_path / 'fuente'
    (src_dir / 'gray_scott' / 'runs').mkdir(parents=True)
    (src_dir / 'gray_scott' / 'runs' / 'gs_test__seed42.json').write_text(json.dumps(run))
    return src_dir, models_root, run


def test_run_maps_punta_a_punta(tmp_path, monkeypatch):
    src_dir, models_root, run = _setup(tmp_path, monkeypatch)
    out = tmp_path / 'salida'
    args = ['--models-root', str(models_root), '--out-dir', str(out), '--sources', str(src_dir),
            '--limit', str(LIMIT), '--batch-size', '2']
    assert run_maps.main(args) == 0
    rec = json.loads((out / 'records' / 'gs_test__seed42.json').read_text())
    assert rec['smoke'] and rec['n'] == 2 * LIMIT and rec['methods'] == {'map': 'grad_cam'}
    assert max(rec['p_dep_max_abs_diff'].values()) < 1e-6
    z = np.load(out / 'maps' / 'gs_test__seed42.npz')
    assert list(z['sample_id']) == [f'val_{i:04d}' for i in range(LIMIT)] + [f'test_{i:04d}' for i in range(LIMIT)]
    assert z['map'].shape == (2 * LIMIT, 224, 224) and z['map'].dtype == np.float16
    assert float(z['map'].min()) >= 0 and float(z['map'].max()) <= 1
    raw = z['map_raw'].astype(np.float32) * z['map_raw_scale'][:, None, None]
    assert (raw >= 0).all() and np.allclose(raw.reshape(len(raw), -1).max(1), z['map_raw_scale'], rtol=1e-3)
    assert 'backbone_map' not in z.files
    # reanudable: la segunda vez no recalcula
    assert run_maps.main(args) == 0
    assert (out / 'maps_manifest.csv').read_text().count('\n') == 2


def test_run_maps_rechaza_modelo_alterado(tmp_path, monkeypatch):
    src_dir, models_root, run = _setup(tmp_path, monkeypatch)
    p = models_root / run['model_file']['path']
    p.write_bytes(p.read_bytes() + b'x')
    out = tmp_path / 'salida'
    assert run_maps.main(['--models-root', str(models_root), '--out-dir', str(out), '--sources', str(src_dir),
                          '--limit', str(LIMIT)]) == 1
    assert not (out / 'records' / 'gs_test__seed42.json').exists()


def test_run_maps_rechaza_p_dep_distinto(tmp_path, monkeypatch):
    src_dir, models_root, run = _setup(tmp_path, monkeypatch)
    run['val_probs'][0] = 1 - run['val_probs'][0]
    (src_dir / 'gray_scott' / 'runs' / 'gs_test__seed42.json').write_text(json.dumps(run))
    out = tmp_path / 'salida'
    assert run_maps.main(['--models-root', str(models_root), '--out-dir', str(out), '--sources', str(src_dir),
                          '--limit', str(LIMIT)]) == 1
