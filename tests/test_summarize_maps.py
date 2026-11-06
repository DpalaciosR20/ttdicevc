"""F3.7: summarize_maps con npz sintéticos."""
import csv
import json

import numpy as np

from src.interpretability import summarize_maps as sm
from src.verify_artefacts import sha256


def _npz(path, maps, backbone=None):
    n = len(maps)
    arr = {'sample_id': np.array([f'val_{i:04d}' for i in range(n // 2)] + [f'test_{i:04d}' for i in range(n - n // 2)]),
           'split': np.array(['val'] * (n // 2) + ['test'] * (n - n // 2)),
           'y_true': np.array([0, 1] * (n // 2), dtype=np.int8), 'map': maps.astype(np.float16)}
    if backbone is not None:
        arr['backbone_map'] = backbone.astype(np.float16)
    np.savez_compressed(path, **arr)


def test_resumen(tmp_path):
    rng = np.random.default_rng(0)
    root, recs = tmp_path / 'in', tmp_path / 'records'
    root.mkdir(); recs.mkdir()
    real = rng.random((4, 224, 224))
    real[0] = 0                                   # un mapa constante
    _npz(root / 'r.npz', real, backbone=real)
    _npz(root / 'p.npz', real)                    # control idéntico al real
    for run, fase, f, methods in (('r', 'final', 'r.npz', {'map': 'gradient_weighted_rollout', 'backbone_map': 'attention_rollout'}),
                                  ('p', 'control_permutacion', 'p.npz', {'map': 'gradient_weighted_rollout'})):
        (recs / f'{run}.json').write_text(json.dumps({'run': run, 'fase': fase, 'technique': 'perlin', 'arch': 'vit_b16',
                                                     'seed': 42, 'methods': methods,
                                                     'npz': {'path': f'x/{f}', 'sha256': sha256(str(root / f))}}))
    out = tmp_path / 'out'
    assert sm.main(['--root', str(root), '--records', str(recs), '--out-dir', str(out)]) == 0
    rows = {(r['run'], r['mapa']): r for r in csv.DictReader(open(out / 'per_run.csv'))}
    assert float(rows[('r', 'map')]['frac_constantes']) == 0.25
    assert abs(float(rows[('r', 'map')]['corr_con_backbone']) - 1) < 1e-6
    assert abs(float(rows[('p', 'map')]['corr_con_real']) - 1) < 1e-6
    z = np.load(out / 'class_means.npz')
    assert z['r|map|val|y1'].shape == (56, 56)


def test_concentracion_uniforme_y_puntual():
    u = np.ones((1, 56, 56))
    d = np.zeros((1, 56, 56)); d[0, 3, 3] = 1
    _, ent_u, top_u = sm.concentration(u)
    _, ent_d, top_d = sm.concentration(d)
    assert abs(ent_u[0] - np.log2(56 * 56)) < 1e-9 and abs(top_u[0] - 0.1) < 1e-3
    assert ent_d[0] == 0 and top_d[0] == 1
