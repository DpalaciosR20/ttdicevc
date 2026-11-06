"""F3.7: integridad de los registros de mapas frente a los modelos de F3.5/F3.6."""
import csv
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, 'results', 'f37_mapas')
MANIFEST = os.path.join(REPO, 'meta', 'artefactos_manifest.csv')


def _records():
    return [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob(os.path.join(RES, 'records', '*.json')))]


def test_treinta_registros_completos():
    recs = _records()
    assert len(recs) == 30
    assert json.load(open(os.path.join(RES, 'maps_meta.json'))) == {'task': 'F3.7', 'smoke': False, 'limit': None}
    modelos = {os.path.basename(r['rel_path']): r for r in csv.DictReader(open(MANIFEST)) if r['task'] in ('F3.5', 'F3.6')}
    vistos = set()
    for r in recs:
        run = json.load(open(os.path.join(REPO, r['run_json'])))
        assert r['n'] == 228 and not r['smoke'] and r['target_logit_class'] == 1
        assert r['model_sha256_verified'] == run['model_file']['sha256'] == modelos[r['run'] + '.pt']['sha256']
        assert max(r['p_dep_max_abs_diff'].values()) <= 1e-4 and r['logits_hooked_max_abs_diff'] <= 1e-4
        esperado = {'map': 'grad_cam'} if r['arch'] == 'efficientnet_b0' else \
            {'map': 'gradient_weighted_rollout', 'backbone_map': 'attention_rollout'}
        assert r['methods'] == esperado
        vistos.add((r['fase'], r['technique'], r['seed']))
    assert len(vistos) == 30


def test_manifiestos_de_mapas():
    recs = {r['run']: r for r in _records()}
    mm = list(csv.DictReader(open(os.path.join(RES, 'maps_manifest.csv'))))
    filas = [r for r in csv.DictReader(open(MANIFEST)) if r['task'] == 'F3.7']
    assert len(mm) == len(filas) == 30
    for m in mm:
        assert m['sha256'] == recs[m['run']]['npz']['sha256']
    for f in filas:
        rec = json.load(open(os.path.join(REPO, f['run_json'])))
        assert f['sha256'] == rec['npz']['sha256'] and f['kind'] == 'maps_npz'
        assert f['dataset'] == 'palaciosroblesdiego/tt2-mapas'


def test_verificacion_de_modelos_en_kaggle():
    rep = json.load(open(os.path.join(RES, 'modelos_verificacion.json')))
    assert rep['passed'] and rep['n_ok'] == rep['n_expected'] == 30


def test_verificacion_de_tt2_mapas_y_resumen():
    rep = json.load(open(os.path.join(RES, 'mapas_verificacion.json')))
    assert rep['passed'] and rep['n_ok'] == rep['n_expected'] == 30 and rep['task'] == 'F3.7' and not rep['ambiguous']
    assert all(f['found_at'].startswith('datasets/palaciosroblesdiego/tt2-mapas/ttdicevc/results/f37_mapas/maps/')
               for f in rep['files'])
    recs = {r['run'] + '.npz': r['npz']['sha256'] for r in _records()}
    assert {f['file']: f['sha256'] for f in rep['files']} == recs
    filas = list(csv.DictReader(open(os.path.join(RES, 'resumen', 'per_run.csv'))))
    assert len(filas) == 50 and {f['run'] for f in filas} == {r['run'] for r in _records()}
