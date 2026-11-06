"""F3.6: integridad del control de permutación (matriz §7).

15 corridas de los finalistas con y_train permutado por default_rng(99),
val y prueba con las etiquetas reales (iguales a F3.5), hiperparámetros de
F3.4, F1 recalculable y huellas coherentes con meta/artefactos_manifest.csv.
"""
import csv
import glob
import json
import os

import numpy as np
import pytest
from sklearn.metrics import f1_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, 'results', 'f36_permutacion')
FINALISTAS = {'gray_scott': ('efficientnet_b0', 'finetune', 'px224-flip'),
              'perlin': ('vit_b16', 'frozen', 'px224-noflip'),
              'lorenz': ('vit_b16', 'frozen', 'px224-noflip')}
SEMILLAS = {42, 7, 123, 2024, 99}
PERM = [int(i) for i in np.random.default_rng(99).permutation(908)]


def _runs(tech):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, tech, 'runs', '*.json')))]


@pytest.mark.parametrize('tech', sorted(FINALISTAS))
def test_corridas_permutadas(tech):
    arch, strat, var = FINALISTAS[tech]
    runs = _runs(tech)
    assert {r['seed'] for r in runs} == SEMILLAS and len(runs) == 5
    params = json.load(open(os.path.join(REPO, 'results', 'f34_hpo', tech, 'hpo_result.json')))['decision']['params']
    for r in runs:
        assert (r['arch'], r['strategy'], r['variant']) == (arch, strat, f'{var}-perm99')
        assert r['fase'] == 'control_permutacion' and r['evaluate_test'] and not r['smoke']
        assert r['permute_labels'] == 99 and r['permutation_indices'] == PERM
        assert r['protocol']['tag'] == 'protocolo-v1' and r['selection']['matches']
        assert (r['lr'], r['weight_decay'], r['batch_size']) == (params['lr'], params['weight_decay'], params['batch_size'])
        ref = json.load(open(os.path.join(REPO, 'results', 'f35_final', tech, 'runs',
                                          f"{tech}__{arch}__{strat}__{var}__seed{r['seed']}.json")))
        assert r['val_true'] == ref['val_true'] and r['test_true'] == ref['test_true']
        assert r['class_weights'] == ref['class_weights']
        for split, f1 in (('val', r['best_f1_val']), ('test', r['f1_test'])):
            pred = (np.array(r[f'{split}_probs']) > 0.5).astype(int)
            assert f1_score(r[f'{split}_true'], pred, average='macro') == pytest.approx(f1, abs=1e-12)


def test_permutacion_conserva_conteo_y_no_alinea():
    import pandas as pd
    si = pd.read_csv(os.path.join(REPO, 'meta', 'sample_index.csv'))
    y = si[si.split == 'train'].sort_values('idx').y_true.to_numpy()
    yp = y[PERM]
    assert yp.sum() == y.sum() == 353
    assert abs(np.corrcoef(y, yp)[0, 1]) < 0.05


def test_manifiesto_f36():
    filas = [f for f in csv.DictReader(open(os.path.join(REPO, 'meta', 'artefactos_manifest.csv'))) if f['task'] == 'F3.6']
    assert len(filas) == 15
    for f in filas:
        run = json.load(open(os.path.join(REPO, f['run_json'])))
        assert run['model_file']['sha256'] == f['sha256'] and os.path.basename(f['rel_path']) == run['model_file']['path']
    assert len(glob.glob(os.path.join(REPO, 'results', 'predictions', 'control_permutacion', '*.csv'))) == 15


def test_verificacion_de_artefactos_f35_y_f36():
    rep = json.load(open(os.path.join(RES, 'artefactos_verificacion.json')))
    assert rep['passed'] and rep['n_ok'] == rep['n_expected'] == 30 and rep['task'] is None
    filas = {os.path.basename(f['rel_path']): f for f in
             csv.DictReader(open(os.path.join(REPO, 'meta', 'artefactos_manifest.csv'))) if f['task'] in ('F3.5', 'F3.6')}
    assert len(filas) == 30
    for f in rep['files']:
        fila = filas[f['file']]
        assert f['status'] == 'ok' and f['sha256'] == fila['sha256']
        assert f['found_at'] == f"datasets/{fila['dataset']}/{fila['rel_path']}"
