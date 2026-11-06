"""F3.5: integridad de los resultados del entrenamiento final con prueba.

Comprueba lo que la verificación de F3.5 revisó al recibir el zip, para que
un cambio posterior en el repo no lo rompa en silencio: 15 corridas de los
finalistas con los hiperparámetros de F3.4, F1 recalculable desde las
probabilidades y huellas de modelos coherentes con meta/artefactos_manifest.csv.
"""
import csv
import glob
import json
import os

import numpy as np
import pytest
from sklearn.metrics import f1_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(REPO, 'results', 'f35_final')
FINALISTAS = {'gray_scott': ('efficientnet_b0', 'finetune'),
              'perlin': ('vit_b16', 'frozen'),
              'lorenz': ('vit_b16', 'frozen')}
SEMILLAS = {42, 7, 123, 2024, 99}


def _runs(tech):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(FINAL, tech, 'runs', '*.json')))]


@pytest.mark.parametrize('tech', sorted(FINALISTAS))
def test_corridas_de_finalistas(tech):
    runs = _runs(tech)
    assert {r['seed'] for r in runs} == SEMILLAS and len(runs) == 5
    params = json.load(open(os.path.join(REPO, 'results', 'f34_hpo', tech, 'hpo_result.json')))['decision']['params']
    for r in runs:
        assert (r['arch'], r['strategy']) == FINALISTAS[tech]
        assert r['fase'] == 'final' and r['evaluate_test'] and not r['smoke']
        assert r['protocol']['tag'] == 'protocolo-v1'
        assert r['selection']['matches'] and r['selection']['override'] is None
        assert (r['lr'], r['weight_decay'], r['batch_size']) == (params['lr'], params['weight_decay'], params['batch_size'])
        assert r['warning'] == ''


@pytest.mark.parametrize('tech', sorted(FINALISTAS))
def test_f1_recalculado(tech):
    for r in _runs(tech):
        for split, ref in (('val', r['best_f1_val']), ('test', r['f1_test'])):
            pred = (np.array(r[f'{split}_probs']) >= 0.5).astype(int)
            assert f1_score(r[f'{split}_true'], pred, average='macro') == pytest.approx(ref, abs=1e-12)


def test_manifiesto_de_artefactos():
    filas = list(csv.DictReader(open(os.path.join(REPO, 'meta', 'artefactos_manifest.csv'))))
    assert len(filas) == 15
    for f in filas:
        run = json.load(open(os.path.join(REPO, f['run_json'])))
        assert run['model_file']['sha256'] == f['sha256']
        assert os.path.basename(f['rel_path']) == run['model_file']['path']
    huellas = {os.path.basename(r['path']): r['sha256']
               for r in csv.DictReader(open(os.path.join(FINAL, 'models_sha256.csv')))}
    assert huellas == {os.path.basename(f['rel_path']): f['sha256'] for f in filas}


def test_predicciones_final():
    csvs = glob.glob(os.path.join(REPO, 'results', 'predictions', 'final', '*.csv'))
    assert len(csvs) == 15
