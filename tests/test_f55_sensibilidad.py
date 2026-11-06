"""F5.5 (entrenamiento de CE): integridad de la fase sensibilidad.

15 corridas de los segundos de F3.3 (select_second) con la configuración
común, prueba con etiquetas reales iguales a F3.5, F1 recalculable y
reproducción bit a bit de F3.2 en las semillas 42, 7 y 123.
"""
import glob
import json
import os

import numpy as np
import pytest
from sklearn.metrics import f1_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, 'results', 'f55_sensibilidad')
SEGUNDOS = {'gray_scott': ('resnet18', 'finetune', 'px224-flip'),
            'perlin': ('efficientnet_b0', 'frozen', 'px224-noflip'),
            'lorenz': ('efficientnet_b0', 'finetune', 'px224-noflip')}
LR = {'finetune': 3e-5, 'frozen': 1e-3}


@pytest.mark.parametrize('tech', sorted(SEGUNDOS))
def test_corridas_de_sensibilidad(tech):
    arch, strat, var = SEGUNDOS[tech]
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, tech, 'runs', '*.json')))]
    assert {r['seed'] for r in runs} == {42, 7, 123, 2024, 99}
    ref = json.load(open(glob.glob(os.path.join(REPO, 'results', 'f35_final', tech, 'runs', '*seed42.json'))[0]))
    for r in runs:
        assert (r['arch'], r['strategy'], r['variant']) == (arch, strat, var)
        assert r['fase'] == 'sensibilidad' and r['evaluate_test'] and r['permute_labels'] is None
        assert r['selection']['role'] == 'second' and r['selection']['matches'] and r['protocol']['tag'] == 'protocolo-v1'
        assert (r['lr'], r['weight_decay'], r['batch_size']) == (LR[strat], 0.0, 32)
        assert r['val_true'] == ref['val_true'] and r['test_true'] == ref['test_true']
        for split, f1 in (('val', r['best_f1_val']), ('test', r['f1_test'])):
            pred = (np.array(r[f'{split}_probs']) > 0.5).astype(int)
            assert f1_score(r[f'{split}_true'], pred, average='macro') == pytest.approx(f1, abs=1e-12)
        if r['seed'] in (42, 7, 123):
            e = json.load(open(os.path.join(REPO, 'results', 'f32_primera_pasada', 'runs',
                                            f"{tech}__{arch}__{strat}__{var}__seed{r['seed']}.json")))
            assert e['val_probs'] == r['val_probs']


def test_predicciones_sensibilidad():
    assert len(glob.glob(os.path.join(REPO, 'results', 'predictions', 'sensibilidad', '*.csv'))) == 15
