"""F4.5 — referencia de longitud: sin métricas de test y fiel a configs/protocolo.json.

Comprueba:
  · que results/f45_longitud/resultado.json, el log y el manifiesto no registran
    ninguna métrica de test (ninguna clave con «test» tiene valor) y que el
    script no evalúa el split de test;
  · que el CSV archivado pasa el validador, tiene val y test, y que su F1-Val
    reproduce el registrado;
  · que la configuración usada es la de configs/protocolo.json;
  · las características (ln y {0, 1}) sobre datos sintéticos.

Antes de ejecutar el script (commit previo a la corrida) las pruebas que leen
los resultados se omiten.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / 'results' / 'f45_longitud'
SCRIPT = OUT / 'f45_longitud.py'
CSV = REPO / 'results' / 'predictions' / 'referencia' / 'text_length__logreg__none__seed42.csv'
RESULT = OUT / 'resultado.json'
LOG = OUT / 'f45_longitud.log'
CONFIG = REPO / 'configs' / 'protocolo.json'


def _require(*paths):
    missing = [p for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f'aún no se ejecutó f45_longitud.py: faltan {[p.name for p in missing]}')


def test_keys(obj, path=()):
    """Rutas de claves que mencionan test y tienen un valor no nulo."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'test' in str(k).lower() and v not in (None, [], {}):
                hits.append(path + (k,))
            hits += test_keys(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += test_keys(v, path + (i,))
    return hits


test_keys.__test__ = False  # utilidad, no prueba


def load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location('f45_longitud', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_utilidad_detecta_claves_de_test():
    assert test_keys({'a': {'f1_test': 0.7}}) == [('a', 'f1_test')]
    assert test_keys({'f1_test': None, 'b': [{'test_probs': []}]}) == []


def test_script_no_evalua_test():
    src = SCRIPT.read_text(encoding='utf-8')
    # el único uso de test es archivar p_dep; nunca se pasa a f1_val_of ni a una métrica
    assert not re.search(r"f1_val_of\(\s*y\[['\"]test['\"]\]", src)
    assert not re.search(r"(f1_score|confusion|accuracy)\w*\([^)]*test", src)
    assert "y['test']" not in src.replace("y[s]", "")


def test_resultado_sin_metricas_de_test():
    _require(RESULT)
    res = json.loads(RESULT.read_text(encoding='utf-8'))
    assert test_keys(res) == []
    assert res['splits_evaluados'] == ['val']
    assert set(res['seleccion']) == {'criterio', 'orden_en_rejilla', 'combinacion', 'f1_val', 'tabla'}


def test_log_y_manifiesto_sin_metricas_de_test():
    _require(LOG, CSV)
    log = LOG.read_text(encoding='utf-8')
    assert not re.search(r'F1[-_ ]?Test|f1_test|test\s*=\s*\d', log, re.I)
    for line in log.splitlines():
        if 'F1' in line:
            assert 'Val' in line, line
    man = json.loads(CSV.with_name(CSV.stem + '.manifest.json').read_text(encoding='utf-8'))
    assert test_keys(man['model_selection']) == [] and test_keys(man['config']) == []
    rej = pd.read_csv(OUT / 'rejilla_longitud.csv')
    assert not [c for c in rej.columns if 'test' in c.lower()]


def test_csv_valido_y_f1_val_reproducible():
    _require(CSV, RESULT)
    from src.predictions.validate import validate_predictions
    df = validate_predictions(CSV)
    assert (df['split'] == 'val').sum() == 114 and (df['split'] == 'test').sum() == 114
    assert set(df['technique']) == {'text_length'} and set(df['arch']) == {'logreg'}
    assert set(df['strategy']) == {'none'} and set(df['seed']) == {42}
    man = json.loads(CSV.with_name(CSV.stem + '.manifest.json').read_text(encoding='utf-8'))
    assert man['task_id'] == 'F4.5' and man['fase'] == 'referencia' and not man['git_commit'].endswith('-dirty')
    val = df[df['split'] == 'val']
    f1 = f1_score(val['y_true'], (val['p_dep'] > 0.5).astype(int), average='macro')
    res = json.loads(RESULT.read_text(encoding='utf-8'))
    assert f1 == pytest.approx(res['f1_val'], abs=1e-12)


def test_configuracion_desde_protocolo():
    _require(RESULT)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    ref, clf = cfg['referencias']['longitud'], cfg['enmienda_4']['longitud']['clasificador']
    res = json.loads(RESULT.read_text(encoding='utf-8'))
    p = res['estimator_params']
    grid = cfg['techo_5d']['familias']['logreg']
    assert p['max_iter'] == clf['max_iter'] and p['random_state'] == ref['seed']
    assert (p['penalty'], p['solver']) == (grid['penalty'], grid['solver'])
    assert p['class_weight'] in grid['class_weight']
    Cs = np.logspace(grid['C']['log10_low'], grid['C']['log10_high'], grid['C']['n'])
    assert np.isclose(Cs, p['C'], rtol=0, atol=1e-15).any()
    assert res['n_combinaciones'] == grid['C']['n'] * len(grid['class_weight'])
    rej = pd.read_csv(OUT / 'rejilla_longitud.csv')
    best = rej['f1_val'].max()
    first = rej[rej['f1_val'] >= best - cfg['techo_5d']['selection']['tie_tolerance']]['orden'].min()
    assert res['seleccion']['orden_en_rejilla'] == first


def test_caracteristicas():
    mod = load_script()
    clf = json.loads(CONFIG.read_text(encoding='utf-8'))['enmienda_4']['longitud']['clasificador']
    df = pd.DataFrame({'n_tokens_beto': [1, 128, 400], 'truncated_128': [False, 'False', 'True']})
    X = mod.features(df, clf)
    assert np.allclose(X[:, 0], np.log([1, 128, 400])) and X[:, 1].tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(mod.Stop):
        mod.features(pd.DataFrame({'n_tokens_beto': [0], 'truncated_128': [True]}), clf)
