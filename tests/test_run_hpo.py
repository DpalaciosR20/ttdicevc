"""
tests/test_run_hpo.py
═════════════════════
F3.4 — run_hpo.py sin GPU ni datos reales:

  - configuracion: espacio por estrategia y arquitectura, trial 0 = §3,
    presupuesto y semillas fijos fuera de --smoke;
  - regla final de la matriz §6 en sus dos ramas;
  - la configuracion debe ser el finalista sugerido (o llevar un motivo);
  - extremo a extremo con el TT_DATA_ROOT sintetico de test_run_grid.py y
    train_model sustituido: test nunca se construye, el trial 0 es la
    configuracion comun, y al volver a correr el estudio se reconstruye igual
    sin entrenar nada; si una corrida guardada no coincide, se detiene.

Ejecucion (desde la raiz del repo):
    python -m pytest tests/test_run_hpo.py -v
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'training'))
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

pytest.importorskip('optuna')

import config  # noqa: E402
import run_hpo  # noqa: E402
from test_run_grid import fake_data  # noqa: E402,F401  (fixture)

MATRIX = config.load_matrix()
SETTINGS = run_hpo.hpo_settings(MATRIX)
DEFAULTS = config.training_defaults(MATRIX)


def test_presupuesto_de_la_matriz():
    assert (SETTINGS['sampler_seed'], SETTINGS['n_trials'], SETTINGS['seeds']) == (42, 12, [42, 7, 123])


@pytest.mark.parametrize('arch, strategy, batch', [
    ('efficientnet_b0', 'finetune', [16, 32, 64]), ('vit_b16', 'finetune', [16, 32]),
    ('vit_b16', 'frozen', [16, 32, 64])])
def test_espacio_por_estrategia(arch, strategy, batch):
    s = run_hpo.search_space(SETTINGS['space'], arch, strategy)
    assert s['batch_size'] == batch and s['lr']['log'] and not s['weight_decay']['log']
    assert (s['lr']['low'], s['lr']['high']) == ((5e-6, 3e-4) if strategy == 'finetune' else (1e-4, 1e-2))
    assert run_hpo.trial0_params(DEFAULTS, strategy, s) == {
        'lr': 3e-5 if strategy == 'finetune' else 1e-3, 'weight_decay': 0.0, 'batch_size': 32}


def _t(n, mean, std, lr):
    return {'trial': n, 'f1_val_mean': mean, 'f1_val_std': std, 'params': {'lr': lr}}


def test_regla_final():
    adopta = run_hpo.final_rule([_t(0, 0.70, 0.010, 1e-3), _t(1, 0.72, 0.02, 2e-3), _t(2, 0.69, 0.0, 3e-3)])
    assert adopta['adopted_best'] and adopta['adopted_trial'] == 1 and adopta['params'] == {'lr': 2e-3}
    conserva = run_hpo.final_rule([_t(0, 0.70, 0.010, 1e-3), _t(1, 0.705, 0.0, 2e-3)])  # ganancia < 1 DE
    assert not conserva['adopted_best'] and conserva['adopted_trial'] == 0 and conserva['best_trial'] == 1
    gana0 = run_hpo.final_rule([_t(0, 0.73, 0.010, 1e-3), _t(1, 0.71, 0.0, 2e-3)])
    assert gana0['best_trial'] == 0 and gana0['adopted_trial'] == 0


def test_solo_el_finalista_sugerido():
    info = run_hpo.check_finalist('gray_scott', 'efficientnet_b0', 'finetune', None, MATRIX)
    assert info['is_suggested_finalist']
    with pytest.raises(SystemExit, match='override-selection'):
        run_hpo.check_finalist('gray_scott', 'resnet18', 'finetune', None, MATRIX)
    info = run_hpo.check_finalist('gray_scott', 'resnet18', 'finetune', 'decision de SV en F3.3', MATRIX)
    assert not info['is_suggested_finalist'] and info['override'] == 'decision de SV en F3.3'


# ─────────────────────────────────────────────────────────────
# Extremo a extremo
# ─────────────────────────────────────────────────────────────

CALLS = []


def _fake_train_model(arch_name, freeze_backbone, train_loader, val_loader, test_loader, class_weights,
                      device, lr=None, n_epochs=15, patience=4, seed=None, weight_decay=0.0):
    import torch
    from sklearn.metrics import f1_score
    assert test_loader is None
    CALLS.append((lr, weight_decay, train_loader.batch_size, seed))
    y = np.array(val_loader.dataset.labels)
    rng = np.random.default_rng([seed, int(lr * 1e9), train_loader.batch_size])
    probs = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.25, len(y)), 0, 1)
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(y, preds, average='macro')
    res = {'best_epoch': 1, 'epochs_run': 1, 'n_trainable_params': 10, 'n_total_params': 20, 'best_f1_val': f1,
           'val_preds': preds, 'val_probs': probs, 'val_true': y}
    return torch.nn.Linear(1, 1), {'train_loss': [0.7], 'f1_val': [f1]}, res


def _run(monkeypatch, argv):
    import train_model
    monkeypatch.setattr(train_model, 'train_model', _fake_train_model)
    monkeypatch.setattr(sys, 'argv', ['run_hpo.py', *argv])
    run_hpo.main()


def _trials(out):
    with open(out / 'trials.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_extremo_a_extremo_y_reanudacion(fake_data, tmp_path, monkeypatch):
    out = tmp_path / 'hpo'
    argv = ['--config', 'lorenz:vit_b16:frozen', '--out-dir', str(out), '--smoke', '--n-trials', '4',
            '--seeds', '42', '7', '--epochs', '1', '--num-workers', '0']
    CALLS.clear()
    _run(monkeypatch, argv)
    assert len(CALLS) == 4 * 2
    assert CALLS[0][:3] == (1e-3, 0.0, 32)                                  # trial 0 = §3
    assert all(1e-4 <= lr <= 1e-2 and 0 <= wd <= 0.1 and bs in (16, 32, 64) for lr, wd, bs, _ in CALLS)

    rows = _trials(out)
    assert [int(r['trial']) for r in rows] == [0, 1, 2, 3]
    runs = sorted((out / 'runs').glob('*.json'))
    assert len(runs) == 8
    rec = json.loads(runs[0].read_text(encoding='utf-8'))
    assert rec['fase'] == 'hpo' and 'test_probs' not in rec and rec['variant'] == 'px224-noflip'
    result = json.loads((out / 'hpo_result.json').read_text(encoding='utf-8'))
    assert result['decision']['trial0'] == 0 and '--lr' in result['run_grid_flags']
    assert result['selection']['is_suggested_finalist']
    assert (out / 'study.pkl').exists()

    # Reanudar: el estudio se reconstruye igual y no se entrena nada
    CALLS.clear()
    _run(monkeypatch, argv)
    assert CALLS == []
    assert _trials(out) == [{**r, 'origen': 'disco'} for r in rows]

    # Una corrida guardada que no corresponde al muestreo detiene el estudio
    p = out / 'runs' / 'trial02__seed7.json'
    rec = json.loads(p.read_text(encoding='utf-8'))
    rec['lr'] = 5e-3
    p.write_text(json.dumps(rec), encoding='utf-8')
    with pytest.raises(run_hpo.ReplayMismatch):
        _run(monkeypatch, argv)


def test_presupuesto_fijo_fuera_de_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_hpo.py', '--config', 'lorenz:vit_b16:frozen', '--out-dir',
                                      str(tmp_path / 'o'), '--n-trials', '3'])
    with pytest.raises(SystemExit, match='smoke'):
        run_hpo.main()
