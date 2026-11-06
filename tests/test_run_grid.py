"""
tests/test_run_grid.py
══════════════════════
F3.2 (repeticion) — run_grid.py con las configuraciones de F1.3/F1.4 y la
salida de F1.5, sin GPU ni datos reales:

  - configuracion: preprocesamiento por tecnica desde el JSON, linea de
    comandos que lo contradice, hiperparametros fijos en primera_pasada, fase
    segun el modo;
  - extremo a extremo con un TT_DATA_ROOT sintetico (PNG de 8x8 con los
    nombres de meta/sample_index.csv) y train_model sustituido por uno falso:
    los CSV pasan el validador de F1.5, p_dep sin redondear, F1 recalculado =
    F1 del JSON, variante del manifiesto = JSON de F1.4, y no se sobrescribe un
    archivo de otra variante.

Ejecucion (desde la raiz del repo):
    python -m pytest tests/test_run_grid.py -v
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'training'))
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import run_grid  # noqa: E402
from src import paths  # noqa: E402


def test_preprocesamiento_por_tecnica():
    prep = config.load_preprocessing(['perlin', 'gray_scott', 'lorenz'])
    assert {t: prep[t]['variant'] for t in prep} == {
        'perlin': 'px224-noflip', 'gray_scott': 'px224-flip', 'lorenz': 'px224-noflip'}
    assert all(prep[t]['size'] == 224 and prep[t]['antialias'] is True for t in prep)


def test_tecnica_ausente_falla(tmp_path):
    p = tmp_path / 'prep.json'
    data = json.loads(Path(config.PREPROCESSING_PATH).read_text(encoding='utf-8'))
    del data['lorenz']
    p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(config.ConfigError, match='lorenz'):
        config.load_preprocessing(['lorenz'], path=p)


def test_cli_no_contradice_json():
    prep = config.load_preprocessing(['gray_scott', 'lorenz'])
    run_grid.check_cli_preprocessing(prep, 224, False)
    with pytest.raises(SystemExit, match='img-size'):
        run_grid.check_cli_preprocessing(prep, 256, False)
    with pytest.raises(SystemExit, match='no-hflip'):
        run_grid.check_cli_preprocessing(prep, None, True)          # gray_scott permite volteo
    run_grid.check_cli_preprocessing({'lorenz': prep['lorenz']}, None, True)


@pytest.mark.parametrize('evaluate, perm, sens, fase', [
    (False, None, False, 'primera_pasada'), (True, None, False, 'final'),
    (True, 99, False, 'control_permutacion'), (True, None, True, 'sensibilidad')])
def test_fase_segun_modo(evaluate, perm, sens, fase):
    assert run_grid.phase_for(evaluate, perm, sens) == fase


def test_permutacion_y_sensibilidad_exigen_test():
    with pytest.raises(SystemExit):
        run_grid.phase_for(False, 99, False)
    with pytest.raises(SystemExit):
        run_grid.phase_for(False, None, True)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(dict(lr=None, weight_decay=None, batch_size=None, epochs=None, patience=None,
                                  num_workers=None, smoke=False), **kw)


def test_hiperparametros_de_la_matriz():
    d = config.training_defaults(config.load_matrix())
    assert d['lr'] == {'finetune': 3e-5, 'frozen': 1e-3} and d['weight_decay'] == 0.0
    assert (d['batch_size'], d['epochs'], d['patience']) == (32, 15, 4)
    assert run_grid.resolve_hparams(d, _Args(), 'primera_pasada') == d
    with pytest.raises(SystemExit, match='lr'):
        run_grid.resolve_hparams(d, _Args(lr=1e-4), 'primera_pasada')
    with pytest.raises(SystemExit, match='epochs'):
        run_grid.resolve_hparams(d, _Args(epochs=2), 'primera_pasada')
    assert run_grid.resolve_hparams(d, _Args(epochs=2, smoke=True), 'primera_pasada')['epochs'] == 2
    assert run_grid.resolve_hparams(d, _Args(lr=1e-4), 'final')['lr'] == {'finetune': 1e-4, 'frozen': 1e-4}


def test_permutacion_guarda_indices():
    from data import permute_labels
    labels = [0] * 6 + [1] * 4
    perm, idx = permute_labels(labels, 99)
    assert perm == [labels[i] for i in idx] and sorted(idx) == list(range(10)) and sum(perm) == 4


# ─────────────────────────────────────────────────────────────
# Extremo a extremo con datos sinteticos
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    """TT_DATA_ROOT con PNG de 8x8 nombrados como en el indice maestro y un
    data_manifest.csv construido sobre ellos."""
    from PIL import Image
    from src import build_manifest
    root = tmp_path / 'insumos'
    with open(paths.SAMPLE_INDEX, newline='', encoding='utf-8') as f:
        index = list(csv.DictReader(f))
    rng = np.random.default_rng(0)
    for tech in paths.TECHNIQUES:
        d = root / paths.REL['IMAGES'][tech]
        d.mkdir(parents=True)
        for r in index:
            Image.fromarray(rng.integers(0, 255, (8, 8), dtype=np.uint8), 'L').save(d / r[f'img_{tech}'])
    monkeypatch.setenv('TT_DATA_ROOT', str(root))
    rows = []
    for tech in paths.TECHNIQUES:
        rel = paths.REL['IMAGES'][tech]
        children = [build_manifest.file_row(f'IMAGES[{tech}]', f'{rel}/{p.name}', root)
                    for p in paths.list_files(root / rel)[0]]
        rows += children + [{'logical_name': f'IMAGES[{tech}]', 'rel_path': rel, 'kind': 'dir',
                             'size_bytes': 0, 'sha256': build_manifest.dir_digest(children),
                             'n_files': len(children)}]
    monkeypatch.setattr(build_manifest, 'read_manifest', lambda path=None: [dict(r) for r in rows])
    return root


def _fake_train_model(arch_name, freeze_backbone, train_loader, val_loader, test_loader, class_weights,
                      device, lr=None, n_epochs=15, patience=4, seed=None, weight_decay=0.0):
    import torch
    from sklearn.metrics import f1_score
    y = np.array(val_loader.dataset.labels)
    rng = np.random.default_rng(seed)
    probs = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, len(y)), 0, 1) + 1e-9 * rng.random(len(y))
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(y, preds, average='macro')
    res = {'arch': arch_name, 'freeze_backbone': freeze_backbone, 'lr': lr, 'weight_decay': weight_decay,
           'seed': seed, 'best_epoch': 1, 'epochs_run': 1, 'n_trainable_params': 10, 'n_total_params': 20,
           'best_f1_val': f1, 'f1_test': None, 'val_preds': preds, 'val_probs': probs, 'val_true': y,
           'test_preds': None, 'test_probs': None, 'test_true': None}
    return torch.nn.Linear(1, 1), {'train_loss': [0.7], 'f1_val': [f1]}, res


def _run(monkeypatch, argv):
    import train_model
    monkeypatch.setattr(train_model, 'train_model', _fake_train_model)
    monkeypatch.setattr(sys, 'argv', ['run_grid.py', *argv])
    run_grid.main()


def test_extremo_a_extremo(fake_data, tmp_path, monkeypatch):
    from sklearn.metrics import f1_score
    from src.predictions.validate import load_predictions, validate_predictions
    out, pred_root = tmp_path / 'out', tmp_path / 'pred'
    argv = ['--out-dir', str(out), '--predictions-root', str(pred_root), '--techniques', 'gray_scott', 'lorenz',
            '--archs', 'resnet18', 'vit_b16', '--strategies', 'frozen', '--seeds', '42', '7']
    _run(monkeypatch, argv)

    files = sorted((pred_root / 'primera_pasada').glob('*.csv'))
    assert len(files) == 2 * 2 * 2
    assert {f.name.split('__')[1] for f in files} == {'resnet18', 'vit_b16'}
    for f in files:
        validate_predictions(f)
    preds = load_predictions(str(pred_root / 'primera_pasada' / '*.csv'))
    assert len(preds) == 8 * 114 and set(preds['split']) == {'val'}

    for f in files:
        manifest = json.loads(f.with_name(f.name[:-4] + '.manifest.json').read_text(encoding='utf-8'))
        tech = f.name.split('__')[0]
        assert manifest['fase'] == 'primera_pasada' and manifest['task_id'] == 'F3.2'
        assert manifest['config']['variant'] == config.load_preprocessing([tech])[tech]['variant']
        assert manifest['config']['optimizer'] == 'AdamW' and manifest['config']['weight_decay'] == 0.0
        assert manifest['inputs'][f'IMAGES[{tech}]']['n_files'] == 1136
        run = json.loads((out / 'runs' / manifest['config']['run_json']).read_text(encoding='utf-8'))
        df = preds[(preds['technique'] == tech) & (preds['arch'] == run['arch']) & (preds['seed'] == run['seed'])]
        assert df['p_dep'].tolist() == run['val_probs']                       # sin redondear
        f1 = f1_score(df['y_true'], (df['p_dep'] > 0.5).astype(int), average='macro')
        assert f1 == run['best_f1_val']

    with open(out / 'summary.csv', encoding='utf-8') as fh:
        assert len(list(csv.DictReader(fh))) == 8

    # Otra variante con la misma clave: no se sobrescribe
    f = files[0]
    mpath = f.with_name(f.name[:-4] + '.manifest.json')
    m = json.loads(mpath.read_text(encoding='utf-8'))
    m['config']['variant'] = 'px256-noflip'
    mpath.write_text(json.dumps(m), encoding='utf-8')
    with pytest.raises(SystemExit, match='otra configuracion'):
        _run(monkeypatch, argv)


def test_carpeta_alterada_aborta(fake_data, tmp_path, monkeypatch):
    extra = fake_data / paths.REL['IMAGES']['perlin'] / 'perlin_train_0838_label0 (1).png'
    extra.write_bytes(b'x')
    with pytest.raises(SystemExit, match='data_manifest'):
        _run(monkeypatch, ['--out-dir', str(tmp_path / 'o'), '--predictions-root', str(tmp_path / 'p'),
                           '--techniques', 'perlin', '--archs', 'resnet18', '--strategies', 'frozen',
                           '--seeds', '42'])


def test_test_bloqueado_sin_etiqueta(monkeypatch):
    monkeypatch.setattr(run_grid, 'git', lambda *a: None if a[0] == 'rev-parse' else '')
    with pytest.raises(SystemExit, match='protocolo-v1'):
        run_grid.check_protocol_frozen(run_grid.PROTOCOL_DEFAULT, 'protocolo-v1')


# ─────────────────────────────────────────────────────────────
# Selección en las fases con test (F3.3 H10; matriz §11.10)
# ─────────────────────────────────────────────────────────────

FINALISTAS = [('gray_scott', 'efficientnet_b0', 'finetune'), ('perlin', 'vit_b16', 'frozen'),
              ('lorenz', 'vit_b16', 'frozen')]
SEGUNDOS = [('gray_scott', 'resnet18', 'finetune'), ('perlin', 'efficientnet_b0', 'frozen'),
            ('lorenz', 'efficientnet_b0', 'finetune')]


@pytest.mark.parametrize('fase', ['final', 'control_permutacion'])
def test_fases_finales_exigen_finalistas(fase):
    m = config.load_matrix()
    info = run_grid.check_selection(fase, FINALISTAS, None, m)
    assert info['matches'] and info['role'] == 'finalist'
    assert info['source'] == 'results/predictions/primera_pasada'
    with pytest.raises(SystemExit, match='override-selection'):
        run_grid.check_selection(fase, SEGUNDOS[:1], None, m)
    info = run_grid.check_selection(fase, SEGUNDOS[:1], 'decision de SV', m)
    assert not info['matches'] and info['override'] == 'decision de SV'


def test_sensibilidad_exige_segundos():
    m = config.load_matrix()
    info = run_grid.check_selection('sensibilidad', SEGUNDOS, None, m)
    assert info['matches'] and info['role'] == 'second'
    assert info['expected'] == {t: [a, s] for t, a, s in SEGUNDOS}
    with pytest.raises(SystemExit, match='override-selection'):
        run_grid.check_selection('sensibilidad', FINALISTAS, None, m)


def test_seleccion_no_aplica_en_primera_pasada_ni_smoke():
    m = config.load_matrix()
    assert run_grid.check_selection('primera_pasada', SEGUNDOS, None, m) is None
    assert run_grid.check_selection('final', SEGUNDOS, None, m, smoke=True) is None
