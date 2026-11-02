"""
tests/test_matriz_experimentos.py
═════════════════════════════════
F1.3 — configs/matriz_experimentos.json debe coincidir con:

  1. docs/matriz_experimentos.md (§2–§8), que es donde se registran las
     decisiones;
  2. configs/preprocesamiento.json (F1.4), para el preprocesamiento por técnica;
  3. los conteos de parámetros de results/f13_matriz/reconocimiento_ce.md
     (bloque A), que se obtuvieron instanciando los modelos en CPU;
  4. el vocabulario controlado de F1.5 (src/predictions/schema.py).

Si alguien cambia uno sin el otro, esta prueba falla. No importa torch.

Ejecución (desde la raíz del repo):
    python -m pytest tests/test_matriz_experimentos.py -v
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.predictions import schema  # noqa: E402

DOC = REPO / 'docs' / 'matriz_experimentos.md'
CONFIG = REPO / 'configs' / 'matriz_experimentos.json'
PREPRO = REPO / 'configs' / 'preprocesamiento.json'
RECON = REPO / 'results' / 'f13_matriz' / 'reconocimiento_ce.md'
DATA_PY = REPO / 'src' / 'training' / 'data.py'

# El código de entrenamiento usa el nombre interno de torchvision (§2)
CODE_ARCH = {'vit_b_16': 'vit_b16'}


# ── Utilidades de lectura del documento ──────────────────────────────────────

def section(text, n):
    """Texto de la sección '## n.' hasta la siguiente '## '."""
    m = re.search(rf'^## {n}\. .*?$(.*?)(?=^## |\Z)', text, re.M | re.S)
    assert m, f'no se encontró la sección {n}'
    return m.group(1)


def table_rows(text):
    """Filas de las tablas Markdown (sin encabezado ni separador) como listas de celdas."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?', c) for c in cells):
            continue
        rows.append(cells)
    return rows[1:] if rows else rows  # descarta el encabezado de la primera tabla


def table_dict(text):
    """{primera celda: resto} de la primera tabla de la sección."""
    return {r[0]: r[1:] for r in table_rows(text)}


def ticks(cell):
    return re.findall(r'`([^`]+)`', cell)


def ints(cell):
    return [int(x) for x in re.findall(r'\b\d+\b', cell)]


def floats(cell):
    return [float(x) for x in re.findall(r'\d+(?:\.\d+)?(?:e-?\d+)?', cell)]


@pytest.fixture(scope='module')
def doc():
    return DOC.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


# ── Vocabulario (F1.5) ───────────────────────────────────────────────────────

def test_vocabulary_is_f15(cfg):
    f = cfg['factores']
    assert set(f['technique']) <= schema.VOCAB['technique']
    assert set(f['arch']) <= schema.VOCAB['arch']
    assert set(f['strategy']) <= schema.VOCAB['strategy']
    assert 'vit_b16' in f['arch'] and 'vit_b_16' not in f['arch']
    for block in (cfg['parametros'], cfg['configuracion_comun']['head']['replaces'],
                  cfg['interpretabilidad']['methods']):
        assert set(k for k in block if not k.startswith('_')) == set(f['arch'])


def test_fases_are_f15(cfg):
    for fase in ('primera_pasada', 'final', 'control_permutacion', 'sensibilidad'):
        assert fase in schema.FASES
        assert set(cfg['fases'][fase]['prediction_splits']) <= schema.FASES[fase]
    for fase in schema.FASES_TEST_REQUIRES_TAG:
        assert cfg['fases'][fase]['requires']['git_tag'] == schema.PROTOCOL_TAG
    assert cfg['seleccion']['decision_threshold'] == schema.DECISION_THRESHOLD


# ── §2 Factores y preprocesamiento (F1.4) ────────────────────────────────────

def test_factors_match_doc(doc, cfg):
    rows = table_dict(section(doc, 2))
    f = cfg['factores']
    assert ticks(rows['Técnica'][0]) == f['technique']
    assert ticks(rows['Arquitectura'][0]) == f['arch']
    assert ticks(rows['Estrategia'][0]) == f['strategy']
    pairs = re.findall(r'`(\w+)` = `([\w-]+)`', rows['Preprocesamiento'][0])
    assert dict(pairs) == {k: v for k, v in f['preprocesamiento'].items() if not k.startswith('_')}
    n = len(f['technique']) * len(f['arch']) * len(f['strategy'])
    assert f'{n} configuraciones' in section(doc, 2)


def test_preprocessing_matches_f14(cfg):
    prepro = json.loads(PREPRO.read_text(encoding='utf-8'))
    ours = {k: v for k, v in cfg['factores']['preprocesamiento'].items() if not k.startswith('_')}
    assert ours == {t: prepro[t]['variant'] for t in prepro if not t.startswith('_')}


# ── Conteos de parámetros (reconocimiento, §5) ───────────────────────────────

def _recon_params():
    text = RECON.read_text(encoding='utf-8')
    out = {}
    for arch, strat, tr, tot in re.findall(
            r'^\| `(\w+)` \| `(\w+)` \| ([\d ]+) \| ([\d ]+) \|', text, re.M):
        out[(CODE_ARCH.get(arch, arch), strat)] = (int(tr.replace(' ', '')), int(tot.replace(' ', '')))
    return out


def test_params_match_recon(cfg):
    recon = _recon_params()
    ours = {(a, s): (v['trainable'], v['total'])
            for a, by in cfg['parametros'].items() if not a.startswith('_') for s, v in by.items()}
    assert len(recon) == 6, recon
    assert ours == recon


def test_params_internally_consistent(cfg):
    head = cfg['configuracion_comun']['head']
    for arch in cfg['factores']['arch']:
        p = cfg['parametros'][arch]
        assert p['finetune']['trainable'] == p['finetune']['total'] == p['frozen']['total']
        # frozen: solo la cabeza lineal (in_features × 2 + 2 sesgos)
        assert p['frozen']['trainable'] == head['in_features'][arch] * head['out_features'] + head['out_features']


def test_complexity_order_in_doc(doc, cfg):
    """La línea «Orden resultante» de §5 coincide con ordenar por (entrenables, totales)."""
    line = re.search(r'Orden resultante: (.*)', section(doc, 5)).group(1)
    items = re.findall(r'`(\w+) (\w+)` \(([\d.,]+)( M)?\)', line)
    assert len(items) == 6, line
    combos = sorted(((a, s) for a in cfg['factores']['arch'] for s in cfg['factores']['strategy']),
                    key=lambda c: (cfg['parametros'][c[0]][c[1]]['trainable'],
                                   cfg['parametros'][c[0]][c[1]]['total']))
    assert [(a, s) for a, s, _, _ in items] == combos
    for a, s, num, mega in items:
        n = cfg['parametros'][a][s]['trainable']
        if mega:
            assert float(num) == round(n / 1e6, 2), (a, s, num)
        else:
            assert int(num.replace(',', '')) == n, (a, s, num)


# ── §3 Configuración común ───────────────────────────────────────────────────

def test_common_config_matches_doc(doc, cfg):
    rows = {k: v[0] for k, v in table_dict(section(doc, 3)).items()}
    c = cfg['configuracion_comun']

    for arch, w in c['weights'].items():
        enum, member = w.split('.')
        assert member == 'IMAGENET1K_V1' and member in rows['Pesos preentrenados']
        assert f'`{enum}`' in rows['Pesos preentrenados'], arch

    assert 'Grayscale(3)' in rows['Entrada'] and 'ImageNet' in rows['Entrada']
    assert c['input']['size_and_flip'] in rows['Entrada']

    assert f"nn.Linear(·, {c['head']['out_features']})" in rows['Capa de clasificación']
    for arch, attr in c['head']['replaces'].items():
        assert f"`{attr}` ({c['head']['in_features'][arch]})" in rows['Capa de clasificación'], arch

    assert '`eval()`' in rows['Modo `frozen`'] and '`train()`' in rows['Modo `frozen`']
    assert c['frozen_mode'] == 'backbone_eval_head_train'

    assert rows['Optimizador'].startswith(c['optimizer']['name'] + ' ')
    assert c['optimizer']['scheduler'] is None and 'sin *scheduler*' in rows['Optimizador']

    lr = dict(re.findall(r'([\d.e-]+) \(`(\w+)`\)', rows['Tasa de aprendizaje']))
    assert {s: float(v) for v, s in lr.items()} == c['lr']
    assert floats(rows['*Weight decay*']) == [c['weight_decay']]
    assert ints(rows['Tamaño de lote']) == [c['batch_size']]
    assert ints(rows['Épocas máximas']) == [c['max_epochs']]

    es = rows['*Early stopping*']
    assert 'F1-Macro de validación' in es and '`argmax`' in es and 'mejora estricta' in es
    assert f"paciencia {c['early_stopping']['patience']}" in es and 'se restaura la mejor época' in es
    assert c['early_stopping'] == {'metric': 'f1_macro', 'split': 'val', 'prediction': 'argmax_logits',
                                   'improvement': 'strict', 'patience': 4, 'restore_best': True}

    assert '`CrossEntropyLoss`' in rows['Pérdida'] and '`n / (2 · n_c)`' in rows['Pérdida']
    assert c['loss']['class_weights'] == 'n / (2 * n_c)' and '`train`' in rows['Pérdida']
    assert c['precision'] == 'fp32' and c['amp'] is False and 'fp32, sin AMP' in rows['Precisión']
    assert f"`num_workers = {c['num_workers']}`" in rows['Semillas']

    det, d = rows['Determinismo'], c['determinism']
    assert d['cudnn_deterministic'] is True and '`cudnn.deterministic = True`' in det
    assert d['cudnn_benchmark'] is False and '`cudnn.benchmark = False`' in det
    later = d['desde_repeticion_f32']
    assert later['use_deterministic_algorithms'] is True and '`torch.use_deterministic_algorithms(True)`' in det
    assert f"`CUBLAS_WORKSPACE_CONFIG={later['CUBLAS_WORKSPACE_CONFIG']}`" in det
    assert later['flash_sdp'] is False and later['mem_efficient_sdp'] is False
    assert '*flash*' in det and '*memory-efficient*' in det and 'deshabilitados' in det


def test_input_normalization_matches_code(cfg):
    """Media y DE de ImageNet iguales a las de data.py (leído como texto: sin torch)."""
    text = DATA_PY.read_text(encoding='utf-8')
    for key, name in (('mean', 'IMAGENET_MEAN'), ('std', 'IMAGENET_STD')):
        vals = re.search(rf'^{name} = \[(.*?)\]', text, re.M).group(1)
        assert [float(x) for x in vals.split(',')] == cfg['configuracion_comun']['input'][key]


# ── §4 Fases, semillas y total de entrenamientos ─────────────────────────────

def _phase_rows(doc):
    """{fase o tarea: fila} de la tabla de §4."""
    out = {}
    for r in table_rows(section(doc, 4)):
        key = ticks(r[0])[0] if ticks(r[0]) else r[1]
        out[key] = r
    return out


@pytest.mark.parametrize('fase, key', [('primera_pasada', 'primera_pasada'), ('hpo', 'F3.4'),
                                        ('final', 'final'), ('control_permutacion', 'control_permutacion'),
                                        ('sensibilidad', 'sensibilidad')])
def test_phase_seeds_and_splits(doc, cfg, fase, key):
    rows = _phase_rows(doc)
    row = rows[key]
    ph = cfg['fases'][fase]
    assert row[1] == ph['task']
    assert ints(row[3]) == ph['seeds']
    assert [t for t in ticks(row[4]) if t in schema.SPLIT_ORDER] == ph['prediction_splits']
    if 'requires' in ph:
        req = row[5]
        same = re.fullmatch(r'Ídem `(\w+)`', req)
        if same:  # «Ídem `final`»: mismo requisito que esa fase
            assert cfg['fases'][same.group(1)]['requires'] == ph['requires']
            req = rows[same.group(1)][5]
        assert f"`{ph['requires']['protocol_file']}`" in req
        assert f"`{ph['requires']['git_tag']}`" in req


def test_selection_and_attribution_rows(doc, cfg):
    rows = _phase_rows(doc)
    assert cfg['fases']['seleccion']['task'] == 'F3.3' and 'F3.3' in rows
    assert rows['F3.7'][3] == 'Las 5 de F3.5' and cfg['fases']['interpretabilidad']['seeds_from'] == 'final'


def test_total_trainings(doc, cfg):
    f, ph = cfg['factores'], cfg['fases']
    n_cfg = len(f['technique']) * len(f['arch']) * len(f['strategy'])
    n_fin = len(f['technique'])  # un finalista por técnica
    total = (n_cfg * len(ph['primera_pasada']['seeds'])
             + n_fin * cfg['hpo']['n_trials'] * len(ph['hpo']['seeds'])
             + n_fin * len(ph['final']['seeds'])
             + n_fin * len(ph['control_permutacion']['seeds'])
             + n_fin * len(ph['sensibilidad']['seeds']))  # un segundo mejor por técnica
    assert total == ph['total_entrenamientos']
    m = re.search(r'Total de entrenamientos: (\d+) \(F3\.2\) \+ (\d+) × (\d+) × (\d+) = (\d+) \(F3\.4\) '
                  r'\+ (\d+) \(F3\.5\) \+ (\d+) \(F3\.6\) \+ (\d+) \(F5\.5\) = (\d+)', section(doc, 4))
    assert m, 'no se encontró la línea del total en §4'
    g = [int(x) for x in m.groups()]
    assert g[0] == n_cfg * len(ph['primera_pasada']['seeds'])
    assert g[1:4] == [n_fin, cfg['hpo']['n_trials'], len(ph['hpo']['seeds'])] and g[4] == g[1] * g[2] * g[3]
    assert g[5] == n_fin * len(ph['final']['seeds']) and g[6] == n_fin * len(ph['control_permutacion']['seeds'])
    assert g[7] == n_fin * len(ph['sensibilidad']['seeds'])
    assert g[8] == total == sum(g[i] for i in (0, 4, 5, 6, 7))


def test_sensitivity_phase(cfg):
    """Enmienda de F1.2: segundo mejor modelo, hiperparámetros de la primera pasada, semillas de final."""
    s = cfg['fases']['sensibilidad']
    assert s['seeds'] == cfg['fases']['final']['seeds']
    assert s['requires'] == cfg['fases']['final']['requires']
    assert s['selection'] == 'select_second' and s['hyperparams'] == 'configuracion_comun'


# ── §5 Regla de selección ────────────────────────────────────────────────────

def test_selection_rule_matches_doc(doc, cfg):
    sec, s = section(doc, 5), cfg['seleccion']
    assert s['fase'] == 'primera_pasada' and '`primera_pasada`' in sec and '`load_predictions`' in sec
    assert s['seeds'] == cfg['fases']['primera_pasada']['seeds']
    assert ints(re.search(r'semillas ([\d, y]+?), a partir', sec).group(1)) == s['seeds']
    assert s['metric'] == 'f1_macro' and 'F1-Macro de validación' in sec and s['split'] == 'val'
    assert s['aggregate'] == 'mean' and s['dispersion'] == {'name': 'std', 'ddof': 1} and '(`ddof = 1`)' in sec
    assert s['tie_tolerance'] == 'leader_std' and '≥ media del líder − DE del líder' in sec
    assert s['multiple_leaders_std'] == 'max' and 'se usa la mayor de sus DE' in sec
    assert s['complexity_order'] == ['trainable', 'total']
    i_tr, i_tot = sec.index('menor número de parámetros entrenables'), sec.index('menor número de parámetros totales')
    assert i_tr < i_tot
    assert s['residual_tiebreak'] == 'max_mean' and 'Si persiste el empate, la de mayor media' in sec
    assert s['unresolved_tie'] == 'error' and 'se detiene con un error' in sec
    assert '`src/selection/select_finalist.py`' in sec


# ── §6 HPO ───────────────────────────────────────────────────────────────────

def _interval(cell, name):
    m = re.search(rf'`{name}` ∈ \[([\d.e-]+), ([\d.e-]+)\], (logarítmico|uniforme)', cell)
    assert m, (name, cell)
    return {'low': float(m.group(1)), 'high': float(m.group(2)), 'log': m.group(3) == 'logarítmico'}


def _batches(cell):
    m = re.search(r'lote ∈ \{([\d, ]+)\}', cell)
    return [int(x) for x in m.group(1).split(',')]


def test_hpo_matches_doc(doc, cfg):
    rows = {k: v[0] for k, v in table_dict(section(doc, 6)).items()}
    h = cfg['hpo']
    assert h['algorithm'] == 'optuna_tpe' and rows['Algoritmo'].startswith('Optuna, TPE')
    assert f"`seed = {h['sampler_seed']}`" in rows['Algoritmo'] and 'parámetros por defecto' in rows['Algoritmo']
    assert h['pruner'] is None and 'sin *pruner*' in rows['Algoritmo']
    assert ints(rows['Presupuesto']) == [h['n_trials']]
    assert h['objective']['seeds'] == cfg['fases']['hpo']['seeds']
    assert ints(rows['Objetivo']) == h['objective']['seeds']
    assert '`enqueue_trial`' in rows['*Trial* 0'] and '`weight_decay = 0`' in rows['*Trial* 0']
    assert f"lote {cfg['configuracion_comun']['batch_size']}" in rows['*Trial* 0']

    for strat in ('finetune', 'frozen'):
        cell, sp = rows[f'Espacio (`{strat}`)'], h['space'][strat]
        assert _interval(cell, 'lr') == sp['lr']
        assert _interval(cell, 'weight_decay') == sp['weight_decay']
        assert _batches(cell) == sp['batch_size']
        # La configuración común (trial 0) está dentro del espacio
        assert sp['lr']['low'] <= cfg['configuracion_comun']['lr'][strat] <= sp['lr']['high']
        assert cfg['configuracion_comun']['batch_size'] in sp['batch_size']
    ov = re.search(r'\(`(\w+)`: \{([\d, ]+)\}\)', rows['Espacio (`finetune`)'])
    assert h['space']['finetune']['batch_size_override'] == {ov.group(1): [int(x) for x in ov.group(2).split(',')]}
    assert 'batch_size_override' not in h['space']['frozen']

    fixed = rows['Fijos']
    words = {'arch': 'Arquitectura', 'strategy': 'estrategia', 'preprocesamiento': 'preprocesamiento',
             'optimizer': 'optimizador', 'max_epochs': 'épocas máximas', 'patience': 'paciencia',
             'class_weights': 'pesos de clase'}
    assert set(h['fixed']) == set(words) and all(w in fixed for w in words.values())
    assert 'más de una DE del *trial* 0' in rows['Regla final']


# ── §7 Permutación y §8 Interpretabilidad ────────────────────────────────────

def test_permutation_matches_doc(doc, cfg):
    sec, p = section(doc, 7), cfg['fases']['control_permutacion']['permutation']
    assert f"`numpy.random.default_rng({p['seed']})`" in sec and p['rng'] == 'numpy.random.default_rng'
    assert p['permuted_split'] == 'train' and '`y_train`' in sec
    assert p['save_indices'] is True and 'Se guardan los índices de la permutación' in sec


def test_interpretability_matches_doc(doc, cfg):
    sec, it = section(doc, 8), cfg['interpretabilidad']
    rows = {ticks(k)[0]: v for k, v in table_dict(sec).items()}
    assert set(rows) == set(it['methods'])
    names = {'grad_cam': 'Grad-CAM', 'attention_rollout': '*Attention rollout*'}
    layers = {'layer4': 'Salida de `layer4`', 'features[-1]': 'Salida del último bloque de `features`'}
    for arch, m in it['methods'].items():
        assert rows[arch][0].startswith(names[m['method']]), arch
        if 'layer' in m:
            assert rows[arch][1] == layers[m['layer']], arch
    vit, cell = it['methods']['vit_b16'], rows['vit_b16'][1]
    assert vit['head_fusion'] == 'mean' and 'cabezas por media' in cell
    assert vit['add_identity'] and vit['renormalize'] and 'identidad añadida y renormalizada' in cell
    assert vit['discard_ratio'] == 0.0 and 'sin descarte' in cell
    assert vit['exclude_cls_token'] and 'se excluye el *token* de clase' in cell

    assert it['target'] == {'logit_class': 1} and 'el *logit* de la clase Depresivo (1)' in sec
    assert it['splits'] == ['val', 'test'] and '114 de `val` y las 114 de `test`' in sec
    assert f"{it['resolution']} × {it['resolution']}" in sec
    assert it['normalization'] == 'per_sample_minmax' and 'normalizados por muestra a [0, 1]' in sec
    assert it['save_unnormalized'] and 'se guarda además el mapa sin normalizar' in sec
    assert f"`{it['dtype']}`" in sec and it['format'] == 'npz_compressed' and '`.npz` comprimido' in sec
    assert f"`{it['index']}`" in sec


# ── §10 Contenido mínimo ─────────────────────────────────────────────────────

def test_json_has_section10_blocks(cfg):
    for key in ('factores', 'configuracion_comun', 'fases', 'seleccion', 'hpo', 'interpretabilidad', 'parametros'):
        assert key in cfg, key
