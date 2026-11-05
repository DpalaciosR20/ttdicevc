"""
tests/test_protocolo.py
═══════════════════════
F1.2 — configs/protocolo.json debe coincidir con:

  1. docs/protocolo_preregistrado.md (§2, §4–§8, §10, §13 y el encabezado);
  2. results/canonical_numbers.json, para toda cifra que el documento cita;
  3. configs/matriz_experimentos.json (semillas y fases) y el esquema de
     predicciones de F1.5 (src/predictions/schema.py).

Si alguien cambia uno sin el otro, esta prueba falla. No importa torch.

Ejecución (desde la raíz del repo):
    python -m pytest tests/test_protocolo.py -v
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.canonical.build import f1_macro_from_confusion, text_sha256  # noqa: E402
from src.predictions import schema  # noqa: E402

DOC = REPO / 'docs' / 'protocolo_preregistrado.md'
CONFIG = REPO / 'configs' / 'protocolo.json'
MATRIX = REPO / 'configs' / 'matriz_experimentos.json'
CANON = REPO / 'results' / 'canonical_numbers.json'
INDEX = REPO / 'meta' / 'sample_index.csv'


# ── Lectura ──────────────────────────────────────────────────────────────────

def norm(text):
    """Signo menos tipográfico → ASCII; separador de miles fuera."""
    return text.replace('−', '-').replace('10,000', '10000')


def section(text, n):
    """Texto de '## n.' (o '### n') hasta el siguiente encabezado del mismo nivel o superior."""
    level = '###' if '.' in str(n) else '##'
    pat = rf'^{level} {re.escape(str(n))}\.? .*?$(.*?)(?=^#{{2,{len(level)}}} |\Z)'
    m = re.search(pat, text, re.M | re.S)
    assert m, f'no se encontró la sección {n}'
    return m.group(1)


def table_rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?', c) for c in cells):
            continue
        rows.append(cells)
    return rows[1:] if rows else rows


def table_dict(text):
    return {r[0]: r[1:] for r in table_rows(text)}


def nums(cell):
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:e-?\d+)?', norm(cell))]


def r4(x):
    return f'{x:.4f}'


@pytest.fixture(scope='module')
def doc():
    return DOC.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def matrix():
    return json.loads(MATRIX.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def canon():
    return json.loads(CANON.read_text(encoding='utf-8'))['numbers']


def tval(canon, nid):
    return canon[nid]['value']['test']


# ── Encabezado: congelamiento y huellas ─────────────────────────────────────

def header(doc):
    return doc.split('\n## ', 1)[0]


# Huella LF vigente de configs/protocolo.json. La Enmienda 2 (2026-11-04) añadió
# pisos_triviales, piso_brillo, descriptores y referencias.longitud.{technique,...};
# el encabezado conserva la huella de protocolo-v1 (el documento solo recibe la
# enmienda al final). Todo cambio posterior del config exige una enmienda nueva
# y actualizar esta constante. Única excepción prevista por la propia Enmienda 2
# (punto 1): F2.4 llenó descriptores.{implementation,software,params,tabla11}
# antes de calcular (huella de la Enmienda 2 sin ese bloque: 8538487a…).
CONFIG_SHA256_ENMIENDA_2 = '708ead595fe59bd0c5a58f0a11e156b3e3cb48f350cfa82d39152d3bd3b70e4a'


def header_sha(doc, path):
    rel = path.relative_to(REPO).as_posix()
    m = re.search(rf'`{re.escape(rel)}`[^\n]*?SHA-256 `([0-9a-f]{{64}})`', header(doc))
    assert m, f'el encabezado no registra el SHA-256 de {rel}'
    return m.group(1)


def sha_at_tag(path):
    """SHA-256 LF del archivo en la etiqueta del protocolo; skip si no hay git o etiqueta."""
    rel = path.relative_to(REPO).as_posix()
    try:
        blob = subprocess.run(['git', '-C', str(REPO), 'show', f'{schema.PROTOCOL_TAG}:{rel}'],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f'git o la etiqueta {schema.PROTOCOL_TAG} no están disponibles')
    return hashlib.sha256(blob.replace(b'\r\n', b'\n')).hexdigest()


def test_header_hashes_match_configs(doc):
    assert header_sha(doc, MATRIX) == text_sha256(MATRIX), 'SHA-256 de la matriz desactualizado en el encabezado'
    assert header_sha(doc, CONFIG) == sha_at_tag(CONFIG), \
        f'el encabezado debe registrar la huella de configs/protocolo.json en {schema.PROTOCOL_TAG}'
    assert text_sha256(CONFIG) == CONFIG_SHA256_ENMIENDA_2, \
        'configs/protocolo.json cambió después de la Enmienda 2: registra una enmienda y actualiza la constante'


def test_header_version_tag_and_amendment(doc, cfg):
    head = header(doc)
    assert 'v1 (congelado)' in head and cfg['_version'] == 'v1'
    assert f"`{cfg['_git_tag']}`" in head and cfg['_git_tag'] == schema.PROTOCOL_TAG
    assert 'Enmienda previa de la matriz: fase `sensibilidad` (F1.2)' in head
    assert '`configs/protocolo.json`' in head


# ── §2 Datos ─────────────────────────────────────────────────────────────────

def test_data_table_matches_canonical_and_index(doc, cfg, canon):
    rows = table_dict(section(doc, 2))
    idx = pd.read_csv(INDEX, usecols=['split', 'y_true'])
    c = cfg['datos']['canonical']
    for label, split in (('Entrenamiento', 'train'), ('Validación', 'val'), ('Prueba', 'test')):
        n, sup = rows[label][0], rows[label][1]
        assert int(n) == canon[c['n'][split]]['value'] == (idx['split'] == split).sum()
        n0, n1 = (int(x) for x in re.findall(r'\d+', sup))
        assert n0 == canon[f"{c['supports'][split]}.label0"]['value']
        assert n1 == canon[f"{c['supports'][split]}.label1"]['value']
        assert (n0, n1) == tuple(int(((idx['split'] == split) & (idx['y_true'] == k)).sum()) for k in (0, 1))


# ── §3 Información conocida ──────────────────────────────────────────────────

def test_known_information_matches_canonical(doc, canon):
    rows = table_dict(section(doc, 3))
    beto = canon['ref.beto_exp41.f1_macro']['value']
    cell = next(k for k in rows if k.startswith('F1-Macro de BETO'))
    assert f'validación {r4(beto["val"])}' in cell and f'prueba {r4(beto["test"])}' in cell
    cell = next(k for k in rows if k.startswith('Δ exploratorio de WITCOM'))
    assert nums(cell.split(':')[-1])[0] == round(tval(canon, 'witcom.delta_beto_gs_resnet18.f1_macro'), 4)
    cell = next(k for k in rows if k.startswith('F1-Val del clasificador sobre 768D'))
    assert cell.endswith(r4(canon['ref.emb768_logreg.f1_macro']['value']['val']))


# ── §4 Modelos ───────────────────────────────────────────────────────────────

def test_models_seeds_and_rule(doc, cfg, matrix):
    sec = section(doc, '4.1')
    seeds = [int(x) for x in re.findall(r'\d+', re.search(r'semillas ([\d, y]+) \(F3\.5\)', sec).group(1))]
    m = cfg['modelos']
    assert seeds == m['seeds'] == matrix['fases'][m['fase']]['seeds']
    assert m['decision_threshold'] == schema.DECISION_THRESHOLD and '`p_dep > 0.5`' in sec
    assert m['fase'] in schema.FASES


def test_references(doc, cfg, canon):
    rows = table_dict(section(doc, '4.2'))
    ref = cfg['referencias']
    assert f"C = 10^{ref['emb768']['log10_C']}".replace('-', '−') in rows['**768D**'][0]
    assert ref['beto']['seed'] == 42 and 'semilla 42' in rows['**BETO**'][0]
    for key in ('beto', 'emb768'):
        assert ref[key]['canonical'] in canon
        for col in ('technique', 'arch', 'strategy'):
            assert ref[key][col] in schema.VOCAB[col]
    assert '`log(n_tokens_beto)`' in rows['**Longitud**'][0] and '`truncated_128`' in rows['**Longitud**'][0]
    assert ref['longitud']['features'] == ['log(n_tokens_beto)', 'truncated_128']
    assert ref['longitud']['scaler']['fit_split'] == 'train'


def _grid(spec):
    return list(np.logspace(spec['log10_low'], spec['log10_high'], spec['n']))


def test_ceiling_5d_matches_doc(doc, cfg):
    sec = section(doc, '4.3')
    t = cfg['techo_5d']
    assert f"`random_state = {t['random_state']}`" in sec
    assert f"tolerancia {t['selection']['tie_tolerance']:.0e}".replace('e-12', 'e-12') in sec
    assert 'familia anterior en esta tabla' in sec and 'orden de la rejilla' in sec
    rows = table_rows(sec)
    names = {'LogReg (L2, `lbfgs`)': 'logreg', 'SVM-RBF': 'svm_rbf', 'Random Forest': 'rf',
             '*Gradient boosting* (`HistGradientBoostingClassifier`)': 'gbm', 'MLP': 'mlp'}
    assert [names[r[0]] for r in rows] == t['family_order']
    cells = {names[r[0]]: norm(r[1]) for r in rows}
    f = t['familias']

    lr = f['logreg']
    assert (lr['penalty'], lr['solver']) == ('l2', 'lbfgs')
    assert f"C ∈ 10^[{lr['C']['log10_low']}, {lr['C']['log10_high']}], {lr['C']['n']} valores" in cells['logreg']
    for fam in ('logreg', 'svm_rbf', 'rf'):
        assert f[fam]['class_weight'] == [None, 'balanced'] and '`class_weight` ∈ {None, `balanced`}' in cells[fam]
    assert 'class_weight' not in f['gbm'] and 'class_weight' not in f['mlp']

    sv = f['svm_rbf']
    assert f"C ∈ 10^[{sv['C']['log10_low']}, {sv['C']['log10_high']}], {sv['C']['n']} valores" in cells['svm_rbf']
    g = sv['gamma']
    assert f"γ ∈ {{`scale`}} ∪ 10^[{g['log10_low']}, {g['log10_high']}], {g['n']} valores" in cells['svm_rbf']
    assert sv['probability'] is True and '`probability=True`' in cells['svm_rbf']

    rf = f['rf']
    assert f"{rf['n_estimators']} árboles" in cells['rf']
    assert rf['max_depth'] == [None, 4, 8] and '`max_depth` ∈ {None, 4, 8}' in cells['rf']
    assert rf['min_samples_leaf'] == [1, 5] and '`min_samples_leaf` ∈ {1, 5}' in cells['rf']

    gb = f['gbm']
    assert gb['learning_rate'] == [0.03, 0.1] and '`learning_rate` ∈ {0.03, 0.1}' in cells['gbm']
    assert gb['max_depth'] == [2, 3, None] and '`max_depth` ∈ {2, 3, None}' in cells['gbm']
    assert gb['max_iter'] == [100, 300] and '`max_iter` ∈ {100, 300}' in cells['gbm']

    ml = f['mlp']
    assert [tuple(h) for h in ml['hidden_layer_sizes']] == [(16,), (32,), (32, 16)]
    assert 'Capas ocultas ∈ {(16,), (32,), (32, 16)}' in cells['mlp']
    assert ml['alpha'] == [1e-4, 1e-2] and '`alpha` ∈ {1e-4, 1e-2}' in cells['mlp']
    assert f"`max_iter = {ml['max_iter']}`" in cells['mlp']

    assert set(t['scaler']['families']) == {'logreg', 'svm_rbf', 'mlp'} and 'LogReg, SVM y MLP' in sec
    # Tamaños de las rejillas
    assert len(_grid(lr['C'])) == 13 and _grid(lr['C'])[0] == pytest.approx(1e-4)
    assert len(_grid(sv['C'])) == 6 and len(_grid(g)) == 5
    for fam in t['family_order']:
        assert fam in schema.VOCAB['arch']


def test_floors_match_doc_and_canonical(doc, cfg, canon):
    sec = section(doc, '4.4')
    rows = table_rows(sec)
    p = cfg['pisos']
    assert [r[0] for r in rows] == ['Mayoritaria', 'Aleatorio estratificado', 'Brillo, por técnica',
                                    'Visual permutado, por técnica']
    assert p['order'] == ['majority', 'stratified', 'brightness', 'visual_permutado']
    maj = tval(canon, p['majority']['canonical'])
    assert rows[0][2].startswith(r4(maj)) and 'clase 0' in rows[0][1] and p['majority']['predict'] == 0
    # Recalculado desde los soportes de prueba
    n0, n1 = canon['support.test.label0']['value'], canon['support.test.label1']['value']
    assert maj == f1_macro_from_confusion({'tp': 0, 'fp': 0, 'fn': n1, 'tn': n0})
    assert nums(rows[1][2]) == [p['stratified']['expected_f1']] and p['stratified']['proportion_split'] == 'train'
    assert 'rejilla de LogReg de §4.3' in rows[2][1] and p['brightness']['C_grid'] == 'techo_5d.familias.logreg.C'
    assert 'media de las 5 semillas' in rows[3][1] and len(p['visual_permutado']['seeds']) == 5
    assert 'mayor F1-Macro de prueba entre los cuatro' in sec and p['rule'] == 'max_f1_test'
    for key in ('majority', 'stratified', 'brightness'):
        assert p[key]['arch'] in schema.VOCAB['arch']


# ── §5 Contraste principal ───────────────────────────────────────────────────

def test_statistic_and_threshold(doc, cfg, canon):
    sec = section(doc, '5.1')
    c = cfg['contraste']
    assert f"**r = {c['r']:.2f}**" in sec and f"F1_{{azar}} = {c['f1_azar']}" in sec
    f1_beto = tval(canon, 'ref.beto_exp41.f1_macro')
    assert f'F1_{{BETO}} = {r4(f1_beto)}' in sec
    umbral = tval(canon, cfg['cifras_diseno']['umbral_f1_visual'])
    assert umbral == pytest.approx(c['f1_azar'] + c['r'] * (f1_beto - c['f1_azar']), abs=1e-15)
    assert f'> {r4(umbral)}' in sec
    assert 'θ_t ≤ 0' in sec and c['h1'] == 'theta > 0'


def test_bootstrap(doc, cfg, canon):
    sec = norm(section(doc, '5.2'))
    b = cfg['contraste']['bootstrap']
    n_test = canon[b['n']]['value']
    assert f"B = {b['B']} réplicas de los {n_test} índices de prueba con reemplazo" in sec
    assert f"`{b['rng']}({b['seed']})`" in sec
    assert 'p < 1/B' in norm(sec).replace('*p* < 1/B', 'p < 1/B') and cfg['contraste']['p_value']['if_zero'] == 'p < 1/B'


def test_floor_superiority(doc, cfg):
    sec = section(doc, '5.3')
    s = cfg['contraste']['superioridad_piso']
    assert 'D_t = \\overline{F1}_{img,t} - F1_{piso,t}' in sec
    assert f"valor esperado ({s['stratified_as_constant']})" in sec
    assert 'se promedian sus cinco semillas en cada réplica' in sec


def test_multiplicity(doc, cfg):
    sec = section(doc, '5.4')
    m = cfg['contraste']['multiplicidad']
    assert f"α = {m['alpha']} unilateral" in sec and m['sided'] == 'one' and m['method'] == 'holm'
    assert 'Dos familias de tres contrastes' in sec and m['family_size'] == 3
    assert m['families'] == ['superioridad_piso', 'no_inferioridad']
    assert 'intervalo percentil bilateral al nivel 1 − 2α_h' in sec and m['interval']['level'] == '1 - 2 * alpha_h'
    assert ('Si varias técnicas tienen el mismo valor p, todas reciben el menor α_h de los rangos que ocupan.'
            in sec) and m['ties'] == 'min_alpha_h'


def test_verdicts(doc, cfg):
    sec = section(doc, '5.5')
    order = [re.match(r'\*\*(V\d)\.', r[0]).group(1) for r in table_rows(sec)]
    v = cfg['contraste']['veredictos']
    assert order == v['order'] == ['V3', 'V1', 'V2', 'V4']
    g = cfg['contraste']['global']
    assert g['order'] == ['G1', 'G2', 'G3', 'G4']
    for code in g['order']:
        assert code in sec
    assert 'si las tres obtienen V3' in sec


def test_retention(doc, cfg):
    sec = section(doc, '5.6')
    assert '95 %' in sec and cfg['contraste']['retencion']['interval_level'] == 0.95
    assert 'descriptiva' in sec and cfg['contraste']['retencion']['descriptive'] is True


# ── §6–§8 ────────────────────────────────────────────────────────────────────

def test_length_control(doc, cfg):
    sec = section(doc, 6)
    l = cfg['longitud']
    assert 'Para cada técnica con V1, V2 o V4' in sec and l['applies_to'] == ['V1', 'V2', 'V4']
    assert f"α = {l['alpha']}" in sec and f"«{l['qualification']}»" in sec
    assert 'terciles de `n_tokens_beto`' in sec and l['descriptivo']['cuts_split'] == 'train'


def test_secondary(doc, cfg):
    sec = section(doc, 7)
    s = cfg['secundarios']
    assert 'Q de Cochran' in sec and 'McNemar exacto por pares con Holm (3 pares)' in sec
    assert s['entre_tecnicas']['pairs'] == 3 and s['entre_tecnicas']['alpha'] == 0.05
    assert 'media de `p_dep` sobre las cinco semillas' in sec
    assert s['prediccion_por_muestra'] == {'aggregate': 'mean_p_dep', 'seeds': 5, 'threshold': 0.5}


def test_sensitivity(doc, cfg, canon, matrix):
    rows = table_dict(section(doc, 8))
    s = cfg['sensibilidad']
    assert nums(rows['Retención exigida'][0])[:2] == s['retencion_exigida']['r']
    f1_beto = tval(canon, 'ref.beto_exp41.f1_macro')
    assert f'(1 − r)({r4(f1_beto)} − 0.5)' in rows['Formulación A'][0]
    for r, nid in s['formulacion_A']['canonical_delta'].items():
        assert tval(canon, nid) == pytest.approx((1 - float(r)) * (f1_beto - 0.5), abs=1e-15)
    shift = canon[s['variabilidad_beto']['canonical']]['value']['val']
    assert f'±{r4(shift)}' in rows['Variabilidad de BETO'][0]
    seg = rows['Segundo mejor modelo'][0]
    assert ('Se excluye al finalista y se aplica de nuevo la regla completa de §5 de la matriz '
            '(líderes, tolerancia, complejidad) a las cinco configuraciones restantes de la técnica.') in seg
    ph = matrix['fases'][s['segundo_mejor']['fase']]
    n = len(matrix['factores']['technique']) * len(ph['seeds'])
    assert f"{n} entrenamientos adicionales" in seg and s['segundo_mejor']['trainings'] == n
    assert s['segundo_mejor']['seeds'] == ph['seeds'] and ph['selection'] == s['segundo_mejor']['selection']
    assert 'mayoritaria, estratificado y brillo' in rows['Piso sin el visual permutado'][0]
    assert s['piso_sin_permutado'] == ['majority', 'stratified', 'brightness']


# ── §10 Acceso a prueba y §12 dependencias ───────────────────────────────────

def test_test_access_gate(doc, cfg, matrix):
    sec = section(doc, 10)
    assert f"`{schema.PROTOCOL_TAG}`" in sec
    for fase in ('final', 'control_permutacion', 'sensibilidad'):
        assert fase in schema.FASES_TEST_REQUIRES_TAG
        assert matrix['fases'][fase]['requires']['git_tag'] == schema.PROTOCOL_TAG
        assert matrix['fases'][fase]['requires']['protocol_file'] == DOC.relative_to(REPO).as_posix()


# ── §13 Cifras de diseño ─────────────────────────────────────────────────────

def test_design_figures(doc, cfg, canon):
    rows = {k: norm(v[0]) for k, v in table_dict(section(doc, 13)).items()}
    d, c = cfg['cifras_diseno'], cfg['contraste']
    assert nums(rows['r (principal)']) == [d['r']] and d['r'] == c['r']
    assert nums(rows['F1_azar']) == [d['f1_azar']] and d['f1_azar'] == c['f1_azar']
    assert nums(rows['α'])[0] == d['alpha'] == c['multiplicidad']['alpha']
    assert 'unilateral, Holm por familia' in rows['α']
    assert nums(rows['B']) == [d['B']] and d['B'] == c['bootstrap']['B']
    assert nums(rows['Semilla del *bootstrap*']) == [d['bootstrap_seed']] and d['bootstrap_seed'] == c['bootstrap']['seed']
    umbral_row = next(v for k, v in rows.items() if k.startswith('Umbral equivalente'))
    assert umbral_row == r4(tval(canon, d['umbral_f1_visual']))
    delta_row = next(v for k, v in rows.items() if k.startswith('δ de la formulación A'))
    assert delta_row == ' / '.join(r4(tval(canon, nid)) for nid in d['delta_formulacion_A'].values())
    assert list(map(float, d['delta_formulacion_A'])) == [d['r']] + cfg['sensibilidad']['retencion_exigida']['r']


def test_canonical_ids_exist(cfg, canon):
    ids = [cfg['cifras_diseno']['umbral_f1_visual'], cfg['pisos']['majority']['canonical'],
           cfg['sensibilidad']['variabilidad_beto']['canonical'], cfg['referencias']['beto']['canonical'],
           cfg['referencias']['emb768']['canonical'], *cfg['cifras_diseno']['delta_formulacion_A'].values()]
    for nid in ids:
        assert nid in canon, nid


# ── Enmienda 2 (2026-11-04): F2.3, F2.4 y longitud ───────────────────────────

def enmienda_2(doc):
    m = re.search(r'^## Enmienda 2 \(2026-11-04\)$(.*)\Z', doc, re.M | re.S)
    assert m, 'falta la Enmienda 2 al final del protocolo'
    assert not re.search(r'^## ', m.group(1), re.M), 'la Enmienda 2 debe ser la última sección'
    return m.group(1)


def test_enmienda_2_es_la_ultima_y_sigue_a_la_1(doc):
    sec = enmienda_2(doc)
    assert doc.index('## Enmienda 1 (2026-11-02)') < doc.index('## Enmienda 2 (2026-11-04)')
    assert '**antes de cualquier consulta de prueba de los finalistas**' in sec
    for n in (1, 2, 3, 4):
        assert f'**{n}. ' in sec


def test_enmienda_2_pisos_triviales(doc, cfg):
    sec = enmienda_2(doc)
    t = cfg['pisos_triviales']
    maj, st = t['majority'], t['stratified']
    assert '`p_dep = 0.0` en todas las muestras' in sec and maj['p_dep'] == 0.0
    assert maj['arch'] == cfg['pisos']['majority']['arch'] and maj['technique'] == cfg['pisos']['majority']['technique']
    assert 'como la constante 0.5 (§5.3)' in sec
    assert st['contrast_constant'] == cfg['pisos']['stratified']['expected_f1'] \
        == cfg['contraste']['superioridad_piso']['stratified_as_constant']
    assert f"`{st['rng']}({st['rng_seed']})`" in sec and st['proportion_split'] == 'train'
    assert '`p_dep ∈ {0.0, 1.0}`' in sec and st['p_dep_values'] == [0.0, 1.0]
    assert 'no interviene en ningún contraste' in sec and st['used_in_contrasts'] is False
    for e in (maj, st):
        assert (e['strategy'], e['seed'], e['fase']) == ('none', 42, 'referencia')


def test_enmienda_2_piso_brillo(doc, cfg):
    sec = enmienda_2(doc)
    b = cfg['piso_brillo']
    assert '256 × 256 en escala de grises, dividida entre 255' in sec
    assert b['image'] == {'source': 'nativa', 'size': [256, 256], 'mode': 'L', 'scale': 'mean / 255',
                          'visual_preprocessing': False}
    assert b['arch'] == cfg['pisos']['brightness']['arch'] and b['per_technique'] is True
    assert 'rejilla completa de LogReg de §4.3 (C y `class_weight`)' in sec
    assert b['grid'] == 'techo_5d.familias.logreg' and b['selection_rule'] == 'techo_5d.selection'
    assert b['scaler'] == {'name': 'StandardScaler', 'fit_split': 'train'}
    assert (b['strategy'], b['seed'], b['fase']) == ('none', 42, 'referencia')


def test_enmienda_2_descriptores(doc, cfg):
    sec = enmienda_2(doc)
    d = cfg['descriptores']
    assert 'entropía de Shannon, contraste, densidad de bordes y energía LBP' in sec
    assert d['features'] == ['shannon_entropy', 'contrast', 'edge_density', 'lbp_energy']
    assert '`configs/protocolo.json` → `descriptores`' in sec
    assert set(d['params']) == set(d['features'])
    assert d['image']['source'] == 'nativa' and d['image']['mode'] == 'L'


def test_f24_descriptores_fijados(cfg):
    """F2.4 fija la implementación de la celda 20 del notebook de CE antes de calcular."""
    d = cfg['descriptores']
    assert '_pendiente' not in d and d['_fijado'].startswith('F2.4, antes de calcular')
    impl = d['implementation']
    assert impl['original'] == 'notebooks/historial/camera_ready_CE.ipynb'
    assert (impl['original_cell'], impl['original_commit']) == (20, 'f126052')
    assert impl['reimplementation'] == 'results/f24_descriptores/f24_descriptores.py'
    assert (REPO / impl['original']).is_file()
    assert d['software']['scikit-image'] == '0.25.2'
    assert d['software']['scikit-image_kaggle'] is None and d['software']['_nota']
    p = d['params']
    assert all(v is not None for q in p.values() for v in q.values())
    assert (p['shannon_entropy']['bins'], p['shannon_entropy']['range'], p['shannon_entropy']['log_base']) ==         (256, [0, 256], 2)
    assert p['contrast']['ddof'] == 0 and 'no GLCM' in p['contrast']['definition']
    e = p['edge_density']
    assert e['detector'].startswith('skimage.filters.sobel') and e['input_scale'] == 'img / 255'
    assert 'media + 1 DE (ddof 0)' in e['thresholds'] and e['comparison'] == '>'
    lbp = p['lbp_energy']
    assert (lbp['P'], lbp['R'], lbp['method'], lbp['bins']) == (24, 3, 'uniform', 26)
    assert lbp['bins'] == lbp['P'] + 2
    t = d['tabla11']
    assert (t['split'], t['multiplicity'], t['k'], t['alpha']) == ('train', 'holm', 12, 0.05)
    assert t['k'] == 3 * len(d['features'])
    assert 'welch' in t['test'] and 'ddof 0' in t['effect_size']
    assert d['image']['size'] == [256, 256] and d['image']['visual_preprocessing'] is False
    assert d['families'] == 'techo_5d.family_order' and d['grids'] == 'techo_5d.familias'
    assert d['scaler']['families'] == cfg['techo_5d']['scaler']['families']
    assert d['arch'] == cfg['referencias']['descriptores']['arch']


def test_enmienda_2_longitud(doc, cfg):
    sec = enmienda_2(doc)
    lo = cfg['referencias']['longitud']
    assert ('`technique = text_length`, `arch = logreg`, `strategy = none`, `seed = 42`, fase `referencia`'
            in sec)
    assert (lo['technique'], lo['arch'], lo['strategy'], lo['seed'], lo['fase']) == \
        ('text_length', 'logreg', 'none', 42, 'referencia')
    assert lo['grid'] == 'techo_5d.familias.logreg' and lo['selection_rule'] == 'techo_5d.selection'
    assert lo['features'] == ['log(n_tokens_beto)', 'truncated_128']
    assert lo['fase'] in schema.FASES


def test_enmienda_2_vocabulario(cfg):
    combos = [cfg['pisos_triviales']['majority'], cfg['pisos_triviales']['stratified'],
              {**cfg['piso_brillo'], 'technique': 'perlin'}, cfg['referencias']['longitud']]
    for e in combos:
        for col in ('technique', 'arch', 'strategy'):
            assert e[col] in schema.VOCAB[col], (col, e[col])
    assert cfg['descriptores']['arch'] in schema.VOCAB['arch']
    assert schema.FILENAME_RE.match(schema.file_name('text_length', 'logreg', 'none', 42)).groupdict() == \
        {'technique': 'text_length', 'arch': 'logreg', 'strategy': 'none', 'seed': '42'}
