"""
tests/test_canonical_numbers.py
═══════════════════════════════
F0.4 — Pruebas del archivo de cifras canónicas (results/canonical_numbers.json).
Plan de Trabajo TT2 (v2) · Rol SV

Esquema          El archivo y cada fragmento validan contra src/canonical/schema.json;
                 toda fuente del repo existe y toda referencia a otra cifra resuelve.
Determinismo     Dos builds seguidos dan los mismos bytes; con umap_params.pkl,
                 el build coincide con el archivo commiteado.
Alcanzabilidad   Cada F1-Macro (no derivado) es alcanzable con una matriz de
                 confusión entera con los soportes reales del split; exacto si es
                 verificado, al redondeo de sus decimales si es transcrito. Se
                 eximen las cifras de diseño (DESIGN_PREFIXES: márgenes y F1
                 mínimos detectables de F1.1), que no son F1 observados.
Confusión        El F1 recalculado de cada matriz es el valor guardado y la matriz
                 suma los soportes del split.
Estructura       908 + 114 + 114 = 1,136; soportes por clase coherentes; 54 corridas
                 de F3.2 (repetición, f32.*) con el mismo y_true de val; la v0
                 sustituida se conserva como f32v0.* (54 + 18 + 18, con nota).
Selección F3.3   f33.*: finalista, segundo y líder por técnica coinciden con
                 select_finalist/select_second; límite, holgura y volteos coherentes.
Derivadas        Cada derivada se recalcula desde depends_on; cada agregado
                 transcrito (summarizes) es compatible con sus cifras.
Fusión           Una clave repetida entre fragmentos o colectores es un error,
                 salvo sustitución explícita (supersedes); la sustituida se
                 conserva intacta en superseded.
Predicciones     Cada F1 verificado cuya fuente es un archivo de
                 results/predictions/ se recalcula desde ese archivo (F2.2).

Ejecución (desde la raíz del repo):
    CANON_UMAP_PARAMS=<umap_params.pkl> CANON_STRICT=1 \
        python -m pytest tests/test_canonical_numbers.py -v -rxs

Variables de entorno (las mismas de src/canonical/build.py):
    CANON_UMAP_PARAMS  umap_params.pkl. Sin definir → se omite la comparación con
                       el archivo commiteado; definida pero inexistente → falla.
    CANON_STRICT       1 → una variable sin definir hace fallar en vez de omitir.

Cifras conocidas como inconsistentes llevan "xfail" en el archivo y se marcan
xfail(strict=True): si dejan de fallar, la prueba lo señala (XPASS → falla).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.canonical import build as canon  # noqa: E402

UMAP_PATH = os.environ.get('CANON_UMAP_PARAMS') or None
STRICT = os.environ.get('CANON_STRICT', '0') == '1'

DOC = json.loads(canon.OUT_PATH.read_text(encoding='utf-8'))
NUMBERS = DOC['numbers']


def need_umap():
    if UMAP_PATH:
        return UMAP_PATH
    if STRICT:
        pytest.fail('CANON_STRICT=1 y CANON_UMAP_PARAMS no está definida')
    pytest.skip('CANON_UMAP_PARAMS no está definida')


def supports(entry, split):
    prefix = entry['supports_ref'][split]
    return NUMBERS[f'{prefix}.label1']['value'], NUMBERS[f'{prefix}.label0']['value']


# Cifras de diseño en escala de F1 que no provienen de una matriz de confusión
DESIGN_PREFIXES = ('f11.', 'f12.')
# Cifras extraídas de la Fase 5 y de F4.6 (bootstrap, p, α, κ...): tampoco salen de una matriz de confusión
F5_PREFIXES = ('f5.', 'f46.')
# F4.1, F4.7, F4.2, F4.3 y F4.4: R², alpha, ρ, ΔF1, p... Sus F1 de prueba verificados se prueban aparte (test_f41_*, test_f42_*)
F4_PREFIXES = ('f41.', 'f47.', 'f42.', 'f43.', 'f44.')


def is_leaf_metric(entry):
    return entry['kind'] == 'metric' and 'depends_on' not in entry and 'summarizes' not in entry


def leaf_cases():
    """(id, split) de cada F1 no derivado y no nulo; xfail si la entrada lo declara."""
    cases = []
    for nid, e in sorted(NUMBERS.items()):
        if not is_leaf_metric(e) or nid.startswith(DESIGN_PREFIXES + F5_PREFIXES + F4_PREFIXES):
            continue
        for s in canon.METRIC_SPLITS:
            if e['value'][s] is None:
                continue
            marks = [pytest.mark.xfail(strict=True, reason=e['xfail']['reason'])] if 'xfail' in e else []
            cases.append(pytest.param(nid, s, id=f'{nid}[{s}]', marks=marks))
    return cases


# ── Esquema ───────────────────────────────────────────────────────────────
def test_esquema_archivo():
    canon.validate(DOC)
    assert DOC['schema_version'] == canon.SCHEMA_VERSION


@pytest.mark.parametrize('path', sorted(REPO.glob(canon.FRAGMENT_GLOB)), ids=lambda p: canon.rel(p))
def test_esquema_fragmento(path):
    canon.validate(json.loads(path.read_text(encoding='utf-8')), fragment=True)


def test_referencias_resuelven():
    for nid, e in NUMBERS.items():
        if 'path' in e['source']:
            assert (REPO / e['source']['path']).is_file(), f'{nid}: no existe {e["source"]["path"]}'
        for s, prefix in e.get('supports_ref', {}).items():
            for k in (0, 1):
                assert f'{prefix}.label{k}' in NUMBERS, f'{nid}: falta {prefix}.label{k}'
        for dep in e.get('depends_on', []) + e.get('summarizes', {}).get('of', []):
            assert dep in NUMBERS, f'{nid}: referencia inexistente {dep}'


def test_evaluator_y_null_reason():
    """El esquema ya lo exige; aquí se comprueba que no queden vacíos semánticos."""
    for nid, e in NUMBERS.items():
        assert e['evaluator'].strip(), nid
        if e['kind'] == 'metric':
            train = nid.startswith('f47.') or (nid.startswith('f42.') and '.estrato_prueba.' not in nid)
            assert set(e['value']) == ({'val', 'test', 'train'} if train else {'val', 'test'}), nid
            for s in canon.METRIC_SPLITS:
                if e['value'][s] is None:
                    assert e['null_reason'][s].strip(), f'{nid}[{s}]'


# ── Determinismo ──────────────────────────────────────────────────────────
def test_determinismo_dos_builds():
    a = canon.dumps(canon.build(UMAP_PATH))
    b = canon.dumps(canon.build(UMAP_PATH))
    assert a == b


def test_coincide_con_archivo_commiteado():
    text = canon.dumps(canon.build(need_umap(), strict=True))
    committed = canon.OUT_PATH.read_bytes().replace(b'\r\n', b'\n').decode('utf-8')
    assert text == committed, 'results/canonical_numbers.json no está al día: python -m src.canonical.build'


def test_conteos_sin_pkl_coinciden():
    """La vía sin pkl (tabla del hallazgo 7) da los mismos conteos que el pkl."""
    alt = canon.build(None)['numbers']
    for nid, e in NUMBERS.items():
        if e['kind'] == 'count':
            assert alt[nid]['value'] == e['value'], nid


# ── Alcanzabilidad del F1-Macro ───────────────────────────────────────────
@pytest.mark.parametrize('nid,split', leaf_cases())
def test_alcanzabilidad_f1(nid, split):
    e = NUMBERS[nid]
    value = e['value'][split]
    pos, neg = supports(e, split)
    best = min(
        (abs(canon.f1_macro_from_confusion({'tp': tp, 'fn': pos - tp, 'tn': tn, 'fp': neg - tn}) - value), tp, tn)
        for tp in range(pos + 1) for tn in range(neg + 1))
    tol = 1e-12 if e['status'] == 'verificado' else 0.5 * 10 ** -e['decimals'] + 1e-12
    assert best[0] <= tol, (f'{nid}[{split}] = {value} no alcanzable con soportes {neg}/{pos}; '
                            f'más cercano a {best[0]:.6f} (TP={best[1]}, TN={best[2]})')


# ── Matriz de confusión ───────────────────────────────────────────────────
CONFUSION_CASES = [(nid, s) for nid, e in sorted(NUMBERS.items()) for s in e.get('confusion', {})]


@pytest.mark.parametrize('nid,split', CONFUSION_CASES, ids=[f'{n}[{s}]' for n, s in CONFUSION_CASES])
def test_matriz_confusion(nid, split):
    e = NUMBERS[nid]
    cm = e['confusion'][split]
    pos, neg = supports(e, split)
    assert cm['tp'] + cm['fn'] == pos and cm['tn'] + cm['fp'] == neg
    assert canon.f1_macro_from_confusion(cm) == e['value'][split]


def test_verificados_tienen_confusion():
    for nid, e in NUMBERS.items():
        if e['status'] == 'verificado' and is_leaf_metric(e):
            for s in canon.METRIC_SPLITS:
                assert (e['value'][s] is None) == (s not in e['confusion']), f'{nid}[{s}]'


# ── Conteos estructurales ─────────────────────────────────────────────────
def test_conteos_estructurales():
    n = {s: NUMBERS[f'n.{s}']['value'] for s in canon.SPLITS}
    assert n == {'train': 908, 'val': 114, 'test': 114}
    assert NUMBERS['n.total']['value'] == sum(n.values()) == 1136
    for s in canon.SPLITS:
        assert NUMBERS[f'support.{s}.label0']['value'] + NUMBERS[f'support.{s}.label1']['value'] == n[s]
    assert (NUMBERS['support.total.label0']['value'], NUMBERS['support.total.label1']['value']) == (695, 441)
    assert (NUMBERS['support.val.label0']['value'], NUMBERS['support.val.label1']['value']) == (70, 44)
    assert (NUMBERS['support.test.label0']['value'], NUMBERS['support.test.label1']['value']) == (70, 44)


PREP = json.loads((REPO / 'configs' / 'preprocesamiento.json').read_text(encoding='utf-8'))


def _f32_keys(prefix):
    seeds = sorted(k for k in NUMBERS if k.startswith(prefix + '.') and '.seed' in k)
    means = sorted(k for k in NUMBERS if k.startswith(prefix + '.') and k.endswith('.f1_macro_mean'))
    stds = sorted(k for k in NUMBERS if k.startswith(prefix + '.') and k.endswith('.f1_macro_std'))
    return seeds, means, stds


def test_f32_estructura():
    """F3.2, repetición: 54 CSV F1.5 de primera_pasada con el mismo y_true de val y
    claves con los identificadores de la repetición (vit_b16, variante de F1.4)."""
    y_by_run = {}
    for path in sorted(canon.F32_PRED_DIR.glob('*.csv')):
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert {r['split'] for r in rows} == {'val'}, f'{path.name}: F3.2 solo tiene val'
        y_by_run[path.name] = tuple(int(r['y_true']) for r in rows)
    assert len(y_by_run) == 54
    assert len(set(y_by_run.values())) == 1, 'y_true distinto entre corridas'
    y = next(iter(y_by_run.values()))
    assert Counter(y) == {0: NUMBERS['support.val.label0']['value'], 1: NUMBERS['support.val.label1']['value']}
    seeds, means, stds = _f32_keys('f32')
    assert len(seeds) == 54 and len(means) == 18 and len(stds) == 18
    assert all(NUMBERS[k]['value']['test'] is None for k in seeds + means + stds), 'F3.2 no debe tener test'
    for k in seeds:
        _, tech, arch, _, variant = k.split('.')[:5]
        assert arch in {'resnet18', 'efficientnet_b0', 'vit_b16'}, k
        assert variant == PREP[tech]['variant'].replace('-', '_'), k
        assert NUMBERS[k]['source']['path'].startswith('results/predictions/primera_pasada/'), k


def test_f32v0_estructura():
    """F3.2 v0 (sustituida): mismas 54 + 18 + 18 cifras que antes de la repetición,
    con prefijo f32v0.* y marcadas como sustituidas."""
    with open(canon.F32V0_DIR / 'predictions.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    y_by_run = defaultdict(list)
    for r in rows:
        y_by_run[r['run_id']].append(int(r['y_true']))
    assert len(y_by_run) == 54
    assert len({tuple(y) for y in y_by_run.values()}) == 1, 'y_true distinto entre corridas'
    seeds, means, stds = _f32_keys('f32v0')
    assert len(seeds) == 54 and len(means) == 18 and len(stds) == 18
    for k in seeds + means + stds:
        e = NUMBERS[k]
        assert e['value']['test'] is None, k
        assert e['note'] == canon.F32V0_NOTE, k
        assert k.split('.')[4] == 'px224_flip', k
        assert all(d.startswith('f32v0.') for d in e.get('depends_on', [])), k
        if 'depends_on' not in e:
            assert e['source']['path'] == 'results/f32_primera_pasada_v0/predictions.csv', k


# ── Selección de F3.3 (f33.*) ─────────────────────────────────────────────
F33_ROLES = ('finalista', 'segundo', 'lider')
F33_PER_TECH = 3 * 2 + 4   # 3 roles × (media, DE) + límite + holgura + 2 volteos


def _f33_role(tech, role):
    keys = sorted(k for k in NUMBERS if k.startswith(f'f33.{tech}.{role}.'))
    configs = {tuple(k.split('.')[3:5]) for k in keys}
    assert len(configs) == 1 and len(keys) == 2, f'f33.{tech}.{role}: {keys}'
    (arch, strat), = configs
    return arch, strat, f'f33.{tech}.{role}.{arch}.{strat}'


@pytest.fixture(scope='module')
def seleccion():
    from src.selection.select_finalist import select_finalists, select_second
    table, _ = select_finalists(str(canon.F32_PRED_DIR / '*.csv'))
    return select_second(table)


def test_f33_presencia():
    keys = [k for k in NUMBERS if k.startswith('f33.')]
    assert len(keys) == 3 * F33_PER_TECH
    for tech in ('perlin', 'gray_scott', 'lorenz'):
        for role in F33_ROLES:
            _f33_role(tech, role)
        for k in ('limite_empate.f1_macro', 'holgura_finalista.f1_macro',
                  'volteos_min.una_corrida', 'volteos_min.repartido'):
            assert f'f33.{tech}.{k}' in NUMBERS, f'falta f33.{tech}.{k}'


@pytest.mark.parametrize('tech', ['perlin', 'gray_scott', 'lorenz'])
def test_f33_consistencia(tech, seleccion):
    g = seleccion[seleccion['technique'] == tech]
    cols = {'finalista': 'finalist', 'segundo': 'second', 'lider': 'leader'}
    base = {}
    for role, col in cols.items():
        arch, strat, base[role] = _f33_role(tech, role)
        row = g[g[col]]
        assert len(row) == 1 and (row['arch'].iloc[0], row['strategy'].iloc[0]) == (arch, strat), \
            f'{tech}/{role}: canónico {arch}/{strat} != select_finalist'
        # La cifra f33 es la misma que la media/DE f32 de esa configuración
        variant = PREP[tech]['variant'].replace('-', '_')
        for s in ('mean', 'std'):
            f32 = NUMBERS[f'f32.{tech}.{arch}.{strat}.{variant}.f1_macro_{s}']
            e = NUMBERS[f'{base[role]}.f1_macro_{s}']
            assert e['value'] == f32['value'] and e['depends_on'] == f32['depends_on'], f'{base[role]}.{s}'
        assert NUMBERS[f'{base[role]}.f1_macro_mean']['value']['val'] == pytest.approx(row['f1_val_mean'].iloc[0], abs=1e-12)
        assert NUMBERS[f'{base[role]}.f1_macro_std']['value']['val'] == pytest.approx(row['f1_val_std'].iloc[0], abs=1e-12)
    lim = NUMBERS[f'f33.{tech}.limite_empate.f1_macro']
    hol = NUMBERS[f'f33.{tech}.holgura_finalista.f1_macro']
    assert lim['depends_on'] == [f'{base["lider"]}.f1_macro_mean', f'{base["lider"]}.f1_macro_std']
    assert hol['depends_on'] == [f'{base["finalista"]}.f1_macro_mean', f'f33.{tech}.limite_empate.f1_macro']
    lead = g[g['leader']].iloc[0]
    assert lim['value']['val'] == pytest.approx(lead['f1_val_mean'] - lead['tolerance'], abs=1e-12)
    assert hol['value']['val'] >= 0, 'el finalista debe estar dentro de la zona de empate'
    assert lim['value']['test'] is None and hol['value']['test'] is None
    # Volteos: del CSV de fragilidad, que se refiere a este finalista
    with open(canon.F33_VOLTEOS, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['tecnica'] == tech]
    single = [int(r['min_volteos']) for r in rows if r['busqueda'] == 'una corrida']
    spread = [int(r['min_volteos']) for r in rows if r['busqueda'].startswith('repartida')]
    assert NUMBERS[f'f33.{tech}.volteos_min.una_corrida']['value'] == min(single)
    assert NUMBERS[f'f33.{tech}.volteos_min.repartido']['value'] == spread[0]
    assert 1 <= NUMBERS[f'f33.{tech}.volteos_min.repartido']['value'] <= min(single)
    assert DOC['inputs'][canon.rel(canon.F33_VOLTEOS)] == canon.text_sha256(canon.F33_VOLTEOS)


# ── Derivadas y agregados transcritos ─────────────────────────────────────
DERIVED = sorted(k for k, e in NUMBERS.items() if 'depends_on' in e)


@pytest.mark.parametrize('nid', DERIVED)
def test_derivadas(nid):
    e = NUMBERS[nid]
    deps = [NUMBERS[d] for d in e['depends_on']]
    assert e['status'] == canon.weakest([d['status'] for d in deps])
    if e['kind'] == 'count':
        assert e['value'] == canon.derive(e['derivation'], [d['value'] for d in deps])
        return
    for s in canon.METRIC_SPLITS:
        vals = [d['value'][s] for d in deps]
        absent = any if e['derivation'] == 'diff' else all
        expected = None if absent(v is None for v in vals) else canon.derive(e['derivation'], vals)
        assert e['value'][s] == expected, f'{nid}[{s}]'


SUMMARIES = sorted(k for k, e in NUMBERS.items() if 'summarizes' in e)
SUMMARY_OPS = {'mean': np.mean, 'std_ddof0': lambda x: np.std(x, ddof=0), 'std_ddof1': lambda x: np.std(x, ddof=1)}


@pytest.mark.parametrize('nid', SUMMARIES)
def test_agregados_transcritos(nid):
    e = NUMBERS[nid]
    op = SUMMARY_OPS[e['summarizes']['op']]
    for s in canon.METRIC_SPLITS:
        if e['value'][s] is None:
            continue
        vals = [NUMBERS[d]['value'][s] for d in e['summarizes']['of']]
        assert round(float(op(vals)), e['decimals']) == e['value'][s], f'{nid}[{s}]'


# ── Fusión de fragmentos ──────────────────────────────────────────────────
def _fragment(tmp_path, name, numbers):
    path = tmp_path / name / 'canonical_fragment.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'schema_version': canon.SCHEMA_VERSION, 'numbers': numbers}), encoding='utf-8')
    return path


def _entry(value):
    return {'kind': 'count', 'status': 'reportado', 'value': value, 'evaluator': 'prueba',
            'source': {'path': 'README.md', 'locator': 'prueba'}}


def test_claves_duplicadas_entre_fragmentos(tmp_path):
    a = _fragment(tmp_path, 'fa', {'x.dup': _entry(1)})
    b = _fragment(tmp_path, 'fb', {'x.dup': _entry(2)})
    with pytest.raises(ValueError, match="clave duplicada 'x.dup'"):
        canon.build(None, fragment_paths=[a, b])


def test_clave_duplicada_con_colector(tmp_path):
    frag = _fragment(tmp_path, 'fz', {'n.total': _entry(1136)})
    with pytest.raises(ValueError, match="clave duplicada 'n.total'"):
        canon.build(None, fragment_paths=[frag])


def test_ciclo_y_dependencia_inexistente():
    base = {'kind': 'count', 'evaluator': 'prueba', 'source': {'path': 'README.md', 'locator': 'prueba'},
            'derivation': 'sum'}
    with pytest.raises(ValueError, match='ciclo'):
        canon.resolve_derived({'a': {**base, 'depends_on': ['b']}, 'b': {**base, 'depends_on': ['a']}})
    with pytest.raises(ValueError, match='inexistente'):
        canon.resolve_derived({'a': {**base, 'depends_on': ['zz']}})


def test_derivada_declarada_debe_coincidir():
    leaf = _entry(2)
    bad = {**_entry(5), 'depends_on': ['l1', 'l2'], 'derivation': 'sum'}
    with pytest.raises(ValueError, match='declarado'):
        canon.resolve_derived({'l1': dict(leaf), 'l2': dict(leaf), 'd': bad})


# ── Sustitución (supersedes) ──────────────────────────────────────────────
def _metric(value_test, status='reportado', **extra):
    e = {'kind': 'metric', 'metric': 'f1_macro', 'status': status, 'value': {'val': None, 'test': value_test},
         'null_reason': {'val': 'prueba'}, 'evaluator': 'prueba', 'decimals': 4,
         'supports_ref': {'val': 'support.val', 'test': 'support.test'},
         'source': {'path': 'README.md', 'locator': 'prueba'}}
    e.update(extra)
    return e


@pytest.mark.parametrize('order', ['vieja_primero', 'nueva_primero'])
def test_supersedes_misma_clave(order):
    old, new = _metric(0.7586), _metric(0.7590, supersedes='x.f1')
    srcs = [('f_old', {'x.f1': old}), ('f_new', {'x.f1': new})]
    numbers, superseded = canon.merge(dict(srcs if order == 'vieja_primero' else srcs[::-1]))
    assert numbers['x.f1'] is new
    assert superseded == {'x.f1': {'by': 'f_new:x.f1', 'origin': 'f_old', 'entry': old}}


def test_supersedes_otra_clave():
    old, new = _metric(0.7586), _metric(0.7590, supersedes='x.old')
    numbers, superseded = canon.merge({'a': {'x.old': old}, 'b': {'x.new': new}})
    assert set(numbers) == {'x.new'} and superseded['x.old']['entry'] is old


def test_supersedes_objetivo_inexistente():
    with pytest.raises(ValueError, match='no existe'):
        canon.merge({'a': {'x.new': _metric(0.7, supersedes='x.zz')}})
    with pytest.raises(ValueError, match='no existe'):
        canon.merge({'a': {'x.f1': _metric(0.7, supersedes='x.f1')}})


def test_supersedes_doble():
    with pytest.raises(ValueError, match='clave duplicada'):
        canon.merge({'a': {'x.f1': _metric(0.7)}, 'b': {'x.f1': _metric(0.71, supersedes='x.f1')},
                     'c': {'x.f1': _metric(0.72, supersedes='x.f1')}})
    with pytest.raises(ValueError, match='dos veces'):
        canon.merge({'a': {'x.old': _metric(0.7)}, 'b': {'x.n1': _metric(0.71, supersedes='x.old'),
                                                       'x.n2': _metric(0.72, supersedes='x.old')}})


def test_superseded_se_conserva_intacta():
    """Cada entrada sustituida del archivo es idéntica a la de su fragmento de origen."""
    for nid, rec in DOC.get('superseded', {}).items():
        frag = json.loads((REPO / rec['origin']).read_text(encoding='utf-8'))
        assert frag['numbers'][nid] == rec['entry'], nid
        assert NUMBERS[rec['by'].rsplit(':', 1)[1]]['supersedes'] == nid


# ── Derivación diff ───────────────────────────────────────────────────────
def test_diff_con_split_nulo():
    a = _metric(0.76, status='verificado')
    a['value']['val'] = 0.72
    b = _metric(0.70)
    d = {'kind': 'metric', 'metric': 'f1_macro', 'evaluator': 'prueba', 'derivation': 'diff',
         'depends_on': ['a', 'b'], 'supports_ref': dict(a['supports_ref']),
         'source': {'path': 'README.md', 'locator': 'prueba'}}
    numbers = {'a': a, 'b': b, 'd': d}
    canon.resolve_derived(numbers)
    assert d['value']['test'] == pytest.approx(0.06) and d['value']['val'] is None
    assert 'b' in d['null_reason']['val'] and d['status'] == 'reportado'


def test_diff_exige_dos_dependencias():
    base = {'kind': 'count', 'status': 'reportado', 'evaluator': 'prueba',
            'source': {'path': 'README.md', 'locator': 'prueba'}}
    numbers = {'a': {**base, 'value': 1}, 'b': {**base, 'value': 2}, 'c': {**base, 'value': 3},
               'd': {k: v for k, v in base.items() if k != 'status'} | {'depends_on': ['a', 'b', 'c'],
                                                                         'derivation': 'diff'}}
    with pytest.raises(ValueError, match='dos dependencias'):
        canon.resolve_derived(numbers)


# ── Verificados desde archivos de predicciones (F2.2) ─────────────────────
PRED_CASES = sorted(nid for nid, e in NUMBERS.items()
                    if e['status'] == 'verificado' and is_leaf_metric(e)
                    and e['source'].get('path', '').startswith('results/predictions/'))


@pytest.mark.parametrize('nid', PRED_CASES)
def test_verificado_desde_predicciones(nid):
    e = NUMBERS[nid]
    with open(REPO / e['source']['path'], newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for s in canon.METRIC_SPLITS:
        if e['value'][s] is None:
            continue
        sel = [r for r in rows if r['split'] == s]
        cm = canon.confusion_counts([int(r['y_true']) for r in sel],
                                    [int(float(r['p_dep']) > 0.5) for r in sel])
        assert cm == e['confusion'][s], f'{nid}[{s}]'
        assert canon.f1_macro_from_confusion(cm) == e['value'][s], f'{nid}[{s}]'


# ── F3.4 (f34.*), F3.5/F3.6/F5.5 (f35/f36/f55.*) y longitud (ref.longitud) ─
TECHS = ('gray_scott', 'perlin', 'lorenz')
F34_PER_TECH = 2 * 3 + 2 + 1 + 1 + 2   # 2 roles × 3 semillas + media/DE trial 0 + media mejor + ganancia + 2 conteos
F3X = {'f35': ('final', 'finalista', 'finalist'), 'f36': ('control_permutacion', 'permutado', 'finalist'),
       'f55': ('sensibilidad', 'segundo', 'second')}
F3X_SEEDS = (42, 7, 123, 2024, 99)


def test_f34_presencia_y_conteos():
    keys = [k for k in NUMBERS if k.startswith('f34.')]
    assert len(keys) == 3 * F34_PER_TECH
    for tech in TECHS:
        for role in ('trial0', 'mejor_trial'):
            for s in (42, 7, 123):
                e = NUMBERS[f'f34.{tech}.{role}.seed{s}.f1_macro']
                assert e['status'] == 'verificado' and e['value']['test'] is None
                assert e['null_reason']['test'] == 'F3.4 solo validación (matriz §4)'
        for k in ('trial0.f1_macro_mean', 'trial0.f1_macro_std', 'mejor_trial.f1_macro_mean', 'ganancia.f1_macro'):
            assert NUMBERS[f'f34.{tech}.{k}']['value']['test'] is None, k
        assert NUMBERS[f'f34.{tech}.trials']['value'] == 12


@pytest.mark.parametrize('tech', TECHS)
def test_f34_recalculo_y_hpo_result(tech):
    d = REPO / 'results' / 'f34_hpo' / tech
    dec = json.loads((d / 'hpo_result.json').read_text(encoding='utf-8'))['decision']
    best = dec['best_trial']
    for role, trial in (('trial0', 0), ('mejor_trial', best)):
        for s in (42, 7, 123):
            e = NUMBERS[f'f34.{tech}.{role}.seed{s}.f1_macro']
            run = json.loads((d / 'runs' / f'trial{trial:02d}__seed{s}.json').read_text(encoding='utf-8'))
            assert e['source']['path'] == canon.rel(d / 'runs' / f'trial{trial:02d}__seed{s}.json')
            cm = canon.confusion_counts(run['val_true'], [int(p > 0.5) for p in run['val_probs']])
            assert cm == e['confusion']['val'] and canon.f1_macro_from_confusion(cm) == e['value']['val']
            assert DOC['inputs'][e['source']['path']] == canon.text_sha256(REPO / e['source']['path'])
    v = lambda k: NUMBERS[f'f34.{tech}.{k}']['value']['val']  # noqa: E731
    assert v('trial0.f1_macro_mean') == pytest.approx(dec['trial0_mean'], abs=1e-12)
    assert v('trial0.f1_macro_std') == pytest.approx(dec['trial0_std'], abs=1e-12)
    assert v('mejor_trial.f1_macro_mean') == pytest.approx(dec['best_mean'], abs=1e-12)
    assert v('ganancia.f1_macro') == pytest.approx(dec['gain'], abs=1e-12)
    assert NUMBERS[f'f34.{tech}.trial_adoptado']['value'] == dec['adopted_trial']
    # regla de la matriz §6 sobre las cifras canónicas
    adopted = best if v('ganancia.f1_macro') > v('trial0.f1_macro_std') else 0
    assert NUMBERS[f'f34.{tech}.trial_adoptado']['value'] == adopted


def _f3x_role_keys(prefix, tech, role):
    keys = sorted(k for k in NUMBERS if k.startswith(f'{prefix}.{tech}.{role}.') and '.seed' in k)
    configs = {tuple(k.split('.')[3:5]) for k in keys}
    assert len(configs) == 1, f'{prefix}.{tech}: {configs}'
    return keys, configs.pop()


def test_f3x_presencia_y_conteos():
    for prefix in F3X:
        assert len([k for k in NUMBERS if k.startswith(f'{prefix}.')]) == 3 * (5 + 2), prefix
        for tech in TECHS:
            _, role, _ = F3X[prefix]
            keys, _ = _f3x_role_keys(prefix, tech, role)
            assert sorted(int(k.split('.seed')[1].split('.')[0]) for k in keys) == sorted(F3X_SEEDS)
            for suffix in ('mean', 'std'):
                e = NUMBERS[f'{prefix}.{tech}.{role}.f1_macro_{suffix}']
                assert sorted(e['depends_on']) == keys
                assert e['derivation'] == {'mean': 'mean', 'std': 'std_ddof1'}[suffix]
                assert e['value']['val'] is not None and e['value']['test'] is not None


@pytest.mark.parametrize('prefix', sorted(F3X))
def test_f3x_configuracion_de_seleccion(prefix, seleccion):
    """f35/f36 usan exactamente el finalista de select_finalist; f55 el de select_second."""
    _, role, col = F3X[prefix]
    for tech in TECHS:
        row = seleccion[(seleccion['technique'] == tech) & seleccion[col]]
        assert len(row) == 1
        _, (arch, strat) = _f3x_role_keys(prefix, tech, role)
        assert (arch, strat) == (row['arch'].iloc[0], row['strategy'].iloc[0]), f'{prefix}.{tech}'


@pytest.mark.parametrize('prefix', sorted(F3X))
def test_f3x_recalculo_desde_predicciones(prefix):
    fase, role, _ = F3X[prefix]
    for tech in TECHS:
        keys, (arch, strat) = _f3x_role_keys(prefix, tech, role)
        for nid in keys:
            e = NUMBERS[nid]
            seed = int(nid.split('.seed')[1].split('.')[0])
            path = f'results/predictions/{fase}/{tech}__{arch}__{strat}__seed{seed}.csv'
            assert e['source']['path'] == path and e['status'] == 'verificado'
            assert DOC['inputs'][path] == canon.text_sha256(REPO / path)
            with open(REPO / path, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            for s in canon.METRIC_SPLITS:
                sel = [r for r in rows if r['split'] == s]
                cm = canon.confusion_counts([int(r['y_true']) for r in sel], [int(float(r['p_dep']) > 0.5) for r in sel])
                assert cm == e['confusion'][s] and canon.f1_macro_from_confusion(cm) == e['value'][s], f'{nid}[{s}]'
        # contra summary_by_config.csv de la fase (4 decimales)
        rdir = {'final': 'f35_final', 'control_permutacion': 'f36_permutacion', 'sensibilidad': 'f55_sensibilidad'}[fase]
        with open(REPO / 'results' / rdir / tech / 'summary_by_config.csv', newline='', encoding='utf-8') as f:
            row, = list(csv.DictReader(f))
        for suffix in ('mean', 'std'):
            e = NUMBERS[f'{prefix}.{tech}.{role}.f1_macro_{suffix}']
            for s in canon.METRIC_SPLITS:
                assert round(e['value'][s], 4) == float(row[f'f1_{s}_{suffix}']), f'{prefix}.{tech}.{suffix}[{s}]'


def test_f36_nota_enmienda_4_a16():
    for tech in TECHS:
        keys, _ = _f3x_role_keys('f36', tech, 'permutado')
        for nid in keys + [f'f36.{tech}.permutado.f1_macro_mean', f'f36.{tech}.permutado.f1_macro_std']:
            assert 'Enmienda 4, A16' in NUMBERS[nid]['note'], nid


def test_ref_longitud():
    e = NUMBERS['ref.longitud.f1_macro']
    assert e['status'] == 'verificado' and e['value']['test'] is not None and 'test' in e['confusion']
    assert 'null_reason' not in e
    path = 'results/predictions/referencia/text_length__logreg__none__seed42.csv'
    assert e['source'] == {'path': path, 'locator': 'split=val, split=test'}
    res = json.loads((REPO / 'results' / 'f45_longitud' / 'resultado.json').read_text(encoding='utf-8'))
    assert e['value']['val'] == pytest.approx(res['f1_val'], abs=1e-12)


# ── Fase 5 (f5.*), F4.6 (f46.*) y desbloqueo de ref.* (protocolo §10) ─────
F5_RES = json.loads((REPO / 'results' / 'f5_contraste' / 'resultados.json').read_text(encoding='utf-8'))
F46_RES = json.loads((REPO / 'results' / 'f5_contraste' / 'f46_resultados.json').read_text(encoding='utf-8'))
REF_PREFIXES = ('ref.techo5d', 'ref.piso.', 'ref.descriptores.', 'ref.emb768', 'ref.longitud')
# Cifras commiteadas antes de la Fase 5 (merge de F5-ejecución, PR #36)
PRE_F5_COMMIT = 'dd2e462ca7d2f40326392da6997916a34c7c3606'


def _f1_csv(path, split):
    with open(REPO / path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split]
    return canon.f1_macro_from_confusion(canon.confusion_counts(
        [int(r['y_true']) for r in rows], [int(float(r['p_dep']) > 0.5) for r in rows]))


def test_referencias_desbloqueadas_tras_fase5():
    """Tras la Fase 5, ninguna referencia queda bloqueada y su prueba sale de su CSV."""
    keys = [k for k, e in NUMBERS.items() if k.startswith(REF_PREFIXES) and e['kind'] == 'metric']
    assert len(keys) == 16
    for nid in keys:
        e = NUMBERS[nid]
        assert e['status'] == 'verificado' and e['value']['test'] is not None, nid
        assert 'bloqueado' not in json.dumps(e.get('null_reason', {}), ensure_ascii=False), nid
        assert e['value']['test'] == _f1_csv(e['source']['path'], 'test'), nid
        assert DOC['inputs'][e['source']['path']] == canon.text_sha256(REPO / e['source']['path'])
        assert canon.REF_UNLOCK_NOTE in e['note'], nid


@pytest.mark.parametrize('nid,key', sorted(canon.REF_TO_OBSERVADO.items()))
def test_referencias_coinciden_con_observados(nid, key):
    assert NUMBERS[nid]['value']['test'] == F5_RES['resultados']['observados'][key]


def test_desbloqueo_se_niega_si_no_coincide_con_observados(tmp_path, monkeypatch):
    doc = json.loads(json.dumps(F5_RES))
    doc['resultados']['observados']['majority'] += 1e-9
    fake = tmp_path / 'resultados.json'
    fake.write_text(json.dumps(doc), encoding='utf-8')
    monkeypatch.setattr(canon, 'F5_RESULT', fake)
    monkeypatch.setattr(canon, 'rel', lambda p: str(p))
    e = json.loads(json.dumps(NUMBERS['ref.piso.majority.f1_macro']))
    e['value']['test'] = None
    e['null_reason'] = {'test': canon.REF_TEST_BLOCKED}
    with pytest.raises(canon.CanonicalError, match='observados'):
        canon.unlock_ref_test({'ref.piso.majority.f1_macro': e}, {})


def test_f5_observados_desde_predicciones():
    """Cada F1 observado de resultados.json se recalcula desde los CSV de sus insumos."""
    obs = F5_RES['resultados']['observados']
    inputs = F5_RES['meta']['inputs']
    for path, h in inputs.items():
        assert canon.sha256(REPO / path) == h, path      # mismos bytes que leyó run_fase5
    seeds = (42, 7, 123, 2024, 99)
    for fase in ('final', 'control_permutacion', 'sensibilidad'):
        for t in TECHS:
            paths = sorted((p for p in inputs if p.startswith(f'results/predictions/{fase}/{t}__')),
                           key=lambda p: seeds.index(int(p.split('__seed')[1].split('.')[0])))
            assert len(paths) == 5
            assert [_f1_csv(p, 'test') for p in paths] == obs[str(('img', fase, t))], (fase, t)
    refs = {'beto': 'text__beto__finetune', 'emb768': 'emb768__logreg__none', 'techo5d': 'umap5d__svm_rbf__none',
            'majority': 'none__majority__none', 'longitud': 'text_length__logreg__none',
            **{str((k, t)): f'{t}__{k}__none' for t in TECHS for k in ('brightness', 'descriptors')}}
    assert set(refs) | {k for k in obs if k.startswith("('img'")} == set(obs)
    for key, stem in refs.items():
        path = f'results/predictions/referencia/{stem}__seed42.csv'
        assert path in inputs
        assert _f1_csv(path, 'test') == obs[key], key
    for t in TECHS:
        r = F5_RES['resultados']['principal']['techniques'][t]
        assert r['f1_img_obs'] == pytest.approx(np.mean(obs[str(('img', 'final', t))]), abs=1e-15)
        assert r['piso_f1_obs']['visual_permutado'] == pytest.approx(
            np.mean(obs[str(('img', 'control_permutacion', t))]), abs=1e-15)


def test_f5_insumos_con_hash():
    for path in ('results/f5_contraste/resultados.json', 'results/f5_contraste/f46_resultados.json'):
        assert DOC['inputs'][path] == canon.text_sha256(REPO / path)


def _at(doc, locator):
    """Lee a.b[0]["r=0.5"] (formato de build._F5Collector._loc)."""
    o = doc
    for quoted, index, plain in re.findall(r'\["([^"]*)"\]|\[(\d+)\]|([^.\[\]"]+)', locator):
        o = o[quoted] if quoted else (o[int(index)] if index else o[plain])
    return o


@pytest.mark.parametrize('prefix,doc', [('f5.', 'f5'), ('f46.', 'f46')])
def test_f5_f46_valores_en_su_localizador(prefix, doc):
    """Cada cifra f5/f46 es exactamente el valor de resultados.json en su localizador."""
    src = {'f5': F5_RES, 'f46': F46_RES}[doc]
    path = {'f5': 'results/f5_contraste/resultados.json', 'f46': 'results/f5_contraste/f46_resultados.json'}[doc]
    keys = [k for k in NUMBERS if k.startswith(prefix)]
    assert keys
    for nid in keys:
        e = NUMBERS[nid]
        assert e['status'] == 'extraido' and e['source']['path'] == path, nid
        v = _at(src, e['source']['locator'])
        if e['kind'] == 'metric':
            assert e['value'] == {'val': None, 'test': v} and e['null_reason']['val'], nid
        elif '.veredicto.' in nid or '.global.' in nid or '.piso.' in nid:
            # identidad en la clave: el último segmento es el valor del localizador
            ident = v[0] if isinstance(v, list) else v
            assert e['value'] == 1 and nid.split('.')[-1] == ident.lower(), nid
        else:
            assert e['value'] == int(v), nid


def test_f5_veredictos_y_conteos():
    R = F5_RES['resultados']
    for t in TECHS:
        v = R['principal']['techniques'][t]['verdict'].lower()
        assert [k for k in NUMBERS if k.startswith(f'f5.principal.{t}.veredicto.')] == [f'f5.principal.{t}.veredicto.{v}']
        assert NUMBERS[f'f5.sensibilidad.por_semilla.n_semillas_v1.{t}']['value'] == \
            R['sensibilidad']['por_semilla']['n_semillas_V1'][t]
        a14 = next(a for a in R['robustez'] if a['alternativa'].startswith('A14'))
        assert NUMBERS[f'f5.robustez.a14.n_semillas_v1.{t}']['value'] == a14['n_semillas_V1'][t]
    g = R['principal']['global'].lower()
    assert [k for k in NUMBERS if k.startswith('f5.principal.global.')] == [f'f5.principal.global.{g}']
    # una identidad de veredicto por técnica y variante
    groups = Counter(k.rsplit('.veredicto.', 1)[0] for k in NUMBERS if '.veredicto.' in k)
    assert set(groups.values()) == {1}


def test_f46_condicion_1():
    for t in TECHS:
        e = NUMBERS[f'f46.contencion.{t}.condicion_1']
        assert e['value'] == int(F46_RES['resultados']['contencion'][t]['condicion_1'])
        assert 'PENDIENTE (P6)' in e['note']


def test_sin_cambios_previos_salvo_desbloqueo():
    """Frente al archivo previo a la Fase 5, solo se añaden f5/f46 y solo cambian las ref.* desbloqueadas."""
    import subprocess
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'{PRE_F5_COMMIT}:results/canonical_numbers.json'],
                       capture_output=True)
    if r.returncode != 0:
        pytest.skip(f'git show {PRE_F5_COMMIT[:7]} no disponible')
    old = json.loads(r.stdout.decode('utf-8'))
    assert set(old['numbers']) <= set(NUMBERS)
    assert {k for k in NUMBERS if k not in old['numbers']} == {k for k in NUMBERS if k.startswith(F5_PREFIXES + F4_PREFIXES)}
    changed = {k for k, e in old['numbers'].items() if NUMBERS[k] != e}
    unlocked = {k for k in changed if k.startswith(REF_PREFIXES)}
    assert changed == unlocked and len(unlocked) == 16
    for k in unlocked:
        o, n = old['numbers'][k], NUMBERS[k]
        assert o['value']['val'] == n['value']['val'] and o['value']['test'] is None
        assert o['confusion']['val'] == n['confusion']['val']
        assert {f for f in set(o) | set(n) if o.get(f) != n.get(f)} <= {'value', 'confusion', 'null_reason', 'source', 'note'}
    assert old.get('superseded') == DOC.get('superseded')
    assert all(DOC['inputs'][k] == h for k, h in old['inputs'].items())


# ── F4.1 (f41.*) y F4.7 (f47.*) ───────────────────────────────────────────
F41_VAL_RES = json.loads((REPO / 'results' / 'f41_recuperabilidad' / 'val' / 'resultados_val.json').read_text(encoding='utf-8'))
F41_TEST_RES = json.loads((REPO / 'results' / 'f41_recuperabilidad' / 'prueba' / 'resultados_prueba.json').read_text(encoding='utf-8'))
F47_RES = json.loads((REPO / 'results' / 'f47_geometria' / 'resultados.json').read_text(encoding='utf-8'))
F41_PATHS = {'results/f41_recuperabilidad/val/resultados_val.json': (F41_VAL_RES, 'val'),
             'results/f41_recuperabilidad/prueba/resultados_prueba.json': (F41_TEST_RES, 'test'),
             'results/f47_geometria/resultados.json': (F47_RES, 'train'),
             'results/f42_regimen/resultados.json': (json.loads((REPO / 'results' / 'f42_regimen' / 'resultados.json')
                                                                .read_text(encoding='utf-8')), 'train')}
F41_PRED_PATH = 'results/f41_recuperabilidad/prueba/predicciones_prueba.csv'


def test_f41_f47_presencia():
    for t in TECHS:
        for b in F41_VAL_RES['resultados']['tecnicas'][t]:
            for split in ('val', 'prueba'):
                base = f'f41.{split}.{t}.{b}'
                assert all(f'{base}.r2.u{i}' in NUMBERS for i in range(5)), base
                assert f'{base}.logreg_u_hat.f1_macro' in NUMBERS
            assert f'f41.val.{t}.{b}.alpha' in NUMBERS
            for k in ('delta_f1', 'delta_f1_ic_inf', 'delta_f1_ic_sup', 'delta_f1.nivel', 'delta_f1.ic_contiene_0'):
                assert f'f41.prueba.{t}.{b}.{k}' in NUMBERS, k
        assert f'f41.prueba.{t}.condicion_2_f46' in NUMBERS
        for k in ('rho_spearman', 'p', 'k', 'n_perm', 'n', 'holm.alpha_h', 'holm.rango', 'holm.rechaza'):
            assert f'f47.{t}.{k}' in NUMBERS, k
    assert 'f41.val.u.logreg.f1_macro' in NUMBERS and 'f41.prueba.u.logreg.f1_macro' in NUMBERS


def test_f41_f47_valores_en_su_localizador():
    """Cada cifra extraída es exactamente el valor del JSON en su localizador, en su split."""
    keys = [k for k in NUMBERS if k.startswith(('f41.', 'f47.', 'f42.')) and NUMBERS[k]['status'] == 'extraido']
    assert keys
    for nid in keys:
        e = NUMBERS[nid]
        src, split = F41_PATHS[e['source']['path']]
        v = _at(src, e['source']['locator'])
        if e['kind'] == 'metric':
            assert e['value'][split] == v, nid
            assert all(e['value'][s] is None and e['null_reason'][s] for s in ('val', 'test') if s != split), nid
        else:
            assert e['value'] == int(v), nid


def test_f41_f1_prueba_desde_predicciones():
    """Los F1 de prueba (û y u) son verificados: se recalculan desde predicciones_prueba.csv."""
    with open(REPO / F41_PRED_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert DOC['inputs'][F41_PRED_PATH] == canon.text_sha256(REPO / F41_PRED_PATH)
    f1 = lambda rr, col: canon.f1_macro_from_confusion(canon.confusion_counts(  # noqa: E731
        [int(r['y_true']) for r in rr], [int(float(r[col]) > 0.5) for r in rr]))
    for t in TECHS:
        for b, x in F41_TEST_RES['resultados']['tecnicas'][t].items():
            if not isinstance(x, dict):
                continue
            rr = [r for r in rows if r['tecnica'] == t and r['backbone'] == b]
            assert [r['sample_id'] for r in rr] == [f'test_{i:04d}' for i in range(114)]
            e = NUMBERS[f'f41.prueba.{t}.{b}.logreg_u_hat.f1_macro']
            assert e['status'] == 'verificado' and e['value']['test'] == f1(rr, 'p_dep_u_hat') == x['f1_u_hat']
            assert f1(rr, 'p_dep_u') == x['f1_u'] == NUMBERS['f41.prueba.u.logreg.f1_macro']['value']['test']


def test_f41_u_coincide_con_f21():
    ref = NUMBERS['ref.techo5d.logreg.f1_macro']['value']
    assert NUMBERS['f41.val.u.logreg.f1_macro']['value']['val'] == ref['val']
    assert NUMBERS['f41.prueba.u.logreg.f1_macro']['value']['test'] == ref['test']


def test_f41_condicion_2_y_contiene_0():
    for t in TECHS:
        T = F41_TEST_RES['resultados']['tecnicas'][t]
        prin = canon.PRINCIPAL_F41[t]
        assert T[prin]['principal'] is True
        assert NUMBERS[f'f41.prueba.{t}.condicion_2_f46']['value'] == int(T['condicion_2_F46']) == int(T[prin]['contiene_0'])
        for b, x in T.items():
            if isinstance(x, dict):
                lo, hi = x['ic']
                assert x['contiene_0'] == (lo <= 0 <= hi)
                assert x['delta_f1_obs'] == pytest.approx(x['f1_u'] - x['f1_u_hat'], abs=1e-15)


def test_f47_solo_train_y_holm():
    from src.protocol import verdict
    P = F47_RES['resultados']['principal']
    assert set(P) == set(TECHS) and F47_RES['resultados']['sensibilidad'] == {}
    holm = verdict.holm({t: P[t]['p'] for t in TECHS}, 0.05)
    for t in TECHS:
        assert P[t]['holm'] == holm[t]
        assert P[t]['p'] == (P[t]['k'] + 1) / (P[t]['n_perm'] + 1) and P[t]['n'] == 908
        e = NUMBERS[f'f47.{t}.rho_spearman']
        assert e['value']['train'] == P[t]['rho_spearman'] and e['value']['val'] is None and e['value']['test'] is None


# ── F4.2 (f42.*) ──────────────────────────────────────────────────────────
F42_RES = json.loads((REPO / 'results' / 'f42_regimen' / 'resultados.json').read_text(encoding='utf-8'))
F42_PRUEBAS = {'lorenz_fisher_regimen': 'lorenz.fisher_regimen', 'lorenz_mw_rho': 'lorenz.mw_rho',
               'perlin_mw_octavas': 'perlin.mw_octavas', 'gray_scott_fisher_reintentos': 'gray_scott.fisher_reintentos'}


def test_f42_presencia_y_split():
    for b in F42_PRUEBAS.values():
        for k in ('p', 'p_holm', 'holm.alpha_h', 'holm.rango', 'holm.rechaza'):
            assert f'f42.{b}.{k}' in NUMBERS, (b, k)
        e = NUMBERS[f'f42.{b}.p']
        assert e['value']['val'] is None and e['value']['test'] is None and e['value']['train'] is not None
    for c in ('dep', 'nodep'):
        for k in (f'lorenz.rho_menor_24_74.proporcion.{c}', f'gray_scott.reintentos.tasa.{c}'):
            assert f'f42.{k}' in NUMBERS, k
    for est in ('rho_menor_24_74', 'rho_mayor_igual_24_74'):
        e = NUMBERS[f'f42.lorenz.estrato_prueba.{est}.f1_macro']
        assert e['status'] == 'verificado' and e['value']['val'] is None and 'train' not in e['value']
        assert e['supports_ref'] == {'test': f'f42.lorenz.estrato_prueba.{est}.support'}


def test_f42_holm_y_p_ajustado():
    """El rechazo es el de verdict.holm (m = 4) y p_holm es el ajustado de Holm (paso descendente)."""
    import math
    from src.protocol import verdict
    R = F42_RES['resultados']
    p = {k: R['pruebas'][k]['p'] for k in F42_PRUEBAS}
    holm = verdict.holm(p, 0.05)
    order = sorted(p, key=p.get)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    for k, b in F42_PRUEBAS.items():
        h = R['holm']['familia'][k]
        assert h['reject'] == holm[k]['reject'] and h['rank'] == holm[k]['rank'], k
        assert math.isclose(h['p_holm'], adj[k], rel_tol=1e-12, abs_tol=0), k
        assert h['reject'] == (h['p_holm'] <= 0.05), k
        assert NUMBERS[f'f42.{b}.p_holm']['value']['train'] == h['p_holm']
        assert NUMBERS[f'f42.{b}.holm.rechaza']['value'] == int(h['reject'])


def test_f42_proporciones_y_tasas_cuadran_con_conteos():
    n = {c: NUMBERS[f'f42.n_train.{c}']['value'] for c in ('dep', 'nodep')}
    assert n['dep'] + n['nodep'] == NUMBERS['f42.n_train.total']['value'] == 908
    for c in ('dep', 'nodep'):
        assert NUMBERS[f'f42.lorenz.rho_menor_24_74.proporcion.{c}']['value']['train'] == \
            NUMBERS[f'f42.lorenz.rho_menor_24_74.n.{c}']['value'] / n[c]
        assert NUMBERS[f'f42.gray_scott.reintentos.tasa.{c}']['value']['train'] == \
            NUMBERS[f'f42.gray_scott.reintentos.n.{c}']['value'] / n[c]
    T = F42_RES['resultados']['pruebas']
    nd, nn = (NUMBERS[f'f42.lorenz.rho_menor_24_74.n.{c}']['value'] for c in ('dep', 'nodep'))
    assert T['lorenz_fisher_regimen']['tabla'] == [[nd, n['dep'] - nd], [nn, n['nodep'] - nn]]
    rd, rn = (NUMBERS[f'f42.gray_scott.reintentos.n.{c}']['value'] for c in ('dep', 'nodep'))
    assert T['gray_scott_fisher_reintentos']['tabla'] == [[rd, n['dep'] - rd], [rn, n['nodep'] - rn]]


def test_f42_reintentos_desde_el_registro():
    """Los conteos de reintento por clase salen del CSV de Kaggle, cuyo sha256 está en el meta."""
    import hashlib
    path = REPO / 'results' / 'f42_regimen' / 'reintentos' / 'reintentos_gs_train.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        F42_RES['meta']['inputs']['results/f42_regimen/reintentos/reintentos_gs_train.csv']
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for c, lab in (('dep', '1'), ('nodep', '0')):
        assert NUMBERS[f'f42.gray_scott.reintentos.n.{c}']['value'] == sum(
            int(r['reintentos']) > 0 for r in rows if r['y_true'] == lab)
    assert NUMBERS['f42.gray_scott.bit_a_bit.n_identicas']['value'] == sum(r['pixeles_coinciden'] == 'True' for r in rows)


def test_f42_estratos_suman_prueba():
    e = {s: {k: NUMBERS[f'f42.lorenz.estrato_prueba.{s}.{k}']['value'] for k in ('n', 'support.label0', 'support.label1')}
         for s in ('rho_menor_24_74', 'rho_mayor_igual_24_74')}
    assert sum(x['n'] for x in e.values()) == 114
    for lab in ('label0', 'label1'):
        assert sum(x[f'support.{lab}'] for x in e.values()) == NUMBERS[f'support.test.{lab}']['value']
    for x in e.values():
        assert x['n'] == x['support.label0'] + x['support.label1']


def test_f42_resoluciones_versionadas():
    import hashlib
    path = REPO / 'results' / 'f42_regimen' / 'resoluciones.json'
    m = F42_RES['meta']['resoluciones']
    assert m['archivo'] == 'results/f42_regimen/resoluciones.json' and m['autor'] == 'SV'
    raw = path.read_bytes()
    assert m['sha256'] in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
    res = json.loads(raw.decode('utf-8'))
    assert F42_RES['resultados']['resoluciones'] == {k: v for k, v in res.items() if not k.startswith('_')}


# ── F4.3 (f43.*) ──────────────────────────────────────────────────────────
F43_DIR = REPO / 'results' / 'f43_atribucion'
F43_RES = json.loads((F43_DIR / 'local' / 'resultados.json').read_text(encoding='utf-8'))
F43_CAS = json.loads((F43_DIR / 'gpu' / 'cascada.json').read_text(encoding='utf-8'))
F43_ABL = json.loads((F43_DIR / 'gpu' / 'ablacion.json').read_text(encoding='utf-8'))
F43_TECHS = ('gray_scott', 'perlin', 'lorenz')
F43_SEEDS = ('42', '7', '123', '2024', '99')


def _f43_ident(prefix):
    """Sufijo de la única clave de identidad bajo prefix (p. ej. el veredicto)."""
    ks = [k[len(prefix) + 1:] for k in NUMBERS if k.startswith(prefix + '.')]
    assert len(ks) == 1, (prefix, ks)
    assert NUMBERS[f'{prefix}.{ks[0]}']['value'] == 1
    return ks[0]


def test_f43_split_val_y_replica_de_prueba():
    """Todo f43.* es de val; solo la fracción de constantes lleva además la réplica de prueba."""
    ks = [k for k in NUMBERS if k.startswith('f43.')]
    assert len(ks) == 207
    for k in ks:
        e = NUMBERS[k]
        if e['kind'] != 'metric':
            continue
        assert e['value']['val'] is not None and 'train' not in e['value'], k
        if '.constantes.' in k:
            assert e['value']['test'] is not None and 'null_reason' not in e, k
        else:
            assert e['value']['test'] is None and e['null_reason']['test'].strip(), k


def test_f43_holm_y_p_de_permutacion():
    import math
    from src.protocol import verdict
    R = F43_RES['resultados']
    p = {t: R['tecnicas'][t]['permutacion_val']['p'] for t in F43_TECHS}
    holm = verdict.holm(p, 0.05)
    order = sorted(p, key=p.get)
    adj, run = {}, 0.0
    for i, t in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[t]))
        adj[t] = run
    assert NUMBERS['f43.holm.m']['value'] == 3 and NUMBERS['f43.holm.alpha']['value']['val'] == 0.05
    for t in F43_TECHS:
        b = f'f43.{t}.permutacion'
        k, B = NUMBERS[f'{b}.k']['value'], NUMBERS[f'{b}.b']['value']
        assert B == 5000 and NUMBERS[f'{b}.p']['value']['val'] == (k + 1) / (B + 1), t
        assert math.isclose(NUMBERS[f'{b}.p_holm']['value']['val'], adj[t], rel_tol=1e-12, abs_tol=0), t
        assert NUMBERS[f'{b}.holm.rechaza']['value'] == int(holm[t]['reject']) == int(adj[t] <= 0.05), t
        assert NUMBERS[f'{b}.holm.rango']['value'] == holm[t]['rank'], t


def test_f43_veredictos_de_cordura_y_criterio_relativo():
    """El veredicto global es la mayoría de las semillas y cada semilla «cambia» si
    su mediana queda por debajo del mínimo de los 10 pares reales."""
    for t in F43_TECHS:
        for k in (('map', 'backbone_map') if t != 'gray_scott' else ('map',)):
            b = f'f43.{t}.cordura_datos.{k}'
            lo = NUMBERS[f'{b}.spearman56.referencia_min']['value']['val']
            ps = ['cambia' if NUMBERS[f'{b}.spearman56.seed{s}']['value']['val'] < lo else 'no_cambia' for s in F43_SEEDS]
            maj = max(set(ps), key=ps.count)
            assert ps.count(maj) > 2 and _f43_ident(f'{b}.veredicto') == maj, (t, k, ps)
            ref = F43_RES['resultados']['tecnicas'][t]['cordura_datos'][k]['val']['spearman56']['referencia_pares']
            assert len(ref) == 10 and min(ref.values()) == lo
            g = F43_CAS['tecnicas'][t]['pasos']['cabeza'][k]['spearman56']['global']
            assert _f43_ident(f'f43.{t}.cascada.cabeza.{k}.veredicto') == g
            orden = F43_CAS['tecnicas'][t]['orden']
            assert NUMBERS[f'f43.{t}.cascada.{k}.n_pasos']['value'] == len(orden)
            assert NUMBERS[f'f43.{t}.cascada.{k}.n_pasos_cambia']['value'] == sum(
                F43_CAS['tecnicas'][t]['pasos'][p][k]['spearman56']['global'] == 'cambia' for p in orden)


def test_f43_cosenos_y_lectura_vit():
    from src.fase4 import atribucion_comun as ac
    res = F43_RES['resultados']['resoluciones']
    for t in ('perlin', 'lorenz'):
        niveles = {}
        for v in ('w1', 'w1_menos_w0'):
            b = f'f43.{t}.coseno.{v}'
            lo = NUMBERS[f'{b}.referencia_min']['value']['val']
            ps = ['bajo' if NUMBERS[f'{b}.seed{s}']['value']['val'] < lo else 'alto' for s in F43_SEEDS]
            niveles[v] = _f43_ident(f'{b}.nivel')
            assert ps.count(niveles[v]) > 2, (t, v, ps)
        cd = _f43_ident(f'f43.{t}.cordura_datos.map.veredicto')
        cab = _f43_ident(f'f43.{t}.cascada.cabeza.map.veredicto')
        assert _f43_ident(f'f43.{t}.lectura_vit') == ac.lectura_vit(cd, niveles[res['coseno_lectura']], cab)


def test_f43_ablacion_delta_f1():
    import math
    for t in F43_TECHS:
        x = F43_ABL['tecnicas'][t]
        f0 = NUMBERS[f'f43.{t}.ablacion.original.f1_macro_media']['value']['val']
        assert math.isclose(f0, sum(x['original']['f1_por_semilla'].values()) / 5, rel_tol=0, abs_tol=1e-12)
        for banda in ('paso_bajo', 'paso_alto'):
            for sg in (1, 2, 4, 8):
                d = NUMBERS[f'f43.{t}.ablacion.{banda}.s{sg}.delta_f1']['value']['val']
                c = x[f'{banda}_s{sg}']
                assert math.isclose(d, c['f1_media'] - f0, rel_tol=0, abs_tol=1e-12), (t, banda, sg)
                assert math.isclose(d, sum(c['delta_f1_por_semilla'].values()) / 5, rel_tol=0, abs_tol=1e-12)


def test_f43_fraccion_de_constantes_cuadra_con_conteos():
    for t in F43_TECHS:
        for clase in ('real', 'predicha'):
            tot = {'val': 0, 'prueba': 0}
            for y in ('0', '1'):
                b = f'f43.{t}.constantes.clase_{clase}.y{y}'
                for sp, s in (('val', 'val'), ('prueba', 'test')):
                    n, k = NUMBERS[f'{b}.{sp}.n_mapas']['value'], NUMBERS[f'{b}.{sp}.n_constantes']['value']
                    assert NUMBERS[f'{b}.fraccion']['value'][s] == k / n, (b, sp)
                    tot[sp] += n
            assert tot == {'val': 5 * 114, 'prueba': 5 * 114}, (t, clase)


def test_f43_insumos_encadenados():
    """La parte local leyó la cascada versionada y ambas partes usan las resoluciones versionadas."""
    import hashlib
    meta = F43_RES['meta']
    cas = F43_DIR / 'gpu' / 'cascada.json'
    assert meta['cascada']['sha256'] == hashlib.sha256(cas.read_bytes()).hexdigest()
    res = (F43_DIR / 'resoluciones.json').read_bytes()
    shas = (hashlib.sha256(res).hexdigest(), hashlib.sha256(res.replace(b'\r\n', b'\n')).hexdigest())
    gpu = json.loads((F43_DIR / 'gpu' / 'atribucion_gpu_meta.json').read_text(encoding='utf-8'))
    assert meta['resoluciones']['sha256'] in shas and gpu['resoluciones']['sha256'] in shas
    assert not F43_CAS['meta']['smoke'] and not F43_ABL['meta']['smoke'] and gpu['split'] == 'val'


def test_f43_valores_en_su_localizador():
    """Cada cifra f43 es el valor del JSON en su localizador (val; la réplica de prueba
    de las constantes en el suyo). Las identidades nombran el valor del localizador; los
    conteos derivados (pasos, transiciones) se prueban en los tests de arriba."""
    docs = {'results/f43_atribucion/local/resultados.json': F43_RES,
            'results/f43_atribucion/gpu/cascada.json': F43_CAS,
            'results/f43_atribucion/gpu/ablacion.json': F43_ABL}
    derivados = ('.n_pasos', '.n_pasos_cambia', '.n_transiciones_a_constante')
    n = 0
    for nid, e in NUMBERS.items():
        if not nid.startswith('f43.'):
            continue
        assert e['status'] == 'extraido', nid
        src, loc = docs[e['source']['path']], e['source']['locator']
        if e['kind'] == 'metric':
            lv, _, lt = loc.partition('; test: ')
            assert e['value']['val'] == _at(src, lv), nid
            if lt:
                assert e['value']['test'] == _at(src, lt), nid
        elif 'identidad en la clave' in e.get('note', ''):
            assert e['value'] == 1 and nid.endswith('.' + str(_at(src, loc))), nid
        elif nid.endswith(derivados):
            continue
        else:
            assert e['value'] == int(_at(src, loc)), nid
        n += 1
    assert n > 150


# ── F4.4 (f44.*) ──────────────────────────────────────────────────────────
F44_RES = json.loads((REPO / 'results' / 'f44_convergencia' / 'resultados.json').read_text(encoding='utf-8'))
F44_DESC = {'gray_scott': ('shannon_entropy', 'contrast', 'edge_density', 'lbp_energy'),
            'perlin': ('contrast', 'edge_density'), 'lorenz': ('shannon_entropy', 'lbp_energy')}


def test_f44_presencia_split_y_localizador():
    ks = [k for k in NUMBERS if k.startswith('f44.')]
    assert len(ks) == 8 * 7
    for nid in ks:
        e = NUMBERS[nid]
        assert e['source']['path'] == 'results/f44_convergencia/resultados.json'
        v = _at(F44_RES, e['source']['locator'])
        if e['kind'] == 'metric':
            assert e['value']['val'] == v and e['value']['test'] is None and 'train' not in e['value'], nid
        else:
            assert e['value'] == int(v), nid


def test_f44_recalculo_desde_por_muestra():
    """Media, IC (bootstrap con default_rng(2026), cuantiles linear) y n se recalculan desde spearman_por_muestra.csv."""
    import math
    from src.fase4 import convergencia as cv
    with open(REPO / 'results' / 'f44_convergencia' / 'spearman_por_muestra.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for t, ds in F44_DESC.items():
        idx = cv.indices_bootstrap(114, np.random.default_rng(2026))
        for d in ds:
            v = np.array([float(r['spearman']) if r['spearman'] else np.nan for r in rows
                          if r['tecnica'] == t and r['mapa'] == 'map' and r['descriptor'] == d])
            assert len(v) == 114
            r = cv.ic_bootstrap(v, idx, metodo='linear')
            b = f'f44.{t}.{d}'
            assert math.isclose(NUMBERS[f'{b}.spearman_media']['value']['val'], r['media'], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(NUMBERS[f'{b}.spearman_media_ic_inf']['value']['val'], r['ic_inf'], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(NUMBERS[f'{b}.spearman_media_ic_sup']['value']['val'], r['ic_sup'], rel_tol=0, abs_tol=1e-12)
            assert NUMBERS[f'{b}.n']['value'] + NUMBERS[f'{b}.n_indefinidos']['value'] == 114
            assert NUMBERS[f'{b}.ic_contiene_0']['value'] == int(r['contiene_0'])


def test_f44_resoluciones_versionadas():
    import hashlib
    raw = (REPO / 'results' / 'f44_convergencia' / 'resoluciones.json').read_bytes()
    m = F44_RES['meta']['resoluciones']
    assert m['sha256'] in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
    assert F44_RES['meta']['limit'] is None and len(F44_RES['meta']['git_commit']) == 40
