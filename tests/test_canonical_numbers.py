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
                 verificado, al redondeo de sus decimales si es transcrito.
Confusión        El F1 recalculado de cada matriz es el valor guardado y la matriz
                 suma los soportes del split.
Estructura       908 + 114 + 114 = 1,136; soportes por clase coherentes; 54 corridas
                 de F3.2 con el mismo y_true de val.
Derivadas        Cada derivada se recalcula desde depends_on; cada agregado
                 transcrito (summarizes) es compatible con sus cifras.
Fusión           Una clave repetida entre fragmentos o colectores es un error.

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


def is_leaf_metric(entry):
    return entry['kind'] == 'metric' and 'depends_on' not in entry and 'summarizes' not in entry


def leaf_cases():
    """(id, split) de cada F1 no derivado y no nulo; xfail si la entrada lo declara."""
    cases = []
    for nid, e in sorted(NUMBERS.items()):
        if not is_leaf_metric(e):
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
            assert set(e['value']) == {'val', 'test'}, nid
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


def test_f32_estructura():
    with open(canon.F32_DIR / 'predictions.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    y_by_run = defaultdict(list)
    for r in rows:
        y_by_run[r['run_id']].append(int(r['y_true']))
    assert len(y_by_run) == 54
    assert len({tuple(y) for y in y_by_run.values()}) == 1, 'y_true distinto entre corridas'
    y = next(iter(y_by_run.values()))
    assert Counter(y) == {0: NUMBERS['support.val.label0']['value'], 1: NUMBERS['support.val.label1']['value']}
    seeds = [k for k in NUMBERS if k.startswith('f32.') and '.seed' in k]
    means = [k for k in NUMBERS if k.startswith('f32.') and k.endswith('.f1_macro_mean')]
    assert len(seeds) == 54 and len(means) == 18
    assert all(NUMBERS[k]['value']['test'] is None for k in seeds + means), 'F3.2 no debe tener test'


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
        expected = None if all(v is None for v in vals) else canon.derive(e['derivation'], vals)
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
