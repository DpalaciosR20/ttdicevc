"""
tests/test_verdict.py
═════════════════════
F1.2 — Regla de decisión del Objetivo V (src/protocol/verdict.py,
docs/protocolo_preregistrado.md §5.2–§5.5) con réplicas sintéticas
deterministas. Cubre V1–V4, G1–G4, los niveles de Holm (α/3, α/2, α), la
parada escalonada, los empates de p (incluidos tres p = 0) y la validación de
entradas. No usa resultados reales.

    python -m pytest tests/test_verdict.py -v
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.protocol import verdict as vd  # noqa: E402

B = 1000
ALPHA = 0.05
TECHS = ('perlin', 'gray_scott', 'lorenz')


def reps(n_nonpos, neg=-0.01, pos=0.01, b=B):
    """Réplicas con exactamente n_nonpos valores ≤ 0 → p = n_nonpos / b."""
    return np.array([neg] * n_nonpos + [pos] * (b - n_nonpos))


def run(theta, d, **kw):
    return vd.verdicts(dict(zip(TECHS, theta)), dict(zip(TECHS, d)), **kw)


def verdict_of(res):
    return tuple(res['techniques'][t]['verdict'] for t in TECHS)


SUP = reps(0)            # supera el piso: p = 0
NO_SUP = reps(500)       # no supera el piso: p = 0.5
NI = reps(0)             # se rechaza H₀ de no inferioridad
LOSS = reps(B, neg=-0.05)  # θ claramente negativo: p = 1, límite superior < 0
INCONCLUSIVE = reps(300)   # p = 0.3, intervalo que cruza 0


# ── Configuración ────────────────────────────────────────────────────────────

def test_contrast_config_is_implemented_rule():
    c = vd.load_contrast()
    assert c['multiplicidad']['alpha'] == ALPHA and c['bootstrap']['B'] == 10000


def test_config_mismatch_raises(tmp_path):
    cfg = json.loads(vd.PROTOCOL_PATH.read_text(encoding='utf-8'))
    bad = copy.deepcopy(cfg)
    bad['contraste']['multiplicidad']['ties'] = 'fixed_order'
    path = tmp_path / 'protocolo.json'
    path.write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(vd.VerdictInputError):
        vd.load_contrast(path)


# ── p y percentiles ──────────────────────────────────────────────────────────

def test_p_value_counts_le_zero():
    assert vd.p_value(reps(37)) == (0.037, False)
    assert vd.p_value([0.0, 1.0, 2.0, 3.0]) == (0.25, False)  # 0 cuenta como ≤ 0
    assert vd.p_value(reps(0)) == (0.0, True)


def test_percentile_interval_level():
    r = np.arange(1001, dtype=float)
    assert vd.percentile_interval(r, 0.05) == (50.0, 950.0)
    assert vd.percentile_interval(r, ALPHA / 3) == pytest.approx((1000 * ALPHA / 3, 1000 * (1 - ALPHA / 3)))


# ── Holm ─────────────────────────────────────────────────────────────────────

def test_holm_levels_by_rank():
    h = vd.holm({'a': 0.001, 'b': 0.02, 'c': 0.04}, ALPHA)
    assert [h[k]['rank'] for k in 'abc'] == [1, 2, 3]
    assert [h[k]['alpha_h'] for k in 'abc'] == pytest.approx([ALPHA / 3, ALPHA / 2, ALPHA])
    assert all(h[k]['reject'] for k in 'abc')


def test_holm_order_independent_of_input_order():
    h = vd.holm({'c': 0.04, 'a': 0.001, 'b': 0.02}, ALPHA)
    assert (h['a']['rank'], h['b']['rank'], h['c']['rank']) == (1, 2, 3)


def test_holm_step_down_stops():
    """Rango 2 falla (0.03 > α/2): el rango 3 no se rechaza aunque p ≤ α."""
    h = vd.holm({'a': 0.001, 'b': 0.03, 'c': 0.04}, ALPHA)
    assert [h[k]['reject'] for k in 'abc'] == [True, False, False]
    assert h['c']['alpha_h'] == ALPHA and 0.04 <= h['c']['alpha_h']


def test_holm_first_rank_fails():
    h = vd.holm({'a': 0.02, 'b': 0.021, 'c': 0.022}, ALPHA)
    assert not any(h[k]['reject'] for k in 'abc')


def test_holm_three_zero_p_ties():
    h = vd.holm({'a': 0.0, 'b': 0.0, 'c': 0.0}, ALPHA)
    assert all(h[k]['rank'] == 1 and h[k]['alpha_h'] == pytest.approx(ALPHA / 3) and h[k]['reject'] for k in 'abc')


def test_holm_partial_tie_gets_strictest_level():
    h = vd.holm({'a': 0.01, 'b': 0.01, 'c': 0.04}, ALPHA)
    assert (h['a']['rank'], h['b']['rank'], h['c']['rank']) == (1, 1, 3)
    assert h['a']['alpha_h'] == h['b']['alpha_h'] == pytest.approx(ALPHA / 3) and h['c']['alpha_h'] == ALPHA
    assert all(h[k]['reject'] for k in 'abc')


def test_holm_tie_rejection_unaffected():
    """Empate en el rango 2-3: el rechazo es el del procedimiento sin empates."""
    h = vd.holm({'a': 0.001, 'b': 0.03, 'c': 0.03}, ALPHA)
    assert (h['b']['rank'], h['c']['rank']) == (2, 2) and h['b']['alpha_h'] == ALPHA / 2
    assert [h[k]['reject'] for k in 'abc'] == [True, False, False]


# ── Veredictos por técnica (V1–V4) ───────────────────────────────────────────

def test_v1_v2_v3_v4():
    res = run(theta=[NI, LOSS, INCONCLUSIVE], d=[SUP, SUP, SUP])
    assert verdict_of(res) == ('V1', 'V2', 'V4')
    res = run(theta=[NI, NI, NI], d=[NO_SUP, SUP, SUP])
    assert verdict_of(res)[0] == 'V3'


def test_v3_takes_precedence_over_v1():
    """Sin superar el piso es V3 aunque se rechace la no inferioridad (orden de §5.5)."""
    res = run(theta=[NI, NI, NI], d=[NO_SUP, NO_SUP, NO_SUP])
    assert verdict_of(res) == ('V3', 'V3', 'V3')
    assert all(r['no_inferioridad']['reject'] for r in res['techniques'].values())


def test_floor_family_uses_holm():
    """p = 0.03 en superioridad: pasa α sin corrección, pero no en el rango 2 de Holm (α/2)."""
    res = run(theta=[NI, NI, NI], d=[reps(0), reps(30), reps(40)])
    assert verdict_of(res) == ('V1', 'V3', 'V3')


def test_v2_depends_on_holm_level():
    """θ con 970/1000 réplicas ≤ 0: el percentil 95 es negativo y el 98.3 positivo.
    Con α_h = α (rango 3) es V2; con α_h = α/3 (rango 1) es V4."""
    tight = reps(970)
    res = run(theta=[NI, NI, tight], d=[SUP, SUP, SUP])          # rango 3 en NI
    r = res['techniques']['lorenz']
    assert r['no_inferioridad']['rank'] == 3 and r['no_inferioridad']['alpha_h'] == ALPHA
    assert r['interval_level'] == pytest.approx(0.90) and r['theta_interval'][1] < 0 and r['verdict'] == 'V2'

    res = run(theta=[LOSS, LOSS, tight], d=[SUP, SUP, SUP])      # rango 1 en NI (p = 0.97 < 1)
    r = res['techniques']['lorenz']
    assert r['no_inferioridad']['rank'] == 1 and r['no_inferioridad']['alpha_h'] == pytest.approx(ALPHA / 3)
    assert r['theta_interval'][1] > 0 and r['verdict'] == 'V4'


def test_step_down_failure_gives_v4():
    """Rango 3 con p ≤ α pero no rechazado por la parada de Holm: no es V1."""
    res = run(theta=[reps(1), reps(30), reps(40)], d=[SUP, SUP, SUP])
    assert verdict_of(res) == ('V1', 'V4', 'V4')


def test_three_zero_p_in_both_families():
    res = run(theta=[NI, NI, NI], d=[SUP, SUP, SUP])
    for r in res['techniques'].values():
        for fam in ('superioridad_piso', 'no_inferioridad'):
            assert r[fam]['p'] == 0 and r[fam]['below_resolution'] and r[fam]['alpha_h'] == pytest.approx(ALPHA / 3)
        assert r['interval_level'] == pytest.approx(1 - 2 * ALPHA / 3)
    assert verdict_of(res) == ('V1', 'V1', 'V1') and res['global'] == 'G1'


# ── Veredicto global (G1–G4) ─────────────────────────────────────────────────

@pytest.mark.parametrize('theta, d, expected, g', [
    ([NI, LOSS, INCONCLUSIVE], [SUP, SUP, SUP], ('V1', 'V2', 'V4'), 'G1'),
    ([NI, LOSS, INCONCLUSIVE], [NO_SUP, SUP, SUP], ('V3', 'V2', 'V4'), 'G2'),
    ([NI, NI, NI], [NO_SUP, NO_SUP, NO_SUP], ('V3', 'V3', 'V3'), 'G3'),
    ([INCONCLUSIVE, INCONCLUSIVE, NI], [SUP, SUP, NO_SUP], ('V4', 'V4', 'V3'), 'G4'),
])
def test_global_verdicts(theta, d, expected, g):
    res = run(theta, d)
    assert verdict_of(res) == expected and res['global'] == g


def test_global_verdict_function():
    assert vd.global_verdict(['V4', 'V1', 'V2']) == 'G1'
    assert vd.global_verdict(['V4', 'V3', 'V2']) == 'G2'
    assert vd.global_verdict(['V3', 'V3', 'V3']) == 'G3'
    assert vd.global_verdict(['V3', 'V4', 'V4']) == 'G4'
    with pytest.raises(vd.VerdictInputError):
        vd.global_verdict(['V5'])


# ── Entradas ─────────────────────────────────────────────────────────────────

def test_different_techniques_raise():
    with pytest.raises(vd.VerdictInputError, match='técnicas distintas'):
        vd.verdicts({'perlin': NI}, {'lorenz': SUP})


def test_expected_techniques_enforced():
    with pytest.raises(vd.VerdictInputError, match='se esperan las técnicas'):
        vd.verdicts({'perlin': NI}, {'perlin': SUP}, techniques=TECHS)


def test_different_B_raises():
    with pytest.raises(vd.VerdictInputError, match='número de réplicas'):
        run(theta=[NI, NI, reps(0, b=999)], d=[SUP, SUP, SUP])
    with pytest.raises(vd.VerdictInputError, match='B = 10000'):
        run(theta=[NI, NI, NI], d=[SUP, SUP, SUP], B=10000)


def test_non_finite_raises():
    bad = NI.copy()
    bad[0] = np.nan
    with pytest.raises(vd.VerdictInputError, match='no finitas'):
        run(theta=[bad, NI, NI], d=[SUP, SUP, SUP])


def test_alpha_from_config_by_default():
    res = run(theta=[NI, NI, NI], d=[SUP, SUP, SUP])
    assert res['alpha'] == ALPHA and res['B'] == B
