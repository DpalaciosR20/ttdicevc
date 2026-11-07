"""Fase 5 — contraste, piso, descriptivos, McNemar/Cochran, kappa y sensibilidad (datos sintéticos)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import fase5_sintetico as S  # noqa: E402
from src.protocol import bootstrap as bs  # noqa: E402
from src.protocol import contraste as cx  # noqa: E402
from src.protocol import run_fase5 as rf  # noqa: E402
from src.protocol import sensibilidad as sens  # noqa: E402

CFG = S.cfg()
IDX = S.idx()
W = bs.weights(IDX, S.N)


def table(ins):
    return cx.f1_table(W, ins.y, rf.predicciones(ins, CFG))


def verdicts(res):
    return {t: r['verdict'] for t, r in res['techniques'].items()}


# ── Casos de respuesta conocida ───────────────────────────────────────────
def test_clasificador_perfecto_V1():
    res = cx.contraste(table(S.make(img_errors={t: 0 for t in S.TECHS})), CFG)
    assert verdicts(res) == {t: 'V1' for t in S.TECHS} and res['global'] == 'G1'


def test_identico_a_beto_V1_si_beto_supera_el_umbral():
    ins = S.make()
    for t in S.TECHS:
        ins.visual[('final', t)]['p'] = np.column_stack([ins.refs['beto']] * 5)
    F = table(ins)
    assert F['beto']['obs'] > CFG['contraste']['f1_azar']
    res = cx.contraste(F, CFG)
    r = CFG['contraste']['r']
    for t in S.TECHS:
        # θ = (1 − r)(F1_BETO − 0.5) > 0 en el observado y en cada réplica con F1_BETO* > 0.5
        assert res['techniques'][t]['theta_obs'] == pytest.approx((1 - r) * (F['beto']['obs'] - 0.5), abs=1e-15)
    assert verdicts(res) == {t: 'V1' for t in S.TECHS}


def test_azar_V3():
    res = cx.contraste(table(S.make(img_errors={t: None for t in S.TECHS})), CFG)
    assert verdicts(res) == {t: 'V3' for t in S.TECHS} and res['global'] == 'G3'


@pytest.mark.parametrize('k,esperado', [(30, 'V4'), (40, 'V2'), (48, 'V3')])
def test_V2_V3_V4(k, esperado):
    res = cx.contraste(table(S.make(img_errors={'gray_scott': 0, 'perlin': 0, 'lorenz': k}, beto_errors=4)), CFG)
    r = res['techniques']['lorenz']
    assert r['verdict'] == esperado
    if esperado == 'V2':
        assert r['superioridad_piso']['reject'] and not r['no_inferioridad']['reject'] and r['theta_interval'][1] < 0
    if esperado == 'V4':
        assert r['superioridad_piso']['reject'] and r['theta_interval'][0] <= 0 <= r['theta_interval'][1]


def test_theta_y_D_por_replica():
    F = table(S.make())
    res = cx.contraste(F, CFG)
    b = F['beto']['rep']
    for t in S.TECHS:
        i = F[('img', 'final', t)]['rep'].mean(axis=1)
        th = (i - 0.5) - 0.5 * (b - 0.5)
        p = float(np.mean(th <= 0))
        assert res['techniques'][t]['no_inferioridad']['p'] == p


# ── Piso (A2, A3, estratificado, permutado) ───────────────────────────────
def _cands():
    B = 6
    return {'majority': (0.38, np.full(B, 0.38)),
            'stratified': (0.5, np.full(B, 0.5)),
            'brightness': (0.55, np.array([0.55, 0.60, 0.50, 0.70, 0.40, 0.55])),
            'visual_permutado': (0.53, np.array([0.53, 0.65, 0.45, 0.50, 0.75, 0.50]))}


def test_piso_identidad_fija_frente_a_maximo_por_replica():
    img = np.full(6, 0.7)
    d_obs, d, info = cx.d_reps(0.7, img, _cands(), mode='identidad')
    assert info['piso'] == ['brightness'] and d_obs == pytest.approx(0.15)
    assert np.allclose(d, 0.7 - _cands()['brightness'][1])
    _, dmax, _ = cx.d_reps(0.7, img, _cands(), mode='max_por_replica')
    assert np.allclose(dmax, 0.7 - np.max([v[1] for v in _cands().values()], axis=0))
    assert (dmax <= d + 1e-15).all()


def test_empate_A3_menor_D():
    c = _cands()
    c['visual_permutado'] = (0.55, c['visual_permutado'][1])        # empate exacto en el observado
    img = np.full(6, 0.7)
    _, d, info = cx.d_reps(0.7, img, c)
    assert info['empate'] and sorted(info['piso']) == ['brightness', 'visual_permutado']
    assert np.allclose(d, np.minimum(0.7 - c['brightness'][1], 0.7 - c['visual_permutado'][1]))
    _, dmax, _ = cx.d_reps(0.7, img, c, ties='max')
    assert np.allclose(dmax, np.maximum(0.7 - c['brightness'][1], 0.7 - c['visual_permutado'][1]))


def test_estratificado_constante_y_permutado_promediado():
    F = table(S.make())
    c = cx.floor_candidates(F, 'perlin', CFG)
    assert c['stratified'][0] == 0.5 and (c['stratified'][1] == 0.5).all()
    perm = F[('img', 'control_permutacion', 'perlin')]
    assert c['visual_permutado'][0] == pytest.approx(perm['obs'].mean())
    assert np.allclose(c['visual_permutado'][1], perm['rep'].mean(axis=1))
    assert list(c) == CFG['pisos']['order']
    sin = cx.floor_candidates(F, 'perlin', CFG, include=CFG['sensibilidad']['piso_sin_permutado'])
    assert 'visual_permutado' not in sin


def test_piso_estratificado_como_identidad():
    """Si el estratificado es el máximo observado, D* = F1_img* − 0.5."""
    F = table(S.make())
    for t in S.TECHS:
        for k in ('brightness',):
            F[(k, t)] = {'obs': 0.3, 'rep': np.full(W.shape[0], 0.3)}
        F[('img', 'control_permutacion', t)] = {'obs': np.full(5, 0.2), 'rep': np.full((W.shape[0], 5), 0.2)}
    res = cx.contraste(F, CFG)
    for t in S.TECHS:
        assert res['techniques'][t]['piso'] == ['stratified']
        assert res['techniques'][t]['D_obs'] == pytest.approx(F[('img', 'final', t)]['obs'].mean() - 0.5)


def test_ic_de_D_con_alpha_h_de_superioridad():
    res = cx.contraste(table(S.make()), CFG)
    for t, r in res['techniques'].items():
        a = r['superioridad_piso']['alpha_h']
        assert r['D_interval_level'] == pytest.approx(1 - 2 * a)


# ── ρ̂ (A5, A6), descomposición, descriptores ─────────────────────────────
def test_retencion_menos_infinito_y_conteo():
    F = table(S.make())
    rep = F['beto']['rep'].copy()
    rep[:300] = 0.45                                                  # 300 réplicas con F1_BETO* ≤ 0.5
    F['beto'] = {'obs': F['beto']['obs'], 'rep': rep}
    out = cx.retencion(F, CFG)
    for t in S.TECHS:
        assert out[t]['n_denominador_no_positivo'] == 300
        assert out[t]['interval'][0] == float('-inf')
    F['beto'] = {'obs': F['beto']['obs'], 'rep': np.where(rep <= 0.5, 0.7, rep)}
    assert all(cx.retencion(F, CFG)[t]['n_denominador_no_positivo'] == 0 for t in S.TECHS)


def test_descomposicion_suma_y_descriptores():
    F = table(S.make())
    d = cx.descomposicion(F, CFG)
    for t in S.TECHS:
        assert d[t]['total_obs'] == pytest.approx(d[t]['beto_menos_img_obs'], abs=1e-12)
    v = cx.frente_a_descriptores(F, CFG)
    for t in S.TECHS:
        exp = F[('img', 'final', t)]['rep'].mean(axis=1) - F[('descriptors', t)]['rep']
        assert v[t]['interval'] == tuple(np.quantile(exp, [0.025, 0.975], method='linear'))


# ── Longitud (A7) ─────────────────────────────────────────────────────────
def test_longitud_solo_V1_V2_V4_y_m():
    F = table(S.make())
    out = cx.contraste_longitud(F, CFG, {'gray_scott': 'V1', 'perlin': 'V3', 'lorenz': 'V4'})
    assert out['m'] == 2 and set(out['techniques']) == {'gray_scott', 'lorenz'}
    for t, r in out['techniques'].items():
        L = F[('img', 'final', t)]['rep'].mean(axis=1) - F['longitud']['rep']
        assert r['p'] == float(np.mean(L <= 0))
        assert r['alpha_h'] in (0.05 / 2, 0.05)
        assert (r['calificacion'] is None) == r['reject']
    assert cx.contraste_longitud(F, CFG, {t: 'V3' for t in S.TECHS})['m'] == 0
    assert cx.contraste_longitud(F, CFG, {t: 'V3' for t in S.TECHS}, m_mode='m3')['m'] == 3


# ── McNemar, Cochran, kappa ───────────────────────────────────────────────
def test_mcnemar_y_cochran_contra_statsmodels_y_scipy():
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
    rng = np.random.default_rng(5)
    for _ in range(30):
        a, b, c = (rng.uniform(size=(3, S.N)) < [[0.7], [0.6], [0.8]]).astype(int)
        m = cx.mcnemar_bilateral(a, b)
        tab = [[int(((a == 1) & (b == 1)).sum()), m['b']], [m['c'], int(((a == 0) & (b == 0)).sum())]]
        assert m['p'] == pytest.approx(mcnemar(tab, exact=True).pvalue, abs=1e-12)
        u = cx.mcnemar_unilateral_more_errors(a, b)
        n = u['b'] + u['c']
        ref = 1.0 if n == 0 else stats.binomtest(u['b'], n, 0.5, alternative='greater').pvalue
        assert u['p'] == pytest.approx(ref, abs=1e-12)
        q, p, gl = cx.cochran_q(np.column_stack([a, b, c]))
        r = cochrans_q(np.column_stack([a, b, c]))
        assert q == pytest.approx(r.statistic, abs=1e-10) and p == pytest.approx(r.pvalue, abs=1e-12) and gl == 2
    with pytest.raises(cx.PendienteError):
        cx.cochran_q(np.ones((S.N, 3)))


def test_comparaciones_mcnemar_lados_y_holm():
    ins = S.make()
    yh = {'beto': bs.predict(ins.refs['beto'], 0.5), 'techo5d': bs.predict(ins.refs['techo5d'], 0.5)}
    for t in S.TECHS:
        yh[t] = cx.yhat_mean(ins.visual[('final', t)]['p'], CFG)
    out = cx.comparaciones_mcnemar(ins.y, yh, CFG)
    assert all(v['lado'] == 'unilateral' for fam in out['frente_a'].values() for v in fam.values())
    bil = cx.comparaciones_mcnemar(ins.y, yh, CFG, lado_vs_ref={'beto': 'bilateral', 'techo5d': 'bilateral'})
    assert all(v['lado'] == 'bilateral' for fam in bil['frente_a'].values() for v in fam.values())
    if out['cochran']['p'] < 0.05:
        assert len(out['pares']) == 3
    else:
        assert out['pares'] is None


def test_yhat_media_de_p_dep():
    p = np.array([[0.4, 0.6, 0.5, 0.5, 0.5], [0.6, 0.6, 0.4, 0.4, 0.51]])
    assert cx.yhat_mean(p, CFG).tolist() == [0, 1]                    # media 0.5 → 0 (estricto)


def test_kappa_contra_sklearn():
    rng = np.random.default_rng(7)
    a, b = rng.integers(0, 2, (2, S.N))
    assert cx.kappa_weighted(bs.observed_weights(S.N), a, b)[0] == pytest.approx(cohen_kappa_score(a, b), abs=1e-12)
    k = cx.kappa_weighted(W[:200], a, b)
    for j in range(200):
        assert k[j] == pytest.approx(cohen_kappa_score(a[IDX[j]], b[IDX[j]]), abs=1e-12)
    with pytest.raises(cx.PendienteError):
        cx.kappa_weighted(bs.observed_weights(S.N), np.ones(S.N), np.ones(S.N))


def test_kappa_y_contencion_f46():
    ins = S.make()
    out = rf.ejecutar_f46(ins, CFG, IDX)
    assert set(out['kappa']) == {'y_hat', 'vector_acierto'}
    assert 'beto__emb768' in out['kappa']['y_hat'] and 'techo5d__lorenz' in out['kappa']['y_hat']
    yv = cx.yhat_mean(ins.visual[('final', 'perlin')]['p'], CFG)
    yr = bs.predict(ins.refs['techo5d'], 0.5)
    c = out['contencion']['perlin']['techo5d']
    ev, er = yv != ins.y, yr != ins.y
    assert c['C_obs'] == pytest.approx((ev & er).sum() / ev.sum()) and c['nula_obs'] == pytest.approx(er.mean())
    j = 3
    num, den = (ev & er)[IDX[j]].sum(), ev[IDX[j]].sum()
    reps = (W @ (ev & er).astype(float)) / (W @ ev.astype(float))
    assert reps[j] == pytest.approx(num / den)
    assert c['interval'] == bs.percentile_interval(reps, 0.025, 0.975)
    assert out['contencion']['perlin']['condicion_2_F41'].startswith('PENDIENTE')
    with pytest.raises(cx.PendienteError):
        cx.contencion(W, ins.y, ins.y, yr, CFG)


# ── Terciles y Spearman (A8) ──────────────────────────────────────────────
def test_terciles_cortes_y_asignacion():
    ins = S.make()
    cuts = cx.tercile_cuts(ins.train['n_tokens_beto'], CFG)
    assert np.allclose(cuts, np.quantile(ins.train['n_tokens_beto'], [1 / 3, 2 / 3], method='linear'))
    lab = cx.assign_terciles([cuts[0], cuts[0] + 1e-9, cuts[1], cuts[1] + 1e-9], cuts)
    assert lab.tolist() == ['T1', 'T2', 'T2', 'T3']
    yh = {t: bs.predict(ins.visual[('final', t)]['p'], 0.5) for t in S.TECHS}
    out = cx.terciles_longitud(ins.y, ins.test_index['n_tokens_beto'].to_numpy(), ins.train['n_tokens_beto'].to_numpy(),
                               yh, bs.predict(ins.refs['beto'], 0.5), CFG)
    assert sum(e['n'] for e in out['estratos'].values()) == S.N
    m = cx.assign_terciles(ins.test_index['n_tokens_beto'], cuts) == 'T2'
    exp = np.mean([f1_score(ins.y[m], yh['lorenz'][m, s], average='macro', labels=[0, 1], zero_division=0)
                   for s in range(5)])
    assert out['estratos']['T2']['lorenz']['media'] == pytest.approx(exp, abs=1e-12)


def test_spearman_u_longitud():
    ins = S.make()
    out = cx.spearman_u_longitud(ins.u_train, ins.train['n_tokens_beto'], ins.train['truncated_128'], CFG)
    assert len(out) == 10
    r = stats.spearmanr(ins.u_train[:, 2], np.log(ins.train['n_tokens_beto'].astype(float)))
    assert out['u2__log(n_tokens_beto)']['rho'] == pytest.approx(r.statistic, abs=1e-12)


# ── Sensibilidad ──────────────────────────────────────────────────────────
def test_formulacion_A_p_y_V2():
    F = table(S.make(img_errors={'gray_scott': 0, 'perlin': 0, 'lorenz': 40}, beto_errors=4))
    out = sens.formulacion_a(F, CFG, S.CANON)
    assert set(out) == {'r=0.5', 'r=0.75', 'r=0.8'}
    delta = S.CANON['f11.delta_margen.r0_50']['value']['test']
    res = out['r=0.5']
    for t, r in res['techniques'].items():
        Delta = F['beto']['rep'] - F[('img', 'final', t)]['rep'].mean(axis=1)
        assert r['no_inferioridad']['p'] == float(np.mean(Delta >= delta))
        lo_delta = -r['theta_interval'][1] + delta                    # límite inferior del IC de Δ
        a = r['no_inferioridad']['alpha_h']
        assert lo_delta == pytest.approx(np.quantile(Delta, a, method='linear'), abs=1e-12)
        if r['verdict'] == 'V2':
            assert lo_delta > delta


def test_beto_desplazado_traslada_theta():
    F = table(S.make())
    base = cx.contraste(F, CFG)
    shift = S.CANON['f12.variabilidad_beto']['value']['val']
    out = sens.variabilidad_beto(F, CFG, S.CANON)
    assert len(out) == 2
    for key, s in zip(out, (1, -1)):
        for t in S.TECHS:
            d = out[key]['techniques'][t]['theta_obs'] - base['techniques'][t]['theta_obs']
            assert d == pytest.approx(-s * CFG['contraste']['r'] * shift, abs=1e-12)


def test_por_semilla_y_segundo_mejor_y_piso_sin_permutado():
    F = table(S.make())
    ps = sens.por_semilla(F, CFG)
    assert set(ps['por_semilla']) == set(CFG['modelos']['seeds'])
    for s, res in ps['por_semilla'].items():
        i = CFG['modelos']['seeds'].index(s)
        for t in S.TECHS:
            assert res['techniques'][t]['f1_img_obs'] == F[('img', 'final', t)]['obs'][i]
    sm = sens.segundo_mejor(F, CFG)
    for t in S.TECHS:
        assert sm['techniques'][t]['f1_img_obs'] == pytest.approx(F[('img', 'sensibilidad', t)]['obs'].mean())
        perm = F[('img', 'control_permutacion', t)]['obs'].mean()
        assert sm['techniques'][t]['piso_f1_obs']['visual_permutado'] == pytest.approx(perm)
    sp = sens.piso_sin_permutado(F, CFG)
    assert all('visual_permutado' not in r['piso_f1_obs'] for r in sp['techniques'].values())


def test_A14_emparejada_es_pendiente_sin_comparacion():
    F = table(S.make())
    with pytest.raises(cx.PendienteError):
        sens.por_semilla(F, S.cfg(comparacion=None), permutado='emparejada', comparacion=None)
    for comp in ('observado', 'por_replica'):
        out = sens.por_semilla(F, CFG, permutado='emparejada', comparacion=comp)
        assert set(out['n_semillas_V1']) == set(S.TECHS)


def test_tabla_de_robustez():
    ins = S.make()
    res = rf.ejecutar(ins, CFG, S.CANON, IDX, log=lambda *_: None)
    rows = res['robustez']
    alts = [r['alternativa'].split(':')[0] for r in rows]
    assert alts == ['A2', 'A3', 'A7', 'A12', 'A14']
    for r in rows[:4]:
        assert set(r['cambia_tecnica']) == set(S.TECHS) and isinstance(r['cambia_global'], bool)
    assert 'calificaciones' in rows[2] and 'principal' in rows[2]['calificaciones']
