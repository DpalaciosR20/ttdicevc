"""FX.1 (docs/plan_exploratorio_pixeles.md): pruebas sintéticas, sin datos reales."""
import copy
import json
import math
import os

import numpy as np
import pytest

from src.fase4.atribucion_comun import Detener
from src.fase_x import pixeles_comun as pc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_FILE = os.path.join(REPO, 'results', 'fx_pixeles', 'resoluciones.json')
RESUELTO = {'sobel_borde': 'reflect', 'c1_orden_azar': 'tras_los_desplazados_de_cada_imagen',
            'c1_muestreo_desplazamiento': 'rechazo_dx_luego_dy', 'c1_muestreo_azar': 'choice_sin_reemplazo',
            'b2_mapas': 'grad_y_log', 'wilcoxon_metodo': 'normal_con_correccion',
            'd_sensibilidad': 'dos_variantes_sigma2_y_t329'}


def res_resuelto(**cambios):
    r = json.load(open(RES_FILE, encoding='utf-8'))
    r.update(RESUELTO)
    r.update(cambios)
    return pc.check_resoluciones(r)


# ── Resoluciones ──────────────────────────────────────────────────────────
def test_resoluciones_versionadas_coinciden_con_el_plan_y_exigen_pendientes():
    r = json.load(open(RES_FILE, encoding='utf-8'))
    assert (r['_autor'], r['_fecha']) == ('SV', '2026-11-10')
    assert r['_nota'] == 'Exploratorio (plan_exploratorio_pixeles.md), fijado antes de calcular'
    assert {k: r[k] for k in pc.FIJOS} == pc.FIJOS
    assert {k: r[k] for k in pc.PENDIENTES} == RESUELTO                # resueltos por CE (2026-11-13)
    assert r['_pendientes_resueltos']['fecha'] == '2026-11-13'
    assert pc.check_resoluciones(r)['wilcoxon_metodo'] == 'normal_con_correccion'
    r_null = dict(r, b2_mapas=None)
    with pytest.raises(Detener, match='PENDIENTE sin resolver'):
        pc.check_resoluciones(r_null)


@pytest.mark.parametrize('cambio, error', [
    ({'clave_nueva': 1}, 'no previstas'),
    ({'k_sorteos': 10}, 'distintos de los del plan'),
    ({'umbral_t': 2.5}, 'distintos de los del plan'),
    ({'sobel_borde': 'constant'}, 'no previstos'),
])
def test_resoluciones_rechaza_claves_y_valores(cambio, error):
    with pytest.raises(Detener, match=error):
        res_resuelto(**cambio)


def test_resoluciones_rechaza_clave_faltante():
    r = json.load(open(RES_FILE, encoding='utf-8'))
    r.update(RESUELTO)
    del r['b_bootstrap']
    with pytest.raises(Detener, match='faltan claves'):
        pc.check_resoluciones(r)


# ── Regla 2.1: aborto ante prueba ─────────────────────────────────────────
def test_aborta_ante_split_test():
    with pytest.raises(Detener, match='prueba'):
        pc.exigir_split([{'split': 'val', 'sample_id': 'val_0000'}, {'split': 'test', 'sample_id': 'test_0000'}], 'val')
    with pytest.raises(Detener, match='prueba'):
        pc.exigir_split([{'split': 'test', 'sample_id': 'test_0001'}], 'train')
    with pytest.raises(Detener):
        pc.exigir_split([{'split': 'test'}], 'test')
    with pytest.raises(Detener, match='prueba'):
        pc.exigir_arreglo_split(np.array(['val', 'test']), 'test')
    with pytest.raises(Detener, match="se esperaba split 'val'"):
        pc.exigir_split([{'split': 'train', 'sample_id': 'train_0000'}], 'val')
    assert pc.exigir_arreglo_split(np.array(['val', 'test', 'val']), 'val').tolist() == [True, False, True]


# ── A: pendiente ──────────────────────────────────────────────────────────
def _escalon(px=64, col=32):
    img = np.zeros((px, px), np.uint8)
    img[:, col:] = 200
    return img


def test_grad_y_log_sobre_un_borde_conocido():
    img = _escalon()
    m = pc.mapas_pendiente(img, 1, 'reflect')
    perfil_g, perfil_l = m['grad'][32], m['log'][32]
    assert set(np.argsort(perfil_g)[-2:]) == {31, 32}             # máximo en el borde
    assert perfil_g[:20].max() < 1e-6 and perfil_g[45:].max() < 1e-6
    # gaussian_laplace truncado deja un residuo ∝ intensidad en zonas planas (≈ 1e-4 a 0.78), < 1e-3
    assert perfil_l[:20].max() < 1e-3 and perfil_l[45:].max() < 1e-3
    assert perfil_l[30] > 0.1 and perfil_l[33] > 0.1               # |LoG| a ambos lados del borde
    assert perfil_l[31:33].min() < perfil_l[30]                     # cruce por cero en el borde
    np.testing.assert_allclose(m['grad'], m['grad'][:1].repeat(64, 0))   # invariante en la vertical


def test_grad_crece_con_el_contraste_y_sigma2_lo_ensancha():
    a = pc.mapas_pendiente(_escalon(), 1, 'reflect')['grad'][32]
    b = pc.mapas_pendiente(_escalon() // 2, 1, 'reflect')['grad'][32]
    assert a.max() > 1.9 * b.max()
    s2 = pc.mapas_pendiente(_escalon(), 2, 'reflect')['grad'][32]
    assert (s2 > 1e-3).sum() > (a > 1e-3).sum()


def test_canny_solo_figuras():
    e = pc.mapa_canny(pc.a_unidad(_escalon()), 1)
    assert e.dtype == bool and e[:, 31:33].any() and not e[:, :20].any()


def test_a_unidad_exige_uint8():
    with pytest.raises(Detener):
        pc.a_unidad(np.zeros((4, 4), np.float32))


# ── B ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('g, n', [(7, 391), (14, 1567)])
def test_nula_b2_numero_de_transformaciones_sin_identidad(g, n):
    t = pc.transformaciones_rejilla(g)
    assert t.shape == (n, g * g)
    assert not (t == np.arange(g * g)).all(1).any()
    assert all(sorted(r) == list(range(g * g)) for r in t)
    assert len({tuple(r) for r in t}) == n                          # todas distintas


def test_k_del_plan():
    assert [pc.k_superior(q, 49) for q in (0.1, 0.2)] == [5, 10]
    assert [pc.k_superior(q, 196) for q in (0.1, 0.2)] == [20, 40]
    assert [pc.k_superior(q, 256 * 256) for q in (0.1, 0.2)] == [6554, 13108]


def test_dice_y_empates_con_orden_estable():
    a = np.array([5., 5., 5., 1., 0.])
    assert pc.top_k(a, 2).tolist() == [0, 1]
    assert pc.bottom_k(np.array([0., 0., 3., 0.]), 2).tolist() == [0, 1]
    assert pc.dice_topk(a, np.array([0., 9., 8., 0., 0.]), 2) == 0.5
    assert pc.dice_topk(np.arange(10.), np.arange(10.), 3) == 1.0
    assert pc.dice_topk(np.arange(10.), -np.arange(10.), 3) == 0.0


def test_dice_nulo_es_la_media_sobre_las_transformaciones():
    rng = np.random.default_rng(0)
    a, b = rng.random((7, 7)), rng.random((7, 7))
    t = pc.transformaciones_rejilla(7)
    ref = np.mean([pc.dice_topk(a, b.ravel()[p], 5) for p in t])
    assert pc.dice_nulo(a, b, 5, t) == pytest.approx(ref)


def test_parches_suma_y_media():
    m = np.arange(224 * 224, dtype=float).reshape(1, 224, 224)
    np.testing.assert_allclose(pc.suma_por_parche(m, 7)[0, 0, 0], m[0, :32, :32].sum())
    np.testing.assert_allclose(pc.media_por_parche(m, 14)[0, 1, 2], m[0, 16:32, 32:48].mean())


def test_analizar_b_sintetico():
    from src.fase_x.pixeles_local import analizar_b
    res = res_resuelto()
    rng = np.random.default_rng(1)
    n = 12
    pend = {'grad': rng.random((n, 224, 224)), 'log': rng.random((n, 224, 224))}
    real = pend['grad'] * 3 + 0.01 * rng.random((n, 224, 224))     # atribución = pendiente
    perm = rng.random((n, 224, 224))
    real[0] = 1.0                                                     # un mapa constante: se excluye
    y = np.array([0, 1] * (n // 2))
    out, filas, npz = analizar_b('gray_scott', real, perm, pend, pend, y, res, pc.indices_bootstrap(n, 2026, 200))
    assert out['n_mapa_constante']['real'] == 1 and out['B1']['real|grad']['n_excluidas'] == 1
    assert out['B1']['real|grad']['media'] > 0.9 and out['B1']['real|grad']['sobre_0']
    assert out['B2']['real|grad|q0.1']['delta']['media'] > 0.5
    assert out['B3']['B1|grad']['sobre_0']
    assert set(out['B4']) == {'grad', 'log'} and len(filas) == n
    assert npz['gray_scott|atrib|real'].shape == (n, 7, 7)


# ── C1 ────────────────────────────────────────────────────────────────────
def test_mascaras_c1_area_exacta_y_desplazamiento():
    rng = np.random.default_rng(3)
    pend = rng.random((256, 256))
    pend[:10] = 0.5                                                   # empates
    for k in (6554, 13108):
        alta = pc.mascara(pc.top_k(pend, k))
        baja = pc.mascara(pc.bottom_k(pend, k))
        assert alta.sum() == k and baja.sum() == k and not (alta & baja).any()
        assert pc.desplazar(alta, 40, -100).sum() == k
    for _ in range(500):
        dy, dx = pc._desplazamiento(rng, 'rechazo_dx_luego_dy')
        assert -128 <= dx <= 127 and -128 <= dy <= 127 and abs(dx) + abs(dy) >= 32
    v = pc.pares_validos()
    assert len(v) == 256 * 256 - (2 * 31 * 32 + 1) and (np.abs(v).sum(1) >= 32).all()
    for _ in range(200):
        dy, dx = pc._desplazamiento(rng, 'uniforme_sobre_pares_validos', validos=v)
        assert abs(dx) + abs(dy) >= 32
    for metodo in ('choice_sin_reemplazo', 'prefijo_de_permutacion'):
        ix = pc._azar(rng, 6554, metodo)
        assert len(np.unique(ix)) == 6554 and pc.mascara(ix).sum() == 6554


@pytest.mark.parametrize('orden', ['tras_los_desplazados_de_cada_imagen', 'tras_todos_los_desplazados_de_la_tecnica'])
def test_generador_c1_reproducible_y_en_el_orden_del_plan(orden):
    res = res_resuelto(c1_orden_azar=orden)
    k_q = {0.1: 50, 0.2: 100}

    def correr():
        return list(pc.sorteos_c1(np.random.default_rng(2026), ('grad', 'log'), [0.1, 0.2], 3, 4, k_q, res))

    a, b = correr(), correr()
    assert [(m, q, i) for m, q, i, _, _ in a] == [(m, q, i) for m in ('grad', 'log') for q in (0.1, 0.2) for i in range(3)]
    assert all(x[3] == y[3] and all((u == v).all() for u, v in zip(x[4], y[4])) for x, y in zip(a, b))
    # el primer sorteo de la técnica es el primer desplazamiento de (grad, 10 %, imagen 0)
    rng = np.random.default_rng(2026)
    assert a[0][3][0] == pc._desplazamiento(rng, 'rechazo_dx_luego_dy')
    # orden explícito: reconstruir a mano la secuencia
    rng = np.random.default_rng(2026)
    claves = [(m, q, i) for m in ('grad', 'log') for q in (0.1, 0.2) for i in range(3)]
    if orden == 'tras_los_desplazados_de_cada_imagen':
        for m, q, i in claves:
            d = [pc._desplazamiento(rng, 'rechazo_dx_luego_dy') for _ in range(4)]
            z = [rng.choice(65536, size=k_q[q], replace=False) for _ in range(4)]
            got = a[claves.index((m, q, i))]
            assert got[3] == d and all((u == v).all() for u, v in zip(got[4], z))
    else:
        d = {c: [pc._desplazamiento(rng, 'rechazo_dx_luego_dy') for _ in range(4)] for c in claves}
        for c in claves:
            z = [rng.choice(65536, size=k_q[c[1]], replace=False) for _ in range(4)]
            got = a[claves.index(c)]
            assert got[3] == d[c] and all((u == v).all() for u, v in zip(got[4], z))


def test_sorteos_cambian_con_la_tecnica_solo_por_el_generador_nuevo():
    res = res_resuelto()
    k_q = {0.1: 10, 0.2: 20}
    a = list(pc.sorteos_c1(np.random.default_rng(2026), ('grad',), [0.1], 2, 3, k_q, res))
    b = list(pc.sorteos_c1(np.random.default_rng(2026), ('grad',), [0.1], 2, 3, k_q, res))
    assert a[0][3] == b[0][3]                                        # default_rng(2026) nuevo por técnica


def test_recorte_y_redondeo_uint8():
    img = np.array([[0, 255], [100, 101]], np.uint8)
    out = pc.suavizar(img, np.ones((2, 2), bool), np.array([[-20.0, 300.0], [100.4, 100.6]]))
    assert out.dtype == np.uint8 and out.tolist() == [[0, 255], [100, 101]]
    keep = pc.suavizar(img, np.zeros((2, 2), bool), np.full((2, 2), 7.0))
    assert keep.tolist() == img.tolist()


def test_imagenes_c1_orden_y_regiones():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    pend = pc.mapas_pendiente(img, 1, 'reflect')['grad']
    d4 = pc.desenfocar(img, 4)
    desp, azar = [(40, 0), (0, -64)], [pc._azar(rng, 6554, 'choice_sin_reemplazo') for _ in range(2)]
    ims = pc.imagenes_c1(img, pend, 0.1, 6554, desp, azar, d4)
    assert len(ims) == 2 + 2 * 2
    alta = pc.mascara(pc.top_k(pend, 6554))
    assert (ims[0][~alta] == img[~alta]).all()                       # fuera de la máscara no cambia
    assert (ims[2][~pc.desplazar(alta, 40, 0)] == img[~pc.desplazar(alta, 40, 0)]).all()
    assert (ims[4][~pc.mascara(azar[0])] == img[~pc.mascara(azar[0])]).all()


def test_estadistico_s():
    p = np.array([0.5, 0.2])
    s = pc.estadistico_s(p, np.array([0.9, 0.2]), np.array([[0.6, 0.4], [0.3, 0.1]]))
    np.testing.assert_allclose(s, [0.4 - 0.1, 0.0 - 0.1])


# ── C2 ────────────────────────────────────────────────────────────────────
def test_oclusion_8x8():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    d8 = pc.desenfocar(img, 8)
    ims = pc.imagenes_c2(img, d8)
    assert len(ims) == 64 and pc.parches_c2()[9] == (1, 1)
    for j, (r, c) in enumerate(pc.parches_c2()):
        cambio = ims[j] != img
        fuera = np.ones((256, 256), bool)
        fuera[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = False
        assert not cambio[fuera].any() and cambio[~fuera].any()
        assert (ims[j][~fuera] == pc.a_uint8(d8)[~fuera]).all()
    o = pc.mapa_oclusion(np.tile(np.linspace(0, 0.63, 64), (2, 1)), [0.0, 0.63])
    assert o.shape == (2, 8, 8) and o[0, 7, 7] == pytest.approx(0.63) and o[1, 7, 7] == 0


def test_cordura_c2_criterio_relativo():
    rng = np.random.default_rng(6)
    base = rng.random((30, 8, 8))
    real = {s: base + 0.05 * rng.random((30, 8, 8)) for s in pc.SEEDS}
    perm = {s: rng.random((30, 8, 8)) for s in pc.SEEDS}
    crit = pc.FIJOS['c2_criterio_f43']
    assert pc.cordura_c2(real, perm, crit)['global'] == 'cambia'
    assert pc.cordura_c2(real, {s: base + 0.05 * rng.random((30, 8, 8)) for s in pc.SEEDS}, crit)['global'] == 'no_cambia'
    perm_c = {s: v.copy() for s, v in perm.items()}
    perm_c[42][0] = 0.3                                              # mapa constante: se excluye y se cuenta
    d = pc.cordura_c2(real, perm_c, crit)
    assert d['por_semilla_detalle']['42']['n_nan'] == 1 and d['por_semilla_detalle']['42']['n_perm_constante'] == 1


# ── D ─────────────────────────────────────────────────────────────────────
def test_welch_permutaciones_igual_al_directo():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 50))
    y = np.r_[np.ones(15, int), np.zeros(25, int)]
    Y = np.stack([rng.permutation(y) for _ in range(5)])
    T = pc.welch_t_permutaciones(X, Y)
    for i in range(5):
        np.testing.assert_allclose(T[i], pc.welch_t(X[Y[i] == 1], X[Y[i] == 0]), atol=1e-10)
    with pytest.raises(Detener):
        pc.welch_t_permutaciones(X, np.vstack([Y[0], np.ones(40)]))


def test_clusteres_conectividad_8_y_signos():
    t = np.zeros((10, 10))
    t[2, 2] = t[3, 3] = 5.0                                          # diagonales: un clúster con conectividad 8
    t[7, 7] = -4.0
    c = pc.clusteres(t, 2.58)
    assert [(x['signo'], x['area'], x['masa']) for x in c] == [('+', 2, 10.0), ('-', 1, 4.0)]
    assert c[0]['centroide_yx'] == [2.5, 2.5]
    assert pc.masa_maxima(t, 2.58) == 10.0 and pc.masa_maxima(np.zeros((5, 5)), 2.58) == 0.0


def test_prueba_por_clusteres_con_senal_plantada():
    rng = np.random.default_rng(8)
    n, h = 60, 24
    y = np.r_[np.ones(30, int), np.zeros(30, int)]
    X = rng.normal(size=(n, h, h))
    X[y == 1, 5:11, 5:11] += 1.5                                      # señal en un bloque
    r = pc.prueba_clusteres(X, y, 2.58, 99, 2026)
    top = r['clusteres'][0]
    assert top['signo'] == '+' and top['p'] == pytest.approx(1 / 100) and r['p_mayor'] == top['p']
    assert 5 <= top['centroide_yx'][0] <= 11 and 5 <= top['centroide_yx'][1] <= 11
    r0 = pc.prueba_clusteres(rng.normal(size=(n, h, h)), y, 2.58, 99, 2026)
    assert r0['p_mayor'] > 0.05
    r2 = pc.prueba_clusteres(X, y, 2.58, 99, 2026)
    np.testing.assert_array_equal(r['nula_masa_maxima'], r2['nula_masa_maxima'])   # reproducible


def test_analizar_d_sintetico():
    from src.fase_x.pixeles_local import analizar_d
    res = res_resuelto()
    res = {**res, 'b_permutaciones_d': 49}
    rng = np.random.default_rng(9)
    y = np.r_[np.ones(20, int), np.zeros(20, int)]
    X = {m: rng.random((40, 16, 16)) for m in ('grad', 'log')}
    X['grad'][y == 1, 4:8, 4:8] += 1.0
    out, npz = analizar_d('lorenz', X, y, res, X)
    assert out['grad']['p_familia'] == pytest.approx(1 / 50) and out['log']['p_familia'] > 0.05
    assert set(out['grad']['sensibilidad']) == {'sigma2_t258', 'sigma1_t329'}
    out_p, _ = analizar_d('perlin', X, y, res, X)
    assert out_p['grad']['biserial'] > 0.5 and 'sensibilidad_sigma2' in out_p['grad']
    res1 = {**res, 'd_sensibilidad': 'una_variante_sigma2_con_t329'}
    assert set(analizar_d('gray_scott', X, y, res1, X)[0]['grad']['sensibilidad']) == {'sigma2_t329'}


# ── Estadística ───────────────────────────────────────────────────────────
def test_mann_whitney_y_biserial():
    r = pc.mann_whitney([3., 4., 5., np.nan], [1., 2.])
    assert r['n1'] == 3 and r['biserial'] == 1.0 and 0 < r['p'] < 1
    assert pc.mann_whitney([1., 2.], [3., 4., 5.])['biserial'] == -1.0


def test_wilcoxon_metodos():
    s = np.random.default_rng(10).normal(0.2, 1, 114)
    a, b = pc.wilcoxon(s, 'normal_sin_correccion'), pc.wilcoxon(s, 'normal_con_correccion')
    assert a['n'] == 114 and a['p'] < b['p']
    assert pc.wilcoxon(np.r_[0.0, s], 'normal_sin_correccion')['n_ceros'] == 1


@pytest.mark.parametrize('familia, m', [('H-B', 4), ('H-C', 12), ('H-D', 6)])
def test_holm_por_familia_con_m_del_plan(familia, m):
    assert pc.FIJOS['familias'][familia] == m
    p = {f'p{i}': (i + 1) * 0.004 for i in range(m)}
    out = pc.holm(p, m)
    assert out['p0']['p_holm'] == pytest.approx(min(1, m * 0.004))
    assert out['p0']['reject']
    with pytest.raises(Detener, match='m = '):
        pc.holm(dict(list(p.items())[:-1]), m)


def test_ic_bootstrap_compartido_por_tecnica():
    idx = pc.indices_bootstrap(114, 2026, 500)
    np.testing.assert_array_equal(idx, pc.indices_bootstrap(114, 2026, 500))
    v = np.r_[np.full(57, 0.3), np.full(57, 0.1)]
    v[0] = np.nan
    r = pc.ic(v, idx)
    assert r['n'] == 113 and r['sobre_0'] and r['ic_inf'] <= r['media'] <= r['ic_sup']


# ── De punta a punta (CPU, modelos e imágenes falsos; compuerta sustituida) ──
def _resoluciones_tmp(tmp_path):
    r = json.load(open(RES_FILE, encoding='utf-8'))
    r.update(RESUELTO)
    p = tmp_path / 'resoluciones.json'
    p.write_text(json.dumps(r), encoding='utf-8')
    return str(p)


def _imagenes_falsas(n_max=200):
    rng = np.random.default_rng(11)
    banco = rng.integers(0, 256, (n_max, 256, 256), dtype=np.uint8)

    def imagenes(tech, rows, split):
        pc.exigir_split(rows, split)
        return banco[:len(rows)], 'huella-sintetica'
    return imagenes, banco


def test_pixeles_gpu_de_punta_a_punta(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from src.fase_x import pixeles_gpu as pg
    from src.fase_x import pixeles_local as pl
    from src.fase4.atribucion_gpu import tensores
    from data import get_transforms
    imagenes, banco = _imagenes_falsas()
    monkeypatch.setattr(pc, 'check_gate', lambda *a, **k: 'commit-sintetico')
    monkeypatch.setattr(pl, 'imagenes', imagenes)
    tf = get_transforms(False, img_size=224, hflip=False)

    class Diminuto(torch.nn.Module):
        def __init__(self, semilla):
            super().__init__()
            torch.manual_seed(semilla)
            self.c = torch.nn.Conv2d(3, 4, 5, stride=4)
            self.h = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.h(torch.relu(self.c(x)).mean((2, 3)))

    def cargar(ub, tech, perm, device):
        ms = {s: Diminuto(s + (1000 if perm else 0)).eval() for s in pc.SEEDS}
        X = tensores(list(banco[:3]), tf)
        with torch.no_grad():
            info = {str(s): {'sha256': 'x', 'val_probs': torch.softmax(m(X).double(), 1)[:, 1].tolist()}
                    for s, m in ms.items()}
        return ms, info
    monkeypatch.setattr(pg, 'cargar_modelos', cargar)
    out = tmp_path / 'gpu'
    assert pg.main(['--models-root', str(tmp_path), '--resoluciones', _resoluciones_tmp(tmp_path),
                    '--out-dir', str(out), '--only', 'gray_scott', '--limit', '3', '--k-sorteos', '10']) == 0
    r = json.load(open(out / 'resultados.json', encoding='utf-8'))
    assert r['meta']['smoke'] and r['meta']['k_sorteos'] == 10 and 'estimacion_114_s' in r['meta']
    assert max(r['meta']['reproduccion_pdep']['gray_scott'].values()) < 1e-12
    c1 = r['resultados']['C1']['gray_scott']['por_condicion']
    assert set(c1) == {'grad|q0.1', 'grad|q0.2', 'log|q0.1', 'log|q0.2'}
    assert 'familia incompleta' in r['resultados']['C1']['H-C']['_nota']
    assert r['resultados']['C2']['gray_scott']['lectura'] in ('interpretable', 'sin validación por datos')
    z = np.load(out / 'arreglos.npz')
    assert z['C1|gray_scott|grad|q0.1|p_por_semilla'].shape == (3, 2 + 2 * 10, 5)
    d = z['C1|gray_scott|grad|q0.1|desplazamientos']
    assert (np.abs(d).sum(2) >= 32).all()
    assert z['C2|gray_scott|real|s42'].shape == (3, 8, 8)
    # no se permite otro K que 20 o la reducción autorizada a 10
    with pytest.raises(Detener, match='K = 7'):
        pg.main(['--models-root', str(tmp_path), '--resoluciones', _resoluciones_tmp(tmp_path),
                 '--out-dir', str(tmp_path / 'otra'), '--limit', '3', '--k-sorteos', '7'])


def test_pixeles_gpu_se_detiene_si_no_reproduce_pdep(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from src.fase_x import pixeles_gpu as pg
    from src.fase_x import pixeles_local as pl
    imagenes, _ = _imagenes_falsas()
    monkeypatch.setattr(pc, 'check_gate', lambda *a, **k: 'commit-sintetico')
    monkeypatch.setattr(pl, 'imagenes', imagenes)

    def cargar(ub, tech, perm, device):
        m = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2)).eval()
        return {s: m for s in pc.SEEDS}, {str(s): {'sha256': 'x', 'val_probs': [0.123] * 114} for s in pc.SEEDS}
    monkeypatch.setattr(pg, 'cargar_modelos', cargar)
    with pytest.raises(Detener, match='no reproduce'):
        pg.main(['--models-root', str(tmp_path), '--resoluciones', _resoluciones_tmp(tmp_path),
                 '--out-dir', str(tmp_path / 'g'), '--only', 'gray_scott', '--limit', '2', '--partes', 'C1'])


def test_pixeles_local_de_punta_a_punta(tmp_path, monkeypatch):
    from src.fase_x import pixeles_local as pl
    from src.fase4 import atribucion_local as al
    imagenes, _ = _imagenes_falsas()
    monkeypatch.setattr(pc, 'check_gate', lambda *a, **k: 'commit-sintetico')
    monkeypatch.setattr(pl, 'imagenes', imagenes)

    class UbFalso:
        def __init__(self, *a):
            self.verificados = {}
    monkeypatch.setattr(al, 'Ubicador', UbFalso)
    rng = np.random.default_rng(12)

    def mapas(ub, tech, perm, rows):
        return rng.random((len(rows), 224, 224)), {s: 1.0 for s in pc.SEEDS}
    monkeypatch.setattr(pl, 'mapas_val', mapas)
    out = tmp_path / 'local'
    assert pl.main(['--root', str(tmp_path), '--resoluciones', _resoluciones_tmp(tmp_path), '--out-dir', str(out),
                    '--limit', '12', '--partes', 'B']) == 0
    r = json.load(open(out / 'resultados.json', encoding='utf-8'))
    assert set(r['resultados']['B']) == {'gray_scott', 'perlin', 'H-B'}
    assert set(r['resultados']['B']['H-B']) == {'gray_scott|grad', 'gray_scott|log', 'perlin|grad', 'perlin|log'}
    assert (out / 'b_por_muestra.csv').exists() and 'estimacion_s' in r['meta']
