"""Pruebas sintéticas de F4.4 (convergencia.py) y F4.8 (figura_ilustrativa.py).
Sin datos reales, salvo archivos versionados del repositorio (Tabla 11, canonical_numbers.json)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.fase4 import convergencia as cv  # noqa: E402
from src.fase4 import figura_ilustrativa as fi  # noqa: E402
from src.fase4.atribucion_comun import Detener  # noqa: E402

RES_F44 = {k: v[0] for k, v in cv.RESOLUCIONES.items()}
RES_F48 = {k: v[0] for k, v in fi.RESOLUCIONES.items()}


def _img(seed=0, n=256):
    return np.random.default_rng(seed).integers(0, 256, size=(n, n), dtype=np.uint8)


# ── F4.4: parches y descriptores ──────────────────────────────────────────
def test_limites_exactos_y_proporcionales():
    assert list(cv.limites(224, 7)) == list(range(0, 225, 32))
    assert list(cv.limites(224, 14)) == list(range(0, 225, 16))
    b = cv.limites(256, 7)
    assert b[0] == 0 and b[-1] == 256 and set(np.diff(b)) <= {36, 37}
    b = cv.limites(256, 14)
    assert b[-1] == 256 and set(np.diff(b)) <= {18, 19}


@pytest.mark.parametrize('g', [7, 14])
@pytest.mark.parametrize('nombre', ['shannon_entropy', 'contrast', 'edge_density', 'lbp_energy'])
def test_descriptor_por_parche_es_la_funcion_de_f24_en_el_recorte(nombre, g):
    img = _img(1)
    fn = cv.funcion_descriptor(nombre)
    out = cv.descriptor_por_parche(img, g, fn)
    assert out.shape == (g, g)
    b = cv.limites(256, g)
    for i, j in ((0, 0), (g - 1, g - 1), (2, g // 2)):
        assert out[i, j] == fn(img[b[i]:b[i + 1], b[j]:b[j + 1]])


def test_funcion_descriptor_es_la_de_f24():
    m = cv.f24()
    assert m.__file__.replace('\\', '/').endswith('results/f24_descriptores/f24_descriptores.py')
    for d in m.FEATURES:
        assert cv.funcion_descriptor(d) is getattr(m, d)
    with pytest.raises(Detener):
        cv.funcion_descriptor('otro')


def test_descriptor_exige_uint8():
    with pytest.raises(Detener):
        cv.descriptor_por_parche(_img().astype(float), 7, cv.funcion_descriptor('contrast'))


def test_redimensionar_224_igual_que_f14():
    prep = {'size': 224, 'interpolation': 'bilinear', 'antialias': True}
    out = cv.redimensionar_224(_img(2), prep)
    assert out.shape == (224, 224) and out.dtype == np.uint8
    assert cv.imagen_para_parche(_img(2), RES_F44, prep).shape == (256, 256)
    assert cv.imagen_para_parche(_img(2), {**RES_F44, 'geometria_parche': 'redimensionada_224'}, prep).shape == (224, 224)


def test_masa_por_parche_conserva_la_suma():
    m = np.random.default_rng(3).random((4, 224, 224))
    for g in (7, 14):
        s = cv.masa_por_parche(m, g)
        assert s.shape == (4, g, g)
        np.testing.assert_allclose(s.sum((1, 2)), m.sum((1, 2)), rtol=1e-10)
        k = 224 // g
        np.testing.assert_allclose(s[0, 1, 2], m[0, k:2 * k, 2 * k:3 * k].sum(), rtol=1e-10)


def test_descriptores_significativos_de_la_tabla11():
    sig, filas = cv.descriptores_significativos()
    assert sig == cv.ESPERADOS
    assert sum(len(v) for v in sig.values()) == 8
    assert all(filas[(t, d)]['sig_holm'] == 'True' for t in sig for d in sig[t])


def test_descriptores_significativos_detiene_si_cambia(tmp_path):
    rows = list(csv.DictReader(open(cv.TABLA11, encoding='utf-8')))
    rows[0]['sig_holm'] = 'False'
    p = tmp_path / 't11.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(Detener):
        cv.descriptores_significativos(p)


# ── F4.4: Spearman e IC ───────────────────────────────────────────────────
def test_spearman_por_muestra_igual_a_scipy_y_nan_si_constante():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(4)
    a, b = rng.random((5, 7, 7)), rng.random((5, 7, 7))
    a[1, :3] = 0.5                                                # empates
    b[3] = 1.0                                                    # constante
    r = cv.spearman_por_muestra(a, b)
    for i in (0, 1, 2, 4):
        np.testing.assert_allclose(r[i], spearmanr(a[i].ravel(), b[i].ravel()).statistic, rtol=1e-12)
    assert np.isnan(r[3])
    np.testing.assert_allclose(cv.spearman_por_muestra(a, a), 1.0)
    np.testing.assert_allclose(cv.spearman_por_muestra(a, -a), -1.0)


def test_tratar_indefinidos():
    v = np.array([0.5, np.nan, -0.2])
    x, n = cv.tratar_indefinidos(v, 'excluir_y_contar')
    assert n == 1 and np.isnan(x[1])
    x, n = cv.tratar_indefinidos(v, 'cero')
    assert n == 1 and list(x) == [0.5, 0.0, -0.2]


def test_ic_bootstrap_constante_y_determinista():
    idx = cv.indices_bootstrap(20, np.random.default_rng(2026), 500)
    r = cv.ic_bootstrap(np.full(20, 0.3), idx)
    assert r['media'] == pytest.approx(0.3) and r['ic_inf'] == pytest.approx(0.3) and r['ic_sup'] == pytest.approx(0.3)
    assert r['sobre_0'] and not r['contiene_0']
    v = np.random.default_rng(5).normal(0.1, 0.3, 50)
    i1 = cv.indices_bootstrap(50, np.random.default_rng(2026), 2000)
    i2 = cv.indices_bootstrap(50, np.random.default_rng(2026), 2000)
    a, b = cv.ic_bootstrap(v, i1), cv.ic_bootstrap(v, i2)
    assert a == b
    assert a['ic_inf'] < a['media'] < a['ic_sup']
    medias = v[i1].mean(1)
    assert a['ic_inf'] == pytest.approx(np.quantile(medias, 0.025, method='linear'))


def test_ic_bootstrap_excluye_nan_en_cada_replica():
    v = np.array([1.0, np.nan, 3.0, np.nan])
    idx = np.array([[0, 1, 2, 3], [1, 1, 0, 0], [3, 3, 3, 2]])
    r = cv.ic_bootstrap(v, idx, nivel=0.5)
    assert r['media'] == 2.0 and r['n'] == 2 and r['b_validas'] == 3
    # medias por réplica: 2, 1, 3
    assert r['ic_inf'] == pytest.approx(np.quantile([2, 1, 3], 0.25)) and r['ic_sup'] == pytest.approx(np.quantile([2, 1, 3], 0.75))


def _sinteticos(n=30, g=7):
    rng = np.random.default_rng(6)
    masa = rng.random((n, g, g))
    desc = {'gray_scott': {'shannon_entropy': masa + 0.1 * rng.random((n, g, g)),
                           'contrast': masa.copy(), 'edge_density': -masa, 'lbp_energy': rng.random((n, g, g))}}
    desc['gray_scott']['contrast'][0] = 2.0                      # una muestra constante
    return desc, {'gray_scott': {'map': masa}}, {'gray_scott': cv.ESPERADOS['gray_scott']}


def test_analizar_signos_indefinidos_y_replicas_compartidas():
    desc, masas, sig = _sinteticos()
    r, filas = cv.analizar(desc, masas, sig, RES_F44, b=1000)
    g = r['gray_scott']['map']
    assert g['contrast']['n_indefinidos'] == 1 and g['contrast']['n'] == 29 and g['contrast']['media'] == pytest.approx(1.0)
    assert g['edge_density']['media'] == pytest.approx(-1.0) and g['edge_density']['bajo_0']
    assert g['shannon_entropy']['sobre_0']
    assert g['lbp_energy']['contiene_0']
    assert len(filas) == 4 * 30 and sum(f['spearman'] == '' for f in filas) == 1
    # réplicas compartidas por técnica: dos descriptores idénticos dan el mismo IC
    desc['gray_scott']['lbp_energy'] = desc['gray_scott']['shannon_entropy']
    r2, _ = cv.analizar(desc, masas, sig, RES_F44, b=1000)
    assert r2['gray_scott']['map']['lbp_energy'] == r2['gray_scott']['map']['shannon_entropy']
    r3, _ = cv.analizar(desc, masas, sig, {**RES_F44, 'bootstrap_rng': 'por_tecnica_descriptor'}, b=1000)
    assert r3['gray_scott']['map']['lbp_energy'] == r3['gray_scott']['map']['shannon_entropy']


def test_analizar_modo_cero():
    desc, masas, sig = _sinteticos()
    r, _ = cv.analizar(desc, masas, sig, {**RES_F44, 'spearman_indefinido': 'cero'}, b=500)
    c = r['gray_scott']['map']['contrast']
    assert c['n'] == 30 and c['n_indefinidos'] == 1 and c['media'] == pytest.approx(29 / 30)


def test_check_resoluciones_f44():
    assert cv.check_resoluciones({**RES_F44, '_autor': 'SV'}) == RES_F44
    with pytest.raises(Detener):
        cv.check_resoluciones({k: v for k, v in RES_F44.items() if k != 'mapa'})
    with pytest.raises(Detener):
        cv.check_resoluciones({**RES_F44, 'mapa': 'otro'})
    with pytest.raises(Detener):
        cv.check_resoluciones({**RES_F44, 'extra': 'x'})


# ── F4.4: contexto desde canonical_numbers (sin cálculos nuevos) ──────────
NUMBERS = json.loads(cv.CANON.read_text(encoding='utf-8'))['numbers']


def test_tabla_cualitativa_y_contexto_desde_canonical():
    sig, filas = cv.descriptores_significativos()
    t = cv.tabla_cualitativa(NUMBERS, sig, filas, RES_F44)
    assert set(t) == set(cv.TECNICAS)
    for tech in cv.TECNICAS:
        assert set(t[tech]['delta_f1']) == {'paso_bajo', 'paso_alto'}
        for banda in ('paso_bajo', 'paso_alto'):
            for s in (1, 2, 4, 8):
                assert t[tech]['delta_f1'][banda][f's{s}'] == NUMBERS[f'f43.{tech}.ablacion.{banda}.s{s}.delta_f1']['value']['val']
        ds = t[tech]['descriptores']
        assert [x['descriptor'] for x in ds] == sorted(sig[tech], key=lambda d: -abs(float(filas[(tech, d)]['cohen_d'])))
        assert all(x['naturaleza'] for x in ds)
    t2 = cv.tabla_cualitativa(NUMBERS, sig, filas, {**RES_F44, 'naturaleza_descriptores': 'sin_clasificar'})
    assert all(x['naturaleza'] is None for v in t2.values() for x in v['descriptores'])
    a11 = cv.contexto_a11(NUMBERS)
    for tech in cv.TECNICAS:
        assert a11[tech]['delta'] == NUMBERS[f'f5.frente_a_descriptores.{tech}']['value']['test']
        assert a11[tech]['ic_inf'] <= a11[tech]['delta'] <= a11[tech]['ic_sup']
    assert cv.advertencias(NUMBERS)['lorenz'] == 'dependiente_de_la_cabeza_sin_validacion_por_datos'


# ── F4.8: recorridos ──────────────────────────────────────────────────────
def test_centroides_y_recorrido_lineal():
    u = np.array([[0.0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0], [1.0, 1, 1, 1, 1], [0.8, 1, 1, 1, 1]])
    y = np.array([0, 0, 1, 1])
    c0, c1 = fi.centroides(u, y)
    np.testing.assert_allclose(c0, [0.1, 0, 0, 0, 0]); np.testing.assert_allclose(c1, [0.9, 1, 1, 1, 1])
    p, t = fi.recorrido_centroides(c0, c1)
    assert p.shape == (7, 5) and list(t) == list(np.linspace(0, 1, 7))
    np.testing.assert_allclose(p[0], c0); np.testing.assert_allclose(p[-1], c1)
    np.testing.assert_allclose(np.diff(p, axis=0), np.tile((c1 - c0) / 6, (6, 1)))


def test_recorrido_direccion_y_recorte():
    rng = np.random.default_rng(7)
    u = rng.random((200, 5))
    d = np.array([1.0, 0, 0, 0, 0])
    p, s, de = fi.recorrido_direccion(u, d, 'proyeccion_train', 7)
    assert de == pytest.approx(u[:, 0].std())
    np.testing.assert_allclose(p[3], u.mean(0))
    np.testing.assert_allclose(p[-1] - p[3], 1.5 * de * d)
    np.testing.assert_allclose(p[0] - p[3], -1.5 * de * d)
    _, _, de2 = fi.recorrido_direccion(u, d, 'media_de_componentes', 5)
    assert de2 == pytest.approx(u.std(0).mean())
    pr, marca = fi.recortar(np.array([[-0.1, 0.5, 1.2, 0, 1], [0.2, 0.3, 0.4, 0.5, 0.6]]))
    np.testing.assert_allclose(pr[0], [0, 0.5, 1, 0, 1])
    assert marca[0].tolist() == [True, False, True, False, False] and not marca[1].any()


def test_direccion_logreg_en_u_y_monotonia():
    rng = np.random.default_rng(8)
    u = rng.random((300, 5))
    y = (u[:, 0] * 2 - u[:, 4] + 0.1 * rng.normal(size=300) > 0.5).astype(int)
    model, m = fi.ajustar_logreg(u, y)
    lr, sc = model[-1], model[0]
    assert lr.C == 0.1 and lr.class_weight == 'balanced'
    d = fi.direccion(model, 'u')
    w = lr.coef_[0] / sc.scale_
    np.testing.assert_allclose(d, w / np.linalg.norm(w))
    np.testing.assert_allclose(np.linalg.norm(fi.direccion(model, 'estandarizado')), 1.0)
    p, _, _ = fi.recorrido_direccion(u, d, 'proyeccion_train', 7)
    assert np.all(np.diff(m.p_dep_of(model, p)) > 0)


def test_verificar_pdep_val(tmp_path):
    rng = np.random.default_rng(9)
    u, y = rng.random((100, 5)), rng.integers(0, 2, 100)
    model, m = fi.ajustar_logreg(u, y)
    uv = rng.random((4, 5))
    ids = [f'val_{i:04d}' for i in range(4)]
    p = m.p_dep_of(model, uv)
    csvp = tmp_path / 'pred.csv'
    with open(csvp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'p_dep'])
        for i, x in zip(ids, p):
            w.writerow([i, 'val', repr(float(x))])
        w.writerow(['test_0000', 'test', '0.5'])                   # prueba se ignora
    assert fi.verificar_pdep_val(model, m, uv, ids, 1e-12, csvp) == 0.0
    with pytest.raises(Detener):
        fi.verificar_pdep_val(model, m, uv + 0.3, ids, 1e-9, csvp)


def test_check_resoluciones_f48():
    assert fi.check_resoluciones(dict(RES_F48)) == RES_F48
    with pytest.raises(Detener):
        fi.check_resoluciones({**RES_F48, 'paleta': 'jet'})


# ── F4.8: renderizado determinista ────────────────────────────────────────
@pytest.mark.parametrize('tech', ['gray_scott', 'perlin', 'lorenz'])
def test_render_determinista(tech):
    u = np.array([0.4, 0.5, 0.3, 0.0, 0.0])                      # u3 = u4 = 0: el render más barato
    a, b = fi.render(tech, u), fi.render(tech, u)
    assert a.shape == (256, 256) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert a.std() > 0
