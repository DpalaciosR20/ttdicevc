"""
Pruebas sintéticas de F4.2 (src/fase4/regimen.py y src/fase4/reintentos_gs.py).
No usan datos reales ni TT_DATA_ROOT.

  python -m pytest tests/test_fase4_regimen.py -v
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'src' / 'generation'), str(REPO / 'src' / 'training')):
    if p not in sys.path:
        sys.path.insert(0, p)

import gray_scott_lib as gsl  # noqa: E402
import lorenz_lib  # noqa: E402
from src.fase4 import regimen as rg  # noqa: E402
from src.fase4 import reintentos_gs as rgs  # noqa: E402

RES = {'fisher_alternativa': 'two-sided', 'mw_continuidad': True, 'reintento_evento': 'reintentos_mayor_0',
       'f1_estrato_sin_clase': 'cero', 'bit_a_bit': 'sha256_archivo',
       'no_identicas': 'incluir_y_listar_con_conteo_por_clase', 'pearson': 'sin_superposicion_sin_fuente_versionada'}


@pytest.fixture(scope='module')
def U():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(400, 5)).astype(np.float32)


# ── Mapeos ────────────────────────────────────────────────────────────────
def test_mapeos_coinciden_con_el_plan(U):
    mp = rg.mapear(U)
    U64 = U.astype(np.float64)
    assert np.allclose(mp['f'], 0.022 + 0.024 * U64[:, 0], rtol=0, atol=1e-6)   # u en float32, como el pkl: error de redondeo ~1e-8
    assert np.allclose(mp['k'], 0.059 + 0.007 * U64[:, 1], rtol=0, atol=1e-6)   # u en float32, como el pkl: error de redondeo ~1e-8
    assert np.allclose(mp['rho'], 24 + 8 * U64[:, 1], rtol=0, atol=1e-6)   # u en float32, como el pkl: error de redondeo ~1e-8
    assert list(mp['octavas']) == [int(1 + u * (8 - 1)) for u in U[:, 4]]   # int(lerp(u4, 1, 8))


def test_mapeos_son_los_de_src_generation(U):
    pytest.importorskip('noise')
    import perlin_lib
    mp = rg.mapear(U)
    for i, u in enumerate(U):
        p = gsl.umap_to_rd_params(u)
        assert (mp['f'][i], mp['k'][i]) == (p['f'], p['k'])
        assert mp['rho'][i] == lorenz_lib.map_umap_to_lorenz(u)['rho']
        assert mp['octavas'][i] == perlin_lib.map_umap_to_perlin(u)['n_octaves']
    assert lorenz_lib.PARAM_RANGES['rho'] == (24.0, 32.0)
    assert perlin_lib.PARAM_RANGES['n_octaves'] == (1, 8)


def test_rangos_de_octavas_y_rho():
    pytest.importorskip('noise')
    extremos = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.float64)
    mp = rg.mapear(extremos)
    assert list(mp['octavas']) == [1, 8]
    assert list(mp['rho']) == [24.0, 32.0]


# ── Estratos de Lorenz ────────────────────────────────────────────────────
def test_estrato_lorenz_equivale_a_u1(U):
    u1 = U[:, 1].astype(np.float64)
    lejos = np.abs(u1 - rg.LORENZ_U1_UMBRAL) > 1e-6
    bajo = rg.estrato_lorenz(rg.mapear(U)['rho'])
    assert np.array_equal(bajo[lejos], (u1 < rg.LORENZ_U1_UMBRAL)[lejos])
    assert rg.estrato_lorenz([24.0, 24.7399, 24.74, 25.0]).tolist() == [True, True, False, False]


def test_f1_por_estrato_y_clase_ausente():
    y = np.array([1, 0, 1, 0, 0, 0])
    yhat = np.array([1, 0, 0, 0, 0, 1])
    bajo = np.array([True, True, True, False, False, False])
    r = rg.f1_por_estrato(y, yhat, bajo, 'cero')
    from sklearn.metrics import f1_score
    assert r['rho_menor_24.74']['f1_macro'] == pytest.approx(f1_score(y[:3], yhat[:3], average='macro'))
    assert r['rho_menor_24.74']['n'] == 3 and r['rho_menor_24.74']['errores'] == 1
    # Estrato alto: solo clase 0 en y; ŷ predice un 1 → F1 de la clase 1 = 0 (definido, no 0/0)
    assert r['rho_mayor_igual_24.74']['f1_macro'] == pytest.approx(
        f1_score(y[3:], yhat[3:], average='macro', labels=[0, 1], zero_division=0))
    # Clase ausente en y y en ŷ → 0/0
    assert rg.f1_macro([0, 0], [0, 0], 'cero') == pytest.approx(0.5)
    assert rg.f1_macro([0, 0], [0, 0], 'no_definido') is None
    with pytest.raises(rg.Detener):
        rg.f1_macro([0, 0], [0, 0], 'otra')


# ── Fisher ────────────────────────────────────────────────────────────────
def _fisher_manual(t):
    """Bilateral: suma de las probabilidades hipergeométricas ≤ la observada (con margen relativo)."""
    (a, b), (c, d) = t
    n1, n0, k = a + b, c + d, a + c
    lo, hi = max(0, k - n0), min(k, n1)
    pmf = {x: math.comb(n1, x) * math.comb(n0, k - x) / math.comb(n1 + n0, k) for x in range(lo, hi + 1)}
    return sum(p for p in pmf.values() if p <= pmf[a] * (1 + 1e-7))


@pytest.mark.parametrize('t', [[[3, 6], [8, 1]], [[0, 10], [5, 5]], [[12, 340], [20, 536]], [[4, 4], [4, 4]]])
def test_fisher_contra_scipy_y_manual(t):
    r = rg.fisher(t, 'two-sided')
    assert r['p'] == pytest.approx(stats.fisher_exact(t).pvalue, rel=1e-12)
    assert r['p'] == pytest.approx(_fisher_manual(t), rel=1e-9)
    for alt in ('greater', 'less'):
        assert rg.fisher(t, alt)['p'] == pytest.approx(stats.fisher_exact(t, alternative=alt).pvalue, rel=1e-12)


def test_tabla_2x2_orientacion():
    ev = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    y = np.array([1, 1, 1, 0, 0, 0])
    assert rg.tabla_2x2(ev, y) == [[2, 1], [1, 2]]
    with pytest.raises(rg.Detener):
        rg.tabla_2x2(ev, [0, 1, 2, 0, 1, 0])


# ── Mann–Whitney ──────────────────────────────────────────────────────────
def _mw_manual(x1, x0, cc):
    """U de x1 por conteo de pares y p asintótico bilateral con corrección por empates."""
    U1 = sum((a > b) + 0.5 * (a == b) for a in x1 for b in x0)
    n1, n0 = len(x1), len(x0)
    n = n1 + n0
    _, counts = np.unique(np.concatenate([x1, x0]), return_counts=True)
    s = math.sqrt(n1 * n0 / 12 * ((n + 1) - (counts ** 3 - counts).sum() / (n * (n - 1))))
    U = max(U1, n1 * n0 - U1)
    z = (U - n1 * n0 / 2 - (0.5 if cc else 0)) / s
    return U1, min(1.0, 2 * stats.norm.sf(z))


@pytest.mark.parametrize('cc', [True, False])
def test_mann_whitney_contra_scipy_y_manual(cc):
    rng = np.random.default_rng(1)
    x1 = rng.integers(1, 9, 60).astype(float)          # con empates, como las octavas
    x0 = rng.integers(2, 9, 90).astype(float)
    r = rg.mann_whitney(x1, x0, cc)
    ref = stats.mannwhitneyu(x1, x0, alternative='two-sided', method='asymptotic', use_continuity=cc)
    assert (r['U_dep'], r['p']) == (pytest.approx(ref.statistic), pytest.approx(ref.pvalue, rel=1e-12))
    U1, p = _mw_manual(x1, x0, cc)
    assert r['U_dep'] == pytest.approx(U1)
    assert r['p'] == pytest.approx(p, rel=1e-9)
    assert -1 <= r['efecto_rbc'] <= 1


def test_mann_whitney_sin_diferencia_y_separacion_total():
    x = np.arange(20.0)
    assert rg.mann_whitney(x, x, True)['p'] == pytest.approx(1.0)
    r = rg.mann_whitney(np.arange(30.0) + 100, np.arange(30.0), False)
    assert r['efecto_rbc'] == pytest.approx(1.0) and r['p'] < 1e-9


# ── Holm ──────────────────────────────────────────────────────────────────
def test_holm_familia_y_ajustados():
    p = dict(zip(rg.FAMILIA, (0.01, 0.04, 0.03, 0.20)))
    h = rg.familia_holm(p)
    # Ordenados: 0.01·4 = 0.04, 0.03·3 = 0.09, 0.04·2 = 0.08 → max acumulado 0.09, 0.20·1 → 0.20
    assert h['lorenz_fisher_regimen']['p_holm'] == pytest.approx(0.04)
    assert h['perlin_mw_octavas']['p_holm'] == pytest.approx(0.09)
    assert h['lorenz_mw_rho']['p_holm'] == pytest.approx(0.09)
    assert h['gray_scott_fisher_reintentos']['p_holm'] == pytest.approx(0.20)
    for k in rg.FAMILIA:
        assert h[k]['reject'] == (h[k]['p_holm'] <= rg.ALPHA)
    with pytest.raises(rg.Detener):
        rg.familia_holm({'lorenz_mw_rho': 0.01})


def test_holm_contra_statsmodels_si_esta():
    sm = pytest.importorskip('statsmodels.stats.multitest')
    rng = np.random.default_rng(3)
    for _ in range(50):
        ps = rng.uniform(0, 0.1, 4)
        h = rg.holm_ajustado(dict(zip(rg.FAMILIA, ps)))
        ref = sm.multipletests(ps, alpha=0.05, method='holm')[1]
        assert np.allclose([h[k] for k in rg.FAMILIA], ref)


# ── Resoluciones ──────────────────────────────────────────────────────────
def test_resoluciones_pendientes_detienen():
    with pytest.raises(rg.Detener, match='PENDIENTE'):
        rg.check_resoluciones({})
    with pytest.raises(rg.Detener, match='no previstos'):
        rg.check_resoluciones({**RES, 'bit_a_bit': 'aprox'})
    assert rg.check_resoluciones(RES) == RES
    with pytest.raises(rg.Detener, match='PENDIENTE'):
        rg.check_resoluciones({k: v for k, v in RES.items() if k != 'no_identicas'})
    with pytest.raises(rg.Detener, match='no previstos'):
        rg.check_resoluciones({**RES, 'no_identicas': 'excluir'})
    with pytest.raises(rg.Detener, match='no previstos'):
        rg.check_resoluciones({**RES, 'pearson': 'superponer'})
    with pytest.raises(rg.Detener, match='claves no previstas'):
        rg.check_resoluciones({**RES, 'otra_cosa': 1})


def test_resoluciones_del_repositorio_y_metadatos():
    import json
    raw = json.loads((REPO / 'results' / 'f42_regimen' / 'resoluciones.json').read_text(encoding='utf-8'))
    res = rg.check_resoluciones(raw)                    # claves «_» aceptadas como metadatos
    assert set(res) == set(rg.RESOLUCIONES) and '_autor' not in res
    meta = rg.meta_resoluciones(raw)
    assert meta['autor'] == 'SV' and meta['fecha'] == '2026-11-10' and 'nota' in meta


# ── Análisis completo con datos sintéticos ────────────────────────────────
def _filas(y, reint, sha_ok=None):
    sha_ok = [True] * len(y) if sha_ok is None else sha_ok
    return [{'sample_id': f'train_{i:04d}', 'y_true': str(int(c)), 'reintentos': str(r), 'usa_respaldo': str(r == 3),
             'sha256_coincide': str(s), 'pixeles_coinciden': 'True'} for i, (c, r, s) in enumerate(zip(y, reint, sha_ok))]


def test_analizar_familia_y_estructura(U):
    pytest.importorskip('noise')
    rng = np.random.default_rng(5)
    y = (U[:, 1] > 0.5).astype(int)                     # ρ distinto por clase
    reint = rng.choice([0, 0, 0, 1, 3], size=len(y))
    sha_ok = [True] * len(y)
    sha_ok[7] = False
    U_te, y_te = U[:60], y[:60]
    r = rg.analizar(U, y, _filas(y, reint, sha_ok), U_te, y_te, y_te, RES)
    assert set(r['holm']['familia']) == set(rg.FAMILIA) and r['holm']['m'] == 4
    assert r['pruebas']['lorenz_mw_rho']['p'] < 1e-10
    assert r['gray_scott']['reintentos']['n_reintentos_mayor_0'] == int((reint > 0).sum())
    assert r['gray_scott']['reintentos']['n_respaldo'] == int((reint == 3).sum())
    assert r['gray_scott']['bit_a_bit']['no_identicas'] == ['train_0007']
    assert r['gray_scott']['bit_a_bit']['no_identicas_por_clase'] == {'dep': int(y[7] == 1), 'nodep': int(y[7] == 0)}
    assert r['gray_scott']['bit_a_bit']['tratamiento_no_identicas'] == 'incluir_y_listar_con_conteo_por_clase'
    # P6: las no idénticas se incluyen en el Fisher de reintentos (la tabla suma las 400)
    assert sum(map(sum, r['pruebas']['gray_scott_fisher_reintentos']['tabla'])) == len(y)
    est = r['lorenz_estratos_prueba']
    assert est['rho_menor_24.74']['n'] + est['rho_mayor_igual_24.74']['n'] == 60
    assert est['rho_mayor_igual_24.74']['errores'] == 0
    assert r['lorenz']['discordancias_rho_vs_u1'] == 0
    # Con el evento «usa_respaldo» la tabla cambia
    r2 = rg.analizar(U, y, _filas(y, reint), U_te, y_te, y_te, {**RES, 'reintento_evento': 'usa_respaldo'})
    t = r2['pruebas']['gray_scott_fisher_reintentos']['tabla']
    assert t[0][0] + t[1][0] == int((reint == 3).sum())


def test_analizar_detiene_si_el_registro_no_cuadra(U):
    y = np.zeros(len(U), dtype=int)
    with pytest.raises(rg.Detener, match='filas'):
        rg.analizar(U, y, _filas(y[:-1], [0] * (len(y) - 1)), U[:5], y[:5], y[:5], RES)
    filas = _filas(y, [0] * len(y))
    filas[0]['y_true'] = '1'
    with pytest.raises(rg.Detener, match='alineado'):
        rg.analizar(U, y, filas, U[:5], y[:5], y[:5], RES)


# ── Instrumentación de Gray-Scott ─────────────────────────────────────────
def test_registro_no_altera_la_salida():
    kw = dict(f=0.03, k=0.06, Du=0.16, Dv=0.06, seed_radius=4, n_iter=120, size=32)
    V0, i0 = gsl.run_gray_scott_safe(**kw)
    with rgs.registro() as llamadas:
        V1, i1 = gsl.run_gray_scott_safe(**kw)
    assert np.array_equal(V0, V1) and i0 == i1
    assert len(llamadas) == (rgs.MAX_RETRIES + 1 if i1 == rgs.MAX_RETRIES else i1 + 1)
    assert gsl.run_gray_scott is not None and gsl.run_gray_scott.__name__ == 'run_gray_scott'   # restaurada


def test_registro_misma_imagen_por_la_ruta_de_produccion(tmp_path):
    """Mismo PNG (sha256) con y sin registro, por generate_and_save_grayscale (≈ 3–8 s: 256², ≥ 2000 iteraciones)."""
    u = np.array([0.4, 0.5, 0.5, 0.2, 0.0], dtype=np.float32)       # n_iter = 2000, el mínimo
    f0, r0, _ = gsl.generate_and_save_grayscale(3, u, 1, 'train', str(tmp_path / 'sin'))
    f1, r1, _, respaldo, llamadas = rgs.regenerar(3, u, 1, tmp_path / 'con')
    assert rgs.sha256_file(f0) == rgs.sha256_file(f1) and r0 == r1
    assert Path(f1).name == 'rd_train_0003_label1.png'
    assert respaldo == (r1 == rgs.MAX_RETRIES) and len(llamadas) >= 1


def test_registro_detecta_reintentos_y_respaldo(monkeypatch):
    original = gsl.run_gray_scott
    estado = {'n': 0}

    def plano_tres_veces(**kw):          # varianza 0 en los intentos → fuerza el respaldo
        estado['n'] += 1
        if kw.get('f') == rgs.RESPALDO['f'] and kw.get('n_iter') == rgs.RESPALDO['n_iter']:
            return original(**{**kw, 'n_iter': 30, 'size': 16})
        return np.zeros((16, 16))

    monkeypatch.setattr(gsl, 'run_gray_scott', plano_tres_veces)
    with rgs.registro() as llamadas:
        V, intento = gsl.run_gray_scott_safe(f=0.03, k=0.06, Du=0.16, Dv=0.08, seed_radius=4, n_iter=10, size=16)
    assert intento == rgs.MAX_RETRIES and len(llamadas) == rgs.MAX_RETRIES + 1
    assert rgs.coherencia(intento, llamadas) is True
    # Dv y el radio cambian de forma determinista entre intentos (lógica intacta)
    assert [c['Dv'] for c in llamadas[:3]] == pytest.approx([0.08, 0.068, 0.0578])
    assert [c['seed_radius'] for c in llamadas[:3]] == [4, 8, 12]
    assert gsl.run_gray_scott is plano_tres_veces   # registro restaura lo que encontró


def test_coherencia_detiene_si_no_cuadra():
    with pytest.raises(rgs.Detener):
        rgs.coherencia(1, [{'var': 1.0}])
    with pytest.raises(rgs.Detener, match='respaldo'):
        rgs.coherencia(rgs.MAX_RETRIES, [{'var': 0.0}] * 3 + [{'f': 0.02, 'var': 1.0}])


def test_respaldo_coincide_con_el_generador():
    import inspect
    src = inspect.getsource(gsl.run_gray_scott_safe)
    assert 'run_gray_scott(f=0.035, k=0.061, Du=0.16, Dv=0.06, seed_radius=10, n_iter=5000' in src
    assert rgs.MAX_RETRIES == 3


# ── Negativa a sobrescribir ───────────────────────────────────────────────
def test_reintentos_no_sobrescribe(tmp_path):
    out, regen = tmp_path / 'out', tmp_path / 'regen'
    rgs.check_dirs(out, regen)                       # nada existe: pasa
    regen.mkdir()
    rgs.check_dirs(out, regen)                       # vacío: pasa
    (regen / 'x.png').write_bytes(b'')
    with pytest.raises(rgs.Detener, match='no está vacío'):
        rgs.check_dirs(out, regen)
    out.mkdir()
    (out / rgs.CSV_NAME).write_text('x')
    with pytest.raises(rgs.Detener, match='ya existe'):
        rgs.check_dirs(out, tmp_path / 'otro')


def test_regimen_no_sobrescribe(tmp_path):
    (tmp_path / 'resultados.json').write_text('{}')
    with pytest.raises(rg.Detener, match='ya existe'):
        rg.refuse_overwrite(tmp_path / 'resultados.json')
    with pytest.raises(rg.Detener, match='ya existe'):
        rg.main(['--resoluciones', str(_res_file(tmp_path)), '--out', str(tmp_path)])


def _res_file(tmp_path):
    import json
    p = tmp_path / 'res.json'
    p.write_text(json.dumps(RES))
    return p


def test_csv_de_reintentos_columnas(tmp_path):
    fila = {c: '' for c in rgs.COLUMNAS}
    rgs.write_csv(tmp_path / 'r.csv', [fila])
    assert (tmp_path / 'r.csv').read_text(encoding='utf-8').splitlines()[0].split(',') == rgs.COLUMNAS


def test_resumen_por_clase():
    filas = [{'y_true': c, 'reintentos': r, 'usa_respaldo': r == 3, 'sha256_coincide': True, 'pixeles_coinciden': True,
              'almacenada_integra': True, 'sample_id': f's{i}'} for i, (c, r) in enumerate([(0, 0), (0, 1), (1, 3), (1, 0)])]
    s = rgs.resumen(filas)
    assert s['reintentos_mayor_0_por_clase'] == {0: 1, 1: 1}
    assert s['usa_respaldo_por_clase'] == {0: 0, 1: 1}
    assert s['distribucion_reintentos'] == {'0': 2, '1': 1, '2': 0, '3': 1}


def test_tareas_muestreo_de_humo():
    sel = np.unique(np.linspace(0, 907, 16).round().astype(int))
    assert len(sel) == 16 and sel[0] == 0 and sel[-1] == 907


def test_permutaciones_son_independientes_del_orden():
    """La tabla 2×2 y el Fisher no dependen del orden de las muestras."""
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 100)
    ev = rng.random(100) < 0.2
    perm = rng.permutation(100)
    assert rg.tabla_2x2(ev, y) == rg.tabla_2x2(ev[perm], y[perm])
    assert list(itertools.chain(*rg.tabla_2x2(ev, y))) and sum(itertools.chain(*rg.tabla_2x2(ev, y))) == 100


def test_trabajo_verifica_contra_la_almacenada(tmp_path):
    """_trabajo regenera, compara sha256 y píxeles con la «almacenada» y arma la fila del CSV (≈ 15 s)."""
    u = np.array([0.4, 0.5, 0.5, 0.2, 0.0], dtype=np.float32)
    stored, _, _ = gsl.generate_and_save_grayscale(5, u, 0, 'train', str(tmp_path / 'almacen'))
    sha = rgs.sha256_file(stored)
    fila = rgs._trabajo(('train_0005', 5, 0, u, str(tmp_path / 'regen'), stored, sha))
    assert set(fila) == set(rgs.COLUMNAS)
    assert fila['sha256_coincide'] and fila['pixeles_coinciden'] and fila['almacenada_integra']
    assert fila['archivo'] == 'rd_train_0005_label0.png' and fila['n_iter'] == 2000
    # Una «almacenada» distinta: no coincide y no se oculta
    from PIL import Image
    arr = np.array(Image.open(stored))
    arr[0, 0] ^= 1
    otra = tmp_path / 'otra.png'
    Image.fromarray(arr).save(otra)
    fila2 = rgs._trabajo(('train_0005', 5, 0, u, str(tmp_path / 'regen2'), str(otra), sha))
    assert fila2['sha256_coincide'] and not fila2['pixeles_coinciden'] and not fila2['almacenada_integra']


def test_figuras_sinteticas(U, tmp_path):
    pytest.importorskip('noise')
    y = (U[:, 0] > 0.5).astype(int)
    r = rg.analizar(U, y, _filas(y, [0] * len(y)), U[:40], y[:40], y[:40], RES)
    paths = rg.figuras(r, tmp_path)
    assert [Path(p).name for p in paths] == ['f42_01_gray_scott_fk.png', 'f42_02_lorenz_rho.png', 'f42_03_perlin_octavas.png']
    assert all(Path(p).stat().st_size > 10_000 for p in paths)
    assert r['gray_scott']['pearson_superpuesto'] is False
    assert r['gray_scott']['pearson_resolucion'] == 'sin_superposicion_sin_fuente_versionada'
