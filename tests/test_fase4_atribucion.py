"""F4.3 (atribución): pruebas sintéticas. Sin datos reales ni TT_DATA_ROOT."""
import builtins
import io
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / 'src' / 'training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.fase4 import atribucion_comun as ac  # noqa: E402
from src.fase4 import atribucion_gpu as ag  # noqa: E402
from src.fase4 import atribucion_local as al  # noqa: E402

RES = {'clase_predicha': 'yhat_media_5_semillas', 'escala_entre_semillas': 'sin_reescalar',
       'umbral_constantes_unidad': 'agrupado', 'l2_resolucion': '224', 'perm_p': 'k_mas_1_sobre_B_mas_1',
       'perm_rng': 'por_tecnica', 'similitud_constantes': 'excluir', 'sobre_maximo': 'como_dentro',
       'criterio_semillas': 'mediana_de_semillas', 'ssim_resolucion': '56',
       'gs_corr_pdep': 'f36_prueba_media_real', 'gs_corr_mapas': 'spearman56_mediana',
       'coseno_lectura': 'w1_menos_w0', 'vit_ln_final': 'con_cabeza', 'vit_embeddings': 'solo_conv_proj',
       'mha_in_proj': 'reset_mha_completo', 'semilla_cascada': 'una_vez_por_modelo',
       'ablacion_borde': 'reflect', 'ablacion_rango': 'recortar_redondear'}


# ── Resoluciones ──────────────────────────────────────────────────────────
def test_resoluciones():
    assert set(RES) == set(ac.RESOLUCIONES)
    assert ac.check_resoluciones({**RES, '_autor': 'SV'}) == RES
    with pytest.raises(ac.Detener, match='PENDIENTE'):
        ac.check_resoluciones({})
    with pytest.raises(ac.Detener, match='no previstos'):
        ac.check_resoluciones({**RES, 'perm_p': 'otro'})
    with pytest.raises(ac.Detener, match='claves no previstas'):
        ac.check_resoluciones({**RES, 'x': 1})
    assert ac.meta_resoluciones({**RES, '_autor': 'SV'}) == {'autor': 'SV'}


# ── Similitud ─────────────────────────────────────────────────────────────
def test_pool_y_constantes():
    m = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8)
    p = ac.pool(m, 4)
    assert p.shape == (2, 4, 4) and p[0, 0, 0] == np.mean([0, 1, 8, 9])
    with pytest.raises(ac.Detener):
        ac.pool(np.zeros((1, 10, 10)), 4)
    assert list(ac.es_constante(np.stack([np.zeros((4, 4)), np.eye(4)]))) == [True, False]
    n = ac.minmax(np.stack([np.zeros((4, 4)), 3 * np.eye(4)]))
    assert n[0].max() == 0 and n[1].max() == 1


def test_spearman_contra_scipy_y_constante():
    rng = np.random.default_rng(0)
    a, b = rng.random((5, 56, 56)), rng.random((5, 56, 56))
    b[1] = a[1] ** 3                                   # monótona: 1
    s = ac.spearman_filas(a, b)
    for i in range(5):
        assert s[i] == pytest.approx(stats.spearmanr(a[i].ravel(), b[i].ravel()).statistic, abs=1e-12)
    assert s[1] == pytest.approx(1.0)
    a[2] = 0.0
    assert np.isnan(ac.spearman_filas(a, b)[2])
    # a 224 se promedia a 56 antes de los rangos
    big = rng.random((2, 224, 224))
    assert ac.spearman_filas(big, big)[0] == pytest.approx(1.0)


def test_ssim_contra_skimage():
    sk = pytest.importorskip('skimage.metrics')
    rng = np.random.default_rng(1)
    a = rng.random((3, 56, 56))
    b = np.clip(a + 0.2 * rng.standard_normal((3, 56, 56)), 0, 1)
    s = ac.ssim_filas(a, b)
    for i in range(3):
        ref = sk.structural_similarity(a[i], b[i], data_range=1.0, gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False)
        assert s[i] == pytest.approx(ref, abs=1e-10)
    assert ac.ssim_filas(a, a)[0] == pytest.approx(1.0)
    big = rng.random((1, 224, 224))
    assert ac.ssim_filas(big, big, g=56)[0] == pytest.approx(1.0)


def test_resumen_similitud():
    s = [0.5, np.nan, 0.9, 0.1]
    assert ac.resumen_similitud(s, 'excluir') == (0.5, 3, 1)
    assert ac.resumen_similitud(s, 'cero') == (pytest.approx(0.3), 4, 1)
    m, n, k = ac.resumen_similitud([np.nan], 'excluir')
    assert np.isnan(m) and (n, k) == (0, 1)


# ── Criterio relativo ─────────────────────────────────────────────────────
def test_pares_y_clasificar():
    assert len(ac.pares_semillas()) == 10
    ref = [0.6, 0.7, 0.8]
    assert ac.clasificar(0.5, ref, 'como_dentro') == 'debajo'
    assert ac.clasificar(0.6, ref, 'como_dentro') == 'dentro'      # el mínimo está dentro
    assert ac.clasificar(0.8, ref, 'indeterminado') == 'dentro'
    assert ac.clasificar(0.9, ref, 'como_dentro') == 'dentro'
    assert ac.clasificar(0.9, ref, 'indeterminado') == 'indeterminado'
    assert ac.clasificar(np.nan, ref, 'como_dentro') == 'indeterminado'


def test_combinar_semillas():
    ref = [0.6, 0.8]
    v = [0.1, 0.2, 0.7, 0.3, 0.75]                       # 3 debajo, 2 dentro; mediana 0.3
    assert ac.combinar_semillas(v, ref, 'mediana_de_semillas', 'como_dentro')['global'] == 'debajo'
    assert ac.combinar_semillas(v, ref, 'mayoria', 'como_dentro')['global'] == 'debajo'
    assert ac.combinar_semillas(v, ref, 'unanimidad', 'como_dentro')['global'] == 'indeterminado'
    assert ac.combinar_semillas([0.7] * 5, ref, 'unanimidad', 'como_dentro')['global'] == 'dentro'
    assert ac.combinar_semillas([0.1, 0.7, np.nan, 0.9, 0.95], ref, 'mayoria', 'indeterminado')['global'] == 'indeterminado'


def test_criterio_cambia_y_no_cambia():
    """Semillas reales parecidas entre sí; un mapa aleatorizado distinto «cambia»;
    uno idéntico «no cambia»."""
    rng = np.random.default_rng(2)
    base = rng.random((20, 56, 56))
    reales = {s: base + 0.1 * rng.random((20, 56, 56)) for s in ac.SEEDS}
    ref = ac.referencia_mapas(reales, ac.spearman_filas, 'excluir')
    assert len(ref) == 10 and min(ref.values()) > 0.8
    otro = {s: ac.resumen_similitud(ac.spearman_filas(reales[s], rng.random((20, 56, 56))), 'excluir')[0] for s in ac.SEEDS}
    mismo = {s: ac.resumen_similitud(ac.spearman_filas(reales[s], reales[s]), 'excluir')[0] for s in ac.SEEDS}
    assert ac.criterio_similitud(otro, ref, RES)['global'] == 'cambia'
    assert ac.criterio_similitud(mismo, ref, RES)['global'] == 'no_cambia'          # por encima del máximo
    assert ac.criterio_similitud(mismo, ref, {**RES, 'sobre_maximo': 'indeterminado'})['global'] == 'indeterminado'


# ── Agregación ────────────────────────────────────────────────────────────
def test_medias_clase_predicha_y_escala():
    rng = np.random.default_rng(3)
    maps = {s: rng.random((6, 4, 4)) * (i + 1) for i, s in enumerate(ac.SEEDS)}
    y = np.array([0, 0, 0, 1, 1, 1])
    m, n = ac.medias_por_clase(ac.media_entre_semillas(maps), y)
    assert n == {0: 3, 1: 3}
    mg, _ = ac.medias_agrupadas(maps, {s: y for s in ac.SEEDS})
    assert np.allclose(m[1], mg[1])                                   # clase real: lo mismo
    p = {s: np.array([0.2, 0.6, 0.4, 0.7, 0.55, 0.1]) for s in ac.SEEDS}
    p[42] = np.array([0.9] * 6)
    cm = ac.clase_predicha(p, 'yhat_media_5_semillas')
    assert all((cm[s] == cm[42]).all() for s in ac.SEEDS)
    assert list(ac.clase_predicha(p, 'yhat_por_semilla')[42]) == [1] * 6
    esc, f = ac.escalar_semillas(maps, 'dividir_media_semilla')
    assert all(np.mean(esc[s]) == pytest.approx(1.0) for s in ac.SEEDS)
    assert ac.escalar_semillas(maps, 'sin_reescalar')[1] == {s: 1.0 for s in ac.SEEDS}


def test_constantes_y_sensibilidad():
    y = np.array([0] * 12 + [1] * 12)
    const = {s: np.array([True] * 10 + [False] * 2 + [False] * 12) for s in ac.SEEDS}
    norm = {s: np.ones((24, 4, 4)) for s in ac.SEEDS}
    fr = ac.fraccion_constantes(const, {s: y for s in ac.SEEDS})
    assert fr['agrupada']['0'] == {'n': 60, 'constantes': 50, 'fraccion': 50 / 60}
    assert fr['por_semilla']['42']['1']['constantes'] == 0
    ag_ = ac.sensibilidad_sin_constantes(norm, const, {s: y for s in ac.SEEDS}, 'agrupado')
    assert ag_['n_no_constantes'] == {0: 10, 1: 60} and ag_['medias'][0] is not None
    ps = ac.sensibilidad_sin_constantes(norm, const, {s: y for s in ac.SEEDS}, 'por_semilla')
    assert ps['medias'][0] is None and ps['semillas_usadas'] == {0: 0, 1: 5}      # 2 < 10 por semilla


# ── Perfil radial y terciles ──────────────────────────────────────────────
def test_anillos_y_perfil_radial():
    idx = ac.anillos()
    assert idx.max() == 27 and idx.min() == -1
    assert (idx == idx[::-1, ::-1]).all() and (idx == idx.T).all()               # simétrico respecto del centro
    c = (224 - 1) / 2
    yy, xx = np.mgrid[0:224, 0:224]
    r = np.hypot(yy - c, xx - c)
    prof = ac.perfil_radial(r, idx)
    assert len(prof) == 28
    assert all(4 * i <= v < 4 * i + 4 for i, v in enumerate(prof))
    assert (r[idx == -1] >= 112).all() and (r[idx >= 0] < 112).all()


def test_terciles():
    u4_train = np.linspace(0, 1, 301)
    cortes = ac.cortes_terciles(u4_train)
    assert cortes == pytest.approx([1 / 3, 2 / 3])
    t = ac.tercil([0.0, cortes[0], cortes[0] + 1e-9, cortes[1], 0.99], cortes)
    assert list(t) == [0, 0, 1, 1, 2]


# ── Permutación y Holm ────────────────────────────────────────────────────
def test_estadistico_y_permutacion():
    rng = np.random.default_rng(4)
    y = np.array([0] * 30 + [1] * 20)
    maps = rng.normal(size=(50, 8, 8))
    d = maps[y == 1].mean(0) - maps[y == 0].mean(0)
    assert ac.estadistico_l2(maps, y) == pytest.approx(np.sqrt((d ** 2).sum()))
    fuerte = maps.copy()
    fuerte[y == 1] += 2.0
    r = ac.prueba_permutacion(fuerte, y, np.random.default_rng(2026), B=200)
    assert r['k'] == 0 and r['p'] == pytest.approx(1 / 201)
    assert ac.prueba_permutacion(fuerte, y, np.random.default_rng(2026), B=200, regla='k_sobre_B')['p'] == 0.0
    nula = ac.prueba_permutacion(maps, y, np.random.default_rng(2026), B=200)
    assert nula['p'] > 0.05
    assert nula == ac.prueba_permutacion(maps, y, np.random.default_rng(2026), B=200)          # reproducible
    # La identidad cuenta como T ≥ T_obs (tolerancia relativa)
    class Ident:
        def permutation(self, v):
            return np.array(v)
    assert ac.prueba_permutacion(maps, y, Ident(), B=3)['k'] == 3


def test_rngs_permutacion():
    a = ac.rngs_permutacion('por_tecnica')
    assert a['gray_scott'] is not a['perlin']
    assert a['gray_scott'].integers(0, 10 ** 9) == a['perlin'].integers(0, 10 ** 9)
    b = ac.rngs_permutacion('secuencial')
    assert b['gray_scott'] is b['lorenz']


def test_holm_y_p_ajustado():
    h = ac.holm({'gray_scott': 0.01, 'perlin': 0.04, 'lorenz': 0.03})
    assert h['gray_scott']['p_holm'] == pytest.approx(0.03)
    assert h['lorenz']['p_holm'] == pytest.approx(0.06) and h['perlin']['p_holm'] == pytest.approx(0.06)
    assert [h[t]['reject'] for t in ('gray_scott', 'lorenz', 'perlin')] == [True, False, False]
    assert all(v['reject'] == (v['p_holm'] <= 0.05) for v in h.values())


# ── Cosenos y lectura de los ViT ─────────────────────────────────────────
def test_cosenos_cabezas():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(2, 16))
    reales = {s: base + 0.1 * rng.normal(size=(2, 16)) for s in ac.SEEDS}
    alto = ac.cosenos_cabezas(reales, {s: reales[s] for s in ac.SEEDS}, RES)
    assert alto['w1']['nivel'] == 'alto' and len(alto['w1']['referencia']) == 10
    bajo = ac.cosenos_cabezas(reales, {s: rng.normal(size=(2, 16)) for s in ac.SEEDS}, RES)
    assert bajo['w1_menos_w0']['nivel'] == 'bajo'
    assert ac.vectores_cabeza(np.array([[1., 0.], [0., 1.]]))['w1_menos_w0'].tolist() == [-1, 1]
    with pytest.raises(ac.Detener):
        ac.vectores_cabeza(np.zeros((3, 4)))
    mix = {'w1': {'nivel': 'alto'}, 'w1_menos_w0': {'nivel': 'bajo'}}
    assert ac.nivel_coseno(mix, 'w1') == 'alto' and ac.nivel_coseno(mix, 'ambos') == 'indeterminado'


@pytest.mark.parametrize('cd,cos,cab,esperado', [
    ('cambia', 'alto', None, 'cordura_por_datos_superada'),
    ('no_cambia', 'bajo', None, 'atencion_del_backbone'),
    ('no_cambia', 'alto', 'cambia', 'dependiente_de_la_cabeza_sin_validacion_por_datos'),
    ('no_cambia', 'alto', 'no_cambia', 'atencion_del_backbone'),
    ('no_cambia', 'alto', None, 'pendiente_de_la_cascada'),
    ('no_cambia', 'indeterminado', 'cambia', 'indeterminado'),
    ('indeterminado', 'alto', 'cambia', 'indeterminado'),
])
def test_lectura_vit(cd, cos, cab, esperado):
    assert ac.lectura_vit(cd, cos, cab) == esperado


def test_seguimiento_gs():
    r = ac.seguimiento_gs({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}, {1: 0.0, 2: 0.3, 3: 0.4, 4: 0.6, 5: 0.9})
    assert r['n'] == 5 and r['spearman_entre_semillas'] == pytest.approx(1.0) and r['pearson_entre_semillas'] > 0.9


# ── Local: artefactos y .npz ──────────────────────────────────────────────
def _art(tmp_path, nombre, contenido, rel):
    import hashlib
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(contenido)
    return {'logical_name': 'X', 'rel_path': rel, 'size_bytes': str(len(contenido)),
            'sha256': hashlib.sha256(contenido).hexdigest()}


def test_ubicador_rechaza_humo_y_sha(tmp_path):
    rel = 'ttdicevc/results/f37_mapas/maps/a.npz'
    row = _art(tmp_path, 'a.npz', b'bueno', rel)
    (tmp_path / 'humo' / 'f37_humo' / 'maps').mkdir(parents=True)
    (tmp_path / 'humo' / 'f37_humo' / 'maps' / 'a.npz').write_bytes(b'humo!')
    ub = al.Ubicador(tmp_path, {'MAPS[a]': row})
    p, _ = ub.ruta('MAPS[a]')                                             # elige la de la ruta del manifiesto
    assert p.replace(os.sep, '/').endswith(rel) and ub.verificados['MAPS[a]']['sha256'] == row['sha256']
    solo_humo = tmp_path / 'otro'
    (solo_humo / 'f37_humo').mkdir(parents=True)
    (solo_humo / 'f37_humo' / 'a.npz').write_bytes(b'bueno')
    with pytest.raises(ac.Detener, match='humo'):
        al.Ubicador(solo_humo, {'MAPS[a]': row}).ruta('MAPS[a]')
    malo = tmp_path / 'malo'
    (malo / 'ttdicevc/results/f37_mapas/maps').mkdir(parents=True)
    (malo / rel).write_bytes(b'malo!')
    with pytest.raises(ac.Detener, match='sha256'):
        al.Ubicador(malo, {'MAPS[a]': row}).ruta('MAPS[a]')


def _npz_sintetico(path, index, rng, vit=False, size=56, senal=1.0):
    n = {s: len(index[s]) for s in ('val', 'test')}
    sid = [r['sample_id'] for s in ('val', 'test') for r in index[s]]
    y = np.array([int(r['y_true']) for s in ('val', 'test') for r in index[s]], dtype=np.int8)
    raw = rng.random((len(sid), size, size)).astype(np.float32) + senal * y[:, None, None]
    scale = raw.reshape(len(raw), -1).max(1).astype(np.float32)
    arr = {'sample_id': np.array(sid), 'split': np.array(['val'] * n['val'] + ['test'] * n['test']), 'y_true': y,
           'p_dep': np.clip(0.3 + 0.4 * y + 0.1 * rng.standard_normal(len(y)), 0, 1),
           'map': ac.minmax(raw).astype(np.float16), 'map_raw': (raw / scale[:, None, None]).astype(np.float16),
           'map_raw_scale': scale}
    if vit:
        arr.update({'backbone_map': arr['map'], 'backbone_map_raw': arr['map_raw'], 'backbone_map_raw_scale': scale})
    np.savez_compressed(path, **arr)
    return path


def test_cargar_npz(tmp_path):
    index = al.load_index()
    rng = np.random.default_rng(6)
    p = _npz_sintetico(tmp_path / 'r.npz', index, rng, vit=True)
    rec = {'run': 'r', 'npz': {'sha256': al.sha256_file(p)}}
    d = al.cargar_npz(p, rec, index)
    assert set(d) == {'val', 'test'} and d['val']['map_crudo'].shape == (114, 56, 56) and 'backbone_crudo' in d['test']
    with pytest.raises(ac.Detener, match='sha256'):
        al.cargar_npz(p, {'run': 'r', 'npz': {'sha256': '0' * 64}}, index)
    z = dict(np.load(p))
    z['sample_id'] = z['sample_id'][::-1]
    np.savez(tmp_path / 'mal.npz', **z)
    with pytest.raises(ac.Detener, match='orden'):
        al.cargar_npz(tmp_path / 'mal.npz', {'run': 'r', 'npz': {'sha256': al.sha256_file(tmp_path / 'mal.npz')}}, index)


def _datos_sinteticos(rng, index, senal):
    def d(split, seed, perm):
        y = np.array([int(r['y_true']) for r in index[split]])
        base = rng.random((len(y), 56, 56)).astype(np.float32) * 0.1
        raw = base + senal * y[:, None, None] * np.linspace(0, 1, 56)[None, None, :] + (0.3 if perm else 0)
        raw[:3] = 0.0                                                     # 3 mapas constantes
        return {'y': y, 'p_dep': np.clip(0.3 + 0.4 * y + 0.1 * rng.standard_normal(len(y)), 0, 1),
                'sample_id': [r['sample_id'] for r in index[split]], 'map_crudo': raw,
                'map_norm': ac.minmax(raw).astype(np.float16), 'backbone_crudo': raw,
                'backbone_norm': ac.minmax(raw).astype(np.float16)}
    return {s: {sp: d(sp, s, False) for sp in ('val', 'test')} for s in ac.SEEDS}, \
           {s: {sp: d(sp, s, True) for sp in ('val', 'test')} for s in ac.SEEDS}


def test_analizar_sintetico(monkeypatch):
    monkeypatch.setattr(ac, 'PERM_B', 199)
    index = al.load_index()
    rng = np.random.default_rng(7)
    real, perm = {}, {}
    for t, senal in (('gray_scott', 1.0), ('perlin', 0.0), ('lorenz', 1.0)):
        real[t], perm[t] = _datos_sinteticos(rng, index, senal)
        if t == 'gray_scott':
            for s in ac.SEEDS:
                for sp in ('val', 'test'):
                    for k in ('backbone_crudo', 'backbone_norm'):
                        real[t][s][sp].pop(k), perm[t][s][sp].pop(k)
    W = rng.normal(size=(2, 8))
    cabezas = {t: {'real': {s: W + 0.05 * rng.normal(size=(2, 8)) for s in ac.SEEDS},
                   'perm': {s: W + 0.05 * rng.normal(size=(2, 8)) for s in ac.SEEDS}} for t in ac.VIT}
    u4 = {sp: rng.random(114) for sp in ('val', 'test')}
    cortes = ac.cortes_terciles(rng.random(908))
    cascada = {'tecnicas': {t: {'pasos': {'cabeza': {'map': {'spearman56': {'global': 'cambia'}}}}} for t in ac.VIT}}
    r, medias, filas = al.analizar(real, perm, cabezas, u4, cortes, RES, cascada)
    assert r['holm']['m'] == 3 and set(r['holm']['familia']) == set(ac.TECHNIQUES)
    assert r['tecnicas']['gray_scott']['permutacion_val']['p'] == pytest.approx(1 / 200)
    assert r['tecnicas']['perlin']['permutacion_val']['p'] > 0.05
    gs = r['tecnicas']['gray_scott']
    assert len(gs['agregacion']['val']['perfil_radial']['clase_real']['0']) == 10     # 56 px: radio máximo ≈ 39, 10 anillos de 4
    const = gs['agregacion']['val']['constantes_clase_real']['agrupada']
    assert const['0']['constantes'] + const['1']['constantes'] == 15                  # 3 por semilla
    lt = r['tecnicas']['lorenz']['agregacion']['test']['terciles_u4']
    assert sum(v['0'] + v['1'] for v in lt['conteos'].values()) == 114
    assert set(r['lectura_vit']) == set(ac.VIT)
    assert r['lectura_vit']['lorenz']['cabeza_cascada'] == 'cambia'
    assert 'principal' in gs['gs_mapas_vs_pdep'] and len(gs['gs_mapas_vs_pdep']['variantes_corr_pdep']) == 3
    cd = r['tecnicas']['perlin']['cordura_datos']
    assert set(cd) == {'map', 'backbone_map'} and len(cd['map']['val']['spearman56']['referencia_pares']) == 10
    assert any(k.startswith('lorenz|map|val|tercil2') for k in medias)
    assert {f['comparacion'] for f in filas} == {'par_real', 'real_vs_permutado'}
    json.dumps(ac.limpiar(r))                                                        # serializable


def test_local_no_sobrescribe(tmp_path):
    (tmp_path / 'resultados.json').write_text('{}')
    res = tmp_path / 'res.json'
    res.write_text(json.dumps(RES))
    with pytest.raises(ac.Detener, match='ya existe'):
        al.main(['--root', str(tmp_path), '--resoluciones', str(res), '--out', str(tmp_path)])


# ── GPU: G2 (solo val) ────────────────────────────────────────────────────
def test_g2_se_niega_a_prueba():
    with pytest.raises(ac.Detener, match='G2'):
        ag.exigir_val('test')
    from src.fase4 import caracteristicas as car
    rows = ag.filas_val(car.load_index())
    assert len(rows) == 114 and all(r['split'] == 'val' for r in rows)
    with pytest.raises(ac.Detener, match='G2'):
        ag.nombres_val([{**rows[0], 'split': 'test'}], 'gray_scott')
    with pytest.raises(ac.Detener, match='G2'):
        ag.nombres_val([{**rows[0], 'img_gray_scott': 'rd_test_0000_label0.png'}], 'gray_scott')
    with pytest.raises(SystemExit):
        ag.main(['--models-root', '.', '--resoluciones', 'x.json', '--split', 'test'])


def test_g2_solo_abre_imagenes_de_val(tmp_path, monkeypatch):
    import hashlib
    from PIL import Image
    from src.fase4 import caracteristicas as car
    rows = ag.filas_val(car.load_index())[:4]
    expected = {}
    for split, rr in (('val', rows), ('test', [{'img_perlin': f'perlin_test_000{i}_label0.png'} for i in range(4)])):
        for r in rr:
            buf = io.BytesIO()
            Image.fromarray(np.full((256, 256), 7, np.uint8), 'L').save(buf, format='PNG')
            (tmp_path / r['img_perlin']).write_bytes(buf.getvalue())
            expected[f'img/{r["img_perlin"]}'] = hashlib.sha256(buf.getvalue()).hexdigest()
    abiertos = []
    real_open = builtins.open

    def espia(f, *a, **k):
        abiertos.append(str(f))
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, 'open', espia)
    data, _ = ag.leer_imagenes_val(str(tmp_path), 'img', rows, 'perlin', expected)
    monkeypatch.setattr(builtins, 'open', real_open)
    assert len(data) == 4 and abiertos and all('_val_' in a and '_test_' not in a for a in abiertos)


# ── GPU: cascada ──────────────────────────────────────────────────────────
torch = pytest.importorskip('torch')


def tiny_vit():
    from torchvision.models.vision_transformer import VisionTransformer
    torch.manual_seed(1)
    m = VisionTransformer(image_size=32, patch_size=8, num_layers=3, num_heads=2, hidden_dim=16, mlp_dim=32,
                          num_classes=2).eval()
    with torch.no_grad():                                                  # torchvision inicia la cabeza en cero
        torch.nn.init.normal_(m.heads.head.weight, std=1.0)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.heads.parameters():
        p.requires_grad = True
    return m


def test_pasos_cascada():
    from torchvision.models import efficientnet_b0
    m = efficientnet_b0(weights=None)
    pe = ag.pasos_cascada(m, 'efficientnet_b0', RES)
    assert [p for p, _ in pe] == ['cabeza'] + [f'features.{i}' for i in range(8, -1, -1)]
    v = tiny_vit()
    pv = ag.pasos_cascada(v, 'vit_b16', RES)
    assert [p for p, _ in pv] == ['cabeza', 'encoder.layers.2', 'encoder.layers.1', 'encoder.layers.0', 'embeddings']
    assert pv[0][1] == ['heads.head', 'encoder.ln'] and pv[-1][1] == ['conv_proj']
    pp = ag.pasos_cascada(v, 'vit_b16', {**RES, 'vit_ln_final': 'paso_propio', 'vit_embeddings': 'conv_proj_y_tokens'})
    assert [p for p, _ in pp][:2] == ['cabeza', 'encoder.ln'] and pp[-1][1] == ['conv_proj', 'class_token', 'encoder.pos_embedding']
    assert ag.pasos_cascada(v, 'vit_b16', {**RES, 'vit_ln_final': 'sin_reinicializar'})[0][1] == ['heads.head']
    for _, nombres in pv:
        for n in nombres:
            v.get_submodule(n)                                             # los nombres existen


def test_reiniciar_determinista_y_acotado():
    import copy
    v1 = tiny_vit()
    v2 = copy.deepcopy(v1)
    torch.manual_seed(2026)
    c1 = ag.reiniciar(v1, ['encoder.layers.encoder_layer_2'], RES)
    torch.manual_seed(2026)
    ag.reiniciar(v2, ['encoder.layers.encoder_layer_2'], RES)
    assert all(torch.equal(a, b) for a, b in zip(v1.state_dict().values(), v2.state_dict().values()))
    assert c1 and all(k.startswith('encoder.layers.encoder_layer_2.') for k in c1)
    assert any('in_proj_weight' in k for k in c1)
    v3 = tiny_vit()
    c3 = ag.reiniciar(v3, ['encoder.layers.encoder_layer_2'], {**RES, 'mha_in_proj': 'solo_reset_parameters'})
    assert not any('in_proj' in k for k in c3)
    v4 = tiny_vit()
    with torch.no_grad():
        v4.class_token.fill_(1.0)                                         # entrenado: distinto de cero
    c4 = ag.reiniciar(v4, ['class_token', 'encoder.pos_embedding'], RES)
    assert set(c4) == {'class_token', 'encoder.pos_embedding'} and float(v4.class_token.abs().sum()) == 0.0


def test_cascada_vit_cabeza_cambia_mapa_y_no_atencion():
    """Enmienda 3: al aleatorizar la cabeza, el rollout ponderado cambia y la atención del backbone no."""
    from src.interpretability.run_maps import method_for
    import config
    metodo = method_for('vit_b16', config.load_matrix())
    torch.manual_seed(3)
    X = torch.randn(6, 3, 32, 32)
    y = np.array([0, 1, 0, 1, 0, 1])
    base, pasos, sims, medias = ag.cascada_modelo(tiny_vit(), 'vit_b16', X, metodo, 'cpu', RES, grid=16, y=y)
    assert [p['paso'] for p in pasos] == ['cabeza', 'encoder.layers.2', 'encoder.layers.1', 'encoder.layers.0', 'embeddings']
    assert np.allclose(sims[('cabeza', 'backbone_map')]['spearman56'], 1.0)
    assert np.nanmedian(sims[('cabeza', 'map')]['spearman56']) < 0.999
    assert not np.allclose(sims[('encoder.layers.2', 'backbone_map')]['spearman56'], 1.0)
    assert ('cabeza', 'map', 1) in medias
    # Reproducible: misma semilla de la cascada, mismas similitudes
    _, _, sims2, _ = ag.cascada_modelo(tiny_vit(), 'vit_b16', X, metodo, 'cpu', RES, grid=16)
    assert all(np.allclose(sims[k]['spearman56'], sims2[k]['spearman56'], equal_nan=True) for k in sims)


def test_cascada_efficientnet_y_resumen():
    from torchvision.models import efficientnet_b0
    from src.interpretability.run_maps import method_for
    import config
    metodo = method_for('efficientnet_b0', config.load_matrix())
    torch.manual_seed(4)
    X = torch.randn(4, 3, 64, 64)
    bases, sims = {}, {}
    for i, s in enumerate(ac.SEEDS):
        torch.manual_seed(100 + i)
        m = efficientnet_b0(weights=None, num_classes=2).eval()
        bases[s], pasos, sims[s], _ = ag.cascada_modelo(m, 'efficientnet_b0', X, metodo, 'cpu', RES, grid=16)
    assert len(pasos) == 10 and all(p['n_tensores_cambiados'] > 0 for p in pasos)
    resumen, ref = ag.resumen_cascada(bases, sims, pasos, RES, grid=16)
    assert set(resumen) == {p['paso'] for p in pasos} and len(ref['map']['spearman56']) == 10
    assert resumen['cabeza']['map']['spearman56']['global'] in ('cambia', 'no_cambia', 'indeterminado')
    json.dumps(ac.limpiar(resumen))


# ── GPU: ablación ─────────────────────────────────────────────────────────
def test_filtros():
    rng = np.random.default_rng(8)
    img = rng.integers(40, 200, (256, 256)).astype(np.uint8)
    assert len(ag.condiciones()) == 9
    lows = [ag.filtrar(img, 'paso_bajo', s, 'reflect', 'recortar_redondear') for s in ac.SIGMAS]
    assert all(abs(l.mean() - img.mean()) < 1.0 for l in lows)
    assert all(a.std() > b.std() for a, b in zip(lows, lows[1:]))                 # más σ, más suave
    hi = ag.filtrar(img, 'paso_alto', 4, 'reflect', 'recortar_redondear')
    assert abs(hi.mean() - img.mean()) < 1.0 and hi.dtype == np.uint8
    y = np.array([[-10.0, 100.0], [300.0, 50.0]])
    assert ag.a_uint8(y, 'recortar_redondear').tolist() == [[0, 100], [255, 50]]
    assert ag.a_uint8(y, 'reescalar_min_max').tolist() == [[0, 90], [255, 49]]     # (y + 10) / 310 · 255
    with pytest.raises(ac.Detener):
        ag.filtrar(img, 'banda', 1, 'reflect', 'recortar_redondear')


class Varianza(torch.nn.Module):
    """Clasificador de juguete: logit depresivo = desviación estándar de la imagen − umbral."""
    def __init__(self, umbral):
        super().__init__()
        self.umbral = umbral

    def forward(self, x):
        s = x.flatten(1).std(1) - self.umbral
        return torch.stack([torch.zeros_like(s), 50 * s], 1)


def test_ablacion_modelo_y_resumen():
    from torchvision import transforms
    rng = np.random.default_rng(9)
    tf = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])
    y = np.array([0] * 6 + [1] * 6)
    imgs = [np.full((256, 256), 120, np.uint8) + (rng.integers(0, 60, (256, 256)).astype(np.uint8) if c else 0)
            for c in y]
    por = {s: ag.ablacion_modelo(Varianza(0.02), imgs, y, tf, 'cpu', RES) for s in ac.SEEDS}
    assert por[42]['original']['f1'] == 1.0
    assert por[42]['paso_bajo_s8']['f1'] < 1.0                                     # quita la varianza
    r = ag.resumen_ablacion(por)
    assert r['original']['delta_f1_media'] == 0.0 and r['paso_bajo_s8']['delta_f1_media'] < 0
    assert r['paso_bajo_s8']['delta_f1_media'] == pytest.approx(r['paso_bajo_s8']['f1_media'] - 1.0)


# ── GPU: compuerta y no sobrescribir ──────────────────────────────────────
def test_gpu_compuerta_y_no_sobrescribe(tmp_path):
    fuera = tmp_path / 'res.json'
    fuera.write_text(json.dumps(RES))
    with pytest.raises(ac.Detener, match='versionado'):
        ag.check_gate(str(fuera))
    (tmp_path / 'cascada.json').write_text('{}')
    with pytest.raises(ac.Detener, match='nunca sobrescribe'):
        ag.refuse_overwrite(str(tmp_path))


# ── GPU: main de punta a punta en CPU (Gray-Scott, 2 imágenes de val) ─────
def test_gpu_main_punta_a_punta(tmp_path, monkeypatch):
    import hashlib
    from PIL import Image
    from build_model import build_model
    from data import get_transforms
    from src import paths
    from src.fase4 import caracteristicas as car

    rows = ag.filas_val(car.load_index())[:2]
    root = tmp_path / 'datos'
    img_dir = root / paths.REL['IMAGES']['gray_scott']
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(10)
    expected = {}
    for r in rows:
        Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8), 'L').save(img_dir / r['img_gray_scott'])
        expected[f"{paths.REL['IMAGES']['gray_scott']}/{r['img_gray_scott']}"] = \
            hashlib.sha256((img_dir / r['img_gray_scott']).read_bytes()).hexdigest()
    monkeypatch.setenv('TT_DATA_ROOT', str(root))
    monkeypatch.setattr(car, 'image_rows', lambda manifest, tech: (expected, {}))
    torch.manual_seed(0)
    model = build_model('efficientnet_b0', freeze_backbone=False, pretrained=False).eval()
    mpath = tmp_path / 'm.pt'
    torch.save(model.state_dict(), mpath)
    monkeypatch.setattr(car, 'verified_model', lambda root_, art, tech, seed, run: (str(mpath), 'sha'))
    prep = {'size': 224, 'resize_mode': 'resize', 'interpolation': 'bilinear', 'antialias': True,
            'hflip': True, 'vit': 'same', 'variant': 'px224-flip'}
    tf = get_transforms(False, img_size=224, hflip=False)
    X = torch.stack([tf(Image.open(img_dir / r['img_gray_scott']).convert('L')) for r in rows])
    with torch.no_grad():
        p = torch.softmax(model(X).double(), 1)[:, 1].tolist()
    run = {'preprocessing': prep, 'val_true': [int(r['y_true']) for r in rows] + [0] * 112,
           'val_probs': p + [0.5] * 112}
    (tmp_path / 'run.json').write_text(json.dumps(run))
    monkeypatch.setattr(car, 'run_json_path', lambda tech, seed: str(tmp_path / 'run.json'))
    monkeypatch.setattr(ag, 'check_gate', lambda path: 'abc1234')
    res = tmp_path / 'res.json'
    res.write_text(json.dumps({**RES, '_autor': 'SV'}))
    out = tmp_path / 'out'
    args = ['--models-root', str(tmp_path), '--resoluciones', str(res), '--out-dir', str(out),
            '--only', 'gray_scott', '--limit', '2']
    assert ag.main(args) == 0
    meta = json.loads((out / 'atribucion_gpu_meta.json').read_text(encoding='utf-8'))
    assert meta['smoke'] is True and meta['split'] == 'val' and meta['resoluciones']['autor'] == 'SV'
    assert meta['tecnicas']['gray_scott']['modelos']['42']['p_dep_max_abs_diff'] < 1e-6
    assert 'estimacion_114_total' in meta
    casc = json.loads((out / 'cascada.json').read_text(encoding='utf-8'))
    assert casc['tecnicas']['gray_scott']['orden'] == ['cabeza'] + [f'features.{i}' for i in range(8, -1, -1)]
    abl = json.loads((out / 'ablacion.json').read_text(encoding='utf-8'))
    assert len(abl['tecnicas']['gray_scott']) == 9 and abl['tecnicas']['gray_scott']['original']['delta_f1_media'] == 0.0
    z = np.load(out / 'cascada_similitudes.npz')
    assert 'gray_scott|42|cabeza|map|spearman56' in z.files and z['gray_scott|42|cabeza|map|spearman56'].shape == (2,)
    with pytest.raises(ac.Detener, match='nunca sobrescribe'):
        ag.main(args)


# ── P9: en la cascada, pasar a mapa constante cuenta como cambio ─────────
def test_p9_transicion_a_constante_cuenta_como_cambio():
    rng = np.random.default_rng(11)
    n = 20
    base = rng.random((n, 56, 56))
    # Paso: 12 de 20 mapas quedan constantes (ReLU anulado); los otros 8 idénticos a la base
    nuevo = base.copy()
    nuevo[:12] = 0.0
    sims = ag.similitudes(base, nuevo, 56)
    assert sims['const_base'].sum() == 0 and sims['const_nuevo'].sum() == 12
    assert np.isnan(sims['spearman56'][:12]).all()
    # Regla general (excluir): solo quedarían los 8 idénticos → mediana 1, «no cambia»
    assert ac.resumen_similitud(sims['spearman56'], 'excluir')[0] == pytest.approx(1.0)
    # P9: las 12 transiciones se imputan a −1 → la mediana cae, y se cuentan
    m, usado, n_nan, tr, dc = ac.resumen_cascada_spearman(sims['spearman56'], sims['const_base'],
                                                          sims['const_nuevo'], 'excluir')
    assert (m, usado, n_nan, tr, dc) == (-1.0, 20, 0, 12, 0)
    # Un mapa ya constante sin aleatorizar no es transición: sigue la regla general (se excluye)
    base2 = base.copy()
    base2[0] = 0.0
    s2 = ag.similitudes(base2, nuevo, 56)
    m2, usado2, nan2, tr2, dc2 = ac.resumen_cascada_spearman(s2['spearman56'], s2['const_base'], s2['const_nuevo'],
                                                             'excluir')
    assert (usado2, nan2, tr2, dc2) == (19, 1, 11, 1)


def test_p9_en_resumen_cascada():
    """Con 5 semillas, un paso que anula la mayoría de los mapas se lee como «cambia»
    y el detalle reporta las transiciones."""
    rng = np.random.default_rng(12)
    common = rng.random((10, 56, 56))
    bases = {s: {'map': common + 0.05 * rng.random((10, 56, 56)), 'backbone_map': None} for s in ac.SEEDS}
    sims = {}
    for s in ac.SEEDS:
        nuevo = bases[s]['map'].copy()
        nuevo[:7] = 0.0
        sims[s] = {('cabeza', 'map'): ag.similitudes(bases[s]['map'], nuevo, 56)}
    res = {**RES, 'criterio_semillas': 'mayoria', 'similitud_constantes': 'excluir'}
    out, _ = ag.resumen_cascada(bases, sims, [{'paso': 'cabeza', 'modulos': ['classifier.1']}], res)
    sp = out['cabeza']['map']['spearman56']
    assert sp['global'] == 'cambia'
    assert all(sp['por_semilla_detalle'][str(s)]['n_transiciones_a_constante'] == 7 for s in ac.SEEDS)
    assert '_regla_P9' in sp['por_semilla_detalle']
