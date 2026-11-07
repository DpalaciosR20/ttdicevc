"""F4.1 y F4.7 (src/fase4): pruebas sintéticas, sin datos reales ni modelos entrenados."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.fase4 import caracteristicas as car  # noqa: E402
from src.fase4 import geometria as geo  # noqa: E402
from src.fase4 import recuperabilidad as rec  # noqa: E402
from src.protocol import bootstrap as bs  # noqa: E402
from src.protocol import insumos as ins_mod  # noqa: E402

CFG = ins_mod.load_config()
RES = {'ridge_escalado': 'ninguno', 'alpha_desempate': 'detener', 'alpha_crossfit': 'seleccionado'}


def _u_and_x(n, d, noise, seed, informative=True):
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(n, 5))
    if informative:
        A = rng.normal(size=(5, d))
        X = U @ A + noise * rng.normal(size=(n, d))
    else:
        X = rng.normal(size=(n, d))
    return U, X


# ── Sonda Ridge ───────────────────────────────────────────────────────────
@pytest.mark.parametrize('escalado', ['ninguno', 'standard_train'])
def test_ridge_recupera_u_conocida(escalado):
    U, X = _u_and_x(500, 40, 1e-3, 0)
    alpha, table = rec.select_alpha(X[:400], U[:400], X[400:], U[400:], escalado, 'menor')
    m = rec.make_ridge(alpha, escalado).fit(X[:400], U[:400])
    r2 = rec.r2_raw(U[400:], rec.predict_clip(m, X[400:]))
    assert r2.shape == (5,) and r2.min() > 0.999
    assert len(table) == 13 and [t['alpha'] for t in table] == list(rec.ALPHAS)


def test_ridge_ruido_r2_cercano_a_cero():
    U, X = _u_and_x(600, 40, 0, 1, informative=False)
    alpha, _ = rec.select_alpha(X[:400], U[:400], X[400:], U[400:], 'ninguno', 'menor')
    m = rec.make_ridge(alpha, 'ninguno').fit(X[:400], U[:400])
    r2 = rec.r2_raw(U[400:], rec.predict_clip(m, X[400:]))
    assert np.all(np.abs(r2) < 0.1)


def test_alphas_del_plan():
    assert rec.ALPHAS == tuple(float(a) for a in np.logspace(-3, 3, 13))
    assert rec.ALPHAS[0] == pytest.approx(1e-3) and rec.ALPHAS[-1] == pytest.approx(1e3)


def test_recorte_a_0_1():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    U = np.clip(3 * X[:, :1] @ np.ones((1, 5)), -5, 5)          # objetivo fuera de [0, 1]
    m = rec.make_ridge(1e-3, 'ninguno').fit(X, U)
    raw = m.predict(X)
    assert raw.min() < 0 and raw.max() > 1
    P = rec.predict_clip(m, X)
    assert P.min() == 0.0 and P.max() == 1.0
    U_hat, _ = rec.crossfit(lambda: rec.make_ridge(1e-3, 'ninguno'), X, U)
    assert U_hat.min() >= 0 and U_hat.max() <= 1


def test_empate_de_alpha_no_se_resuelve_solo():
    U, _ = _u_and_x(100, 3, 0, 3)
    X = np.zeros((100, 3))                                     # predicción constante: R² igual para todo alpha
    with pytest.raises(rec.Detener, match='empate'):
        rec.select_alpha(X[:80], U[:80], X[80:], U[80:], 'ninguno', 'detener')
    assert rec.select_alpha(X[:80], U[:80], X[80:], U[80:], 'ninguno', 'menor')[0] == rec.ALPHAS[0]
    assert rec.select_alpha(X[:80], U[:80], X[80:], U[80:], 'ninguno', 'mayor')[0] == rec.ALPHAS[-1]


class _Espia:
    """Estimador que registra qué filas vio al ajustar (columna 0 = id) y falla si predice una de ellas."""
    registro = []

    def fit(self, X, U):
        self.vistos = set(X[:, 0].astype(int))
        self.mean = U.mean(axis=0)
        _Espia.registro.append(self.vistos)
        return self

    def predict(self, X):
        ids = set(X[:, 0].astype(int))
        assert not ids & self.vistos, 'fuga: se predice una fila usada en el ajuste'
        return np.tile(self.mean, (len(X), 1))


def test_crossfit_sin_fuga():
    _Espia.registro = []
    n = 103
    X = np.column_stack([np.arange(n), np.random.default_rng(4).normal(size=(n, 2))])
    U = np.random.default_rng(5).uniform(size=(n, 5))
    U_hat, fold = rec.crossfit(_Espia, X, U)
    assert len(_Espia.registro) == 5 and np.bincount(fold).sum() == n
    for k, vistos in enumerate(_Espia.registro):
        assert vistos == set(np.flatnonzero(fold != k))          # cada modelo vio exactamente los otros pliegues
    from sklearn.model_selection import KFold
    ref = np.full(n, -1)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
        ref[te] = k
    assert (fold == ref).all()                                  # KFold(5, shuffle=True, random_state=42)


def test_etapa_val_y_logreg_de_4_3():
    U, X = _u_and_x(400, 20, 0.05, 6)
    y = (U[:, 0] + 0.2 * np.random.default_rng(7).normal(size=400) > 0.5).astype(int)
    Xd, Ud, yd = ({'train': A[:300], 'val': A[300:]} for A in (X, U, y))
    out, objs = rec.etapa_val(Xd, Ud, yd, CFG, RES)
    assert out['alpha'] in rec.ALPHAS and out['pliegues'] == [60] * 5
    assert len(out['r2_val']) == 5 and min(out['r2_val']) > 0.9
    grid, fixed, tol = rec.logreg_spec(CFG)
    assert len(grid) == 26 and fixed['max_iter'] == 5000 and tol == 1e-12
    assert grid[0] == {'C': 1e-4, 'class_weight': None} and grid[1]['class_weight'] == 'balanced'
    sel = out['logreg_u_hat']
    assert grid[sel['indice']] == sel['combo']


def test_resoluciones_pendientes_detienen():
    with pytest.raises(rec.Detener, match='PENDIENTE'):
        rec.check_resoluciones({'ridge_escalado': 'ninguno'})
    with pytest.raises(rec.Detener, match='no previstos'):
        rec.check_resoluciones(RES | {'ridge_escalado': 'pca'})
    with pytest.raises(rec.Detener, match='PENDIENTE'):
        rec.check_resoluciones({}, geo.RESOLUCIONES)


# ── ΔF1 con las réplicas de A1 ────────────────────────────────────────────
def test_delta_f1_e_ic_con_indices_de_A1():
    rng = np.random.default_rng(8)
    y = (rng.uniform(size=114) < 44 / 114).astype(int)
    p_u = np.clip(y * 0.4 + rng.uniform(size=114) * 0.6, 0, 1)
    p_hat = np.clip(y * 0.3 + rng.uniform(size=114) * 0.7, 0, 1)
    idx = bs.indices(CFG)
    assert idx.shape == (10000, 114)
    d, reps = rec.delta_f1(y, p_u, p_hat, idx, CFG['enmienda_4']['ic']['descriptivo'])
    f1 = lambda yy, p: f1_score(yy, (p > 0.5).astype(int), average='macro')  # noqa: E731
    assert d['delta_f1_obs'] == pytest.approx(f1(y, p_u) - f1(y, p_hat), abs=1e-12)
    for b in range(5):
        i = idx[b]
        assert reps[b] == pytest.approx(f1(y[i], p_u[i]) - f1(y[i], p_hat[i]), abs=1e-12)
    lo, hi = np.quantile(reps, [0.025, 0.975], method='linear')
    assert d['ic'] == [pytest.approx(lo, abs=1e-15), pytest.approx(hi, abs=1e-15)]
    assert d['contiene_0'] == (lo <= 0 <= hi)


def test_idx_sha256_igual_al_de_la_fase_5():
    f5 = json.loads((REPO / rec.F5_RESULT).read_text(encoding='utf-8'))
    assert rec.check_idx(bs.indices(CFG), f5['meta']) == f5['meta']['bootstrap']['idx_sha256']
    with pytest.raises(rec.Detener, match='idx_sha256'):
        rec.check_idx(bs.indices(CFG)[:, ::-1], f5['meta'])


# ── No sobrescribe ────────────────────────────────────────────────────────
def test_write_json_nunca_sobrescribe(tmp_path):
    p = tmp_path / 'x' / 'r.json'
    rec.write_json(p, {'a': 1})
    with pytest.raises(rec.Detener, match='nunca sobrescribe'):
        rec.write_json(p, {'a': 2})
    assert json.loads(p.read_text(encoding='utf-8')) == {'a': 1}


def _resoluciones(tmp_path):
    p = tmp_path / 'res.json'
    p.write_text(json.dumps(RES), encoding='utf-8')
    return str(p)


def test_prueba_se_niega_a_sobrescribir(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, 'check_gate_prueba', lambda val_rel: None)
    man = tmp_path / 'm.csv'
    man.write_text('tecnica,backbone,archivo,sha256\n', encoding='utf-8')
    (tmp_path / 'prueba').mkdir()
    (tmp_path / 'prueba' / 'resultados_prueba.json').write_text('{}', encoding='utf-8')
    index = rec.load_index()
    monkeypatch.setattr(rec, 'load_index', lambda: index)
    monkeypatch.setattr(rec, 'REPO', tmp_path)       # val_path relativo a REPO
    with pytest.raises(rec.Detener, match='nunca sobrescribe'):
        rec.main(['--prueba', '--features-dir', str(tmp_path), '--manifest', str(man),
                  '--resoluciones', _resoluciones(tmp_path), '--out-dir', str(tmp_path)])


def test_val_se_niega_a_sobrescribir(tmp_path):
    man = tmp_path / 'm.csv'
    man.write_text('tecnica,backbone,archivo,sha256\n', encoding='utf-8')
    (tmp_path / 'val').mkdir()
    (tmp_path / 'val' / 'resultados_val.json').write_text('{}', encoding='utf-8')
    with pytest.raises(rec.Detener, match='nunca sobrescribe'):
        rec.main(['--val', '--features-dir', str(tmp_path), '--manifest', str(man),
                  '--resoluciones', _resoluciones(tmp_path), '--out-dir', str(tmp_path)])


def test_compuerta_de_prueba_exige_val_versionado(monkeypatch):
    with pytest.raises(rec.Detener):
        rec.check_gate_prueba('results/f41_recuperabilidad/val/no_existe.json')


# ── Mantel ────────────────────────────────────────────────────────────────
def _mantel_referencia(Da, Db, n_perm, rng):
    """Implementación directa: permuta filas y columnas, scipy.stats.spearmanr sobre los pares i < j."""
    b = geo.condensed(Db)
    r_obs = stats.spearmanr(geo.condensed(Da), b).statistic
    perm = []
    for _ in range(n_perm):
        p = rng.permutation(Da.shape[0])
        perm.append(stats.spearmanr(geo.condensed(Da[np.ix_(p, p)]), b).statistic)
    perm = np.array(perm)
    return r_obs, int(np.sum(perm >= r_obs - 1e-12)), int(np.sum(np.abs(perm) >= abs(r_obs) - 1e-12))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_mantel_igual_a_implementacion_de_referencia(seed):
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(14, 5))
    X = U @ rng.normal(size=(5, 8)) + 0.8 * rng.normal(size=(14, 8))
    Da, Db = geo.dist_u(U), geo.dist_features(X)
    r_ref, k_may, k_bil = _mantel_referencia(Da, Db, 199, np.random.default_rng(2026))
    m = geo.mantel(Da, Db, 199, np.random.default_rng(2026), 'mayor', 'k_mas_1_sobre_B_mas_1')
    assert m['rho_spearman'] == pytest.approx(r_ref, abs=1e-12)
    assert m['k'] == k_may and m['p'] == (k_may + 1) / 200
    mb = geo.mantel(Da, Db, 199, np.random.default_rng(2026), 'bilateral', 'k_sobre_B')
    assert mb['k'] == k_bil and mb['p'] == k_bil / 199
    assert geo.spearman_dist(Da, Db) == pytest.approx(r_ref, abs=1e-12)


def test_mantel_con_empates_en_distancias():
    rng = np.random.default_rng(9)
    U = rng.integers(0, 3, size=(12, 5)).astype(float) / 2       # muchas distancias repetidas
    X = rng.normal(size=(12, 6))
    Da, Db = geo.dist_u(U), geo.dist_features(X)
    r_ref, k_may, _ = _mantel_referencia(Da, Db, 99, np.random.default_rng(1))
    m = geo.mantel(Da, Db, 99, np.random.default_rng(1), 'mayor', 'k_mas_1_sobre_B_mas_1')
    assert m['rho_spearman'] == pytest.approx(r_ref, abs=1e-12) and m['k'] == k_may


def test_mantel_estructura_perfecta():
    rng = np.random.default_rng(10)
    U = rng.uniform(size=(20, 5))
    Da = geo.dist_u(U)
    m = geo.mantel(Da, np.sqrt(Da) + 3.0, 99, np.random.default_rng(2026), 'mayor', 'k_mas_1_sobre_B_mas_1')
    assert m['rho_spearman'] == pytest.approx(1.0) and m['k'] == 0 and m['p'] == 1 / 100


def test_mantel_p_uniforme_bajo_H0():
    rng = np.random.default_rng(11)
    ps = []
    for _ in range(300):
        Da = geo.dist_u(rng.uniform(size=(12, 5)))
        Db = geo.dist_features(rng.normal(size=(12, 4)))
        ps.append(geo.mantel(Da, Db, 99, rng, 'mayor', 'k_mas_1_sobre_B_mas_1')['p'])
    ps = np.array(ps)
    assert stats.kstest(ps, 'uniform').pvalue > 0.001
    assert 0.42 < ps.mean() < 0.58 and 0.02 < np.mean(ps <= 0.05) < 0.10


def test_mantel_constantes_del_plan():
    assert geo.N_PERM == 9999 and geo.RNG_SEED == 2026


# ── Extracción de características ─────────────────────────────────────────
def _png_bytes(seed, size=256):
    from PIL import Image
    arr = (np.random.default_rng(seed).uniform(size=(size, size)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode='L').save(buf, format='PNG')
    return buf.getvalue()


def _transform():
    sys.path.insert(0, str(REPO / 'src' / 'training'))
    from data import get_transforms
    return get_transforms(False, img_size=224, hflip=False, interpolation='bilinear', antialias=True)


def test_penultima_capa_efficientnet():
    import torch
    sys.path.insert(0, str(REPO / 'src' / 'training'))
    from build_model import build_model
    torch.manual_seed(0)
    model = build_model('efficientnet_b0', pretrained=False).eval()
    imgs = [_png_bytes(s) for s in range(3)]
    x = torch.stack([_transform()(__import__('PIL.Image', fromlist=['Image']).open(io.BytesIO(b)).convert('L')) for b in imgs])
    with torch.no_grad():
        ref = torch.flatten(model.avgpool(model.features(x)), 1).numpy()
        logits = model(x)
    fe = car.to_feature_extractor(model, 'efficientnet_b0')
    X = car.extract(fe, imgs, _transform(), 'cpu', batch_size=2)
    assert X.shape == (3, 1280) and X.dtype == np.float32
    np.testing.assert_allclose(X, ref, rtol=1e-5, atol=1e-6)
    assert logits.shape == (3, 2)                                 # la cabeza original ya no está en fe
    assert isinstance(fe.classifier, torch.nn.Identity)


def test_penultima_capa_vit():
    import torch
    sys.path.insert(0, str(REPO / 'src' / 'training'))
    from build_model import build_model
    torch.manual_seed(0)
    model = build_model('vit_b16', pretrained=False).eval()
    imgs = [_png_bytes(s) for s in range(2)]
    from PIL import Image
    x = torch.stack([_transform()(Image.open(io.BytesIO(b)).convert('L')) for b in imgs])
    with torch.no_grad():
        z = model._process_input(x)
        z = torch.cat([model.class_token.expand(z.shape[0], -1, -1), z], dim=1)
        ref = model.encoder(z)[:, 0].numpy()
    fe = car.to_feature_extractor(model, 'vit_b16')
    X = car.extract(fe, imgs, _transform(), 'cpu')
    assert X.shape == (2, 768)
    np.testing.assert_allclose(X, ref, rtol=1e-5, atol=1e-5)


def test_compare_backbone_identico_salvo_cabeza():
    import torch
    a = {'w': torch.ones(3), 'b': torch.zeros(2), 'heads.head.weight': torch.ones(2, 3)}
    b = {'w': torch.ones(3), 'b': torch.zeros(2), 'heads.head.weight': torch.zeros(1000, 3)}
    c = car.compare_backbone(a, b, 'heads.head.')
    assert c['identico'] and c['n_tensores'] == 2
    b['w'] = torch.ones(3) + 1e-7
    c = car.compare_backbone(a, b, 'heads.head.')
    assert not c['identico'] and c['distintos'] == ['w']
    del b['b']
    assert car.compare_backbone(a, b, 'heads.head.')['faltan_en_b'] == ['b']


def test_dir_digest_igual_a_build_manifest():
    from src.build_manifest import dir_digest
    rows = [{'rel_path': f'images/x/{i}.png', 'sha256': f'{i:064x}'} for i in (3, 1, 2)]
    assert car.dir_digest({r['rel_path']: r['sha256'] for r in rows}) == dir_digest(rows)


def test_dir_digest_real_de_data_manifest():
    """El cálculo reproduce las filas kind=dir de meta/data_manifest.csv a partir de sus filas de archivo."""
    man = car.load_data_manifest()
    for tech in car.TECHNIQUES:
        files, d = car.image_rows(man, tech)
        assert len(files) == int(d['n_files']) == 1136
        assert car.dir_digest(files) == d['sha256']


def test_imagenes_con_hash_distinto_se_detienen(tmp_path):
    b = _png_bytes(0)
    (tmp_path / 'a.png').write_bytes(b)
    ok = car.sha256_bytes(b)
    data, h = car.read_verified_images(str(tmp_path), 'rel', ['a.png'], {'rel/a.png': ok})
    assert data == [b] and h == {'rel/a.png': ok}
    with pytest.raises(car.Detener, match='sha256'):
        car.read_verified_images(str(tmp_path), 'rel', ['a.png'], {'rel/a.png': '0' * 64})
    with pytest.raises(car.Detener, match='no está'):
        car.read_verified_images(str(tmp_path), 'rel', ['a.png'], {})


def test_npz_nunca_sobrescribe(tmp_path):
    p = str(tmp_path / 't__b.npz')
    h = car.save_npz(p, np.ones((2, 3), np.float32), ['a', 'b'], ['train', 'val'])
    assert h == car.sha256_file(p)
    z = np.load(p)
    assert z['features'].dtype == np.float32 and list(z['sample_id']) == ['a', 'b']
    with pytest.raises(car.Detener, match='nunca sobrescribe'):
        car.save_npz(p, np.zeros((2, 3), np.float32), ['a', 'b'], ['train', 'val'])


def test_modelos_y_backbones_del_plan():
    man = car.load_artefactos()
    for tech in car.TECHNIQUES:
        for s in car.FINAL_SEEDS:
            row = man[car.model_logical(tech, s)]
            assert row['task'] == 'F3.5'
            d = json.load(open(car.run_json_path(tech, s), encoding='utf-8'))
            assert d['model_file']['sha256'] == row['sha256']
            assert d['preprocessing']['size'] == 224
    assert car.plan_backbones('perlin') == [('vit_b16_imagenet', ('imagenet', None))]
    gs = car.plan_backbones('gray_scott')
    assert [b for b, _ in gs] == [f'efficientnet_b0_finetune_seed{s}' for s in car.FINAL_SEEDS] + ['efficientnet_b0_imagenet']
    assert set(rec.PRINCIPAL.values()) | {b for v in rec.SENSIBILIDAD.values() for b in v} == \
        {b for t in car.TECHNIQUES for b, _ in car.plan_backbones(t)}


def test_load_index_orden_y_tamanos():
    rows = car.load_index()
    assert len(rows) == 1136 and [r['split'] for r in rows[:1]] == ['train']
    assert rows[908]['sample_id'] == 'val_0000' and rows[1022]['sample_id'] == 'test_0000'
