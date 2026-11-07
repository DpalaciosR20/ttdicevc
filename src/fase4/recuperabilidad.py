"""
recuperabilidad.py
──────────────────
F4.1 — Recuperabilidad de parámetros (docs/plan_fase4.md, F4.1). Local.

Método (plan, F4.1-1…7):
  1. Sonda lineal: Ridge multisalida de las características de la penúltima
     capa (src/fase4/caracteristicas.py) a u (5 componentes, escala [0, 1]),
     con las predicciones recortadas a [0, 1].
  2. alpha ∈ logspace(-3, 3, 13), elegido por el R² medio en val.
  3. Ajuste en train; R² en val (principal) y en prueba (una vez; G1).
     û de train por cross-fitting de 5 pliegues (KFold(shuffle=True,
     random_state=42)); û de val y prueba con el modelo ajustado en todo train.
  4. LogReg con la rejilla, el escalado y la regla de §4.3
     (configs/protocolo.json: techo_5d) sobre û y, de la misma forma, sobre u.
     La de u se contrasta con umap5d__logreg de F2.1.
  5. ΔF1 = F1(u) − F1(û) en prueba, con IC percentil bilateral al 95 % sobre
     las réplicas de A1 (mismos índices que la Fase 5; se verifica idx_sha256).
  6. R²: sklearn.metrics.r2_score(multioutput='raw_values').
  Condición 2 de F4.6 (docs/plan_fase4.md): «el IC de ΔF1 de F4.1 contiene 0».

Puntos que el plan no fija (reporte_e1a.md, PENDIENTES). No se resuelven aquí:
las funciones los reciben como parámetros y la CLI exige el archivo
--resoluciones con todos ellos; si falta alguno, se detiene.
  ridge_escalado     'ninguno' | 'standard_train'   (escalado de X antes de Ridge)
  alpha_desempate    'menor' | 'mayor' | 'detener'  (empate a 1e-12 en R² medio de val)
  alpha_crossfit     'seleccionado'                 (alpha de los 5 pliegues)

Etapas (G1):
  --val     train y val: alpha, R² de val, LogReg sobre û y sobre u (selección
            en val), contraste de val con F2.1. Escribe <out>/val/resultados_val.json.
  --prueba  una sola vez. Exige: la etiqueta enmienda-5 en la historia de HEAD;
            docs/plan_fase4.md sin cambios frente a ella; resultados.json de la
            Fase 5 y resultados_val.json versionados y sin cambios; árbol
            limpio; que no exista <out>/prueba/resultados_prueba.json. Recalcula
            la etapa val y exige que coincida con la versionada. Calcula R² de
            prueba, F1 de prueba, ΔF1 con su IC y la condición 2.
Ninguna etapa sobrescribe su salida.

Uso:
  python -m src.fase4.recuperabilidad --val    --features-dir <dir> --resoluciones <json>
  python -m src.fase4.recuperabilidad --prueba --features-dir <dir> --resoluciones <json>
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / 'src' / 'training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TECHNIQUES = ('gray_scott', 'perlin', 'lorenz')
PRINCIPAL = {'gray_scott': 'efficientnet_b0_finetune_seed42', 'perlin': 'vit_b16_imagenet',
             'lorenz': 'vit_b16_imagenet'}
SENSIBILIDAD = {'gray_scott': [f'efficientnet_b0_finetune_seed{s}' for s in (7, 123, 2024, 99)]
                + ['efficientnet_b0_imagenet'], 'perlin': [], 'lorenz': []}
ALPHAS = tuple(float(a) for a in np.logspace(-3, 3, 13))
KFOLD = {'n_splits': 5, 'shuffle': True, 'random_state': 42}
CLIP = (0.0, 1.0)
ALPHA_TIE_TOL = 1e-12
LOGREG_MAX_ITER = 5000           # §4.3 no lo fija; igual que F2.1, F2.2 y F4.5
GATE_TAG = 'enmienda-5'
PLAN = 'docs/plan_fase4.md'
F5_RESULT = 'results/f5_contraste/resultados.json'
F21_LOGREG = REPO / 'results' / 'predictions' / 'referencia' / 'umap5d__logreg__none__seed42.csv'
OUT = REPO / 'results' / 'f41_recuperabilidad'
RESOLUCIONES = {'ridge_escalado': ('ninguno', 'standard_train'),
                'alpha_desempate': ('menor', 'mayor', 'detener'),
                'alpha_crossfit': ('seleccionado',)}


class Detener(RuntimeError):
    """El análisis se detiene: una comprobación o una compuerta no se cumple."""


# ── Resoluciones de los puntos PENDIENTES ─────────────────────────────────
def check_resoluciones(res, spec=RESOLUCIONES):
    faltan = [k for k in spec if res.get(k) is None]
    if faltan:
        raise Detener('PENDIENTE sin resolver en --resoluciones: ' + ', '.join(faltan))
    malos = {k: res[k] for k in spec if res[k] not in spec[k]}
    if malos:
        raise Detener(f'valores no previstos en --resoluciones: {malos}')
    return {k: res[k] for k in spec}


# ── Sonda Ridge ───────────────────────────────────────────────────────────
def make_ridge(alpha, escalado):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if escalado == 'ninguno':
        return Ridge(alpha=alpha)
    if escalado == 'standard_train':
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    raise Detener(f'escalado desconocido {escalado!r}')


def predict_clip(model, X):
    return np.clip(model.predict(X), *CLIP)


def r2_raw(U, U_hat):
    from sklearn.metrics import r2_score
    return np.asarray(r2_score(U, U_hat, multioutput='raw_values'), dtype=np.float64)


def select_alpha(X_tr, U_tr, X_va, U_va, escalado, desempate, alphas=ALPHAS):
    """alpha con el mayor R² medio en val (predicciones recortadas). Devuelve (alpha, tabla)."""
    table = []
    for a in alphas:
        m = make_ridge(a, escalado).fit(X_tr, U_tr)
        r2 = r2_raw(U_va, predict_clip(m, X_va))
        table.append({'alpha': a, 'r2_val_media': float(r2.mean()), 'r2_val': r2.tolist()})
    best = max(t['r2_val_media'] for t in table)
    tied = [t['alpha'] for t in table if t['r2_val_media'] >= best - ALPHA_TIE_TOL]
    if len(tied) > 1:
        if desempate == 'detener':
            raise Detener(f'empate de alpha en R² medio de val: {tied}')
        alpha = min(tied) if desempate == 'menor' else max(tied)
    else:
        alpha = tied[0]
    return alpha, table


def crossfit(make_model, X, U, kfold=KFOLD):
    """û de train por cross-fitting: cada fila se predice con el modelo de los otros pliegues.

    Devuelve (U_hat recortada, pliegue de cada fila)."""
    from sklearn.model_selection import KFold
    U_hat = np.full(U.shape, np.nan)
    fold = np.full(len(X), -1, dtype=int)
    for k, (tr, te) in enumerate(KFold(**kfold).split(X)):
        m = make_model().fit(X[tr], U[tr])
        U_hat[te] = predict_clip(m, X[te])
        fold[te] = k
    if np.isnan(U_hat).any() or (fold < 0).any():
        raise Detener('cross-fitting incompleto')
    return U_hat, fold


# ── LogReg de §4.3 ────────────────────────────────────────────────────────
def logreg_spec(cfg):
    t = cfg['techo_5d']
    spec = t['familias']['logreg']
    if (t['scaler']['name'], t['scaler']['fit_split']) != ('StandardScaler', 'train') or 'logreg' not in t['scaler']['families']:
        raise Detener(f'escalado de §4.3 inesperado: {t["scaler"]}')
    sel = t['selection']
    if (sel['metric'], sel['split']) != ('f1_macro', 'val'):
        raise Detener(f'selección de §4.3 inesperada: {sel}')
    if spec['estimator'] != 'LogisticRegression':
        raise Detener(f'estimador inesperado {spec["estimator"]}')
    Cs = [float(v) for v in np.logspace(spec['C']['log10_low'], spec['C']['log10_high'], spec['C']['n'])]
    grid = [{'C': c, 'class_weight': w} for c in Cs for w in spec['class_weight']]   # C varía más lento
    fixed = {'penalty': spec['penalty'], 'solver': spec['solver'], 'max_iter': LOGREG_MAX_ITER,
             'random_state': t['random_state']}
    return grid, fixed, float(sel['tie_tolerance'])


def p_dep_of(model, X):
    pos = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(X)[:, pos].astype(np.float64)


def f1_of(y, p, thr=0.5):
    from sklearn.metrics import f1_score
    return float(f1_score(y, (p > thr).astype(int), average='macro'))


def logreg_select(Z_tr, y_tr, Z_va, y_va, cfg):
    """Rejilla de §4.3 con StandardScaler ajustado en train; F1 de val; empate → anterior en la rejilla."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    grid, fixed, tol = logreg_spec(cfg)
    f1s, models, warned = [], [], 0
    for combo in grid:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            m = make_pipeline(StandardScaler(), LogisticRegression(**fixed, **combo)).fit(Z_tr, y_tr)
        warned += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        f1s.append(f1_of(y_va, p_dep_of(m, Z_va)))
        models.append(m)
    best = max(f1s)
    i = next(k for k, f in enumerate(f1s) if f >= best - tol)
    return {'combo': grid[i], 'indice': i, 'f1_val': f1s[i],
            'n_empate': sum(f >= best - tol for f in f1s), 'avisos_convergencia': warned}, models[i]


# ── Datos ─────────────────────────────────────────────────────────────────
def load_index():
    with open(REPO / 'meta' / 'sample_index.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {s: sorted((r for r in rows if r['split'] == s), key=lambda r: r['sample_id'])
            for s in ('train', 'val', 'test')}


def load_u(index, splits):
    """u (n × 5, [0, 1]) y etiquetas por split, en el orden de sample_id (alineadas por idx)."""
    from src import paths
    from src.build_manifest import verify_manifest
    from umap_io import load_umap_arrays
    problems = verify_manifest('UMAP_PARAMS')
    if problems:
        raise Detener(f'umap_params.pkl: {problems}')
    arr = load_umap_arrays(paths.umap_params_path())
    out = {}
    for s in splits:
        order = np.array([int(r['idx']) for r in index[s]])
        u, lab = arr[s]['params'][order].astype(np.float64), arr[s]['labels'][order].astype(int)
        if list(lab) != [int(r['y_true']) for r in index[s]]:
            raise Detener(f'umap_params.pkl ({s}): etiquetas no alineadas con sample_index por idx')
        if u.shape[1] != 5:
            raise Detener(f'u ({s}) con {u.shape[1]} componentes')
        out[s] = (u, lab)
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_features(features_dir, manifest_csv, tech, backbone, index, splits):
    """Características por split (orden de sample_id), verificando el sha256 del .npz."""
    with open(manifest_csv, newline='', encoding='utf-8') as f:
        rows = {(r['tecnica'], r['backbone']): r for r in csv.DictReader(f)}
    row = rows.get((tech, backbone))
    if row is None:
        raise Detener(f'{tech}/{backbone} no está en {manifest_csv}')
    path = Path(features_dir) / row['archivo']
    if sha256_file(path) != row['sha256']:
        raise Detener(f'{path}: sha256 distinto de {manifest_csv}')
    z = np.load(path, allow_pickle=False)
    X, sid, spl = z['features'], [str(s) for s in z['sample_id']], [str(s) for s in z['split']]
    out = {}
    for s in splits:
        want = [r['sample_id'] for r in index[s]]
        pos = {k: i for i, k in enumerate(sid)}
        if any(k not in pos or spl[pos[k]] != s for k in want):
            raise Detener(f'{path}: faltan muestras de {s} o su split no coincide')
        out[s] = X[[pos[k] for k in want]].astype(np.float64)
    return out, row['sha256']


def load_f21(split, index):
    with open(F21_LOGREG, newline='', encoding='utf-8') as f:
        rows = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == split}
    return np.array([float(rows[r['sample_id']]['p_dep']) for r in index[split]])


# ── Etapa val ─────────────────────────────────────────────────────────────
def etapa_val(X, U, y, cfg, res):
    """X, U, y: {split: array} con train y val. Devuelve (resultado serializable, objetos para prueba)."""
    alpha, table = select_alpha(X['train'], U['train'], X['val'], U['val'], res['ridge_escalado'],
                                res['alpha_desempate'])
    full = make_ridge(alpha, res['ridge_escalado']).fit(X['train'], U['train'])
    U_hat = {'val': predict_clip(full, X['val'])}
    U_hat['train'], fold = crossfit(lambda: make_ridge(alpha, res['ridge_escalado']), X['train'], U['train'])
    sel_hat, m_hat = logreg_select(U_hat['train'], y['train'], U_hat['val'], y['val'], cfg)
    out = {'alpha': alpha, 'alphas': list(ALPHAS), 'r2_por_alpha': table,
           'r2_val': r2_raw(U['val'], U_hat['val']).tolist(),
           'r2_train_crossfit': r2_raw(U['train'], U_hat['train']).tolist(),
           'logreg_u_hat': sel_hat, 'pliegues': np.bincount(fold).tolist()}
    return out, {'ridge': full, 'logreg_u_hat': m_hat, 'U_hat': U_hat}


def logreg_u(U, y, cfg, index=None):
    sel, m = logreg_select(U['train'], y['train'], U['val'], y['val'], cfg)
    out = {'logreg_u': sel}
    if index is not None:
        f21 = load_f21('val', index)
        out['f21_val_p_dep_max_abs_diff'] = float(np.max(np.abs(p_dep_of(m, U['val']) - f21)))
    return out, m


# ── Etapa prueba ──────────────────────────────────────────────────────────
def delta_f1(y, p_u, p_hat, idx, ic_cfg, thr=0.5):
    """ΔF1 = F1(u) − F1(û) en el observado y en cada réplica (A1); IC percentil bilateral."""
    from src.protocol import bootstrap as bs
    from src.protocol import contraste as cx
    W = bs.weights(idx, len(y))
    yu, yh = bs.predict(p_u, thr), bs.predict(p_hat, thr)
    W_obs = bs.observed_weights(len(y))
    obs = float(bs.f1_macro(W_obs, y, yu)[0] - bs.f1_macro(W_obs, y, yh)[0])
    reps = bs.f1_macro(W, y, yu) - bs.f1_macro(W, y, yh)
    lo, hi = cx._ci95(ic_cfg)
    ic = bs.percentile_interval(reps, lo, hi)
    return {'delta_f1_obs': obs, 'ic': list(ic), 'nivel': ic_cfg['nivel'], 'contiene_0': bool(ic[0] <= 0 <= ic[1]),
            'f1_u': float(bs.f1_macro(W_obs, y, yu)[0]), 'f1_u_hat': float(bs.f1_macro(W_obs, y, yh)[0])}, reps


def idx_sha256(idx):
    return hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()


def check_idx(idx, f5_meta):
    h = idx_sha256(idx)
    if h != f5_meta['bootstrap']['idx_sha256']:
        raise Detener(f'idx_sha256 {h} distinto del de la Fase 5 {f5_meta["bootstrap"]["idx_sha256"]}')
    return h


# ── Compuertas ────────────────────────────────────────────────────────────
def _git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True)


def check_versionado(rel):
    if _git('ls-files', '--error-unmatch', rel).returncode != 0:
        raise Detener(f'{rel} no está versionado')
    if _git('diff', '--quiet', 'HEAD', '--', rel).returncode != 0:
        raise Detener(f'{rel} tiene cambios sin commitear')


def check_gate_prueba(val_rel):
    if _git('merge-base', '--is-ancestor', GATE_TAG, 'HEAD').returncode != 0:
        raise Detener(f'la etiqueta {GATE_TAG} no es ancestro de HEAD')
    if _git('diff', '--quiet', GATE_TAG, '--', PLAN).returncode != 0:
        raise Detener(f'{PLAN} cambió frente a {GATE_TAG}')
    check_versionado(F5_RESULT)
    check_versionado(val_rel)
    st = _git('status', '--porcelain')
    if st.returncode != 0 or st.stdout.strip():
        raise Detener('el árbol de trabajo no está limpio:\n' + st.stdout)


def refuse_overwrite(path):
    if Path(path).exists():
        raise Detener(f'{path} ya existe: recuperabilidad nunca sobrescribe')


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(path, obj):
    refuse_overwrite(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


# ── CLI ───────────────────────────────────────────────────────────────────
def compute_val(features_dir, manifest_csv, res, cfg, index, techniques):
    U_all = load_u(index, ('train', 'val'))
    U = {s: U_all[s][0] for s in U_all}
    y = {s: U_all[s][1] for s in U_all}
    lu, m_u = logreg_u(U, y, cfg, index)
    out = {'u': lu, 'tecnicas': {}}
    objs = {'logreg_u': m_u}
    for tech in techniques:
        out['tecnicas'][tech] = {}
        for backbone in [PRINCIPAL[tech]] + SENSIBILIDAD[tech]:
            X, digest = load_features(features_dir, manifest_csv, tech, backbone, index, ('train', 'val'))
            r, o = etapa_val(X, U, y, cfg, res)
            out['tecnicas'][tech][backbone] = r | {'principal': backbone == PRINCIPAL[tech], 'npz_sha256': digest,
                                                   'd': X['train'].shape[1]}
            objs[(tech, backbone)] = o
    return out, objs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--val', action='store_true')
    g.add_argument('--prueba', action='store_true')
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--manifest', default=str(OUT / 'caracteristicas' / 'caracteristicas_manifest.csv'))
    ap.add_argument('--resoluciones', required=True)
    ap.add_argument('--out-dir', default=str(OUT))
    a = ap.parse_args(argv)
    res = check_resoluciones(json.loads(Path(a.resoluciones).read_text(encoding='utf-8')))
    from src.protocol import bootstrap as bs
    from src.protocol import insumos as ins_mod
    cfg = ins_mod.load_config()
    index = load_index()
    out_dir = Path(a.out_dir)
    val_path = out_dir / 'val' / 'resultados_val.json'
    meta = {'task': 'F4.1', 'resoluciones': res, 'git_commit': _git('rev-parse', 'HEAD').stdout.strip(),
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'manifest_sha256': sha256_file(a.manifest), 'kfold': KFOLD, 'clip': CLIP,
            'logreg_max_iter': LOGREG_MAX_ITER}
    if a.val:
        refuse_overwrite(val_path)
        val, _ = compute_val(a.features_dir, a.manifest, res, cfg, index, TECHNIQUES)
        write_json(val_path, {'meta': meta, 'resultados': val})
        print(f'escrito {val_path}')
        return 0

    val_rel = val_path.relative_to(REPO).as_posix()
    check_gate_prueba(val_rel)
    test_path = out_dir / 'prueba' / 'resultados_prueba.json'
    refuse_overwrite(test_path)
    committed = json.loads(val_path.read_text(encoding='utf-8'))['resultados']
    val, objs = compute_val(a.features_dir, a.manifest, res, cfg, index, TECHNIQUES)
    if _clean(val) != committed:
        raise Detener('la etapa val recalculada no coincide con resultados_val.json versionado')
    f5 = json.loads((REPO / F5_RESULT).read_text(encoding='utf-8'))
    idx = bs.indices(cfg)
    meta['idx_sha256'] = check_idx(idx, f5['meta'])
    U_all = load_u(index, ('train', 'test'))
    U_te, y_te = U_all['test']
    p_u = p_dep_of(objs['logreg_u'], U_te)
    out = {'u': {'f1_test': f1_of(y_te, p_u),
                 'f21_test_p_dep_max_abs_diff': float(np.max(np.abs(p_u - load_f21('test', index))))},
           'tecnicas': {}}
    preds = []
    for tech in TECHNIQUES:
        out['tecnicas'][tech] = {}
        for backbone in [PRINCIPAL[tech]] + SENSIBILIDAD[tech]:
            X, _ = load_features(a.features_dir, a.manifest, tech, backbone, index, ('test',))
            o = objs[(tech, backbone)]
            U_hat_te = predict_clip(o['ridge'], X['test'])
            p_hat = p_dep_of(o['logreg_u_hat'], U_hat_te)
            d, _ = delta_f1(y_te, p_u, p_hat, idx, cfg['enmienda_4']['ic']['descriptivo'])
            out['tecnicas'][tech][backbone] = {'principal': backbone == PRINCIPAL[tech],
                                               'r2_test': r2_raw(U_te, U_hat_te).tolist(), **d}
            preds += [{'tecnica': tech, 'backbone': backbone, 'sample_id': r['sample_id'], 'y_true': int(yy),
                       'p_dep_u_hat': float(p), 'p_dep_u': float(pu)}
                      for r, yy, p, pu in zip(index['test'], y_te, p_hat, p_u)]
        prin = out['tecnicas'][tech][PRINCIPAL[tech]]
        out['tecnicas'][tech]['condicion_2_F46'] = prin['contiene_0']
    write_json(test_path, {'meta': meta, 'resultados': out})
    pred_path = out_dir / 'prueba' / 'predicciones_prueba.csv'
    refuse_overwrite(pred_path)
    with open(pred_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(preds)
    print(f'escrito {test_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
