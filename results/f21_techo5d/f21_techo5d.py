"""
f21_techo5d.py
──────────────
F2.1 — Techo 5D (docs/protocolo_preregistrado.md §4.3; configs/protocolo.json
["techo_5d"]). Cinco familias de clasificadores sobre los 5 componentes u de
umap_params.pkl, con selección por F1-Macro de validación.

Reglas (protocolo §4.3 y decisiones de F2.1 no fijadas por el protocolo):
  - Rejillas leídas de configs/protocolo.json; random_state = 42.
  - StandardScaler ajustado solo en train (dentro del pipeline) para logreg,
    svm_rbf y mlp.
  - SVC con probability=True; HistGradientBoosting con early_stopping=False.
  - LogisticRegression con max_iter=5000 (no fijado por el protocolo; igual que
    F2.2). Los demás hiperparámetros no nombrados: valores por defecto de
    scikit-learn.
  - Orden de la rejilla: itertools.product en el orden de claves del config (el
    primer hiperparámetro varía más lento); en SVM, γ='scale' antes del logspace.
  - p_dep = predict_proba[:, clase 1]; F1-Val con ŷ = 1[p_dep > 0.5].
  - Empate (tolerancia 1e-12): familia anterior en family_order y, dentro de
    ella, combinación anterior en el orden de la rejilla.

No calcula ni imprime métricas de test (protocolo §10): las predicciones de
test solo se archivan hasta la Fase 5.

Uso (desde la raíz del repo, .venv, CPU, TT_DATA_ROOT definida):
    python results/f21_techo5d/f21_techo5d.py            # corrida completa
    python results/f21_techo5d/f21_techo5d.py --estimar  # una combinación por familia, sin escribir

Salidas:
    results/predictions/referencia/umap5d__{arch}__none__seed42.csv (+ manifiesto), arch ∈ familias
    results/f21_techo5d/rejilla_completa.csv
    results/f21_techo5d/seleccion.csv
    results/f21_techo5d/f21_techo5d.log
"""

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from src import paths  # noqa: E402
from src.predictions.write import write_predictions  # noqa: E402
from src.training.umap_io import load_umap_arrays, sha256  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CONFIG = REPO / 'configs' / 'protocolo.json'
TASK_ID = 'F2.1'
FASE = 'referencia'
TECHNIQUE = 'umap5d'
STRATEGY = 'none'
PRED_SPLITS = ('val', 'test')
LOGREG_MAX_ITER = 5000  # no fijado por §4.3; igual que F2.2


class Stop(RuntimeError):
    """Verificación que obliga a detener."""


class Log:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, msg=''):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def write(self):
        self.path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8', newline='\n')


def text_sha256(path):
    """SHA-256 con fines de línea LF (independiente de core.autocrlf)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def verify_file(rel, log):
    """SHA-256 de un archivo de TT_DATA_ROOT contra su fila de meta/data_manifest.csv."""
    with open(paths.DATA_MANIFEST, newline='', encoding='utf-8') as f:
        row = next(r for r in csv.DictReader(f) if r['rel_path'] == rel and r['kind'] == 'file')
    digest = sha256(paths.resolve(rel))
    if digest != row['sha256']:
        raise Stop(f'{rel}: sha256 {digest} != manifiesto {row["sha256"]}')
    log(f'  sha256 OK  {rel}  {digest}')
    return digest


def load_index():
    idx = pd.read_csv(paths.SAMPLE_INDEX, dtype={'sample_id': str, 'split': str})
    return {s: idx[idx['split'] == s].sort_values('idx').reset_index(drop=True) for s in paths.SPLITS}


# ── Rejillas (configs/protocolo.json §4.3) ────────────────────────────────
def _logspace(spec):
    return [float(v) for v in np.logspace(spec['log10_low'], spec['log10_high'], spec['n'])]


def grid_axes(fam, spec):
    """Ejes de la rejilla en el orden de claves del config (el primero varía más lento)."""
    if fam == 'logreg':
        return [('C', _logspace(spec['C'])), ('class_weight', spec['class_weight'])]
    if fam == 'svm_rbf':
        g = spec['gamma']
        return [('C', _logspace(spec['C'])), ('gamma', list(g['named']) + _logspace(g)),
                ('class_weight', spec['class_weight'])]
    if fam == 'rf':
        return [('max_depth', spec['max_depth']), ('min_samples_leaf', spec['min_samples_leaf']),
                ('class_weight', spec['class_weight'])]
    if fam == 'gbm':
        return [('learning_rate', spec['learning_rate']), ('max_depth', spec['max_depth']),
                ('max_iter', spec['max_iter'])]
    if fam == 'mlp':
        return [('hidden_layer_sizes', [tuple(h) for h in spec['hidden_layer_sizes']]),
                ('alpha', spec['alpha'])]
    raise Stop(f'familia desconocida {fam}')


def combinations(fam, spec):
    axes = grid_axes(fam, spec)
    names = [n for n, _ in axes]
    return [dict(zip(names, vals)) for vals in itertools.product(*(v for _, v in axes))]


def fixed_params(fam, spec, rs):
    """Hiperparámetros fijos de la familia (no varían en la rejilla)."""
    if fam == 'logreg':
        assert spec['estimator'] == 'LogisticRegression'
        return {'penalty': spec['penalty'], 'solver': spec['solver'], 'max_iter': LOGREG_MAX_ITER,
                'random_state': rs}
    if fam == 'svm_rbf':
        assert spec['estimator'] == 'SVC'
        return {'kernel': spec['kernel'], 'probability': spec['probability'], 'random_state': rs}
    if fam == 'rf':
        assert spec['estimator'] == 'RandomForestClassifier'
        return {'n_estimators': spec['n_estimators'], 'random_state': rs}
    if fam == 'gbm':
        assert spec['estimator'] == 'HistGradientBoostingClassifier'
        return {'early_stopping': False, 'random_state': rs}
    if fam == 'mlp':
        assert spec['estimator'] == 'MLPClassifier'
        return {'max_iter': spec['max_iter'], 'random_state': rs}
    raise Stop(f'familia desconocida {fam}')


ESTIMATORS = {'logreg': LogisticRegression, 'svm_rbf': SVC, 'rf': RandomForestClassifier,
              'gbm': HistGradientBoostingClassifier, 'mlp': MLPClassifier}


def make_model(fam, fixed, combo, scaled):
    est = ESTIMATORS[fam](**fixed, **combo)
    return make_pipeline(StandardScaler(), est) if scaled else make_pipeline(est)


def jsonable(combo):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in combo.items()}


def n_iter_of(model):
    est = model[-1]
    if hasattr(est, 'n_iter_'):
        n = est.n_iter_
        return int(np.max(n)) if np.ndim(n) else int(n)
    return None


def p_dep_of(model, X):
    pos = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(X)[:, pos].astype(np.float64)


def f1_val_of(y, p):
    return float(f1_score(y, (p > 0.5).astype(int), average='macro'))


def select(f1s, tol):
    """Primer índice cuyo F1 está a ≤ tol del máximo (empate → anterior en la rejilla)."""
    best = max(f1s)
    return next(i for i, f in enumerate(f1s) if f >= best - tol)


# ── Principal ─────────────────────────────────────────────────────────────
def main(estimar, log):
    t0 = time.time()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    tcfg = cfg['techo_5d']
    rs = tcfg['random_state']
    tol = tcfg['selection']['tie_tolerance']
    order = tcfg['family_order']
    scaled_fams = set(tcfg['scaler']['families'])
    if (tcfg['scaler']['name'], tcfg['scaler']['fit_split']) != ('StandardScaler', 'train'):
        raise Stop(f'escalador inesperado {tcfg["scaler"]}')
    if (tcfg['selection']['metric'], tcfg['selection']['split']) != ('f1_macro', 'val'):
        raise Stop(f'selección inesperada {tcfg["selection"]}')
    if tcfg['selection']['tie_break'] != ['family_order', 'grid_order']:
        raise Stop(f'desempate inesperado {tcfg["selection"]["tie_break"]}')

    log(f'F2.1 — Techo 5D. scikit-learn {sklearn.__version__}, numpy {np.__version__}, '
        f'pandas {pd.__version__}' + ('  [--estimar: no escribe nada]' if estimar else ''))
    log('\n— Integridad')
    inputs = {paths.REL['UMAP_PARAMS']: verify_file(paths.REL['UMAP_PARAMS'], log),
              'meta/sample_index.csv (repo)': sha256(paths.SAMPLE_INDEX),
              'configs/protocolo.json (repo, LF)': text_sha256(CONFIG)}
    log(f'  sha256     meta/sample_index.csv  {inputs["meta/sample_index.csv (repo)"]}')
    log(f'  sha256 LF  configs/protocolo.json  {inputs["configs/protocolo.json (repo, LF)"]}')

    index = load_index()
    u = load_umap_arrays(paths.umap_params_path())
    X, y = {}, {}
    for s in paths.SPLITS:
        X[s] = np.asarray(u[s]['params'], dtype=np.float64)
        y[s] = np.asarray(u[s]['labels']).astype(int)
        if len(index[s]) != paths.EXPECTED_SIZES[s] or not np.array_equal(y[s], index[s]['y_true'].values):
            raise Stop(f'{s}: etiquetas de umap_params.pkl ≠ índice maestro')
    log('  formas (908/114/114, 5) y etiquetas de umap_params.pkl = índice maestro')

    log('\n— Rejilla (F1-Macro de validación con ŷ = 1[p_dep > 0.5])')
    rows, best = [], {}
    for fam in order:
        spec = tcfg['familias'][fam]
        fixed = fixed_params(fam, spec, rs)
        combos = combinations(fam, spec)
        if estimar:
            combos = combos[:1]
        log(f'\n  [{fam}] {len(combos)} combinaciones; fijos {json.dumps(jsonable(fixed))}; '
            f'escalador: {"StandardScaler (train)" if fam in scaled_fams else "ninguno"}')
        fitted, f1s = [], []
        tf = time.time()
        for i, combo in enumerate(combos):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', ConvergenceWarning)
                model = make_model(fam, fixed, combo, fam in scaled_fams).fit(X['train'], y['train'])
            conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
            f1 = f1_val_of(y['val'], p_dep_of(model, X['val']))
            fitted.append(model)
            f1s.append(f1)
            rows.append({'familia': fam, 'orden': i, 'combinacion': json.dumps(jsonable(combo)),
                         'f1_val': f1, 'n_iter': n_iter_of(model), 'avisos_convergencia': conv})
            log(f'    {i:>2} {json.dumps(jsonable(combo))}  F1-Val = {f1:.6f}'
                + (f'  n_iter={n_iter_of(model)}' if n_iter_of(model) is not None else '')
                + (f'  ConvergenceWarning×{conv}' if conv else ''))
        k = select(f1s, tol)
        n_tie = sum(f >= max(f1s) - tol for f in f1s)
        best[fam] = {'orden': k, 'combo': combos[k], 'f1_val': f1s[k], 'model': fitted[k],
                     'fixed': fixed, 'n_empate': n_tie}
        log(f'  [{fam}] seleccionada #{k} {json.dumps(jsonable(combos[k]))}  F1-Val = {f1s[k]:.6f} '
            f'({n_tie} combinación(es) dentro de la tolerancia); {time.time() - tf:.1f} s')

    techo = order[select([best[f]['f1_val'] for f in order], tol)]
    log(f'\n— Techo 5D: {techo}  F1-Val = {best[techo]["f1_val"]:.6f}')
    for f in order:
        log(f'  {f:<8} F1-Val = {best[f]["f1_val"]:.6f}' + ('  ← techo' if f == techo else ''))

    svm = best['svm_rbf']['model']
    p_svm = p_dep_of(svm, X['val'])
    n_diff = int(((p_svm > 0.5).astype(int) != svm.predict(X['val'])).sum())
    f1_predict = float(f1_score(y['val'], svm.predict(X['val']), average='macro'))
    log(f'\n— SVM seleccionado: predicciones de val que difieren entre predict() y p_dep > 0.5: '
        f'{n_diff}/114 (F1-Val con predict() = {f1_predict:.6f}, no usado en la selección)')

    if estimar:
        log(f'\nDuración (estimación, 1 combinación por familia): {time.time() - t0:.1f} s')
        return

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / 'rejilla_completa.csv', index=False, lineterminator='\n')
    sel = pd.DataFrame([{'familia': f, 'orden': best[f]['orden'],
                         'combinacion': json.dumps(jsonable(best[f]['combo'])),
                         'f1_val': best[f]['f1_val'], 'n_empate_tolerancia': best[f]['n_empate'],
                         'techo': f == techo} for f in order])
    sel.to_csv(OUT_DIR / 'seleccion.csv', index=False, lineterminator='\n')
    log('\n  escrito results/f21_techo5d/rejilla_completa.csv y seleccion.csv')

    log('\n— Predicciones (fase referencia; test archivado sin métricas)')
    for fam in order:
        b = best[fam]
        frames = []
        for s in PRED_SPLITS:
            p = p_dep_of(b['model'], X[s])
            if s == 'val' and f1_val_of(y[s], p) != b['f1_val']:
                raise RuntimeError(f'{fam}: p_dep > 0.5 no reproduce el F1-Val de la selección')
            frames.append(pd.DataFrame({'sample_id': index[s]['sample_id'], 'split': s, 'y_true': y[s],
                                        'technique': TECHNIQUE, 'arch': fam, 'strategy': STRATEGY,
                                        'seed': rs, 'p_dep': p}))
        df = pd.concat(frames, ignore_index=True)
        config = {'features': 'umap_params.pkl, 5 componentes u (params)',
                  'pipeline': ('StandardScaler (ajustado en train) + ' if fam in scaled_fams else '')
                  + ESTIMATORS[fam].__name__,
                  'estimator_params': jsonable({**b['fixed'], **b['combo']}),
                  'other_params': f'valores por defecto de scikit-learn {sklearn.__version__}',
                  'fit': 'solo train', 'decision': 'p_dep > 0.5',
                  'protocol': 'docs/protocolo_preregistrado.md §4.3',
                  'protocol_config': 'configs/protocolo.json',
                  'protocol_config_sha256_lf': inputs['configs/protocolo.json (repo, LF)'],
                  'script': 'results/f21_techo5d/f21_techo5d.py',
                  'script_sha256_lf': text_sha256(Path(__file__))}
        selection = {'criterio': ('F1-Macro de validación (ŷ = 1[p_dep > 0.5]) sobre la rejilla de §4.3; '
                                  'empate con tolerancia 1e-12 → combinación anterior en la rejilla'),
                     'orden_en_rejilla': b['orden'], 'combinacion': jsonable(b['combo']),
                     'f1_val': b['f1_val'], 'techo_5d': fam == techo,
                     'tabla': 'results/f21_techo5d/rejilla_completa.csv'}
        csv_path = write_predictions(df, fase=FASE, task_id=TASK_ID, config=config, inputs=inputs,
                                     model_selection=selection)
        back = pd.read_csv(csv_path, float_precision='round_trip')
        if not np.array_equal(back['p_dep'].values, df['p_dep'].values):
            raise RuntimeError(f'{fam}: p_dep no hace ida y vuelta exacta por el CSV')
        log(f'  escrito {csv_path.relative_to(paths.REPO).as_posix()} ({len(df)} filas)')
    log(f'\nDuración: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--estimar', action='store_true', help='una combinación por familia; no escribe nada')
    args = ap.parse_args()
    log = Log(OUT_DIR / ('f21_estimacion.log' if args.estimar else 'f21_techo5d.log'))
    try:
        main(args.estimar, log)
    except Stop as e:
        log(f'\nDETENIDO: {e}')
        raise SystemExit(2)
    finally:
        if not args.estimar:
            log.write()
