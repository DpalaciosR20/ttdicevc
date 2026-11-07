"""
f45_longitud.py
───────────────
F4.5 — Referencia de longitud (docs/protocolo_preregistrado.md §4.2 y §6;
Enmienda 2, punto 4; Enmienda 4, A9). Solo ajusta el clasificador y archiva
sus predicciones: el contraste L_t es de la Fase 5.

  technique text_length, arch logreg, strategy none, seed 42, fase referencia.
  Características (meta/sample_index.csv): log natural de n_tokens_beto y
  truncated_128 → {0, 1}. LogReg con la rejilla completa de §4.3
  (techo_5d.familias.logreg: 13 C × class_weight {None, balanced}),
  StandardScaler ajustado solo en train (dentro del pipeline), selección por
  F1-Macro de validación con la regla de §4.3 (tolerancia 1e-12 → combinación
  anterior en la rejilla).

Reutiliza las funciones de results/f23_pisos/f23_pisos.py (las mismas de F2.1):
logreg_grid, p_dep_of, f1_val_of, select, n_iter_of, load_index. Todos los
valores salen de configs/protocolo.json (referencias.longitud,
enmienda_4.longitud.clasificador, techo_5d); ninguno se escribe a mano.

No calcula, imprime ni registra ninguna métrica de test (protocolo §10): las
predicciones de test se archivan sin evaluarlas. tests/test_f45_longitud.py lo
comprueba sobre el log y el JSON de resultados.

Uso (desde la raíz del repo, .venv, CPU):
    python results/f45_longitud/f45_longitud.py

Salidas:
    results/predictions/referencia/text_length__logreg__none__seed42.csv (+ manifiesto)
    results/f45_longitud/rejilla_longitud.csv   (F1-Val por combinación)
    results/f45_longitud/resultado.json         (selección y F1-Val; sin test)
    results/f45_longitud/f45_longitud.log
"""

import importlib.util
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
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import paths  # noqa: E402
from src.predictions.write import write_predictions  # noqa: E402
from src.training.umap_io import sha256  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CONFIG = REPO / 'configs' / 'protocolo.json'
TASK_ID = 'F4.5'
PRED_SPLITS = ('val', 'test')


def _load_f23():
    spec = importlib.util.spec_from_file_location('f23_pisos', REPO / 'results' / 'f23_pisos' / 'f23_pisos.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f23 = _load_f23()
Stop, Log, text_sha256 = f23.Stop, f23.Log, f23.text_sha256


def dig(cfg, dotted):
    for k in dotted.split('.'):
        cfg = cfg[k]
    return cfg


def load_spec(cfg):
    """Valores de configs/protocolo.json, comprobados contra lo que este script implementa."""
    ref = cfg['referencias']['longitud']
    clf = cfg['enmienda_4']['longitud']['clasificador']
    grid = dig(cfg, ref['grid'])
    sel = dig(cfg, ref['selection_rule'])
    checks = {
        'features': (ref['features'], ['log(n_tokens_beto)', 'truncated_128']),
        'scaler': (ref['scaler'], {'name': 'StandardScaler', 'fit_split': 'train'}),
        'identidad': ((ref['technique'], ref['arch'], ref['strategy'], ref['fase']),
                      ('text_length', 'logreg', 'none', 'referencia')),
        'log': (clf['log'], 'natural'),
        'truncated_128': (clf['truncated_128'], {'false': 0, 'true': 1}),
        'seleccion': ((clf['seleccion'], clf['split_seleccion'], sel['metric'], sel['split']),
                      (ref['selection_rule'], 'val', 'f1_macro', 'val')),
        'archiva': (clf['archiva'], list(PRED_SPLITS)),
        'metricas_al_ajustar': (clf['metricas_al_ajustar'], ['f1_val']),
        'estimator': (grid['estimator'], 'LogisticRegression'),
        'semilla': (ref['seed'], cfg['techo_5d']['random_state']),
    }
    bad = {k: v for k, (v, want) in checks.items() if v != want}
    if bad:
        raise Stop(f'configs/protocolo.json pide algo distinto de lo implementado: {bad}')
    if clf['max_iter'] != f23.LOGREG_MAX_ITER:
        raise Stop(f'max_iter {clf["max_iter"]} distinto del de F2.1/F2.3 ({f23.LOGREG_MAX_ITER})')
    fixed = {'penalty': grid['penalty'], 'solver': grid['solver'], 'max_iter': clf['max_iter'],
             'random_state': ref['seed']}
    return ref, clf, grid, sel, fixed


def features(df, clf):
    """[ln(n_tokens_beto), truncated_128 → {0, 1}] en el orden de df."""
    n = df['n_tokens_beto'].astype(np.int64).to_numpy()
    if (n <= 0).any():
        raise Stop('n_tokens_beto ≤ 0')
    tr = df['truncated_128'].map(lambda v: str(v).strip().lower()).map(clf['truncated_128'])
    if tr.isna().any():
        raise Stop(f'valores de truncated_128 fuera de {list(clf["truncated_128"])}')
    return np.column_stack([np.log(n.astype(np.float64)), tr.to_numpy(dtype=np.float64)])


def main(log):
    t0 = time.time()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    ref, clf, grid, sel, fixed = load_spec(cfg)
    log(f'F4.5 — Referencia de longitud. scikit-learn {sklearn.__version__}, numpy {np.__version__}, '
        f'pandas {pd.__version__}')
    inputs = {'meta/sample_index.csv (repo)': sha256(paths.SAMPLE_INDEX),
              'configs/protocolo.json (repo, LF)': text_sha256(CONFIG)}
    for k, v in inputs.items():
        log(f'  sha256  {k}  {v}')

    index = f23.load_index()
    y = {s: index[s]['y_true'].astype(int).to_numpy() for s in paths.SPLITS}
    X = {s: features(index[s], clf) for s in paths.SPLITS}
    log(f'  índice maestro: train {len(y["train"])}, val {len(y["val"])}, test {len(y["test"])}; '
        f'características {ref["features"]} (ln, {{False: 0, True: 1}})')

    combos = f23.logreg_grid(grid)
    tol = sel['tie_tolerance']
    log(f'\n— Rejilla: {len(combos)} combinaciones; fijos {json.dumps(fixed)}; StandardScaler (train)')
    fitted, f1s, rows = [], [], []
    for i, combo in enumerate(combos):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            model = make_pipeline(StandardScaler(), LogisticRegression(**fixed, **combo)).fit(X['train'], y['train'])
        conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        f1 = f23.f1_val_of(y['val'], f23.p_dep_of(model, X['val']))
        fitted.append(model)
        f1s.append(f1)
        rows.append({'orden': i, 'combinacion': json.dumps(combo), 'f1_val': f1,
                     'n_iter': f23.n_iter_of(model), 'avisos_convergencia': conv})
        log(f'  {i:>2} {json.dumps(combo)}  F1-Val = {f1:.6f}  n_iter={f23.n_iter_of(model)}'
            + (f'  ConvergenceWarning×{conv}' if conv else ''))
    k = f23.select(f1s, tol)
    n_tie = sum(f >= max(f1s) - tol for f in f1s)
    best = fitted[k]
    log(f'\n  seleccionada #{k} {json.dumps(combos[k])}  F1-Val = {f1s[k]:.6f} '
        f'({n_tie} combinación(es) dentro de la tolerancia {tol})')

    p = {s: f23.p_dep_of(best, X[s]) for s in PRED_SPLITS}
    if f23.f1_val_of(y['val'], p['val']) != f1s[k]:
        raise RuntimeError('p_dep > 0.5 no reproduce el F1-Val de la selección')

    pd.DataFrame(rows).to_csv(OUT_DIR / 'rejilla_longitud.csv', index=False, lineterminator='\n')
    log('\n— Predicciones (fase referencia, seed 42; test archivado sin métricas)')
    df = pd.concat([pd.DataFrame({'sample_id': index[s]['sample_id'], 'split': s, 'y_true': y[s],
                                  'technique': ref['technique'], 'arch': ref['arch'], 'strategy': ref['strategy'],
                                  'seed': ref['seed'], 'p_dep': p[s]}) for s in PRED_SPLITS], ignore_index=True)
    config = {'features': 'ln(n_tokens_beto), truncated_128 → {0, 1} (meta/sample_index.csv)',
              'pipeline': 'StandardScaler (ajustado en train) + LogisticRegression',
              'estimator_params': {**fixed, **combos[k]},
              'other_params': f'valores por defecto de scikit-learn {sklearn.__version__}',
              'fit': 'solo train',
              'script': 'results/f45_longitud/f45_longitud.py', 'script_sha256_lf': text_sha256(Path(__file__)),
              'reused': 'results/f23_pisos/f23_pisos.py (logreg_grid, p_dep_of, f1_val_of, select, load_index)',
              'protocol': 'docs/protocolo_preregistrado.md §4.2, §6; Enmienda 2, punto 4; Enmienda 4, A9',
              'protocol_config': 'configs/protocolo.json',
              'protocol_config_sha256_lf': inputs['configs/protocolo.json (repo, LF)'], 'decision': 'p_dep > 0.5'}
    selection = {'criterio': ('F1-Macro de validación (ŷ = 1[p_dep > 0.5]) sobre la rejilla completa de LogReg '
                              'de §4.3; empate con tolerancia 1e-12 → combinación anterior'),
                 'orden_en_rejilla': k, 'combinacion': combos[k], 'f1_val': f1s[k],
                 'tabla': 'results/f45_longitud/rejilla_longitud.csv'}
    path = write_predictions(df, fase=ref['fase'], task_id=TASK_ID, config=config, inputs=inputs,
                             model_selection=selection)
    back = pd.read_csv(path, float_precision='round_trip')
    if not np.array_equal(back['p_dep'].values, df['p_dep'].values):
        raise RuntimeError(f'{path.name}: p_dep no hace ida y vuelta exacta por el CSV')
    log(f'  escrito {path.relative_to(paths.REPO).as_posix()} ({len(df)} filas: '
        + ', '.join(f'{s} {int((df["split"] == s).sum())}' for s in PRED_SPLITS) + ')')

    result = {'task_id': TASK_ID, 'predictions': path.relative_to(paths.REPO).as_posix(),
              'n_combinaciones': len(combos), 'seleccion': selection, 'n_empate_tolerancia': n_tie,
              'f1_val': f1s[k], 'splits_evaluados': ['val'], 'estimator_params': {**fixed, **combos[k]},
              'inputs': inputs, 'sklearn': sklearn.__version__}
    (OUT_DIR / 'resultado.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n',
                                           encoding='utf-8', newline='\n')
    log('  escrito results/f45_longitud/resultado.json y rejilla_longitud.csv')
    log(f'\nDuración: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    log = Log(OUT_DIR / 'f45_longitud.log')
    try:
        main(log)
    except Stop as e:
        log(f'\nDETENIDO: {e}')
        raise SystemExit(2)
    finally:
        log.write()
