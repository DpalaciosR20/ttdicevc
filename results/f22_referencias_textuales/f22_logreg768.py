"""
f22_logreg768.py
────────────────
F2.2 — Clasificador sobre los embeddings de 768 dimensiones de Exp 4.1
(docs/especificacion_f22.md §2): StandardScaler (ajustado solo en train) +
LogisticRegression (L2, lbfgs, max_iter=5000, sin pesos de clase,
random_state=42). C ∈ 10^[−4, 2], 13 valores; selección por F1-Macro de
validación, en empate el menor C. Ajuste final solo sobre train.

No calcula ni imprime métricas de test (§4): las predicciones de test solo
se archivan hasta la Fase 5.

Uso (desde la raíz del repo, .venv, CPU):
    python results/f22_referencias_textuales/f22_logreg768.py

Salidas:
    results/predictions/referencia/emb768__logreg__none__seed42.csv (+ manifiesto)
    results/f22_referencias_textuales/seleccion_C.csv
    results/f22_referencias_textuales/f22_logreg768.log
"""

import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (FASE, OUT_DIR, PRED_SPLITS, SEED, TASK_ID, Log, Stop, load_index, paths,
                    sha256, verify_file)
from src.predictions.write import write_predictions

C_GRID = np.logspace(-4, 2, 13)
EMB_KEYS = {'train': ('X_train', 'y_train'), 'val': ('X_val', 'y_val'), 'test': ('X_test', 'y_test')}

log = Log(OUT_DIR / 'f22_logreg768.log')


def make_model(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=5000,
                                            class_weight=None, random_state=SEED))


def main():
    t0 = time.time()
    log(f'F2.2 — LogReg sobre 768D (EMB_41). scikit-learn {sklearn.__version__}')
    log('\n— Integridad')
    inputs = {paths.REL['EMB_41']: verify_file(paths.REL['EMB_41'], log),
              'meta/sample_index.csv (repo)': sha256(paths.SAMPLE_INDEX)}

    index = load_index()
    emb = joblib.load(paths.emb_41_path())
    X, y = {}, {}
    for s, (kx, ky) in EMB_KEYS.items():
        X[s] = np.asarray(emb[kx], dtype=np.float64)
        y[s] = np.asarray(emb[ky]).astype(int)
        if X[s].shape != (paths.EXPECTED_SIZES[s], 768):
            raise Stop(f'{s}: forma {X[s].shape}')
        if not np.array_equal(y[s], index[s]['y_true'].values):
            raise Stop(f'{s}: etiquetas de EMB_41 ≠ índice maestro')
    log('  formas (908/114/114, 768) y etiquetas = índice maestro')

    log('\n— Selección de C por F1-Macro de validación')
    rows = []
    for C in C_GRID:
        model = make_model(C).fit(X['train'], y['train'])
        f1 = f1_score(y['val'], model.predict(X['val']), average='macro')
        n_iter = int(model[-1].n_iter_[0])
        rows.append({'C': C, 'log10_C': np.log10(C), 'f1_val': f1, 'n_iter': n_iter})
        log(f'  C = {C:.6g}  F1-Val = {f1:.6f}  (iteraciones {n_iter})')
    table = pd.DataFrame(rows)
    best_f1 = table['f1_val'].max()
    C_star = float(table.loc[table['f1_val'] == best_f1, 'C'].min())  # empate exacto → menor C
    n_tie = int((table['f1_val'] == best_f1).sum())
    table['selected'] = table['C'] == C_star
    table.to_csv(OUT_DIR / 'seleccion_C.csv', index=False, lineterminator='\n')
    log(f'  seleccionado C = {C_star:.6g} (F1-Val = {best_f1:.6f}; {n_tie} valor(es) con ese F1)')

    model = make_model(C_star).fit(X['train'], y['train'])
    pos = int(np.flatnonzero(model.classes_ == 1)[0])
    frames = []
    for s in PRED_SPLITS:
        p = model.predict_proba(X[s])[:, pos].astype(np.float64)
        if s == 'val' and f1_score(y[s], (p > 0.5).astype(int), average='macro') != best_f1:
            raise RuntimeError('la regla p_dep > 0.5 no reproduce el F1-Val de la selección')
        frames.append(pd.DataFrame({'sample_id': index[s]['sample_id'], 'split': s, 'y_true': y[s],
                                    'technique': 'emb768', 'arch': 'logreg', 'strategy': 'none',
                                    'seed': SEED, 'p_dep': p}))
    df = pd.concat(frames, ignore_index=True)

    config = {'features': 'EMB_41 [CLS] 768D (X_train/X_val/X_test)',
              'pipeline': 'StandardScaler (ajustado en train) + LogisticRegression',
              'logreg': {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 5000, 'class_weight': None,
                         'random_state': SEED, 'C': C_star},
              'fit': 'solo train (sin reajuste con train + val)', 'decision': 'p_dep > 0.5',
              'script': 'results/f22_referencias_textuales/f22_logreg768.py',
              'script_sha256': sha256(OUT_DIR / 'f22_logreg768.py')}
    selection = {'criterio': 'F1-Macro de validación sobre C ∈ logspace(-4, 2, 13); empate → menor C',
                 'C': C_star, 'f1_val': best_f1, 'tabla': 'results/f22_referencias_textuales/seleccion_C.csv'}
    csv_path = write_predictions(df, fase=FASE, task_id=TASK_ID, config=config, inputs=inputs,
                                 model_selection=selection)
    back = pd.read_csv(csv_path)
    if not np.array_equal(back['p_dep'].values, df['p_dep'].values):
        raise RuntimeError('p_dep no hace ida y vuelta exacta por el CSV')
    log(f'\n  escrito {csv_path.relative_to(paths.REPO).as_posix()} ({len(df)} filas; test archivado, '
        f'sin métricas hasta la Fase 5)')
    log(f'Duración: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    try:
        main()
    except Stop as e:
        log(f'\nDETENIDO: {e}')
        raise SystemExit(2)
    finally:
        log.write()
