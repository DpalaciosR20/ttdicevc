"""
verificar_insumos.py — F3.3 etapa A, puntos 1 (b–d, f) y 2.

Verificación independiente de results/predictions/primera_pasada/ (54 CSV):
  1b) todas las filas con split == val;
  1c) sample_id == los 114 de val de meta/sample_index.csv, y_true == índice;
  1d) manifiestos frente a configs/preprocesamiento.json y
      configs/matriz_experimentos.json (y sus sha256 en 36c6398);
  1f) evaluate_test / test_* en los JSON de runs/;
  2)  F1-Macro desde p_dep (> 0.5) frente a runs/*.json, manifiesto y summary.csv;
      p_dep frente a val_probs bit a bit; conteo de |p_dep − 0.5| < 1e-6.

No usa filas de test: el índice se filtra a split == 'val' justo al leerlo.
No importa código del repo (solo pandas, numpy, sklearn).

Uso (desde la raíz del repo):
    python results/f33_seleccion/verificar_insumos.py
"""

import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
PRED = REPO / 'results' / 'predictions' / 'primera_pasada'
F32 = REPO / 'results' / 'f32_primera_pasada'
OUT = REPO / 'results' / 'f33_seleccion'
COMMIT = '36c6398d894bdb4dfedc5252b6e5957d4620cd9c'

problems = []


def bad(msg):
    problems.append(msg)
    print('PROBLEMA:', msg)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_show(rev_path):
    return subprocess.run(['git', 'show', rev_path], cwd=REPO, capture_output=True, check=True).stdout


# Índice: solo val (las filas de test se descartan al leer y no se usan).
idx = pd.read_csv(REPO / 'meta' / 'sample_index.csv', usecols=['sample_id', 'split', 'y_true'], dtype=str)
idx = idx[idx['split'] == 'val'].copy()
idx['y_true'] = idx['y_true'].astype(int)
val_ids = set(idx['sample_id'])
val_y = dict(zip(idx['sample_id'], idx['y_true']))
assert len(val_ids) == 114, len(val_ids)

prep = json.loads((REPO / 'configs' / 'preprocesamiento.json').read_text(encoding='utf-8'))
mat = json.loads((REPO / 'configs' / 'matriz_experimentos.json').read_text(encoding='utf-8'))
cc = mat['configuracion_comun']
seeds_ok = set(mat['fases']['primera_pasada']['seeds'])

sha_now = {k: sha_bytes((REPO / 'configs' / f).read_bytes())
           for k, f in [('prep', 'preprocesamiento.json'), ('mat', 'matriz_experimentos.json')]}
sha_at_commit = {k: sha_bytes(git_show(f'{COMMIT}:configs/{f}'))
                 for k, f in [('prep', 'preprocesamiento.json'), ('mat', 'matriz_experimentos.json')]}
print('sha256 configs actual   :', sha_now)
print('sha256 configs @36c6398 :', sha_at_commit)

summary = pd.read_csv(F32 / 'summary.csv')
csvs = sorted(glob.glob(str(PRED / '*.csv')))
print('CSV encontrados:', len(csvs))
if len(csvs) != 54:
    bad(f'se esperaban 54 CSV, hay {len(csvs)}')

rows = []
created = []
manifest_values = {}
for c in csvs:
    c = Path(c)
    tech, arch, strat, sd = c.stem.split('__')
    seed = int(sd.replace('seed', ''))
    df = pd.read_csv(c, float_precision='round_trip')
    df_txt = pd.read_csv(c, dtype=str)

    # 1b
    splits = set(df['split'])
    if splits != {'val'}:
        bad(f'{c.name}: splits {splits}')
    # nombre vs contenido
    for col, v in [('technique', tech), ('arch', arch), ('strategy', strat)]:
        if set(df[col]) != {v}:
            bad(f'{c.name}: {col} {set(df[col])}')
    if set(df['seed']) != {seed}:
        bad(f'{c.name}: seed {set(df["seed"])}')
    # 1c
    ids = list(df['sample_id'])
    if set(ids) != val_ids or len(ids) != 114:
        bad(f'{c.name}: sample_id distinto de val del índice (n={len(ids)})')
    if any(df['sample_id'].map(val_y) != df['y_true']):
        bad(f'{c.name}: y_true distinto del índice')

    # 1d manifiesto
    man = json.loads(c.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    cfg = man['config']
    created.append(man['created_at'])
    exp_lr = cc['lr'][strat]
    checks = {
        'git_commit': (man['git_commit'], COMMIT),
        'fase': (man['fase'], 'primera_pasada'),
        'variant': (cfg['variant'], prep[tech]['variant']),
        'preprocessing': (cfg['preprocessing'], prep[tech]),
        'optimizer': (cfg['optimizer'], cc['optimizer']['name']),
        'lr': (cfg['lr'], exp_lr),
        'weight_decay': (cfg['weight_decay'], cc['weight_decay']),
        'batch_size': (cfg['batch_size'], cc['batch_size']),
        'max_epochs': (cfg['max_epochs'], cc['max_epochs']),
        'patience': (cfg['patience'], cc['early_stopping']['patience']),
        'loss': (cfg['loss'], cc['loss']['name']),
        'permute_labels': (cfg['permute_labels'], None),
        'smoke': (cfg['smoke'], False),
        'prep_sha_vs_commit': (cfg['preprocessing_file']['sha256'], sha_at_commit['prep']),
        'mat_sha_vs_commit': (cfg['matrix_file']['sha256'], sha_at_commit['mat']),
        'csv_sha256': (man['csv_sha256'], sha_bytes(c.read_bytes())),
        'n_rows': (man['n_rows'], {'val': 114}),
    }
    for k, (got, exp) in checks.items():
        if got != exp:
            bad(f'{c.name}: manifiesto {k} = {got!r}; se espera {exp!r}')
    if seed not in seeds_ok:
        bad(f'{c.name}: semilla {seed} fuera de {seeds_ok}')
    cw = cfg['class_weights']
    manifest_values.setdefault('class_weights', set()).add(tuple(cw))
    manifest_values.setdefault('determinism', set()).add(json.dumps(cfg['determinism'], sort_keys=True))
    manifest_values.setdefault('environment', set()).add(json.dumps(man['environment'], sort_keys=True))

    # JSON de runs
    rj_path = F32 / 'runs' / cfg['run_json']
    rj = json.loads(rj_path.read_text(encoding='utf-8'))
    for k in ['evaluate_test', 'f1_test', 'acc_test', 'n_test', 'test_preds', 'test_probs', 'test_true']:
        if rj.get(k) not in (None, False):
            bad(f'{rj_path.name}: {k} = {rj.get(k)!r}')
    if rj['seed'] != seed or rj['technique'] != tech or rj['arch'].replace('vit_b_16', 'vit_b16') != arch:
        bad(f'{rj_path.name}: identidad distinta del CSV')

    # 2: F1 independiente
    p = df['p_dep'].to_numpy(dtype=float)
    y = df['y_true'].to_numpy(dtype=int)
    yhat = (p > 0.5).astype(int)
    f1 = float(f1_score(y, yhat, labels=[0, 1], average='macro', zero_division=0))
    vp = np.asarray(rj['val_probs'], dtype=float)
    bitwise = bool(np.array_equal(p.view(np.uint64), vp.view(np.uint64)))
    # p_dep del lector por omisión de pandas (H1 de F3.2)
    p_default = pd.read_csv(c)['p_dep'].to_numpy(dtype=float)
    n_default_diff = int((p_default != p).sum())
    vt = np.asarray(rj['val_true'], dtype=int)
    vpred = np.asarray(rj['val_preds'], dtype=int)
    s = summary[(summary['technique'] == tech) & (summary['arch'].replace('vit_b_16', 'vit_b16') == arch)
                & (summary['strategy'] == strat) & (summary['seed'] == seed)]
    if len(s) != 1:
        bad(f'{c.name}: {len(s)} filas en summary.csv')
    rows.append({
        'technique': tech, 'arch': arch, 'strategy': strat, 'seed': seed,
        'f1_recalc': f1, 'f1_json': rj['best_f1_val'], 'f1_manifest': man['model_selection']['best_f1_val'],
        'f1_summary': float(s['f1_val'].iloc[0]),
        'd_json': abs(f1 - rj['best_f1_val']), 'd_manifest': abs(f1 - man['model_selection']['best_f1_val']),
        'd_summary': abs(f1 - float(s['f1_val'].iloc[0])),
        'p_dep_eq_val_probs_bitwise': bitwise,
        'n_pdep_default_reader_differs': n_default_diff,
        'y_true_eq_val_true': bool(np.array_equal(y, vt)),
        'yhat_eq_val_preds': bool(np.array_equal(yhat, vpred)),
        'n_near_half_1e-6': int((np.abs(p - 0.5) < 1e-6).sum()),
        'min_abs_pdep_minus_half': float(np.abs(p - 0.5).min()),
        'n_pos_pred': int(yhat.sum()), 'n_pos_true': int(y.sum()),
        'best_epoch': rj['best_epoch'], 'epochs_run': rj['epochs_run'],
        'created_at': man['created_at'],
    })

res = pd.DataFrame(rows)
res.to_csv(OUT / 'f1_recalculado.csv', index=False, lineterminator='\n', float_format='%.17g')

print('\n== 1d valores únicos en manifiestos ==')
for k, v in manifest_values.items():
    print(k, f'({len(v)} valores distintos)')
    for x in v:
        print('   ', x)

print('\n== 1e created_at ==')
print('min', min(created), 'max', max(created))

print('\n== 2 F1 ==')
print('max |recalc - json|     =', repr(res['d_json'].max()))
print('max |recalc - manifest| =', repr(res['d_manifest'].max()))
print('max |recalc - summary|  =', repr(res['d_summary'].max()), '(summary.csv redondea a 4 decimales)')
print('p_dep == val_probs bit a bit:', int(res['p_dep_eq_val_probs_bitwise'].sum()), '/', len(res))
print('y_true == val_true:', int(res['y_true_eq_val_true'].sum()), '/', len(res))
print('yhat(p>0.5) == val_preds:', int(res['yhat_eq_val_preds'].sum()), '/', len(res))
print('corridas con algún p_dep distinto con el lector por omisión:',
      int((res['n_pdep_default_reader_differs'] > 0).sum()), '; valores distintos en total:',
      int(res['n_pdep_default_reader_differs'].sum()))
print('|p_dep-0.5|<1e-6: total', int(res['n_near_half_1e-6'].sum()),
      '; mínimo global |p_dep-0.5| =', repr(res['min_abs_pdep_minus_half'].min()))
print('y_true: positivos por corrida', sorted(set(res['n_pos_true'])))

print('\nPROBLEMAS:', len(problems))
sys.exit(1 if problems else 0)
