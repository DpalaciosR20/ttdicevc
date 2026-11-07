"""
verificar_45.py — Bloque A (SV), punto 1: integridad de los 45 CSV de CE.

Fases: final (F3.5), control_permutacion (F3.6), sensibilidad (F5.5, entrenamiento).
Barrera: de results/predictions/referencia/ solo se lee sample_id, split e
y_true del CSV de BETO (nunca p_dep). No importa src.protocol.verdict.

Uso (desde la raíz):  python results/f5_preparacion/verificar_45.py
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import paths  # noqa: E402
from src.predictions.validate import validate_predictions, load_index  # noqa: E402
from src.selection import select_finalist as sf  # noqa: E402

PHASES = {'final': 'f35_final', 'control_permutacion': 'f36_permutacion', 'sensibilidad': 'f55_sensibilidad'}
SEEDS = {42, 7, 123, 2024, 99}
TECHS = ['gray_scott', 'perlin', 'lorenz']
TOL = 1e-12
fails = []


def check(cond, msg):
    print(('  OK    ' if cond else '  FALLA ') + msg)
    if not cond:
        fails.append(msg)


def git(*a):
    r = subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


index = load_index(paths.SAMPLE_INDEX)
ix = pd.read_csv(paths.SAMPLE_INDEX)
ix_val = ix[ix.split == 'val'].sort_values('sample_id')
ix_test = ix[ix.split == 'test'].sort_values('sample_id')

# BETO: solo sample_id / split / y_true (sin p_dep)
beto = pd.read_csv(REPO / 'results/predictions/referencia/text__beto__finetune__seed42.csv',
                   usecols=['sample_id', 'split', 'y_true'])
beto_test = beto[beto.split == 'test'].sort_values('sample_id')
print('Barrera: CSV de BETO leído con usecols=[sample_id, split, y_true]; p_dep no se carga.')

# ── c) selección oficial
table, _ = sf.select_finalists()
table2 = sf.select_second(table)
fin = {r.technique: (r.arch, r.strategy) for r in table[table['finalist']].itertuples()}
sec = {r.technique: (r.arch, r.strategy) for r in table2[table2['second']].itertuples()}
print('\nselect_finalist:', fin)
print('select_second:  ', sec)

matrix = json.loads((REPO / 'configs/matriz_experimentos.json').read_text(encoding='utf-8'))
cc = matrix['configuracion_comun']
perm_expected = np.random.default_rng(99).permutation(908).tolist()

ref_ytrue = None
rows = []
for fase, rdir in PHASES.items():
    print(f'\n== Fase {fase}')
    csvs = sorted((REPO / 'results/predictions' / fase).glob('*.csv'))
    check(len(csvs) == 15, f'{fase}: {len(csvs)} CSV (se esperan 15)')
    seen = set()
    for c in csvs:
        df = validate_predictions(c, index=index)  # a) lanza PredictionFormatError si es inválido
        tech, arch, strat = df.technique.iloc[0], df.arch.iloc[0], df.strategy.iloc[0]
        seed = int(df.seed.iloc[0])
        seen.add((tech, seed))
        man = json.loads(c.with_name(c.stem + '.manifest.json').read_text(encoding='utf-8'))
        run_json = sorted(glob.glob(str(REPO / 'results' / rdir / tech / 'runs' /
                                        f'{tech}__{arch}__{strat}__*__seed{seed}.json')))
        check(len(run_json) == 1, f'{c.name}: un JSON de corrida ({len(run_json)})')
        run = json.loads(Path(run_json[0]).read_text(encoding='utf-8'))
        val = df[df.split == 'val'].sort_values('sample_id')
        tst = df[df.split == 'test'].sort_values('sample_id')
        # b) ids e y_true
        yt = (val.y_true.tolist(), tst.y_true.tolist())
        if ref_ytrue is None:
            ref_ytrue = yt
        okb = (val.sample_id.tolist() == ix_val.sample_id.tolist()
               and tst.sample_id.tolist() == ix_test.sample_id.tolist()
               and yt[0] == ix_val.y_true.tolist() and yt[1] == ix_test.y_true.tolist()
               and tst.sample_id.tolist() == beto_test.sample_id.tolist()
               and yt[1] == beto_test.y_true.tolist() and yt == ref_ytrue)
        # c) configuración
        exp = (sec if fase == 'sensibilidad' else fin)[tech]
        okc = ((arch, strat) == exp and run['fase'] == fase and run['technique'] == tech
               and man['fase'] == fase)
        # d) hiperparámetros
        if fase == 'sensibilidad':
            hp = {'lr': cc['lr'][strat], 'weight_decay': cc['weight_decay'], 'batch_size': cc['batch_size']}
        else:
            hp = json.loads((REPO / 'results/f34_hpo' / tech / 'hpo_result.json')
                            .read_text(encoding='utf-8'))['decision']['params']
        okd = (all(run[k] == hp[k] and man['config'][k] == hp[k] for k in ('lr', 'weight_decay', 'batch_size'))
               and run['n_epochs_max'] == cc['max_epochs'] and run['patience'] == cc['early_stopping']['patience'])
        # e) permutación
        if fase == 'control_permutacion':
            oke = (run['permute_labels'] == 99 and run['permutation_indices'] == perm_expected
                   and man['config']['permute_labels'] == 99 and man['config']['permutation_indices'] == perm_expected)
        else:
            oke = run['permute_labels'] is None and run['permutation_indices'] is None
        # f) F1 recalculado (p_dep > 0.5) frente al JSON
        f1v = f1_score(val.y_true, (val.p_dep > 0.5).astype(int), average='macro')
        f1t = f1_score(tst.y_true, (tst.p_dep > 0.5).astype(int), average='macro')
        csv_eq_json = (np.array_equal(val.p_dep.to_numpy(), np.array(run['val_probs']))
                       and np.array_equal(tst.p_dep.to_numpy(), np.array(run['test_probs']))
                       and list(run['val_true']) == yt[0] and list(run['test_true']) == yt[1])
        preds_rule = (list(run['val_preds']) == (np.array(run['val_probs']) > 0.5).astype(int).tolist()
                      and list(run['test_preds']) == (np.array(run['test_probs']) > 0.5).astype(int).tolist())
        n_half = int((df.p_dep == 0.5).sum())
        okf = abs(f1v - run['best_f1_val']) <= TOL and abs(f1t - run['f1_test']) <= TOL and csv_eq_json and preds_rule
        print(f'{c.name}: b={okb} c={okc} d={okd} e={oke} f={okf}  F1v={f1v:.6f} F1t={f1t:.6f} '
              f'(json {run["best_f1_val"]:.6f}/{run["f1_test"]:.6f}; |dv|={abs(f1v - run["best_f1_val"]):.1e} '
              f'|dt|={abs(f1t - run["f1_test"]):.1e}) p_dep==0.5: {n_half}')
        for k, v in (('b', okb), ('c', okc), ('d', okd), ('e', oke), ('f', okf)):
            if not v:
                fails.append(f'{c.name}: {k}')
        rows.append({'fase': fase, 'technique': tech, 'arch': arch, 'strategy': strat, 'seed': seed,
                     'f1_val': f1v, 'f1_test': f1t, 'git_commit': run['git_commit'], 'created_at': run['created_at'],
                     'manifest_commit': man['git_commit'], 'task_id': man['task_id'],
                     'protocol_commit': run['protocol']['commit'],
                     'selection_role': run['selection'].get('role'),
                     'selection_matches': run['selection'].get('matches'),
                     'override': run['selection'].get('override')})
    check(all({s for t, s in seen if t == tt} == SEEDS for tt in TECHS),
          f'{fase}: semillas {sorted(SEEDS)} en las 3 técnicas')

R = pd.DataFrame(rows)
print('\n== f) Resúmenes por técnica y fase (ddof=0 frente a ddof=1)')
for (fase, tech), g in R.groupby(['fase', 'technique'], sort=False):
    sbc = pd.read_csv(REPO / 'results' / PHASES[fase] / tech / 'summary_by_config.csv').iloc[0]
    v, t = g.f1_val.to_numpy(), g.f1_test.to_numpy()
    print(f'  {fase:<20} {tech:<11} val {v.mean():.4f} sd1 {v.std(ddof=1):.4f} sd0 {v.std(ddof=0):.4f} | '
          f'test {t.mean():.4f} sd1 {t.std(ddof=1):.4f} sd0 {t.std(ddof=0):.4f} | '
          f'summary_by_config: val {sbc.f1_val_mean} ± {sbc.f1_val_std}, test {sbc.f1_test_mean} ± {sbc.f1_test_std}')
    ok = (abs(round(v.mean(), 4) - sbc.f1_val_mean) < 1e-9 and abs(round(v.std(ddof=1), 4) - sbc.f1_val_std) < 1e-9
          and abs(round(t.mean(), 4) - sbc.f1_test_mean) < 1e-9 and abs(round(t.std(ddof=1), 4) - sbc.f1_test_std) < 1e-9)
    check(ok, f'{fase}/{tech}: summary_by_config.csv = media y DE ddof=1 (4 decimales)')

print('\n== g) Ancestría')
for fase in PHASES:
    sub = R[R.fase == fase]
    for commit in sorted(set(sub.git_commit) | set(sub.manifest_commit)):
        for tag in ('protocolo-v1', 'seleccion-f33', '9027167'):
            rc, _ = git('merge-base', '--is-ancestor', tag, commit)
            check(rc == 0, f'{fase}: {tag} es ancestro de {commit[:7]} (rc={rc})')
    print(f'  {fase}: task_id={sorted(set(sub.task_id))}; protocol.commit={sorted(set(sub.protocol_commit))}; '
          f'selection.role={sorted(set(sub.selection_role))}; matches={sorted(set(map(str, sub.selection_matches)))}; '
          f'override={sorted(set(map(str, sub.override)))}')
    print(f'  {fase}: created_at [{sub.created_at.min()}, {sub.created_at.max()}]')
_, e3 = git('log', '-1', '--format=%H %cI', '9027167')
f35_min = R[R.fase == 'final'].created_at.min()
e3_utc = pd.Timestamp(e3.split()[1]).tz_convert('UTC')
print(f'  Enmienda 3: {e3} (= {e3_utc} UTC); created_at mínimo de F3.5 = {f35_min}')
check(e3_utc < pd.Timestamp(f35_min), 'Enmienda 3 anterior a la primera corrida de F3.5')

R.to_csv(REPO / 'results/f5_preparacion/f1_recalculado_45.csv', index=False)
print(f'\nFALLAS: {len(fails)}')
for f in fails:
    print('  -', f)
sys.exit(1 if fails else 0)
