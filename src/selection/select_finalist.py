"""
select_finalist.py
──────────────────
Selección del finalista por técnica (F3.3) con la regla de
docs/matriz_experimentos.md §5, leída de configs/matriz_experimentos.json
(bloque «seleccion»):

  1. F1-Macro de validación por corrida (ŷ = 1 si p_dep > 0.5, F1.5 §3.3) a
     partir de las predicciones de la fase primera_pasada, cargadas y validadas
     con load_predictions. Media y DE muestral (ddof = 1) por configuración
     sobre las semillas 42, 7 y 123.
  2. Líder: la configuración de mayor media de la técnica.
  3. Empatadas: media ≥ media del líder − DE del líder. Si varias
     configuraciones comparten la media máxima, se usa la mayor de sus DE.
  4. Finalista: entre las empatadas, la de menos parámetros entrenables y,
     después, menos parámetros totales.
  5. Si persiste el empate, la de mayor media; si aún persiste, error
     (UnresolvedTieError) y la resuelve SV.

Segundo mejor modelo (select_second, fase sensibilidad, protocolo §8): se
excluye al finalista y se aplican de nuevo los pasos 2-5 a las configuraciones
restantes de la técnica.

Los conteos de parámetros se leen del JSON (bloque «parametros»); no se
instancian modelos ni se importa torch.

La selección oficial es la de este script. La columna «seleccion» de
summary_by_config.csv (run_grid.py) es informativa.

Uso (desde la raíz del repo):
    python -m src.selection.select_finalist
    python -m src.selection.select_finalist --pattern "results/predictions/primera_pasada/*.csv" \\
        --out results/f33_seleccion/seleccion.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import paths
from src.predictions import schema
from src.predictions.validate import load_predictions, protocol_tag_exists

MATRIX_PATH = paths.REPO / 'configs' / 'matriz_experimentos.json'
DEFAULT_PATTERN = str(paths.RESULTS_DIR / 'predictions' / 'primera_pasada' / '*.csv')
CONFIG_KEY = ['technique', 'arch', 'strategy']

TABLE_COLUMNS = CONFIG_KEY + ['n_seeds', 'seeds', 'f1_val_mean', 'f1_val_std', 'trainable', 'total',
                              'tolerance', 'leader', 'tied', 'finalist']

# Valores del bloque «seleccion» que este script implementa; si el JSON dice
# otra cosa, se rechaza en vez de aplicar una regla distinta en silencio.
IMPLEMENTED_RULE = {
    'fase': 'primera_pasada', 'split': 'val', 'metric': 'f1_macro', 'aggregate': 'mean',
    'dispersion': {'name': 'std', 'ddof': 1}, 'tie_tolerance': 'leader_std', 'multiple_leaders_std': 'max',
    'complexity_order': ['trainable', 'total'], 'residual_tiebreak': 'max_mean', 'unresolved_tie': 'error',
}


class SelectionError(ValueError):
    """Las predicciones no permiten aplicar la regla (faltan corridas, sobran, etc.)."""


class UnresolvedTieError(SelectionError):
    """Empate que la regla de §5 no resuelve: lo decide SV."""


def load_matrix(path=MATRIX_PATH):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_rule(matrix):
    rule = matrix['seleccion']
    wrong = {k: rule.get(k) for k, v in IMPLEMENTED_RULE.items() if rule.get(k) != v}
    if wrong:
        raise SelectionError(f'el bloque «seleccion» del JSON pide {wrong}; este script implementa {IMPLEMENTED_RULE}')
    if rule['decision_threshold'] != schema.DECISION_THRESHOLD:
        raise SelectionError(f'umbral {rule["decision_threshold"]} distinto del de F1.5 ({schema.DECISION_THRESHOLD})')
    return rule


def param_table(matrix):
    """DataFrame arch, strategy, trainable, total desde el bloque «parametros»."""
    rows = [{'arch': a, 'strategy': s, 'trainable': int(v['trainable']), 'total': int(v['total'])}
            for a, by in matrix['parametros'].items() if not a.startswith('_') for s, v in by.items()]
    return pd.DataFrame(rows, columns=['arch', 'strategy', 'trainable', 'total'])


def f1_by_run(preds, matrix):
    """F1-Macro de validación por corrida (technique, arch, strategy, seed)."""
    rule = check_rule(matrix)
    fases = set(preds['fase'])
    if fases != {rule['fase']}:
        raise SelectionError(f'se esperan solo predicciones de la fase {rule["fase"]!r}; hay {sorted(fases)}')
    val = preds[preds['split'] == rule['split']]
    if val.empty:
        raise SelectionError(f'no hay filas de {rule["split"]!r}')
    out = []
    for key, g in val.groupby(CONFIG_KEY + ['seed'], sort=True):
        y_hat = (g['p_dep'] > rule['decision_threshold']).astype(int)
        f1 = f1_score(g['y_true'], y_hat, labels=[0, 1], average='macro', zero_division=0)
        out.append(dict(zip(CONFIG_KEY + ['seed'], key), f1_val=float(f1), n_val=len(g)))
    return pd.DataFrame(out)


def config_table(runs, matrix):
    """Media y DE por configuración, con los conteos de parámetros. Exige que
    cada técnica presente tenga exactamente las configuraciones de la matriz
    (arquitecturas × estrategias) y cada una exactamente las semillas de §5."""
    rule = check_rule(matrix)
    seeds = sorted(rule['seeds'])
    archs, strategies = matrix['factores']['arch'], matrix['factores']['strategy']
    expected = {(a, s) for a in archs for s in strategies}

    bad_tech = sorted(set(runs['technique']) - set(matrix['factores']['technique']))
    if bad_tech:
        raise SelectionError(f'técnicas fuera de la matriz: {bad_tech}')
    for tech, g in runs.groupby('technique'):
        got = set(zip(g['arch'], g['strategy']))
        if got != expected:
            raise SelectionError(f'{tech}: faltan configuraciones {sorted(expected - got)} '
                                 f'y sobran {sorted(got - expected)}')

    params = param_table(matrix)
    rows = []
    for (tech, arch, strat), g in runs.groupby(CONFIG_KEY, sort=True):
        got = sorted(g['seed'])
        if got != seeds:
            raise SelectionError(f'{tech}/{arch}/{strat}: semillas {got}; se esperan {seeds}')
        p = params[(params['arch'] == arch) & (params['strategy'] == strat)]
        if len(p) != 1:
            raise SelectionError(f'{arch}/{strat}: sin conteo de parámetros en el JSON')
        f1 = g['f1_val'].to_numpy(dtype=float)
        rows.append({'technique': tech, 'arch': arch, 'strategy': strat, 'n_seeds': len(f1),
                     'seeds': ' '.join(str(s) for s in got), 'f1_val_mean': float(f1.mean()),
                     'f1_val_std': float(f1.std(ddof=rule['dispersion']['ddof'])),
                     'trainable': int(p['trainable'].iloc[0]), 'total': int(p['total'].iloc[0])})
    return pd.DataFrame(rows)


def select(table):
    """Aplica §5 pasos 2-5 por técnica. Devuelve la tabla con tolerance,
    leader, tied y finalist; lanza UnresolvedTieError si la regla no decide."""
    out = []
    for tech, g in table.groupby('technique', sort=False):
        g = g.copy()
        best = g['f1_val_mean'].max()
        leaders = g['f1_val_mean'] == best
        tol = g.loc[leaders, 'f1_val_std'].max()          # varios líderes: la mayor DE
        tied = g['f1_val_mean'] >= best - tol
        cands = g[tied].sort_values(['trainable', 'total', 'f1_val_mean'], ascending=[True, True, False],
                                    kind='stable')
        top = cands.iloc[0]
        rivals = cands[(cands['trainable'] == top['trainable']) & (cands['total'] == top['total'])
                       & (cands['f1_val_mean'] == top['f1_val_mean'])]
        if len(rivals) > 1:
            names = [f'{r.arch}/{r.strategy}' for r in rivals.itertuples()]
            raise UnresolvedTieError(f'{tech}: empate no resuelto por §5 entre {names} (mismos parámetros '
                                     f'entrenables y totales y misma media {top["f1_val_mean"]:.6f}); lo decide SV')
        g['tolerance'] = tol
        g['leader'] = leaders
        g['tied'] = tied
        g['finalist'] = g.index == top.name
        out.append(g)
    res = pd.concat(out)
    order = {t: i for i, t in enumerate(paths.TECHNIQUES)}
    res = res.assign(_o=res['technique'].map(order)).sort_values(
        ['_o', 'f1_val_mean', 'trainable'], ascending=[True, False, True], kind='stable').drop(columns='_o')
    return res[TABLE_COLUMNS].reset_index(drop=True)


def select_from_runs(runs, matrix):
    return select(config_table(runs, matrix))


def select_second(table):
    """Segundo mejor modelo por técnica (fase sensibilidad, protocolo §8):
    se excluye al finalista y se aplica de nuevo la regla completa de §5
    (líderes, tolerancia, complejidad) a las configuraciones restantes.

    `table` es la salida de config_table (o de select). Devuelve la tabla de
    select con la columna adicional `second`; lanza UnresolvedTieError si la
    regla no decide entre las restantes."""
    first = select(table)
    fin = set(first.loc[first['finalist'], CONFIG_KEY].itertuples(index=False, name=None))
    keys = list(table[CONFIG_KEY].itertuples(index=False, name=None))
    rest = table[[k not in fin for k in keys]].reset_index(drop=True)
    if rest.empty:
        raise SelectionError('no quedan configuraciones después de excluir al finalista')
    second = select(rest)
    sec = set(second.loc[second['finalist'], CONFIG_KEY].itertuples(index=False, name=None))
    first['second'] = [k in sec for k in first[CONFIG_KEY].itertuples(index=False, name=None)]
    return first


def select_second_from_runs(runs, matrix):
    return select_second(config_table(runs, matrix))


def select_finalists(pattern=DEFAULT_PATTERN, matrix=None, index=None, tag_exists=protocol_tag_exists):
    """Carga las predicciones con load_predictions y aplica §5.
    Devuelve (tabla por configuración, F1 por corrida)."""
    matrix = matrix or load_matrix()
    check_rule(matrix)
    preds = load_predictions(pattern, index=index, tag_exists=tag_exists)
    runs = f1_by_run(preds, matrix)
    return select_from_runs(runs, matrix), runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pattern', default=DEFAULT_PATTERN, help='glob de predicciones de primera_pasada')
    ap.add_argument('--matrix', default=str(MATRIX_PATH))
    ap.add_argument('--index', default=str(paths.SAMPLE_INDEX), help='índice maestro (meta/sample_index.csv)')
    ap.add_argument('--out', default=None, help='CSV de salida con la tabla por configuración')
    args = ap.parse_args()

    try:
        table, runs = select_finalists(args.pattern, matrix=load_matrix(args.matrix), index=args.index)
    except SelectionError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nFinalistas:')
    for r in table[table['finalist']].itertuples():
        print(f'  {r.technique:<11} {r.arch}/{r.strategy}  F1-Val {r.f1_val_mean:.4f} ± {r.f1_val_std:.4f}')
    status = 0
    try:
        table = select_second(table)
        print('\nSegundo mejor (fase sensibilidad, protocolo §8):')
        for r in table[table['second']].itertuples():
            print(f'  {r.technique:<11} {r.arch}/{r.strategy}  F1-Val {r.f1_val_mean:.4f} ± {r.f1_val_std:.4f}')
    except SelectionError as e:
        print(f'ERROR (segundo mejor): {e}', file=sys.stderr)
        status = 1
    if args.out:
        out = Path(args.out)
        paths.assert_outside_data_root(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False, lineterminator='\n')
        runs.to_csv(out.with_name(out.stem + '_corridas.csv'), index=False, lineterminator='\n')
        print(f'\nEscrito {out}')
    return status


if __name__ == '__main__':
    sys.exit(main())
