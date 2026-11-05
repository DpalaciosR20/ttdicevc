"""
regla_independiente.py — F3.3 etapa A, puntos 3 y 4a.

Reimplementación de la regla de docs/matriz_experimentos.md §5 sin importar
src.selection (solo pandas, numpy y el JSON de la matriz para los parámetros):

  1. F1-Macro de val por corrida desde p_dep (> 0.5), media y DE ddof=1 por configuración.
  2. Líder = mayor media (si hay varios, tolerancia = la mayor de sus DE).
  3. Empatadas: media >= media_líder − tolerancia.
  4. Finalista: menos entrenables, luego menos totales, luego mayor media; si persiste, error.
  Segundo: se excluye al finalista y se repiten 2–4.

Compara con results/f33_seleccion/seleccion.csv (salida de select_finalist) y
escribe tabla_seleccion.csv y fragilidad_tolerancias.csv (4a, descriptivo).

Uso (desde la raíz del repo):
    python results/f33_seleccion/regla_independiente.py
"""

import glob
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PRED = REPO / 'results' / 'predictions' / 'primera_pasada'
OUT = REPO / 'results' / 'f33_seleccion'
TECHS = ['gray_scott', 'perlin', 'lorenz']
SEEDS = [42, 7, 123]


def f1_from_counts(tp, fp, fn, tn):
    def f(a, b, c):  # 2a / (2a + b + c), zero_division = 0
        d = 2 * a + b + c
        return 0.0 if d == 0 else 2 * a / d
    return (f(tp, fp, fn) + f(tn, fn, fp)) / 2


def counts(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return (int(((y == 1) & (yhat == 1)).sum()), int(((y == 0) & (yhat == 1)).sum()),
            int(((y == 1) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 0)).sum()))


def load_runs():
    """dict (tech, arch, strat) -> {seed: (y, yhat)}; solo filas de val."""
    runs = {}
    for c in sorted(glob.glob(str(PRED / '*.csv'))):
        tech, arch, strat, sd = Path(c).stem.split('__')
        df = pd.read_csv(c, float_precision='round_trip')
        assert set(df['split']) == {'val'}
        y = df['y_true'].to_numpy(int)
        yhat = (df['p_dep'].to_numpy(float) > 0.5).astype(int)
        runs.setdefault((tech, arch, strat), {})[int(sd[4:])] = (y, yhat)
    return runs


def params():
    m = json.loads((REPO / 'configs' / 'matriz_experimentos.json').read_text(encoding='utf-8'))['parametros']
    return {(a, s): (v['trainable'], v['total']) for a, by in m.items() if not a.startswith('_') for s, v in by.items()}


def config_stats(f1s_by_cfg, par):
    """f1s_by_cfg: dict cfg -> list de 3 F1 (orden SEEDS)."""
    rows = []
    for (tech, arch, strat), f1s in f1s_by_cfg.items():
        assert len(f1s) == 3
        tr, tot = par[(arch, strat)]
        rows.append({'technique': tech, 'arch': arch, 'strategy': strat,
                     'mean': statistics.fmean(f1s), 'sd': statistics.stdev(f1s),
                     'sd0': statistics.pstdev(f1s), 'trainable': tr, 'total': tot})
    return rows


def pick(rows, tied_fn):
    """rows de una técnica; tied_fn(row, leader_rows) -> bool. Devuelve (finalista, tied, líderes)."""
    best = max(r['mean'] for r in rows)
    leaders = [r for r in rows if r['mean'] == best]
    tied = [r for r in rows if tied_fn(r, leaders)]
    tied_sorted = sorted(tied, key=lambda r: (r['trainable'], r['total'], -r['mean']))
    top = tied_sorted[0]
    rivals = [r for r in tied_sorted if (r['trainable'], r['total'], r['mean']) == (top['trainable'], top['total'], top['mean'])]
    if len(rivals) > 1:
        raise RuntimeError(f'empate no resuelto: {[(r["arch"], r["strategy"]) for r in rivals]}')
    return top, tied, leaders


def rule_tied(r, leaders):  # regla §5
    best = leaders[0]['mean']
    tol = max(l['sd'] for l in leaders)
    return r['mean'] >= best - tol


def name(r):
    return f'{r["arch"]}/{r["strategy"]}'


def finalist_and_second(rows, tied_fn=rule_tied):
    fin, tied, leaders = pick(rows, tied_fn)
    rest = [r for r in rows if r is not fin]
    sec, _, _ = pick(rest, tied_fn)
    return fin, sec, tied, leaders


def main():
    runs = load_runs()
    par = params()
    f1s = {cfg: [f1_from_counts(*counts(*runs[cfg][s])) for s in SEEDS] for cfg in runs}
    rows = config_stats(f1s, par)

    # --- 3: tabla y comparación con select_finalist ---
    off = pd.read_csv(OUT / 'seleccion.csv')
    table, ok = [], True
    for tech in TECHS:
        tr = [r for r in rows if r['technique'] == tech]
        fin, sec, tied, leaders = finalist_and_second(tr)
        best = leaders[0]['mean']
        tol = max(l['sd'] for l in leaders)
        limit = best - tol
        o = off[off['technique'] == tech]
        o_fin = o[o['finalist']].iloc[0]
        o_sec = o[o['second']].iloc[0]
        same = (o_fin['arch'], o_fin['strategy']) == (fin['arch'], fin['strategy']) and \
               (o_sec['arch'], o_sec['strategy']) == (sec['arch'], sec['strategy'])
        ok &= same
        print(f'{tech:<11} finalista {name(fin):<26} segundo {name(sec):<26} '
              f'select_finalist: {o_fin["arch"]}/{o_fin["strategy"]}, {o_sec["arch"]}/{o_sec["strategy"]}  '
              f'{"COINCIDE" if same else "DIFIERE"}')
        for r in sorted(tr, key=lambda r: -r['mean']):
            om = o[(o['arch'] == r['arch']) & (o['strategy'] == r['strategy'])].iloc[0]
            table.append({
                'tecnica': tech, 'configuracion': name(r), 'media': r['mean'], 'de': r['sd'],
                'limite_empate': limit, 'holgura': r['mean'] - limit, 'en_empate': r in tied,
                'entrenables': r['trainable'], 'totales': r['total'],
                'rol': 'finalista' if r is fin else ('segundo' if r is sec else '—'),
                'lider': r in leaders,
                'f1_semillas_42_7_123': ' '.join(repr(x) for x in f1s[(tech, r['arch'], r['strategy'])]),
                'd_media_vs_select_finalist': abs(r['mean'] - om['f1_val_mean']),
                'd_de_vs_select_finalist': abs(r['sd'] - om['f1_val_std']),
            })
    tab = pd.DataFrame(table)
    tab.to_csv(OUT / 'tabla_seleccion.csv', index=False, lineterminator='\n', float_format='%.17g')
    print('max |media − select_finalist| =', repr(tab['d_media_vs_select_finalist'].max()),
          '; max |DE − select_finalist| =', repr(tab['d_de_vs_select_finalist'].max()))
    print('Reimplementación coincide con select_finalist/select_second:', ok)

    # --- 4a: variantes de la zona de empate (descriptivo) ---
    variants = {
        'regla_§5 (DE líder, ddof=1)': rule_tied,
        'DE líder ddof=0': lambda r, L: r['mean'] >= L[0]['mean'] - max(l['sd0'] for l in L),
        'EE líder = DE/√3': lambda r, L: r['mean'] >= L[0]['mean'] - max(l['sd'] for l in L) / math.sqrt(3),
        'DE del retador (ddof=1)': lambda r, L: r in L or r['mean'] >= L[0]['mean'] - r['sd'],
        'DE combinada √(DE_l²+DE_r²)': lambda r, L: r in L or r['mean'] >= L[0]['mean'] - math.sqrt(
            max(l['sd'] for l in L) ** 2 + r['sd'] ** 2),
    }
    frag = []
    for tech in TECHS:
        tr = [r for r in rows if r['technique'] == tech]
        for vname, fn in variants.items():
            fin, tied, leaders = pick(tr, fn)
            frag.append({'tecnica': tech, 'variante': vname, 'finalista': name(fin),
                         'empatadas': '; '.join(name(r) for r in sorted(tied, key=lambda r: -r['mean']))})
    frag = pd.DataFrame(frag)
    frag.to_csv(OUT / 'fragilidad_tolerancias.csv', index=False, lineterminator='\n')
    with pd.option_context('display.width', 250, 'display.max_colwidth', 120):
        print(frag.to_string(index=False))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
