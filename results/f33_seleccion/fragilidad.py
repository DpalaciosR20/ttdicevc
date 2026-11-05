"""
fragilidad.py — F3.3 etapa A, puntos 4b y 5 (descriptivo; no cambia la selección).

4b) Granularidad del F1-Macro (n = 114 de val). Para cada corrida del finalista
    y del líder por media de cada técnica:
      - ΔF1 al voltear la predicción de una sola muestra (mín. y máx. sobre las 114);
      - número mínimo de volteos que cambiaría el finalista de la regla §5,
        (i) concentrados en una sola corrida y (ii) repartidos entre las
        corridas de esas dos configuraciones. Las demás configuraciones de la
        técnica quedan fijas. Búsqueda exacta: un volteo solo depende de su tipo
        (VP→FN, FN→VP, VN→FP, FP→VN), así que se enumeran los conteos por tipo.
5)  Tope de épocas para finalista y segundo: best_epoch, épocas ejecutadas,
    causa de parada y F1-val de las últimas 4 épocas (runs/*.json).

Uso (desde la raíz del repo):
    python results/f33_seleccion/fragilidad.py
"""

import itertools
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regla_independiente import (OUT, REPO, SEEDS, TECHS, config_stats, counts, f1_from_counts,  # noqa: E402
                                 finalist_and_second, load_runs, name, params, pick, rule_tied)

F32 = REPO / 'results' / 'f32_primera_pasada'
# tipo de volteo -> cambio en (tp, fp, fn, tn)
FLIPS = {'VP→FN': (-1, 0, 1, 0), 'FN→VP': (1, 0, -1, 0), 'VN→FP': (0, 1, 0, -1), 'FP→VN': (0, -1, 0, 1)}
AVAIL = {'VP→FN': 0, 'FN→VP': 2, 'VN→FP': 3, 'FP→VN': 1}   # índice del conteo de origen en (tp, fp, fn, tn)
K_SPREAD = 6     # tope de la búsqueda exacta repartida
K_SINGLE = 40    # tope de la búsqueda exacta en una sola corrida


def apply(c, combo):
    tp, fp, fn, tn = c
    for t, n in combo.items():
        d = FLIPS[t]
        tp, fp, fn, tn = tp + n * d[0], fp + n * d[1], fn + n * d[2], tn + n * d[3]
    return tp, fp, fn, tn


def feasible(c, combo):
    return all(n <= c[AVAIL[t]] for t, n in combo.items())


def main():
    runs = load_runs()
    par = params()
    base_counts = {(cfg, s): counts(*runs[cfg][s]) for cfg in runs for s in SEEDS}
    base_f1 = {k: f1_from_counts(*v) for k, v in base_counts.items()}

    def finalist_with(f1_over):
        f1s = {cfg: [f1_over.get((cfg, s), base_f1[(cfg, s)]) for s in SEEDS] for cfg in runs}
        return f1s

    single_rows, min_rows, ep_rows = [], [], []
    for tech in TECHS:
        cfgs = [c for c in runs if c[0] == tech]
        rows = config_stats({c: [base_f1[(c, s)] for s in SEEDS] for c in cfgs}, par)
        fin, sec, tied, leaders = finalist_and_second(rows)
        lead = leaders[0]
        key = lambda r: (r['technique'], r['arch'], r['strategy'])  # noqa: E731
        targets = [key(fin)] + ([key(lead)] if key(lead) != key(fin) else [])
        roles = {key(fin): 'finalista', key(lead): 'líder por media'}
        if len(targets) == 1:
            roles[key(fin)] = 'finalista y líder por media'
        orig = key(fin)

        def decision(f1_over):
            f1s = {c: [f1_over.get((c, s), base_f1[(c, s)]) for s in SEEDS] for c in cfgs}
            return key(pick(config_stats(f1s, par), rule_tied)[0])

        assert decision({}) == orig
        target_runs = [(c, s) for c in targets for s in SEEDS]

        # 4b-1: un solo volteo, por corrida
        for (c, s) in target_runs:
            y, yhat = runs[c][s]
            deltas = []
            for i in range(len(y)):
                yh = yhat.copy()
                yh[i] = 1 - yh[i]
                deltas.append(f1_from_counts(*counts(y, yh)) - base_f1[(c, s)])
            ad = [abs(d) for d in deltas]
            single_rows.append({'tecnica': tech, 'configuracion': f'{c[1]}/{c[2]}', 'rol': roles[c], 'semilla': s,
                                'f1': base_f1[(c, s)], 'tp_fp_fn_tn': ' '.join(map(str, base_counts[(c, s)])),
                                'delta_min': min(deltas), 'delta_max': max(deltas),
                                'abs_delta_min': min(ad), 'abs_delta_max': max(ad)})

        # 4b-2 (i): volteos concentrados en una sola corrida
        best_single = None
        for (c, s) in target_runs:
            bc = base_counts[(c, s)]
            found = None
            for k in range(1, K_SINGLE + 1):
                for parts in itertools.product(range(k + 1), repeat=3):
                    if sum(parts) > k:
                        continue
                    combo = dict(zip(FLIPS, (*parts, k - sum(parts))))
                    if not feasible(bc, combo):
                        continue
                    new = decision({(c, s): f1_from_counts(*apply(bc, combo))})
                    if new != orig:
                        found = (k, combo, new)
                        break
                if found:
                    break
            min_rows.append({'tecnica': tech, 'busqueda': 'una corrida', 'corrida': f'{c[1]}/{c[2]} s{s}',
                             'min_volteos': found[0] if found else f'>{K_SINGLE}',
                             'ejemplo': ({t: n for t, n in found[1].items() if n} if found else ''),
                             'nuevo_finalista': f'{found[2][1]}/{found[2][2]}' if found else ''})
            if found and (best_single is None or found[0] < best_single):
                best_single = found[0]

        # 4b-2 (ii): volteos repartidos entre las corridas objetivo (exacta hasta K_SPREAD)
        items = [(r, t) for r in target_runs for t in FLIPS]
        found = None
        for k in range(1, K_SPREAD + 1):
            for multi in itertools.combinations_with_replacement(range(len(items)), k):
                per_run = {}
                for j in multi:
                    r, t = items[j]
                    per_run.setdefault(r, {}).setdefault(t, 0)
                    per_run[r][t] += 1
                if not all(feasible(base_counts[r], cb) for r, cb in per_run.items()):
                    continue
                over = {r: f1_from_counts(*apply(base_counts[r], cb)) for r, cb in per_run.items()}
                new = decision(over)
                if new != orig:
                    found = (k, per_run, new)
                    break
            if found:
                break
        min_rows.append({'tecnica': tech, 'busqueda': 'repartida (cualquier corrida de finalista/líder)',
                         'corrida': ', '.join(f'{c[1]}/{c[2]} s{s}' for (c, s) in target_runs),
                         'min_volteos': found[0] if found else f'>{K_SPREAD}',
                         'ejemplo': ({f'{r[0][1]}/{r[0][2]} s{r[1]}': {t: n for t, n in cb.items()}
                                      for r, cb in found[1].items()} if found else ''),
                         'nuevo_finalista': f'{found[2][1]}/{found[2][2]}' if found else ''})

        # 5: épocas del finalista y del segundo
        for r, rol in [(fin, 'finalista'), (sec, 'segundo')]:
            for s in SEEDS:
                arch_file = r['arch']
                p = F32 / 'runs' / f'{tech}__{arch_file}__{r["strategy"]}__' \
                                   f'{json.loads((REPO / "configs" / "preprocesamiento.json").read_text())[tech]["variant"]}__seed{s}.json'
                rj = json.loads(p.read_text(encoding='utf-8'))
                h = rj['history']['f1_val']
                assert len(h) == rj['epochs_run']
                assert h[rj['best_epoch'] - 1] == rj['best_f1_val'] == max(h)
                by_pat = rj['epochs_run'] - rj['best_epoch'] >= rj['patience']
                causa = 'paciencia' if by_pat else ('tope' if rj['epochs_run'] == rj['n_epochs_max'] else '¿?')
                if by_pat and rj['epochs_run'] == rj['n_epochs_max']:
                    causa = 'paciencia (coincide con el tope)'
                ep_rows.append({'tecnica': tech, 'rol': rol, 'configuracion': name(r), 'semilla': s,
                                'best_epoch': rj['best_epoch'], 'epocas_ejecutadas': rj['epochs_run'],
                                'tope': rj['n_epochs_max'], 'paciencia': rj['patience'], 'causa': causa,
                                'f1_val_ultimas_4': ' '.join(repr(x) for x in h[-4:]),
                                'best_f1_val': rj['best_f1_val']})

    for fname, rows in [('fragilidad_volteo_unico.csv', single_rows), ('fragilidad_volteos.csv', min_rows),
                        ('epocas_finalista_segundo.csv', ep_rows)]:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / fname, index=False, lineterminator='\n', float_format='%.17g')
        print(f'\n== {fname} ==')
        with pd.option_context('display.width', 300, 'display.max_columns', None, 'display.max_colwidth', 200):
            print(df.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
