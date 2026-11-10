"""Descriptivo (E1b): signo de ΔF1 en las réplicas de A1, recalculado desde
predicciones_prueba.csv con recuperabilidad.delta_f1 (sin modelos nuevos ni
nueva consulta de prueba). Comprueba que el IC coincide con resultados_prueba.json.
Uso: python results/f41_recuperabilidad/e1b/replicas_delta_f1_signo.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.fase4 import recuperabilidad as rec  # noqa: E402
from src.protocol import bootstrap as bs  # noqa: E402
from src.protocol import insumos as ins_mod  # noqa: E402
D = REPO / 'results' / 'f41_recuperabilidad'
cfg = ins_mod.load_config()
idx = bs.indices(cfg)
T = json.loads((D / 'prueba' / 'resultados_prueba.json').read_text(encoding='utf-8'))['resultados']
with open(D / 'prueba' / 'predicciones_prueba.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
out = {}
for t, bb in T['tecnicas'].items():
    for b, x in bb.items():
        if not isinstance(x, dict):
            continue
        rr = [r for r in rows if r['tecnica'] == t and r['backbone'] == b]
        y = np.array([int(r['y_true']) for r in rr])
        pu = np.array([float(r['p_dep_u']) for r in rr])
        ph = np.array([float(r['p_dep_u_hat']) for r in rr])
        d, reps = rec.delta_f1(y, pu, ph, idx, cfg['enmienda_4']['ic']['descriptivo'])
        assert d['ic'] == x['ic'] and d['delta_f1_obs'] == x['delta_f1_obs'], (t, b)
        out[f'{t}/{b}'] = {'reps_lt0': int((reps < 0).sum()), 'reps_eq0': int((reps == 0).sum()),
                           'reps_gt0': int((reps > 0).sum()), 'n_pred_distintas': int(((pu > 0.5) != (ph > 0.5)).sum())}
(D / 'e1b' / 'replicas_delta_f1_signo.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps(out))
