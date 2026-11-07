"""
geometria.py
────────────
F4.7 — Correspondencia geométrica (docs/plan_fase4.md, F4.7). Local; solo train.

Método (plan, F4.7-1…5):
  1. Características: las de la penúltima capa de F4.1 (Gray-Scott: semilla 42
     principal).
  2. Distancias: euclídea en u (escala [0, 1]); coseno en las características.
  3. train (n = 908). ρ de Spearman entre las matrices de distancia (pares i < j)
     y prueba de Mantel con 9,999 permutaciones, default_rng(2026).
  Familia de Holm: las tres pruebas de Mantel (una por técnica, m = 3).

Mantel: se permutan a la vez filas y columnas de la matriz de u; el estadístico
es el ρ de Spearman de los pares i < j. Como permutar objetos solo reordena los
pares, los rangos de la matriz permutada son los rangos permutados: se calculan
una vez y cada permutación es una correlación de Pearson entre matrices de
rangos centradas (diagonal 0). Es exacto, no una aproximación.

Puntos que el plan no fija (reporte_e1a.md, PENDIENTES). No se resuelven aquí:
la CLI exige --resoluciones con:
  mantel_lado          'mayor' | 'bilateral'
  mantel_p             'k_mas_1_sobre_B_mas_1' | 'k_sobre_B'
  mantel_rng           'por_tecnica' (default_rng(2026) nuevo en cada técnica) |
                       'continuo' (un solo generador, técnicas en el orden fijo)
  mantel_sensibilidad  true | false (Mantel también en los backbones de
                       sensibilidad de Gray-Scott, fuera de la familia de Holm)

Uso:
  python -m src.fase4.geometria --features-dir <dir> --resoluciones <json>
Escribe results/f47_geometria/resultados.json; nunca sobrescribe.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.fase4 import recuperabilidad as rec  # noqa: E402

N_PERM = 9999
RNG_SEED = 2026
ALPHA = 0.05
OUT = REPO / 'results' / 'f47_geometria'
RESOLUCIONES = {'mantel_lado': ('mayor', 'bilateral'),
                'mantel_p': ('k_mas_1_sobre_B_mas_1', 'k_sobre_B'),
                'mantel_rng': ('por_tecnica', 'continuo'),
                'mantel_sensibilidad': (True, False)}


def condensed(D):
    i, j = np.triu_indices(D.shape[0], k=1)
    return D[i, j]


def dist_u(U):
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(U, metric='euclidean'))


def dist_features(X):
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(X, metric='cosine'))


def _rank_matrix(D):
    """Matriz simétrica de rangos centrados de los pares i < j (empates: rango medio); diagonal 0."""
    from scipy.stats import rankdata
    n = D.shape[0]
    i, j = np.triu_indices(n, k=1)
    r = rankdata(D[i, j])
    r = r - r.mean()
    R = np.zeros((n, n))
    R[i, j] = r
    R[j, i] = r
    return R


def spearman_dist(Da, Db):
    """ρ de Spearman entre los pares i < j de dos matrices de distancia."""
    Ra, Rb = _rank_matrix(Da), _rank_matrix(Db)
    return float((Ra * Rb).sum() / np.sqrt((Ra ** 2).sum() * (Rb ** 2).sum()))


def mantel(Da, Db, n_perm, rng, lado, p_formula):
    """Mantel con el ρ de Spearman. Permuta objetos de Da. Devuelve dict con r, p y conteo."""
    Ra, Rb = _rank_matrix(Da), _rank_matrix(Db)
    norm = np.sqrt((Ra ** 2).sum() * (Rb ** 2).sum())
    r_obs = float((Ra * Rb).sum() / norm)
    n = Da.shape[0]
    perm_r = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n)
        perm_r[b] = (Ra[np.ix_(p, p)] * Rb).sum() / norm
    tol = 1e-12
    if lado == 'mayor':
        k = int(np.sum(perm_r >= r_obs - tol))
    elif lado == 'bilateral':
        k = int(np.sum(np.abs(perm_r) >= abs(r_obs) - tol))
    else:
        raise rec.Detener(f'lado desconocido {lado!r}')
    if p_formula == 'k_mas_1_sobre_B_mas_1':
        p_val = (k + 1) / (n_perm + 1)
    elif p_formula == 'k_sobre_B':
        p_val = k / n_perm
    else:
        raise rec.Detener(f'fórmula de p desconocida {p_formula!r}')
    return {'rho_spearman': r_obs, 'p': p_val, 'k': k, 'n_perm': n_perm, 'lado': lado,
            'perm_media': float(perm_r.mean()), 'perm_de': float(perm_r.std(ddof=1))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--manifest', default=str(rec.OUT / 'caracteristicas' / 'caracteristicas_manifest.csv'))
    ap.add_argument('--resoluciones', required=True)
    ap.add_argument('--out', default=str(OUT / 'resultados.json'))
    a = ap.parse_args(argv)
    res = rec.check_resoluciones(json.loads(Path(a.resoluciones).read_text(encoding='utf-8')), RESOLUCIONES)
    rec.refuse_overwrite(a.out)
    from src.protocol import verdict
    index = rec.load_index()
    U_tr = rec.load_u(index, ('train',))['train'][0]
    Du = dist_u(U_tr)
    rng_cont = np.random.default_rng(RNG_SEED)
    out = {'principal': {}, 'sensibilidad': {}}
    for tech in rec.TECHNIQUES:
        backbones = [rec.PRINCIPAL[tech]] + (rec.SENSIBILIDAD[tech] if res['mantel_sensibilidad'] else [])
        for backbone in backbones:
            X, digest = rec.load_features(a.features_dir, a.manifest, tech, backbone, index, ('train',))
            rng = np.random.default_rng(RNG_SEED) if res['mantel_rng'] == 'por_tecnica' else rng_cont
            m = mantel(Du, dist_features(X['train']), N_PERM, rng, res['mantel_lado'], res['mantel_p'])
            m |= {'backbone': backbone, 'npz_sha256': digest, 'n': int(X['train'].shape[0])}
            (out['principal'] if backbone == rec.PRINCIPAL[tech] else out['sensibilidad'])[
                tech if backbone == rec.PRINCIPAL[tech] else f'{tech}/{backbone}'] = m
            print(f"{tech:<11} {backbone:<32} ρ = {m['rho_spearman']:.4f}  p = {m['p']:.4g}", flush=True)
    holm = verdict.holm({t: out['principal'][t]['p'] for t in rec.TECHNIQUES}, ALPHA)
    for t in rec.TECHNIQUES:
        out['principal'][t]['holm'] = holm[t]
    meta = {'task': 'F4.7', 'resoluciones': res, 'n_perm': N_PERM, 'rng_seed': RNG_SEED, 'alpha': ALPHA,
            'distancias': {'u': 'euclidean', 'caracteristicas': 'cosine'}, 'split': 'train',
            'git_commit': rec._git('rev-parse', 'HEAD').stdout.strip(),
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    rec.write_json(a.out, {'meta': meta, 'resultados': out})
    print(f'escrito {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
