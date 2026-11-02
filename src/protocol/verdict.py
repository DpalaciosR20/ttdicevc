"""
verdict.py
──────────
Regla de decisión del Objetivo V (docs/protocolo_preregistrado.md §5.2–§5.5),
con parámetros de configs/protocolo.json (bloque «contraste»).

No calcula F1 ni genera réplicas: recibe, por técnica, las réplicas bootstrap
ya calculadas (las mismas B réplicas pareadas de §5.2) de

    θ*_t = (F1_img − 0.5) − r (F1_BETO − 0.5)      no inferioridad (§5.1)
    D*_t = F1_img − F1_piso                         superioridad sobre el piso (§5.3)

y aplica:

  1. p unilateral por bootstrap: p = #{réplica ≤ 0} / B (§5.2); si es 0 se
     marca below_resolution (se reporta p < 1/B).
  2. Holm a α unilateral en cada familia de tres contrastes (§5.4). En el
     rango k (p ascendente) el nivel es α_h = α / (m − k + 1); el rechazo es
     escalonado: se detiene en el primer p > α_h. Si varias técnicas tienen
     el mismo p, todas reciben el menor α_h de los rangos que ocupan; el
     rechazo no se ve afectado.
  3. Intervalo percentil bilateral de θ al nivel 1 − 2α_h, con α_h de la
     familia de no inferioridad (cuantiles α_h y 1 − α_h, numpy «linear»).
  4. Veredicto por técnica, en este orden (§5.5):
        V3  no se rechaza H₀ de superioridad sobre el piso
        V1  se rechaza H₀ de no inferioridad
        V2  límite superior del intervalo de θ < 0
        V4  en otro caso
  5. Veredicto global: G1 si alguna V1; si no, G2 si alguna V2; si no, G3 si
     todas V3; en otro caso G4.

La Fase 5 (cálculo de las réplicas) no está aquí.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / 'configs' / 'protocolo.json'

VERDICTS = ('V1', 'V2', 'V3', 'V4')
GLOBAL_VERDICTS = ('G1', 'G2', 'G3', 'G4')


class VerdictInputError(ValueError):
    """Las réplicas no permiten aplicar la regla (técnicas o B distintos, NaN, etc.)."""


def load_contrast(path=PROTOCOL_PATH):
    """Bloque «contraste» del protocolo; verifica que pida lo que este módulo implementa."""
    c = json.loads(Path(path).read_text(encoding='utf-8'))['contraste']
    m = c['multiplicidad']
    expected = {'method': 'holm', 'sided': 'one', 'ties': 'min_alpha_h',
                'families': ['superioridad_piso', 'no_inferioridad']}
    wrong = {k: m.get(k) for k, v in expected.items() if m.get(k) != v}
    if wrong or m['interval']['alpha_h_from'] != 'no_inferioridad' or m['interval']['quantile_method'] != 'linear' \
            or c['veredictos']['order'] != ['V3', 'V1', 'V2', 'V4'] or c['p_value']['formula'] != '#{rep <= 0} / B':
        raise VerdictInputError(f'configs/protocolo.json pide una regla distinta de la implementada: {wrong or m}')
    return c


def p_value(replicates):
    """p unilateral de H₁: estadístico > 0. Devuelve (p, below_resolution)."""
    reps = np.asarray(replicates, dtype=float)
    p = float(np.mean(reps <= 0))
    return p, p == 0.0


def holm(pvalues, alpha):
    """Holm escalonado sobre {técnica: p}. Devuelve {técnica: {rank, alpha_h, reject}}.

    rank es la posición (1 = menor p); con empates, cada técnica empatada recibe
    el menor de sus rangos y el α_h más estricto (el del menor rango)."""
    if not pvalues:
        raise VerdictInputError('familia vacía')
    m = len(pvalues)
    ps = sorted(pvalues.values())
    out, stopped = {}, False
    reject_by_rank = []
    for k, p in enumerate(ps, start=1):
        if not stopped and p <= alpha / (m - k + 1):
            reject_by_rank.append(True)
        else:
            stopped = True
            reject_by_rank.append(False)
    for tech, p in pvalues.items():
        first = ps.index(p) + 1                      # menor rango que ocupa el valor
        last = len(ps) - ps[::-1].index(p)           # mayor rango que ocupa
        out[tech] = {'rank': first, 'alpha_h': alpha / (m - first + 1),
                     # Con empates el rechazo es el mismo en todos los rangos que ocupan
                     'reject': all(reject_by_rank[first - 1:last])}
    return out


def percentile_interval(replicates, alpha_h):
    """Intervalo percentil bilateral al nivel 1 − 2α_h."""
    lo, hi = np.quantile(np.asarray(replicates, dtype=float), [alpha_h, 1 - alpha_h], method='linear')
    return float(lo), float(hi)


def technique_verdict(beats_floor, ni_reject, theta_upper):
    if not beats_floor:
        return 'V3'
    if ni_reject:
        return 'V1'
    if theta_upper < 0:
        return 'V2'
    return 'V4'


def global_verdict(verdicts):
    vs = list(verdicts)
    if not vs or any(v not in VERDICTS for v in vs):
        raise VerdictInputError(f'veredictos inválidos: {vs}')
    if 'V1' in vs:
        return 'G1'
    if 'V2' in vs:
        return 'G2'
    if all(v == 'V3' for v in vs):
        return 'G3'
    return 'G4'


def _check(theta_reps, d_reps, techniques):
    if set(theta_reps) != set(d_reps):
        raise VerdictInputError(f'técnicas distintas en θ {sorted(theta_reps)} y D {sorted(d_reps)}')
    if techniques is not None and set(theta_reps) != set(techniques):
        raise VerdictInputError(f'se esperan las técnicas {sorted(techniques)}; hay {sorted(theta_reps)}')
    sizes = set()
    arrays = {}
    for name, reps in (('θ', theta_reps), ('D', d_reps)):
        for tech, r in reps.items():
            a = np.asarray(r, dtype=float)
            if a.ndim != 1 or a.size == 0:
                raise VerdictInputError(f'{name}[{tech}]: se espera un vector de réplicas')
            if not np.isfinite(a).all():
                raise VerdictInputError(f'{name}[{tech}]: réplicas no finitas')
            sizes.add(a.size)
            arrays[(name, tech)] = a
    if len(sizes) != 1:
        raise VerdictInputError(f'número de réplicas distinto entre técnicas o estadísticos: {sorted(sizes)}')
    return arrays, sizes.pop()


def verdicts(theta_reps, d_reps, alpha=None, B=None, techniques=None, contrast=None):
    """Aplica §5.2–§5.5.

    theta_reps, d_reps: {técnica: réplicas (B,)}, sobre las mismas réplicas.
    alpha, B: por defecto los de configs/protocolo.json; si se da B, se exige.
    techniques: si se da, el conjunto exacto de técnicas esperado.

    Devuelve {'techniques': {técnica: {...}}, 'global': 'G?', 'alpha': α, 'B': B}."""
    c = contrast or load_contrast()
    alpha = c['multiplicidad']['alpha'] if alpha is None else alpha
    arrays, n = _check(theta_reps, d_reps, techniques)
    if B is not None and n != B:
        raise VerdictInputError(f'se esperan B = {B} réplicas; hay {n}')

    techs = list(theta_reps)
    p_sup = {t: p_value(arrays[('D', t)]) for t in techs}
    p_ni = {t: p_value(arrays[('θ', t)]) for t in techs}
    h_sup = holm({t: p for t, (p, _) in p_sup.items()}, alpha)
    h_ni = holm({t: p for t, (p, _) in p_ni.items()}, alpha)

    out = {}
    for t in techs:
        ci = percentile_interval(arrays[('θ', t)], h_ni[t]['alpha_h'])
        v = technique_verdict(h_sup[t]['reject'], h_ni[t]['reject'], ci[1])
        out[t] = {
            'superioridad_piso': {'p': p_sup[t][0], 'below_resolution': p_sup[t][1], **h_sup[t]},
            'no_inferioridad': {'p': p_ni[t][0], 'below_resolution': p_ni[t][1], **h_ni[t]},
            'theta_interval': ci,
            'interval_level': 1 - 2 * h_ni[t]['alpha_h'],
            'verdict': v,
        }
    return {'techniques': out, 'global': global_verdict(r['verdict'] for r in out.values()),
            'alpha': alpha, 'B': n}
