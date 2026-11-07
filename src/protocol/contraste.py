"""
contraste.py
────────────
Cálculos de la Fase 5 sobre réplicas pareadas (docs/protocolo_preregistrado.md
§5–§7 y Enmiendas 4 y 5), con la regla de decisión de src/protocol/verdict.py (sin
modificarla). Todo parámetro sale de configs/protocolo.json.

Entrada común: un «F1-tabla» con, para cada clasificador, su F1-Macro observado
(«obs», con todas las muestras una vez) y sus B réplicas («rep»):
    F[nombre] = {'obs': float, 'rep': (B,)}                 referencias
    F[('img', fase, técnica)] = {'obs': (S,), 'rep': (B, S)}  visuales, por semilla
Se construye con f1_table() a partir de las predicciones y de la matriz de
multiplicidades de src/protocol/bootstrap.py.

Funciones → protocolo:
    theta_reps                θ* (§5.1–§5.2); formulación A como δ − Δ* (A13)
    floor_candidates          pisos de §4.4: mayoritaria, estratificado = 0.5
                              constante (§5.3), brillo, permutado = media de 5
                              semillas por réplica (§5.3)
    d_reps                    D* (§5.3) con A2 (identidad del máximo observado,
                              o máximo por réplica como sensibilidad) y A3
                              (empate con la tolerancia de la Enmienda 5; P2)
    veredicto                 verdict.verdicts con B del protocolo; IC de D (A4)
    retencion                 ρ̂ (§5.6; A5, A6)
    descomposicion            §7 (A5)
    frente_a_descriptores     §7 (A11)
    contraste_longitud        §6 (A7)
    cochran_q, mcnemar_*      §7 (A12; Enmienda 5, P5)
    kappa, contencion         F4.6 (A10; docs/plan_fase4.md; Enmienda 5, P3 y P4)
    terciles_longitud,
    spearman_u_longitud       §6 descriptivo (A8)
"""

from decimal import Decimal
from itertools import combinations

import numpy as np
from scipy import stats

from src.protocol import bootstrap as bs
from src.protocol import verdict

TECHNIQUES = ('gray_scott', 'perlin', 'lorenz')


NO_APLICA = 'no_aplica'


def empate_tolerancia(cfg):
    """Tolerancia del empate de A3 (Enmienda 5): |F1_a − F1_b| ≤ tol en el observado."""
    return float(cfg['enmienda_5']['piso']['empate_tolerancia'])


# ── Tabla de F1 ───────────────────────────────────────────────────────────
def f1_table(W, y, preds):
    """preds: {nombre: ŷ (n,) o (n, S)} → {nombre: {'obs', 'rep'}}."""
    W_obs = bs.observed_weights(len(y))
    out = {}
    for name, yhat in preds.items():
        yhat = np.asarray(yhat)
        out[name] = {'obs': bs.f1_macro(W_obs, y, yhat)[0], 'rep': bs.f1_macro(W, y, yhat)}
    return out


def img(F, fase, tech, seed_index=None):
    """F1 visual: media de las semillas (§4.1) o una semilla concreta (§8, por semilla)."""
    e = F[('img', fase, tech)]
    if seed_index is None:
        return float(np.mean(e['obs'])), e['rep'].mean(axis=1)
    return float(e['obs'][seed_index]), e['rep'][:, seed_index]


def ref(F, name, shift=0.0):
    return F[name]['obs'] + shift, F[name]['rep'] + shift


# ── θ y D ─────────────────────────────────────────────────────────────────
def theta_reps(img_x, beto_x, r, f1_azar):
    """θ = (F1_img − azar) − r (F1_BETO − azar), elemento a elemento (§5.1)."""
    return (img_x - f1_azar) - r * (beto_x - f1_azar)


def formulation_a_reps(img_x, beto_x, delta):
    """Formulación A (A13) en la forma que consume verdict.verdicts: δ − Δ con Δ = F1_BETO − F1_img.
    p = #{δ − Δ* ≤ 0}/B = #{Δ* ≥ δ}/B; límite superior de δ − Δ < 0 ⇔ límite inferior de Δ > δ."""
    return delta - (beto_x - img_x)


def floor_candidates(F, tech, cfg, include=None, permutado_fase='control_permutacion', permutado_seed=None):
    """Pisos de la técnica (§4.4) como {nombre: (obs, rep)}.

    include: lista de pisos (por omisión pisos.order); permutado = media de las
    5 semillas de control_permutacion en cada réplica (§5.3), o la semilla
    permutado_seed si se indica."""
    order = include or cfg['pisos']['order']
    B = F['beto']['rep'].shape[0]
    const = cfg['contraste']['superioridad_piso']['stratified_as_constant']
    out = {}
    for name in order:
        if name == 'majority':
            out[name] = ref(F, 'majority')
        elif name == 'stratified':
            out[name] = (float(const), np.full(B, float(const)))
        elif name == 'brightness':
            out[name] = ref(F, ('brightness', tech))
        elif name == 'visual_permutado':
            out[name] = img(F, permutado_fase, tech, permutado_seed)
        else:
            raise ValueError(f'piso desconocido {name!r}')
    return out


def d_reps(img_obs, img_rep, cands, mode='identidad', ties='min', tol=0.0):
    """D = F1_img − F1_piso (§5.3).

    mode='identidad' (A2): el piso es la referencia con el mayor F1 observado y
    esa identidad se remuestrea. Hay empate si |F1_a − F1_max| ≤ tol en el
    observado (A3; tolerancia de la Enmienda 5).
      ties='min' (A3): en cada réplica, el menor D* entre los empatados.
      ties='primero' (Enmienda 5, P2; alternativa de robustez): el primero de
      los empatados según el orden de cands (= pisos.order).
    mode='max_por_replica' (sensibilidad de A2): en cada réplica, el máximo de los pisos.
    Devuelve (D_obs, D_rep, info)."""
    obs = {k: float(v[0]) for k, v in cands.items()}
    best = max(obs.values())
    tied = [k for k, v in obs.items() if abs(v - best) <= tol]
    if mode == 'identidad':
        if ties == 'min':
            D_obs = min(img_obs - obs[k] for k in tied)
            D_rep = np.stack([img_rep - cands[k][1] for k in tied]).min(axis=0)
        elif ties == 'primero':
            D_obs, D_rep = img_obs - obs[tied[0]], img_rep - cands[tied[0]][1]
        else:
            raise ValueError(ties)
    elif mode == 'max_por_replica':
        D_obs = img_obs - best
        D_rep = img_rep - np.stack([v[1] for v in cands.values()]).max(axis=0)
    else:
        raise ValueError(mode)
    return D_obs, D_rep, {'piso': tied, 'piso_f1_obs': obs, 'empate': len(tied) > 1, 'mode': mode, 'ties': ties}


# ── Veredicto (§5.2–§5.5) ─────────────────────────────────────────────────
def veredicto(theta, D, cfg, extras=None):
    """theta, D: {técnica: (obs, rep)}. Aplica verdict.verdicts con B del protocolo y añade
    los observados, el IC de D (A4) y los extras por técnica."""
    B = cfg['contraste']['bootstrap']['B']
    res = verdict.verdicts({t: v[1] for t, v in theta.items()}, {t: v[1] for t, v in D.items()},
                           B=B, techniques=list(theta))
    icD = cfg['enmienda_4']['ic']['D']
    if (icD['tipo'], icD['lado'], icD['alpha_h_from'], icD['cuantiles']) != \
            ('percentil', 'bilateral', 'superioridad_piso', 'linear'):
        raise ValueError(f'enmienda_4.ic.D inesperado {icD}')
    for t, r in res['techniques'].items():
        a = r['superioridad_piso']['alpha_h']
        r['theta_obs'], r['D_obs'] = float(theta[t][0]), float(D[t][0])
        r['D_interval'] = bs.percentile_interval(D[t][1], a, 1 - a)
        r['D_interval_level'] = 1 - 2 * a
        if extras and t in extras:
            r.update(extras[t])
    return res


def contraste(F, cfg, fase='final', r=None, formulation='B', delta=None, beto_shift=0.0,
              floor_mode='identidad', ties='min', floors=None, seed_index=None, permutado_seed=None):
    """Contraste completo por técnica: θ (o la formulación A), D con el piso, veredicto."""
    c = cfg['contraste']
    r = c['r'] if r is None else r
    b_obs, b_rep = ref(F, 'beto', beto_shift)
    theta, D, extras = {}, {}, {}
    for t in TECHNIQUES:
        i_obs, i_rep = img(F, fase, t, seed_index)
        if formulation == 'B':
            theta[t] = (theta_reps(i_obs, b_obs, r, c['f1_azar']), theta_reps(i_rep, b_rep, r, c['f1_azar']))
        elif formulation == 'A':
            theta[t] = (formulation_a_reps(i_obs, b_obs, delta), formulation_a_reps(i_rep, b_rep, delta))
        else:
            raise ValueError(formulation)
        ps = permutado_seed(t) if callable(permutado_seed) else permutado_seed
        cands = floor_candidates(F, t, cfg, include=floors, permutado_seed=ps)
        d_obs, d_rep, info = d_reps(i_obs, i_rep, cands, floor_mode, ties, empate_tolerancia(cfg))
        D[t] = (d_obs, d_rep)
        extras[t] = {'f1_img_obs': i_obs, 'f1_beto_obs': b_obs, **info}
    res = veredicto(theta, D, cfg, extras)
    res['params'] = {'fase': fase, 'r': r, 'formulation': formulation, 'delta': delta, 'beto_shift': beto_shift,
                     'floor_mode': floor_mode, 'ties': ties, 'floors': floors or cfg['pisos']['order'],
                     'seed_index': seed_index}
    return res


# ── Descriptivos sobre las mismas réplicas ────────────────────────────────
def _ci95(cfg_ic):
    if (cfg_ic['tipo'], cfg_ic['lado'], cfg_ic.get('cuantiles', 'linear')) != ('percentil', 'bilateral', 'linear'):
        raise ValueError(f'IC inesperado {cfg_ic}')
    # (1 − 0.95)/2 en coma flotante es 0.025000000000000022; con Decimal, 0.025 y 0.975 exactos (A5: 2.5–97.5)
    a = (Decimal(1) - Decimal(str(cfg_ic['nivel']))) / 2
    return float(a), float(1 - a)


def retencion(F, cfg, fase='final'):
    """ρ̂ = (F1_img − azar)/(F1_BETO − azar) (§5.6). Réplicas con F1_BETO* ≤ azar → −inf y se
    cuentan (A6); IC percentil bilateral al 95 %, cuantiles linear (A5)."""
    azar = cfg['contraste']['f1_azar']
    e4 = cfg['enmienda_4']['retencion']
    if (e4['valor_replica'], e4['reportar_conteo']) != ('-inf', True):
        raise ValueError(f'enmienda_4.retencion inesperado {e4}')
    lo, hi = _ci95(cfg['enmienda_4']['ic']['descriptivo'])
    b_obs, b_rep = ref(F, 'beto')
    out = {}
    for t in TECHNIQUES:
        i_obs, i_rep = img(F, fase, t)
        den = b_rep - azar
        bad = den <= 0
        rho = np.where(bad, -np.inf, (i_rep - azar) / np.where(bad, 1.0, den))
        rho_obs = (i_obs - azar) / (b_obs - azar) if b_obs > azar else float('-inf')
        out[t] = {'rho_obs': float(rho_obs), 'interval': bs.percentile_interval(rho, lo, hi),
                  'n_denominador_no_positivo': int(bad.sum())}
    return out


def descomposicion(F, cfg, fase='final'):
    """§7: cabeza (BETO − 768D), reducción (768D − 5D), renderizado (5D − img); IC 95 % (A5)."""
    lo, hi = _ci95(cfg['enmienda_4']['ic']['descriptivo'])
    out = {}
    for t in TECHNIQUES:
        i_obs, i_rep = img(F, fase, t)
        stages = {'cabeza': ('beto', 'emb768'), 'reduccion': ('emb768', 'techo5d'), 'renderizado': ('techo5d', None)}
        res = {}
        total_obs = 0.0
        for s, (a, b) in stages.items():
            a_obs, a_rep = ref(F, a)
            b_obs, b_rep = (i_obs, i_rep) if b is None else ref(F, b)
            res[s] = {'obs': float(a_obs - b_obs), 'interval': bs.percentile_interval(a_rep - b_rep, lo, hi)}
            total_obs += a_obs - b_obs
        res['total_obs'] = float(total_obs)
        res['beto_menos_img_obs'] = float(ref(F, 'beto')[0] - i_obs)
        out[t] = res
    return out


def frente_a_descriptores(F, cfg, fase='final'):
    """§7: F1_img − F1_descriptores con IC percentil bilateral 95 % sobre las réplicas de §5.2 (A11)."""
    e = cfg['enmienda_4']['ic']['descriptores']
    if e['replicas'] != 'contraste.bootstrap':
        raise ValueError(e)
    lo, hi = _ci95(e)
    out = {}
    for t in TECHNIQUES:
        i_obs, i_rep = img(F, fase, t)
        d_obs, d_rep = ref(F, ('descriptors', t))
        out[t] = {'obs': float(i_obs - d_obs), 'interval': bs.percentile_interval(i_rep - d_rep, lo, hi)}
    return out


# ── Longitud (§6; A7) ─────────────────────────────────────────────────────
def contraste_longitud(F, cfg, verdicts_by_tech, fase='final', m_mode='principal', seed_index=None):
    """L = F1_img − F1_longitud, p = #{L* ≤ 0}/B, Holm con α de §6.

    m_mode='principal' (A7): solo técnicas con V1, V2 o V4; m = su número.
    m_mode='m3' (sensibilidad de A7): las tres técnicas, m = 3."""
    lcfg = cfg['longitud']
    e7 = cfg['enmienda_4']['longitud']['contraste']
    if e7['p_value'] != '#{L* <= 0} / B' or lcfg['applies_to'] != ['V1', 'V2', 'V4']:
        raise ValueError('bloque longitud inesperado')
    if m_mode == 'principal':
        techs = [t for t in TECHNIQUES if verdicts_by_tech[t] in lcfg['applies_to']]
    elif m_mode == 'm3':
        techs = list(TECHNIQUES)
    else:
        raise ValueError(m_mode)
    out = {'m': len(techs), 'techniques': {}}
    if not techs:
        return out
    l_obs, l_rep = ref(F, 'longitud')
    p = {}
    for t in techs:
        i_obs, i_rep = img(F, fase, t, seed_index)
        p[t], below = verdict.p_value(i_rep - l_rep)
        out['techniques'][t] = {'L_obs': float(i_obs - l_obs), 'p': p[t], 'below_resolution': below}
    h = verdict.holm(p, lcfg['alpha'])
    for t in techs:
        out['techniques'][t].update(h[t])
        out['techniques'][t]['calificacion'] = None if h[t]['reject'] else lcfg['qualification']
    return out


# ── Predicción por muestra, Cochran y McNemar (§7; A12) ──────────────────
def yhat_mean(p_seeds, cfg):
    """ŷ = 1 si la media de p_dep de las 5 semillas > 0.5 (§7)."""
    s = cfg['secundarios']['prediccion_por_muestra']
    p = np.asarray(p_seeds, dtype=np.float64)
    if s['aggregate'] != 'mean_p_dep' or p.shape[1] != s['seeds']:
        raise ValueError(f'predicción por muestra inesperada {s}, {p.shape}')
    return (p.mean(axis=1) > s['threshold']).astype(np.int64)


def cochran_q(correct, cfg):
    """Q de Cochran asintótica (χ², gl = k − 1) sobre una matriz n × k de aciertos 0/1.

    Si todos los bloques son concordantes (0/0), Q y p son los de la Enmienda 5,
    P5 (Q = 0, p = 1). Devuelve (Q, p, gl, todos_concordantes)."""
    X = np.asarray(correct, dtype=np.float64)
    k = X.shape[1]
    C, R, N = X.sum(axis=0), X.sum(axis=1), X.sum()
    den = k * N - (R ** 2).sum()
    if den == 0:
        tc = cfg['enmienda_5']['cochran']['todos_concordantes']
        return float(tc['Q']), float(tc['p']), k - 1, True
    q = (k - 1) * (k * (C ** 2).sum() - N ** 2) / den
    return float(q), float(stats.chi2.sf(q, k - 1)), k - 1, False


def _discordant(correct_a, correct_b):
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    return int((a & ~b).sum()), int((~a & b).sum())


def mcnemar_bilateral(correct_a, correct_b):
    """McNemar exacto bilateral: p = min(1, 2·Binom(n, ½).cdf(min(b, c)))."""
    b, c = _discordant(correct_a, correct_b)
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))
    return {'b': b, 'c': c, 'p': p}


def mcnemar_unilateral_more_errors(correct_ref, correct_vis):
    """McNemar exacto unilateral, H1: el visual comete más errores que la referencia (A12).
    b = ref acierta y visual falla; c = al revés. p = P(X ≥ b), X ~ Binom(b + c, ½)."""
    b, c = _discordant(correct_ref, correct_vis)
    n = b + c
    p = 1.0 if n == 0 else float(stats.binom.sf(b - 1, n, 0.5))
    return {'b': b, 'c': c, 'p': p}


def comparaciones_mcnemar(y, yhat, cfg, lado_vs_ref=None):
    """Q de Cochran entre los tres finalistas y, si p < α, McNemar bilateral por pares con Holm;
    McNemar frente a BETO y al techo 5D por técnica, con Holm en cada familia de tres (A12).
    yhat: {'beto', 'techo5d', técnica…: ŷ (n,)}."""
    e = cfg['enmienda_4']['cochran_mcnemar']
    if e['cochran'] != {'tipo': 'asintotico', 'distribucion': 'chi2', 'gl': 2} or e['mcnemar']['tipo'] != 'exacto':
        raise ValueError(f'enmienda_4.cochran_mcnemar inesperado {e}')
    m = e['mcnemar']
    lado = lado_vs_ref or {'beto': m['lado_vs_beto'], 'techo5d': m['lado_vs_techo']}
    alpha_ct = cfg['secundarios']['entre_tecnicas']['alpha']
    alpha = cfg['contraste']['multiplicidad']['alpha']          # §13: α = 0.05, Holm por familia
    ok = {k: (np.asarray(v) == y).astype(int) for k, v in yhat.items()}
    q, p_q, gl, concordantes = cochran_q(np.column_stack([ok[t] for t in TECHNIQUES]), cfg)
    if gl != e['cochran']['gl']:
        raise ValueError('gl de Cochran')
    out = {'cochran': {'Q': q, 'p': p_q, 'gl': gl, 'alpha': alpha_ct, 'todos_concordantes': concordantes},
           'pares': None, 'frente_a': {}}
    pares_ok = not concordantes or cfg['enmienda_5']['cochran']['todos_concordantes']['mcnemar_pares']
    if pares_ok and p_q < alpha_ct:
        pares = {f'{a}__{b}': mcnemar_bilateral(ok[a], ok[b]) for a, b in combinations(TECHNIQUES, 2)}
        h = verdict.holm({k: v['p'] for k, v in pares.items()}, alpha)
        out['pares'] = {k: {**v, **h[k], 'lado': m['lado_entre_tecnicas']} for k, v in pares.items()}
    for refname in ('beto', 'techo5d'):
        fam = {}
        for t in TECHNIQUES:
            if lado[refname] == 'unilateral':
                fam[t] = mcnemar_unilateral_more_errors(ok[refname], ok[t])
            elif lado[refname] == 'bilateral':
                fam[t] = mcnemar_bilateral(ok[refname], ok[t])
            else:
                raise ValueError(lado[refname])
        h = verdict.holm({t: v['p'] for t, v in fam.items()}, alpha)
        out['frente_a'][refname] = {t: {**fam[t], **h[t], 'lado': lado[refname]} for t in TECHNIQUES}
    return out


# ── F4.6: kappa y contención (A10; docs/plan_fase4.md) ───────────────────
def _joint(W, a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n11 = W @ (a * b)
    n10 = W @ (a * (1 - b))
    n01 = W @ ((1 - a) * b)
    n00 = W @ ((1 - a) * (1 - b))
    return n11, n10, n01, n00


def kappa_weighted(W, a, b):
    """Kappa de Cohen de dos vectores 0/1 bajo cada fila de W; nan donde 1 − p_e = 0 (indefinida)."""
    n11, n10, n01, n00 = _joint(W, a, b)
    N = n11 + n10 + n01 + n00
    po = (n11 + n00) / N
    pa, pb = (n11 + n10) / N, (n11 + n01) / N
    pe = pa * pb + (1 - pa) * (1 - pb)
    den = 1 - pe
    return np.divide(po - pe, den, out=np.full(den.shape, np.nan), where=den != 0)


def _indefinida_spec(cfg, name):
    s = cfg['enmienda_5'][name]['indefinida']
    if s != {'replicas': 'excluir_y_contar', 'observado': NO_APLICA}:
        raise ValueError(f'enmienda_5.{name}.indefinida inesperado {s}')
    return s


def _summary_excluding(obs, reps, lo, hi):
    """Enmienda 5, P3/P4: observado indefinido → «no aplica»; réplicas indefinidas se excluyen
    del IC y se cuentan."""
    reps = np.asarray(reps, dtype=np.float64)
    bad = np.isnan(reps)
    if np.isnan(obs):
        return {'obs': NO_APLICA, 'interval': NO_APLICA, 'n_replicas_indefinidas': int(bad.sum())}
    ok = reps[~bad]
    return {'obs': float(obs), 'interval': bs.percentile_interval(ok, lo, hi) if ok.size else NO_APLICA,
            'n_replicas_indefinidas': int(bad.sum())}


def kappa(W, y, yhat, cfg):
    """Kappa por pares entre BETO, 768D, techo 5D y visual (por técnica), sobre ŷ (principal)
    y sobre aciertos (secundario), con IC percentil bilateral 95 % (A10)."""
    k = cfg['enmienda_4']['kappa']
    lo, hi = _ci95(k['ic'])
    if (k['principal'], k['secundario']) != ('y_hat', 'vector_acierto'):
        raise ValueError(k)
    _indefinida_spec(cfg, 'kappa')
    W_obs = bs.observed_weights(len(y))
    names = ['beto', 'emb768', 'techo5d'] + list(TECHNIQUES)
    out = {}
    for base in ('y_hat', 'vector_acierto'):
        vec = {n: (np.asarray(yhat[n]) if base == 'y_hat' else (np.asarray(yhat[n]) == y).astype(int)) for n in names}
        res = {}
        for a, b in combinations(names, 2):
            if a in TECHNIQUES and b in TECHNIQUES:
                continue
            res[f'{a}__{b}'] = _summary_excluding(kappa_weighted(W_obs, vec[a], vec[b])[0],
                                                  kappa_weighted(W, vec[a], vec[b]), lo, hi)
        out[base] = res
    return out


def contencion(W, y, yhat_vis, yhat_ref, cfg):
    """C = |E_vis ∩ E_ref| / |E_vis|, referencia nula |E_ref| / n, IC percentil bilateral 95 %
    sobre las réplicas de §5.2 (docs/plan_fase4.md, F4.6). |E_vis| = 0 → Enmienda 5, P4
    (observado «no aplica»; réplicas excluidas y contadas)."""
    _indefinida_spec(cfg, 'contencion')
    lo, hi = _ci95(cfg['enmienda_4']['ic']['descriptivo'])
    ev = (np.asarray(yhat_vis) != y).astype(np.float64)
    er = (np.asarray(yhat_ref) != y).astype(np.float64)
    num, den = W @ (ev * er), W @ ev
    reps = np.divide(num, den, out=np.full(den.shape, np.nan), where=den > 0)
    obs = (ev * er).sum() / ev.sum() if ev.sum() > 0 else np.nan
    s = _summary_excluding(obs, reps, lo, hi)
    return {'C_obs': s['obs'], 'nula_obs': float(er.mean()), 'interval': s['interval'],
            'n_replicas_indefinidas': s['n_replicas_indefinidas'],
            'n_errores_vis': int(ev.sum()), 'n_errores_ref': int(er.sum()), 'n_interseccion': int((ev * er).sum())}


# ── §6 descriptivo (A8) ───────────────────────────────────────────────────
def tercile_cuts(n_tokens_train, cfg):
    t = cfg['enmienda_4']['longitud']['terciles']
    if t['cortes'] != "np.quantile(n_tokens_beto[train], [1/3, 2/3], method='linear')" \
            or t['quantile_method'] != 'linear' or t['intervalos'] != {'T1': '<= c1', 'T2': '(c1, c2]', 'T3': '> c2'}:
        raise ValueError(f'enmienda_4.longitud.terciles inesperado {t}')
    return np.quantile(np.asarray(n_tokens_train, dtype=np.float64), [1 / 3, 2 / 3], method='linear')


def assign_terciles(n_tokens, cuts):
    n = np.asarray(n_tokens, dtype=np.float64)
    c1, c2 = cuts
    return np.where(n <= c1, 'T1', np.where(n <= c2, 'T2', 'T3'))


def terciles_longitud(y, n_tokens_test, n_tokens_train, yhat_seeds, yhat_beto, cfg):
    """F1-Macro por tercil de n_tokens_beto (cortes en train): finalista = F1 por semilla y luego
    la media; BETO. yhat_seeds: {técnica: (n × S)}. Descriptivo (A8)."""
    t = cfg['enmienda_4']['longitud']['terciles']
    if t['f1'] != 'por_semilla_luego_media' or t['modelos'] != ['finalista', 'beto']:
        raise ValueError(t)
    cuts = tercile_cuts(n_tokens_train, cfg)
    lab = assign_terciles(n_tokens_test, cuts)
    out = {'cortes': [float(c) for c in cuts], 'estratos': {}}
    for T in ('T1', 'T2', 'T3'):
        m = lab == T
        W1 = bs.observed_weights(int(m.sum()))
        e = {'n': int(m.sum()), 'n_dep': int(y[m].sum()), 'beto': float(bs.f1_macro(W1, y[m], yhat_beto[m])[0])}
        for tech, Y in yhat_seeds.items():
            per_seed = bs.f1_macro(W1, y[m], np.asarray(Y)[m])[0]
            e[tech] = {'por_semilla': [float(v) for v in per_seed], 'media': float(per_seed.mean())}
        out['estratos'][T] = e
    return out


def spearman_u_longitud(u_train, n_tokens_train, truncated_train, cfg):
    """Spearman de u0–u4 con log(n_tokens_beto) y truncated_128 en train (A8). Descriptivo."""
    c = cfg['enmienda_4']['longitud']['terciles']['correlacion']
    clf = cfg['enmienda_4']['longitud']['clasificador']
    if c['metodo'] != 'spearman' or c['split'] != 'train' or clf['log'] != 'natural':
        raise ValueError(c)
    tr = np.array([clf['truncated_128'][str(v).strip().lower()] for v in truncated_train], dtype=np.float64)
    covs = {'log(n_tokens_beto)': np.log(np.asarray(n_tokens_train, dtype=np.float64)), 'truncated_128': tr}
    out = {}
    for j, name in enumerate(c['variables']):
        for cov in c['contra']:
            res = stats.spearmanr(u_train[:, j], covs[cov])
            out[f'{name}__{cov}'] = {'rho': float(res.statistic), 'p': float(res.pvalue)}
    return out
