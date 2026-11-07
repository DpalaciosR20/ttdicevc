"""
sensibilidad.py
───────────────
Análisis de sensibilidad (§8) con las resoluciones de las Enmiendas 4 y 5, y la
tabla de robustez que exige la relación con el §11 de la Enmienda 4. Ninguna
variante modifica el veredicto principal.

Variantes de §8:
  retencion_exigida     r ∈ sensibilidad.retencion_exigida.r
  formulacion_A         r ∈ enmienda_4.formulacion_A.r, δ fijo de las cifras
                        canónicas f11.delta_margen.* (A13)
  por_semilla           una semilla a la vez; piso permutado = media de 5;
                        Holm entre técnicas dentro de cada semilla (A14)
  variabilidad_beto     F1_BETO ± f12.variabilidad_beto en el observado y en
                        cada réplica (A15)
  segundo_mejor         fase sensibilidad con el piso permutado del finalista (A17)
  piso_sin_permutado    piso = máximo de sensibilidad.piso_sin_permutado

Tabla de robustez (alternativas declaradas): A2 máximo por réplica; A3 con la
alternativa de la Enmienda 5, P2 (el primero de los empatados según
pisos.order); A7 con m = 3 y en cada variante; A12 bilateral frente a BETO y al
techo; A14 semilla emparejada con la comparación por réplica de la Enmienda 5,
P1 (en cada réplica, el mayor entre el permutado de la semilla emparejada y la
media de las 5 semillas).
"""

import numpy as np

from src.protocol import contraste as cx
from src.protocol.contraste import TECHNIQUES


def _verdicts(res):
    return {t: r['verdict'] for t, r in res['techniques'].items()}


def canonical_value(canon, nid, split='test'):
    e = canon[nid]
    v = e['value'][split] if isinstance(e['value'], dict) else e['value']
    if v is None:
        raise ValueError(f'{nid}[{split}] es nulo')
    return v


def retencion_exigida(F, cfg):
    return {f'r={r}': cx.contraste(F, cfg, r=r) for r in cfg['sensibilidad']['retencion_exigida']['r']}


def formulacion_a(F, cfg, canon):
    e = cfg['enmienda_4']['formulacion_A']
    if (e['h0'], e['p_value'], e['delta']) != ('Delta >= delta', '#{Delta* >= delta} / B', 'fijo'):
        raise ValueError(f'enmienda_4.formulacion_A inesperado {e}')
    out = {}
    for r in e['r']:
        key = {0.5: '0.5', 0.75: '0.75', 0.8: '0.8'}[r]
        delta = canonical_value(canon, e['canonical_delta'][key])
        # r solo se registra en params (la formulación A usa δ); antes quedaba el r por omisión (erratum D3)
        out[f'r={r}'] = cx.contraste(F, cfg, r=r, formulation='A', delta=delta)
    return out


def por_semilla(F, cfg, permutado='media'):
    """Veredicto con cada semilla por separado (A14). Holm entre técnicas dentro de cada semilla
    (cada llamada a contraste es una familia de tres). Piso permutado = media de 5 (principal) o,
    como sensibilidad, el de la semilla emparejada si es mayor, comparado en cada réplica
    (Enmienda 5, P1)."""
    e = cfg['enmienda_4']['por_semilla']
    if (e['piso_permutado'], e['holm'], e['reporte']) != \
            ('media_5_semillas', 'entre_tecnicas_dentro_de_cada_semilla', 'n_semillas_V1'):
        raise ValueError(f'enmienda_4.por_semilla inesperado {e}')
    seeds = cfg['modelos']['seeds']
    out = {}
    for i, s in enumerate(seeds):
        if permutado == 'media':
            out[s] = cx.contraste(F, cfg, seed_index=i)
        elif permutado == 'emparejada':
            out[s] = _por_semilla_emparejada(F, cfg, i)
        else:
            raise ValueError(permutado)
    n_v1 = {t: sum(out[s]['techniques'][t]['verdict'] == 'V1' for s in seeds) for t in TECHNIQUES}
    return {'por_semilla': out, 'n_semillas_V1': n_v1}


def _por_semilla_emparejada(F, cfg, i):
    comparacion = cfg['enmienda_5']['por_semilla']['sensibilidad']['comparacion']
    if comparacion != 'por_replica':
        raise ValueError(f'enmienda_5.por_semilla.sensibilidad.comparacion inesperado {comparacion!r}')
    seeds = cfg['modelos']['seeds']
    floors = [f for f in cfg['pisos']['order'] if f != 'visual_permutado']
    c = cfg['contraste']
    b_obs, b_rep = cx.ref(F, 'beto')
    theta, D, extras = {}, {}, {}
    for t in TECHNIQUES:
        i_obs, i_rep = cx.img(F, 'final', t, i)
        theta[t] = (cx.theta_reps(i_obs, b_obs, c['r'], c['f1_azar']), cx.theta_reps(i_rep, b_rep, c['r'], c['f1_azar']))
        cands = cx.floor_candidates(F, t, cfg, include=floors)
        m_obs, m_rep = cx.img(F, 'control_permutacion', t)
        s_obs, s_rep = cx.img(F, 'control_permutacion', t, i)
        # P1: en cada réplica (y en el observado), el mayor entre la semilla emparejada y la media de 5
        cands['visual_permutado'] = (max(s_obs, m_obs), np.maximum(s_rep, m_rep))
        d_obs, d_rep, info = cx.d_reps(i_obs, i_rep, cands, tol=cx.empate_tolerancia(cfg))
        D[t] = (d_obs, d_rep)
        extras[t] = {'f1_img_obs': i_obs, **info, 'semilla': seeds[i]}
    return cx.veredicto(theta, D, cfg, extras)


def variabilidad_beto(F, cfg, canon):
    v = cfg['enmienda_4']['variabilidad_beto']
    if v['aplica_a'] != ['observado', 'replicas'] or v['signos'] != [1, -1]:
        raise ValueError(v)
    shift = canonical_value(canon, v['canonical'], 'val')
    return {f'{"+" if s > 0 else "-"}{shift}': cx.contraste(F, cfg, beto_shift=s * shift) for s in v['signos']}


def segundo_mejor(F, cfg):
    e = cfg['enmienda_4']['segundo_mejor']
    if e['piso_permutado'] != 'finalista_de_la_tecnica' or cfg['sensibilidad']['segundo_mejor']['fase'] != 'sensibilidad':
        raise ValueError(e)
    # floor_candidates toma el permutado de control_permutacion (finalista de la técnica)
    return cx.contraste(F, cfg, fase='sensibilidad')


def piso_sin_permutado(F, cfg):
    return cx.contraste(F, cfg, floors=cfg['sensibilidad']['piso_sin_permutado'])


def todas(F, cfg, canon):
    return {'retencion_exigida': retencion_exigida(F, cfg),
            'formulacion_A': formulacion_a(F, cfg, canon),
            'por_semilla': por_semilla(F, cfg),
            'variabilidad_beto': variabilidad_beto(F, cfg, canon),
            'segundo_mejor': segundo_mejor(F, cfg),
            'piso_sin_permutado': piso_sin_permutado(F, cfg)}


def _global(res):
    return res['global']


def _row(alt, principal, res, nota=None):
    v0, v1 = _verdicts(principal), _verdicts(res)
    return {'alternativa': alt, 'veredictos': v1, 'global': _global(res),
            'cambia_tecnica': {t: v1[t] != v0[t] for t in TECHNIQUES},
            'cambia_global': _global(res) != _global(principal), 'nota': nota}


def robustez(F, cfg, principal, variantes, mcnemar_principal=None, mcnemar_bilateral=None):
    """Tabla de robustez (Enmienda 4, relación con el §11): veredicto por técnica y global bajo
    cada alternativa declarada y si cambia respecto del principal."""
    rows = []
    rows.append(_row('A2: máximo por réplica', principal, cx.contraste(F, cfg, floor_mode='max_por_replica')))
    if cfg['enmienda_5']['piso']['empate_alternativa'] != 'primero_en_pisos_order':
        raise ValueError(cfg['enmienda_5']['piso'])
    empate = {t: principal['techniques'][t]['empate'] for t in TECHNIQUES}
    rows.append(_row('A3: primero de los empatados según pisos.order (Enmienda 5, P2)', principal,
                     cx.contraste(F, cfg, ties='primero'),
                     nota=('empate en el máximo observado en: ' + ', '.join(t for t in TECHNIQUES if empate[t]))
                     if any(empate.values()) else 'sin empate en el máximo observado: igual al principal'))
    # A7: m = 3 y calificación en cada variante (no cambia V ni G; solo la calificación)
    cal = {'principal': cx.contraste_longitud(F, cfg, _verdicts(principal), m_mode='m3')}
    for name, group in variantes.items():
        if name == 'por_semilla':
            for s, res in group['por_semilla'].items():
                i = cfg['modelos']['seeds'].index(s)
                cal[f'por_semilla/{s}'] = cx.contraste_longitud(F, cfg, _verdicts(res), m_mode='m3', seed_index=i)
        elif name in ('segundo_mejor', 'piso_sin_permutado'):
            fase = 'sensibilidad' if name == 'segundo_mejor' else 'final'
            cal[name] = cx.contraste_longitud(F, cfg, _verdicts(group), fase=fase, m_mode='m3')
        else:
            for k, res in group.items():
                cal[f'{name}/{k}'] = cx.contraste_longitud(F, cfg, _verdicts(res), m_mode='m3')
    rows.append({'alternativa': 'A7: m = 3 y calificación en cada variante de §8', 'veredictos': _verdicts(principal),
                 'global': _global(principal), 'cambia_tecnica': {t: False for t in TECHNIQUES},
                 'cambia_global': False, 'nota': 'L solo califica; V y G no cambian', 'calificaciones': cal})
    if mcnemar_principal is not None and mcnemar_bilateral is not None:
        ch = {ref: {t: mcnemar_principal['frente_a'][ref][t]['reject'] != mcnemar_bilateral['frente_a'][ref][t]['reject']
                    for t in TECHNIQUES} for ref in ('beto', 'techo5d')}
        rows.append({'alternativa': 'A12: McNemar bilateral frente a BETO y al techo', 'veredictos': _verdicts(principal),
                     'global': _global(principal), 'cambia_tecnica': {t: False for t in TECHNIQUES},
                     'cambia_global': False, 'nota': '§7: McNemar no modifica veredictos', 'cambia_rechazo_mcnemar': ch})
    ps = por_semilla(F, cfg, permutado='emparejada')
    base = variantes['por_semilla']
    rows.append({'alternativa': 'A14: piso permutado de la semilla emparejada, si es mayor (por réplica; Enmienda 5, P1)',
                 'n_semillas_V1': ps['n_semillas_V1'],
                 'cambia_n_semillas_V1': {t: ps['n_semillas_V1'][t] != base['n_semillas_V1'][t] for t in TECHNIQUES},
                 'por_semilla': {s: {'veredictos': _verdicts(r), 'global': _global(r)} for s, r in ps['por_semilla'].items()}})
    return rows
