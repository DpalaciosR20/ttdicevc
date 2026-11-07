"""
sensibilidad.py
───────────────
Análisis de sensibilidad (§8) con las resoluciones de la Enmienda 4, y la tabla
de robustez que exige la relación con el §11 de esa enmienda. Ninguna variante
modifica el veredicto principal.

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

Tabla de robustez (alternativas declaradas): A2 máximo por réplica; A3 (solo
aplica si hay empate en el máximo observado); A7 con m = 3 y en cada variante;
A12 bilateral frente a BETO y al techo; A14 semilla emparejada.

PENDIENTE (no resuelto): la sensibilidad de A14, «piso permutado de la semilla
emparejada, si es mayor», no fija frente a qué se compara (el observado o cada
réplica). Se exige enmienda_4.por_semilla.sensibilidad.comparacion ∈
{"observado", "por_replica"}; sin ella, la variante lanza PendienteError.
Del mismo modo, la alternativa de A3 no está declarada: si hay empate en el
máximo observado, la fila de A3 lanza PendienteError.
"""

import numpy as np

from src.protocol import contraste as cx
from src.protocol.contraste import PendienteError, TECHNIQUES


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
        out[f'r={r}'] = cx.contraste(F, cfg, formulation='A', delta=delta)
    return out


def por_semilla(F, cfg, permutado='media', comparacion=None):
    """Veredicto con cada semilla por separado (A14). Holm entre técnicas dentro de cada semilla
    (cada llamada a contraste es una familia de tres). Piso permutado = media de 5 (principal) o,
    como sensibilidad, el de la semilla emparejada si es mayor (comparacion = 'observado' o
    'por_replica', PENDIENTE de fijar)."""
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
            out[s] = _por_semilla_emparejada(F, cfg, i, comparacion)
        else:
            raise ValueError(permutado)
    n_v1 = {t: sum(out[s]['techniques'][t]['verdict'] == 'V1' for s in seeds) for t in TECHNIQUES}
    return {'por_semilla': out, 'n_semillas_V1': n_v1}


def _por_semilla_emparejada(F, cfg, i, comparacion):
    if comparacion not in ('observado', 'por_replica'):
        raise PendienteError('A14 (sensibilidad): «si es mayor» no fija la comparación (observado o por '
                             'réplica); falta enmienda_4.por_semilla.sensibilidad.comparacion')
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
        if comparacion == 'observado':
            cands['visual_permutado'] = (s_obs, s_rep) if s_obs > m_obs else (m_obs, m_rep)
        else:
            cands['visual_permutado'] = (max(s_obs, m_obs), np.maximum(s_rep, m_rep))
        d_obs, d_rep, info = cx.d_reps(i_obs, i_rep, cands)
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


def robustez(F, cfg, principal, variantes, mcnemar_principal=None, mcnemar_bilateral=None, comparacion_a14=None):
    """Tabla de robustez (Enmienda 4, relación con el §11): veredicto por técnica y global bajo
    cada alternativa declarada y si cambia respecto del principal."""
    rows = []
    rows.append(_row('A2: máximo por réplica', principal, cx.contraste(F, cfg, floor_mode='max_por_replica')))
    if any(principal['techniques'][t]['empate'] for t in TECHNIQUES):
        raise PendienteError('A3: hay empate en el máximo observado y la Enmienda 4 no declara alternativa')
    rows.append({'alternativa': 'A3: empate en el máximo observado', 'veredictos': _verdicts(principal),
                 'global': _global(principal), 'cambia_tecnica': {t: False for t in TECHNIQUES},
                 'cambia_global': False, 'nota': 'sin empate en el máximo observado: A3 no aplica'})
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
    ps = por_semilla(F, cfg, permutado='emparejada', comparacion=comparacion_a14)
    base = variantes['por_semilla']
    rows.append({'alternativa': 'A14: piso permutado de la semilla emparejada, si es mayor',
                 'n_semillas_V1': ps['n_semillas_V1'],
                 'cambia_n_semillas_V1': {t: ps['n_semillas_V1'][t] != base['n_semillas_V1'][t] for t in TECHNIQUES},
                 'por_semilla': {s: {'veredictos': _verdicts(r), 'global': _global(r)} for s, r in ps['por_semilla'].items()}})
    return rows
