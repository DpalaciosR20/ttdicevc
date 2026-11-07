"""
run_fase5.py
────────────
Punto de entrada único de la Fase 5 (F5.1–F5.6, contraste de F4.5) y del
análisis de errores de F4.6.

Modos:
  (sin bandera) o --verificar-insumos
        Solo presencia y alineación de los insumos (sample_id, split, y_true,
        identidad y csv_sha256). No carga p_dep. No escribe resultados.
  --ejecutar-prueba
        Ejecuta la Fase 5 sobre las predicciones de prueba. Exige:
          · la etiqueta GATE_TAG (enmienda-5) en la historia de HEAD;
          · docs/protocolo_preregistrado.md, configs/protocolo.json y
            docs/plan_fase4.md sin cambios frente a esa etiqueta;
          · el árbol de trabajo limpio (git status --porcelain vacío);
          · que no quede ningún punto PENDIENTE en configs/protocolo.json;
          · que no exista <out>/resultados.json (nunca sobrescribe).
        Escribe <out>/resultados.json (todas las cifras a precisión completa),
        un CSV por análisis y <out>/run_fase5.log (hashes de insumos y commit).
  --analisis-f46
        Kappa y contención de errores (F4.6). Por docs/plan_fase4.md (G1) usa
        prueba solo después del commit del veredicto: exige además que
        <out>/resultados.json esté versionado y sin cambios, y que no exista
        <out>/f46_resultados.json.

Uso:  python -m src.protocol.run_fase5 [--verificar-insumos | --ejecutar-prueba | --analisis-f46]
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import paths
from src.protocol import bootstrap as bs
from src.protocol import contraste as cx
from src.protocol import insumos as ins_mod
from src.protocol import sensibilidad as sens
from src.protocol.contraste import TECHNIQUES

REPO = paths.REPO
OUT_DEFAULT = paths.RESULTS_DIR / 'f5_contraste'
GATE_TAG = 'enmienda-5'                     # única constante de la compuerta: etiqueta de la Enmienda 5
FROZEN = ('docs/protocolo_preregistrado.md', 'configs/protocolo.json', 'docs/plan_fase4.md')
CANON_PATH = paths.RESULTS_DIR / 'canonical_numbers.json'


class Negativa(RuntimeError):
    """run_fase5 se niega a ejecutar."""


# ── Compuertas ────────────────────────────────────────────────────────────
def _git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True)


def check_gate():
    if _git('merge-base', '--is-ancestor', GATE_TAG, 'HEAD').returncode != 0:
        raise Negativa(f'la etiqueta {GATE_TAG} no es ancestro de HEAD')
    r = _git('diff', '--quiet', GATE_TAG, '--', *FROZEN)
    if r.returncode != 0:
        raise Negativa(f'{", ".join(FROZEN)} cambiaron frente a {GATE_TAG}')


def check_clean_tree():
    r = _git('status', '--porcelain')
    if r.returncode != 0 or r.stdout.strip():
        raise Negativa('el árbol de trabajo no está limpio:\n' + r.stdout)


def pendientes(cfg):
    """Puntos que bloquean --ejecutar-prueba: resoluciones de la Enmienda 5 ausentes del config."""
    e5 = cfg.get('enmienda_5', {})
    need = {'P1': ('por_semilla', 'sensibilidad', 'comparacion'), 'P2': ('piso', 'empate_alternativa'),
            'A3': ('piso', 'empate_tolerancia'), 'P3': ('kappa', 'indefinida'),
            'P4': ('contencion', 'indefinida'), 'P5': ('cochran', 'todos_concordantes')}
    out = []
    for pid, path in need.items():
        d = e5
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            out.append(f'{pid}: falta enmienda_5.{".".join(path)}')
    return out


def avisos(cfg):
    """Pendientes que no bloquean la Fase 5 (se reportan)."""
    return [f'P6: {v}' for k, v in cfg.get('enmienda_5', {}).get('pendiente', {}).items() if k == 'P6']


def check_results_absent(out_dir, name='resultados.json'):
    if (Path(out_dir) / name).exists():
        raise Negativa(f'{Path(out_dir) / name} ya existe: run_fase5 nunca sobrescribe una ejecución previa')


def check_results_committed(out_dir):
    p = Path(out_dir) / 'resultados.json'
    rel = p.relative_to(REPO).as_posix() if p.is_absolute() and REPO in p.parents else str(p)
    if not p.exists() or _git('ls-files', '--error-unmatch', rel).returncode != 0:
        raise Negativa(f'{rel} no está versionado: F4.6 usa prueba solo después del commit del veredicto (G1)')
    if _git('diff', '--quiet', 'HEAD', '--', rel).returncode != 0:
        raise Negativa(f'{rel} tiene cambios sin commitear')


# ── Cálculo (testeable con insumos sintéticos) ────────────────────────────
def predicciones(ins, cfg):
    thr = cfg['modelos']['decision_threshold']
    if cfg['modelos']['decision_rule'] != 'p_dep > threshold':
        raise ValueError(cfg['modelos']['decision_rule'])
    preds = {k: bs.predict(p, thr) for k, p in ins.refs.items()}
    for (fase, t), v in ins.visual.items():
        preds[('img', fase, t)] = bs.predict(v['p'], thr)
    return preds


def ejecutar(ins, cfg, canon, idx, log=print):
    """Fase 5 completa sobre insumos ya cargados e índices de A1."""
    t0 = time.time()
    y = ins.y
    W = bs.weights(idx, len(y))
    F = cx.f1_table(W, y, predicciones(ins, cfg))
    log(f'  F1 de {len(F)} clasificadores en {W.shape[0]} réplicas: {time.time() - t0:.1f} s')
    principal = cx.contraste(F, cfg)
    v = {t: r['verdict'] for t, r in principal['techniques'].items()}
    yhat = {'beto': bs.predict(ins.refs['beto'], cfg['modelos']['decision_threshold']),
            'techo5d': bs.predict(ins.refs['techo5d'], cfg['modelos']['decision_threshold'])}
    for t in TECHNIQUES:
        yhat[t] = cx.yhat_mean(ins.visual[('final', t)]['p'], cfg)
    mcn = cx.comparaciones_mcnemar(y, yhat, cfg)
    mcn_bil = cx.comparaciones_mcnemar(y, yhat, cfg, lado_vs_ref={'beto': 'bilateral', 'techo5d': 'bilateral'})
    variantes = sens.todas(F, cfg, canon)
    res = {
        'principal': principal,
        'observados': {str(k): v['obs'] for k, v in F.items()},
        'retencion': cx.retencion(F, cfg),
        'descomposicion': cx.descomposicion(F, cfg),
        'frente_a_descriptores': cx.frente_a_descriptores(F, cfg),
        'longitud': cx.contraste_longitud(F, cfg, v),
        'mcnemar': mcn,
        'sensibilidad': variantes,
        'robustez': sens.robustez(F, cfg, principal, variantes, mcn, mcn_bil),
        'terciles_longitud': cx.terciles_longitud(
            y, ins.test_index['n_tokens_beto'].to_numpy(), ins.train['n_tokens_beto'].to_numpy(),
            {t: bs.predict(ins.visual[('final', t)]['p'], cfg['modelos']['decision_threshold']) for t in TECHNIQUES},
            yhat['beto'], cfg),
        'spearman_u_longitud': (cx.spearman_u_longitud(ins.u_train, ins.train['n_tokens_beto'].to_numpy(),
                                                       ins.train['truncated_128'].to_numpy(), cfg)
                                if ins.u_train is not None else None),
    }
    log(f'  cálculo completo: {time.time() - t0:.1f} s')
    return res


def ejecutar_f46(ins, cfg, idx):
    y = ins.y
    W = bs.weights(idx, len(y))
    thr = cfg['modelos']['decision_threshold']
    yhat = {k: bs.predict(ins.refs[k], thr) for k in ('beto', 'emb768', 'techo5d')}
    for t in TECHNIQUES:
        yhat[t] = cx.yhat_mean(ins.visual[('final', t)]['p'], cfg)
    out = {'kappa': cx.kappa(W, y, yhat, cfg), 'contencion': {}, 'contencion_por_semilla': {}}
    for t in TECHNIQUES:
        out['contencion'][t] = {r: cx.contencion(W, y, yhat[t], yhat[r], cfg) for r in ('techo5d', 'beto', 'emb768')}
        c = out['contencion'][t]['techo5d']
        out['contencion'][t]['condicion_1'] = (cx.NO_APLICA if c['interval'] == cx.NO_APLICA
                                               else c['interval'][0] > c['nula_obs'])
        out['contencion'][t]['condicion_2_F41'] = 'PENDIENTE (P6): requiere el IC de ΔF1 de F4.1'
        P = bs.predict(ins.visual[('final', t)]['p'], thr)
        out['contencion_por_semilla'][t] = {
            str(s): cx.contencion(W, y, P[:, i], yhat['techo5d'], cfg)
            for i, s in enumerate(ins.visual[('final', t)]['seeds'])}
    return out


# ── Salida ────────────────────────────────────────────────────────────────
def _clean(o):
    """JSON estricto: −inf/inf/nan como cadenas, tuplas como listas, claves como texto."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else ('-inf' if f < 0 else ('inf' if f > 0 else 'nan'))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(_clean(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else _clean(v)
                        for k, v in r.items()})


def _verdict_rows(res, label):
    rows = []
    for t, r in res['techniques'].items():
        rows.append({'analisis': label, 'tecnica': t, 'veredicto': r['verdict'], 'global': res['global'],
                     'theta_obs': r['theta_obs'], 'theta_interval': r['theta_interval'],
                     'interval_level': r['interval_level'],
                     'p_no_inferioridad': r['no_inferioridad']['p'], 'rechaza_NI': r['no_inferioridad']['reject'],
                     'D_obs': r['D_obs'], 'D_interval': r['D_interval'], 'p_superioridad': r['superioridad_piso']['p'],
                     'supera_piso': r['superioridad_piso']['reject'], 'piso': r['piso'],
                     'f1_img_obs': r['f1_img_obs']})
    return rows


def write_outputs(out_dir, res, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {'meta': meta, 'resultados': res}
    (out_dir / 'resultados.json').write_text(json.dumps(_clean(doc), indent=1, ensure_ascii=False) + '\n',
                                            encoding='utf-8', newline='\n')
    rows = _verdict_rows(res['principal'], 'principal')
    for name, group in res['sensibilidad'].items():
        if name == 'por_semilla':
            for s, r in group['por_semilla'].items():
                rows += _verdict_rows(r, f'por_semilla/{s}')
        elif name in ('segundo_mejor', 'piso_sin_permutado'):
            rows += _verdict_rows(group, name)
        else:
            for k, r in group.items():
                rows += _verdict_rows(r, f'{name}/{k}')
    _write_csv(out_dir / 'veredictos.csv', rows)
    _write_csv(out_dir / 'robustez.csv', res['robustez'])
    for name in ('retencion', 'descomposicion', 'frente_a_descriptores'):
        _write_csv(out_dir / f'{name}.csv', [{'tecnica': t, **v} for t, v in res[name].items()])
    _write_csv(out_dir / 'longitud.csv', [{'tecnica': t, 'm': res['longitud']['m'], **v}
                                          for t, v in res['longitud']['techniques'].items()])
    m = res['mcnemar']
    _write_csv(out_dir / 'mcnemar.csv',
               [{'contraste': 'cochran', **m['cochran']}]
               + [{'contraste': f'par/{k}', **v} for k, v in (m['pares'] or {}).items()]
               + [{'contraste': f'{ref}/{t}', **v} for ref, fam in m['frente_a'].items() for t, v in fam.items()])
    _write_csv(out_dir / 'terciles_longitud.csv', [{'estrato': k, **v} for k, v in
                                                   res['terciles_longitud']['estratos'].items()])
    if res['spearman_u_longitud']:
        _write_csv(out_dir / 'spearman_u_longitud.csv', [{'par': k, **v} for k, v in res['spearman_u_longitud'].items()])


def _meta(cfg, files, idx):
    return {'git_commit': _git('rev-parse', 'HEAD').stdout.strip(),
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'gate_tag': GATE_TAG, 'gate_tag_commit': _git('rev-parse', f'{GATE_TAG}^{{commit}}').stdout.strip(),
            'config_sha256_lf': ins_mod.text_sha256(ins_mod.CONFIG_PATH),
            'canonical_sha256_lf': ins_mod.text_sha256(CANON_PATH),
            'bootstrap': {'call': cfg['enmienda_4']['bootstrap']['call'], 'seed': cfg['enmienda_4']['bootstrap']['seed'],
                          'idx_sha256': hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()},
            'inputs': files}


def main(argv=None, out_dir=OUT_DEFAULT, log=print):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--verificar-insumos', action='store_true')
    g.add_argument('--ejecutar-prueba', action='store_true')
    g.add_argument('--analisis-f46', action='store_true')
    args = ap.parse_args(argv)
    out_dir = Path(out_dir)
    cfg = ins_mod.load_config()
    lines = []

    def L(msg=''):
        lines.append(str(msg))
        log(msg)

    if not (args.ejecutar_prueba or args.analisis_f46):
        L('run_fase5 — solo --verificar-insumos (sin --ejecutar-prueba no se carga p_dep)')
        ins_mod.verificar_insumos(cfg, log=L)
        return 0
    try:
        check_gate()
        check_clean_tree()
        p = pendientes(cfg)
        if p:
            raise Negativa('quedan puntos PENDIENTES (Enmienda 5):\n  - ' + '\n  - '.join(p))
        for a in avisos(cfg):
            L(f'AVISO (no bloquea): {a}')
        if args.ejecutar_prueba:
            check_results_absent(out_dir)
        else:
            check_results_committed(out_dir)
            check_results_absent(out_dir, 'f46_resultados.json')
    except Negativa as e:
        L(f'NEGATIVA: {e}')
        return 2
    canon = json.loads(CANON_PATH.read_text(encoding='utf-8'))['numbers']
    idx = bs.indices(cfg)
    L(f'Compuerta {GATE_TAG}: OK; árbol limpio; HEAD {_git("rev-parse", "HEAD").stdout.strip()}')
    ins = ins_mod.cargar(cfg)
    meta = _meta(cfg, ins.files, idx)
    for k, h in sorted(ins.files.items()):
        L(f'  sha256 {h}  {k}')
    if args.ejecutar_prueba:
        res = ejecutar(ins, cfg, canon, idx, log=L)
        write_outputs(out_dir, res, meta)
        L(f'escrito {out_dir / "resultados.json"}')
        (out_dir / 'run_fase5.log').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    else:
        res = ejecutar_f46(ins, cfg, idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'f46_resultados.json').write_text(
            json.dumps(_clean({'meta': meta, 'resultados': res}), indent=1, ensure_ascii=False) + '\n',
            encoding='utf-8', newline='\n')
        (out_dir / 'run_f46.log').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
