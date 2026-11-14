"""
pixeles_gpu.py
──────────────
FX.1 — partes con inferencia (docs/plan_exploratorio_pixeles.md), Kaggle con GPU T4.
Solo val (regla 2.1).

  C1  Suavizado dirigido: para cada imagen de 256 px, mapa m ∈ {grad, log} y q ∈ {10 %, 20 %},
      se reemplaza el q superior de píxeles por la imagen desenfocada (σ = 4) y se compara con
      la misma máscara desplazada (K = 20 sorteos), con una máscara de azar (K = 20) y con el
      q inferior. p_dep = media de las 5 semillas de F3.5. S_i = δ_alta − δ_desp; Wilcoxon
      (familia H-C, m = 12) y ΔF1-val descriptivo.
  C2  Oclusión 8 × 8 (parches de 32 px por su versión desenfocada con σ = 8) en los 15
      finalistas y los 15 permutados; cordura por datos con el criterio relativo de F4.3 y
      Spearman con la pendiente por parche (solo se interpreta si «cambia»).

Antes de modificar ninguna imagen, cada modelo debe reproducir su p_dep de val de F3.5/F3.6
(≤ tol_pdep). Compuerta: plan-pixeles en la historia de HEAD, plan sin cambios, árbol
limpio, resoluciones versionado y --out-dir vacío.

Uso (Kaggle):
  python -m src.fase_x.pixeles_gpu --models-root /kaggle/input \
      --resoluciones results/fx_pixeles/resoluciones.json --out-dir results/fx_pixeles/gpu
  --limit N: prueba acotada con las N primeras imágenes de val (fuera de results/fx_pixeles/);
  imprime la estimación para las 114 y la compara con el límite de 3 h del plan (§5.C3).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING = os.path.join(REPO, 'src', 'training')
for _p in (REPO, TRAINING):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.fase4 import atribucion_comun as ac  # noqa: E402
from src.fase4.atribucion_comun import Detener  # noqa: E402
from src.fase_x import pixeles_comun as pc  # noqa: E402

MAPAS = ('grad', 'log')


def f1(y, p, umbral=0.5):
    from src.fase4.atribucion_gpu import f1_macro
    return f1_macro(y, (np.asarray(p) > umbral).astype(int))


def cargar_modelos(ub, tech, perm, device):
    """{semilla: modelo en eval en device}, con el sha256 verificado contra el manifiesto y el JSON."""
    import torch
    from build_model import build_model
    from src.fase4 import atribucion_local as al
    arch, strat = ac.FINALIST[tech]
    out, info = {}, {}
    for s in pc.SEEDS:
        p, row = ub.ruta(al.logical_model(tech, s, perm))
        run = al.run_json(tech, s, perm)
        if row['sha256'] != run['model_file']['sha256']:
            raise Detener(f'{tech} s{s} perm={perm}: sha256 del modelo distinto de model_file del JSON')
        m = build_model(arch, freeze_backbone=strat == 'frozen', pretrained=False)
        m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True), strict=True)
        out[s] = m.to(device).eval()
        info[str(s)] = {'sha256': row['sha256'], 'val_probs': run['val_probs']}
    return out, info


def p_modelos(modelos, imgs_u8, tf, device, batch_size):
    """{semilla: p_dep (N,)} para una lista de imágenes uint8."""
    from src.fase4.atribucion_gpu import predecir, tensores
    X = tensores(imgs_u8, tf)
    return {s: predecir(m, X, device, batch_size) for s, m in modelos.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models-root', required=True)
    ap.add_argument('--resoluciones', default=os.path.join(REPO, pc.RESOLUCIONES_PATH))
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'results', 'fx_pixeles', 'gpu'))
    ap.add_argument('--partes', nargs='+', default=['C1', 'C2'], choices=['C1', 'C2'])
    ap.add_argument('--only', nargs='+', default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--k-sorteos', type=int, default=None,
                    help='solo la reducción autorizada por el plan (10); por omisión, 20')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args(argv)

    raw = json.loads(open(a.resoluciones, encoding='utf-8').read())
    res = pc.check_resoluciones(raw)
    k_sorteos = a.k_sorteos or res['k_sorteos']
    if k_sorteos not in (res['k_sorteos'], res['k_sorteos_reduccion_autorizada']):
        raise Detener(f'K = {k_sorteos}: el plan solo admite {res["k_sorteos"]} o la reducción a '
                      f'{res["k_sorteos_reduccion_autorizada"]}')
    commit = pc.check_gate(a.resoluciones, a.out_dir, smoke=bool(a.limit))

    import torch
    import config
    from data import get_transforms
    from train_model import configure_determinism
    from src.fase4 import atribucion_local as al
    from src.fase4 import caracteristicas as car
    from src.fase4.convergencia import preprocesamiento
    from src.fase_x.pixeles_local import filas, imagenes

    det = configure_determinism(config.determinism(config.load_matrix()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = filas(car.load_index(), 'val', a.limit)
    y = np.array([int(r['y_true']) for r in rows])
    n = len(rows)
    ub = al.Ubicador(a.models_root, al.load_artefactos())
    tecnicas = a.only or res['tecnicas']['C1']
    k_q = dict(zip(res['q'], res['k_c1']))
    idx = pc.indices_bootstrap(n, res['semilla_rng'], res['b_bootstrap'])
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    meta = {'task': 'FX.1', 'parte': 'gpu', 'smoke': bool(a.limit), 'limit': a.limit, 'k_sorteos': k_sorteos,
            'git_commit': commit, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'determinism': det, 'entorno': pc.entorno(),
            'resoluciones': {'archivo': os.path.relpath(a.resoluciones, REPO).replace(os.sep, '/'),
                             'sha256': car.sha256_file(a.resoluciones), **ac.meta_resoluciones(raw)},
            'plan': {'archivo': pc.PLAN, 'sha256': car.sha256_file(os.path.join(REPO, pc.PLAN))},
            'imagenes_sha256_lista': {}, 'reproduccion_pdep': {}, 'segundos': {}}
    resultados, npz = {'C1': {}, 'C2': {}}, {}
    for tech in tecnicas:
        tt = time.time()
        imgs, meta['imagenes_sha256_lista'][tech] = imagenes(tech, rows, 'val')
        prep = preprocesamiento(tech)
        tf = get_transforms(False, img_size=prep['size'], hflip=False, interpolation=prep['interpolation'],
                            antialias=prep['antialias'])
        reales, info_r = cargar_modelos(ub, tech, False, device)
        perms, info_p = cargar_modelos(ub, tech, True, device) if 'C2' in a.partes else ({}, {})
        meta['segundos'][f'{tech}_carga'] = round(time.time() - tt, 1)

        # Regla 2.5: reproducción de p_dep de val antes de modificar ninguna imagen
        p0 = p_modelos(reales, list(imgs), tf, device, a.batch_size)
        rep = {}
        for s in pc.SEEDS:
            rep[f'real_s{s}'] = float(np.abs(p0[s] - np.asarray(info_r[str(s)]['val_probs'][:n])).max())
        p0_perm = p_modelos(perms, list(imgs), tf, device, a.batch_size) if perms else {}
        for s in p0_perm:
            rep[f'perm_s{s}'] = float(np.abs(p0_perm[s] - np.asarray(info_p[str(s)]['val_probs'][:n])).max())
        meta['reproduccion_pdep'][tech] = rep
        if max(rep.values()) > res['tol_pdep']:
            raise Detener(f'{tech}: p_dep no reproduce val_probs de F3.5/F3.6: {rep}')
        p_orig = np.mean([p0[s] for s in pc.SEEDS], axis=0)

        pend = [pc.mapas_pendiente(im, res['sigma_pendiente'], res['sobel_borde']) for im in imgs]

        if 'C1' in a.partes:
            tc = time.time()
            rng = np.random.default_rng(res['semilla_rng'])           # nuevo por técnica
            d4 = [pc.desenfocar(im, res['sigma_c1_desenfoque'], res['desenfoque_borde']) for im in imgs]
            p_cond = {(m, q): np.full((n, 2 + 2 * k_sorteos, len(pc.SEEDS)), np.nan) for m in MAPAS for q in k_q}
            desp_reg = {(m, q): np.zeros((n, k_sorteos, 2), dtype=np.int16) for m in MAPAS for q in k_q}
            for m, q, i, desp, azar in pc.sorteos_c1(rng, MAPAS, list(k_q), n, k_sorteos, k_q, res):
                ims = pc.imagenes_c1(imgs[i], pend[i][m], q, k_q[q], desp, azar, d4[i])
                ps = p_modelos(reales, ims, tf, device, a.batch_size)
                p_cond[(m, q)][i] = np.stack([ps[s] for s in pc.SEEDS], axis=1)
                desp_reg[(m, q)][i] = np.array(desp, dtype=np.int16)
            meta['segundos'][f'{tech}_C1'] = round(time.time() - tc, 1)
            r_t, pvals = {}, {}
            for (m, q), arr in p_cond.items():
                pm = arr.mean(2)                                       # media de las 5 semillas
                alta, baja = pm[:, 0], pm[:, 1]
                desp, azar = pm[:, 2:2 + k_sorteos], pm[:, 2 + k_sorteos:]
                s_i = pc.estadistico_s(p_orig, alta, desp)
                f0 = f1(y, p_orig)
                df_desp = np.array([f1(y, desp[:, k]) - f0 for k in range(k_sorteos)])
                df_azar = np.array([f1(y, azar[:, k]) - f0 for k in range(k_sorteos)])
                df_alta = f1(y, alta) - f0
                r_t[f'{m}|q{q}'] = {
                    'S': pc.ic(s_i, idx, res['nivel_ic'], res['cuantiles_ic']),
                    'wilcoxon': pc.wilcoxon(s_i, res['wilcoxon_metodo']),
                    'delta_secundarios': {
                        'azar': pc.ic(np.abs(alta - p_orig) - np.abs(azar - p_orig[:, None]).mean(1), idx,
                                      res['nivel_ic'], res['cuantiles_ic']),
                        'baja': pc.ic(np.abs(alta - p_orig) - np.abs(baja - p_orig), idx, res['nivel_ic'],
                                      res['cuantiles_ic'])},
                    'delta_f1': {'original': f0, 'alta': df_alta, 'baja': f1(y, baja) - f0,
                                 'desplazada': {'mediana': float(np.median(df_desp)), 'min': float(df_desp.min()),
                                                'max': float(df_desp.max()),
                                                'n_sorteos_con_caida_mayor_que_alta': int((df_desp < df_alta).sum())},
                                 'azar': {'mediana': float(np.median(df_azar)), 'min': float(df_azar.min()),
                                          'max': float(df_azar.max()),
                                          'n_sorteos_con_caida_mayor_que_alta': int((df_azar < df_alta).sum())}}}
                pvals[f'{tech}|{m}|q{q}'] = r_t[f'{m}|q{q}']['wilcoxon']['p']
                npz[f'C1|{tech}|{m}|q{q}|p_por_semilla'] = arr.astype(np.float64)
                npz[f'C1|{tech}|{m}|q{q}|desplazamientos'] = desp_reg[(m, q)]
                npz[f'C1|{tech}|{m}|q{q}|S'] = s_i
            npz[f'C1|{tech}|p_original'] = np.stack([p0[s] for s in pc.SEEDS], axis=1)
            resultados['C1'][tech] = {'por_condicion': r_t, '_p': pvals}

        if 'C2' in a.partes:
            tc = time.time()
            g, px = res['rejilla_c2'], res['parche_c2_px']
            o_real = {s: np.zeros((n, g, g)) for s in pc.SEEDS}
            o_perm = {s: np.zeros((n, g, g)) for s in pc.SEEDS}
            for i in range(n):
                d8 = pc.desenfocar(imgs[i], res['sigma_c2_desenfoque'], res['desenfoque_borde'])
                ims = pc.imagenes_c2(imgs[i], d8, g, px)
                pr = p_modelos(reales, ims, tf, device, a.batch_size)
                pp = p_modelos(perms, ims, tf, device, a.batch_size)
                for s in pc.SEEDS:
                    o_real[s][i] = pc.mapa_oclusion(pr[s][None], [p0[s][i]], g)[0]
                    o_perm[s][i] = pc.mapa_oclusion(pp[s][None], [p0_perm[s][i]], g)[0]
            meta['segundos'][f'{tech}_C2'] = round(time.time() - tc, 1)
            cord = pc.cordura_c2(o_real, o_perm, res['c2_criterio_f43'])
            o_tec = np.mean([o_real[s] for s in pc.SEEDS], axis=0)
            pend8 = {m: pc.media_por_parche(np.stack([p[m] for p in pend]), g) for m in MAPAS}
            corr = {m: pc.ic(pc.spearman_por_muestra(o_tec, pend8[m]), idx, res['nivel_ic'], res['cuantiles_ic'])
                    for m in MAPAS}
            etiqueta = 'interpretable' if cord['global'] == 'cambia' else 'sin validación por datos'
            resultados['C2'][tech] = {'cordura': cord, 'corr_pendiente': corr, 'lectura': etiqueta,
                                      'n_o_tecnica_constante': int(ac.es_constante(o_tec).sum())}
            for s in pc.SEEDS:
                npz[f'C2|{tech}|real|s{s}'] = o_real[s].astype(np.float32)
                npz[f'C2|{tech}|perm|s{s}'] = o_perm[s].astype(np.float32)
        meta['segundos'][f'{tech}_total'] = round(time.time() - tt, 1)
        print(f'{tech}: {meta["segundos"]}', flush=True)
        reales = perms = None
        if device == 'cuda':
            torch.cuda.empty_cache()

    if 'C1' in a.partes:
        p = {k: v for t in resultados['C1'] for k, v in resultados['C1'][t].pop('_p').items()}
        if not a.only and not a.limit:
            resultados['C1']['H-C'] = pc.holm(p, res['familias']['H-C'], res['alpha'])
        else:
            resultados['C1']['H-C'] = {'_nota': 'familia incompleta (--only o --limit): no se aplica Holm', 'p': p}
    meta['segundos_total'] = round(time.time() - t0, 1)
    if a.limit:
        calc = sum(v for k, v in meta['segundos'].items() if k.endswith(('_C1', '_C2')))
        carga = sum(v for k, v in meta['segundos'].items() if k.endswith('_carga'))
        est = calc * pc.N_VAL / n + carga
        meta['estimacion_114_s'] = round(est, 1)
        meta['supera_limite_plan'] = est > res['limite_gpu_horas'] * 3600
        print(f'estimación para 114 imágenes: {est / 60:.1f} min (límite del plan: {res["limite_gpu_horas"]} h) → '
              f'{"SUPERA: detente y avisa a SV" if meta["supera_limite_plan"] else "dentro del límite"}', flush=True)
    with open(os.path.join(a.out_dir, 'resultados.json'), 'w', encoding='utf-8') as f:
        json.dump(pc.limpiar({'meta': meta, 'resultados': resultados}), f, indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(a.out_dir, 'arreglos.npz'), **npz)
    print(f'FX.1 GPU: {meta["segundos_total"]} s')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
