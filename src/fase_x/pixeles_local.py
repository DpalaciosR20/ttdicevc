"""
pixeles_local.py
────────────────
FX.1 — partes sin inferencia (docs/plan_exploratorio_pixeles.md):
  B  traslape entre atribución (mapas de F3.7, val) y pendiente por parche
     (Gray-Scott 7 × 7, Perlin 14 × 14), con la nula espacial diédrica y el contraste
     con los mapas de F3.6; B4 (familia H-B).
  D  diferencias de pendiente entre clases por píxel en train: prueba por clústeres
     en Gray-Scott y Lorenz; Mann–Whitney de la media por imagen en Perlin (familia H-D).

Insumos: imágenes (TT_DATA_ROOT, verificadas contra meta/data_manifest.csv) y los .npz
de F3.7 bajo --root (verificados contra meta/artefactos_manifest.csv y sus registros).
Prueba no se lee: las filas de sample_index y los arreglos de los .npz se filtran por
split con un aborto explícito ante 'test' (regla 2.1).

Uso:
  python -m src.fase_x.pixeles_local --root <tt2-mapas o /kaggle/input> \
      --resoluciones results/fx_pixeles/resoluciones.json --out-dir results/fx_pixeles/local
  --limit N: prueba acotada (N muestras de val y N de train), fuera de results/fx_pixeles/.
"""
import argparse
import csv
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


# ── Carga ─────────────────────────────────────────────────────────────────
def filas(index_rows, split, limit=None):
    rows = sorted((r for r in index_rows if r['split'] == split), key=lambda r: r['sample_id'])
    pc.exigir_split(rows, split)
    return rows[:limit] if limit else rows


def imagenes(tech, rows, split):
    """uint8 (N, 256, 256) verificadas contra data_manifest; solo el split pedido."""
    from src import paths
    from src.fase4 import caracteristicas as car
    from src.fase4.atribucion_gpu import decodificar
    pc.exigir_split(rows, split)
    names = [r[f'img_{tech}'] for r in rows]
    for n in names:
        if f'_{split}_' not in n or '_test_' in n:
            raise Detener(f'{n} no es una imagen de {split}')
    expected, _ = car.image_rows(car.load_data_manifest(), tech)
    datos, hashes = car.read_verified_images(str(paths.images_dir(tech)), paths.REL['IMAGES'][tech], names, expected)
    return np.stack([decodificar(b) for b in datos]), pc.huella_lista(hashes)


def mapas_val(ub, tech, perm, rows):
    """Mapa crudo de F3.7 en val, dividido por la media de la semilla y promediado entre las
    5 semillas (como F4.3 y F4.4): (N, 224, 224). Solo se materializan las filas de val."""
    from src.fase4 import atribucion_local as al
    ids = [r['sample_id'] for r in rows]
    y = [int(r['y_true']) for r in rows]
    crudo = {}
    for s in pc.SEEDS:
        name = ac.run_name(tech, s, perm)
        rec = json.loads((al.RECORDS / f'{name}.json').read_text(encoding='utf-8'))
        p, row = ub.ruta(al.logical_maps(tech, s, perm))
        if row['sha256'] != rec['npz']['sha256']:
            raise Detener(f'{name}: sha256 del manifiesto distinto del registro de F3.7')
        z = np.load(p)
        m = pc.exigir_arreglo_split(z['split'], 'val')
        sid = z['sample_id'].astype(str)[m]
        if list(sid[:len(ids)]) != ids or list(z['y_true'][m][:len(ids)].astype(int)) != y:
            raise Detener(f'{name}: sample_id o y_true de val distintos de sample_index')
        crudo[s] = ac.mapa_crudo({k: z[k][m][:len(ids)] for k in ('map_raw', 'map_raw_scale')})
    esc, factores = ac.escalar_semillas(crudo, 'dividir_media_semilla')
    return ac.media_entre_semillas(esc), factores


# ── B (puro: recibe arreglos) ─────────────────────────────────────────────
def analizar_b(tech, real, perm, pend, pend_s2, y, res, idx):
    """real, perm: (N, 224, 224) mapas promediados; pend, pend_s2: {'grad','log'}: (N, 224, 224).
    Devuelve el resumen y las filas por muestra."""
    g = res['rejilla_b'][tech]
    k_q = dict(zip(res['q'], res['k_b'][tech]))
    perms = pc.transformaciones_rejilla(g)
    if len(perms) != res['transformaciones_nula_b2'][tech]:
        raise Detener(f'{tech}: {len(perms)} transformaciones; el plan fija {res["transformaciones_nula_b2"][tech]}')
    const = {'real': ac.es_constante(real), 'perm': ac.es_constante(perm)}
    attr = {k: pc.suma_por_parche(v, g) for k, v in (('real', real), ('perm', perm))}
    pp = {m: pc.media_por_parche(pend[m], g) for m in MAPAS}
    pp2 = {m: pc.media_por_parche(pend_s2[m], g) for m in MAPAS}
    n = len(y)
    b1, b2 = {}, {}
    for fuente in ('real', 'perm'):
        for m in MAPAS:
            rho = pc.spearman_por_muestra(attr[fuente], pp[m])
            b1[(fuente, m)] = np.where(const[fuente], np.nan, rho)
        b1[(fuente, 'grad_s2')] = np.where(const[fuente], np.nan, pc.spearman_por_muestra(attr[fuente], pp2['grad']))
        b1[(fuente, 'log_s2')] = np.where(const[fuente], np.nan, pc.spearman_por_muestra(attr[fuente], pp2['log']))
        for m in (MAPAS if res['b2_mapas'] == 'grad_y_log' else ('grad',)):
            for q, k in k_q.items():
                obs, nul = np.full(n, np.nan), np.full(n, np.nan)
                for i in range(n):
                    if const[fuente][i]:
                        continue
                    obs[i] = pc.dice_topk(attr[fuente][i], pp[m][i], k)
                    nul[i] = pc.dice_nulo(attr[fuente][i], pp[m][i], k, perms)
                b2[(fuente, m, q)] = {'dice': obs, 'dice_nulo': nul, 'delta': obs - nul}
    out = {'rejilla': g, 'n': n, 'n_mapa_constante': {f: int(c.sum()) for f, c in const.items()},
           'n_transformaciones_nula': int(len(perms)), 'B1': {}, 'B2': {}, 'B3': {}}
    for (fuente, m), rho in b1.items():
        out['B1'][f'{fuente}|{m}'] = {**pc.ic(rho, idx, res['nivel_ic'], res['cuantiles_ic']),
                                      'n_excluidas': int(np.isnan(rho).sum())}
    for (fuente, m, q), d in b2.items():
        out['B2'][f'{fuente}|{m}|q{q}'] = {'k': k_q[q], 'dice_media': float(np.nanmean(d['dice'])),
                                           'dice_nulo_media': float(np.nanmean(d['dice_nulo'])),
                                           'delta': pc.ic(d['delta'], idx, res['nivel_ic'], res['cuantiles_ic'])}
    for m in MAPAS:
        out['B3'][f'B1|{m}'] = pc.ic(b1[('real', m)] - b1[('perm', m)], idx, res['nivel_ic'], res['cuantiles_ic'])
    for (fuente, m, q), d in b2.items():
        if fuente == 'real':
            out['B3'][f'B2|{m}|q{q}'] = pc.ic(d['delta'] - b2[('perm', m, q)]['delta'], idx, res['nivel_ic'],
                                              res['cuantiles_ic'])
    y = np.asarray(y)
    b4 = {m: pc.mann_whitney(b1[('real', m)][y == 1], b1[('real', m)][y == 0]) for m in MAPAS}
    out['B4'] = b4
    filas_out = []
    for i in range(n):
        f = {'tecnica': tech, 'muestra': i, 'y_true': int(y[i]), 'real_constante': bool(const['real'][i]),
             'perm_constante': bool(const['perm'][i])}
        for (fuente, m), rho in b1.items():
            f[f'spearman_{fuente}_{m}'] = '' if np.isnan(rho[i]) else float(rho[i])
        for (fuente, m, q), d in b2.items():
            f[f'delta_dice_{fuente}_{m}_q{q}'] = '' if np.isnan(d['delta'][i]) else float(d['delta'][i])
        filas_out.append(f)
    npz = {f'{tech}|atrib|{k}': v.astype(np.float32) for k, v in attr.items()}
    npz.update({f'{tech}|pendiente|{m}': v.astype(np.float32) for m, v in pp.items()})
    return out, filas_out, npz


# ── D (puro) ──────────────────────────────────────────────────────────────
def analizar_d(tech, X, y, res, X_s2=None):
    """X: {'grad','log'}: (N, 256, 256) de train. Gray-Scott y Lorenz: prueba por clústeres;
    Perlin: Mann–Whitney de la media por imagen. Sensibilidad descriptiva según d_sensibilidad."""
    y = np.asarray(y).astype(int)
    out, npz = {}, {}
    for m in MAPAS:
        if tech == 'perlin':
            out[m] = pc.mann_whitney(X[m].mean((1, 2))[y == 1], X[m].mean((1, 2))[y == 0])
            out[m]['p_familia'] = out[m]['p']
            if X_s2 is not None:
                out[m]['sensibilidad_sigma2'] = pc.mann_whitney(X_s2[m].mean((1, 2))[y == 1], X_s2[m].mean((1, 2))[y == 0])
            continue
        r = pc.prueba_clusteres(X[m], y, res['umbral_t'], res['b_permutaciones_d'], res['semilla_rng'])
        out[m] = {'p_familia': r['p_mayor'], 'n_clusteres': len(r['clusteres']),
                  'clusteres_p_menor_005': [c for c in r['clusteres'] if c['p'] < 0.05],
                  'masa_mayor': r['clusteres'][0]['masa'] if r['clusteres'] else 0.0,
                  'nula_percentiles': {str(q): float(np.quantile(r['nula_masa_maxima'], q)) for q in (0.5, 0.95, 0.99)}}
        npz[f'{tech}|{m}|t'] = r['t'].astype(np.float32)
        npz[f'{tech}|{m}|nula'] = r['nula_masa_maxima'].astype(np.float32)
        sens = {}
        variantes = ([('sigma2_t258', X_s2, res['umbral_t']), ('sigma1_t329', X, res['umbral_t_sensibilidad'])]
                     if res['d_sensibilidad'] == 'dos_variantes_sigma2_y_t329'
                     else [('sigma2_t329', X_s2, res['umbral_t_sensibilidad'])])
        for nombre, XX, u in variantes:
            if XX is None:
                continue
            rs = pc.prueba_clusteres(XX[m], y, u, res['b_permutaciones_d'], res['semilla_rng'])
            sens[nombre] = {'p_mayor': rs['p_mayor'], 'n_clusteres': len(rs['clusteres']),
                            'clusteres_p_menor_005': [c for c in rs['clusteres'] if c['p'] < 0.05]}
            npz[f'{tech}|{m}|t|{nombre}'] = rs['t'].astype(np.float32)
        out[m]['sensibilidad'] = sens
    return out, npz


# ── Principal ─────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='raíz con los .npz de F3.7 (copia de Drive o /kaggle/input)')
    ap.add_argument('--resoluciones', default=os.path.join(REPO, pc.RESOLUCIONES_PATH))
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'results', 'fx_pixeles', 'local'))
    ap.add_argument('--partes', nargs='+', default=['B', 'D'], choices=['B', 'D'])
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args(argv)

    raw = json.loads(open(a.resoluciones, encoding='utf-8').read())
    res = pc.check_resoluciones(raw)
    commit = pc.check_gate(a.resoluciones, a.out_dir, smoke=bool(a.limit))
    from src.fase4 import atribucion_local as al
    from src.fase4 import caracteristicas as car
    from src.fase4.convergencia import preprocesamiento, redimensionar_224

    index = car.load_index()
    t0, seg = time.time(), {}
    meta = {'task': 'FX.1', 'parte': 'local', 'smoke': bool(a.limit), 'limit': a.limit, 'git_commit': commit,
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'resoluciones': {'archivo': os.path.relpath(a.resoluciones, REPO).replace(os.sep, '/'),
                             'sha256': car.sha256_file(a.resoluciones), **ac.meta_resoluciones(raw)},
            'plan': {'archivo': pc.PLAN, 'sha256': car.sha256_file(os.path.join(REPO, pc.PLAN))},
            'entorno': pc.entorno(), 'imagenes_sha256_lista': {}, 'artefactos_verificados': {}}
    os.makedirs(a.out_dir, exist_ok=True)
    resultados, npz, filas_b = {}, {}, []

    if 'B' in a.partes:
        ub = al.Ubicador(a.root, al.load_artefactos())
        rows = filas(index, 'val', a.limit)
        y = [int(r['y_true']) for r in rows]
        idx = pc.indices_bootstrap(len(rows), res['semilla_rng'], res['b_bootstrap'])
        resultados['B'] = {}
        for tech in res['tecnicas']['B']:
            tt = time.time()
            imgs, meta['imagenes_sha256_lista'][f'{tech}|val'] = imagenes(tech, rows, 'val')
            prep = preprocesamiento(tech)
            i224 = [redimensionar_224(im, prep) for im in imgs]
            pend = {m: [] for m in MAPAS}
            pend2 = {m: [] for m in MAPAS}
            for im in i224:
                for d, s in ((pend, res['sigma_pendiente']), (pend2, res['sigma_pendiente_sensibilidad'])):
                    mm = pc.mapas_pendiente(im, s, res['sobel_borde'])
                    for m in MAPAS:
                        d[m].append(mm[m])
            pend = {m: np.stack(v) for m, v in pend.items()}
            pend2 = {m: np.stack(v) for m, v in pend2.items()}
            real, f_real = mapas_val(ub, tech, False, rows)
            perm, f_perm = mapas_val(ub, tech, True, rows)
            r, fb, nz = analizar_b(tech, real, perm, pend, pend2, y, res, idx)
            r['factores_escala_semilla'] = {'real': f_real, 'perm': f_perm}
            resultados['B'][tech] = r
            filas_b += fb
            npz.update(nz)
            seg[f'B_{tech}'] = round(time.time() - tt, 1)
            print(f'B {tech}: {seg[f"B_{tech}"]} s', flush=True)
        p = {f'{t}|{m}': resultados['B'][t]['B4'][m]['p'] for t in res['tecnicas']['B'] for m in MAPAS}
        resultados['B']['H-B'] = pc.holm(p, res['familias']['H-B'], res['alpha'])
        meta['artefactos_verificados'] = ub.verificados

    if 'D' in a.partes:
        rows = filas(index, 'train', a.limit)
        y = [int(r['y_true']) for r in rows]
        resultados['D'] = {}
        for tech in res['tecnicas']['D']:
            tt = time.time()
            imgs, meta['imagenes_sha256_lista'][f'{tech}|train'] = imagenes(tech, rows, 'train')
            X = {m: [] for m in MAPAS}
            X2 = {m: [] for m in MAPAS}
            for im in imgs:
                for d, s in ((X, res['sigma_pendiente']), (X2, res['sigma_pendiente_sensibilidad'])):
                    mm = pc.mapas_pendiente(im, s, res['sobel_borde'])
                    for m in MAPAS:
                        d[m].append(mm[m])
            X = {m: np.stack(v) for m, v in X.items()}
            X2 = {m: np.stack(v) for m, v in X2.items()}
            r, nz = analizar_d(tech, X, y, res, X2)
            resultados['D'][tech] = r
            npz.update(nz)
            seg[f'D_{tech}'] = round(time.time() - tt, 1)
            print(f'D {tech}: {seg[f"D_{tech}"]} s', flush=True)
        p = {f'{t}|{m}': resultados['D'][t][m]['p_familia'] for t in res['tecnicas']['D'] for m in MAPAS}
        resultados['D']['H-D'] = pc.holm(p, res['familias']['H-D'], res['alpha'])

    meta['segundos'] = seg
    meta['segundos_total'] = round(time.time() - t0, 1)
    if a.limit:
        meta['estimacion_s'] = {k: round(v * ((pc.N_VAL if k.startswith('B') else pc.N_TRAIN) / a.limit), 1)
                                for k, v in seg.items()}
        meta['estimacion_total_s'] = round(sum(meta['estimacion_s'].values()), 1)
        print('estimación (s, escalado lineal):', meta['estimacion_s'], '| total', meta['estimacion_total_s'], flush=True)
    with open(os.path.join(a.out_dir, 'resultados.json'), 'w', encoding='utf-8') as f:
        json.dump(pc.limpiar({'meta': meta, 'resultados': resultados}), f, indent=1, ensure_ascii=False)
    if filas_b:
        with open(os.path.join(a.out_dir, 'b_por_muestra.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(filas_b[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(filas_b)
    np.savez_compressed(os.path.join(a.out_dir, 'arreglos.npz'), **npz)
    print(f'FX.1 local: {meta["segundos_total"]} s')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
