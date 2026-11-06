"""
summarize_maps.py
─────────────────
Resumen descriptivo de los mapas de F3.7 que cabe en git (los .npz, ≈ 1.3 GiB,
viven en el dataset `tt2-mapas`). No es el análisis de F4.3: solo comprueba
que los mapas no son degenerados y deja los insumos agregados para las figuras.

Por cada corrida de results/f37_mapas/records/ busca su .npz bajo --root
(por nombre; si hay copias, la que termina en la ruta del registro, como
verify_artefacts) y comprueba su SHA-256. Salidas en --out-dir:

  per_run.csv     por corrida y tipo de mapa (map, backbone_map):
                    frac_constantes   mapas sin variación (normalizados a 0)
                    entropia_media    entropía de Shannon del mapa normalizado a
                                      suma 1, en bits, sobre la rejilla 56 × 56
                                      (máximo log2(3136) = 11.61: mapa uniforme)
                    masa_top10        fracción de la masa en el 10 % de píxeles
                                      más altos (0.10 = uniforme)
                    corr_con_real     solo en F3.6: correlación de Pearson media,
                                      por muestra, con el mapa del modelo real de
                                      la misma técnica y semilla (F3.5)
                    corr_con_backbone solo en ViT: correlación media, por muestra,
                                      entre map y backbone_map
  class_means.npz claves '<corrida>|<mapa>|<split>|y<0|1>': mapa medio
                  normalizado (float32, 56 × 56), por split y clase real.

Uso (Kaggle, CPU, con el dataset tt2-mapas como input):
  python -m src.interpretability.summarize_maps --root /kaggle/input --out-dir results/f37_mapas/resumen
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRID = 56


def pool(maps, g=GRID):
    """(N, 224, 224) -> (N, g, g) por promedio en bloques."""
    n, h, w = maps.shape
    k = h // g
    return maps.reshape(n, g, k, g, k).mean((2, 4))


def concentration(m):
    """m: (N, g, g) >= 0. Devuelve (constante, entropía en bits, masa del top 10 %) por muestra."""
    flat = m.reshape(len(m), -1).astype(np.float64)
    tot = flat.sum(1)
    const = flat.max(1) == flat.min(1)
    p = np.where(tot[:, None] > 0, flat / np.where(tot > 0, tot, 1)[:, None], 1 / flat.shape[1])
    ent = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1)
    k = int(round(0.1 * flat.shape[1]))
    top = np.sort(p, 1)[:, -k:].sum(1)
    return const, ent, top


def rowcorr(a, b):
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)


def load(path):
    z = np.load(path)
    out = {k: z[k] for k in ('sample_id', 'split', 'y_true')}
    for k in ('map', 'backbone_map'):
        if k in z.files:
            out[k] = pool(z[k].astype(np.float32))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--records', default=os.path.join(REPO, 'results', 'f37_mapas', 'records'))
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'results', 'f37_mapas', 'resumen'))
    a = ap.parse_args(argv)
    from src.verify_artefacts import find_all, locate, sha256

    recs = [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob(os.path.join(a.records, '*.json')))]
    allf = find_all(a.root, ('.npz',))
    paths = {}
    for r in recs:
        p, _, amb = locate(allf.get(os.path.basename(r['npz']['path']), []), r['npz']['path'])
        if p is None:
            raise SystemExit(f"{r['run']}: npz {'ambiguo' if amb else 'ausente'} bajo {a.root}")
        if sha256(p) != r['npz']['sha256']:
            raise SystemExit(f"{r['run']}: SHA-256 del npz distinto del registro")
        paths[r['run']] = p
    print(f'{len(recs)} npz con huella correcta', flush=True)

    data = {r['run']: load(paths[r['run']]) for r in recs}
    by_key = {(r['technique'], r['seed'], r['fase']): r['run'] for r in recs}
    rows, means = [], {}
    for r in recs:
        d = data[r['run']]
        for mk in ('map', 'backbone_map'):
            if mk not in d:
                continue
            m = d[mk]
            const, ent, top = concentration(m)
            row = {'run': r['run'], 'fase': r['fase'], 'technique': r['technique'], 'arch': r['arch'],
                   'seed': r['seed'], 'mapa': mk, 'metodo': r['methods'][mk], 'n': len(m),
                   'frac_constantes': float(const.mean()), 'entropia_media': float(ent.mean()),
                   'masa_top10': float(top.mean()), 'corr_con_real': '', 'corr_con_backbone': ''}
            if r['fase'] == 'control_permutacion':
                real = data[by_key[(r['technique'], r['seed'], 'final')]]
                assert list(real['sample_id']) == list(d['sample_id'])
                row['corr_con_real'] = float(np.nanmean(rowcorr(m, real[mk])))
            if mk == 'map' and 'backbone_map' in d:
                row['corr_con_backbone'] = float(np.nanmean(rowcorr(m, d['backbone_map'])))
            rows.append(row)
            for split in ('val', 'test'):
                for y in (0, 1):
                    sel = (d['split'] == split) & (d['y_true'] == y)
                    means[f"{r['run']}|{mk}|{split}|y{y}"] = m[sel].mean(0).astype(np.float32)
            print(f"{r['run']:<62} {mk:<13} const={row['frac_constantes']:.3f} H={row['entropia_media']:.2f} "
                  f"top10={row['masa_top10']:.3f} r_real={row['corr_con_real'] if row['corr_con_real'] == '' else round(row['corr_con_real'], 3)}",
                  flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, 'per_run.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(a.out_dir, 'class_means.npz'), **means)
    print(f'{len(rows)} filas -> {a.out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
