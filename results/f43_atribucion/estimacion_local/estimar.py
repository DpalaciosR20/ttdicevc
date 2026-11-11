"""F4.3, P20: prueba acotada de la parte local para estimar su duración.

Mide, con datos reales y sin escribir las salidas de atribucion_local.py:
  1. el recorrido de --root (find_all);
  2. la lectura y verificación de 1 .npz de Gray-Scott, 1 .npz de ViT y 1 .pt de ViT
     (Ubicador.ruta + cargar_npz + cabeza_de), de donde sale el rendimiento en MiB/s;
  3. el cálculo por técnica con esos mapas: Spearman y SSIM de 114 pares, la
     agregación de una semilla y la prueba de permutación con 250 réplicas.
Extrapola al total (30 .npz, 20 .pt; 15 comparaciones × 2 medidas × splits × mapas).

Uso: python results/f43_atribucion/estimacion_local/estimar.py --root <tt2-artefactos>
Escribe estimacion.json junto a este archivo.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.fase4 import atribucion_comun as ac  # noqa: E402
from src.fase4 import atribucion_local as al  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    a = ap.parse_args()
    res = ac.check_resoluciones(json.loads((REPO / 'results/f43_atribucion/resoluciones.json').read_text(encoding='utf-8')))
    art = al.load_artefactos()
    t = {}
    t0 = time.time()
    ub = al.Ubicador(a.root, art)
    t['find_all_s'] = time.time() - t0
    index = al.load_index()

    lect, datos = {}, {}
    for tech, s in (('gray_scott', 42), ('perlin', 42)):
        name = ac.run_name(tech, s, False)
        rec = json.loads((al.RECORDS / f'{name}.json').read_text(encoding='utf-8'))
        t0 = time.time()
        p, row = ub.ruta(al.logical_maps(tech, s, False))
        datos[tech] = al.cargar_npz(p, rec, index)
        lect[f'npz_{tech}'] = {'bytes': int(row['size_bytes']), 's': time.time() - t0}
    t0 = time.time()
    pm, rowm = ub.ruta(al.logical_model('perlin', 42, False))
    al.cabeza_de(pm)
    lect['pt_vit'] = {'bytes': int(rowm['size_bytes']), 's': time.time() - t0}

    calc = {}
    for tech, d in datos.items():
        kinds = ['map_norm'] + (['backbone_norm'] if 'backbone_norm' in d['val'] else [])
        A = d['val']['map_norm'].astype(np.float32)
        B = np.roll(A, 1, axis=0)
        t0 = time.time(); ac.spearman_filas(A, B); sp = time.time() - t0
        t0 = time.time(); ac.ssim_filas(A, B, g=ac.GRID if res['ssim_resolucion'] == '56' else None); ss = time.time() - t0
        real = {s: d for s in ac.SEEDS}
        t0 = time.time(); m, y = al.estadistico_val(real, res); agr = time.time() - t0
        rng = np.random.default_rng(ac.PERM_SEED)
        t0 = time.time(); ac.prueba_permutacion(m, y, rng, 250, res['perm_p']); perm = time.time() - t0
        n_comp = 15 * len(al.SPLITS) * len(kinds)          # 10 pares + 5 real–permutado
        calc[tech] = {'spearman_114_s': sp, 'ssim_114_s': ss, 'estadistico_val_s': agr,
                      'permutacion_250_s': perm, 'n_comparaciones': n_comp,
                      'estimado_s': n_comp * (sp + ss) + perm * ac.PERM_B / 250 + 4 * agr}

    tot_npz = sum(int(r['size_bytes']) for k, r in art.items() if k.startswith('MAPS[') and r['task'] == 'F3.7')
    tot_pt = sum(int(art[al.logical_model(tc, s, p)]['size_bytes']) for tc in ac.VIT for s in ac.SEEDS for p in (False, True))
    mibs_npz = sum(v['bytes'] for k, v in lect.items() if k.startswith('npz')) / sum(v['s'] for k, v in lect.items() if k.startswith('npz')) / 2**20
    mibs_pt = lect['pt_vit']['bytes'] / lect['pt_vit']['s'] / 2**20
    lectura_s = tot_npz / 2**20 / mibs_npz + tot_pt / 2**20 / mibs_pt
    calculo_s = calc['gray_scott']['estimado_s'] + 2 * calc['perlin']['estimado_s']
    out = {'tarea': 'F4.3', 'parte': 'estimacion_local_P20', 'tiempos': t, 'lecturas': lect,
           'mib_s': {'npz': mibs_npz, 'pt': mibs_pt}, 'total_bytes': {'npz': tot_npz, 'pt': tot_pt},
           'calculo': calc, 'estimado_s': {'find_all': t['find_all_s'], 'lectura': lectura_s,
                                           'calculo': calculo_s, 'total': t['find_all_s'] + lectura_s + calculo_s},
           'nota': 'la lectura medida puede venir de la caché de Drive; un archivo frío puede ser más lento'}
    out['estimado_min_total'] = out['estimado_s']['total'] / 60
    (Path(__file__).parent / 'estimacion.json').write_text(json.dumps(ac.limpiar(out), indent=1) + '\n', encoding='utf-8')
    print(json.dumps(ac.limpiar(out), indent=1))


if __name__ == '__main__':
    main()
