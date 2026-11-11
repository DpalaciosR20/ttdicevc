"""
atribucion_local.py
───────────────────
F4.3 — Atribución espacial y espectral (docs/plan_fase4.md, F4.3), parte local
(CPU). La cascada y la ablación, que requieren inferencia, están en
atribucion_gpu.py (Kaggle, solo val).

Insumos (copia de Drive MiUnidad/tt2-artefactos/, --root):
  - Mapas de F3.7: los 30 .npz de ttdicevc/results/f37_mapas/maps/ (nunca
    f37_humo/). sha256 y bytes contra meta/artefactos_manifest.csv (MAPS[...])
    y contra results/f37_mapas/records/<corrida>.json.
  - Modelos ViT de F3.5 y F3.6 (.pt), solo para los cosenos de las cabezas.
    sha256 contra el manifiesto y contra model_file.sha256 del JSON de la corrida.
  - u de umap_params.pkl (TT_DATA_ROOT; verificado contra el manifiesto), solo
    para los terciles de u4 de Lorenz (cortes en train).

Análisis (val principal, prueba réplica con los mapas archivados, G1):
  1. Agregación por clase real (principal) y predicha (secundaria) sobre
     map_raw · map_raw_scale, sin normalizar. Media entre semillas por muestra.
  2. Fracción de mapas constantes por clase real y predicha; sensibilidad con
     los mapas normalizados sin constantes (no se reporta con < 10 no constantes).
  3. Gray-Scott: perfil radial, 28 anillos de 4 px. Lorenz: mapas por clase en
     cada tercil de u4.
  4. Prueba por técnica (solo val): permutación de etiquetas (5,000,
     default_rng(2026)) sobre ‖media(y=1) − media(y=0)‖₂; Holm m = 3.
  5. Cordura por aleatorización de datos: similitud por muestra F3.5 frente a
     F3.6 de la misma semilla (Spearman 56 × 56 principal, SSIM secundaria)
     frente al rango de similitudes entre semillas reales (10 pares).
  6. ViT: coseno de la cabeza real y la permutada de la misma semilla (w1 y
     w1 − w0) frente a los 10 pares de cabezas reales.
  7. Gray-Scott: por semilla, correlación de mapas real–permutado frente a la
     correlación de p_dep.
  8. Lectura de los ViT, si se da --cascada (salida de atribucion_gpu.py).
No hay inferencia: p_dep sale de los .npz (las predicciones archivadas).

Compuerta: enmienda-5 en la historia de HEAD, docs/plan_fase4.md sin cambios,
results/f5_contraste/resultados.json versionado (G1) y árbol limpio.
No sobrescribe: si <out>/resultados.json existe, se detiene.

Uso:
  python -m src.fase4.atribucion_local --root <tt2-artefactos> --resoluciones <json> \
      [--cascada results/f43_atribucion/gpu/cascada.json] [--out results/f43_atribucion/local]
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / 'src' / 'training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.fase4 import atribucion_comun as ac  # noqa: E402
from src.fase4.atribucion_comun import Detener  # noqa: E402

ARTEFACTOS = REPO / 'meta' / 'artefactos_manifest.csv'
RECORDS = REPO / 'results' / 'f37_mapas' / 'records'
OUT = REPO / 'results' / 'f43_atribucion' / 'local'
SPLITS = ('val', 'test')
N_SPLIT = 114
PROHIBIDO = 'f37_humo'


# ── Artefactos ────────────────────────────────────────────────────────────
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def load_artefactos(path=ARTEFACTOS):
    with open(path, newline='', encoding='utf-8') as f:
        return {r['logical_name']: r for r in csv.DictReader(f)}


def logical_maps(tech, seed, perm):
    return f'MAPS[{ac.run_name(tech, seed, perm)}]'


def logical_model(tech, seed, perm):
    arch, strat = ac.FINALIST[tech]
    return f'MODEL[{tech}:{arch}:{strat}:{"perm99:" if perm else ""}seed{seed}]'


class Ubicador:
    """Busca bajo root los archivos del manifiesto (por nombre; con copias, la que
    termina en rel_path, como verify_artefacts) y verifica bytes y sha256."""

    def __init__(self, root, artefactos):
        from src.verify_artefacts import find_all
        self.root, self.art = str(root), artefactos
        self.found = find_all(self.root, ('.npz', '.pt'))
        self.verificados = {}

    def ruta(self, logical):
        from src.verify_artefacts import locate
        row = self.art.get(logical)
        if row is None:
            raise Detener(f'{logical} no está en {ARTEFACTOS.name}')
        name = os.path.basename(row['rel_path'])
        p, _, amb = locate(self.found.get(name, []), row['rel_path'])
        if p is None:
            raise Detener(f'{"ambiguo" if amb else "falta"}: {name} bajo {self.root}')
        if PROHIBIDO in p.replace(os.sep, '/'):
            raise Detener(f'{p}: copia de la prueba de humo de F3.7; se exige ttdicevc/results/f37_mapas/maps/')
        size, digest = os.path.getsize(p), sha256_file(p)
        if size != int(row['size_bytes']) or digest != row['sha256']:
            raise Detener(f'{p}: bytes o sha256 distintos de {ARTEFACTOS.name}')
        self.verificados[logical] = {'archivo': os.path.relpath(p, self.root).replace(os.sep, '/'),
                                     'sha256': digest, 'bytes': size}
        return p, row


def load_index():
    with open(REPO / 'meta' / 'sample_index.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {s: sorted((r for r in rows if r['split'] == s), key=lambda r: r['sample_id']) for s in ('train', 'val', 'test')}


def cargar_npz(path, record, index):
    """Arreglos por split. Comprueba el sha256 del registro de F3.7, el orden de
    sample_id y las etiquetas de sample_index."""
    if record['npz']['sha256'] != sha256_file(path):
        raise Detener(f'{path}: sha256 distinto de results/f37_mapas/records/{record["run"]}.json')
    z = np.load(path)
    sid, split, y = z['sample_id'].astype(str), z['split'].astype(str), z['y_true'].astype(int)
    out = {}
    for s in SPLITS:
        want = [r['sample_id'] for r in index[s]]
        m = split == s
        if list(sid[m]) != want:
            raise Detener(f'{path}: sample_id de {s} no está en el orden de sample_index')
        if list(y[m]) != [int(r['y_true']) for r in index[s]]:
            raise Detener(f'{path}: y_true de {s} distinto de sample_index')
        d = {'y': y[m], 'p_dep': z['p_dep'][m].astype(np.float64), 'sample_id': want,
             'map_norm': z['map'][m], 'map_crudo': ac.mapa_crudo({k: z[k][m] for k in ('map_raw', 'map_raw_scale')})}
        if 'backbone_map' in z.files:
            d['backbone_norm'] = z['backbone_map'][m]
            d['backbone_crudo'] = ac.mapa_crudo({k: z[k][m] for k in ('backbone_map_raw', 'backbone_map_raw_scale')},
                                                'backbone_map')
        out[s] = d
    return out


def cabeza_de(path):
    import torch
    st = torch.load(path, map_location='cpu', weights_only=True)
    if 'heads.head.weight' not in st:
        raise Detener(f'{path}: no tiene heads.head.weight (se espera un ViT-B/16)')
    return st['heads.head.weight'].double().numpy()


def run_json(tech, seed, perm):
    fuente = 'f36_permutacion' if perm else 'f35_final'
    p = REPO / 'results' / fuente / tech / 'runs' / f'{ac.run_name(tech, seed, perm)}.json'
    return json.loads(p.read_text(encoding='utf-8'))


# ── Análisis por técnica (puro: recibe arreglos) ─────────────────────────
def _pool56(m):
    return None if m is None else ac.pool(m[None])[0]


def agregacion(real, res, tech, u4=None, cortes=None):
    """Agregación, constantes, sensibilidad, perfiles y terciles. real: {semilla: {split: d}}."""
    seeds = list(real)
    crudo_val = {s: real[s]['val']['map_crudo'] for s in seeds}
    out, medias = {}, {}
    for split in SPLITS:
        y = real[seeds[0]][split]['y']
        if any(list(real[s][split]['y']) != list(y) for s in seeds):
            raise Detener(f'{tech} {split}: y_true distinto entre semillas')
        esc, factores = ac.escalar_semillas({s: real[s][split]['map_crudo'] for s in seeds},
                                            res['escala_entre_semillas'], crudo_val)
        pred = ac.clase_predicha({s: real[s][split]['p_dep'] for s in seeds}, res['clase_predicha'])
        por_muestra = ac.media_entre_semillas(esc)
        m_real, n_real = ac.medias_por_clase(por_muestra, y)
        m_pred, n_pred = ac.medias_agrupadas(esc, pred)
        const = {s: ac.es_constante(real[s][split]['map_crudo']) for s in seeds}
        reales = {s: y for s in seeds}
        sens_real = ac.sensibilidad_sin_constantes({s: real[s][split]['map_norm'].astype(np.float32) for s in seeds},
                                                   const, reales, res['umbral_constantes_unidad'])
        sens_pred = ac.sensibilidad_sin_constantes({s: real[s][split]['map_norm'].astype(np.float32) for s in seeds},
                                                   const, pred, res['umbral_constantes_unidad'])
        d = {'n_por_clase_real': {str(c): n_real[c] for c in (0, 1)},
             'n_por_clase_predicha_agrupado': {str(c): n_pred[c] for c in (0, 1)},
             'factores_escala': {str(s): f for s, f in factores.items()},
             'norma_l2_diferencia_real': float(np.linalg.norm((m_real[1] - m_real[0]).ravel())),
             'constantes_clase_real': ac.fraccion_constantes(const, reales),
             'constantes_clase_predicha': ac.fraccion_constantes(const, pred),
             'sensibilidad_sin_constantes': {
                 'clase_real': {k: v for k, v in sens_real.items() if k != 'medias'} |
                               {'reportada': {str(c): sens_real['medias'][c] is not None for c in (0, 1)}},
                 'clase_predicha': {k: v for k, v in sens_pred.items() if k != 'medias'} |
                                   {'reportada': {str(c): sens_pred['medias'][c] is not None for c in (0, 1)}}}}
        for c in (0, 1):
            medias[f'{tech}|map|{split}|real|y{c}'] = _pool56(m_real[c])
            medias[f'{tech}|map|{split}|predicha|y{c}'] = _pool56(m_pred[c])
            medias[f'{tech}|map_norm_sin_constantes|{split}|real|y{c}'] = _pool56(sens_real['medias'][c])
            medias[f'{tech}|map_norm_sin_constantes|{split}|predicha|y{c}'] = _pool56(sens_pred['medias'][c])
        if 'backbone_crudo' in real[seeds[0]][split]:
            bb = ac.media_entre_semillas({s: real[s][split]['backbone_crudo'] for s in seeds})
            mb, _ = ac.medias_por_clase(bb, y)
            for c in (0, 1):
                medias[f'{tech}|backbone_map|{split}|real|y{c}'] = _pool56(mb[c])
        if tech == 'gray_scott':
            idx = ac.anillos(m_real[0].shape[0])
            d['perfil_radial'] = {
                'clase_real': {str(c): ac.perfil_radial(m_real[c], idx) for c in (0, 1)},
                'diferencia_real': ac.perfil_radial(m_real[1] - m_real[0], idx),
                'clase_predicha': {str(c): ac.perfil_radial(m_pred[c], idx) if m_pred[c] is not None else None
                                   for c in (0, 1)},
                'anillos': {'n': ac.N_ANILLOS, 'ancho_px': ac.ANCHO_ANILLO, 'centro': (m_real[0].shape[0] - 1) / 2}}
        if tech == 'lorenz' and u4 is not None:
            t = ac.tercil(u4[split], cortes)
            d['terciles_u4'] = {'cortes_train': cortes, 'conteos': {}}
            for k in range(3):
                sel = t == k
                mk, nk = ac.medias_por_clase(por_muestra[sel], y[sel])
                d['terciles_u4']['conteos'][str(k)] = {str(c): nk[c] for c in (0, 1)}
                for c in (0, 1):
                    medias[f'{tech}|map|{split}|tercil{k}|y{c}'] = _pool56(mk[c])
        out[split] = d
    return out, medias


def estadistico_val(real, res):
    """Mapas por muestra de val (media entre semillas, con la escala de la resolución) para la prueba."""
    seeds = list(real)
    esc, _ = ac.escalar_semillas({s: real[s]['val']['map_crudo'] for s in seeds}, res['escala_entre_semillas'])
    m = ac.media_entre_semillas(esc)
    if res['l2_resolucion'] == '56':
        m = ac.pool(m)
    return m, real[seeds[0]]['val']['y']


def cordura_datos(real, perm, res, tech, filas_csv):
    """Similitud real frente a permutado de la misma semilla, contra los 10 pares de semillas reales."""
    seeds = list(real)
    g_ssim = ac.GRID if res['ssim_resolucion'] == '56' else None
    kinds = ['map'] + (['backbone_map'] if 'backbone_norm' in real[seeds[0]]['val'] else [])
    out = {}
    for kind in kinds:
        key = 'map_norm' if kind == 'map' else 'backbone_norm'
        out[kind] = {}
        for split in SPLITS:
            M = {s: real[s][split][key].astype(np.float32) for s in seeds}
            P = {s: perm[s][split][key].astype(np.float32) for s in seeds}
            ids = real[seeds[0]][split]['sample_id']
            sim = {}
            for nombre, fn in (('spearman56', ac.spearman_filas),
                               ('ssim', lambda a, b: ac.ssim_filas(a, b, g=g_ssim))):
                rp, rp_vals, ref = {}, {}, {}
                for s in seeds:
                    v = fn(M[s], P[s])
                    rp_vals[s] = v
                    rp[s] = ac.resumen_similitud(v, res['similitud_constantes'])
                for a, b in ac.pares_semillas(tuple(seeds)):
                    v = fn(M[a], M[b])
                    ref[f'{a}-{b}'] = ac.resumen_similitud(v, res['similitud_constantes'])[0]
                    for sid, x in zip(ids, v):
                        filas_csv.append({'tecnica': tech, 'mapa': kind, 'split': split, 'medida': nombre,
                                          'comparacion': 'par_real', 'semilla_a': a, 'semilla_b': b,
                                          'sample_id': sid, 'valor': x})
                for s in seeds:
                    for sid, x in zip(ids, rp_vals[s]):
                        filas_csv.append({'tecnica': tech, 'mapa': kind, 'split': split, 'medida': nombre,
                                          'comparacion': 'real_vs_permutado', 'semilla_a': s, 'semilla_b': s,
                                          'sample_id': sid, 'valor': x})
                crit = ac.criterio_similitud({s: rp[s][0] for s in seeds}, ref, res)
                sim[nombre] = {'real_vs_permutado': {str(s): {'mediana': rp[s][0], 'n_usado': rp[s][1],
                                                              'n_nan': rp[s][2]} for s in seeds},
                               'referencia_pares': ref, **crit}
            out[kind][split] = sim
    return out


def gs_seguimiento(real, perm, res):
    """Gray-Scott: correlación de mapas real–permutado frente a la de p_dep, por semilla."""
    seeds = list(real)
    split = 'val' if res['gs_corr_pdep'] == 'misma_semilla_val' else 'test'
    variantes_pdep = {
        'f36_prueba_media_real': {s: ac.pearson(perm[s]['test']['p_dep'],
                                                np.mean([real[t]['test']['p_dep'] for t in seeds], 0)) for s in seeds},
        'misma_semilla_val': {s: ac.pearson(perm[s]['val']['p_dep'], real[s]['val']['p_dep']) for s in seeds},
        'misma_semilla_prueba': {s: ac.pearson(perm[s]['test']['p_dep'], real[s]['test']['p_dep']) for s in seeds}}
    variantes_mapas = {}
    for sp in SPLITS:
        variantes_mapas[f'spearman56_mediana|{sp}'] = {
            s: ac.resumen_similitud(ac.spearman_filas(real[s][sp]['map_norm'].astype(np.float32),
                                                      perm[s][sp]['map_norm'].astype(np.float32)),
                                    res['similitud_constantes'])[0] for s in seeds}
        variantes_mapas[f'pearson_media|{sp}'] = {
            s: float(np.nanmean(ac.pearson_filas(real[s][sp]['map_norm'].astype(np.float32),
                                                 perm[s][sp]['map_norm'].astype(np.float32)))) for s in seeds}
    principal = ac.seguimiento_gs(variantes_mapas[f"{res['gs_corr_mapas']}|{split}"], variantes_pdep[res['gs_corr_pdep']])
    return {'principal': {'corr_mapas': f"{res['gs_corr_mapas']}|{split}", 'corr_pdep': res['gs_corr_pdep'], **principal},
            'variantes_corr_pdep': {k: {str(s): v for s, v in d.items()} for k, d in variantes_pdep.items()},
            'variantes_corr_mapas': {k: {str(s): v for s, v in d.items()} for k, d in variantes_mapas.items()}}


def lecturas(resultados, cascada, res):
    """Lectura de los ViT con la cordura por datos (Spearman, val), el coseno y, si
    hay --cascada, el paso «cabeza»."""
    out = {}
    for t in ac.VIT:
        if t not in resultados:
            continue
        r = resultados[t]
        cab = None
        if cascada is not None:
            cab = cascada['tecnicas'][t]['pasos']['cabeza']['map']['spearman56']['global']
        cd = r['cordura_datos']['map']['val']['spearman56']['global']
        cn = ac.nivel_coseno(r['cosenos_cabezas'], res['coseno_lectura'])
        out[t] = {'cordura_datos': cd, 'coseno': cn, 'cabeza_cascada': cab, 'lectura': ac.lectura_vit(cd, cn, cab)}
    return out


def analizar(real_por_tecnica, perm_por_tecnica, cabezas, u4, cortes, res, cascada=None):
    """real/perm_por_tecnica: {técnica: {semilla: {split: d}}}; cabezas: {técnica: {'real': {s: W}, 'perm': {s: W}}}."""
    res = ac.check_resoluciones(res)
    rngs = ac.rngs_permutacion(res['perm_rng'], tuple(real_por_tecnica))
    resultados, medias, filas, pvals = {}, {}, [], {}
    for tech in real_por_tecnica:
        real, perm = real_por_tecnica[tech], perm_por_tecnica[tech]
        if list(real) != list(ac.SEEDS) or list(perm) != list(ac.SEEDS):
            raise Detener(f'{tech}: se esperan las semillas {ac.SEEDS} en real y permutado')
        agr, med = agregacion(real, res, tech, u4 if tech == 'lorenz' else None, cortes)
        medias.update(med)
        m, y = estadistico_val(real, res)
        prueba = ac.prueba_permutacion(m, y, rngs[tech], ac.PERM_B, res['perm_p'])
        pvals[tech] = prueba['p']
        r = {'agregacion': agr, 'permutacion_val': prueba, 'cordura_datos': cordura_datos(real, perm, res, tech, filas)}
        if tech in ac.VIT:
            r['cosenos_cabezas'] = ac.cosenos_cabezas(cabezas[tech]['real'], cabezas[tech]['perm'], res)
        if tech == 'gray_scott':
            r['gs_mapas_vs_pdep'] = gs_seguimiento(real, perm, res)
        resultados[tech] = r
    familia = ac.holm(pvals, ac.ALPHA)
    return {'resoluciones': res, 'tecnicas': resultados,
            'holm': {'alpha': ac.ALPHA, 'm': len(pvals), 'familia': familia},
            'lectura_vit': lecturas(resultados, cascada, res)}, medias, filas


# ── Compuerta y escritura ─────────────────────────────────────────────────
def refuse_overwrite(*paths):
    for p in paths:
        if Path(p).exists():
            raise Detener(f'{p} ya existe; no se sobrescribe')


def cargar_todo(root, index, techniques):
    art = load_artefactos()
    ub = Ubicador(root, art)
    real, perm, cabezas = {}, {}, {}
    for tech in techniques:
        real[tech], perm[tech] = {}, {}
        cabezas[tech] = {'real': {}, 'perm': {}}
        for s in ac.SEEDS:
            for es_perm, dest in ((False, real), (True, perm)):
                name = ac.run_name(tech, s, es_perm)
                rec = json.loads((RECORDS / f'{name}.json').read_text(encoding='utf-8'))
                p, row = ub.ruta(logical_maps(tech, s, es_perm))
                if row['sha256'] != rec['npz']['sha256']:
                    raise Detener(f'{name}: sha256 del manifiesto distinto del registro de F3.7')
                dest[tech][s] = cargar_npz(p, rec, index)
                if tech in ac.VIT:
                    pm, rowm = ub.ruta(logical_model(tech, s, es_perm))
                    if rowm['sha256'] != run_json(tech, s, es_perm)['model_file']['sha256']:
                        raise Detener(f'{name}: sha256 del modelo distinto de model_file del JSON de la corrida')
                    cabezas[tech]['perm' if es_perm else 'real'][s] = cabeza_de(pm)
        print(f'{tech}: 10 .npz verificados' + (' y 10 .pt' if tech in ac.VIT else ''), flush=True)
    return real, perm, cabezas, ub.verificados


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--root', required=True, help='copia de Drive MiUnidad/tt2-artefactos/')
    ap.add_argument('--resoluciones', required=True)
    ap.add_argument('--cascada', default=None, help='cascada.json de atribucion_gpu.py (opcional)')
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    raw_res = json.loads(Path(a.resoluciones).read_text(encoding='utf-8'))
    res = ac.check_resoluciones(raw_res)
    out = Path(a.out)
    refuse_overwrite(out / 'resultados.json', out / 'medias_clase.npz', out / 'similitudes_por_muestra.csv')
    from src.fase4.regimen import check_gate, load_u
    commit = check_gate()
    t0 = time.time()
    index = load_index()
    U = load_u(index, ('train', 'val', 'test'))
    cortes = ac.cortes_terciles(U['train'][0][:, 4])
    u4 = {s: np.asarray(U[s][0][:, 4], dtype=np.float64) for s in SPLITS}
    cascada = json.loads(Path(a.cascada).read_text(encoding='utf-8')) if a.cascada else None
    real, perm, cabezas, verificados = cargar_todo(a.root, index, ac.TECHNIQUES)
    r, medias, filas = analizar(real, perm, cabezas, u4, cortes, res, cascada)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'medias_clase.npz', **{k: v.astype(np.float32) for k, v in medias.items() if v is not None})
    with open(out / 'similitudes_por_muestra.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(filas)
    doc = {'meta': {'tarea': 'F4.3', 'parte': 'local', 'git_commit': commit,
                    'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'resoluciones': {'archivo': str(Path(a.resoluciones).as_posix()), 'sha256': sha256_file(a.resoluciones),
                                     **ac.meta_resoluciones(raw_res)},
                    'cascada': {'archivo': a.cascada, 'sha256': sha256_file(a.cascada)} if a.cascada else None,
                    'artefactos_verificados': verificados, 'segundos': round(time.time() - t0, 1),
                    'medias_clase_sha256': sha256_file(out / 'medias_clase.npz'),
                    'similitudes_sha256': sha256_file(out / 'similitudes_por_muestra.csv')},
           'resultados': ac.limpiar(r)}
    (out / 'resultados.json').write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(ac.limpiar({'holm': r['holm'], 'lectura_vit': r['lectura_vit']}), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
