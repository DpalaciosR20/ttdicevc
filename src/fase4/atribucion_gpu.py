"""
atribucion_gpu.py
─────────────────
F4.3 — Atribución espacial y espectral (docs/plan_fase4.md, F4.3), parte con
inferencia. Se ejecuta en Kaggle (GPU). **Solo val** (G2): el código se niega a
cargar cualquier otro split.

1. Cascada de cordura por parámetros (F4.3-2), para las 5 semillas de cada
   finalista de F3.5:
     EfficientNet-B0  cabeza (classifier.1) → features[8] … features[0]
     ViT-B/16         cabeza (heads.head) → encoder.layers 11 … 0 → embeddings
   Reinicialización acumulativa con reset_parameters() de PyTorch y
   torch.manual_seed(2026), sobre una copia en CPU (el resultado no depende del
   dispositivo). En cada paso se recalculan los mapas de F3.7 (Grad-CAM en
   EfficientNet; rollout ponderado y atención del backbone en ViT) y se compara
   cada muestra con el mapa del modelo sin aleatorizar: Spearman a 56 × 56
   (principal) y SSIM a 56 y 224 (secundaria). Criterio relativo: la mediana
   por muestra frente al rango de los 10 pares de semillas reales (mapas sin
   aleatorizar).
2. Ablación espectral (F4.3-6): paso-bajo gaussiano σ ∈ {1, 2, 4, 8} px sobre la
   imagen de 256 px, antes del preprocesamiento de F1.4, y paso-alto
   complementario (imagen − paso-bajo + media de la imagen). ΔF1 = media sobre
   las 5 semillas de (F1-val filtrado − F1-val sin filtrar).

Verificaciones (se detiene ante cualquier discrepancia):
  - sha256 de cada .pt contra meta/artefactos_manifest.csv y model_file.sha256;
  - sha256 de cada imagen de val contra meta/data_manifest.csv;
  - p_dep del modelo sin aleatorizar contra val_probs del JSON (≤ --tol-pdep);
  - con --maps-root, el mapa sin aleatorizar contra el .npz de F3.7 (val).
Compuerta: enmienda-5 en la historia de HEAD, docs/plan_fase4.md sin cambios y
el archivo --resoluciones versionado y sin cambios.
Nunca sobrescribe: si existe alguna salida en --out-dir, se detiene.

Uso (Kaggle):
  python -m src.fase4.atribucion_gpu --models-root /kaggle/input --maps-root /kaggle/input \
      --resoluciones results/f43_atribucion/resoluciones.json --out-dir results/f43_atribucion/gpu
  --limit N: solo las N primeras muestras de val (prueba de humo y estimación;
  escribe "smoke": true; usar un --out-dir temporal).
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
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

SPLIT = 'val'
N_VAL = 114
SALIDAS = ('cascada.json', 'cascada_similitudes.npz', 'cascada_medias.npz', 'ablacion.json', 'ablacion.csv',
           'atribucion_gpu_meta.json')
PLAN = 'docs/plan_fase4.md'
GATE_TAG = 'enmienda-5'


# ── G2: solo val ──────────────────────────────────────────────────────────
def exigir_val(split):
    if split != SPLIT:
        raise Detener(f'G2: atribucion_gpu solo usa val; se pidió {split!r}')


def filas_val(index_rows):
    """Filas de sample_index de val, en orden de sample_id. Ninguna otra."""
    rows = [r for r in index_rows if r['split'] == SPLIT]
    for r in rows:
        exigir_val(r['split'])
    if len(rows) != N_VAL:
        raise Detener(f'sample_index: {len(rows)} filas de val; se esperan {N_VAL}')
    return sorted(rows, key=lambda r: r['sample_id'])


def nombres_val(rows, tech):
    names = [r[f'img_{tech}'] for r in rows]
    for r, n in zip(rows, names):
        exigir_val(r['split'])
        if f'_{SPLIT}_' not in n or '_test_' in n or '_train_' in n:
            raise Detener(f'G2: {n} no es una imagen de val')
    return names


def leer_imagenes_val(image_dir, rel_dir, rows, tech, expected):
    """Bytes verificados (sha256 contra data_manifest) de las imágenes de val."""
    from src.fase4.caracteristicas import read_verified_images
    names = nombres_val(rows, tech)
    data, hashes = read_verified_images(image_dir, rel_dir, names, expected)
    return data, hashes


def decodificar(b):
    from PIL import Image
    a = np.array(Image.open(io.BytesIO(b)).convert('L'))
    if a.shape != (256, 256):
        raise Detener(f'imagen de forma {a.shape}; se espera 256 × 256')
    return a


# ── Mapas ─────────────────────────────────────────────────────────────────
def tensores(imgs_u8, tf):
    import torch
    from PIL import Image
    return torch.stack([tf(Image.fromarray(a, 'L')) for a in imgs_u8])


def calcular_mapas(model, X, metodo, device, batch_size=16, target=1):
    """{'map', 'backbone_map' (ViT) o None, 'p_dep'}; mapas sin normalizar float32 (N, H, W)."""
    import torch
    from src.interpretability import attribution as at
    maps, bb, probs = [], [], []
    for i in range(0, len(X), batch_size):
        x = X[i:i + batch_size].to(device)
        with torch.no_grad():
            plain = model(x)
        probs.append(torch.softmax(plain.double(), 1)[:, 1].cpu().numpy())
        if metodo['method'] == 'grad_cam':
            cam, _ = at.grad_cam(model, x, at.resolve_layer(model, metodo['layer']), target=target)
            maps.append(cam.float().cpu().numpy())
        else:
            r, _ = at.vit_attributions(model, x, target=target)
            maps.append(r['gradient_rollout'].float().cpu().numpy())
            bb.append(r['attention_rollout'].float().cpu().numpy())
    return {'map': np.concatenate(maps), 'backbone_map': np.concatenate(bb) if bb else None,
            'p_dep': np.concatenate(probs)}


# ── Cascada ───────────────────────────────────────────────────────────────
def internal_arch(arch):
    return {'vit_b16': 'vit_b_16'}.get(arch, arch)


def pasos_cascada(model, arch, res):
    """[(paso, [módulos o parámetros])], desde la salida."""
    arch = internal_arch(arch)
    if arch == 'efficientnet_b0':
        return [('cabeza', ['classifier.1'])] + \
               [(f'features.{i}', [f'features.{i}']) for i in reversed(range(len(model.features)))]
    if arch == 'vit_b_16':
        cab = ['heads.head'] + (['encoder.ln'] if res['vit_ln_final'] == 'con_cabeza' else [])
        pasos = [('cabeza', cab)]
        if res['vit_ln_final'] == 'paso_propio':
            pasos.append(('encoder.ln', ['encoder.ln']))
        n = len(model.encoder.layers)
        pasos += [(f'encoder.layers.{i}', [f'encoder.layers.encoder_layer_{i}']) for i in reversed(range(n))]
        emb = ['conv_proj'] + (['class_token', 'encoder.pos_embedding'] if res['vit_embeddings'] == 'conv_proj_y_tokens' else [])
        return pasos + [('embeddings', emb)]
    raise Detener(f'arquitectura no prevista en F4.3: {arch}')


def reiniciar(model, nombres, res):
    """Reinicializa en su lugar los módulos (reset_parameters de cada submódulo) o
    parámetros nombrados. Devuelve las claves del state_dict que cambiaron.
    nn.MultiheadAttention no tiene reset_parameters público: con 'reset_mha_completo'
    se llama a su _reset_parameters (in_proj); su out_proj se reinicia aparte."""
    import torch
    import torch.nn as nn
    antes = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for name in nombres:
            if name == 'class_token':
                nn.init.zeros_(model.class_token)                     # init de torchvision
            elif name == 'encoder.pos_embedding':
                nn.init.normal_(model.encoder.pos_embedding, std=0.02)  # init de torchvision
            else:
                for m in model.get_submodule(name).modules():
                    if isinstance(m, nn.MultiheadAttention):
                        if res['mha_in_proj'] == 'reset_mha_completo':
                            m._reset_parameters()
                    elif callable(getattr(m, 'reset_parameters', None)):
                        m.reset_parameters()
    despues = model.state_dict()
    cambiadas = [k for k in antes if not torch.equal(antes[k], despues[k])]
    if not cambiadas:
        raise Detener(f'el paso {nombres} no cambió ningún parámetro')
    return cambiadas


def similitudes(base, nuevo, grid):
    """Por muestra: Spearman (rejilla grid), SSIM a la rejilla y SSIM a resolución completa,
    y si cada mapa es constante en la rejilla (const_base, const_nuevo; P9)."""
    a, b = ac.minmax(base), ac.minmax(nuevo)
    return {'spearman56': ac.spearman_filas(a, b, grid), 'ssim56': ac.ssim_filas(a, b, g=grid),
            'ssim224': ac.ssim_filas(a, b), 'const_base': ac.es_constante(ac.pool(a, grid)),
            'const_nuevo': ac.es_constante(ac.pool(b, grid))}


def cascada_modelo(model_cpu, arch, X, metodo, device, res, batch_size=16, grid=ac.GRID, y=None):
    """Mapas sin aleatorizar y, en cada paso acumulativo, similitudes con ellos."""
    m_dev = copy.deepcopy(model_cpu).to(device).eval()
    base = calcular_mapas(m_dev, X, metodo, device, batch_size)
    pasos, sims, medias = [], {}, {}
    import torch
    kinds = ['map'] + (['backbone_map'] if base['backbone_map'] is not None else [])
    if res['semilla_cascada'] == 'una_vez_por_modelo':
        torch.manual_seed(ac.CASCADA_SEED)
    for paso, nombres in pasos_cascada(model_cpu, arch, res):
        if res['semilla_cascada'] == 'en_cada_paso':
            torch.manual_seed(ac.CASCADA_SEED)
        cambiadas = reiniciar(model_cpu, nombres, res)
        m_dev.load_state_dict(model_cpu.state_dict())
        m_dev.eval()
        r = calcular_mapas(m_dev, X, metodo, device, batch_size)
        pasos.append({'paso': paso, 'modulos': nombres, 'n_tensores_cambiados': len(cambiadas)})
        for k in kinds:
            sims[(paso, k)] = similitudes(base[k], r[k], grid)
            if y is not None:
                for c in (0, 1):
                    if (np.asarray(y) == c).any():
                        medias[(paso, k, c)] = ac.minmax(r[k])[np.asarray(y) == c].mean(0)
    return base, pasos, sims, medias


def resumen_cascada(bases, sims_por_semilla, pasos, res, grid=ac.GRID):
    """Criterio relativo por paso y tipo de mapa, con la referencia de los pares de semillas."""
    seeds = list(bases)
    kinds = ['map'] + (['backbone_map'] if bases[seeds[0]]['backbone_map'] is not None else [])
    fns = {'spearman56': lambda a, b: ac.spearman_filas(ac.minmax(a), ac.minmax(b), grid),
           'ssim56': lambda a, b: ac.ssim_filas(ac.minmax(a), ac.minmax(b), g=grid),
           'ssim224': lambda a, b: ac.ssim_filas(ac.minmax(a), ac.minmax(b))}
    referencia = {k: {med: ac.referencia_mapas({s: bases[s][k] for s in seeds}, fn, res['similitud_constantes'])
                      for med, fn in fns.items()} for k in kinds}
    out = {}
    for p in pasos:
        out[p['paso']] = {'modulos': p['modulos']}
        for k in kinds:
            out[p['paso']][k] = {}
            for med in fns:
                per = {}
                info = {}
                for s in seeds:
                    d = sims_por_semilla[s][(p['paso'], k)]
                    if med == 'spearman56':
                        m, n, nn_, tr, dc = ac.resumen_cascada_spearman(d[med], d['const_base'], d['const_nuevo'],
                                                                        res['similitud_constantes'])
                        info[str(s)] = {'mediana': m, 'n_usado': n, 'n_nan': nn_,
                                        'n_transiciones_a_constante': tr, 'n_base_constante': dc}
                    else:
                        m, n, nn_ = ac.resumen_similitud(d[med], res['similitud_constantes'])
                        info[str(s)] = {'mediana': m, 'n_usado': n, 'n_nan': nn_}
                    per[s] = m
                if med == 'spearman56':
                    info['_regla_P9'] = ('mapa no constante → constante cuenta como cambio: Spearman imputado −1 '
                                         'antes de la mediana')
                out[p['paso']][k][med] = {'por_semilla_detalle': info, 'referencia_pares': referencia[k][med],
                                          **ac.criterio_similitud(per, referencia[k][med], res)}
    return out, referencia


# ── Ablación ──────────────────────────────────────────────────────────────
def a_uint8(y, rango):
    if rango == 'recortar_redondear':
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)
    if rango == 'reescalar_min_max':
        lo, hi = float(y.min()), float(y.max())
        if hi > lo:
            return np.rint((y - lo) / (hi - lo) * 255).astype(np.uint8)
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)
    raise Detener(f'ablacion_rango desconocido: {rango!r}')


def filtrar(img, tipo, sigma, borde, rango):
    """Imagen uint8 256 × 256 → filtrada uint8. paso_bajo: gaussiana (truncate 4.0);
    paso_alto: imagen − paso_bajo + media de la imagen."""
    from scipy.ndimage import gaussian_filter
    x = img.astype(np.float64)
    low = gaussian_filter(x, sigma=sigma, mode=borde)
    if tipo == 'paso_bajo':
        y = low
    elif tipo == 'paso_alto':
        y = x - low + x.mean()
    else:
        raise Detener(f'tipo de filtro desconocido: {tipo!r}')
    return a_uint8(y, rango)


def condiciones():
    return [('original', None)] + [('paso_bajo', s) for s in ac.SIGMAS] + [('paso_alto', s) for s in ac.SIGMAS]


def f1_macro(y, yhat):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(y), np.asarray(yhat), average='macro', zero_division=0))


def predecir(model, X, device, batch_size=32):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(torch.softmax(model(X[i:i + batch_size].to(device)).double(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def ablacion_modelo(model, imgs_u8, y, tf, device, res, batch_size=32):
    """{condición: {'f1', 'p_dep', 'frac_recortada'}} para un modelo."""
    out = {}
    for tipo, s in condiciones():
        if tipo == 'original':
            filt, frac = imgs_u8, 0.0
        else:
            filt = [filtrar(a, tipo, s, res['ablacion_borde'], res['ablacion_rango']) for a in imgs_u8]
            frac = 0.0
            if tipo == 'paso_alto' and res['ablacion_rango'] == 'recortar_redondear':
                from scipy.ndimage import gaussian_filter
                raw = [a.astype(np.float64) - gaussian_filter(a.astype(np.float64), s, mode=res['ablacion_borde'])
                       + a.mean() for a in imgs_u8]
                frac = float(np.mean([((r < -0.5) | (r > 255.5)).mean() for r in raw]))
        p = predecir(model, tensores(filt, tf), device, batch_size)
        out[f'{tipo}' + (f'_s{s}' if s else '')] = {'f1': f1_macro(y, (p > 0.5).astype(int)), 'p_dep': p,
                                                    'frac_recortada': frac}
    return out


def resumen_ablacion(por_semilla):
    """ΔF1 por condición: por semilla y media sobre semillas (F1 filtrado − F1 sin filtrar)."""
    seeds = list(por_semilla)
    out = {}
    for cond in por_semilla[seeds[0]]:
        f1 = {s: por_semilla[s][cond]['f1'] for s in seeds}
        d = {s: f1[s] - por_semilla[s]['original']['f1'] for s in seeds}
        out[cond] = {'f1_por_semilla': {str(s): v for s, v in f1.items()}, 'f1_media': float(np.mean(list(f1.values()))),
                     'delta_f1_por_semilla': {str(s): v for s, v in d.items()},
                     'delta_f1_media': float(np.mean(list(d.values()))),
                     'frac_recortada_media': float(np.mean([por_semilla[s][cond]['frac_recortada'] for s in seeds]))}
    return out


# ── Compuerta y verificación ──────────────────────────────────────────────
def _git(*args):
    return subprocess.run(['git', '-C', REPO, *args], capture_output=True, text=True)


def check_gate(resoluciones_path):
    if _git('merge-base', '--is-ancestor', GATE_TAG, 'HEAD').returncode != 0:
        raise Detener(f'la etiqueta {GATE_TAG} no está en la historia de HEAD')
    if _git('diff', '--quiet', GATE_TAG, 'HEAD', '--', PLAN).returncode != 0:
        raise Detener(f'{PLAN} cambió frente a {GATE_TAG}')
    rel = os.path.relpath(os.path.abspath(resoluciones_path), REPO).replace(os.sep, '/')
    if _git('ls-files', '--error-unmatch', rel).returncode != 0:
        raise Detener(f'{rel} no está versionado')
    if _git('diff', '--quiet', 'HEAD', '--', rel).returncode != 0:
        raise Detener(f'{rel} tiene cambios sin commit')
    return _git('rev-parse', 'HEAD').stdout.strip()


def refuse_overwrite(out_dir):
    for n in SALIDAS:
        if os.path.exists(os.path.join(out_dir, n)):
            raise Detener(f'{os.path.join(out_dir, n)} ya existe: atribucion_gpu nunca sobrescribe')


def npz_val(maps_root, tech, seed, artefactos, n):
    """Mapas de val del .npz de F3.7 (copia en ttdicevc/results/f37_mapas/maps/, sha256 verificado)."""
    from src.fase4.caracteristicas import sha256_file
    from src.verify_artefacts import find_all, locate
    row = artefactos[f'MAPS[{ac.run_name(tech, seed)}]']
    found = find_all(maps_root, ('.npz',))
    p, _, amb = locate(found.get(os.path.basename(row['rel_path']), []), row['rel_path'])
    if p is None or 'f37_humo' in p.replace(os.sep, '/'):
        raise Detener(f'{row["rel_path"]}: {"ambiguo" if amb else "falta o es copia de humo"} bajo {maps_root}')
    if sha256_file(p) != row['sha256']:
        raise Detener(f'{p}: sha256 distinto de artefactos_manifest.csv')
    z = np.load(p)
    m = z['split'].astype(str) == SPLIT
    out = {'map': z['map'][m][:n].astype(np.float32)}
    if 'backbone_map' in z.files:
        out['backbone_map'] = z['backbone_map'][m][:n].astype(np.float32)
    return out


def comparar_con_npz(base, ref):
    """Diferencia máxima del mapa normalizado y mediana de Spearman frente al .npz."""
    out = {}
    for k, v in ref.items():
        a = ac.minmax(base[k])
        out[k] = {'max_abs_diff_normalizado': float(np.abs(a - v).max()),
                  'spearman56_mediana': float(np.nanmedian(ac.spearman_filas(a, v)))}
    return out


# ── Principal ─────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models-root', required=True)
    ap.add_argument('--maps-root', default=None, help='raíz con tt2-mapas para comparar el mapa sin aleatorizar')
    ap.add_argument('--resoluciones', required=True)
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'results', 'f43_atribucion', 'gpu'))
    ap.add_argument('--only', nargs='+', default=list(ac.TECHNIQUES))
    ap.add_argument('--partes', nargs='+', default=['cascada', 'ablacion'], choices=['cascada', 'ablacion'])
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--tol-pdep', type=float, default=1e-4)
    ap.add_argument('--tol-mapa', type=float, default=1e-2)
    a = ap.parse_args(argv)

    raw_res = json.loads(open(a.resoluciones, encoding='utf-8').read())
    res = ac.check_resoluciones(raw_res)
    refuse_overwrite(a.out_dir)
    commit = check_gate(a.resoluciones)

    import torch
    import config
    from build_model import build_model
    from data import get_transforms
    from train_model import configure_determinism
    from src import paths
    from src.fase4 import caracteristicas as car
    from src.interpretability.run_maps import method_for

    matrix = config.load_matrix()
    det = configure_determinism(config.determinism(matrix))
    target = matrix['interpretabilidad']['target']['logit_class']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    artefactos, data_manifest = car.load_artefactos(), car.load_data_manifest()
    rows = filas_val(car.load_index())
    if a.limit:
        rows = rows[:a.limit]
    y = np.array([int(r['y_true']) for r in rows])
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    meta = {'task': 'F4.3', 'parte': 'gpu', 'split': SPLIT, 'smoke': bool(a.limit), 'limit': a.limit,
            'git_commit': commit, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'torch': torch.__version__, 'python': platform.python_version(), 'determinism': det,
            'resoluciones': {'archivo': a.resoluciones, 'sha256': car.sha256_file(a.resoluciones),
                             **ac.meta_resoluciones(raw_res)},
            'g2': 'solo val: filas_val, nombres_val y leer_imagenes_val rechazan cualquier otro split',
            'tecnicas': {}}
    cascada, ablacion, sims_npz, medias_npz, filas_abl = {'tecnicas': {}}, {'tecnicas': {}}, {}, {}, []
    for tech in a.only:
        tt = time.time()
        arch, strat = ac.FINALIST[tech]
        runs = {s: json.load(open(car.run_json_path(tech, s), encoding='utf-8')) for s in ac.SEEDS}
        prep = runs[42]['preprocessing']
        if any(r['preprocessing'] != prep for r in runs.values()):
            raise Detener(f'{tech}: preprocesamiento distinto entre semillas')
        for s, r in runs.items():
            if r['val_true'][:len(y)] != y.tolist():
                raise Detener(f'{tech} semilla {s}: val_true distinto de sample_index')
        tf = get_transforms(False, img_size=prep['size'], hflip=False, interpolation=prep['interpolation'],
                            antialias=prep['antialias'])
        expected, _ = car.image_rows(data_manifest, tech)
        imgs_b, hashes = leer_imagenes_val(str(paths.images_dir(tech)), paths.REL['IMAGES'][tech], rows, tech, expected)
        imgs = [decodificar(b) for b in imgs_b]
        X = tensores(imgs, tf)
        metodo = method_for(arch, matrix)
        info = {'arch': arch, 'preprocessing': prep, 'n': len(rows),
                'imagenes_sha256_lista': hashlib.sha256(''.join(f'{k}\t{v}\n' for k, v in sorted(hashes.items()))
                                                        .encode()).hexdigest(), 'modelos': {}, 'segundos': {}}
        bases, sims, abl = {}, {}, {}
        for s in ac.SEEDS:
            tl = time.time()
            path, digest = car.verified_model(a.models_root, artefactos, tech, s, runs[s])
            model = build_model(arch, freeze_backbone=strat == 'frozen', pretrained=False)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            model.eval()
            info['segundos'][f'carga_s{s}'] = round(time.time() - tl, 1)
            info['modelos'][str(s)] = {'archivo': os.path.basename(path), 'sha256': digest}
            if 'cascada' in a.partes:
                tc = time.time()
                base, pasos, sims[s], medias_paso = cascada_modelo(model, arch, X, metodo, device, res, a.batch_size, y=y)
                dp = float(np.abs(base['p_dep'] - np.asarray(runs[s]['val_probs'][:len(y)])).max())
                info['modelos'][str(s)]['p_dep_max_abs_diff'] = dp
                if dp > a.tol_pdep:
                    raise Detener(f'{tech} semilla {s}: p_dep no reproduce val_probs ({dp:.2e} > {a.tol_pdep})')
                if a.maps_root:
                    cmp = comparar_con_npz(base, npz_val(a.maps_root, tech, s, artefactos, len(y)))
                    info['modelos'][str(s)]['vs_npz_f37'] = cmp
                    if any(v['max_abs_diff_normalizado'] > a.tol_mapa for v in cmp.values()):
                        raise Detener(f'{tech} semilla {s}: el mapa sin aleatorizar difiere del .npz de F3.7 {cmp}')
                bases[s] = base
                info['pasos'] = pasos
                for (paso, k), d in sims[s].items():
                    for med, v in d.items():
                        sims_npz[f'{tech}|{s}|{paso}|{k}|{med}'] = v.astype(np.float32)
                for (paso, k, c), m in medias_paso.items():
                    medias_npz[f'{tech}|{s}|{paso}|{k}|y{c}'] = ac.pool(m[None])[0].astype(np.float32) \
                        if m.shape[0] % ac.GRID == 0 else m.astype(np.float32)
                for k in ('map', 'backbone_map'):
                    if base[k] is not None:
                        for c in (0, 1):
                            if not (y == c).any():
                                continue
                            mm = ac.minmax(base[k])[y == c].mean(0)
                            medias_npz[f'{tech}|{s}|original|{k}|y{c}'] = ac.pool(mm[None])[0].astype(np.float32)
                info['segundos'][f'cascada_s{s}'] = round(time.time() - tc, 1)
                model = build_model(arch, freeze_backbone=strat == 'frozen', pretrained=False)
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
                model.eval()
            if 'ablacion' in a.partes:
                ta = time.time()
                m_dev = model.to(device).eval()
                abl[s] = ablacion_modelo(m_dev, imgs, y, tf, device, res, a.batch_size)
                dp = float(np.abs(abl[s]['original']['p_dep'] - np.asarray(runs[s]['val_probs'][:len(y)])).max())
                info['modelos'][str(s)]['ablacion_p_dep_original_max_abs_diff'] = dp
                if dp > a.tol_pdep:
                    raise Detener(f'{tech} semilla {s}: p_dep sin filtrar no reproduce val_probs ({dp:.2e})')
                for cond, d in abl[s].items():
                    filas_abl.append({'tecnica': tech, 'semilla': s, 'condicion': cond, 'f1_val': d['f1'],
                                      'delta_f1': d['f1'] - abl[s]['original']['f1'],
                                      'frac_recortada': d['frac_recortada']})
                info['segundos'][f'ablacion_s{s}'] = round(time.time() - ta, 1)
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f'{tech} semilla {s}: {info["segundos"]}', flush=True)
        if 'cascada' in a.partes:
            resumen, ref = resumen_cascada(bases, sims, info['pasos'], res)
            cascada['tecnicas'][tech] = {'pasos': resumen, 'orden': [p['paso'] for p in info['pasos']]}
        if 'ablacion' in a.partes:
            ablacion['tecnicas'][tech] = resumen_ablacion(abl)
        info['segundos']['total'] = round(time.time() - tt, 1)
        meta['tecnicas'][tech] = info
    meta['segundos_total'] = round(time.time() - t0, 1)
    if a.limit:
        meta['estimacion_114'] = {
            t: round(sum(v for k, v in i['segundos'].items() if k.startswith(('cascada', 'ablacion'))) * N_VAL / a.limit
                     + sum(v for k, v in i['segundos'].items() if k.startswith('carga')), 1)
            for t, i in meta['tecnicas'].items()}
        meta['estimacion_114_total'] = round(sum(meta['estimacion_114'].values()), 1)
        print('estimacion_114 (s; cómputo escalado a 114 + carga):', meta['estimacion_114'],
              '| total', meta['estimacion_114_total'], flush=True)
    if 'cascada' in a.partes:
        cascada['meta'] = {'git_commit': commit, 'smoke': bool(a.limit), 'resoluciones': res}
        with open(os.path.join(a.out_dir, 'cascada.json'), 'w', encoding='utf-8') as f:
            json.dump(ac.limpiar(cascada), f, indent=1, ensure_ascii=False)
        np.savez_compressed(os.path.join(a.out_dir, 'cascada_similitudes.npz'), **sims_npz)
        np.savez_compressed(os.path.join(a.out_dir, 'cascada_medias.npz'), **medias_npz)
    if 'ablacion' in a.partes:
        ablacion['meta'] = {'git_commit': commit, 'smoke': bool(a.limit), 'sigmas': list(ac.SIGMAS),
                            'resoluciones': {k: res[k] for k in ('ablacion_borde', 'ablacion_rango')}}
        with open(os.path.join(a.out_dir, 'ablacion.json'), 'w', encoding='utf-8') as f:
            json.dump(ac.limpiar(ablacion), f, indent=1, ensure_ascii=False)
        with open(os.path.join(a.out_dir, 'ablacion.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(filas_abl[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(filas_abl)
    with open(os.path.join(a.out_dir, 'atribucion_gpu_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(ac.limpiar(meta), f, indent=1, ensure_ascii=False)
    print(f'F4.3 GPU: {meta["segundos_total"]} s')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
