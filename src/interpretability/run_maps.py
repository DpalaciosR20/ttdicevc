"""
run_maps.py
───────────
F3.7: mapas de atribución de los modelos guardados de F3.5 (final) y F3.6
(control de permutación) sobre las 114 muestras de val y las 114 de test
(matriz §8, Enmienda 3).

Por cada corrida (un JSON de results/<fuente>/<técnica>/runs/):
  1. Busca su .pt bajo --models-root por nombre y comprueba el SHA-256
     contra model_file.sha256 del JSON. Construye la arquitectura sin pesos
     de ImageNet y carga el state_dict (strict).
  2. Reproduce p_dep con una pasada normal (sin hooks) y lo compara con
     val_probs/test_probs del JSON: si difiere más de --tol-pdep, la corrida
     falla (modelo o preprocesamiento distintos).
  3. Calcula los mapas: Grad-CAM (efficientnet_b0, resnet18) o, en vit_b16,
     el rollout ponderado por gradiente (mapa principal) y el attention
     rollout (atención del backbone). Registra la diferencia máxima entre los
     logits con y sin la atención explícita.
  4. Escribe <maps-dir>/<corrida>.npz (comprimido) y
     <out-dir>/records/<corrida>.json.

Formato del .npz (indexado por sample_id, orden val y luego test):
  sample_id (str), split (str), y_true (int8), p_dep (float64)
  map            float16 (N, 224, 224)  normalizado por muestra a [0, 1]
  map_raw        float16 (N, 224, 224)  sin normalizar, dividido entre map_raw_scale
  map_raw_scale  float32 (N,)           máximo del mapa sin normalizar (1 si es 0)
  En vit_b16, además backbone_map, backbone_map_raw y backbone_map_raw_scale.
  El mapa sin normalizar es map_raw * map_raw_scale[:, None, None]. El factor
  evita que float16 pierda los valores pequeños (Grad-CAM con pesos aleatorios,
  relevancias del rollout del orden de 1e-4).

Uso:
  python -m src.interpretability.run_maps --models-root /kaggle/input \
      --out-dir results/f37_mapas [--limit 4] [--only perlin] [--batch-size 16]
--limit N toma solo las N primeras muestras de cada split (prueba de humo y
estimación de duración); escribe en <out-dir> con "smoke": true y no se
mezcla con una corrida completa.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING = os.path.join(REPO, 'src', 'training')
for p in (REPO, TRAINING):
    if p not in sys.path:
        sys.path.insert(0, p)

SOURCES = ('results/f35_final', 'results/f36_permutacion')
SPLITS = ('val', 'test')
FREEZE = {'frozen': True, 'finetune': False}


def write_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def list_runs(sources, only=None):
    runs = []
    for src in sources:
        for p in sorted(glob.glob(os.path.join(REPO, src, '*', 'runs', '*.json'))):
            d = json.load(open(p, encoding='utf-8'))
            if only and d['technique'] not in only:
                continue
            if 'model_file' not in d:
                raise SystemExit(f'{p} no tiene model_file: la corrida no guardó su modelo')
            runs.append((os.path.relpath(p, REPO).replace(os.sep, '/'), d))
    return runs


def find_models(root):
    from src.verify_artefacts import find_models as fm
    found, dup = fm(root)
    if dup:
        raise SystemExit(f'modelos duplicados bajo {root}: {dup[:5]}')
    return found


def method_for(arch, matrix):
    return matrix['interpretabilidad']['methods'][arch]


def to_npz_arrays(prefix, raw):
    """raw: float32 (N, H, W) -> {prefix: minmax f16, prefix_raw: f16 escalado, prefix_raw_scale}."""
    import torch
    from src.interpretability.attribution import minmax
    t = torch.from_numpy(raw)
    norm = minmax(t).numpy().astype(np.float16)
    scale = raw.reshape(len(raw), -1).max(1).astype(np.float32)
    scale[scale <= 0] = 1.0
    return {prefix: norm, f'{prefix}_raw': (raw / scale[:, None, None]).astype(np.float16),
            f'{prefix}_raw_scale': scale}


def process_run(run_rel, d, model_path, image_cache, expected, matrix, device, batch_size, limit, tol_pdep):
    import torch
    from torch.utils.data import DataLoader
    from build_model import build_model
    from data import ClassBlindImageDataset, PREFIXES, get_transforms, list_split_files
    from umap_io import sha256
    from src import paths
    from src.interpretability import attribution as at
    import config

    t0 = time.time()
    digest = sha256(model_path)
    if digest != d['model_file']['sha256']:
        raise RuntimeError(f'SHA-256 del modelo distinto del JSON: {digest} != {d["model_file"]["sha256"]}')
    arch = config.internal_arch(d['arch'])
    model = build_model(arch, freeze_backbone=FREEZE[d['strategy']], pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
    model.eval().to(device)
    meth = method_for(d['arch'], matrix)
    prep = d['preprocessing']
    tech = d['technique']
    tf = get_transforms(False, img_size=prep['size'], hflip=False, interpolation=prep['interpolation'],
                        antialias=prep['antialias'])

    out = {k: [] for k in ('sample_id', 'split', 'y_true', 'p_dep')}
    maps = {'map': [], 'backbone_map': []}
    pdep_diff, logit_diff = {}, 0.0
    for split in SPLITS:
        key = (tech, split)
        if key not in image_cache:
            files, labels = list_split_files(str(paths.images_dir(tech)), PREFIXES[tech], split, expected[split])
            image_cache[key] = (files, labels)
        files, labels = image_cache[key]
        n = len(files) if limit is None else min(limit, len(files))
        if labels[:n] != d[f'{split}_true'][:n]:
            raise RuntimeError(f'{split}: etiquetas distintas de las del JSON')
        loader = DataLoader(ClassBlindImageDataset(files[:n], labels[:n], tf), batch_size=batch_size,
                            shuffle=False, num_workers=2)
        probs = []
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                plain = model(x)
            probs.append(torch.softmax(plain.double(), 1)[:, 1].cpu().numpy())
            if meth['method'] == 'grad_cam':
                cam, logits = at.grad_cam(model, x, at.resolve_layer(model, meth['layer']),
                                          target=matrix['interpretabilidad']['target']['logit_class'])
                maps['map'].append(cam.float().cpu().numpy())
            else:
                res, logits = at.vit_attributions(model, x, target=matrix['interpretabilidad']['target']['logit_class'])
                maps['map'].append(res['gradient_rollout'].float().cpu().numpy())
                maps['backbone_map'].append(res['attention_rollout'].float().cpu().numpy())
            logit_diff = max(logit_diff, float((logits - plain).abs().max()))
        p = np.concatenate(probs)
        ref = np.asarray(d[f'{split}_probs'][:n], dtype=np.float64)
        pdep_diff[split] = float(np.abs(p - ref).max())
        out['sample_id'] += [f'{split}_{i:04d}' for i in range(n)]
        out['split'] += [split] * n
        out['y_true'] += labels[:n]
        out['p_dep'].append(p)
    if max(pdep_diff.values()) > tol_pdep:
        raise RuntimeError(f'p_dep no reproduce el JSON: {pdep_diff} > {tol_pdep}')
    arrays = {'sample_id': np.array(out['sample_id']), 'split': np.array(out['split']),
              'y_true': np.array(out['y_true'], dtype=np.int8), 'p_dep': np.concatenate(out['p_dep'])}
    arrays.update(to_npz_arrays('map', np.concatenate(maps['map'])))
    if maps['backbone_map']:
        arrays.update(to_npz_arrays('backbone_map', np.concatenate(maps['backbone_map'])))
    methods = {'map': meth['method']}
    if maps['backbone_map']:
        methods['backbone_map'] = meth['secondary']['method']
    return arrays, {'methods': methods, 'model_sha256_verified': digest, 'p_dep_max_abs_diff': pdep_diff,
                    'logits_hooked_max_abs_diff': logit_diff, 'n': len(arrays['sample_id']),
                    'seconds_compute': time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models-root', required=True, help='carpeta donde buscar los .pt (p. ej. /kaggle/input)')
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'results', 'f37_mapas'))
    ap.add_argument('--maps-dir', default=None, help='default: <out-dir>/maps')
    ap.add_argument('--sources', nargs='+', default=list(SOURCES))
    ap.add_argument('--only', nargs='+', default=None, help='técnicas (p. ej. perlin)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None, help='N muestras por split (prueba de humo)')
    ap.add_argument('--tol-pdep', type=float, default=1e-4)
    a = ap.parse_args(argv)

    import torch
    import config
    from train_model import configure_determinism
    from umap_io import sha256
    from src.predictions.write import environment, git_commit
    from src.training.run_grid import load_sample_index
    from src import paths

    matrix = config.load_matrix()
    det = configure_determinism(config.determinism(matrix))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smoke = a.limit is not None
    maps_dir = a.maps_dir or os.path.join(a.out_dir, 'maps')
    rec_dir = os.path.join(a.out_dir, 'records')
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    meta_path = os.path.join(a.out_dir, 'maps_meta.json')
    meta = {'task': 'F3.7', 'smoke': smoke, 'limit': a.limit}
    if os.path.exists(meta_path) and json.load(open(meta_path)) != meta:
        raise SystemExit(f'{a.out_dir} ya tiene salidas con otra configuración ({json.load(open(meta_path))}); usa otro --out-dir')
    write_json_atomic(meta_path, meta)

    runs = list_runs(a.sources, a.only)
    found = find_models(a.models_root)
    missing = [d['model_file']['path'] for _, d in runs if d['model_file']['path'] not in found]
    if missing:
        raise SystemExit(f'faltan {len(missing)} modelos bajo {a.models_root}: {missing[:5]}')
    index = load_sample_index(str(paths.SAMPLE_INDEX))
    expected = {s: [r[1] for r in index[s]] for s in paths.SPLITS}
    env = environment()
    env.update({'torch': torch.__version__, 'device': str(device),
                'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else None})
    commit = git_commit()
    print(f'{len(runs)} corridas | dispositivo {device} | commit {commit} | smoke={smoke}', flush=True)
    matrix_ref = {'path': 'configs/matriz_experimentos.json', 'sha256': sha256(config.MATRIX_PATH)}

    cache, failures = {}, []
    for i, (run_rel, d) in enumerate(runs, 1):
        name = os.path.splitext(d['model_file']['path'])[0]
        rec_path = os.path.join(rec_dir, name + '.json')
        npz_path = os.path.join(maps_dir, name + '.npz')
        if os.path.exists(rec_path) and os.path.exists(npz_path) and \
                json.load(open(rec_path))['npz']['sha256'] == sha256(npz_path):
            print(f'[{i}/{len(runs)}] {name}: ya hecho', flush=True)
            continue
        t0 = time.time()
        try:
            arrays, info = process_run(run_rel, d, found[d['model_file']['path']], cache, expected, matrix,
                                       device, a.batch_size, a.limit, a.tol_pdep)
            np.savez_compressed(npz_path, **arrays)
            rec = {'task': 'F3.7', 'smoke': smoke, 'run': name, 'run_json': run_rel, 'fase': d['fase'],
                   'technique': d['technique'], 'arch': d['arch'], 'strategy': d['strategy'], 'seed': d['seed'],
                   'permute_labels': d.get('permute_labels'), **info,
                   'target_logit_class': matrix['interpretabilidad']['target']['logit_class'],
                   'npz': {'path': os.path.relpath(npz_path, REPO).replace(os.sep, '/'), 'sha256': sha256(npz_path),
                           'bytes': os.path.getsize(npz_path)},
                   'matrix_file': matrix_ref, 'determinism': det, 'environment': env, 'git_commit': commit,
                   'batch_size': a.batch_size, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   'seconds': time.time() - t0}
            write_json_atomic(rec_path, rec)
            print(f"[{i}/{len(runs)}] {name}: n={info['n']} Δp_dep={max(info['p_dep_max_abs_diff'].values()):.2e} "
                  f"Δlogit={info['logits_hooked_max_abs_diff']:.2e} {rec['seconds']:.0f}s "
                  f"{rec['npz']['bytes'] / 2**20:.1f} MiB", flush=True)
        except Exception as e:   # una corrida rota no detiene las demás; se reporta al final
            failures.append(f'{name}: {type(e).__name__}: {e}')
            print(f'[{i}/{len(runs)}] {name}: FALLA {type(e).__name__}: {e}', flush=True)
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    rows = []
    for p in sorted(glob.glob(os.path.join(rec_dir, '*.json'))):
        r = json.load(open(p))
        rows.append({'run': r['run'], 'fase': r['fase'], 'technique': r['technique'], 'arch': r['arch'],
                     'strategy': r['strategy'], 'seed': r['seed'], 'n': r['n'], 'methods': '+'.join(r['methods'].values()),
                     'npz': r['npz']['path'], 'bytes': r['npz']['bytes'], 'sha256': r['npz']['sha256']})
    if rows:
        with open(os.path.join(a.out_dir, 'maps_manifest.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    print(f'\n{len(rows)} mapas registrados, {len(failures)} fallas')
    for fl in failures:
        print('  ' + fl)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
