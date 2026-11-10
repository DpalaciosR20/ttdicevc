"""
caracteristicas.py
──────────────────
F4.1 / F4.7 (docs/plan_fase4.md): extrae las características de la penúltima
capa del backbone de cada finalista para las 1,136 imágenes de cada técnica.
Se ejecuta en Kaggle (GPU).

Solo extrae características: no calcula logits, p_dep ni ninguna predicción
de clase (G2). La cabeza de clasificación se reemplaza por nn.Identity antes
de la pasada, así que el modelo no tiene cómo clasificar.

Backbones (plan, F4.1-1):
  perlin, lorenz  vit_b16_imagenet: ViT-B/16 con los pesos ImageNet de
                  torchvision (IMAGENET1K_V1). Se comprueba, tensor por
                  tensor (torch.equal), que el backbone de cada finalista de
                  F3.5 (las 5 semillas; la 42 es la que exige el plan) es
                  idéntico al de ImageNet: todos los tensores salvo
                  heads.head.*. Si alguno difiere, se detiene.
  gray_scott      efficientnet_b0_finetune_seed<S> para S en 42, 7, 123, 2024,
                  99 (backbone del finalista ajustado; 42 es la principal) y
                  efficientnet_b0_imagenet (sin ajustar; sensibilidad).

Penúltima capa: la entrada de la capa final, es decir, la salida con la cabeza
reemplazada por nn.Identity:
  vit_b_16         token de clase tras el encoder y su LayerNorm (768)
  efficientnet_b0  pooling global tras features (1280); el Dropout del
                   clasificador es la identidad en eval()

Preprocesamiento: el de inferencia de F1.4 que registra el JSON de la corrida
del finalista (preprocessing: tamaño, interpolación, antialias), sin aumentos:
get_transforms(train=False, ...).

Verificaciones (se detiene ante cualquier discrepancia):
  - sha256 de cada .pt contra meta/artefactos_manifest.csv (y contra
    model_file.sha256 del JSON de la corrida);
  - sha256 de cada imagen contra meta/data_manifest.csv y el dir_digest de
    la carpeta contra la fila kind=dir. Los bytes que se decodifican son los
    mismos que se hashean;
  - las 1,136 imágenes de cada técnica en el orden de meta/sample_index.csv.

Salida:
  <features-dir>/<técnica>__<backbone>.npz  (NO se versiona)
      features float32 (1136, D), sample_id (str), split (str)
  <out-dir>/caracteristicas_manifest.csv   técnica, backbone, archivo, n, d,
      sha256 del .npz, modelo y su sha256, dispositivo
  <out-dir>/caracteristicas_meta.json      comprobaciones de identidad,
      digests de imágenes, pesos ImageNet (sha256), entorno, commit
Nunca sobrescribe: si existe un .npz o el manifiesto, se detiene.

Uso (Kaggle):
  python -m src.fase4.caracteristicas --models-root /kaggle/input \
      --features-dir /kaggle/working/f41_caracteristicas \
      --out-dir results/f41_recuperabilidad/caracteristicas
  --limit N: solo las N primeras imágenes de cada técnica (prueba de humo,
  escribe "smoke": true; usar un --features-dir y --out-dir temporales).
"""
import argparse
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

TECHNIQUES = ('gray_scott', 'perlin', 'lorenz')
FINAL_SEEDS = (42, 7, 123, 2024, 99)
FINALIST = {'gray_scott': ('efficientnet_b0', 'finetune'), 'perlin': ('vit_b16', 'frozen'),
            'lorenz': ('vit_b16', 'frozen')}
HEAD_PREFIX = {'vit_b_16': 'heads.head.', 'efficientnet_b0': 'classifier.1.'}
ARTEFACTOS = os.path.join(REPO, 'meta', 'artefactos_manifest.csv')
DATA_MANIFEST = os.path.join(REPO, 'meta', 'data_manifest.csv')
SAMPLE_INDEX = os.path.join(REPO, 'meta', 'sample_index.csv')


class Detener(RuntimeError):
    """La extracción se detiene: una comprobación no se cumple."""


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


# ── Backbones ─────────────────────────────────────────────────────────────
def internal_arch(arch):
    return {'vit_b16': 'vit_b_16'}.get(arch, arch)


def to_feature_extractor(model, arch):
    """Reemplaza la capa final por Identity: la salida es la penúltima capa."""
    import torch.nn as nn
    arch = internal_arch(arch)
    if arch == 'vit_b_16':
        model.heads = nn.Identity()
    elif arch == 'efficientnet_b0':
        model.classifier = nn.Identity()
    else:
        raise Detener(f'arquitectura no prevista en F4.1: {arch}')
    return model.eval()


def imagenet_state(arch):
    """state_dict de torchvision con pesos ImageNet (IMAGENET1K_V1) y el sha256 del archivo de pesos."""
    import torch
    from torchvision import models
    arch = internal_arch(arch)
    weights = {'vit_b_16': models.ViT_B_16_Weights.IMAGENET1K_V1,
               'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1}[arch]
    model = {'vit_b_16': models.vit_b_16, 'efficientnet_b0': models.efficientnet_b0}[arch](weights=weights)
    cache = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(weights.url))
    info = {'weights': str(weights), 'url': weights.url,
            'file_sha256': sha256_file(cache) if os.path.isfile(cache) else None}
    return model, info


def compare_backbone(state_a, state_b, head_prefix):
    """Compara dos state_dict tensor por tensor, salvo la cabeza.

    Devuelve {'n_tensores', 'iguales', 'distintos', 'faltan_en_a', 'faltan_en_b', 'identico'}."""
    import torch
    ka = {k for k in state_a if not k.startswith(head_prefix)}
    kb = {k for k in state_b if not k.startswith(head_prefix)}
    common = sorted(ka & kb)
    distintos = [k for k in common
                 if state_a[k].shape != state_b[k].shape or not torch.equal(state_a[k].cpu(), state_b[k].cpu())]
    out = {'n_tensores': len(common), 'iguales': len(common) - len(distintos), 'distintos': distintos,
           'faltan_en_a': sorted(kb - ka), 'faltan_en_b': sorted(ka - kb)}
    out['identico'] = not distintos and not out['faltan_en_a'] and not out['faltan_en_b']
    return out


# ── Manifiestos ───────────────────────────────────────────────────────────
def load_artefactos(path=ARTEFACTOS):
    with open(path, newline='', encoding='utf-8') as f:
        return {r['logical_name']: r for r in csv.DictReader(f)}


def model_logical(tech, seed):
    arch, strat = FINALIST[tech]
    return f'MODEL[{tech}:{arch}:{strat}:seed{seed}]'


def locate_model(models_root, row):
    from src.verify_artefacts import find_all, locate
    found = find_all(models_root, ('.pt',))
    name = os.path.basename(row['rel_path'])
    p, _, amb = locate(found.get(name, []), row['rel_path'])
    if p is None:
        raise Detener(f'{"ambiguo" if amb else "falta"}: {name} bajo {models_root}')
    return p


def verified_model(models_root, artefactos, tech, seed, run_json):
    row = artefactos.get(model_logical(tech, seed))
    if row is None or row['task'] != 'F3.5':
        raise Detener(f'{model_logical(tech, seed)} no está en {ARTEFACTOS} como F3.5')
    path = locate_model(models_root, row)
    digest = sha256_file(path)
    if digest != row['sha256'] or os.path.getsize(path) != int(row['size_bytes']):
        raise Detener(f'{path}: sha256 o tamaño distintos de artefactos_manifest.csv')
    if digest != run_json['model_file']['sha256']:
        raise Detener(f'{path}: sha256 distinto de model_file.sha256 del JSON de la corrida')
    return path, digest


def load_data_manifest(path=DATA_MANIFEST):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def image_rows(manifest, tech):
    """({rel_path: sha256} de los archivos, fila dir) de IMAGES[tech]."""
    rows = [r for r in manifest if r['logical_name'] == f'IMAGES[{tech}]']
    dirs = [r for r in rows if r['kind'] == 'dir']
    if len(dirs) != 1:
        raise Detener(f'IMAGES[{tech}]: se espera una fila kind=dir en data_manifest.csv')
    return {r['rel_path']: r['sha256'] for r in rows if r['kind'] == 'file'}, dirs[0]


def dir_digest(file_hashes):
    """Mismo cálculo que src.build_manifest.dir_digest: sha256 de "rel\\tsha\\n" ordenado por rel."""
    lines = ''.join(f'{rel}\t{h}\n' for rel, h in sorted(file_hashes.items()))
    return sha256_bytes(lines.encode('utf-8'))


def read_verified_images(image_dir, rel_dir, names, expected):
    """Lee cada imagen una vez, verifica su sha256 y devuelve los bytes y los hashes calculados."""
    data, hashes = [], {}
    for name in names:
        rel = f'{rel_dir}/{name}'
        if rel not in expected:
            raise Detener(f'{rel} no está en data_manifest.csv')
        b = open(os.path.join(image_dir, name), 'rb').read()
        h = sha256_bytes(b)
        if h != expected[rel]:
            raise Detener(f'{rel}: sha256 distinto de data_manifest.csv')
        hashes[rel] = h
        data.append(b)
    return data, hashes


def load_index(path=SAMPLE_INDEX):
    """Filas de sample_index en el orden train, val, test; dentro de cada split por sample_id."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    order = {'train': 0, 'val': 1, 'test': 2}
    rows.sort(key=lambda r: (order[r['split']], r['sample_id']))
    sizes = {s: sum(r['split'] == s for r in rows) for s in order}
    if sizes != {'train': 908, 'val': 114, 'test': 114}:
        raise Detener(f'sample_index.csv: tamaños {sizes}')
    return rows


# ── Extracción ────────────────────────────────────────────────────────────
def extract(model, images_bytes, transform, device, batch_size=32):
    """Penúltima capa de cada imagen (float32). model debe venir de to_feature_extractor."""
    import torch
    from PIL import Image
    feats = []
    model = model.to(device).eval()
    with torch.no_grad():
        for i in range(0, len(images_bytes), batch_size):
            x = torch.stack([transform(Image.open(io.BytesIO(b)).convert('L'))
                             for b in images_bytes[i:i + batch_size]]).to(device)
            feats.append(model(x).float().cpu().numpy())
    out = np.concatenate(feats).astype(np.float32)
    if out.ndim != 2 or not np.isfinite(out).all():
        raise Detener(f'características con forma {out.shape} o valores no finitos')
    return out


def save_npz(path, features, sample_id, split):
    if os.path.exists(path):
        raise Detener(f'{path} ya existe: caracteristicas nunca sobrescribe')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, features=features, sample_id=np.array(sample_id), split=np.array(split))
    os.replace(tmp, path)
    return sha256_file(path)


def run_json_path(tech, seed):
    arch, strat = FINALIST[tech]
    import glob
    hits = glob.glob(os.path.join(REPO, 'results', 'f35_final', tech, 'runs', f'{tech}__{arch}__{strat}__*__seed{seed}.json'))
    if len(hits) != 1:
        raise Detener(f'{tech} semilla {seed}: {len(hits)} JSON de corrida en results/f35_final')
    return hits[0]


def git_commit():
    r = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() or None


def plan_backbones(tech):
    """[(backbone, fuente)]: fuente = ('imagenet', None) o ('finalista', semilla)."""
    if FINALIST[tech][0] == 'vit_b16':
        return [('vit_b16_imagenet', ('imagenet', None))]
    return [(f'efficientnet_b0_finetune_seed{s}', ('finalista', s)) for s in FINAL_SEEDS] + \
           [('efficientnet_b0_imagenet', ('imagenet', None))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models-root', required=True)
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'results', 'f41_recuperabilidad', 'caracteristicas'))
    ap.add_argument('--only', nargs='+', default=list(TECHNIQUES))
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args(argv)

    import torch
    from build_model import build_model
    from data import get_transforms
    from src import paths

    manifest_out = os.path.join(a.out_dir, 'caracteristicas_manifest.csv')
    if os.path.exists(manifest_out):
        raise Detener(f'{manifest_out} ya existe: caracteristicas nunca sobrescribe')
    os.makedirs(a.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    t0 = time.time()
    artefactos, data_manifest, index = load_artefactos(), load_data_manifest(), load_index()
    if a.limit:
        index = index[:a.limit]
    meta = {'task': 'F4.1', 'smoke': bool(a.limit), 'limit': a.limit, 'git_commit': git_commit(),
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'device': device,
            'torch': torch.__version__, 'python': platform.python_version(),
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'g2': 'sin logits ni p_dep: la cabeza se reemplaza por Identity', 'tecnicas': {}}
    rows_out = []
    for tech in a.only:
        tt = time.time()
        arch, strat = FINALIST[tech]
        runs = {s: json.load(open(run_json_path(tech, s), encoding='utf-8')) for s in FINAL_SEEDS}
        prep = runs[42]['preprocessing']
        if any(r['preprocessing'] != prep for r in runs.values()):
            raise Detener(f'{tech}: preprocesamiento distinto entre semillas')
        tf = get_transforms(False, img_size=prep['size'], hflip=False, interpolation=prep['interpolation'],
                            antialias=prep['antialias'])
        rel_dir = paths.REL['IMAGES'][tech]
        expected, dir_row = image_rows(data_manifest, tech)
        names = [r[f'img_{tech}'] for r in index]
        imgs, hashes = read_verified_images(str(paths.images_dir(tech)), rel_dir, names, expected)
        info = {'arch': arch, 'preprocessing': prep, 'n_imagenes': len(imgs)}
        if not a.limit:
            if len(hashes) != int(dir_row['n_files']) or dir_digest(hashes) != dir_row['sha256']:
                raise Detener(f'IMAGES[{tech}]: dir_digest distinto de data_manifest.csv')
            info['dir_digest'] = dir_row['sha256']
        sample_id = [r['sample_id'] for r in index]
        split = [r['split'] for r in index]
        ref_model, ref_info = imagenet_state(arch)
        info['imagenet'] = ref_info
        ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}
        models_info = {}
        for s in FINAL_SEEDS:
            path, digest = verified_model(a.models_root, artefactos, tech, s, runs[s])
            st = torch.load(path, map_location='cpu', weights_only=True)
            entry = {'archivo': os.path.basename(path), 'sha256': digest}
            if strat == 'frozen':
                cmp = compare_backbone(st, ref_state, HEAD_PREFIX[internal_arch(arch)])
                entry['backbone_vs_imagenet'] = {k: v for k, v in cmp.items() if k != 'distintos'} | \
                    {'distintos': cmp['distintos'][:20]}
                if not cmp['identico']:
                    raise Detener(f'{tech} semilla {s}: el backbone congelado difiere del de ImageNet '
                                  f'({len(cmp["distintos"])} tensores)')
            # Solo el ajuste fino necesita el state_dict después; el congelado ya se comparó
            models_info[str(s)] = {'state': st if strat == 'finetune' else None, **entry}
            del st
        info['modelos'] = {s: {k: v for k, v in m.items() if k != 'state'} for s, m in models_info.items()}
        for backbone, (kind, seed) in plan_backbones(tech):
            if kind == 'imagenet':
                model, model_name, model_sha = ref_model, ref_info['weights'], ref_info['file_sha256']
            else:
                model = build_model(arch, freeze_backbone=strat == 'frozen', pretrained=False)
                model.load_state_dict(models_info[str(seed)]['state'], strict=True)
                model_name, model_sha = models_info[str(seed)]['archivo'], models_info[str(seed)]['sha256']
            fe = to_feature_extractor(model, arch)
            tb = time.time()
            X = extract(fe, imgs, tf, device, a.batch_size)
            npz = os.path.join(a.features_dir, f'{tech}__{backbone}.npz')
            digest = save_npz(npz, X, sample_id, split)
            rows_out.append({'tecnica': tech, 'backbone': backbone, 'archivo': os.path.basename(npz),
                             'n': X.shape[0], 'd': X.shape[1], 'sha256': digest, 'bytes': os.path.getsize(npz),
                             'modelo': model_name, 'modelo_sha256': model_sha, 'dispositivo': device,
                             'segundos': round(time.time() - tb, 1)})
            print(f'{tech:<11} {backbone:<32} {X.shape}  {time.time() - tb:6.1f} s  {digest[:12]}', flush=True)
        meta['tecnicas'][tech] = info | {'segundos': round(time.time() - tt, 1)}
        del models_info, ref_model, ref_state
    meta['segundos_total'] = round(time.time() - t0, 1)
    with open(manifest_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows_out)
    with open(os.path.join(a.out_dir, 'caracteristicas_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f'{len(rows_out)} archivos de características; {meta["segundos_total"]} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
