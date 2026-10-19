"""
smoke_test_training.py
───────────────────────
Smoke test de src/training/ (Semana 1, CE): confirma que las 3 arquitecturas
x 2 estrategias cargan, hacen forward y entrenan sin tronar, y que el modo
frozen realmente no toca el backbone. NO mide desempeno.

Uso (CPU basta, ~1-2 min; la primera vez descarga los pesos de ImageNet):
    python tests/smoke_test_training.py
    python tests/smoke_test_training.py --image-dir <carpeta> --technique gray_scott --n 24

Sin --image-dir usa PNGs sinteticos (ruido) con la nomenclatura real
(<prefijo>_<split>_<idx:04d>_label<0o1>.png). Con --image-dir toma las
primeras --n imagenes de train (y n/4 de val/test) del dataset real.
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'training'))
from build_model import build_model, get_head  # noqa: E402
from data import (PREFIXES, ClassBlindImageDataset, compute_class_weights,  # noqa: E402
                  get_transforms, list_split_files)
from train_model import train_model  # noqa: E402

ARCHS = ['resnet18', 'efficientnet_b0', 'vit_b_16']
# Prefijo en state_dict de la capa final reemplazada (ver build_model.get_head)
HEAD_PREFIX = {'resnet18': 'fc.', 'efficientnet_b0': 'classifier.1.', 'vit_b_16': 'heads.head.'}
FAILURES = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAILURES.append(msg)


def write_synthetic(image_dir, prefix, n_per_split):
    rng = np.random.default_rng(0)
    for split, n in n_per_split.items():
        for i in range(n):
            label = i % 2
            img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
            Image.fromarray(img, mode='L').save(
                os.path.join(image_dir, f'{prefix}_{split}_{i:04d}_label{label}.png'))


def test_list_split_files(prefix):
    print('\n[data] validacion de nombres de archivo')
    with tempfile.TemporaryDirectory() as d:
        write_synthetic(d, prefix, {'val': 4})
        paths, labels = list_split_files(d, prefix, 'val', expected_labels=[0, 1, 0, 1])
        check(len(paths) == 4 and labels == [0, 1, 0, 1], 'lee 4 archivos en orden con sus labels')
        try:
            list_split_files(d, prefix, 'val', expected_labels=[1, 1, 0, 1])
            check(False, 'detecta label que no coincide con umap_params')
        except ValueError:
            check(True, 'detecta label que no coincide con umap_params')
        os.remove(os.path.join(d, f'{prefix}_val_0002_label0.png'))
        try:
            list_split_files(d, prefix, 'val')
            check(False, 'detecta idx faltante')
        except ValueError:
            check(True, 'detecta idx faltante')


def make_small_loaders(image_dir, prefix, n_train, batch_size=8):
    loaders = {}
    sizes = {'train': n_train, 'val': max(4, n_train // 4), 'test': max(4, n_train // 4)}
    train_labels = None
    for split, n in sizes.items():
        paths, labels = list_split_files(image_dir, prefix, split)
        paths, labels = paths[:n], labels[:n]
        ds = ClassBlindImageDataset(paths, labels, get_transforms(split == 'train'))
        kw = dict(shuffle=True, generator=torch.Generator().manual_seed(0)) if split == 'train' else {}
        loaders[split] = torch.utils.data.DataLoader(ds, batch_size=batch_size, **kw)
        if split == 'train':
            train_labels = labels
    return loaders, compute_class_weights(train_labels)


def snapshot(model, arch):
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    head = [p.detach().clone() for p in get_head(model, arch).parameters()]
    return state, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-dir', default=None)
    ap.add_argument('--technique', default='gray_scott', choices=list(PREFIXES))
    ap.add_argument('--n', type=int, default=24, help='imagenes de train a usar')
    ap.add_argument('--epochs', type=int, default=2)
    args = ap.parse_args()
    prefix = PREFIXES[args.technique]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')

    test_list_split_files(prefix)

    tmp = None
    image_dir = args.image_dir
    if image_dir is None:
        tmp = tempfile.TemporaryDirectory()
        image_dir = tmp.name
        n_eval = max(4, args.n // 4)
        write_synthetic(image_dir, prefix, {'train': args.n, 'val': n_eval, 'test': n_eval})
        print(f'Usando {args.n}+{n_eval}+{n_eval} imagenes sinteticas')

    loaders, cw = make_small_loaders(image_dir, prefix, args.n)
    x, _ = next(iter(loaders['train']))
    check(tuple(x.shape[1:]) == (3, 224, 224), f'tensor de entrada 3x224x224 (got {tuple(x.shape[1:])})')

    for arch in ARCHS:
        for frozen in (False, True):
            print(f'\n[{arch}, frozen={frozen}]')
            # Mismo seed que usa train_model -> mismo modelo inicial para comparar
            torch.manual_seed(42)
            ref = build_model(arch, freeze_backbone=frozen)
            before_backbone, before_head = snapshot(ref, arch)
            del ref

            model, history, res = train_model(arch, frozen, loaders['train'], loaders['val'],
                                              loaders['test'], cw, device,
                                              n_epochs=args.epochs, patience=args.epochs, seed=42)
            model = model.cpu()
            check(len(res['test_preds']) == len(loaders['test'].dataset), 'test_preds con el tamano del test')
            check(np.isfinite(history['train_loss']).all(), 'loss finita')
            check(res['lr'] == (1e-3 if frozen else 3e-5), f"lr por defecto correcto ({res['lr']:g})")

            after = model.state_dict()
            backbone_changed = [k for k in after if not k.startswith(HEAD_PREFIX[arch])
                                and not torch.equal(after[k], before_backbone[k])]
            head_changed = any(not torch.equal(a, b) for a, b in
                               zip(get_head(model, arch).parameters(), before_head))
            check(head_changed, 'la cabeza se actualizo')
            if frozen:
                check(not backbone_changed,
                      f'backbone intacto, incluidas running stats de BN ({len(backbone_changed)} tensores cambiaron)')
            else:
                check(len(backbone_changed) > 0, 'fine-tuning actualiza el backbone')

    print('\n[determinismo] resnet18 dos veces con seed=42')
    runs = [train_model('resnet18', False, loaders['train'], loaders['val'], loaders['test'],
                        cw, device, n_epochs=1, seed=42)[2] for _ in range(2)]
    check(np.allclose(runs[0]['test_probs'], runs[1]['test_probs'], atol=1e-6),
          'mismas probabilidades de test con la misma semilla')

    if tmp is not None:
        tmp.cleanup()
    print('\n' + ('TODO OK' if not FAILURES else f'{len(FAILURES)} FALLAS:\n  - ' + '\n  - '.join(FAILURES)))
    sys.exit(1 if FAILURES else 0)


if __name__ == '__main__':
    main()
