"""
data.py
────────
Preprocesamiento y carga compartidos por las 3 arquitecturas (ResNet-18,
EfficientNet-B0, ViT-B-16), para que los 18 experimentos usen exactamente
la misma entrada.

Preprocesamiento por defecto (el mismo del camera-ready, ver
docs/resumen_modulo3_camera_ready.md):
  - PNG 256x256 modo 'L' -> replicada a 3 canales (R=G=B, sin color nuevo)
  - Resize a 224x224 (ViT-B-16 de torchvision EXIGE 224; con 256 truena con
    "Wrong image height! Expected 224")
  - Normalizacion estandar de ImageNet
  - Solo en train: RandomHorizontalFlip

Resolucion y volteo son configurables (img_size, hflip) porque el plan v2
(F1.4) pide validar por tecnica que no destruyan informacion de u -- p. ej.
el volteo en Lorenz o el resize 256->224 en los trazos finos de Perlin. La
decision la toma SV; mientras tanto se usan los valores del camera-ready.

Nombres de archivo: <prefijo>_<split>_<idx:04d>_label<0o1>.png (ver
docs/artefactos_datos.md). El label se lee del nombre del archivo; si se
pasan los labels de umap_params.pkl, se verifica que coincidan 1 a 1.
"""

import os
import re
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Prefijo de archivo por tecnica (igual que expected_path en src/generation/)
PREFIXES = {'gray_scott': 'rd', 'perlin': 'perlin', 'lorenz': 'lorenz'}


def get_transforms(train, img_size=IMG_SIZE, hflip=True):
    """Transformaciones compartidas. train=True y hflip=True agregan
    RandomHorizontalFlip. img_size=256 deja la imagen sin redimensionar."""
    tfs = [transforms.Grayscale(num_output_channels=3)]
    if img_size != 256:
        tfs.append(transforms.Resize((img_size, img_size)))
    if train and hflip:
        tfs.append(transforms.RandomHorizontalFlip())
    tfs += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(tfs)


def list_split_files(image_dir, prefix, split, expected_labels=None):
    """Devuelve (paths, labels) del split, ordenados por idx.

    Verifica que los idx sean exactamente 0..n-1 (sin huecos ni duplicados) y,
    si se da 'expected_labels' (p. ej. umap_params['val']['labels']), que el
    label de cada archivo coincida con el de esa posicion."""
    pattern = re.compile(rf'^{re.escape(prefix)}_{re.escape(split)}_(\d{{4}})_label([01])\.png$')
    found = {}
    for fname in os.listdir(image_dir):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            if idx in found:
                raise ValueError(f'idx duplicado {idx} en {split}: {fname} y {found[idx][0]}')
            found[idx] = (fname, int(m.group(2)))
    n = len(found)
    if n == 0:
        raise FileNotFoundError(f'No hay archivos {prefix}_{split}_*.png en {image_dir}')
    if sorted(found) != list(range(n)):
        missing = sorted(set(range(max(found) + 1)) - set(found))
        raise ValueError(f'Faltan idx en {split}: {missing[:10]}...')

    paths = [os.path.join(image_dir, found[i][0]) for i in range(n)]
    labels = [found[i][1] for i in range(n)]
    if expected_labels is not None:
        expected = [int(l) for l in expected_labels]
        if len(expected) != n:
            raise ValueError(f'{split}: {n} archivos pero {len(expected)} labels esperados')
        bad = [i for i in range(n) if labels[i] != expected[i]]
        if bad:
            raise ValueError(f'{split}: labels del nombre no coinciden con los esperados en idx {bad[:10]}')
    return paths, labels


class ClassBlindImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('L')
        return self.transform(img), self.labels[i]


def compute_class_weights(labels):
    """Pesos 'balanced' (n / (n_clases * conteo_c)), calculados SOLO con train."""
    counts = np.bincount(np.asarray(labels), minlength=2).astype(np.float64)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def permute_labels(labels, seed):
    """Permutacion determinista de etiquetas (control de permutacion, plan v2
    F3.6). Conserva el conteo por clase, asi que los class weights no cambian."""
    return [int(l) for l in np.random.default_rng(seed).permutation(np.asarray(labels))]


def make_loaders(image_dir, technique, seed, batch_size=32, num_workers=2, expected_labels=None,
                 img_size=IMG_SIZE, hflip=True, permute_train_seed=None, with_test=True):
    """Crea train/val/test loaders + class_weights para una tecnica.

    technique: 'gray_scott' | 'perlin' | 'lorenz'
    expected_labels: opcional, dict {'train': [...], 'val': [...], 'test': [...]}
    seed: fija el orden de barajado de train (via torch.Generator). train_model
    vuelve a sembrar este generador al inicio, asi que reusar los loaders entre
    semillas da el mismo orden que crearlos de nuevo.
    permute_train_seed: si no es None, permuta SOLO las etiquetas de train
    (val/test conservan las reales) -- control de permutacion.
    with_test=False: no se construye el loader de test (test_loader=None), para
    la primera pasada del plan v2, que solo infiere sobre validacion.

    Devuelve: train_loader, val_loader, test_loader, class_weights
    """
    prefix = PREFIXES[technique]
    expected_labels = expected_labels or {}
    split_names = ('train', 'val', 'test') if with_test else ('train', 'val')
    splits = {s: list_split_files(image_dir, prefix, s, expected_labels.get(s)) for s in split_names}

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=_seed_worker)

    tr_p, tr_l = splits['train']
    if permute_train_seed is not None:
        tr_l = permute_labels(tr_l, permute_train_seed)
    train_loader = DataLoader(ClassBlindImageDataset(tr_p, tr_l, get_transforms(True, img_size, hflip)),
                              shuffle=True, generator=g, **common)
    val_loader = DataLoader(ClassBlindImageDataset(*splits['val'], get_transforms(False, img_size)),
                            shuffle=False, **common)
    test_loader = None
    if with_test:
        test_loader = DataLoader(ClassBlindImageDataset(*splits['test'], get_transforms(False, img_size)),
                                 shuffle=False, **common)
    return train_loader, val_loader, test_loader, compute_class_weights(tr_l)
