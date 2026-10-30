"""
f14_recuperabilidad.py
───────────────────────
Pruebas P1-P4 de F1.4 (docs/decisiones_preprocesamiento.md, secciones 3-5 y
precisiones operativas de 6.1). Solo las 908 imagenes de train; el objetivo es
u (umap_params.pkl), nunca la etiqueta: y_true no se lee y no se abre ninguna
imagen de val ni de test.

  P1  R2 por componente de ridge sobre features ResNet-18 (ImageNet, congelada)
      para V0 (256 nativo), V1 (resize 224 de data.py) y V2 (recorte central 224);
      dR2(V) = R2(V0) - R2(V).
  P2  mismos regresores, particion retenida volteada completa (p=1);
      dR2(F) = R2(V) - R2(F(V)).
  P3  fraccion de energia espectral con max(|fx|,|fy|) > 112 ciclos/imagen vs u_j
      (Spearman, explicativa).
  P4  centroide horizontal y skewness ponderada de x vs u_j (Spearman, Holm 30).

Al final aplica la regla de la seccion 5 tal como esta escrita y escribe
decision.csv.

Uso (desde la raiz del repo, TT_DATA_ROOT definida, CPU):
    .venv/Scripts/python.exe results/f14_preprocesamiento/f14_recuperabilidad.py
"""

import csv
import hashlib
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import torch
import torchvision
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'src', 'training'))
import paths  # noqa: E402
from data import IMAGENET_MEAN, IMAGENET_STD, get_transforms  # noqa: E402
from umap_io import load_umap_arrays, sha256  # noqa: E402

SEED = 2026
N_SPLITS = 5
ALPHAS = np.logspace(-3, 3, 13)
B = 2000
THRESHOLD = 0.02          # regla 5.1 y 5.3
ALPHA_HOLM = 0.05
NYQ_224 = 112             # ciclos por imagen
TECHNIQUES = ['perlin', 'gray_scott', 'lorenz']
VARIANTS = ['V0', 'V1', 'V2']
PREFERENCE = ['V1', 'V2', 'V0']   # regla 5.1
N_COMP = 5
BATCH = 64

LOG_LINES = []


def log(msg=''):
    print(msg, flush=True)
    LOG_LINES.append(msg)


# ─────────────────────────────────────────────────────────────
# Transformaciones (importadas de src/training/data.py)
# ─────────────────────────────────────────────────────────────

def _flip_p1(tf):
    """Copia de la cadena con el RandomHorizontalFlip de data.py forzado a p=1
    (precision 3 de 6.1: la particion retenida se voltea completa)."""
    steps = list(tf.transforms)
    flips = [i for i, s in enumerate(steps) if isinstance(s, transforms.RandomHorizontalFlip)]
    assert len(flips) == 1, f'se esperaba un RandomHorizontalFlip en {tf}'
    steps[flips[0]] = transforms.RandomHorizontalFlip(p=1.0)
    return transforms.Compose(steps)


def build_transforms():
    """{V: (transform sin aumentar, transform volteado)}. V0 y V1 son exactamente
    las de data.py (eval y train con hflip); V2 no existe en el codigo y se arma
    con las mismas constantes, con el volteo en la misma posicion."""
    v2 = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    v2_flip = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.CenterCrop(224),
                                  transforms.RandomHorizontalFlip(p=1.0),
                                  transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return {
        'V0': (get_transforms(False, 256), _flip_p1(get_transforms(True, 256, hflip=True))),
        'V1': (get_transforms(False, 224), _flip_p1(get_transforms(True, 224, hflip=True))),
        'V2': (v2, v2_flip),
    }


# ─────────────────────────────────────────────────────────────
# Insumos
# ─────────────────────────────────────────────────────────────

def load_inputs():
    manifest = pd.read_csv(paths.DATA_MANIFEST)
    umap_path = paths.umap_params_path()
    umap_sha = sha256(umap_path)
    expected = manifest.loc[manifest.logical_name == 'UMAP_PARAMS', 'sha256'].item()
    assert umap_sha == expected, f'SHA-256 de umap_params.pkl no coincide: {umap_sha} != {expected}'
    log(f'umap_params.pkl sha256 {umap_sha} (coincide con meta/data_manifest.csv)')
    u = load_umap_arrays(umap_path)['train']['params'].astype(np.float64)

    # Solo columnas sin etiqueta: y_true no se lee
    cols = ['sample_id', 'split', 'idx'] + [f'img_{t}' for t in TECHNIQUES]
    idx = pd.read_csv(paths.SAMPLE_INDEX, usecols=cols)
    idx = idx[idx.split == 'train'].sort_values('idx').reset_index(drop=True)
    assert len(idx) == 908 and (idx.idx.values == np.arange(908)).all()
    assert u.shape == (908, N_COMP)
    file_sha = {r.rel_path: r.sha256 for r in manifest.itertuples() if r.kind == 'file'}
    return u, idx, file_sha, umap_sha


def load_images(technique, names, file_sha):
    assert all('_train_' in n and '_val_' not in n and '_test_' not in n for n in names)
    folder = paths.images_dir(technique)
    rel_folder = paths.REL['IMAGES'][technique]
    imgs = []
    for n in names:
        p = folder / n
        with open(p, 'rb') as f:
            raw = f.read()
        h = hashlib.sha256(raw).hexdigest()
        assert h == file_sha[f'{rel_folder}/{n}'], f'SHA-256 no coincide: {p}'
        img = Image.open(p)
        img.load()
        assert img.size == (256, 256) and img.mode == 'L', (p, img.size, img.mode)
        imgs.append(img.convert('L'))
    log(f'  {technique}: {len(imgs)} imagenes de train leidas, SHA-256 verificado contra el manifiesto')
    return imgs


# ─────────────────────────────────────────────────────────────
# P1 / P2
# ─────────────────────────────────────────────────────────────

def build_extractor():
    torch.manual_seed(SEED)
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    return m


@torch.no_grad()
def extract(model, imgs, tf):
    out = []
    for i in range(0, len(imgs), BATCH):
        x = torch.stack([tf(im) for im in imgs[i:i + BATCH]])
        out.append(model(x).numpy().astype(np.float64))
    feats = np.concatenate(out)
    assert feats.shape == (len(imgs), 512)
    return feats


def oof_predictions(X, X_eval_sets, u):
    """Ajusta un ridge por componente en cada particion con X y predice la
    particion retenida en X y en cada X_eval_sets[k]. Devuelve
    (oof (n,5), {k: oof (n,5)}, alphas elegidos (folds,5))."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros_like(u)
    oof_eval = {k: np.zeros_like(u) for k in X_eval_sets}
    alphas = np.zeros((N_SPLITS, N_COMP))
    for f, (tr, te) in enumerate(kf.split(X)):
        for j in range(N_COMP):
            reg = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            reg.fit(X[tr], u[tr, j])
            alphas[f, j] = reg[-1].alpha_
            oof[te, j] = reg.predict(X[te])
            for k, Xe in X_eval_sets.items():
                oof_eval[k][te, j] = reg.predict(Xe[te])
    return oof, oof_eval, alphas


def r2(y, p):
    """R2 por columna; acepta indices de bootstrap en la primera dimension."""
    ss_res = ((y - p) ** 2).sum(axis=-2)
    ss_tot = ((y - y.mean(axis=-2, keepdims=True)) ** 2).sum(axis=-2)
    return 1 - ss_res / ss_tot


def boot_indices():
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 908, size=(B, 908))


def delta_ci(u, pa, pb, bidx):
    """IC 95 % de R2(pa) - R2(pb) por bootstrap pareado (mismos indices)."""
    d = r2(u[bidx], pa[bidx]) - r2(u[bidx], pb[bidx])   # (B, 5)
    return np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


# ─────────────────────────────────────────────────────────────
# P3 / P4
# ─────────────────────────────────────────────────────────────

def high_freq_fraction(a):
    """Fraccion de |FFT2|^2 con max(|fx|,|fy|) > 112 ciclos/imagen, sin DC."""
    n = a.shape[0]
    P = np.abs(np.fft.fft2(a)) ** 2
    f = np.abs(np.fft.fftfreq(n, d=1.0 / n))       # ciclos por imagen
    fmax = np.maximum(f[:, None], f[None, :])
    P[0, 0] = 0.0
    total = P.sum()
    return float(P[fmax > NYQ_224].sum() / total) if total > 0 else 0.0


def orientation_stats(a):
    """(desplazamiento del centroide horizontal, skewness ponderada de x)."""
    w = a.sum(axis=0)                     # perfil por columnas p(x)
    x = np.arange(a.shape[1], dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return 0.0, 0.0
    mu = (w * x).sum() / s
    var = (w * (x - mu) ** 2).sum() / s
    skew = (w * (x - mu) ** 3).sum() / s / var ** 1.5 if var > 0 else 0.0
    return float(mu - (a.shape[1] - 1) / 2), float(skew)


# ─────────────────────────────────────────────────────────────
# Regla de decision (seccion 5, literal)
# ─────────────────────────────────────────────────────────────

def decide(tech, dr2_size, dr2_flip, p4_sig):
    admitted = [v for v in PREFERENCE if (dr2_size[v] <= THRESHOLD).all()]
    size = admitted[0]                    # V0 siempre se admite (dR2 = 0)
    if size in ('V1', 'V2'):
        vit = size
    else:
        vit = 'V0 (256 px, positional embeddings interpolados; sujeto a confirmacion de CE, regla 5.2)'
    flip = bool((dr2_flip[size] <= THRESHOLD).all() and not p4_sig)
    return {'technique': tech, 'admitidas': ' '.join(admitted), 'tamano': size, 'volteo': flip, 'vit': vit}


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log('F1.4 - pruebas P1-P4 (docs/decisiones_preprocesamiento.md)')
    paths.assert_outside_data_root(HERE)
    commit = subprocess.check_output(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    log(f'commit: {commit}')
    log(f'python {platform.python_version()} | torch {torch.__version__} | torchvision {torchvision.__version__} | '
        f'numpy {np.__version__} | scipy {scipy.__version__} | sklearn {sklearn.__version__} | '
        f'statsmodels {statsmodels.__version__} | pandas {pd.__version__} | {platform.platform()}')

    u, idx, file_sha, _ = load_inputs()
    tfs = build_transforms()
    model = build_extractor()
    bidx = boot_indices()

    r2_rows, flip_rows, spec_rows, orient_rows, decisions = [], [], [], [], []
    p4_raw = []   # (tech, stat, comp, rho, p)

    for tech in TECHNIQUES:
        log(f'\n== {tech} ==')
        imgs = load_images(tech, idx[f'img_{tech}'].tolist(), file_sha)

        feats, feats_flip = {}, {}
        for v in VARIANTS:
            feats[v] = extract(model, imgs, tfs[v][0])
            feats_flip[v] = extract(model, imgs, tfs[v][1])
            log(f'  features {v} y F({v}) extraidas')

        oof, oof_flip = {}, {}
        for v in VARIANTS:
            oof[v], ev, alphas = oof_predictions(feats[v], {'F': feats_flip[v]}, u)
            oof_flip[v] = ev['F']
            log(f'  {v}: alphas elegidos por componente (mediana entre particiones) '
                f'{np.median(alphas, axis=0).tolist()}')

        R2 = {v: r2(u, oof[v]) for v in VARIANTS}
        R2F = {v: r2(u, oof_flip[v]) for v in VARIANTS}
        dr2_size, dr2_flip = {}, {}
        for v in VARIANTS:
            dr2_size[v] = R2['V0'] - R2[v]
            lo, hi = delta_ci(u, oof['V0'], oof[v], bidx)
            dr2_flip[v] = R2[v] - R2F[v]
            flo, fhi = delta_ci(u, oof[v], oof_flip[v], bidx)
            for j in range(N_COMP):
                r2_rows.append({'technique': tech, 'variant': v, 'component': f'u{j}', 'R2': R2[v][j],
                                'dR2': dr2_size[v][j], 'ic_lo': lo[j], 'ic_hi': hi[j],
                                'admitida_j': bool(dr2_size[v][j] <= THRESHOLD)})
                flip_rows.append({'technique': tech, 'variant': v, 'component': f'u{j}', 'R2_V': R2[v][j],
                                  'R2_FV': R2F[v][j], 'dR2_F': dr2_flip[v][j], 'ic_lo': flo[j],
                                  'ic_hi': fhi[j], 'cumple_j': bool(dr2_flip[v][j] <= THRESHOLD)})
            log(f'  {v}: R2={np.round(R2[v], 4).tolist()} dR2={np.round(dr2_size[v], 4).tolist()} '
                f'dR2_F={np.round(dr2_flip[v], 4).tolist()}')
        assert np.all(dr2_size['V0'] == 0)

        arrs = [np.asarray(im, dtype=np.float64) / 255.0 for im in imgs]
        hf = np.array([high_freq_fraction(a) for a in arrs])
        ori = np.array([orientation_stats(a) for a in arrs])
        for j in range(N_COMP):
            rho, p = spearmanr(hf, u[:, j])
            spec_rows.append({'technique': tech, 'component': f'u{j}', 'hf_mean': hf.mean(),
                              'hf_median': np.median(hf), 'hf_max': hf.max(), 'rho': rho, 'p': p})
            for s, name in enumerate(('centroide_x', 'skew_x')):
                rho, p = spearmanr(ori[:, s], u[:, j])
                p4_raw.append((tech, name, f'u{j}', rho, p))
        log(f'  P3: fraccion de energia > {NYQ_224} ciclos: media={hf.mean():.3e} max={hf.max():.3e}')
        decisions.append((tech, dr2_size, dr2_flip))

    # P4: Holm sobre los 30 contrastes
    reject, p_holm, _, _ = multipletests([r[4] for r in p4_raw], alpha=ALPHA_HOLM, method='holm')
    for (tech, name, comp, rho, p), ph, rj in zip(p4_raw, p_holm, reject):
        orient_rows.append({'technique': tech, 'statistic': name, 'component': comp, 'rho': rho, 'p': p,
                            'p_holm': ph, 'significativo': bool(rj)})
    assert len(orient_rows) == 30

    log('\n== Decision (regla de la seccion 5) ==')
    dec_rows = []
    for tech, dr2_size, dr2_flip in decisions:
        sig = any(r['significativo'] for r in orient_rows if r['technique'] == tech)
        d = decide(tech, dr2_size, dr2_flip, sig)
        d['p4_significativo'] = sig
        dec_rows.append(d)
        log(f'  {tech}: admitidas={d["admitidas"]} tamano={d["tamano"]} volteo={d["volteo"]} '
            f'(P4 significativo={sig}) vit={d["vit"]}')

    def write(name, rows):
        with open(os.path.join(HERE, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f'{v:.6g}' if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})

    write('r2_por_componente.csv', r2_rows)
    write('sensibilidad_volteo.csv', flip_rows)
    write('espectro.csv', spec_rows)
    write('orientacion.csv', orient_rows)
    write('decision.csv', dec_rows)
    log(f'\nTiempo total: {(time.time() - t0) / 60:.1f} min')
    with open(os.path.join(HERE, 'f14.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG_LINES) + '\n')


if __name__ == '__main__':
    main()
