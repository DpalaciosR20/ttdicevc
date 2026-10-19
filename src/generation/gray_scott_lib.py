"""
gray_scott_lib.py
──────────────────
Generación class-blind de imágenes Gray-Scott (reacción-difusión) a partir
de los 5 componentes UMAP normalizados. NINGUNA función de este archivo
recibe la etiqueta (label) como entrada para el proceso de simulación —
el label solo se usa para nombrar el archivo de salida.

Contrato técnico:
  - Du = 0.16 fijo
  - f  in [0.022, 0.046]
  - k  in [0.059, 0.066]
  - Dv in [0.040, 0.080]
  - radio de semilla in [3, 18] px
  - n_iter in [2000, 7000]  (el paper dice 9000; el código real usa 7000 -- ver
    docs/hallazgos_metodologicos.md)
  - Imagen 256x256, escala de grises (1 canal)

Incluye dos rutas de ejecución:
  - CPU (numpy puro): mas lenta (~5-9s/imagen), no requiere GPU.
  - GPU (PyTorch, batched): mucho mas rapida, usar cuando haya GPU disponible.
"""

import os
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Mapeo UMAP -> parametros fisicos (identico en ambas rutas)
# ─────────────────────────────────────────────────────────────

def umap_to_rd_params(umap_vec):
    """5 valores UMAP normalizados [0,1] -> parametros fisicos de Gray-Scott."""
    u0, u1, u2, u3, u4 = umap_vec
    f = 0.022 + u0 * (0.046 - 0.022)
    k = 0.059 + u1 * (0.066 - 0.059)
    Dv = 0.04 + u2 * (0.08 - 0.04)
    seed_radius = int(3 + u3 * (18 - 3))
    n_iter = int(2000 + u4 * (7000 - 2000))
    return dict(f=f, k=k, Du=0.16, Dv=Dv, seed_radius=seed_radius, n_iter=n_iter)


def expected_path(idx, label, tag, output_dir):
    """Nombre de archivo esperado -- usar SIEMPRE esta funcion para construir
    rutas de lectura/escritura (evita el bug de prefijo inconsistente que se
    encontro durante el camera-ready)."""
    return os.path.join(output_dir, f"rd_{tag}_{idx:04d}_label{label}.png")


# ─────────────────────────────────────────────────────────────
# Ruta CPU (numpy puro) -- referencia / fallback sin GPU
# ─────────────────────────────────────────────────────────────

def gray_scott_step(U, V, Du, Dv, f, k, dt=1.0):
    lap_U = (np.roll(U, 1, 0) + np.roll(U, -1, 0) + np.roll(U, 1, 1) + np.roll(U, -1, 1) - 4 * U)
    lap_V = (np.roll(V, 1, 0) + np.roll(V, -1, 0) + np.roll(V, 1, 1) + np.roll(V, -1, 1) - 4 * V)
    uvv = U * V * V
    return U + dt * (Du * lap_U - uvv + f * (1 - U)), V + dt * (Dv * lap_V + uvv - (f + k) * V)


def run_gray_scott(f, k, size=256, n_iter=3000, Du=0.16, Dv=0.08, seed_radius=5, dt=1.0):
    U = np.ones((size, size))
    V = np.zeros((size, size))
    cx, cy = size // 2, size // 2
    r = seed_radius
    U[cx - r:cx + r, cy - r:cy + r] = 0.50
    V[cx - r:cx + r, cy - r:cy + r] = 0.25
    rng = np.random.default_rng(42)  # semilla FIJA, igual para todas las muestras -- no depende de label ni indice
    U += rng.uniform(-0.01, 0.01, U.shape)
    V += rng.uniform(-0.01, 0.01, V.shape)
    U, V = np.clip(U, 0, 1), np.clip(V, 0, 1)
    for _ in range(n_iter):
        U, V = gray_scott_step(U, V, Du, Dv, f, k, dt)
    return V


def run_gray_scott_safe(f, k, Du, Dv, seed_radius, n_iter, size=256, max_retries=3):
    """Wrapper con reintentos: si la varianza espacial es muy baja (patron
    trivial), ajusta Dv y el radio de semilla de forma deterministica y
    reintenta, hasta max_retries veces."""
    for intento in range(max_retries):
        V = run_gray_scott(f=f, k=k, Du=Du, Dv=Dv, seed_radius=seed_radius, n_iter=n_iter, size=size)
        if V.var() > 1e-4:
            return V, intento
        Dv = max(0.04, Dv * 0.85)
        seed_radius = min(25, seed_radius + 4)
    V = run_gray_scott(f=0.035, k=0.061, Du=0.16, Dv=0.06, seed_radius=10, n_iter=5000, size=size)
    return V, max_retries


def generate_and_save_grayscale(idx, umap_vec, label, split_name, output_dir):
    """Genera UNA imagen (ruta CPU) y la guarda en escala de grises real.
    Recibe 'label' SOLO para nombrar el archivo -- nunca entra al render."""
    os.makedirs(output_dir, exist_ok=True)
    params = umap_to_rd_params(umap_vec)                       # <- SOLO umap_vec
    V, reintentos = run_gray_scott_safe(**params, size=256)     # <- SOLO parametros fisicos
    V_norm = (V - V.min()) / (V.max() - V.min() + 1e-8)
    gray = (V_norm * 255).astype(np.uint8)
    fpath = expected_path(idx, label, split_name, output_dir)
    Image.fromarray(gray).convert('L').save(fpath)
    return fpath, reintentos, params


def generate_batch_cpu(coords_5d, labels, tag, output_dir):
    """Genera un lote de imagenes por la ruta CPU (lenta, ~5-9s/imagen).
    Usar solo si no hay GPU disponible."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(len(labels)):
        label = int(labels[i])
        fpath = expected_path(i, label, tag, output_dir)
        if os.path.exists(fpath):
            paths.append(fpath)
            continue
        fpath, _, _ = generate_and_save_grayscale(i, coords_5d[i], label, tag, output_dir)
        paths.append(fpath)
    return paths


# ─────────────────────────────────────────────────────────────
# Ruta GPU (PyTorch, batched) -- usar cuando haya GPU disponible
# ─────────────────────────────────────────────────────────────
# Requiere `import torch` en el entorno donde se use este modulo.

_NOISE_CACHE = {}


def _fixed_noise(size, device):
    import torch
    key = (size, str(device))
    if key not in _NOISE_CACHE:
        rng = np.random.default_rng(42)
        noise_U = rng.uniform(-0.01, 0.01, (size, size)).astype(np.float32)
        noise_V = rng.uniform(-0.01, 0.01, (size, size)).astype(np.float32)
        _NOISE_CACHE[key] = (
            torch.from_numpy(noise_U).to(device),
            torch.from_numpy(noise_V).to(device),
        )
    return _NOISE_CACHE[key]


def _simulate_batch_gpu(f, k, Du, Dv, seed_radius, n_iter, size, device):
    import torch
    B = len(f)
    f_t = torch.tensor(f, dtype=torch.float32, device=device).view(B, 1, 1)
    k_t = torch.tensor(k, dtype=torch.float32, device=device).view(B, 1, 1)
    Du_t = torch.tensor(Du, dtype=torch.float32, device=device).view(B, 1, 1)
    Dv_t = torch.tensor(Dv, dtype=torch.float32, device=device).view(B, 1, 1)
    n_iter_t = torch.tensor(n_iter, dtype=torch.int64, device=device)

    noise_U, noise_V = _fixed_noise(size, device)
    U = torch.ones((B, size, size), dtype=torch.float32, device=device)
    V = torch.zeros((B, size, size), dtype=torch.float32, device=device)
    cx = cy = size // 2
    for i in range(B):
        r = int(seed_radius[i])
        U[i, cx - r:cx + r, cy - r:cy + r] = 0.50
        V[i, cx - r:cx + r, cy - r:cy + r] = 0.25
    U = (U + noise_U.unsqueeze(0)).clamp(0, 1)
    V = (V + noise_V.unsqueeze(0)).clamp(0, 1)

    max_iter = int(n_iter_t.max().item())
    for step in range(max_iter):
        lap_U = (torch.roll(U, 1, 1) + torch.roll(U, -1, 1) + torch.roll(U, 1, 2) + torch.roll(U, -1, 2) - 4 * U)
        lap_V = (torch.roll(V, 1, 1) + torch.roll(V, -1, 1) + torch.roll(V, 1, 2) + torch.roll(V, -1, 2) - 4 * V)
        uvv = U * V * V
        newU = U + (Du_t * lap_U - uvv + f_t * (1 - U))
        newV = V + (Dv_t * lap_V + uvv - (f_t + k_t) * V)
        active = (step < n_iter_t).view(B, 1, 1)
        U = torch.where(active, newU, U)
        V = torch.where(active, newV, V)
    return V


def _save_batch(V, indices, labels, tag, output_dir):
    Vmin = V.amin(dim=(1, 2), keepdim=True)
    Vmax = V.amax(dim=(1, 2), keepdim=True)
    V_norm = (V - Vmin) / (Vmax - Vmin + 1e-8)
    gray = (V_norm * 255).clamp(0, 255).to(__import__('torch').uint8).cpu().numpy()
    saved_paths = {}
    for j, idx in enumerate(indices):
        label = int(labels[j])
        fpath = expected_path(idx, label, tag, output_dir)
        Image.fromarray(gray[j]).convert('L').save(fpath)
        saved_paths[idx] = fpath
    return saved_paths


def generate_batch_gpu(coords_5d, labels, tag, output_dir, device, chunk_size=1536, max_retries=3, size=256):
    """Genera un lote de imagenes en GPU, por chunks. Mucho mas rapido que
    generate_batch_cpu -- usar esta version siempre que haya GPU disponible.
    Reanudable: si un archivo ya existe, se salta."""
    import torch  # noqa: F401 (solo para confirmar que esta disponible)
    os.makedirs(output_dir, exist_ok=True)
    n = len(labels)
    paths = [None] * n
    pending, n_skipped = [], 0
    for i in range(n):
        fpath = expected_path(i, int(labels[i]), tag, output_dir)
        if os.path.exists(fpath):
            paths[i] = fpath
            n_skipped += 1
        else:
            pending.append(i)

    for start in range(0, len(pending), chunk_size):
        chunk_idx = pending[start:start + chunk_size]
        params = [umap_to_rd_params(coords_5d[i]) for i in chunk_idx]
        f = [p['f'] for p in params]
        k = [p['k'] for p in params]
        Du = [p['Du'] for p in params]
        Dv = [p['Dv'] for p in params]
        seed_radius = [p['seed_radius'] for p in params]
        n_iter = [p['n_iter'] for p in params]

        active_local = list(range(len(chunk_idx)))
        final_V = [None] * len(chunk_idx)
        for intento in range(max_retries):
            if not active_local:
                break
            sub_f = [f[j] for j in active_local]
            sub_k = [k[j] for j in active_local]
            sub_Du = [Du[j] for j in active_local]
            sub_Dv = [Dv[j] for j in active_local]
            sub_r = [seed_radius[j] for j in active_local]
            sub_n = [n_iter[j] for j in active_local]
            V = _simulate_batch_gpu(sub_f, sub_k, sub_Du, sub_Dv, sub_r, sub_n, size, device)
            var = V.var(dim=(1, 2)).cpu().numpy()
            still_active = []
            for pos, j in enumerate(active_local):
                if var[pos] > 1e-4:
                    final_V[j] = V[pos]
                else:
                    Dv[j] = max(0.04, Dv[j] * 0.85)
                    seed_radius[j] = min(25, seed_radius[j] + 4)
                    still_active.append(j)
            active_local = still_active
        if active_local:
            sub_n = [5000] * len(active_local)
            V = _simulate_batch_gpu(
                [0.035] * len(active_local), [0.061] * len(active_local),
                [0.16] * len(active_local), [0.06] * len(active_local),
                [10] * len(active_local), sub_n, size, device,
            )
            for pos, j in enumerate(active_local):
                final_V[j] = V[pos]

        import torch
        V_stack = torch.stack(final_V, dim=0)
        chunk_labels = [labels[i] for i in chunk_idx]
        saved = _save_batch(V_stack, chunk_idx, chunk_labels, tag, output_dir)
        for i, p in saved.items():
            paths[i] = p
        print(f'  [{tag}] {n_skipped + len(chunk_idx)}/{n} listas', flush=True)

    return paths
