"""
perlin_lib.py
─────────────
Generación class-blind de imágenes Perlin Flow Fields a partir de los 5
componentes UMAP normalizados.

IMPORTANTE (historial): la version original de este generador usaba
`seed = label*10000 + i`, lo que hacia que la geometria (no solo el color)
dependiera de la etiqueta -- una fuga real, ya corregida aqui. La semilla
ahora es un hash determinista de los propios valores UMAP, y ninguna
funcion de este archivo recibe el label para el proceso de generacion.

Requiere el paquete `noise` (pip install noise). En Windows puede fallar
la instalacion por requerir Microsoft C++ Build Tools -- si eso pasa,
generar las imagenes de Perlin en Colab/Kaggle (Linux) en vez de local.

Contrato técnico:
  - frecuencia in [2.0, 10.0]
  - turbulencia in [0.5, 3.0]
  - velocidad in [0.5, 3.0] px/paso
  - longitud de trayectoria in [30, 200] pasos
  - numero de octavas in [1, 8]
  - 800 particulas
  - Imagen 256x256, escala de grises (1 canal)
"""

import os
import hashlib
import numpy as np
from PIL import Image
from noise import pnoise2

PARAM_RANGES = {
    'frequency':   (2.0, 10.0),
    'turbulence':  (0.5, 3.0),
    'speed':       (0.5, 3.0),
    'traj_length': (30, 200),
    'n_octaves':   (1, 8),
}


def lerp(t, lo, hi):
    return lo + t * (hi - lo)


def map_umap_to_perlin(params):
    return {
        'frequency':   lerp(params[0], *PARAM_RANGES['frequency']),
        'turbulence':  lerp(params[1], *PARAM_RANGES['turbulence']),
        'speed':       lerp(params[2], *PARAM_RANGES['speed']),
        'traj_length': int(lerp(params[3], *PARAM_RANGES['traj_length'])),
        'n_octaves':   int(lerp(params[4], *PARAM_RANGES['n_octaves'])),
    }


def seed_from_umap(params):
    """Semilla determinista derivada SOLO de los 5 valores UMAP -- nunca del
    label ni del indice de la muestra."""
    key = ",".join(f"{v:.8f}" for v in params)
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest, 16) % (2 ** 32)


def generate_noise_grid(size, frequency, n_octaves, seed=0):
    scale = frequency / size

    def _noise_at(x, y):
        return pnoise2(x * scale, y * scale, octaves=n_octaves,
                        persistence=0.5, lacunarity=2.0, base=seed)

    xs, ys = np.arange(size), np.arange(size)
    xx, yy = np.meshgrid(xs, ys)
    return np.vectorize(_noise_at)(xx, yy)


def generate_flow_field(noise_grid, turbulence):
    return noise_grid * np.pi * turbulence


def trace_particles(flow_field, n_particles, speed, traj_length, rng):
    h, w = flow_field.shape
    trajectories = []
    for _ in range(n_particles):
        x, y = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        path = [(x, y)]
        for _ in range(traj_length):
            xi, yi = int(x) % w, int(y) % h
            angle = flow_field[yi, xi]
            x += np.cos(angle) * speed
            y += np.sin(angle) * speed
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            path.append((x, y))
        if len(path) > 1:
            trajectories.append(path)
    return trajectories


def draw_trajectories_grayscale(trajectories, noise_grid, size):
    """Intensidad de gris derivada del valor de ruido normalizado -- SIN
    colormap, sin ninguna dependencia de clase."""
    nmin, nmax = noise_grid.min(), noise_grid.max()
    noise_norm = (noise_grid - nmin) / (nmax - nmin + 1e-8)
    canvas = np.zeros((size, size), dtype=np.float64)
    for path in trajectories:
        sx, sy = int(path[0][0]) % size, int(path[0][1]) % size
        gray_val = noise_norm[sy, sx]
        for i in range(len(path) - 1):
            x0, y0 = int(path[i][0]), int(path[i][1])
            if 0 <= x0 < size and 0 <= y0 < size:
                canvas[y0, x0] = max(canvas[y0, x0], gray_val)
    return (canvas * 255).astype(np.uint8)


def generate_perlin_image_classblind(params, size=256, n_particles=800):
    """Genera UNA imagen Perlin en escala de grises. Recibe SOLO 'params'
    (el vector UMAP de 5 componentes) -- no hay argumento de label en
    ningun punto de esta funcion ni de las que llama."""
    seed = seed_from_umap(params)
    rng = np.random.default_rng(seed)
    p = map_umap_to_perlin(params)
    noise_grid = generate_noise_grid(size, p['frequency'], p['n_octaves'], seed=seed % 1024)
    flow_field = generate_flow_field(noise_grid, p['turbulence'])
    trajectories = trace_particles(flow_field, n_particles, p['speed'], p['traj_length'], rng)
    return draw_trajectories_grayscale(trajectories, noise_grid, size)


def expected_path(idx, label, tag, output_dir):
    return os.path.join(output_dir, f"perlin_{tag}_{idx:04d}_label{label}.png")


def generate_batch(coords_5d, labels, tag, output_dir):
    """Genera un lote completo (reanudable: salta archivos ya existentes).
    Rapido incluso en CPU (~0.2-0.3s/imagen), no requiere GPU."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(len(labels)):
        label = int(labels[i])
        fpath = expected_path(i, label, tag, output_dir)
        if os.path.exists(fpath):
            paths.append(fpath)
            continue
        img = generate_perlin_image_classblind(coords_5d[i])  # <- SOLO el vector UMAP
        Image.fromarray(img).convert('L').save(fpath)
        paths.append(fpath)
    return paths
