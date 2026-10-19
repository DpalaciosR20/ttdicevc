"""
lorenz_lib.py
─────────────
Generación class-blind de imágenes del Atractor de Lorenz a partir de los
5 componentes UMAP normalizados.

Historial: el codigo original ya era class-blind en su motor fisico (el
color por clase estaba comentado); solo faltaba reemplazar el color
comentado por intensidad de gris real (para no perder la informacion de
z(t) que originalmente iba en el color) y forzar la conversion final a un
solo canal ('L'). Ambos ajustes ya estan aplicados aqui.

Contrato técnico (mapeo UMAP -> parametros):
  umap[0] -> sigma      in [8.0, 14.0]
  umap[1] -> rho        in [24.0, 32.0]
  umap[2] -> beta       in [2.0, 4.0]
  umap[3] -> n_puntos   in [5_000, 100_000]
  umap[4] -> elevacion  in [0.0, 180.0] grados
  Imagen 256x256, escala de grises (1 canal), fondo negro.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')  # backend sin GUI -- compatible con Colab/Kaggle/headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

IMG_SIZE = 256
LW_RANGE = (0.1, 1.5)
ALPHA_RANGE = (0.3, 0.7)

PARAM_RANGES = {
    'sigma':     (8.0, 14.0),
    'rho':       (24.0, 32.0),
    'beta':      (2.0, 4.0),
    'n_puntos':  (5_000, 100_000),
    'elevacion': (0.0, 180.0),
}


def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def map_umap_to_lorenz(umap_components):
    c = np.asarray(umap_components, dtype=np.float64)
    assert c.shape == (5,), f"Se esperan 5 componentes, recibidos {c.shape}"

    def _lerp(v, lo, hi):
        return lo + v * (hi - lo)

    return {
        'sigma':     _lerp(c[0], *PARAM_RANGES['sigma']),
        'rho':       _lerp(c[1], *PARAM_RANGES['rho']),
        'beta':      _lerp(c[2], *PARAM_RANGES['beta']),
        'n_puntos':  int(_lerp(c[3], *PARAM_RANGES['n_puntos'])),
        'elevacion': _lerp(c[4], *PARAM_RANGES['elevacion']),
    }


def solve_attractor(sigma, rho, beta, n_puntos, t_max=50.0, initial_state=None, t_discard=5.0):
    """Integracion 100% deterministica (RK45, sin ninguna semilla aleatoria)."""
    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]
    n_total = int(n_puntos * (t_max / (t_max - t_discard)))
    t_eval = np.linspace(0, t_max, n_total)
    sol = solve_ivp(lorenz_system, (0, t_max), initial_state, args=(sigma, rho, beta),
                     method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"solve_ivp fallo: {sol.message}")
    mask = sol.t >= t_discard
    return {'x': sol.y[0, mask], 'y': sol.y[1, mask], 'z': sol.y[2, mask], 't': sol.t[mask]}


def compute_velocity(trajectory):
    dx, dy, dz = np.diff(trajectory['x']), np.diff(trajectory['y']), np.diff(trajectory['z'])
    dt = np.diff(trajectory['t'])
    dt = np.where(dt == 0, 1e-12, dt)
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / dt


def normalize_array(arr):
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - vmin) / (vmax - vmin)


def render_attractor_enriched(trajectory, elevacion, output_path=None, azimuth=45.0,
                               lw_range=LW_RANGE, alpha_range=ALPHA_RANGE,
                               invert_alpha=False, bg_color='black'):
    """Renderiza en escala de grises real. z(t) -> intensidad de gris,
    velocidad -> grosor y opacidad. Sin ningun parametro de label."""
    speed = compute_velocity(trajectory)
    speed_norm = normalize_array(speed)

    z = trajectory['z']
    z_mid = (z[:-1] + z[1:]) / 2.0
    z_norm = normalize_array(z_mid)

    gray_val = 0.3 + 0.7 * z_norm
    colors_rgba = np.zeros((len(gray_val), 4))
    colors_rgba[:, 0] = colors_rgba[:, 1] = colors_rgba[:, 2] = gray_val

    alpha_min, alpha_max = alpha_range
    alphas = (alpha_max - speed_norm * (alpha_max - alpha_min)) if invert_alpha \
        else (alpha_min + speed_norm * (alpha_max - alpha_min))
    colors_rgba[:, 3] = alphas

    lw_min, lw_max = lw_range
    linewidths = lw_min + speed_norm * (lw_max - lw_min)

    x, y = trajectory['x'], trajectory['y']
    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    lc = Line3DCollection(segments, colors=colors_rgba, linewidths=linewidths)
    ax.add_collection3d(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.view_init(elev=elevacion, azim=azimuth)
    ax.set_axis_off()
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')

    if output_path is not None:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(output_path, dpi=100, facecolor=bg_color, bbox_inches='tight', pad_inches=0)
        img = Image.open(output_path).convert('L')
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE))  # salvaguarda por bbox_inches='tight'
        img.save(output_path)
    plt.close(fig)
    return fig


def expected_path(idx, label, tag, output_dir):
    return os.path.join(output_dir, f"lorenz_{tag}_{idx:04d}_label{label}.png")


def generate_batch(coords_5d, labels, tag, output_dir):
    """Genera un lote completo (reanudable). Mas lento que Perlin, similar
    a Gray-Scott CPU (~10-12s/imagen) por el costo del solver ODE."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(len(labels)):
        label = int(labels[i])
        fpath = expected_path(i, label, tag, output_dir)
        if os.path.exists(fpath):
            paths.append(fpath)
            continue
        params = map_umap_to_lorenz(coords_5d[i])              # <- SOLO el vector UMAP
        trajectory = solve_attractor(sigma=params['sigma'], rho=params['rho'],
                                      beta=params['beta'], n_puntos=params['n_puntos'])
        render_attractor_enriched(trajectory, elevacion=params['elevacion'], output_path=fpath)
        paths.append(fpath)
    return paths
