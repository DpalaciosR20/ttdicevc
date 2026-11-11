"""
figura_ilustrativa.py
─────────────────────
F4.8 — Figura ilustrativa (docs/plan_fase4.md, F4.8). Sin peso argumentativo:
solo ilustra cómo se ve la dirección discriminativa; no produce métricas.

1. Dirección: la LogReg umap5d__logreg de F2.1 (C = 0.1, class_weight =
   balanced, StandardScaler ajustado en train), reproducida con el código de
   F2.1 (results/f21_techo5d/f21_techo5d.py). Sus p_dep de val se verifican
   contra results/predictions/referencia/umap5d__logreg__none__seed42.csv.
   Prueba no se usa (ninguna inferencia sobre prueba).
2. Recorridos en el espacio de u:
   - centroides de clase en train y 7 puntos lineales entre ellos (extremos incluidos);
   - ±1.5 DE sobre la dirección normalizada de w desde la media global de train.
   Los puntos fuera de [0, 1] se recortan y se marcan (componentes recortadas).
3. Renderizado con las funciones de src/generation (Gray-Scott por la ruta de
   CPU), en escala de grises; una versión con paleta solo como ilustración.

Las decisiones que el plan no fija están en RESOLUCIONES (PENDIENTES de
results/f44_convergencia/reporte_e4a.md).

Uso:
  python -m src.fase4.figura_ilustrativa --resoluciones <json> [--out results/f48_figura]
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / 'src' / 'training'), str(REPO / 'src' / 'generation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.fase4.atribucion_comun import Detener  # noqa: E402

F21_SCRIPT = 'results/f21_techo5d/f21_techo5d.py'
F21_SELECCION = REPO / 'results' / 'f21_techo5d' / 'seleccion.csv'
F21_PRED = REPO / 'results' / 'predictions' / 'referencia' / 'umap5d__logreg__none__seed42.csv'
OUT = REPO / 'results' / 'f48_figura'
TECNICAS = ('gray_scott', 'perlin', 'lorenz')
N_PASOS = 7
DE_RECORRIDO = 1.5

RESOLUCIONES = {
    # P7: espacio donde se normaliza w (la LogReg se ajusta sobre u estandarizado)
    'direccion_espacio': ('u', 'estandarizado'),
    # P8: qué DE escala el recorrido de ±1.5 DE
    'de_recorrido': ('proyeccion_train', 'media_de_componentes'),
    # P9: puntos del recorrido de ±1.5 DE
    'n_pasos_recorrido': ('7', '5', '9'),
    # P10: tolerancia de la verificación de p_dep de val contra el CSV de F2.1
    'tol_pdep': ('1e-9', '1e-6'),
    # P11: paleta de la versión ilustrativa
    'paleta': ('viridis', 'magma', 'ninguna'),
    # P12: cómo se marca un punto recortado en la figura
    'marca_recorte': ('borde_y_lista', 'solo_lista'),
}


def check_resoluciones(res):
    res = {k: v for k, v in res.items() if not k.startswith('_')}
    falta = [k for k in RESOLUCIONES if k not in res]
    sobra = [k for k in res if k not in RESOLUCIONES]
    if falta or sobra:
        raise Detener(f'resoluciones: faltan {falta}, sobran {sobra}')
    for k, v in res.items():
        if v not in RESOLUCIONES[k]:
            raise Detener(f'resoluciones: {k} = {v!r} no es una de {RESOLUCIONES[k]}')
    return res


# ── LogReg de F2.1 ────────────────────────────────────────────────────────
def f21():
    spec = importlib.util.spec_from_file_location('f21_techo5d', REPO / F21_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def combinacion_logreg(path=F21_SELECCION):
    with open(path, newline='', encoding='utf-8') as f:
        fila = next(r for r in csv.DictReader(f) if r['familia'] == 'logreg')
    combo = json.loads(fila['combinacion'])
    if combo != {'C': 0.1, 'class_weight': 'balanced'}:
        raise Detener(f'F2.1 logreg seleccionada {combo}; el plan fija C = 0.1, balanced')
    return combo


def ajustar_logreg(u_train, y_train):
    """Pipeline (StandardScaler, LogisticRegression) con el código y la configuración de F2.1."""
    m = f21()
    cfg = json.loads((REPO / 'configs' / 'protocolo.json').read_text(encoding='utf-8'))['techo_5d']
    if 'logreg' not in cfg['scaler']['families']:
        raise Detener('F2.1: la LogReg no lleva StandardScaler en el config')
    fixed = m.fixed_params('logreg', cfg['familias']['logreg'], cfg['random_state'])
    model = m.make_model('logreg', fixed, combinacion_logreg(), True)
    return model.fit(np.asarray(u_train, np.float64), np.asarray(y_train, int)), m


def verificar_pdep_val(model, m, u_val, ids_val, tol, path=F21_PRED):
    """p_dep de val del modelo reproducido contra el CSV de F2.1 (solo val)."""
    with open(path, newline='', encoding='utf-8') as f:
        filas = [r for r in csv.DictReader(f) if r['split'] == 'val']
    ref = {r['sample_id']: float(r['p_dep']) for r in filas}
    if sorted(ref) != sorted(ids_val):
        raise Detener(f'{path.name}: sample_id de val distintos del índice')
    p = m.p_dep_of(model, np.asarray(u_val, np.float64))
    diff = float(np.max(np.abs(p - np.array([ref[i] for i in ids_val]))))
    if diff > tol:
        raise Detener(f'p_dep de val reproducido difiere de F2.1 en {diff:.3g} > {tol}')
    return diff


def direccion(model, espacio):
    """Vector unitario de w: en el espacio de u (w_std / σ) o en el estandarizado."""
    scaler, lr = model[0], model[-1]
    w = lr.coef_[0].astype(np.float64)
    if espacio == 'u':
        w = w / scaler.scale_
    elif espacio != 'estandarizado':
        raise Detener(f'direccion_espacio desconocido: {espacio!r}')
    return w / np.linalg.norm(w)


# ── Recorridos ────────────────────────────────────────────────────────────
def centroides(u, y):
    u, y = np.asarray(u, np.float64), np.asarray(y, int)
    return u[y == 0].mean(0), u[y == 1].mean(0)


def recorrido_centroides(c0, c1, n=N_PASOS):
    t = np.linspace(0.0, 1.0, n)
    return c0[None, :] + t[:, None] * (c1 - c0)[None, :], t


def recorrido_direccion(u, d, modo_de, n, k=DE_RECORRIDO):
    u = np.asarray(u, np.float64)
    m = u.mean(0)
    if modo_de == 'proyeccion_train':
        de = float(np.std((u - m) @ d))
    elif modo_de == 'media_de_componentes':
        de = float(np.mean(u.std(0)))
    else:
        raise Detener(f'de_recorrido desconocido: {modo_de!r}')
    s = np.linspace(-k, k, n)
    return m[None, :] + (s * de)[:, None] * d[None, :], s, de


def recortar(puntos):
    """Recorte a [0, 1] y máscara de componentes recortadas (n, 5)."""
    p = np.asarray(puntos, np.float64)
    marca = (p < 0) | (p > 1)
    return np.clip(p, 0.0, 1.0), marca


# ── Renderizado (src/generation, class-blind) ─────────────────────────────
def render(tech, u):
    """uint8 (256, 256) con el generador de la técnica; la etiqueta nunca entra."""
    from PIL import Image
    u = np.asarray(u, np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        if tech == 'gray_scott':
            import gray_scott_lib as gs
            path, _, _ = gs.generate_and_save_grayscale(0, u, 0, 'f48', tmp)   # ruta CPU; label solo nombra el archivo
            return np.array(Image.open(path).convert('L'))
        if tech == 'perlin':
            import perlin_lib as pl
            return np.asarray(pl.generate_perlin_image_classblind(u), dtype=np.uint8)
        if tech == 'lorenz':
            import lorenz_lib as ll
            p = ll.map_umap_to_lorenz(u)
            tr = ll.solve_attractor(sigma=p['sigma'], rho=p['rho'], beta=p['beta'], n_puntos=p['n_puntos'])
            path = os.path.join(tmp, 'lorenz.png')
            ll.render_attractor_enriched(tr, elevacion=p['elevacion'], output_path=path)
            return np.array(Image.open(path).convert('L'))
    raise Detener(f'técnica desconocida {tech!r}')


# ── Figura ────────────────────────────────────────────────────────────────
def figura(imagenes, recorridos, res, out_png, paleta=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from viz import style
    style.apply()
    filas = [(t, r) for t in TECNICAS for r in recorridos]
    n = max(len(imagenes[t][r]) for t, r in filas)
    fig, axes = plt.subplots(len(filas), n, figsize=(1.25 * n, 1.35 * len(filas)))
    for i, (t, r) in enumerate(filas):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(imagenes[t][r]):
                ax.set_axis_off()
                continue
            ax.imshow(imagenes[t][r][j], cmap=paleta or 'gray', vmin=0, vmax=255)
            recortado = recorridos[r]['recortado'][j]
            for sp in ax.spines.values():
                sp.set_visible(bool(recortado) and res['marca_recorte'] == 'borde_y_lista')
                sp.set_color(style.THRESHOLD_COLOR)
                sp.set_linewidth(1.6)
            if j == 0:
                ax.set_ylabel(f'{style.TECH_LABEL[t]}\n{recorridos[r]["etiqueta"]}', fontsize=style.FONT_SIZES['annot'])
            ax.set_title(recorridos[r]['ticks'][j], fontsize=style.FONT_SIZES['annot'] - 1)
    titulo = 'Dirección discriminativa de la LogReg 5D (F2.1), renderizada por cada técnica'
    fig.suptitle(titulo + (' — versión con paleta, solo ilustrativa' if paleta else ''), y=1.01)
    fig.text(0.01, -0.01, 'Ilustración sin peso argumentativo (plan, F4.8). Borde rojo: punto con alguna componente '
                          'recortada a [0, 1].', fontsize=style.FONT_SIZES['annot'], color=style.INK_SECONDARY)
    style.save(fig, out_png)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='F4.8: figura ilustrativa de la dirección discriminativa')
    ap.add_argument('--resoluciones', required=True)
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    raw = json.loads(Path(a.resoluciones).read_text(encoding='utf-8'))
    res = check_resoluciones(raw)
    out = Path(a.out)
    if (out / 'recorridos.json').exists():
        raise Detener(f'{out / "recorridos.json"} ya existe; no se sobrescribe')
    from src.fase4 import atribucion_comun as ac
    from src.fase4 import atribucion_local as al
    from src.fase4.regimen import check_gate, load_u
    commit = check_gate()
    t0 = time.time()
    index = al.load_index()
    U = load_u(index, ('train', 'val'))                       # sin prueba
    model, m = ajustar_logreg(*U['train'])
    diff = verificar_pdep_val(model, m, U['val'][0], [r['sample_id'] for r in index['val']], float(res['tol_pdep']))
    d = direccion(model, res['direccion_espacio'])
    c0, c1 = centroides(*U['train'])
    pc, t = recorrido_centroides(c0, c1)
    pd_, s, de = recorrido_direccion(U['train'][0], d, res['de_recorrido'], int(res['n_pasos_recorrido']))
    recorridos = {}
    for nombre, (p, ticks, etiqueta) in {'centroides': (pc, [f't={x:.2f}' for x in t], 'centroide 0 → 1'),
                                         'direccion': (pd_, [f'{x:+.2f} DE' for x in s], '±1.5 DE sobre ŵ')}.items():
        pr, marca = recortar(p)
        recorridos[nombre] = {'etiqueta': etiqueta, 'ticks': ticks, 'u_sin_recortar': p, 'u': pr,
                              'componentes_recortadas': marca, 'recortado': marca.any(1),
                              'p_dep_logreg': m.p_dep_of(model, pr)}
    imagenes, tiempos = {}, {}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'imagenes').mkdir(exist_ok=True)
    from PIL import Image
    for tech in TECNICAS:
        tt = time.time()
        imagenes[tech] = {}
        for r, info in recorridos.items():
            imagenes[tech][r] = [render(tech, u) for u in info['u']]
            for j, im in enumerate(imagenes[tech][r]):
                Image.fromarray(im, 'L').save(out / 'imagenes' / f'{tech}_{r}_{j}.png')
        tiempos[tech] = round(time.time() - tt, 1)
        print(f'{tech}: {sum(len(v) for v in imagenes[tech].values())} imágenes, {tiempos[tech]} s', flush=True)
    figura(imagenes, recorridos, res, out / 'f48_direccion_grises')
    if res['paleta'] != 'ninguna':
        figura(imagenes, recorridos, res, out / 'f48_direccion_paleta_ilustrativa', paleta=res['paleta'])
    doc = {'meta': {'tarea': 'F4.8', 'git_commit': commit, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'resoluciones': {'archivo': Path(a.resoluciones).as_posix(), 'sha256': al.sha256_file(a.resoluciones),
                                     **ac.meta_resoluciones(raw)},
                    'logreg': {'combinacion': combinacion_logreg(), 'p_dep_val_max_abs_diff': diff,
                               'csv_f21_sha256': al.sha256_file(F21_PRED)},
                    'segundos': tiempos, 'segundos_total': round(time.time() - t0, 1),
                    'nota': 'sin peso argumentativo (plan, F4.8); sin métricas'},
           'direccion': {'espacio': res['direccion_espacio'], 'w_unitario': d, 'de_recorrido': de,
                         'centroide_0': c0, 'centroide_1': c1, 'media_train': U['train'][0].astype(np.float64).mean(0)},
           'recorridos': recorridos}
    (out / 'recorridos.json').write_text(json.dumps(ac.limpiar(_listas(doc)), indent=1, ensure_ascii=False) + '\n',
                                         encoding='utf-8')


def _listas(o):
    if isinstance(o, dict):
        return {k: _listas(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_listas(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


if __name__ == '__main__':
    try:
        main()
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
