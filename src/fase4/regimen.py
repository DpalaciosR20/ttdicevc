"""
regimen.py
──────────
F4.2: régimen dinámico (C). Especificación en docs/plan_fase4.md, F4.2. Se corre en local.

Pregunta (plan v2): ¿la separación de clases se debe a cruzar fronteras de
régimen en vez de una codificación continua? Es un análisis explicativo
(protocolo §9): no cambia veredictos ni selecciona modelos.

Método (plan, decisiones F4.2-1 a F4.2-6):
  Mapeos. Se toman de src/generation y no se reescriben aquí:
    Gray-Scott  gray_scott_lib.umap_to_rd_params   f = 0.022 + 0.024·u0, k = 0.059 + 0.007·u1
    Lorenz      lorenz_lib.map_umap_to_lorenz      ρ = 24 + 8·u1
    Perlin      perlin_lib.map_umap_to_perlin      octavas = int(lerp(u4, 1, 8))
  Las distribuciones usan solo train (908 muestras, sin modelos).
  Gray-Scott  Dispersión de (f, k) por clase. Descriptivo, sin prueba.
              El diagrama de Pearson (1993) se superpone solo si hay en el
              repositorio una fuente versionada de sus fronteras. Hoy no la
              hay, así que la figura va sin superposición (PENDIENTE).
  Lorenz      Proporción por clase con ρ < 24.74 (u1 < 0.0925): Fisher exacto.
              ρ por clase: Mann–Whitney bilateral.
  Perlin      Octavas por clase: Mann–Whitney bilateral.
  Reintentos  Tasa de reintentos de Gray-Scott por clase: Fisher. Sale del
              registro de src/fase4/reintentos_gs.py (Kaggle, CPU).
  Holm        Familia F4.2, m = 4 (Fisher y Mann–Whitney de Lorenz,
              Mann–Whitney de Perlin, Fisher de reintentos), α = 0.05.
  Estratos    F1 del finalista de Lorenz (ViT-B/16 congelado, media de p_dep
              de 5 semillas > 0.5, §7) por estrato ρ < 24.74 / ρ ≥ 24.74.
              Usa las predicciones de prueba archivadas de F3.5. Descriptivo;
              cumple G1 porque se usa después del commit del veredicto de la
              Fase 5. No hay inferencia nueva (G2).

PENDIENTES que el plan no fija (results/f42_regimen/reporte_e2a.md). No se
resuelven aquí: las funciones los reciben como parámetros, y la CLI exige el
archivo --resoluciones con todos ellos. Si falta alguno, se detiene.
  fisher_alternativa    'two-sided' | 'greater' | 'less'
  mw_continuidad        true | false        (corrección de continuidad, asintótica)
  reintento_evento      'reintentos_mayor_0' | 'usa_respaldo'
  f1_estrato_sin_clase  'cero' | 'no_definido'
  bit_a_bit             'sha256_archivo' | 'pixeles'   (criterio de «idéntica»)

Compuerta: la etiqueta enmienda-5 está en la historia de HEAD,
docs/plan_fase4.md no cambió frente a ella, results/f5_contraste/resultados.json
está versionado y sin cambios, y el árbol está limpio.
No sobrescribe: si <out>/resultados.json ya existe, se detiene.

Uso:
  python -m src.fase4.regimen --resoluciones <json> [--reintentos <csv>] [--out <dir>]
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / 'src' / 'training'), str(REPO / 'src' / 'generation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LORENZ_RHO_UMBRAL = 24.74
LORENZ_U1_UMBRAL = 0.0925          # (24.74 − 24) / 8
ALPHA = 0.05                       # protocolo, α de las familias de Holm
MW_METODO = 'asymptotic'           # n = 908 con empates (octavas enteras): el exacto no aplica
SEEDS = (42, 7, 123, 2024, 99)
LORENZ_FINAL = 'results/predictions/final/lorenz__vit_b16__frozen__seed{seed}.csv'
GATE_TAG = 'enmienda-5'
PLAN = 'docs/plan_fase4.md'
F5_RESULT = 'results/f5_contraste/resultados.json'
OUT = REPO / 'results' / 'f42_regimen'
REINTENTOS_CSV = OUT / 'reintentos' / 'reintentos_gs_train.csv'
FAMILIA = ('lorenz_fisher_regimen', 'lorenz_mw_rho', 'perlin_mw_octavas', 'gray_scott_fisher_reintentos')
RESOLUCIONES = {'fisher_alternativa': ('two-sided', 'greater', 'less'),
                'mw_continuidad': (True, False),
                'reintento_evento': ('reintentos_mayor_0', 'usa_respaldo'),
                'f1_estrato_sin_clase': ('cero', 'no_definido'),
                'bit_a_bit': ('sha256_archivo', 'pixeles')}


class Detener(RuntimeError):
    """El análisis se detiene porque falla una comprobación o una compuerta."""


def check_resoluciones(res, spec=RESOLUCIONES):
    faltan = [k for k in spec if res.get(k) is None]
    if faltan:
        raise Detener('PENDIENTE sin resolver en --resoluciones: ' + ', '.join(faltan))
    malos = {k: res[k] for k in spec if res[k] not in spec[k]}
    if malos:
        raise Detener(f'valores no previstos en --resoluciones: {malos}')
    return {k: res[k] for k in spec}


# ── Mapeos (de src/generation; no se reescriben) ──────────────────────────
def gs_fk(u):
    import gray_scott_lib
    p = gray_scott_lib.umap_to_rd_params(u)
    return p['f'], p['k']


def lorenz_rho(u):
    import lorenz_lib
    return lorenz_lib.map_umap_to_lorenz(u)['rho']


def perlin_octavas(u):
    import perlin_lib
    return perlin_lib.map_umap_to_perlin(u)['n_octaves']


def mapear(U):
    """Parámetros de régimen de cada fila de U. Se conserva el dtype almacenado, igual que al generar."""
    fk = np.array([gs_fk(u) for u in U], dtype=np.float64).reshape(-1, 2)
    return {'f': fk[:, 0], 'k': fk[:, 1],
            'rho': np.array([lorenz_rho(u) for u in U], dtype=np.float64),
            'octavas': np.array([perlin_octavas(u) for u in U], dtype=np.int64)}


def estrato_lorenz(rho):
    """True si ρ < 24.74, el estrato por debajo de la bifurcación."""
    return np.asarray(rho, dtype=np.float64) < LORENZ_RHO_UMBRAL


# ── Pruebas ───────────────────────────────────────────────────────────────
def tabla_2x2(evento, y):
    """[[evento ∧ y=1, ¬evento ∧ y=1], [evento ∧ y=0, ¬evento ∧ y=0]]. Filas: depresivo, no depresivo."""
    e, y = np.asarray(evento, dtype=bool), np.asarray(y).astype(int)
    if not np.isin(y, (0, 1)).all():
        raise Detener('etiquetas no binarias')
    return [[int((e & (y == 1)).sum()), int((~e & (y == 1)).sum())],
            [int((e & (y == 0)).sum()), int((~e & (y == 0)).sum())]]


def fisher(tabla, alternativa):
    from scipy.stats import fisher_exact
    r = fisher_exact(tabla, alternative=alternativa)
    return {'tabla': tabla, 'odds_ratio': float(r.statistic), 'p': float(r.pvalue), 'alternativa': alternativa}


def mann_whitney(x_dep, x_nodep, continuidad):
    """Mann–Whitney bilateral. U es el de la clase depresiva (scipy: U1 de x)."""
    from scipy.stats import mannwhitneyu
    x1, x0 = np.asarray(x_dep, dtype=np.float64), np.asarray(x_nodep, dtype=np.float64)
    r = mannwhitneyu(x1, x0, alternative='two-sided', method=MW_METODO, use_continuity=bool(continuidad))
    n1, n0 = len(x1), len(x0)
    return {'U_dep': float(r.statistic), 'n_dep': n1, 'n_nodep': n0, 'p': float(r.pvalue),
            'efecto_rbc': float(2 * r.statistic / (n1 * n0) - 1),   # correlación biserial de rangos
            'mediana_dep': float(np.median(x1)), 'mediana_nodep': float(np.median(x0)),
            'metodo': MW_METODO, 'continuidad': bool(continuidad)}


def holm_ajustado(pvalues):
    """p ajustados de Holm: max_{j ≤ k} min(1, (m − j + 1)·p_(j))."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m, run, out = len(keys), 0.0, {}
    for j, k in enumerate(keys, start=1):
        run = max(run, min(1.0, (m - j + 1) * pvalues[k]))
        out[k] = run
    return out


def familia_holm(pvalues, alpha=ALPHA):
    from src.protocol.verdict import holm
    if set(pvalues) != set(FAMILIA):
        raise Detener(f'la familia de Holm de F4.2 es {FAMILIA}; se recibió {sorted(pvalues)}')
    h, adj = holm(pvalues, alpha), holm_ajustado(pvalues)
    return {k: {'p': pvalues[k], 'p_holm': adj[k], **h[k]} for k in FAMILIA}


# ── F1 por estrato (Lorenz) ───────────────────────────────────────────────
def f1_macro(y, yhat, sin_clase):
    """F1-Macro binario. Si una clase falta en y y en ŷ, su F1 es 0/0: vale 0 ('cero') o no se define ('no_definido')."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    f1s = []
    for c in (1, 0):
        tp = int(((yhat == c) & (y == c)).sum())
        den = 2 * tp + int(((yhat == c) & (y != c)).sum()) + int(((yhat != c) & (y == c)).sum())
        if den == 0:
            if sin_clase == 'no_definido':
                return None
            if sin_clase != 'cero':
                raise Detener(f'f1_estrato_sin_clase desconocido {sin_clase!r}')
            f1s.append(0.0)
        else:
            f1s.append(2 * tp / den)
    return float(np.mean(f1s))


def f1_por_estrato(y, yhat, bajo, sin_clase):
    out = {}
    for nombre, m in (('rho_menor_24.74', np.asarray(bajo, dtype=bool)), ('rho_mayor_igual_24.74', ~np.asarray(bajo, dtype=bool))):
        yy, hh = np.asarray(y)[m], np.asarray(yhat)[m]
        out[nombre] = {'n': int(m.sum()), 'n_dep': int((yy == 1).sum()), 'n_nodep': int((yy == 0).sum()),
                       'errores': int((yy != hh).sum()),
                       'f1_macro': f1_macro(yy, hh, sin_clase) if m.any() else None}
    return out


# ── Análisis completo (puro: recibe arreglos) ─────────────────────────────
def eventos_reintento(filas, evento, bit_a_bit):
    """Del registro de reintentos (filas en orden de sample_id): evento por muestra y muestras no idénticas."""
    col = {'reintentos_mayor_0': lambda r: int(r['reintentos']) > 0,
           'usa_respaldo': lambda r: r['usa_respaldo'] in ('True', 'true', '1', True)}
    ident = {'sha256_archivo': 'sha256_coincide', 'pixeles': 'pixeles_coinciden'}
    if evento not in col or bit_a_bit not in ident:
        raise Detener(f'resolución desconocida: {evento!r}, {bit_a_bit!r}')
    ev = np.array([col[evento](r) for r in filas], dtype=bool)
    no_ident = [r['sample_id'] for r in filas if r[ident[bit_a_bit]] not in ('True', 'true', '1', True)]
    return ev, no_ident


def analizar(U_train, y_train, filas_reintentos, U_test, y_test, yhat_test, res):
    res = check_resoluciones(res)
    y_train = np.asarray(y_train).astype(int)
    if len(filas_reintentos) != len(y_train):
        raise Detener(f'registro de reintentos con {len(filas_reintentos)} filas; train tiene {len(y_train)}')
    if [int(r['y_true']) for r in filas_reintentos] != list(y_train):
        raise Detener('registro de reintentos no alineado con las etiquetas de train')
    mp = mapear(U_train)
    dep, nodep = y_train == 1, y_train == 0
    bajo = estrato_lorenz(mp['rho'])
    u1_bajo = np.asarray(U_train, dtype=np.float64)[:, 1] < LORENZ_U1_UMBRAL
    ev, no_ident = eventos_reintento(filas_reintentos, res['reintento_evento'], res['bit_a_bit'])
    pruebas = {
        'lorenz_fisher_regimen': fisher(tabla_2x2(bajo, y_train), res['fisher_alternativa']),
        'lorenz_mw_rho': mann_whitney(mp['rho'][dep], mp['rho'][nodep], res['mw_continuidad']),
        'perlin_mw_octavas': mann_whitney(mp['octavas'][dep], mp['octavas'][nodep], res['mw_continuidad']),
        'gray_scott_fisher_reintentos': fisher(tabla_2x2(ev, y_train), res['fisher_alternativa']),
    }
    holm = familia_holm({k: v['p'] for k, v in pruebas.items()})
    mp_te = mapear(U_test)
    bajo_te = estrato_lorenz(mp_te['rho'])
    octs = sorted(set(mp['octavas'].tolist()))
    return {
        'resoluciones': res,
        'n_train': {'total': int(len(y_train)), 'dep': int(dep.sum()), 'nodep': int(nodep.sum())},
        'lorenz': {
            'proporcion_rho_menor': {'dep': float(bajo[dep].mean()), 'nodep': float(bajo[nodep].mean()),
                                     'n_dep': int(bajo[dep].sum()), 'n_nodep': int(bajo[nodep].sum())},
            'discordancias_rho_vs_u1': int((bajo != u1_bajo).sum()),
            'rho_cuartiles': {c: np.quantile(mp['rho'][m], [0.25, 0.5, 0.75]).tolist() for c, m in (('dep', dep), ('nodep', nodep))},
        },
        'perlin': {'octavas_por_clase': {str(o): {'dep': int((mp['octavas'][dep] == o).sum()),
                                                  'nodep': int((mp['octavas'][nodep] == o).sum())} for o in octs}},
        'gray_scott': {
            'fk_por_clase': {c: {'f_media': float(mp['f'][m].mean()), 'k_media': float(mp['k'][m].mean()),
                                 'f_rango': [float(mp['f'][m].min()), float(mp['f'][m].max())],
                                 'k_rango': [float(mp['k'][m].min()), float(mp['k'][m].max())]}
                             for c, m in (('dep', dep), ('nodep', nodep))},
            'reintentos': {'evento': res['reintento_evento'], 'tasa_dep': float(ev[dep].mean()), 'tasa_nodep': float(ev[nodep].mean()),
                           'n_dep': int(ev[dep].sum()), 'n_nodep': int(ev[nodep].sum()),
                           'n_reintentos_mayor_0': int(sum(int(r['reintentos']) > 0 for r in filas_reintentos)),
                           'n_respaldo': int(sum(r['usa_respaldo'] in ('True', 'true', '1', True) for r in filas_reintentos))},
            'bit_a_bit': {'criterio': res['bit_a_bit'], 'n_identicas': int(len(y_train) - len(no_ident)),
                          'n_total': int(len(y_train)), 'no_identicas': no_ident},
            'pearson_superpuesto': False,
        },
        'pruebas': pruebas,
        'holm': {'alpha': ALPHA, 'm': len(FAMILIA), 'familia': holm},
        'lorenz_estratos_prueba': {
            'discordancias_rho_vs_u1': int((bajo_te != (np.asarray(U_test, dtype=np.float64)[:, 1] < LORENZ_U1_UMBRAL)).sum()),
            **f1_por_estrato(y_test, yhat_test, bajo_te, res['f1_estrato_sin_clase'])},
        '_arrays': {'train': mp, 'y_train': y_train, 'reintento': ev},
    }


# ── Datos ─────────────────────────────────────────────────────────────────
def load_index():
    with open(REPO / 'meta' / 'sample_index.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {s: sorted((r for r in rows if r['split'] == s), key=lambda r: r['sample_id']) for s in ('train', 'val', 'test')}


def load_u(index, splits):
    """u (n × 5, dtype almacenado) y etiquetas por split, en el orden de sample_id (alineadas por idx)."""
    from src import paths
    from src.build_manifest import verify_manifest
    from umap_io import load_umap_arrays
    problems = verify_manifest('UMAP_PARAMS')
    if problems:
        raise Detener(f'umap_params.pkl: {problems}')
    arr = load_umap_arrays(paths.umap_params_path())
    out = {}
    for s in splits:
        order = np.array([int(r['idx']) for r in index[s]])
        u, lab = arr[s]['params'][order], arr[s]['labels'][order].astype(int)
        if list(lab) != [int(r['y_true']) for r in index[s]]:
            raise Detener(f'umap_params.pkl ({s}): etiquetas no alineadas con sample_index por idx')
        out[s] = (u, lab)
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_reintentos(path, index_train):
    with open(path, newline='', encoding='utf-8') as f:
        filas = {r['sample_id']: r for r in csv.DictReader(f)}
    want = [r['sample_id'] for r in index_train]
    if sorted(filas) != sorted(want):
        raise Detener(f'{path}: las muestras no son las {len(want)} de train')
    return [filas[s] for s in want]


def load_lorenz_test(index_test):
    """p_dep de prueba de las 5 semillas del finalista de Lorenz (F3.5, archivadas) → ŷ (§7) y sha256 por archivo."""
    from src.protocol import contraste as cx
    from src.protocol import insumos as ins_mod
    want = [r['sample_id'] for r in index_test]
    P, shas = [], {}
    for seed in SEEDS:
        path = REPO / LORENZ_FINAL.format(seed=seed)
        with open(path, newline='', encoding='utf-8') as f:
            rows = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == 'test'}
        if sorted(rows) != sorted(want):
            raise Detener(f'{path}: las filas de prueba no son las {len(want)} de sample_index')
        if [int(rows[s]['y_true']) for s in want] != [int(r['y_true']) for r in index_test]:
            raise Detener(f'{path}: y_true distinto de sample_index')
        P.append([float(rows[s]['p_dep']) for s in want])
        shas[path.relative_to(REPO).as_posix()] = sha256_file(path)
    return cx.yhat_mean(np.array(P).T, ins_mod.load_config()), shas


# ── Compuerta y escritura ─────────────────────────────────────────────────
def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)


def check_gate():
    if _git('merge-base', '--is-ancestor', GATE_TAG, 'HEAD').returncode != 0:
        raise Detener(f'la etiqueta {GATE_TAG} no está en la historia de HEAD')
    if _git('diff', '--quiet', GATE_TAG, 'HEAD', '--', PLAN).returncode != 0:
        raise Detener(f'{PLAN} cambió frente a {GATE_TAG}')
    if _git('ls-files', '--error-unmatch', F5_RESULT).returncode != 0:
        raise Detener(f'{F5_RESULT} no está versionado (G1: veredicto de la Fase 5 sin commit)')
    st = _git('status', '--porcelain').stdout.strip()
    if st:
        raise Detener('árbol de trabajo con cambios:\n' + st)
    return _git('rev-parse', 'HEAD').stdout.strip()


def refuse_overwrite(path):
    if Path(path).exists():
        raise Detener(f'{path} ya existe; no se sobrescribe')


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def figuras(r, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / 'src'))
    from viz import style
    style.apply()
    fig_dir = Path(out_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    mp, y = r['_arrays']['train'], r['_arrays']['y_train']
    clases = ((1, 'Depresivo', style.TECH_COLOR['perlin'], 'o'), (0, 'No depresivo', style.TECH_COLOR['gray_scott'], 's'))
    n_txt = f"Train: n = {r['n_train']['total']} ({r['n_train']['dep']} depresivas, {r['n_train']['nodep']} no depresivas)."
    paths = []

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for c, lab, col, mk in clases:
        m = y == c
        ax.scatter(mp['f'][m], mp['k'][m], s=9, alpha=0.55, color=col, marker=mk, label=f'{lab} ({int(m.sum())})', linewidths=0)
    ax.set_xlabel('f (alimentación)')
    ax.set_ylabel('k (decaimiento)')
    ax.set_title('Gray-Scott: (f, k) por clase en train')
    ax.legend(loc='upper left', fontsize=7)
    fig.text(0.01, -0.02, n_txt + ' Sin superposición del diagrama de Pearson (1993): PENDIENTE, '
             'no hay en el repositorio una fuente versionada de sus fronteras.', fontsize=6.5, va='top', wrap=True)
    paths.append(style.save(fig, fig_dir / 'f42_01_gray_scott_fk'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = np.linspace(24, 32, 33)
    for c, lab, col, _ in clases:
        ax.hist(mp['rho'][y == c], bins=bins, density=True, histtype='step', lw=1.6, color=col, label=lab)
    ax.axvline(LORENZ_RHO_UMBRAL, color=style.THRESHOLD_COLOR, ls='--', lw=1.1)
    ax.text(LORENZ_RHO_UMBRAL, ax.get_ylim()[1] * 0.95, ' ρ = 24.74', color=style.THRESHOLD_COLOR, fontsize=7, va='top')
    p = r['pruebas']
    ax.set_xlabel('ρ')
    ax.set_ylabel('densidad')
    ax.set_title('Lorenz: ρ por clase en train')
    ax.legend(fontsize=7)
    fig.text(0.01, -0.02, n_txt + f" Fisher (ρ < 24.74): p = {p['lorenz_fisher_regimen']['p']:.4g}; "
             f"Mann–Whitney: p = {p['lorenz_mw_rho']['p']:.4g} (sin ajustar; Holm en resultados.json).", fontsize=6.5, va='top', wrap=True)
    paths.append(style.save(fig, fig_dir / 'f42_02_lorenz_rho'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    octs = sorted(set(mp['octavas'].tolist()))
    w = 0.38
    for j, (c, lab, col, _) in enumerate(clases):
        m = y == c
        ax.bar(np.array(octs) + (j - 0.5) * w, [(mp['octavas'][m] == o).mean() for o in octs], width=w, color=col, label=lab)
    ax.set_xticks(octs)
    ax.set_xlabel('octavas')
    ax.set_ylabel('proporción dentro de la clase')
    ax.set_title('Perlin: octavas por clase en train')
    ax.legend(fontsize=7)
    fig.text(0.01, -0.02, n_txt + f" Mann–Whitney: p = {p['perlin_mw_octavas']['p']:.4g} (sin ajustar; Holm en resultados.json).",
             fontsize=6.5, va='top', wrap=True)
    paths.append(style.save(fig, fig_dir / 'f42_03_perlin_octavas'))
    plt.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--resoluciones', required=True)
    ap.add_argument('--reintentos', default=str(REINTENTOS_CSV))
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    res = check_resoluciones(json.loads(Path(a.resoluciones).read_text(encoding='utf-8')))
    out = Path(a.out)
    refuse_overwrite(out / 'resultados.json')
    refuse_overwrite(out / 'figures')
    commit = check_gate()
    index = load_index()
    U = load_u(index, ('train', 'test'))
    filas = load_reintentos(a.reintentos, index['train'])
    yhat_te, shas = load_lorenz_test(index['test'])
    r = analizar(U['train'][0], U['train'][1], filas, U['test'][0], U['test'][1], yhat_te, res)
    figs = figuras(r, out)
    doc = {'meta': {'tarea': 'F4.2', 'git_commit': commit, 'gate_tag': GATE_TAG,
                    'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'inputs': {Path(a.reintentos).resolve().relative_to(REPO).as_posix(): sha256_file(a.reintentos), **shas},
                    'figuras': [Path(p).resolve().relative_to(REPO).as_posix() for p in figs]},
           'resultados': _clean(r)}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'resultados.json').write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(_clean({'holm': r['holm'], 'lorenz_estratos_prueba': r['lorenz_estratos_prueba']}), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
