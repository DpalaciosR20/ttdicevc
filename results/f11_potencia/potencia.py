"""
potencia.py
───────────
F1.1 — Análisis de potencia del contraste de no inferioridad frente a BETO
(docs/especificacion_f11.md). Implementa los bloques E, S, D y P de §5 con los
parámetros de §6. No calcula ninguna métrica de prueba de modelos visuales:
en prueba solo intervienen y_test y las predicciones de BETO (F2.2, ya
conocidas); todo lo visual de prueba es simulado.

Uso (desde la raíz del repo, .venv, CPU):
    python results/f11_potencia/potencia.py --dry-run     # estimación de tiempo
    python results/f11_potencia/potencia.py               # corrida completa
    python results/f11_potencia/potencia.py --jobs 6      # escenarios en paralelo

Aclaraciones de implementación (acordadas con SV antes de ejecutar)
───────────────────────────────────────────────────────────────────
1. Holm en S. Cada réplica simula TRES técnicas independientes bajo el mismo
   escenario (mismas predicciones de BETO, mismos índices bootstrap) y aplica
   Holm completo sobre las tres. P(V1..V4) se promedia sobre técnicas.
2. Nivel de los intervalos. Para la técnica de rango k en el orden de Holm,
   α_h = α/(3−k+1). El intervalo es bilateral al 1−2α_h: cada límite es una
   cota unilateral a α_h, dual del valor p bootstrap. V1 ⟺ rechazo de Holm
   (p ≤ α_h en el procedimiento escalonado). V2: A, límite inferior de Δ > δ;
   B, límite superior de θ < 0, ambos al α_h de la técnica. En E se reportan
   los anchos para α_h ∈ {α/3, α/2, α}.
3. Piso. Prueba de permutación de y_test (10,000 permutaciones, matriz común a
   todas las réplicas) sobre el F1 medio de las cinco semillas. El nulo de
   permutación se desplaza para que su media sea 0.529 (modelo de
   corrimiento); p = (1 + #{nulo desplazado ≥ observado}) / (N + 1). Holm
   aparte sobre las tres técnicas.
4. El valor p bootstrap es la proporción de réplicas (§3): A, Δ* ≥ δ;
   B, θ* ≤ 0. Holm rechaza con p ≤ α_h.

Generador de S (§5)
───────────────────
Para cada técnica, semilla s y muestra i de prueba (y_i, acierto de BETO b_i):
  m_i ~ Bern(a), común a las cinco semillas: si m_i, el acierto visual es b_i
       (componente de acuerdo con BETO).
  si no: con prob. κ el acierto es un sorteo z_i ~ Bern(q_{y_i}) común a las
       semillas; si no, un sorteo propio z_is ~ Bern(q_{y_i}).
  q_c = expit(logit(R_c) + η), con R_c el recall de BETO-test en la clase c.
  ŷ_is = y_i si acierta, 1 − y_i si no.
(a, η) se resuelven sobre los conteos esperados para que el F1-Macro
esperado de una semilla sea 0.5 + ρ(F1_BETO − 0.5) y la discordancia
esperada BETO–visual sea la objetivo (percentiles 10/50/90 de E). κ se
calibra con una simulación piloto para que la DE (ddof=1) media entre las
cinco semillas sea la objetivo; si con κ = 0 no se alcanza, se usa κ = 0 y
la celda se marca de_factible = False. Si (a, η) no existen, la celda se
marca factible = False y no se simula. Se reportan la retención, la DE y la
discordancia realizadas.

Error de Monte Carlo: SE de cada P(V) = DE entre réplicas del indicador
promediado sobre técnicas / √n_sims; SE del límite del IC usado por V1
(B = 2,000) por la fórmula asintótica del cuantil, con la densidad estimada
por espaciamiento de estadísticos de orden (k = √B).

Salidas (results/f11_potencia/): empirico.csv, empirico_corridas.csv,
malla_escenarios.csv, retencion_minima.csv, potencia_piso.csv,
curva_potencia.png/.pdf, potencia.log (potencia_dryrun.log con --dry-run).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import paths  # noqa: E402
from src.predictions.validate import load_index, load_predictions  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TASK_ID = 'F1.1'
CANON = REPO / 'results' / 'canonical_numbers.json'
F32_PRED = REPO / 'results' / 'f32_primera_pasada_v0' / 'predictions.csv'
BETO_PRED = REPO / 'results' / 'predictions' / 'referencia' / 'text__beto__finetune__seed42.csv'
EMB_PRED = REPO / 'results' / 'predictions' / 'referencia' / 'emb768__logreg__none__seed42.csv'

# ── Parámetros (§2, §3, §5, §6) ───────────────────────────────────────────
MASTER_SEED = 2026
ALPHA = 0.05
F1_AZAR = 0.5
R_VALUES = (0.50, 0.75, 0.80)
RHO_GRID = (0.25, 0.50, 0.75, 0.90, 1.00, 1.10)
DISC_PCTS = (10, 50, 90)
SD_GRID = (0.005, 0.010, 0.015)
N_TECH = 3
N_SEEDS = 5
B_E = 10_000
B_S = 2_000
N_SIMS = 1_000
N_PERM = 10_000
N_VERIF = 100
POWER_TARGET = 0.80
CENTRAL = ('p50', 0.010)          # escenario central (discordancia, DE)
KAPPA_GRID = np.linspace(0.0, 1.0, 21)
N_PILOT = 1_000
VERDICTS = ('V1', 'V2', 'V3', 'V4')
FORMS = ('A', 'B')
ALPHA_LEVELS = (ALPHA / 3, ALPHA / 2, ALPHA)


class Log:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, msg=''):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def write(self):
        self.path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8', newline='\n')


# ── F1-Macro vectorizado ──────────────────────────────────────────────────
def boot_weights(rng, B, n):
    """Matriz B × n de multiplicidades de un bootstrap con reemplazo."""
    idx = rng.integers(0, n, size=(B, n))
    flat = (idx + n * np.arange(B)[:, None]).ravel()
    return np.bincount(flat, minlength=B * n).reshape(B, n).astype(np.float64), idx


def _safe_ratio(num, den):
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def f1_macro_counts(tp, fp, fn, tn):
    """F1-Macro desde conteos (arrays de igual forma); 0/0 → 0, como sklearn."""
    f1_pos = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    f1_neg = _safe_ratio(2 * tn, 2 * tn + fn + fp)
    return (f1_pos + f1_neg) / 2


def f1_macro_weighted(W, y, P):
    """F1-Macro de cada columna de P (n × m, 0/1) bajo cada fila de pesos W (B × n)."""
    y = y.astype(np.float64)
    P = P.astype(np.float64)
    tp = W @ (y[:, None] * P)
    fp = W @ ((1 - y)[:, None] * P)
    pos = (W @ y)[:, None]
    neg = (W @ (1 - y))[:, None]
    return f1_macro_counts(tp, fp, pos - tp, neg - fp)


def f1_macro_permuted(Yp, P):
    """F1-Macro de cada columna de P (n × m) frente a cada fila de etiquetas Yp (N × n)."""
    pos = Yp[0].sum()
    neg = Yp.shape[1] - pos
    tp = Yp @ P
    fp = P.sum(axis=0)[None, :] - tp
    return f1_macro_counts(tp, fp, pos - tp, neg - fp)


# ── Pruebas ───────────────────────────────────────────────────────────────
def holm(p, alpha=ALPHA):
    """(rechazo, α_h) por técnica. α_h = α/(m−k+1) según el rango k de p."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p, kind='stable')
    alpha_h = np.empty(m)
    reject = np.zeros(m, dtype=bool)
    going = True
    for k, j in enumerate(order):
        alpha_h[j] = alpha / (m - k)
        going = going and p[j] <= alpha_h[j]
        reject[j] = going
    return reject, alpha_h


def quantile_se(sorted_x, q):
    """SE asintótico del cuantil q de una muestra ordenada (columnas)."""
    B = sorted_x.shape[0]
    k = int(round(np.sqrt(B)))
    j = int(round(q * (B - 1)))
    lo, hi = max(j - k, 0), min(j + k, B - 1)
    spacing = sorted_x[hi] - sorted_x[lo]
    inv_density = spacing * B / (hi - lo)
    return np.sqrt(q * (1 - q) / B) * inv_density


# ── Insumos ───────────────────────────────────────────────────────────────
def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO, capture_output=True, text=True).stdout
        return sha, bool(dirty.strip())
    except OSError:
        return 'desconocido', None


def canonical_values():
    numbers = json.loads(CANON.read_text(encoding='utf-8'))['numbers']
    floors = {k: v['value']['test'] for k, v in numbers.items() if k.startswith('floor.brightness.')}
    return {
        'f1_beto_test': numbers['ref.beto_exp41.f1_macro']['value']['test'],
        'f1_beto_val': numbers['ref.beto_exp41.f1_macro']['value']['val'],
        'f1_emb768_val': numbers['ref.emb768_logreg.f1_macro']['value']['val'],
        'floor': max(floors.values()),
        'floor_id': max(floors, key=floors.get),
        'witcom_delta': numbers['witcom.delta_beto_gs_resnet18.f1_macro']['value']['test'],
        'f32': {k: v['value']['val'] for k, v in numbers.items()
                if k.startswith('f32v0.') and '.seed' in k and k.endswith('.f1_macro')},
    }


def ordered_split(df, index, split):
    ids = index.loc[index['split'] == split, 'sample_id'].sort_values().to_numpy()
    sub = df[df['split'] == split].set_index('sample_id').loc[ids]
    y_idx = index.set_index('sample_id').loc[ids, 'y_true'].to_numpy()
    if not np.array_equal(sub['y_true'].to_numpy(), y_idx):
        raise ValueError(f'y_true difiere del índice maestro en {split}')
    return ids, y_idx.astype(np.int64), (sub['p_dep'].to_numpy(dtype=np.float64) > 0.5).astype(np.int64)


def load_inputs(log):
    index = load_index()
    beto = load_predictions(str(BETO_PRED), index=index)
    emb = load_predictions(str(EMB_PRED), index=index)
    _, y_val, beto_val = ordered_split(beto, index, 'val')
    _, y_test, beto_test = ordered_split(beto, index, 'test')
    _, _, emb_val = ordered_split(emb, index, 'val')   # solo val (§4): test de LogReg-768D no se usa

    f32 = pd.read_csv(F32_PRED, dtype={'sample_id': str, 'split': str, 'technique': str, 'arch': str,
                                       'strategy': str, 'run_id': str})
    if set(f32['split']) != {'val'}:
        raise ValueError('predictions.csv de F3.2 contiene splits distintos de val')
    runs = []
    for run_id, g in f32.groupby('run_id', sort=True):
        _, y_r, pred = ordered_split(g, index, 'val')
        first = g.iloc[0]
        runs.append({'run_id': run_id, 'technique': first['technique'], 'arch': first['arch'],
                     'strategy': first['strategy'], 'seed': int(first['seed']), 'pred': pred})
    log(f'Insumos: y_val {np.bincount(y_val).tolist()}, y_test {np.bincount(y_test).tolist()} (0/1); '
        f'{len(runs)} corridas de F3.2 (solo val)')
    shas = {str(p.relative_to(REPO)).replace('\\', '/'): sha256(p)
            for p in (paths.SAMPLE_INDEX, BETO_PRED, EMB_PRED, F32_PRED, CANON)}
    for k, v in shas.items():
        log(f'  sha256 {k}  {v}')
    return {'y_val': y_val, 'y_test': y_test, 'beto_val': beto_val, 'beto_test': beto_test,
            'emb_val': emb_val, 'runs': runs, 'sha256': shas}


# ── Verificación del F1 vectorizado ───────────────────────────────────────
def verify_f1(rng, data, log):
    from sklearn.metrics import f1_score
    y = data['y_test']
    # BETO-test y cinco columnas aleatorias (no se usa ninguna predicción visual de prueba)
    P = np.column_stack([data['beto_test'], rng.integers(0, 2, size=(len(y), 5))])
    W, idx = boot_weights(rng, N_VERIF, len(y))
    fast = f1_macro_weighted(W, y, P)
    worst = 0.0
    for b in range(N_VERIF):
        for j in range(P.shape[1]):
            ref = f1_score(y[idx[b]], P[idx[b], j], average='macro', zero_division=0)
            worst = max(worst, abs(ref - fast[b, j]))
    log(f'Verificación F1 vectorizado vs sklearn: {N_VERIF} réplicas × {P.shape[1]} columnas, '
        f'máx |dif| = {worst:.3e}')
    if worst > 1e-12:
        raise AssertionError('F1 vectorizado no coincide con sklearn')


# ── Bloque E ──────────────────────────────────────────────────────────────
def block_e(rng, data, canon, log, B=B_E):
    y, beto = data['y_val'], data['beto_val']
    runs = data['runs']
    cols = [beto, data['emb_val']] + [r['pred'] for r in runs]
    P = np.column_stack(cols)
    f1_obs = f1_macro_weighted(np.ones((1, len(y))), y, P)[0]

    if abs(f1_obs[0] - canon['f1_beto_val']) > 1e-12 or abs(f1_obs[1] - canon['f1_emb768_val']) > 1e-12:
        raise AssertionError('F1-val de BETO o de LogReg-768D difiere de canonical_numbers.json')
    worst = 0.0
    for r, f in zip(runs, f1_obs[2:]):
        key = (f"f32v0.{r['technique']}.{r['arch']}.{r['strategy']}.px224_flip.seed{r['seed']}.f1_macro")
        worst = max(worst, abs(canon['f32'][key] - f))
    log(f'E: F1-val recalculado de las {len(runs)} corridas vs canonical_numbers.json, máx |dif| = {worst:.3e}')
    if worst > 1e-12:
        raise AssertionError('F1-val de F3.2 difiere de canonical_numbers.json')

    W, _ = boot_weights(rng, B, len(y))
    F = f1_macro_weighted(W, y, P)
    beto_b = F[:, 0]

    run_rows = []
    for r, f in zip(runs, f1_obs[2:]):
        run_rows.append({'run_id': r['run_id'], 'technique': r['technique'], 'arch': r['arch'],
                         'strategy': r['strategy'], 'seed': r['seed'], 'f1_val': f,
                         'discordancia_beto': float(np.mean(r['pred'] != beto))})
    run_rows.append({'run_id': 'emb768__logreg__none__seed42', 'technique': 'emb768', 'arch': 'logreg',
                     'strategy': 'none', 'seed': 42, 'f1_val': f1_obs[1],
                     'discordancia_beto': float(np.mean(data['emb_val'] != beto))})
    runs_df = pd.DataFrame(run_rows)

    groups = {}
    for j, r in enumerate(runs):
        groups.setdefault((r['technique'], r['arch'], r['strategy']), []).append(j + 2)
    groups[('emb768', 'logreg', 'none')] = [1]

    rows = []
    for (tech, arch, strat), js in groups.items():
        vis_b = F[:, js].mean(axis=1)
        vis_obs = f1_obs[js].mean()
        stats = {'delta': (f1_obs[0] - vis_obs, beto_b - vis_b)}
        for rv in R_VALUES:
            stats[f'theta_r{rv:.2f}'] = ((vis_obs - F1_AZAR) - rv * (f1_obs[0] - F1_AZAR),
                                         (vis_b - F1_AZAR) - rv * (beto_b - F1_AZAR))
        disc = runs_df.loc[runs_df['technique'].eq(tech) & runs_df['arch'].eq(arch)
                           & runs_df['strategy'].eq(strat), 'discordancia_beto']
        sd = float(np.std(f1_obs[js], ddof=1)) if len(js) > 1 else np.nan
        for name, (point, boot) in stats.items():
            for a_h in ALPHA_LEVELS:
                lo, hi = np.quantile(boot, [a_h, 1 - a_h])
                rows.append({'technique': tech, 'arch': arch, 'strategy': strat, 'n_semillas': len(js),
                             'f1_val_media': vis_obs, 'de_semillas': sd,
                             'discordancia_media': float(disc.mean()), 'estadistico': name,
                             'alpha_h': a_h, 'confianza': 1 - 2 * a_h, 'punto': point,
                             'li': lo, 'ls': hi, 'ancho': hi - lo})
    emp = pd.DataFrame(rows)

    vis_runs = runs_df[runs_df['technique'] != 'emb768']
    pct = {f'p{q}': float(np.percentile(vis_runs['discordancia_beto'], q)) for q in DISC_PCTS}
    sds = emp.drop_duplicates(['technique', 'arch', 'strategy']).query("technique != 'emb768'")['de_semillas']
    log(f'E: discordancia BETO–visual (54 corridas): ' + ', '.join(f'{k} = {v:.4f}' for k, v in pct.items())
        + f'; LogReg-768D = {runs_df.iloc[-1]["discordancia_beto"]:.4f}')
    log(f'E: DE entre semillas (18 configs, ddof=1): min {sds.min():.4f}, mediana {sds.median():.4f}, '
        f'max {sds.max():.4f}')
    w = emp.query("estadistico == 'delta' and technique != 'emb768'").groupby('alpha_h')['ancho']
    log('E: ancho del IC de Δ (18 configs): ' + '; '.join(
        f'α_h={a:.4f}: mediana {s.median():.4f} [{s.min():.4f}, {s.max():.4f}]' for a, s in w))
    return emp, runs_df, pct


# ── Bloque S: calibración ─────────────────────────────────────────────────
def expected_f1(acc0, acc1, n0, n1):
    tp, fn = n1 * acc1, n1 * (1 - acc1)
    tn, fp = n0 * acc0, n0 * (1 - acc0)
    return float(f1_macro_counts(np.array(tp), np.array(fp), np.array(fn), np.array(tn)))


class Beto:
    """Estructura fija de prueba: y_test y aciertos de BETO."""

    def __init__(self, y, pred):
        self.y = y
        self.b = (pred == y).astype(np.int64)
        self.n0, self.n1 = int((y == 0).sum()), int((y == 1).sum())
        self.R = np.array([self.b[y == 0].mean(), self.b[y == 1].mean()])
        self.f1 = float(f1_macro_weighted(np.ones((1, len(y))), y, pred[:, None])[0, 0])

    def q(self, eta):
        return expit(logit(self.R) + eta)

    def f1_of(self, a, eta):
        acc = a * self.R + (1 - a) * self.q(eta)
        return expected_f1(acc[0], acc[1], self.n0, self.n1)

    def disc_of(self, a, eta):
        qi = self.q(eta)[self.y]
        return float((1 - a) * np.mean(self.b * (1 - qi) + (1 - self.b) * qi))


def solve_a_eta(beto, f1_target, disc_target, eta_lim=12.0):
    """(a, η) con F1 esperado = objetivo y discordancia esperada = objetivo; None si no existen."""
    def eta_for(a):
        g = lambda e: beto.f1_of(a, e) - f1_target
        lo, hi = g(-eta_lim), g(eta_lim)
        if lo > 0 or hi < 0:
            return None
        return brentq(g, -eta_lim, eta_lim, xtol=1e-12)

    grid = np.linspace(0.0, 0.995, 400)
    vals = []
    for a in grid:
        e = eta_for(a)
        vals.append(np.nan if e is None else beto.disc_of(a, e) - disc_target)
    vals = np.array(vals)
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] <= 0:
            a = brentq(lambda x: beto.disc_of(x, eta_for(x)) - disc_target, grid[i], grid[i + 1], xtol=1e-12)
            return a, eta_for(a)
    return None


def draw_predictions(rng, beto, a, eta, kappa, n_rep, n_tech=N_TECH, n_seeds=N_SEEDS):
    """ŷ con forma (n_rep, n, n_tech, n_seeds)."""
    n = len(beto.y)
    q = beto.q(eta)[beto.y]
    m = rng.random((n_rep, n, n_tech)) < a
    k = rng.random((n_rep, n, n_tech)) < kappa
    z_sh = rng.random((n_rep, n, n_tech)) < q[None, :, None]
    z_own = rng.random((n_rep, n, n_tech, n_seeds)) < q[None, :, None, None]
    c = np.where(m[..., None], beto.b[None, :, None, None].astype(bool),
                 np.where(k[..., None], z_sh[..., None], z_own))
    y = beto.y[None, :, None, None]
    return np.where(c, y, 1 - y).astype(np.int64)


def obs_f1_batch(beto, yhat):
    """F1-Macro observado de ŷ (..., n, t, s) → (..., t, s)."""
    y = beto.y[:, None, None]
    tp = ((yhat == 1) & (y == 1)).sum(axis=-3)
    fp = ((yhat == 1) & (y == 0)).sum(axis=-3)
    return f1_macro_counts(tp, fp, beto.n1 - tp, beto.n0 - fp)


def calibrate_kappa(rng, beto, a, eta, sd_target):
    """κ con DE media entre semillas = objetivo (interpolación sobre KAPPA_GRID)."""
    sds = []
    for kappa in KAPPA_GRID:
        f1 = obs_f1_batch(beto, draw_predictions(rng, beto, a, eta, kappa, N_PILOT, n_tech=1))
        sds.append(float(np.std(f1, axis=-1, ddof=1).mean()))
    sds = np.array(sds)
    if sd_target >= sds[0]:
        return 0.0, False, sds
    # sds decrece con κ (κ = 1 → DE 0); se fuerza monotonía para interpolar
    mono = np.minimum.accumulate(sds)
    order = np.argsort(mono)
    return float(np.interp(sd_target, mono[order], KAPPA_GRID[order])), True, sds


# ── Bloque S: una réplica ─────────────────────────────────────────────────
def simulate_one(rng, beto, Yp, a, eta, kappa, floor, B=B_S):
    n = len(beto.y)
    yhat = draw_predictions(rng, beto, a, eta, kappa, 1)[0]            # (n, 3, 5)
    P_vis = yhat.reshape(n, N_TECH * N_SEEDS)
    P = np.column_stack([np.where(beto.b == 1, beto.y, 1 - beto.y), P_vis])
    W, _ = boot_weights(rng, B, n)
    F = f1_macro_weighted(W, beto.y, P)                                 # (B, 16)
    beto_b = F[:, 0]
    vis_b = F[:, 1:].reshape(B, N_TECH, N_SEEDS).mean(axis=2)           # (B, 3)

    f_obs = obs_f1_batch(beto, yhat)                                    # (3, 5)
    t_obs = f_obs.mean(axis=1)

    # Piso: permutación con nulo desplazado a 0.529
    null = f1_macro_permuted(Yp, P_vis.astype(np.float32)).reshape(-1, N_TECH, N_SEEDS).mean(axis=2)
    null = null + (floor - null.mean(axis=0))
    p_floor = (1 + (null >= t_obs[None, :]).sum(axis=0)) / (Yp.shape[0] + 1)
    floor_sup, _ = holm(p_floor)

    delta_b = beto_b[:, None] - vis_b
    q_levels = np.array(sorted(set(ALPHA_LEVELS) | {1 - x for x in ALPHA_LEVELS}))
    qi = {round(x, 10): i for i, x in enumerate(q_levels)}
    verdicts = np.zeros((2, len(R_VALUES), N_TECH, 4), dtype=bool)
    se_bound = np.zeros((2, len(R_VALUES)))
    delta_sorted = np.sort(delta_b, axis=0)
    dq = np.quantile(delta_b, q_levels, axis=0)                         # (6, 3)
    for ri, rv in enumerate(R_VALUES):
        delta_m = (1 - rv) * (beto.f1 - F1_AZAR)
        theta_b = (vis_b - F1_AZAR) - rv * (beto_b[:, None] - F1_AZAR)
        tq = np.quantile(theta_b, q_levels, axis=0)
        for fi, form in enumerate(FORMS):
            if form == 'A':
                p = (delta_b >= delta_m).mean(axis=0)
            else:
                p = (theta_b <= 0).mean(axis=0)
            rej, a_h = holm(p)
            for t in range(N_TECH):
                if form == 'A':
                    loss = dq[qi[round(a_h[t], 10)], t] > delta_m
                else:
                    loss = tq[qi[round(1 - a_h[t], 10)], t] < 0
                v = 2 if not floor_sup[t] else (0 if rej[t] else (1 if loss else 3))
                verdicts[fi, ri, t, v] = True
            if form == 'A':
                se_bound[fi, ri] = quantile_se(delta_sorted, 1 - ALPHA / 3).mean()
            else:
                se_bound[fi, ri] = quantile_se(np.sort(theta_b, axis=0), ALPHA / 3).mean()

    realized = {
        'ret': float(((t_obs - F1_AZAR) / (beto.f1 - F1_AZAR)).mean()),
        'sd': float(np.std(f_obs, axis=1, ddof=1).mean()),
        'disc': float((yhat != P[:, 0][:, None, None]).mean()),
    }
    return verdicts, floor_sup, se_bound, realized


def run_scenario(args):
    """Un escenario de la malla: calibración y n_sims réplicas."""
    (scen, seed_calib, seed_sim, beto_y, beto_pred, Yp, floor, n_sims, B) = args
    beto = Beto(beto_y, beto_pred)
    rho, disc_label, disc_target, sd_target = scen
    f1_target = F1_AZAR + rho * (beto.f1 - F1_AZAR)
    base = {'rho': rho, 'f1_objetivo': f1_target, 'disc_etiqueta': disc_label,
            'disc_objetivo': disc_target, 'de_objetivo': sd_target}
    t0 = time.perf_counter()
    sol = solve_a_eta(beto, f1_target, disc_target)
    if sol is None:
        return {**base, 'factible': False, 'de_factible': False, 'a': np.nan, 'eta': np.nan,
                'kappa': np.nan, 'n_sims': 0, 't_calib': time.perf_counter() - t0, 't_sim': 0.0}, None
    a, eta = sol
    kappa, sd_ok, _ = calibrate_kappa(np.random.default_rng(seed_calib), beto, a, eta, sd_target)
    t_calib = time.perf_counter() - t0

    rng = np.random.default_rng(seed_sim)
    V = np.zeros((n_sims, 2, len(R_VALUES), N_TECH, 4), dtype=bool)
    FS = np.zeros((n_sims, N_TECH), dtype=bool)
    SE = np.zeros((n_sims, 2, len(R_VALUES)))
    RL = np.zeros((n_sims, 3))
    t1 = time.perf_counter()
    for i in range(n_sims):
        V[i], FS[i], SE[i], rl = simulate_one(rng, beto, Yp, a, eta, kappa, floor, B)
        RL[i] = (rl['ret'], rl['sd'], rl['disc'])
    t_sim = time.perf_counter() - t1
    info = {**base, 'factible': True, 'de_factible': sd_ok, 'a': a, 'eta': eta, 'kappa': kappa,
            'n_sims': n_sims, 'ret_realizada': RL[:, 0].mean(), 'de_realizada': RL[:, 1].mean(),
            'disc_realizada': RL[:, 2].mean(), 't_calib': t_calib, 't_sim': t_sim}
    return info, (V, FS, SE)


def summarize(infos_results):
    grid_rows, floor_rows = [], []
    for info, res in infos_results:
        meta = {k: v for k, v in info.items() if k not in ('t_calib', 't_sim')}
        if res is None:
            for form in FORMS:
                for rv in R_VALUES:
                    grid_rows.append({**meta, 'formulacion': form, 'r': rv})
            floor_rows.append({**meta})
            continue
        V, FS, SE = res
        n = V.shape[0]
        per_sim = V.mean(axis=3)                                       # (n, 2, R, 4): promedio sobre técnicas
        fs_sim = FS.mean(axis=1)
        for fi, form in enumerate(FORMS):
            for ri, rv in enumerate(R_VALUES):
                row = {**meta, 'formulacion': form, 'r': rv}
                for vi, v in enumerate(VERDICTS):
                    row[f'P_{v}'] = per_sim[:, fi, ri, vi].mean()
                    row[f'se_{v}'] = per_sim[:, fi, ri, vi].std(ddof=1) / np.sqrt(n)
                row['se_mc_limite'] = SE[:, fi, ri].mean()
                row['P_piso'] = fs_sim.mean()
                grid_rows.append(row)
        floor_rows.append({**meta, 'P_piso': fs_sim.mean(), 'se_piso': fs_sim.std(ddof=1) / np.sqrt(n)})
    return pd.DataFrame(grid_rows), pd.DataFrame(floor_rows)


def min_detectable(grid, f1_beto):
    rows = []
    for (form, rv, dl, sd), g in grid.groupby(['formulacion', 'r', 'disc_etiqueta', 'de_objetivo'], sort=True):
        g = g[g['factible']].sort_values('rho')
        rho, pw = g['rho'].to_numpy(), g['P_V1'].to_numpy()
        rho_min, nota = np.nan, 'no alcanzada en la malla'
        hit = np.flatnonzero(pw >= POWER_TARGET)
        if len(hit):
            j = hit[0]
            if j == 0:
                rho_min, nota = rho[0], 'alcanzada en el menor ρ factible (cota superior)'
            else:
                rho_min = rho[j - 1] + (POWER_TARGET - pw[j - 1]) * (rho[j] - rho[j - 1]) / (pw[j] - pw[j - 1])
                nota = 'interpolación lineal'
        rows.append({'formulacion': form, 'r': rv, 'disc_etiqueta': dl, 'de_objetivo': sd,
                     'rho_min': rho_min, 'f1_min': F1_AZAR + rho_min * (f1_beto - F1_AZAR)
                     if np.isfinite(rho_min) else np.nan,
                     'rho_factibles': ' '.join(f'{x:.2f}' for x in rho), 'nota': nota})
    return pd.DataFrame(rows)


# ── Figura ────────────────────────────────────────────────────────────────
def plot_curves(grid, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {0.50: '#2a78d6', 0.75: '#eb6834', 0.80: '#1baf7a'}
    markers = {0.50: 'o', 0.75: 's', 0.80: '^'}
    titles = {'A': 'A. Margen fijo', 'B': 'B. Retención lineal'}
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    g = grid[grid['factible']]
    for ax, form in zip(axes, FORMS):
        for rv in R_VALUES:
            sub = g[(g['formulacion'] == form) & (g['r'] == rv)]
            band = sub.groupby('rho')['P_V1'].agg(['min', 'max'])
            ax.fill_between(band.index, band['min'], band['max'], color=colors[rv], alpha=0.12, linewidth=0)
            c = sub[(sub['disc_etiqueta'] == CENTRAL[0]) & np.isclose(sub['de_objetivo'], CENTRAL[1])]
            c = c.sort_values('rho')
            ax.plot(c['rho'], c['P_V1'], color=colors[rv], lw=2, marker=markers[rv], ms=8,
                    markeredgecolor='#fcfcfb', markeredgewidth=1.5, label=f'r = {rv:.2f}')
            if len(c):
                ax.annotate(f'r = {rv:.2f}', (c['rho'].iloc[-1], c['P_V1'].iloc[-1]), xytext=(6, 0),
                            textcoords='offset points', va='center', fontsize=9, color='#0b0b0b')
        ax.axhline(POWER_TARGET, color='#52514e', lw=1, ls='--')
        ax.text(RHO_GRID[0], POWER_TARGET + 0.02, 'potencia 0.80', fontsize=8, color='#52514e')
        ax.set_title(titles[form], loc='left', fontsize=11)
        ax.set_xlabel('Retención verdadera ρ')
        ax.set_xlim(0.2, 1.25)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis='y', color='#e5e4e0', lw=0.8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('P(V1): conservación declarada')
    axes[1].legend(loc='center left', bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=9)
    fig.suptitle('Potencia del contraste de no inferioridad frente a BETO (n = 114, 5 semillas, Holm sobre 3 técnicas)',
                 fontsize=11, x=0.01, ha='left')
    fig.text(0.01, 0.005, 'Línea: escenario central (discordancia p50, DE entre semillas 0.010). '
             'Banda: mínimo y máximo sobre los escenarios factibles de discordancia y DE.\n'
             'ρ = 0.25 no es factible con la discordancia BETO–visual observada en validación.', fontsize=8, color='#52514e')
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'curva_potencia.{ext}', dpi=200, metadata={'CreationDate': None} if ext == 'pdf' else None)
    plt.close(fig)


# ── Principal ─────────────────────────────────────────────────────────────
def scenarios(pct):
    return [(rho, dl, pct[dl], sd) for rho in RHO_GRID for dl in (f'p{q}' for q in DISC_PCTS) for sd in SD_GRID]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--dry-run', action='store_true', help='estimación de tiempo; no escribe salidas')
    ap.add_argument('--out', type=Path, default=OUT_DIR)
    ap.add_argument('--n-sims', type=int, default=N_SIMS)
    ap.add_argument('--B', type=int, default=B_S)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--solo-figura', action='store_true',
                    help='regenera curva_potencia.png/.pdf desde malla_escenarios.csv, sin simular')
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if args.solo_figura:
        plot_curves(pd.read_csv(args.out / 'malla_escenarios.csv'), args.out)
        return

    log = Log(args.out / ('potencia_dryrun.log' if args.dry_run else 'potencia.log'))
    t_start = time.perf_counter()
    sha, dirty = git_head()
    import scipy
    import sklearn
    log(f'F1.1 — análisis de potencia ({"dry-run" if args.dry_run else "corrida completa"})')
    log(f'Inicio: {time.strftime("%Y-%m-%d %H:%M:%S %z")}; commit {sha}{" (árbol con cambios)" if dirty else ""}')
    log(f'Entorno: Python {platform.python_version()} · numpy {np.__version__} · scipy {scipy.__version__} · '
        f'pandas {pd.__version__} · scikit-learn {sklearn.__version__} · {platform.platform()} · '
        f'{platform.processor() or platform.machine()} · {os.cpu_count()} núcleos · CPU · jobs={args.jobs}')
    log(f'Parámetros: semilla maestra {MASTER_SEED}; α = {ALPHA}; r ∈ {R_VALUES}; ρ ∈ {RHO_GRID}; '
        f'discordancia p{DISC_PCTS}; DE ∈ {SD_GRID}; B_E = {B_E}; B_S = {args.B}; sims = {args.n_sims}; '
        f'permutaciones = {N_PERM}')

    canon = canonical_values()
    log(f'canonical_numbers.json: F1_BETO test = {canon["f1_beto_test"]!r}, val = {canon["f1_beto_val"]!r}; '
        f'piso = {canon["floor"]} ({canon["floor_id"]}); Δ WITCOM = {canon["witcom_delta"]!r}')
    data = load_inputs(log)
    beto = Beto(data['y_test'], data['beto_test'])
    if abs(beto.f1 - canon['f1_beto_test']) > 1e-12:
        raise AssertionError('F1 de BETO-test recalculado difiere de canonical_numbers.json')
    log('δ(r) = (1 − r)(F1_BETO − 0.5): ' + ', '.join(
        f'r={rv:.2f}: {(1 - rv) * (beto.f1 - F1_AZAR):.4f}' for rv in R_VALUES))

    ss = np.random.SeedSequence(MASTER_SEED)
    s_verif, s_e, s_perm, s_calib, s_sim = ss.spawn(5)
    verify_f1(np.random.default_rng(s_verif), data, log)

    t0 = time.perf_counter()
    emp, runs_df, pct = block_e(np.random.default_rng(s_e), data, canon, log)
    t_e = time.perf_counter() - t0
    log(f'E: {t_e:.1f} s (B = {B_E})')

    rng_perm = np.random.default_rng(s_perm)
    Yp = rng_perm.permuted(np.tile(beto.y, (N_PERM, 1)), axis=1).astype(np.float32)

    scen = scenarios(pct)
    calib_seeds = s_calib.spawn(len(scen))
    sim_seeds = s_sim.spawn(len(scen))

    if args.dry_run:
        n_try = 20
        j = scen.index((1.00, 'p50', pct['p50'], 0.010))
        info, _ = run_scenario((scen[j], calib_seeds[j], sim_seeds[j], beto.y, data['beto_test'], Yp,
                                canon['floor'], n_try, args.B))
        per_sim = info['t_sim'] / n_try
        total_s = t_e + len(scen) * (info['t_calib'] + per_sim * args.n_sims)
        log(f'Dry-run: escenario ρ=1.00, p50, DE 0.010 → a = {info["a"]:.4f}, η = {info["eta"]:.4f}, '
            f'κ = {info["kappa"]:.4f}; calibración {info["t_calib"]:.1f} s; {n_try} réplicas en '
            f'{info["t_sim"]:.2f} s ({per_sim * 1000:.1f} ms/réplica)')
        log(f'Estimación: E {t_e:.0f} s + {len(scen)} escenarios × ({info["t_calib"]:.1f} s calibración + '
            f'{args.n_sims} × {per_sim * 1000:.1f} ms) = {total_s / 60:.1f} min en 1 proceso '
            f'(factor de extrapolación {args.n_sims / n_try:.0f}× en réplicas, {len(scen)}× en escenarios)')
        log(f'Duración del dry-run: {time.perf_counter() - t_start:.1f} s')
        log.write()
        return

    emp.to_csv(args.out / 'empirico.csv', index=False, lineterminator='\n')
    runs_df.to_csv(args.out / 'empirico_corridas.csv', index=False, lineterminator='\n')

    tasks = [(scen[j], calib_seeds[j], sim_seeds[j], beto.y, data['beto_test'], Yp, canon['floor'],
              args.n_sims, args.B) for j in range(len(scen))]
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(run_scenario, tasks))
    else:
        for t in tasks:
            results.append(run_scenario(t))
    for info, _ in results:
        if info['factible']:
            log(f'S: ρ={info["rho"]:.2f} {info["disc_etiqueta"]} DE={info["de_objetivo"]:.3f}: '
                f'a={info["a"]:.4f} η={info["eta"]:+.4f} κ={info["kappa"]:.3f}'
                f'{"" if info["de_factible"] else " (DE no alcanzable)"}; realizadas ret={info["ret_realizada"]:.3f} '
                f'DE={info["de_realizada"]:.4f} disc={info["disc_realizada"]:.4f}; '
                f'{info["t_calib"] + info["t_sim"]:.0f} s')
        else:
            log(f'S: ρ={info["rho"]:.2f} {info["disc_etiqueta"]} DE={info["de_objetivo"]:.3f}: no factible')

    grid, floor_df = summarize(results)
    grid['delta'] = np.where(grid['formulacion'] == 'A', (1 - grid['r']) * (beto.f1 - F1_AZAR), np.nan)
    if grid.loc[grid['factible'], [f'P_{v}' for v in VERDICTS]].sum(axis=1).sub(1).abs().max() > 1e-9:
        raise AssertionError('P(V1..V4) no suma 1')
    grid.to_csv(args.out / 'malla_escenarios.csv', index=False, lineterminator='\n')
    floor_df.to_csv(args.out / 'potencia_piso.csv', index=False, lineterminator='\n')
    rmin = min_detectable(grid, beto.f1)
    rmin.to_csv(args.out / 'retencion_minima.csv', index=False, lineterminator='\n')
    plot_curves(grid, args.out)

    central = rmin[(rmin['disc_etiqueta'] == CENTRAL[0]) & np.isclose(rmin['de_objetivo'], CENTRAL[1])]
    for _, r in central.iterrows():
        log(f'D (central): {r["formulacion"]}, r={r["r"]:.2f}: ρ_min = {r["rho_min"]:.3f} '
            f'(F1 = {r["f1_min"]:.4f}; {r["nota"]})')
    log(f'Duración total: {(time.perf_counter() - t_start) / 60:.1f} min')
    log.write()


if __name__ == '__main__':
    main()
