"""
bootstrap.py
────────────
Réplicas pareadas de §5.2 y F1-Macro vectorizado (Fase 5).

  - Índices de la Enmienda 4, A1: numpy.random.default_rng(seed) y
    exactamente la llamada de configs/protocolo.json → enmienda_4.bootstrap.call
    («rng.integers(0, 114, size=(10000, 114))»), sobre las filas de prueba
    ordenadas por sample_id.
  - Matriz de multiplicidades W (B × n): W[b, i] = veces que la muestra i
    aparece en la réplica b. Toda métrica de una réplica se calcula con conteos
    ponderados por W (tp = W @ [y=1 ∧ ŷ=1], etc.), equivalentes a recalcularla
    sobre las filas remuestreadas.
  - F1-Macro con labels = [0, 1] y zero_division = 0, como
    sklearn.metrics.f1_score(average='macro', labels=[0, 1], zero_division=0):
    F1_c = 2·tp_c / (2·tp_c + fp_c + fn_c), y 0 si el denominador es 0.
  - F1 de un clasificador visual con 5 semillas = media de los F1 por semilla
    en cada réplica (§4.1, §5.2).

No lee datos reales: recibe vectores ya cargados (src/protocol/insumos.py).
"""

import numpy as np


class BootstrapConfigError(ValueError):
    """El bloque enmienda_4.bootstrap no coincide con lo que este módulo implementa."""


def bootstrap_spec(cfg):
    """(seed, low, high, size) de configs/protocolo.json, comprobados contra la
    llamada textual de A1 y contra contraste.bootstrap."""
    e = cfg['enmienda_4']['bootstrap']
    c = cfg['contraste']['bootstrap']
    low, high, size = e['low'], e['high'], tuple(e['size'])
    call = f'rng.integers({low}, {high}, size=({size[0]}, {size[1]}))'
    problems = []
    if e['call'] != call:
        problems.append(f'call {e["call"]!r} != {call!r}')
    if (e['rng'], e['seed']) != (c['rng'], c['seed']) or e['rng'] != 'numpy.random.default_rng':
        problems.append(f'rng/seed {e["rng"]}({e["seed"]}) != contraste {c["rng"]}({c["seed"]})')
    if size[0] != c['B'] or high != size[1] or low != 0:
        problems.append(f'size {size}, low {low}, high {high} frente a B = {c["B"]}')
    if e['row_order'] != 'sample_id':
        problems.append(f'row_order {e["row_order"]!r}')
    if problems:
        raise BootstrapConfigError('; '.join(problems))
    return e['seed'], low, high, size


def indices(cfg):
    """Índices (B × n) de A1: default_rng(seed).integers(low, high, size=size)."""
    seed, low, high, size = bootstrap_spec(cfg)
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=size)


def weights(idx, n=None):
    """Matriz de multiplicidades (B × n, float64) de una matriz de índices (B × n)."""
    idx = np.asarray(idx)
    B, k = idx.shape
    n = k if n is None else n
    flat = (idx + n * np.arange(B)[:, None]).ravel()
    return np.bincount(flat, minlength=B * n).reshape(B, n).astype(np.float64)


def observed_weights(n):
    """Una sola «réplica» con todas las muestras una vez (estimador observado)."""
    return np.ones((1, n), dtype=np.float64)


def _ratio(num, den):
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape, dtype=np.float64), where=den > 0)


def f1_macro_from_counts(tp, fp, fn, tn):
    """F1-Macro binario desde conteos (arrays de igual forma); 0/0 → 0."""
    return (_ratio(2 * tp, 2 * tp + fp + fn) + _ratio(2 * tn, 2 * tn + fn + fp)) / 2


def f1_macro(W, y, yhat):
    """F1-Macro de cada columna de yhat (n × m, 0/1) bajo cada fila de W (B × n) → (B × m).

    yhat puede ser un vector (n,): entonces devuelve (B,)."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    vector = yhat.ndim == 1
    P = yhat[:, None] if vector else yhat
    if not np.isin(P, (0.0, 1.0)).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError('y e ŷ deben ser 0/1')
    tp = W @ (y[:, None] * P)
    fp = W @ ((1 - y)[:, None] * P)
    pos = (W @ y)[:, None]
    neg = (W @ (1 - y))[:, None]
    out = f1_macro_from_counts(tp, fp, pos - tp, neg - fp)
    return out[:, 0] if vector else out


def f1_macro_seeds_mean(W, y, yhat_seeds):
    """Media de los F1 por semilla en cada réplica: yhat_seeds (n × S) → (B,)."""
    return f1_macro(W, y, yhat_seeds).mean(axis=1)


def predict(p_dep, threshold):
    """ŷ = 1 si y solo si p_dep > umbral (§4.1)."""
    return (np.asarray(p_dep, dtype=np.float64) > threshold).astype(np.int64)


def percentile_interval(reps, lo, hi):
    """Cuantiles lo y hi con interpolación lineal (numpy «linear»).

    Admite −inf (Enmienda 4, A6): si alguno de los dos estadísticos de orden que
    interpola un cuantil es −inf, el cuantil es −inf (límite de la interpolación),
    en vez del nan que daría numpy."""
    raw = np.asarray(reps, dtype=np.float64)
    if np.isnan(raw).any() or np.isposinf(raw).any():
        raise ValueError('réplicas con nan o +inf')
    # Sin −inf: exactamente np.quantile(reps, [lo, hi]), como verdict.percentile_interval
    q = np.quantile(raw, [lo, hi], method='linear') if not np.isneginf(raw).any() else None
    if q is not None:
        return float(q[0]), float(q[1])
    x = np.sort(raw)
    out = []
    for p in (lo, hi):
        j = int(np.floor((len(x) - 1) * p))
        out.append(float('-inf') if np.isneginf(x[j]) else float(np.quantile(x, p, method='linear')))
    return tuple(out)
