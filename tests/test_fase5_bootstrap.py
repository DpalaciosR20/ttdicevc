"""Fase 5 — bootstrap y F1 vectorizado (datos sintéticos)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fase5_sintetico import CFG, N, labels  # noqa: E402
from src.protocol import bootstrap as bs  # noqa: E402


def test_f1_vectorizado_igual_a_sklearn_replica_a_replica():
    rng = np.random.default_rng(1)
    y = labels()
    B = 250
    idx = rng.integers(0, N, size=(B, N))
    W = bs.weights(idx, N)
    P = np.column_stack([rng.integers(0, 2, N), (rng.uniform(size=N) < 0.1).astype(int),
                         np.zeros(N, int),            # nunca predice la clase 1
                         np.ones(N, int),             # nunca predice la clase 0
                         y.copy()])                   # perfecto
    F = bs.f1_macro(W, y, P)
    for b in range(B):
        for j in range(P.shape[1]):
            ref = f1_score(y[idx[b]], P[idx[b], j], average='macro', labels=[0, 1], zero_division=0)
            assert F[b, j] == pytest.approx(ref, abs=1e-12), (b, j)


def test_f1_replica_sin_una_clase_verdadera():
    """Réplica degenerada (solo clase 0 en y): labels=[0,1] y zero_division=0 como sklearn."""
    y = labels()
    idx = np.zeros((1, N), int)                       # todas las filas = muestra 0 (clase 0)
    W = bs.weights(idx, N)
    for yhat in (np.zeros(N, int), np.ones(N, int)):
        ref = f1_score(y[idx[0]], yhat[idx[0]], average='macro', labels=[0, 1], zero_division=0)
        assert bs.f1_macro(W, y, yhat)[0] == pytest.approx(ref, abs=1e-12)


def test_f1_semillas_media_de_f1_por_semilla():
    rng = np.random.default_rng(2)
    y = labels()
    idx = rng.integers(0, N, size=(50, N))
    W = bs.weights(idx, N)
    P = rng.integers(0, 2, (N, 5))
    m = bs.f1_macro_seeds_mean(W, y, P)
    for b in range(50):
        ref = np.mean([f1_score(y[idx[b]], P[idx[b], s], average='macro', labels=[0, 1], zero_division=0)
                       for s in range(5)])
        assert m[b] == pytest.approx(ref, abs=1e-12)


def test_indices_forma_y_llamada_del_config():
    idx = bs.indices(CFG)
    assert idx.shape == (10000, 114) and idx.dtype.kind == 'i' and idx.min() >= 0 and idx.max() <= 113
    ref = np.random.default_rng(2026).integers(0, 114, size=(10000, 114))
    assert np.array_equal(idx, ref)
    assert np.array_equal(idx[:3], np.random.default_rng(2026).integers(0, 114, size=(10000, 114))[:3])
    # equivalencia con rng.choice (bloque A, A1)
    assert np.array_equal(idx, np.random.default_rng(2026).choice(114, size=(10000, 114), replace=True))
    # reproducible entre llamadas
    assert np.array_equal(idx, bs.indices(CFG))


def test_indices_rechaza_config_distinto():
    for mutate in (lambda e: e.__setitem__('call', 'rng.choice(114, size=(10000, 114))'),
                   lambda e: e.__setitem__('seed', 1),
                   lambda e: e.__setitem__('size', [5000, 114]),
                   lambda e: e.__setitem__('row_order', 'idx')):
        c = copy.deepcopy(CFG)
        mutate(c['enmienda_4']['bootstrap'])
        with pytest.raises(bs.BootstrapConfigError):
            bs.indices(c)


def test_pesos():
    rng = np.random.default_rng(3)
    idx = rng.integers(0, N, size=(20, N))
    W = bs.weights(idx, N)
    assert (W.sum(axis=1) == N).all()
    for b in range(20):
        assert np.array_equal(W[b], np.bincount(idx[b], minlength=N))


def test_umbral_estricto():
    assert bs.predict([0.5, 0.5000001, 0.49], 0.5).tolist() == [0, 1, 0]


def test_percentil_con_menos_infinito():
    x = np.concatenate([np.full(300, -np.inf), np.linspace(0, 1, 9700)])
    lo, hi = bs.percentile_interval(x, 0.025, 0.975)
    assert lo == float('-inf') and hi == pytest.approx(np.quantile(x[300:], (0.975 * 9999 - 300) / 9699))
    y = np.linspace(0, 1, 101)
    assert bs.percentile_interval(y, 0.025, 0.975) == tuple(np.quantile(y, [0.025, 0.975], method='linear'))
    with pytest.raises(ValueError):
        bs.percentile_interval([np.nan, 1.0], 0.025, 0.975)
