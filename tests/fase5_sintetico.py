"""Insumos SINTÉTICOS para las pruebas de la Fase 5 (no lee ninguna predicción real).

El tamaño reproduce el real: n = 114 (70/44), 3 técnicas × 3 fases × 5 semillas,
referencias de una semilla y un train de 908 muestras con n_tokens_beto,
truncated_128 y u (5 componentes)."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CFG = json.loads((REPO / 'configs' / 'protocolo.json').read_text(encoding='utf-8'))
CANON = json.loads((REPO / 'results' / 'canonical_numbers.json').read_text(encoding='utf-8'))['numbers']
TECHS = ('gray_scott', 'perlin', 'lorenz')
N, N0, N1 = 114, 70, 44


def cfg(sin_enmienda_5=False):
    """Config real (con la Enmienda 5) o, para probar la negativa, sin el bloque enmienda_5."""
    c = copy.deepcopy(CFG)
    if sin_enmienda_5:
        c.pop('enmienda_5')
    return c


def labels():
    return np.array([0] * N0 + [1] * N1)


def p_with_errors(y, k, rng, margin=0.3):
    """p_dep con exactamente k errores (tras umbral 0.5) en posiciones aleatorias."""
    yhat = y.copy()
    flip = rng.choice(len(y), size=k, replace=False)
    yhat[flip] = 1 - yhat[flip]
    return np.where(yhat == 1, 0.5 + margin * rng.uniform(0.2, 1, len(y)), 0.5 - margin * rng.uniform(0.2, 1, len(y)))


def p_random(n, rng):
    return rng.uniform(0, 1, n)


class Ins:
    """Mismo contrato que src.protocol.insumos.Insumos."""

    def __init__(self, y, visual, refs, train, test_index, u_train):
        self.y, self.visual, self.refs = y, visual, refs
        self.train, self.test_index, self.u_train = train, test_index, u_train
        self.files = {}
        self.sample_id = list(test_index['sample_id'])


def make(seed=0, img_errors=None, beto_errors=30, perm_random=True, second_errors=40):
    """img_errors: {técnica: k errores por semilla} del finalista (por omisión 25)."""
    rng = np.random.default_rng(seed)
    y = labels()
    img_errors = img_errors or {t: 25 for t in TECHS}
    visual = {}
    for t in TECHS:
        k = img_errors[t]
        visual[('final', t)] = {'arch': 'a', 'strategy': 's', 'seeds': CFG['modelos']['seeds'],
                                'p': np.column_stack([p_with_errors(y, k, rng) if k is not None else p_random(N, rng)
                                                      for _ in range(5)])}
        visual[('control_permutacion', t)] = {
            'arch': 'a', 'strategy': 's', 'seeds': CFG['modelos']['seeds'],
            'p': np.column_stack([p_random(N, rng) if perm_random else p_with_errors(y, 50, rng) for _ in range(5)])}
        visual[('sensibilidad', t)] = {'arch': 'b', 'strategy': 's', 'seeds': CFG['modelos']['seeds'],
                                       'p': np.column_stack([p_with_errors(y, second_errors, rng) for _ in range(5)])}
    refs = {'beto': p_with_errors(y, beto_errors, rng), 'emb768': p_with_errors(y, beto_errors + 3, rng),
            'techo5d': p_with_errors(y, beto_errors + 6, rng), 'majority': np.zeros(N),
            'longitud': p_with_errors(y, 50, rng)}
    for t in TECHS:
        refs[('brightness', t)] = p_with_errors(y, 50, rng)
        refs[('descriptors', t)] = p_with_errors(y, 40, rng)
    ntr = rng.integers(16, 2200, 908)
    train = pd.DataFrame({'sample_id': [f'train_{i:04d}' for i in range(908)], 'n_tokens_beto': ntr,
                          'truncated_128': ntr > 128, 'y_true': rng.integers(0, 2, 908)})
    nte = rng.integers(16, 2200, N)
    test_index = pd.DataFrame({'sample_id': [f'test_{i:04d}' for i in range(N)], 'n_tokens_beto': nte,
                               'truncated_128': nte > 128, 'y_true': y})
    u = rng.uniform(0, 1, (908, 5))
    return Ins(y, visual, refs, train, test_index, u)


def idx(cfg_=None):
    from src.protocol import bootstrap as bs
    return bs.indices(cfg_ or CFG)
