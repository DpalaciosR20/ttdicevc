"""
tests/test_select_finalist.py
═════════════════════════════
F1.3 — Regla de selección del finalista (docs/matriz_experimentos.md §5)
implementada en src/selection/select_finalist.py, con casos sintéticos:

  - líder único;
  - empate resuelto por parámetros entrenables;
  - empate resuelto por parámetros totales;
  - empate residual resuelto por la media;
  - además: varios líderes (mayor DE), empate no resuelto (error), semillas
    o configuraciones incompletas, fase equivocada, regla del JSON distinta y
    una prueba de extremo a extremo con archivos F1.5 y load_predictions.

No usa resultados reales ni requiere TT_DATA_ROOT.

    python -m pytest tests/test_select_finalist.py -v
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.predictions import schema  # noqa: E402
from src.selection import select_finalist as sf  # noqa: E402
from src.training.umap_io import sha256  # noqa: E402

SEEDS = [42, 7, 123]
REAL = sf.load_matrix()
CONFIGS = [(a, s) for a in REAL['factores']['arch'] for s in REAL['factores']['strategy']]


def matrix(techniques=('gray_scott',), params=None):
    """Matriz real restringida a `techniques`, con conteos de parámetros opcionales."""
    m = copy.deepcopy(REAL)
    m['factores']['technique'] = list(techniques)
    for (arch, strat), (tr, tot) in (params or {}).items():
        m['parametros'][arch][strat] = {'trainable': tr, 'total': tot}
    return m


def runs(spec, technique='gray_scott'):
    """spec: {(arch, strategy): (media, de)} → F1 por semilla media−de, media, media+de
    (DE muestral exactamente `de`). Las configuraciones no listadas valen 0.60 ± 0.01."""
    rows = []
    for arch, strat in CONFIGS:
        m, s = spec.get((arch, strat), (0.60, 0.01))
        for seed, f1 in zip(SEEDS, (m - s, m, m + s)):
            rows.append({'technique': technique, 'arch': arch, 'strategy': strat, 'seed': seed, 'f1_val': f1})
    return pd.DataFrame(rows)


def finalist(table, technique='gray_scott'):
    r = table[(table['technique'] == technique) & table['finalist']]
    assert len(r) == 1
    return r['arch'].iloc[0], r['strategy'].iloc[0]


def flags(table, col):
    return {(r.arch, r.strategy) for r in table.itertuples() if getattr(r, col)}


# ── Los cuatro casos pedidos ─────────────────────────────────────────────────

def test_single_leader():
    """Ninguna otra configuración entra en la tolerancia: gana el líder aunque
    haya configuraciones con menos parámetros."""
    t = sf.select_from_runs(runs({('resnet18', 'finetune'): (0.80, 0.01),
                                  ('efficientnet_b0', 'finetune'): (0.70, 0.01)}), matrix())
    assert flags(t, 'leader') == {('resnet18', 'finetune')}
    assert flags(t, 'tied') == {('resnet18', 'finetune')}
    assert finalist(t) == ('resnet18', 'finetune')
    assert t.loc[t['finalist'], 'tolerance'].iloc[0] == pytest.approx(0.01)


def test_tie_resolved_by_trainable():
    """Dentro de 1 DE del líder (ViT finetune) están EfficientNet frozen y
    ResNet finetune; gana el de menos entrenables. ResNet frozen, con menos
    entrenables aún, queda fuera de la tolerancia y no compite."""
    t = sf.select_from_runs(runs({('vit_b16', 'finetune'): (0.75, 0.03),
                                  ('efficientnet_b0', 'frozen'): (0.73, 0.01),
                                  ('resnet18', 'finetune'): (0.74, 0.02),
                                  ('resnet18', 'frozen'): (0.70, 0.01)}), matrix())
    assert flags(t, 'leader') == {('vit_b16', 'finetune')}
    assert flags(t, 'tied') == {('vit_b16', 'finetune'), ('efficientnet_b0', 'frozen'), ('resnet18', 'finetune')}
    assert finalist(t) == ('efficientnet_b0', 'frozen')


def test_tie_resolved_by_total():
    """Dos empatadas con los mismos entrenables: gana la de menos parámetros
    totales, aunque la otra tenga mayor media."""
    params = {('resnet18', 'frozen'): (1538, 11177538), ('vit_b16', 'frozen'): (1538, 85800194)}
    t = sf.select_from_runs(runs({('vit_b16', 'frozen'): (0.75, 0.03),
                                  ('resnet18', 'frozen'): (0.73, 0.01)}), matrix(params=params))
    assert flags(t, 'tied') == {('vit_b16', 'frozen'), ('resnet18', 'frozen')}
    assert finalist(t) == ('resnet18', 'frozen')


def test_residual_tie_resolved_by_mean():
    """Dos empatadas con los mismos entrenables y totales: gana la de mayor media,
    aunque no sea el líder."""
    params = {('resnet18', 'frozen'): (1026, 4010110), ('efficientnet_b0', 'frozen'): (1026, 4010110)}
    t = sf.select_from_runs(runs({('vit_b16', 'finetune'): (0.76, 0.04),
                                  ('efficientnet_b0', 'frozen'): (0.74, 0.01),
                                  ('resnet18', 'frozen'): (0.73, 0.01)}), matrix(params=params))
    assert {('efficientnet_b0', 'frozen'), ('resnet18', 'frozen')} <= flags(t, 'tied')
    assert finalist(t) == ('efficientnet_b0', 'frozen')


# ── Casos límite de la regla ─────────────────────────────────────────────────

def test_multiple_leaders_use_max_std():
    """Dos líderes con la misma media (0.75) y DE 0.25 y 0.125: la tolerancia
    usa 0.25, así que ResNet frozen (0.5625) entra y gana. Con la DE menor
    quedaría fuera y ganaría ResNet finetune. Valores exactos en binario."""
    rows = runs({})
    sel = lambda a, s: (rows['arch'] == a) & (rows['strategy'] == s)  # noqa: E731
    rows.loc[sel('vit_b16', 'finetune'), 'f1_val'] = [0.5, 0.75, 1.0]
    rows.loc[sel('resnet18', 'finetune'), 'f1_val'] = [0.625, 0.75, 0.875]
    rows.loc[sel('resnet18', 'frozen'), 'f1_val'] = [0.5625] * 3
    t = sf.select_from_runs(rows, matrix())
    assert flags(t, 'leader') == {('vit_b16', 'finetune'), ('resnet18', 'finetune')}
    assert set(t['tolerance']) == {0.25}
    assert finalist(t) == ('resnet18', 'frozen')


def test_tolerance_boundary_is_inclusive():
    """Media exactamente igual a media del líder − DE del líder: empatada (≥).
    Valores representables en binario para que la igualdad sea exacta."""
    rows = runs({})
    rows.loc[(rows['arch'] == 'vit_b16') & (rows['strategy'] == 'finetune'), 'f1_val'] = [0.5, 0.75, 1.0]
    rows.loc[(rows['arch'] == 'resnet18') & (rows['strategy'] == 'frozen'), 'f1_val'] = [0.5, 0.5, 0.5]
    t = sf.select_from_runs(rows, matrix())
    assert t.loc[t['finalist'], 'tolerance'].iloc[0] == 0.25
    assert finalist(t) == ('resnet18', 'frozen')


def test_unresolved_tie_raises():
    params = {('resnet18', 'frozen'): (1026, 4010110), ('efficientnet_b0', 'frozen'): (1026, 4010110)}
    with pytest.raises(sf.UnresolvedTieError, match='lo decide SV'):
        sf.select_from_runs(runs({('efficientnet_b0', 'frozen'): (0.74, 0.01),
                                  ('resnet18', 'frozen'): (0.74, 0.01)}), matrix(params=params))


def test_selection_is_per_technique():
    both = pd.concat([runs({('resnet18', 'finetune'): (0.80, 0.01)}, technique='perlin'),
                      runs({('vit_b16', 'frozen'): (0.80, 0.01)}, technique='lorenz')])
    t = sf.select_from_runs(both, matrix(techniques=('perlin', 'lorenz')))
    assert finalist(t, 'perlin') == ('resnet18', 'finetune')
    assert finalist(t, 'lorenz') == ('vit_b16', 'frozen')
    assert list(dict.fromkeys(t['technique'])) == ['perlin', 'lorenz']


def test_real_parameter_counts_are_used():
    t = sf.select_from_runs(runs({}), matrix())
    vit = t[(t['arch'] == 'vit_b16') & (t['strategy'] == 'frozen')].iloc[0]
    assert (vit['trainable'], vit['total']) == (1538, 85800194)


# ── Segundo mejor modelo (F1.2, fase sensibilidad) ───────────────────────────

def second(table, technique='gray_scott'):
    r = table[(table['technique'] == technique) & table['second']]
    assert len(r) == 1
    return r['arch'].iloc[0], r['strategy'].iloc[0]


def test_second_normal_case():
    """Líder único finalista: el segundo es el líder de las cinco restantes."""
    t = sf.select_second_from_runs(runs({('resnet18', 'finetune'): (0.80, 0.01),
                                         ('efficientnet_b0', 'finetune'): (0.70, 0.01)}), matrix())
    assert finalist(t) == ('resnet18', 'finetune')
    assert second(t) == ('efficientnet_b0', 'finetune')


def test_second_when_finalist_is_not_leader():
    """El finalista (EfficientNet frozen) gana por complejidad sin ser el líder;
    al excluirlo, el líder (ViT finetune) queda solo dentro de su tolerancia y
    es el segundo."""
    t = sf.select_second_from_runs(runs({('vit_b16', 'finetune'): (0.75, 0.03),
                                         ('efficientnet_b0', 'frozen'): (0.73, 0.01)}), matrix())
    assert flags(t, 'leader') == {('vit_b16', 'finetune')}
    assert finalist(t) == ('efficientnet_b0', 'frozen')
    assert second(t) == ('vit_b16', 'finetune')


def test_second_reapplies_complexity_rule():
    """Al excluir al finalista se aplica la regla completa: entre el líder (ViT
    finetune) y ResNet finetune, empatados, gana ResNet por menos entrenables,
    aunque ResNet frozen tenga menos parámetros (queda fuera de la tolerancia)."""
    t = sf.select_second_from_runs(runs({('vit_b16', 'finetune'): (0.75, 0.03),
                                         ('efficientnet_b0', 'frozen'): (0.73, 0.01),
                                         ('resnet18', 'finetune'): (0.74, 0.02),
                                         ('resnet18', 'frozen'): (0.70, 0.01)}), matrix())
    assert finalist(t) == ('efficientnet_b0', 'frozen')
    assert second(t) == ('resnet18', 'finetune')


def test_second_unresolved_tie_raises():
    params = {('resnet18', 'frozen'): (1026, 4010110), ('efficientnet_b0', 'frozen'): (1026, 4010110)}
    with pytest.raises(sf.UnresolvedTieError, match='lo decide SV'):
        sf.select_second_from_runs(runs({('vit_b16', 'finetune'): (0.80, 0.01),
                                         ('efficientnet_b0', 'frozen'): (0.74, 0.01),
                                         ('resnet18', 'frozen'): (0.74, 0.01)}), matrix(params=params))


def test_second_is_per_technique_and_distinct():
    both = pd.concat([runs({('resnet18', 'finetune'): (0.80, 0.01), ('vit_b16', 'frozen'): (0.75, 0.01)},
                           technique='perlin'),
                      runs({('vit_b16', 'frozen'): (0.80, 0.01), ('resnet18', 'finetune'): (0.70, 0.01)},
                           technique='lorenz')])
    t = sf.select_second_from_runs(both, matrix(techniques=('perlin', 'lorenz')))
    assert second(t, 'perlin') == ('vit_b16', 'frozen')
    assert second(t, 'lorenz') == ('resnet18', 'finetune')
    assert not (t['finalist'] & t['second']).any()


# ── Entradas incompletas o incoherentes ──────────────────────────────────────

def test_missing_seed_raises():
    r = runs({})
    r = r[~((r['arch'] == 'resnet18') & (r['strategy'] == 'frozen') & (r['seed'] == 123))]
    with pytest.raises(sf.SelectionError, match='semillas'):
        sf.select_from_runs(r, matrix())


def test_extra_seed_raises():
    r = runs({})
    extra = r.iloc[[0]].assign(seed=2024)
    with pytest.raises(sf.SelectionError, match='semillas'):
        sf.select_from_runs(pd.concat([r, extra]), matrix())


def test_missing_configuration_raises():
    r = runs({})
    r = r[~((r['arch'] == 'vit_b16') & (r['strategy'] == 'finetune'))]
    with pytest.raises(sf.SelectionError, match='faltan configuraciones'):
        sf.select_from_runs(r, matrix())


def test_technique_outside_matrix_raises():
    with pytest.raises(sf.SelectionError, match='fuera de la matriz'):
        sf.select_from_runs(runs({}, technique='lorenz'), matrix(techniques=('gray_scott',)))


def test_rule_mismatch_raises():
    m = matrix()
    m['seleccion']['multiple_leaders_std'] = 'min'
    with pytest.raises(sf.SelectionError, match='seleccion'):
        sf.select_from_runs(runs({}), m)


# ── Extremo a extremo con archivos F1.5 ──────────────────────────────────────

N_VAL = 10
Y_VAL = [0, 1, 1, 0, 0, 1, 0, 1, 0, 0]
INDEX = pd.DataFrame([(f'val_{i:04d}', 'val', y) for i, y in enumerate(Y_VAL)]
                     + [(f'train_{i:04d}', 'train', i % 2) for i in range(4)],
                     columns=['sample_id', 'split', 'y_true'])


def p_dep_for(arch, strat, seed):
    """resnet18/frozen acierta todo (con un p_dep = 0.5 en una muestra de clase 0,
    que por la regla estricta se predice 0). Las demás fallan una cantidad de
    muestras que depende de la configuración y la semilla."""
    p = np.where(np.array(Y_VAL) == 1, 0.9, 0.1)
    if (arch, strat) == ('resnet18', 'frozen'):
        p[0] = 0.5
        return p
    n_wrong = 1 + (CONFIGS.index((arch, strat)) + SEEDS.index(seed)) % 3
    p[:n_wrong] = 1 - p[:n_wrong]
    return p


def write_file(folder, technique, arch, strat, seed):
    df = pd.DataFrame({'sample_id': [f'val_{i:04d}' for i in range(N_VAL)], 'split': 'val', 'y_true': Y_VAL,
                       'technique': technique, 'arch': arch, 'strategy': strat, 'seed': seed,
                       'p_dep': p_dep_for(arch, strat, seed)})
    path = folder / schema.file_name(technique, arch, strat, seed)
    df.to_csv(path, index=False, lineterminator='\n')
    m = {'task_id': 'F3.2', 'fase': folder.name, 'git_commit': 'abc1234', 'created_at': '2026-11-02T00:00:00Z',
         'config': {}, 'inputs': {}, 'model_selection': 'mejor F1-Val', 'environment': {},
         'csv_sha256': sha256(path), 'n_rows': {'val': N_VAL}}
    path.with_name(schema.manifest_name(path.name)).write_text(json.dumps(m), encoding='utf-8')


def write_phase(tmp_path, fase='primera_pasada', technique='gray_scott'):
    folder = tmp_path / fase
    folder.mkdir(parents=True, exist_ok=True)
    for arch, strat in CONFIGS:
        for seed in SEEDS:
            write_file(folder, technique, arch, strat, seed)
    return folder


def test_end_to_end_with_load_predictions(tmp_path):
    folder = write_phase(tmp_path)
    table, per_run = sf.select_finalists(str(folder / '*.csv'), matrix=matrix(), index=INDEX,
                                         tag_exists=lambda: False)
    assert len(per_run) == 18 and set(per_run['n_val']) == {N_VAL}
    for r in per_run.itertuples():
        y_hat = (p_dep_for(r.arch, r.strategy, r.seed) > 0.5).astype(int)
        assert r.f1_val == f1_score(Y_VAL, y_hat, average='macro')
    best = per_run[(per_run['arch'] == 'resnet18') & (per_run['strategy'] == 'frozen')]
    assert set(best['f1_val']) == {1.0}
    assert finalist(table) == ('resnet18', 'frozen')
    exp = per_run.groupby(['arch', 'strategy'])['f1_val'].agg(['mean', lambda x: x.std(ddof=1)])
    for r in table.itertuples():
        assert r.f1_val_mean == pytest.approx(exp.loc[(r.arch, r.strategy)].iloc[0])
        assert r.f1_val_std == pytest.approx(exp.loc[(r.arch, r.strategy)].iloc[1])


def test_end_to_end_rejects_other_phase(tmp_path):
    folder = write_phase(tmp_path, fase='referencia')
    with pytest.raises(sf.SelectionError, match='primera_pasada'):
        sf.select_finalists(str(folder / '*.csv'), matrix=matrix(), index=INDEX, tag_exists=lambda: False)


def test_default_pattern_is_primera_pasada():
    assert Path(sf.DEFAULT_PATTERN).parent.name == 'primera_pasada'
