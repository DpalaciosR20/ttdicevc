"""
test_predictions_format.py
──────────────────────────
Pruebas del validador de predicciones (F1.5,
docs/especificacion_predicciones.md §5): un archivo válido de ejemplo y al
menos un caso de falla por cada una de las reglas 1-7.

Usa un índice maestro sintético (no requiere TT_DATA_ROOT). La última prueba
corre contra meta/sample_index.csv real si existe.

    python -m pytest tests/test_predictions_format.py -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import paths  # noqa: E402
from src.predictions import schema  # noqa: E402
from src.predictions.validate import (PredictionFormatError, load_predictions,  # noqa: E402
                                      validate_predictions)
from src.training.umap_io import sha256  # noqa: E402

# Índice sintético: 4 train, 3 val, 3 test
INDEX = pd.DataFrame(
    [(f'{s}_{i:04d}', s, y) for s, ys in (('train', [0, 1, 0, 1]), ('val', [0, 1, 1]), ('test', [1, 0, 0]))
     for i, y in enumerate(ys)],
    columns=['sample_id', 'split', 'y_true'])

TAG_YES = lambda: True  # noqa: E731
TAG_NO = lambda: False  # noqa: E731


def make_rows(splits=('val',), technique='gray_scott', arch='resnet18', strategy='finetune', seed=42):
    sub = INDEX[INDEX['split'].isin(splits)].copy()
    sub['split'] = pd.Categorical(sub['split'], schema.SPLIT_ORDER, ordered=True)
    sub = sub.sort_values(['split', 'sample_id']).astype({'split': str})
    df = sub.assign(technique=technique, arch=arch, strategy=strategy, seed=seed)
    df['p_dep'] = [0.123456789012345, 0.87, 0.5, 0.0, 1.0, 0.25, 0.75, 0.6, 0.4, 0.3][:len(df)]
    return df[schema.COLUMNS].reset_index(drop=True)


def write(tmp_path, df, fase='referencia', name=None, manifest='ok', overrides=None):
    """Escribe el CSV y su manifiesto en tmp_path/{fase}/. manifest: 'ok', None (sin manifiesto)."""
    folder = tmp_path / fase
    folder.mkdir(parents=True, exist_ok=True)
    if name is None:
        r = df.iloc[0]
        name = schema.file_name(r['technique'], r['arch'], r['strategy'], r['seed'])
    path = folder / name
    df.to_csv(path, index=False, lineterminator='\n', float_format=None)
    if manifest is not None:
        m = {'task_id': 'F1.5', 'fase': fase, 'git_commit': 'abc1234', 'created_at': '2026-10-28T00:00:00Z',
             'config': {'lr': 3e-5}, 'inputs': {'IMAGES[gray_scott]': {'rel_path': 'x', 'sha256': '0' * 64}},
             'model_selection': 'mejor F1-Val', 'environment': {'python': '3.13'},
             'csv_sha256': sha256(path), 'n_rows': df['split'].value_counts().to_dict()}
        m.update(overrides or {})
        m = {k: v for k, v in m.items() if v is not DROP}
        path.with_name(schema.manifest_name(name)).write_text(json.dumps(m), encoding='utf-8')
    return path


DROP = object()


def rules_of(path, tag_exists=TAG_YES):
    with pytest.raises(PredictionFormatError) as exc:
        validate_predictions(path, index=INDEX, tag_exists=tag_exists)
    return exc.value.rules


# ── Archivo válido ───────────────────────────────────────────────────────────

def test_valid_file_passes(tmp_path):
    path = write(tmp_path, make_rows(('train', 'val', 'test')))
    df = validate_predictions(path, index=INDEX, tag_exists=TAG_NO)
    assert list(df.columns) == schema.COLUMNS
    assert df['p_dep'].dtype == 'float64' and df['seed'].dtype == 'int64'
    assert df['p_dep'].iloc[0] == 0.123456789012345  # sin redondear


def test_valid_baseline_vocab(tmp_path):
    df = make_rows(('val',), technique='umap5d', arch='svm_rbf', strategy='none', seed=0)
    validate_predictions(write(tmp_path, df), index=INDEX, tag_exists=TAG_NO)


# ── Regla 1: columnas, orden, tipos, faltantes ───────────────────────────────

def test_rule1_extra_column(tmp_path):
    df = make_rows().assign(run_id='x')
    assert 1 in rules_of(write(tmp_path, df))


def test_rule1_column_order(tmp_path):
    df = make_rows()[['split', 'sample_id'] + schema.COLUMNS[2:]]
    assert 1 in rules_of(write(tmp_path, df))


def test_rule1_missing_value(tmp_path):
    df = make_rows()
    df.loc[1, 'p_dep'] = float('nan')
    assert 1 in rules_of(write(tmp_path, df))


def test_rule1_wrong_type(tmp_path):
    df = make_rows().astype({'seed': str})
    df.loc[0, 'seed'] = '42.0'
    assert 1 in rules_of(write(tmp_path, df, name=schema.file_name('gray_scott', 'resnet18', 'finetune', 42)))


def test_rule1_row_order(tmp_path):
    df = make_rows(('val', 'test'))
    df = pd.concat([df[df['split'] == 'test'], df[df['split'] == 'val']], ignore_index=True)
    assert 1 in rules_of(write(tmp_path, df))


# ── Regla 2: vocabulario y rango de p_dep ────────────────────────────────────

def test_rule2_vocab(tmp_path):
    df = make_rows(arch='vit_b_16')
    assert 2 in rules_of(write(tmp_path, df))


def test_rule2_p_dep_range(tmp_path):
    df = make_rows()
    df.loc[0, 'p_dep'] = 1.2
    assert 2 in rules_of(write(tmp_path, df))


# ── Regla 3: sample_id y y_true contra el índice ─────────────────────────────

def test_rule3_missing_sample(tmp_path):
    df = make_rows().iloc[:-1]
    assert 3 in rules_of(write(tmp_path, df))


def test_rule3_foreign_sample(tmp_path):
    df = make_rows()
    df.loc[2, 'sample_id'] = 'val_0099'
    assert 3 in rules_of(write(tmp_path, df))


def test_rule3_y_true_mismatch(tmp_path):
    df = make_rows()
    df.loc[0, 'y_true'] = 1 - df.loc[0, 'y_true']
    assert 3 in rules_of(write(tmp_path, df))


# ── Regla 4: claves duplicadas ───────────────────────────────────────────────

def test_rule4_duplicate_key(tmp_path):
    df = make_rows()
    df = pd.concat([df, df.iloc[[0]]]).sort_values(['sample_id'], kind='stable').reset_index(drop=True)
    assert 4 in rules_of(write(tmp_path, df))


def test_rule4_duplicate_across_files(tmp_path):
    df = make_rows()
    write(tmp_path / 'a', df)
    write(tmp_path / 'b', df)  # misma fase 'referencia' en dos carpetas: misma clave
    with pytest.raises(PredictionFormatError) as exc:
        load_predictions(str(tmp_path / '**' / '*.csv'), index=INDEX, tag_exists=TAG_NO)
    assert 4 in exc.value.rules


# ── Regla 5: nombre de archivo frente a contenido ────────────────────────────

def test_rule5_seed_mismatch(tmp_path):
    df = make_rows(seed=42)
    assert 5 in rules_of(write(tmp_path, df, name=schema.file_name('gray_scott', 'resnet18', 'finetune', 7)))


def test_rule5_bad_name(tmp_path):
    assert 5 in rules_of(write(tmp_path, make_rows(), name='predictions.csv'))


# ── Regla 6: manifiesto ──────────────────────────────────────────────────────

def test_rule6_missing_manifest(tmp_path):
    assert 6 in rules_of(write(tmp_path, make_rows(), manifest=None))


def test_rule6_missing_field(tmp_path):
    assert 6 in rules_of(write(tmp_path, make_rows(), overrides={'git_commit': DROP}))


def test_rule6_sha_mismatch(tmp_path):
    assert 6 in rules_of(write(tmp_path, make_rows(), overrides={'csv_sha256': '0' * 64}))


def test_rule6_phase_mismatch(tmp_path):
    assert 6 in rules_of(write(tmp_path, make_rows(), fase='referencia', overrides={'fase': 'final'}))


def test_rule6_unknown_phase_folder(tmp_path):
    assert 6 in rules_of(write(tmp_path, make_rows(), fase='f32_primera_pasada'))


# ── Regla 7: test según la fase ──────────────────────────────────────────────

def test_rule7_test_forbidden_in_first_pass(tmp_path):
    path = write(tmp_path, make_rows(('val', 'test')), fase='primera_pasada')
    assert rules_of(path, tag_exists=TAG_YES) == {7}


def test_rule7_test_in_final_requires_tag(tmp_path):
    path = write(tmp_path, make_rows(('val', 'test')), fase='final')
    assert rules_of(path, tag_exists=TAG_NO) == {7}
    validate_predictions(path, index=INDEX, tag_exists=TAG_YES)


def test_rule7_first_pass_val_only_passes(tmp_path):
    path = write(tmp_path, make_rows(('train', 'val')), fase='primera_pasada')
    validate_predictions(path, index=INDEX, tag_exists=TAG_NO)


# ── load_predictions ─────────────────────────────────────────────────────────

def test_load_predictions_adds_fase_and_concats(tmp_path):
    write(tmp_path, make_rows(('val',), seed=42), fase='referencia')
    write(tmp_path, make_rows(('val',), seed=7), fase='referencia')
    write(tmp_path, make_rows(('val',), seed=99), fase='primera_pasada')
    out = load_predictions(str(tmp_path / '**' / '*.csv'), index=INDEX, tag_exists=TAG_NO)
    assert len(out) == 9
    assert out.groupby('fase').size().to_dict() == {'primera_pasada': 3, 'referencia': 6}
    assert list(out.columns) == schema.COLUMNS + ['fase']


def test_load_predictions_rejects_invalid_file(tmp_path):
    write(tmp_path, make_rows(seed=42))
    write(tmp_path, make_rows(seed=7, arch='vit_b_16'))
    with pytest.raises(PredictionFormatError):
        load_predictions(str(tmp_path / '**' / '*.csv'), index=INDEX, tag_exists=TAG_NO)


# ── Índice maestro real ──────────────────────────────────────────────────────

@pytest.mark.skipif(not paths.SAMPLE_INDEX.is_file(), reason='meta/sample_index.csv aún no existe')
def test_real_index_structure():
    idx = pd.read_csv(paths.SAMPLE_INDEX)
    assert list(idx.columns) == ['sample_id', 'split', 'idx', 'id', 'y_true', 'text_sha256', 'n_tokens_beto',
                                 'truncated_128', 'img_perlin', 'img_gray_scott', 'img_lorenz']
    assert idx['split'].value_counts().to_dict() == paths.EXPECTED_SIZES
    assert idx['sample_id'].is_unique
    assert not idx['sample_id'].str.contains('label').any()
    assert (idx['truncated_128'] == (idx['n_tokens_beto'] > 128)).all()
    assert idx['img_gray_scott'].str.startswith('rd_').all()
