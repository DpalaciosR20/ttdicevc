"""
insumos.py
──────────
Insumos de la Fase 5: las 45 corridas visuales (final, control_permutacion,
sensibilidad) y las referencias (BETO, 768D, techo 5D, mayoritaria, brillo ×3,
descriptores ×3, longitud), con las filas de prueba ordenadas por sample_id.

Dos modos:
  verificar_insumos(cfg)  solo presencia y alineación: lee sample_id, split,
                          y_true y las columnas de identidad, y compara
                          csv_sha256 con el manifiesto. NO carga p_dep.
  cargar(cfg)             carga p_dep de prueba con
                          src.predictions.validate.validate_predictions (lectura
                          exacta con float()). Solo debe llamarlo run_fase5 con
                          --ejecutar-prueba.

Las identidades salen de configs/protocolo.json y de las selecciones ya
congeladas: select_finalist / select_second sobre la primera pasada (F3.3) y
results/f21_techo5d/seleccion.csv (techo 5D, columna techo).

Uso:  python -m src.protocol.insumos   (equivale a --verificar-insumos)
"""

import csv
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.predictions import schema
from src.training.umap_io import sha256

REPO = paths.REPO
CONFIG_PATH = REPO / 'configs' / 'protocolo.json'
PRED = paths.RESULTS_DIR / 'predictions'
TECHO_SELECCION = paths.RESULTS_DIR / 'f21_techo5d' / 'seleccion.csv'
ID_COLS = ['sample_id', 'split', 'y_true', 'technique', 'arch', 'strategy', 'seed']
TECHNIQUES = ('gray_scott', 'perlin', 'lorenz')
# fase → (rol, columna de la tabla de selección)
VISUAL_PHASES = {'final': ('finalista', 'finalist'),
                 'control_permutacion': ('permutado', 'finalist'),
                 'sensibilidad': ('segundo', 'second')}


class InsumosError(RuntimeError):
    pass


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def text_sha256(path):
    return __import__('hashlib').sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


# ── Identidades ───────────────────────────────────────────────────────────
def _one(pattern):
    hits = sorted(glob.glob(str(pattern)))
    if len(hits) != 1:
        raise InsumosError(f'se espera exactamente un archivo para {pattern}; hay {len(hits)}')
    return Path(hits[0])


def _ref_path(technique, arch, strategy, seed=None):
    s = '*' if seed is None else str(seed)
    return _one(PRED / 'referencia' / f'{technique}__{arch}__{strategy}__seed{s}.csv')


def techo_identity():
    with open(TECHO_SELECCION, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['techo'] == 'True']
    if len(rows) != 1:
        raise InsumosError(f'{TECHO_SELECCION}: {len(rows)} filas con techo=True')
    return rows[0]['familia']


def reference_files(cfg):
    """{nombre: ruta} de las referencias; brillo y descriptores con clave (nombre, técnica)."""
    r, pt = cfg['referencias'], cfg['pisos_triviales']
    out = {
        'beto': _ref_path(r['beto']['technique'], r['beto']['arch'], r['beto']['strategy'], r['beto']['seed']),
        'emb768': _ref_path(r['emb768']['technique'], r['emb768']['arch'], r['emb768']['strategy']),
        'techo5d': _ref_path(r['techo_5d']['technique'], techo_identity(), 'none'),
        'majority': _ref_path(pt['majority']['technique'], pt['majority']['arch'], pt['majority']['strategy'],
                              pt['majority']['seed']),
        'longitud': _ref_path(r['longitud']['technique'], r['longitud']['arch'], r['longitud']['strategy'],
                              r['longitud']['seed']),
    }
    b, d = cfg['piso_brillo'], cfg['descriptores']
    for t in TECHNIQUES:
        out[('brightness', t)] = _ref_path(t, b['arch'], b['strategy'], b['seed'])
        out[('descriptors', t)] = _ref_path(t, d['arch'], d['strategy'])
    return out


def visual_selection():
    """{fase: {técnica: (arch, strategy)}} según select_finalist / select_second (F3.3)."""
    from src.selection import select_finalist as sf
    table, _ = sf.select_finalists()
    table = sf.select_second(table)
    out = {}
    for fase, (_, col) in VISUAL_PHASES.items():
        out[fase] = {r.technique: (r.arch, r.strategy) for r in table[table[col]].itertuples()}
    return out


def visual_files(cfg, selection=None):
    """{(fase, técnica): [rutas por semilla, en el orden de modelos.seeds]}."""
    selection = selection or visual_selection()
    seeds = cfg['modelos']['seeds']
    if cfg['pisos']['visual_permutado']['seeds'] != seeds:
        raise InsumosError('las semillas del piso permutado difieren de modelos.seeds')
    out = {}
    for fase in VISUAL_PHASES:
        present = sorted(Path(p).name for p in glob.glob(str(PRED / fase / '*.csv')))
        expected = []
        for t in TECHNIQUES:
            arch, strat = selection[fase][t]
            out[(fase, t)] = [PRED / fase / schema.file_name(t, arch, strat, s) for s in seeds]
            expected += [p.name for p in out[(fase, t)]]
        if sorted(expected) != present:
            raise InsumosError(f'{fase}: archivos {present} != esperados por la selección {sorted(expected)}')
    return out, selection


# ── Índice ────────────────────────────────────────────────────────────────
def load_index():
    idx = pd.read_csv(paths.SAMPLE_INDEX, dtype={'sample_id': str, 'split': str})
    out = {s: idx[idx['split'] == s].sort_values('sample_id').reset_index(drop=True) for s in paths.SPLITS}
    for s in paths.SPLITS:
        if len(out[s]) != paths.EXPECTED_SIZES[s]:
            raise InsumosError(f'{s}: {len(out[s])} filas en el índice')
    return out


# ── Verificación de alineación (sin p_dep) ────────────────────────────────
def _check_alignment(path, index, ident, log):
    """Lee solo ID_COLS (sin p_dep) y compara con el índice y el manifiesto."""
    man_path = path.with_name(schema.manifest_name(path.name))
    if not man_path.is_file():
        raise InsumosError(f'falta {man_path}')
    man = json.loads(man_path.read_text(encoding='utf-8'))
    if man.get('csv_sha256') != sha256(path):
        raise InsumosError(f'{path.name}: csv_sha256 del manifiesto no coincide')
    df = pd.read_csv(path, usecols=ID_COLS, dtype={'sample_id': str, 'split': str})
    if 'p_dep' in df.columns:  # pragma: no cover - usecols lo impide
        raise AssertionError('p_dep no debe cargarse en la verificación')
    for s in ('val', 'test'):
        g = df[df['split'] == s].sort_values('sample_id')
        if list(g['sample_id']) != list(index[s]['sample_id']):
            raise InsumosError(f'{path.name}: sample_id de {s} != índice')
        if list(g['y_true'].astype(int)) != list(index[s]['y_true'].astype(int)):
            raise InsumosError(f'{path.name}: y_true de {s} != índice')
    got = {c: sorted(set(df[c])) for c in ('technique', 'arch', 'strategy', 'seed')}
    want = {c: [v] for c, v in ident.items()}
    if any(got[c] != want[c] for c in want):
        raise InsumosError(f'{path.name}: identidad {got} != {want}')
    if set(df['split']) - {'val', 'test'}:
        raise InsumosError(f'{path.name}: splits {sorted(set(df["split"]))}')
    log(f'  OK  {path.relative_to(REPO).as_posix()}  fase={man.get("fase")} task={man.get("task_id")} '
        f'val={int((df["split"] == "val").sum())} test={int((df["split"] == "test").sum())}  '
        f'sha256={sha256(path)[:12]}…')
    return sha256(path)


def _ident_from_name(path):
    m = schema.FILENAME_RE.match(path.name)
    d = m.groupdict()
    return {'technique': d['technique'], 'arch': d['arch'], 'strategy': d['strategy'], 'seed': int(d['seed'])}


def verificar_insumos(cfg, log=print, check_umap=True):
    """Presencia y alineación de todos los insumos, sin cargar p_dep. Devuelve {ruta: sha256}."""
    index = load_index()
    hashes = {}
    log('— Corridas visuales (45)')
    vfiles, selection = visual_files(cfg)
    for fase, sel in selection.items():
        log(f'  {fase}: ' + ', '.join(f'{t} {a}/{s}' for t, (a, s) in sel.items()))
    for (fase, t), files in vfiles.items():
        for p in files:
            hashes[p.relative_to(REPO).as_posix()] = _check_alignment(p, index, _ident_from_name(p), log)
    log('— Referencias')
    for name, p in reference_files(cfg).items():
        hashes[p.relative_to(REPO).as_posix()] = _check_alignment(p, index, _ident_from_name(p), log)
    if len(hashes) != 45 + 11:
        raise InsumosError(f'{len(hashes)} archivos; se esperan 56')
    y = index['test']['y_true'].astype(int).to_numpy()
    log(f'— Prueba: n = {len(y)}, soportes 0/1 = {int((y == 0).sum())}/{int((y == 1).sum())}; '
        f'filas ordenadas por sample_id ({index["test"]["sample_id"].iloc[0]} … {index["test"]["sample_id"].iloc[-1]})')
    if check_umap:
        u = load_u_train(index)
        log(f'— u de train (umap_params.pkl): {u.shape}, etiquetas alineadas con el índice por idx')
    return hashes


# ── Carga con p_dep (solo run_fase5 --ejecutar-prueba) ────────────────────
@dataclass
class Insumos:
    sample_id: list
    y: np.ndarray                                   # (n,) y_true de prueba
    visual: dict = field(default_factory=dict)      # {(fase, técnica): {'arch','strategy','seeds','p' (n × S)}}
    refs: dict = field(default_factory=dict)        # {nombre o (nombre, técnica): p (n,)}
    files: dict = field(default_factory=dict)       # {ruta: sha256}
    train: pd.DataFrame = None                      # índice de train (n_tokens_beto, truncated_128, …)
    u_train: np.ndarray = None                      # (908, 5)
    test_index: pd.DataFrame = None


def _load_test_pdep(path, index):
    from src.predictions.validate import validate_predictions
    df = validate_predictions(path)
    t = df[df['split'] == 'test'].sort_values('sample_id')
    if list(t['sample_id']) != list(index['test']['sample_id']):
        raise InsumosError(f'{path.name}: sample_id de prueba != índice')
    if list(t['y_true'].astype(int)) != list(index['test']['y_true'].astype(int)):
        raise InsumosError(f'{path.name}: y_true de prueba != índice')
    return t['p_dep'].to_numpy(dtype=np.float64)


def cargar(cfg, with_u=True):
    index = load_index()
    vfiles, selection = visual_files(cfg)
    ins = Insumos(sample_id=list(index['test']['sample_id']),
                  y=index['test']['y_true'].astype(int).to_numpy(), train=index['train'],
                  test_index=index['test'])
    for (fase, t), files in vfiles.items():
        arch, strat = selection[fase][t]
        P = np.column_stack([_load_test_pdep(p, index) for p in files])
        ins.visual[(fase, t)] = {'arch': arch, 'strategy': strat, 'seeds': list(cfg['modelos']['seeds']), 'p': P}
        for p in files:
            ins.files[p.relative_to(REPO).as_posix()] = sha256(p)
    for name, p in reference_files(cfg).items():
        ins.refs[name] = _load_test_pdep(p, index)
        ins.files[p.relative_to(REPO).as_posix()] = sha256(p)
    if with_u:
        ins.u_train = load_u_train(index)
    return ins


def load_u_train(index):
    """u de train (908 × 5) en el orden del índice (sample_id), alineada por idx y etiquetas."""
    from src.training.umap_io import load_umap_arrays
    arr = load_umap_arrays(paths.umap_params_path())['train']
    tr = index['train']
    order = tr['idx'].astype(int).to_numpy()
    u, lab = arr['params'][order], arr['labels'][order]
    if list(lab.astype(int)) != list(tr['y_true'].astype(int)):
        raise InsumosError('etiquetas de umap_params.pkl (train) no alineadas con el índice por idx')
    return u.astype(np.float64)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--verificar-insumos', action='store_true', default=True)
    ap.parse_args(argv)
    verificar_insumos(load_config())
    return 0


if __name__ == '__main__':
    sys.exit(main())
