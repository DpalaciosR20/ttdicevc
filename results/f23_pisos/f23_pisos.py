"""
f23_pisos.py
────────────
F2.3 — Pisos (docs/protocolo_preregistrado.md §4.4 y Enmienda 2, puntos 2 y 3;
configs/protocolo.json ["pisos_triviales"], ["piso_brillo"]).

  Mayoritaria   technique none, arch majority: p_dep = 0.0 en todas las muestras.
  Estratificado technique none, arch stratified: p_dep ∈ {0.0, 1.0} con
                numpy.random.default_rng(42) y la proporción de clase 1 de train
                (353/908, leída de meta/sample_index.csv). Una sola extracción,
                rng.random(n_val + n_test) < p1, sobre las filas en el orden del
                archivo (val y luego test, cada uno por sample_id). En los
                contrastes interviene la constante 0.5 (§5.3); el archivo se
                guarda solo por completitud del formato.
  Brillo        technique = técnica, arch brightness: intensidad media / 255 de
                la imagen nativa de 256 × 256 (modo L, sin el preprocesamiento de
                los clasificadores visuales). Cada imagen se verifica por SHA-256
                contra meta/data_manifest.csv. LogReg con la rejilla completa de
                §4.3 (13 C × class_weight {None, balanced} = 26 combinaciones),
                StandardScaler ajustado en train (dentro del pipeline).

Reglas no fijadas por el protocolo (iguales que F2.1):
  - LogisticRegression con max_iter=5000 y random_state=42; demás
    hiperparámetros por defecto de scikit-learn.
  - Orden de la rejilla: itertools.product en el orden de claves del config
    (C varía más lento; class_weight con null primero).
  - p_dep = predict_proba[:, clase 1]; F1-Val con ŷ = 1[p_dep > 0.5].
  - Empate (tolerancia 1e-12): combinación anterior en la rejilla.

No calcula ni imprime métricas de test (protocolo §10), salvo la comprobación
de la mayoritaria contra su cifra de diseño f12.piso_mayoritaria.f1_macro
(el log solo dice si coincide).

Uso (desde la raíz del repo, .venv, CPU, TT_DATA_ROOT definida):
    python results/f23_pisos/f23_pisos.py            # corrida completa
    python results/f23_pisos/f23_pisos.py --estimar  # lee y verifica las imágenes de una técnica,
                                                     # una combinación; no escribe nada

Salidas:
    results/predictions/referencia/none__majority__none__seed42.csv (+ manifiesto)
    results/predictions/referencia/none__stratified__none__seed42.csv (+ manifiesto)
    results/predictions/referencia/{perlin,gray_scott,lorenz}__brightness__none__seed42.csv (+ manifiesto)
    results/f23_pisos/rejilla_brillo.csv
    results/f23_pisos/f23_pisos.log
"""

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import PIL  # noqa: E402
import sklearn  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import paths  # noqa: E402
from src.canonical.build import confusion_counts, f1_macro_from_confusion  # noqa: E402
from src.predictions.write import write_predictions  # noqa: E402
from src.training.umap_io import sha256  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CONFIG = REPO / 'configs' / 'protocolo.json'
CANON = REPO / 'results' / 'canonical_numbers.json'
TASK_ID = 'F2.3'
PRED_SPLITS = ('val', 'test')
LOGREG_MAX_ITER = 5000  # no fijado por §4.3; igual que F2.1 y F2.2
TOL_DESIGN = 1e-12


class Stop(RuntimeError):
    """Verificación que obliga a detener."""


class Log:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, msg=''):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def write(self):
        self.path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8', newline='\n')


def text_sha256(path):
    """SHA-256 con fines de línea LF (independiente de core.autocrlf)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_index():
    idx = pd.read_csv(paths.SAMPLE_INDEX, dtype={'sample_id': str, 'split': str})
    out = {s: idx[idx['split'] == s].sort_values('sample_id').reset_index(drop=True) for s in paths.SPLITS}
    for s in paths.SPLITS:
        if len(out[s]) != paths.EXPECTED_SIZES[s]:
            raise Stop(f'{s}: {len(out[s])} filas en el índice maestro')
    return out


def load_manifest():
    with open(paths.DATA_MANIFEST, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ── Rejilla LogReg (configs/protocolo.json → techo_5d.familias.logreg) ────
def logreg_grid(spec):
    Cs = [float(v) for v in np.logspace(spec['C']['log10_low'], spec['C']['log10_high'], spec['C']['n'])]
    return [{'C': c, 'class_weight': w} for c, w in itertools.product(Cs, spec['class_weight'])]


def logreg_fixed(spec, rs):
    assert spec['estimator'] == 'LogisticRegression'
    return {'penalty': spec['penalty'], 'solver': spec['solver'], 'max_iter': LOGREG_MAX_ITER, 'random_state': rs}


def p_dep_of(model, X):
    pos = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(X)[:, pos].astype(np.float64)


def f1_val_of(y, p):
    return float(f1_score(y, (p > 0.5).astype(int), average='macro'))


def select(f1s, tol):
    """Primer índice cuyo F1 está a ≤ tol del máximo (empate → anterior en la rejilla)."""
    best = max(f1s)
    return next(i for i, f in enumerate(f1s) if f >= best - tol)


def n_iter_of(model):
    return int(np.max(model[-1].n_iter_))


# ── Brillo ────────────────────────────────────────────────────────────────
def brightness(technique, index, manifest, size, mode, log):
    """Intensidad media / 255 por split, verificando SHA-256, modo y tamaño de cada imagen."""
    rel_dir = paths.REL['IMAGES'][technique]
    expected = {r['rel_path']: r['sha256'] for r in manifest if r['kind'] == 'file'
                and r['rel_path'].startswith(rel_dir + '/')}
    n_total = sum(len(index[s]) for s in paths.SPLITS)
    if len(expected) != n_total:
        raise Stop(f'{technique}: {len(expected)} imágenes en data_manifest.csv, se esperaban {n_total}')
    folder = paths.images_dir(technique)
    out, t0 = {}, time.time()
    for s in paths.SPLITS:
        vals = []
        for name in index[s][f'img_{technique}']:
            rel = f'{rel_dir}/{name}'
            if rel not in expected:
                raise Stop(f'{rel} no está en data_manifest.csv')
            data = (folder / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected[rel]:
                raise Stop(f'{rel}: sha256 distinto del manifiesto')
            with Image.open(folder / name) as im:
                if im.mode != mode or list(im.size) != size:
                    raise Stop(f'{rel}: modo {im.mode}, tamaño {im.size}; se esperaba {mode} {size}')
                arr = np.asarray(im, dtype=np.float64)
            vals.append(arr.mean() / 255.0)
        out[s] = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
    log(f'  [{technique}] {n_total} imágenes: sha256 = data_manifest.csv, modo {mode}, {size[0]}×{size[1]}; '
        f'{time.time() - t0:.1f} s')
    return out


def frame(index, s, y, technique, arch, seed, p):
    return pd.DataFrame({'sample_id': index[s]['sample_id'], 'split': s, 'y_true': y,
                         'technique': technique, 'arch': arch, 'strategy': 'none', 'seed': seed, 'p_dep': p})


def write(df, config, inputs, selection, log):
    path = write_predictions(df, fase='referencia', task_id=TASK_ID, config=config, inputs=inputs,
                             model_selection=selection)
    back = pd.read_csv(path, float_precision='round_trip')
    if not np.array_equal(back['p_dep'].values, df['p_dep'].values):
        raise RuntimeError(f'{path.name}: p_dep no hace ida y vuelta exacta por el CSV')
    log(f'  escrito {path.relative_to(paths.REPO).as_posix()} ({len(df)} filas)')


# ── Principal ─────────────────────────────────────────────────────────────
def main(estimar, log):
    t0 = time.time()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    triv, bcfg, tcfg = cfg['pisos_triviales'], cfg['piso_brillo'], cfg['techo_5d']
    sel_cfg = tcfg['selection']
    if (bcfg['grid'], bcfg['selection_rule']) != ('techo_5d.familias.logreg', 'techo_5d.selection'):
        raise Stop(f'piso_brillo apunta a otra rejilla o selección: {bcfg}')
    if (bcfg['scaler']['name'], bcfg['scaler']['fit_split']) != ('StandardScaler', 'train'):
        raise Stop(f'escalador inesperado {bcfg["scaler"]}')
    if (sel_cfg['metric'], sel_cfg['split']) != ('f1_macro', 'val'):
        raise Stop(f'selección inesperada {sel_cfg}')
    if bcfg['image'] != {'source': 'nativa', 'size': [256, 256], 'mode': 'L', 'scale': 'mean / 255',
                         'visual_preprocessing': False}:
        raise Stop(f'imagen de brillo inesperada {bcfg["image"]}')
    rs, tol = tcfg['random_state'], sel_cfg['tie_tolerance']

    log(f'F2.3 — Pisos. scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}, '
        f'Pillow {PIL.__version__}' + ('  [--estimar: no escribe nada]' if estimar else ''))
    log('\n— Integridad')
    inputs = {'meta/sample_index.csv (repo)': sha256(paths.SAMPLE_INDEX),
              'meta/data_manifest.csv (repo)': sha256(paths.DATA_MANIFEST),
              'configs/protocolo.json (repo, LF)': text_sha256(CONFIG)}
    for k, v in inputs.items():
        log(f'  sha256  {k}  {v}')
    manifest = load_manifest()
    for t in paths.TECHNIQUES:
        row = next(r for r in manifest if r['kind'] == 'dir' and r['logical_name'] == f'IMAGES[{t}]')
        inputs[f'IMAGES[{t}]'] = {'rel_path': row['rel_path'], 'sha256': row['sha256'],
                                  'n_files': int(row['n_files']),
                                  'verificacion': 'sha256 de cada imagen contra meta/data_manifest.csv'}

    index = load_index()
    y = {s: index[s]['y_true'].astype(int).values for s in paths.SPLITS}
    n1, n = int(y['train'].sum()), len(y['train'])
    log(f'  índice maestro: train {n} ({n - n1}/{n1}), val {len(y["val"])}, test {len(y["test"])}')

    # ── Mayoritaria
    log('\n— Mayoritaria (p_dep = 0.0)')
    maj = triv['majority']
    if maj['p_dep'] != 0.0:
        raise Stop(f'p_dep de la mayoritaria {maj["p_dep"]}')
    p_maj = {s: np.full(len(y[s]), maj['p_dep'], dtype=np.float64) for s in PRED_SPLITS}
    f1_maj_val = f1_val_of(y['val'], p_maj['val'])
    log(f'  F1-Val = {f1_maj_val:.6f}')
    design = json.loads(CANON.read_text(encoding='utf-8'))['numbers'][cfg['pisos']['majority']['canonical']]
    cm = confusion_counts(pd.Series(y['test']), pd.Series((p_maj['test'] > 0.5).astype(int)))
    match = abs(f1_macro_from_confusion(cm) - design['value']['test']) <= TOL_DESIGN
    log(f'  F1-Test de la mayoritaria = cifra de diseño {cfg["pisos"]["majority"]["canonical"]}: '
        f'{"coincide" if match else "NO COINCIDE"}')
    if not match:
        raise Stop('la mayoritaria no reproduce su cifra de diseño')

    # ── Estratificado
    log('\n— Aleatorio estratificado')
    st = triv['stratified']
    if (st['rng'], st['proportion_split'], st['p_dep_values']) != ('numpy.random.default_rng', 'train', [0.0, 1.0]):
        raise Stop(f'estratificado inesperado {st}')
    p1 = n1 / n
    draw = (np.random.default_rng(st['rng_seed']).random(len(y['val']) + len(y['test'])) < p1).astype(np.float64)
    p_st = {'val': draw[:len(y['val'])], 'test': draw[len(y['val']):]}
    f1_st_val = f1_val_of(y['val'], p_st['val'])
    log(f'  p1 = {n1}/{n} = {p1:.6f}; default_rng({st["rng_seed"]}).random({len(draw)}) < p1, val y luego test')
    log(f'  F1-Val = {f1_st_val:.6f}  (en los contrastes interviene la constante {st["contrast_constant"]})')

    # ── Brillo
    log('\n— Brillo (intensidad media / 255, imagen nativa)')
    spec = tcfg['familias']['logreg']
    fixed = logreg_fixed(spec, rs)
    combos = logreg_grid(spec)
    techniques = paths.TECHNIQUES[:1] if estimar else paths.TECHNIQUES
    if estimar:
        combos = combos[:1]
    log(f'  {len(combos)} combinaciones; fijos {json.dumps(fixed)}; escalador StandardScaler (train)')
    rows, best = [], {}
    for t in techniques:
        X = brightness(t, index, manifest, bcfg['image']['size'], bcfg['image']['mode'], log)
        fitted, f1s = [], []
        for i, combo in enumerate(combos):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', ConvergenceWarning)
                model = make_pipeline(StandardScaler(), LogisticRegression(**fixed, **combo)).fit(X['train'],
                                                                                                    y['train'])
            conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
            f1 = f1_val_of(y['val'], p_dep_of(model, X['val']))
            fitted.append(model)
            f1s.append(f1)
            rows.append({'tecnica': t, 'orden': i, 'combinacion': json.dumps(combo), 'f1_val': f1,
                         'n_iter': n_iter_of(model), 'avisos_convergencia': conv})
            log(f'    {i:>2} {json.dumps(combo)}  F1-Val = {f1:.6f}  n_iter={n_iter_of(model)}'
                + (f'  ConvergenceWarning×{conv}' if conv else ''))
        k = select(f1s, tol)
        n_tie = sum(f >= max(f1s) - tol for f in f1s)
        best[t] = {'orden': k, 'combo': combos[k], 'f1_val': f1s[k], 'model': fitted[k], 'X': X, 'n_empate': n_tie}
        log(f'  [{t}] seleccionada #{k} {json.dumps(combos[k])}  F1-Val = {f1s[k]:.6f} '
            f'({n_tie} combinación(es) dentro de la tolerancia)')

    if estimar:
        log(f'\nDuración (estimación: una técnica, una combinación): {time.time() - t0:.1f} s')
        return

    pd.DataFrame(rows).to_csv(OUT_DIR / 'rejilla_brillo.csv', index=False, lineterminator='\n')
    log('\n  escrito results/f23_pisos/rejilla_brillo.csv')

    log('\n— Predicciones (fase referencia, seed 42; test archivado sin métricas)')
    script = {'script': 'results/f23_pisos/f23_pisos.py', 'script_sha256_lf': text_sha256(Path(__file__)),
              'protocol': 'docs/protocolo_preregistrado.md §4.4 y Enmienda 2',
              'protocol_config': 'configs/protocolo.json',
              'protocol_config_sha256_lf': inputs['configs/protocolo.json (repo, LF)'], 'decision': 'p_dep > 0.5'}
    idx_inputs = {k: inputs[k] for k in ('meta/sample_index.csv (repo)', 'configs/protocolo.json (repo, LF)')}

    df = pd.concat([frame(index, s, y[s], maj['technique'], maj['arch'], maj['seed'], p_maj[s])
                    for s in PRED_SPLITS], ignore_index=True)
    write(df, {'rule': 'p_dep = 0.0 en todas las muestras (Enmienda 2, punto 3)', **script}, idx_inputs,
          {'criterio': 'sin selección: predice siempre la clase 0', 'f1_val': f1_maj_val}, log)

    df = pd.concat([frame(index, s, y[s], st['technique'], st['arch'], st['seed'], p_st[s])
                    for s in PRED_SPLITS], ignore_index=True)
    write(df, {'rule': (f'numpy.random.default_rng({st["rng_seed"]}).random({len(draw)}) < {n1}/{n} '
                        '(proporción de clase 1 en train), una extracción sobre val y luego test, '
                        'por sample_id (Enmienda 2, punto 3)'),
               'used_in_contrasts': False, 'contrast_constant': st['contrast_constant'], **script},
          idx_inputs, {'criterio': 'sin selección', 'f1_val': f1_st_val}, log)

    for t in techniques:
        b = best[t]
        p = {s: p_dep_of(b['model'], b['X'][s]) for s in PRED_SPLITS}
        if f1_val_of(y['val'], p['val']) != b['f1_val']:
            raise RuntimeError(f'{t}: p_dep > 0.5 no reproduce el F1-Val de la selección')
        df = pd.concat([frame(index, s, y[s], t, bcfg['arch'], bcfg['seed'], p[s]) for s in PRED_SPLITS],
                       ignore_index=True)
        config = {'features': 'intensidad media / 255 de la imagen nativa de 256 × 256 (modo L)',
                  'pipeline': 'StandardScaler (ajustado en train) + LogisticRegression',
                  'estimator_params': {**fixed, **b['combo']},
                  'other_params': f'valores por defecto de scikit-learn {sklearn.__version__}',
                  'fit': 'solo train', **script}
        selection = {'criterio': ('F1-Macro de validación (ŷ = 1[p_dep > 0.5]) sobre la rejilla completa de '
                                  'LogReg de §4.3; empate con tolerancia 1e-12 → combinación anterior'),
                     'orden_en_rejilla': b['orden'], 'combinacion': b['combo'], 'f1_val': b['f1_val'],
                     'tabla': 'results/f23_pisos/rejilla_brillo.csv'}
        write(df, config, {**idx_inputs, f'IMAGES[{t}]': inputs[f'IMAGES[{t}]']}, selection, log)
    log(f'\nDuración: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--estimar', action='store_true', help='una técnica y una combinación; no escribe nada')
    args = ap.parse_args()
    log = Log(OUT_DIR / ('f23_estimacion.log' if args.estimar else 'f23_pisos.log'))
    try:
        main(args.estimar, log)
    except Stop as e:
        log(f'\nDETENIDO: {e}')
        raise SystemExit(2)
    finally:
        if not args.estimar:
            log.write()
