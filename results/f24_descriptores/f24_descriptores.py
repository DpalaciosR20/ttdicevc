"""
f24_descriptores.py
───────────────────
F2.4 — Clasificador sobre descriptores de bajo nivel (docs/protocolo_preregistrado.md
§4.2, §4.3 y Enmienda 2, punto 1; configs/protocolo.json ["descriptores"]).

Descriptores (Tabla 11 del Módulo 2): las tres funciones de abajo son copia
literal de notebooks/historial/camera_ready_CE.ipynb, celda 20 (commit f126052),
y el contraste es img.std() como en esa celda. La imagen se lee igual que allí,
np.array(Image.open(path).convert('L')), sobre la imagen nativa de 256 × 256.
tests/test_f24_descriptores.py ejecuta la celda original y compara en 10 imágenes.

A diferencia del original:
  - si falta una imagen o su SHA-256 no coincide con meta/data_manifest.csv,
    la ejecución se detiene (el original la saltaba sin avisar);
  - se exigen 908/114/114 imágenes por técnica, modo L y 256 × 256;
  - la etiqueta se toma de meta/sample_index.csv (igual a la de umap_params.pkl,
    verificado en F2.1) y se comprueba contra el sufijo _label<y> del archivo.

Clasificador (uno por técnica): las cinco familias y rejillas de §4.3
(configs/protocolo.json → techo_5d) con el código de F2.1
(results/f21_techo5d/f21_techo5d.py: rejillas, orden, hiperparámetros fijos,
StandardScaler en train para logreg/svm_rbf/mlp, p_dep, F1-Val con
p_dep > 0.5 y desempate 1e-12 → familia anterior en family_order y, dentro de
ella, combinación anterior en la rejilla). Entrada: los 4 descriptores.

Tabla 11 recalculada: solo train; t de Welch (depresivo − no depresivo), d de
Cohen no ponderado con DE ddof 0 y Holm (α = 0.05) sobre las 12 comparaciones
juntas, como en la celda 20. Se compara con la salida guardada de la celda 20
(ejecución A) y con la ejecución posterior documentada en
notebooks/historial/README.md (ejecución B).

No calcula ni imprime métricas de test (protocolo §10): las predicciones de
test se archivan hasta la Fase 5.

Uso (desde la raíz del repo, .venv, CPU, TT_DATA_ROOT definida):
    python results/f24_descriptores/f24_descriptores.py --estimar  # una técnica, 20 imágenes por split,
                                                                   # una combinación por familia; no escribe
    python results/f24_descriptores/f24_descriptores.py            # corrida completa

Salidas:
    results/f24_descriptores/descriptores.csv
    results/f24_descriptores/rejilla_descriptores.csv
    results/f24_descriptores/seleccion.csv
    results/f24_descriptores/tabla11_recalculada.csv
    results/f24_descriptores/f24_descriptores.log
    results/predictions/referencia/{perlin,gray_scott,lorenz}__descriptors__none__seed42.csv (+ manifiesto)
"""

import argparse
import hashlib
import importlib.util
import io
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
import scipy  # noqa: E402
import skimage  # noqa: E402
import sklearn  # noqa: E402
import statsmodels  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from skimage.feature import local_binary_pattern  # noqa: E402
from skimage.filters import sobel  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from src import paths  # noqa: E402
from src.predictions.write import write_predictions  # noqa: E402
from src.training.umap_io import sha256  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F21 = _load('f21_techo5d', 'results/f21_techo5d/f21_techo5d.py')
F23 = _load('f23_pisos', 'results/f23_pisos/f23_pisos.py')

OUT_DIR = Path(__file__).resolve().parent
CONFIG = REPO / 'configs' / 'protocolo.json'
NOTEBOOK = REPO / 'notebooks' / 'historial' / 'camera_ready_CE.ipynb'
NOTEBOOK_CELL = 20
TASK_ID = 'F2.4'
PRED_SPLITS = ('val', 'test')
FEATURES = ('shannon_entropy', 'contrast', 'edge_density', 'lbp_energy')
ORIG_METRIC = {'entropy': 'shannon_entropy', 'contrast': 'contrast', 'edges': 'edge_density', 'lbp': 'lbp_energy'}
TABLE11_ORDER = ('gray_scott', 'perlin', 'lorenz')  # orden de la celda 20 (Holm no depende del orden)
# Ejecución B (notebooks/historial/README.md, «Otra ejecución, con Lorenz distinto»): solo constan los d de
# Lorenz en entropía y LBP; Gray-Scott y Perlin «salen idénticos» a A; Lorenz contraste y bordes no constan.
RUN_B_LORENZ = {'shannon_entropy': -0.412, 'lbp_energy': 0.431}
Stop = F21.Stop
STOPS = (F21.Stop, F23.Stop)


# ── Descriptores: copia literal de la celda 20 ────────────────────────────
def shannon_entropy(img):
    hist, _ = np.histogram(img, bins=256, range=(0,256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def edge_density(img):
    edges = sobel(img.astype(float) / 255.0)
    thresh = edges.mean() + edges.std()   # umbral adaptativo simple
    return float((edges > thresh).mean())


def lbp_energy(img, P=24, R=3):
    lbp = local_binary_pattern(img, P, R, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), density=True)
    return float(np.sum(hist ** 2))


def contrast(img):
    return float(img.std())  # celda 20, línea 42: feats[key]['contrast'].append(img.std())


def descriptors_of(img):
    return {'shannon_entropy': shannon_entropy(img), 'contrast': contrast(img),
            'edge_density': edge_density(img), 'lbp_energy': lbp_energy(img)}


def read_image(data):
    """Como la celda 20: np.array(Image.open(path).convert('L')), desde los bytes ya verificados."""
    with Image.open(io.BytesIO(data)) as im:
        mode, size = im.mode, list(im.size)
        arr = np.array(im.convert('L'))
    return arr, mode, size


def check_params(dcfg):
    """Los parámetros fijados en configs/protocolo.json deben ser los del código de arriba."""
    p = dcfg['params']
    ok = ((p['shannon_entropy']['bins'], p['shannon_entropy']['range'], p['shannon_entropy']['log_base'])
          == (256, [0, 256], 2)
          and p['contrast']['ddof'] == 0
          and (p['lbp_energy']['P'], p['lbp_energy']['R'], p['lbp_energy']['method'], p['lbp_energy']['bins'])
          == (24, 3, 'uniform', 26)
          and p['edge_density']['comparison'] == '>' and p['edge_density']['input_scale'] == 'img / 255'
          and tuple(dcfg['features']) == FEATURES
          and dcfg['image'] == {'source': 'nativa', 'size': [256, 256], 'mode': 'L', 'visual_preprocessing': False}
          and dcfg['software']['scikit-image'] == skimage.__version__)
    if not ok:
        raise Stop('configs/protocolo.json → descriptores no coincide con la implementación o el entorno')


# ── Lectura verificada ────────────────────────────────────────────────────
def compute_descriptors(technique, index, manifest, limit, log):
    rel_dir = paths.REL['IMAGES'][technique]
    expected = {r['rel_path']: r['sha256'] for r in manifest if r['kind'] == 'file'
                and r['rel_path'].startswith(rel_dir + '/')}
    n_total = sum(paths.EXPECTED_SIZES.values())
    if len(expected) != n_total:
        raise Stop(f'{technique}: {len(expected)} imágenes en data_manifest.csv, se esperaban {n_total}')
    folder = paths.images_dir(technique)
    rows, t0 = [], time.time()
    for s in paths.SPLITS:
        sub = index[s] if limit is None else index[s].head(limit)
        for sid, y, name in zip(sub['sample_id'], sub['y_true'].astype(int), sub[f'img_{technique}']):
            rel = f'{rel_dir}/{name}'
            if rel not in expected:
                raise Stop(f'{rel} no está en data_manifest.csv')
            path = folder / name
            if not path.is_file():
                raise Stop(f'falta la imagen {rel}')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected[rel]:
                raise Stop(f'{rel}: sha256 distinto del manifiesto')
            if not name.endswith(f'_label{y}.png'):
                raise Stop(f'{rel}: el sufijo de etiqueta no coincide con y_true = {y}')
            img, mode, size = read_image(data)
            if mode != 'L' or size != [256, 256]:
                raise Stop(f'{rel}: modo {mode}, tamaño {size}; se esperaba L 256×256')
            rows.append({'sample_id': sid, 'technique': technique, 'split': s, 'y_true': int(y),
                         **descriptors_of(img)})
    df = pd.DataFrame(rows)
    counts = df.groupby('split').size().to_dict()
    if limit is None and counts != paths.EXPECTED_SIZES:
        raise Stop(f'{technique}: conteos por split {counts} != {paths.EXPECTED_SIZES}')
    log(f'  [{technique}] {len(df)} imágenes: sha256 = data_manifest.csv, modo L, 256×256; '
        f'train/val/test = {counts.get("train", 0)}/{counts.get("val", 0)}/{counts.get("test", 0)}; '
        f'{time.time() - t0:.1f} s')
    return df


# ── Tabla 11 ──────────────────────────────────────────────────────────────
def original_table11():
    """Salida guardada de la celda 20 (ejecución A), leída del notebook versionado."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    text = ''.join(''.join(o.get('text', '')) for o in nb['cells'][NOTEBOOK_CELL].get('outputs', []))
    lines = [ln.split() for ln in text.strip().splitlines()]
    header = lines[0]
    if header != ['technique', 'metric', 'dep_mean', 'dep_std', 'nodep_mean', 'nodep_std', 't', 'p', 'cohen_d',
                  'p_holm', 'sig_holm']:
        raise Stop(f'salida de la celda {NOTEBOOK_CELL} con encabezado inesperado: {header}')
    out = {}
    for ln in lines[1:]:
        r = dict(zip(header, ln))
        out[(r['technique'], ORIG_METRIC[r['metric']])] = {'d': float(r['cohen_d']), 'sig': r['sig_holm'] == 'True'}
    if len(out) != 12:
        raise Stop(f'la salida de la celda {NOTEBOOK_CELL} tiene {len(out)} filas, se esperaban 12')
    return out


def table11(desc, tcfg, log):
    if (tcfg['split'], tcfg['multiplicity'], tcfg['k'], tcfg['alpha']) != ('train', 'holm', 12, 0.05):
        raise Stop(f'tabla11 inesperada {tcfg}')
    orig = original_table11()
    rows = []
    log('\n— Tabla 11 recalculada (solo train)')
    for t in TABLE11_ORDER:
        tr = desc[(desc['technique'] == t) & (desc['split'] == 'train')]
        n1, n0 = int((tr['y_true'] == 1).sum()), int((tr['y_true'] == 0).sum())
        log(f'  [{t}] conteos por clase usados: depresivo {n1}, no depresivo {n0}')
        for f in FEATURES:
            a = tr.loc[tr['y_true'] == 1, f].to_numpy(dtype=np.float64)
            b = tr.loc[tr['y_true'] == 0, f].to_numpy(dtype=np.float64)
            t_stat, p = stats.ttest_ind(a, b, equal_var=False)
            d = (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)  # celda 20, línea 57
            rows.append({'technique': t, 'descriptor': f, 'n_dep': n1, 'n_nodep': n0,
                         'dep_mean': a.mean(), 'dep_std': a.std(), 'nodep_mean': b.mean(), 'nodep_std': b.std(),
                         't': float(t_stat), 'p': float(p), 'cohen_d': float(d)})
    if len(rows) != tcfg['k']:
        raise Stop(f'{len(rows)} comparaciones, se esperaban {tcfg["k"]}')
    reject, p_holm, _, _ = multipletests([r['p'] for r in rows], alpha=tcfg['alpha'], method='holm')
    for r, ph, rej in zip(rows, p_holm, reject):
        r['p_holm'], r['sig_holm'] = float(ph), bool(rej)
        o = orig[(r['technique'], r['descriptor'])]
        r['cohen_d_orig_A'], r['sig_holm_orig_A'] = o['d'], o['sig']
        r['diff_d_A'] = r['cohen_d'] - o['d']
        r['cambia_sig_A'] = r['sig_holm'] != o['sig']
        r['reproduce_d_2dec_A'] = round(r['cohen_d'], 2) == round(o['d'], 2)
        if r['technique'] != 'lorenz':
            d_b = o['d']  # idéntico a A según la README
        else:
            d_b = RUN_B_LORENZ.get(r['descriptor'], np.nan)  # contraste y bordes de B no constan
        r['cohen_d_orig_B'] = d_b
        r['diff_d_B'] = r['cohen_d'] - d_b if not np.isnan(d_b) else np.nan
    df = pd.DataFrame(rows)
    log(f'  {"técnica":<11}{"descriptor":<17}{"d":>9}{"d_A":>9}{"Δd_A":>9}{"d_B":>9}{"p_holm":>11}'
        f'  sig  sig_A  cambia')
    for r in rows:
        log(f'  {r["technique"]:<11}{r["descriptor"]:<17}{r["cohen_d"]:>9.4f}{r["cohen_d_orig_A"]:>9.4f}'
            f'{r["diff_d_A"]:>+9.4f}{r["cohen_d_orig_B"]:>9.3f}{r["p_holm"]:>11.3e}'
            f'  {"sí" if r["sig_holm"] else "no":<3}  {"sí" if r["sig_holm_orig_A"] else "no":<5}  '
            f'{"SÍ" if r["cambia_sig_A"] else "no"}')
    log(f'  significativas tras Holm (k = {len(rows)}): recalculada {int(df["sig_holm"].sum())}, '
        f'original A {int(df["sig_holm_orig_A"].sum())}')
    no_rep = df[(df['technique'] != 'lorenz') & ~df['reproduce_d_2dec_A']]
    log('  Gray-Scott y Perlin reproducen d a 2 decimales: ' + ('sí (8/8 filas)' if no_rep.empty
        else f'NO en {len(no_rep)}/8 filas → hallazgo: las imágenes de la sesión de Kaggle difieren de tt2-insumos'))
    return df


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
    dcfg, tcfg = cfg['descriptores'], cfg['techo_5d']
    if (dcfg['families'], dcfg['grids'], dcfg['selection_rule']) != \
            ('techo_5d.family_order', 'techo_5d.familias', 'techo_5d.selection'):
        raise Stop(f'descriptores apunta a otras familias, rejillas o selección: {dcfg}')
    if (dcfg['scaler']['name'], dcfg['scaler']['fit_split']) != ('StandardScaler', 'train') or \
            dcfg['scaler']['families'] != tcfg['scaler']['families']:
        raise Stop(f'escalador inesperado {dcfg["scaler"]}')
    sel_cfg = tcfg['selection']
    if (sel_cfg['metric'], sel_cfg['split'], sel_cfg['tie_break']) != ('f1_macro', 'val',
                                                                      ['family_order', 'grid_order']):
        raise Stop(f'selección inesperada {sel_cfg}')
    check_params(dcfg)
    rs, tol, order = tcfg['random_state'], sel_cfg['tie_tolerance'], tcfg['family_order']
    scaled = set(dcfg['scaler']['families'])

    log(f'F2.4 — Descriptores de bajo nivel. scikit-learn {sklearn.__version__}, scikit-image {skimage.__version__}, '
        f'numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, '
        f'statsmodels {statsmodels.__version__}, Pillow {PIL.__version__}'
        + ('  [--estimar: no escribe nada]' if estimar else ''))
    log('\n— Integridad')
    inputs = {'meta/sample_index.csv (repo)': sha256(paths.SAMPLE_INDEX),
              'meta/data_manifest.csv (repo)': sha256(paths.DATA_MANIFEST),
              'configs/protocolo.json (repo, LF)': F21.text_sha256(CONFIG),
              'notebooks/historial/camera_ready_CE.ipynb (repo, LF)': F21.text_sha256(NOTEBOOK)}
    for k, v in inputs.items():
        log(f'  sha256  {k}  {v}')
    manifest = F23.load_manifest()
    for t in paths.TECHNIQUES:
        row = next(r for r in manifest if r['kind'] == 'dir' and r['logical_name'] == f'IMAGES[{t}]')
        inputs[f'IMAGES[{t}]'] = {'rel_path': row['rel_path'], 'sha256': row['sha256'],
                                  'n_files': int(row['n_files']),
                                  'verificacion': 'sha256 de cada imagen contra meta/data_manifest.csv'}
    index = F23.load_index()
    y = {s: index[s]['y_true'].astype(int).values for s in paths.SPLITS}
    log(f'  índice maestro: train {len(y["train"])} ({int((y["train"] == 0).sum())}/{int(y["train"].sum())}), '
        f'val {len(y["val"])}, test {len(y["test"])}')

    log('\n— Descriptores (imagen nativa 256 × 256, celda 20)')
    techniques = paths.TECHNIQUES[:1] if estimar else paths.TECHNIQUES
    desc = pd.concat([compute_descriptors(t, index, manifest, 20 if estimar else None, log) for t in techniques],
                     ignore_index=True)

    log('\n— Rejilla por técnica (F1-Macro de validación con ŷ = 1[p_dep > 0.5])')
    grid_rows, sel_rows, chosen = [], [], {}
    for t in techniques:
        dt = desc[desc['technique'] == t]
        X = {s: dt[dt['split'] == s].sort_values('sample_id')[list(FEATURES)].to_numpy(dtype=np.float64)
             for s in paths.SPLITS}
        yy = {s: dt[dt['split'] == s].sort_values('sample_id')['y_true'].to_numpy(dtype=int) for s in paths.SPLITS}
        if not estimar:
            for s in paths.SPLITS:
                if not np.array_equal(yy[s], y[s]):
                    raise Stop(f'{t}/{s}: orden o etiquetas distintos del índice maestro')
        best = {}
        for fam in order:
            spec = tcfg['familias'][fam]
            fixed = F21.fixed_params(fam, spec, rs)
            combos = F21.combinations(fam, spec)
            if estimar:
                combos = combos[:1]
            fitted, f1s, tf = [], [], time.time()
            for i, combo in enumerate(combos):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model = F21.make_model(fam, fixed, combo, fam in scaled).fit(X['train'], yy['train'])
                conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
                f1 = F21.f1_val_of(yy['val'], F21.p_dep_of(model, X['val']))
                fitted.append(model)
                f1s.append(f1)
                grid_rows.append({'tecnica': t, 'familia': fam, 'orden': i,
                                  'combinacion': json.dumps(F21.jsonable(combo)), 'f1_val': f1,
                                  'n_iter': F21.n_iter_of(model), 'avisos_convergencia': conv})
            k = F21.select(f1s, tol)
            n_tie = sum(f >= max(f1s) - tol for f in f1s)
            best[fam] = {'orden': k, 'combo': combos[k], 'f1_val': f1s[k], 'model': fitted[k], 'fixed': fixed,
                         'n_empate': n_tie}
            n_conv = sum(r['avisos_convergencia'] > 0 for r in grid_rows[-len(combos):])
            log(f'  [{t}/{fam}] {len(combos)} combinaciones; mejor #{k} {json.dumps(F21.jsonable(combos[k]))}  '
                f'F1-Val = {f1s[k]:.6f} ({n_tie} dentro de la tolerancia'
                + (f'; {n_conv} con ConvergenceWarning' if n_conv else '') + f'); {time.time() - tf:.1f} s')
        win = order[F21.select([best[f]['f1_val'] for f in order], tol)]
        chosen[t] = {**best[win], 'familia': win, 'X': X, 'y': yy}
        for f in order:
            sel_rows.append({'tecnica': t, 'familia': f, 'orden': best[f]['orden'],
                             'combinacion': json.dumps(F21.jsonable(best[f]['combo'])), 'f1_val': best[f]['f1_val'],
                             'n_empate_tolerancia': best[f]['n_empate'], 'seleccionada': f == win})
        log(f'  [{t}] seleccionada: {win} #{best[win]["orden"]} {json.dumps(F21.jsonable(best[win]["combo"]))}  '
            f'F1-Val = {best[win]["f1_val"]:.6f}')

    if estimar:
        log(f'\nDuración (estimación: una técnica, 20 imágenes por split, una combinación por familia): '
            f'{time.time() - t0:.1f} s')
        return

    t11 = table11(desc, dcfg['tabla11'], log)

    desc.sort_values(['technique', 'sample_id']).to_csv(OUT_DIR / 'descriptores.csv', index=False,
                                                        lineterminator='\n', float_format='%.17g')
    pd.DataFrame(grid_rows).to_csv(OUT_DIR / 'rejilla_descriptores.csv', index=False, lineterminator='\n')
    pd.DataFrame(sel_rows).to_csv(OUT_DIR / 'seleccion.csv', index=False, lineterminator='\n')
    t11.to_csv(OUT_DIR / 'tabla11_recalculada.csv', index=False, lineterminator='\n', float_format='%.17g')
    log('\n  escrito results/f24_descriptores/{descriptores,rejilla_descriptores,seleccion,tabla11_recalculada}.csv')

    log('\n— Predicciones del modelo seleccionado (fase referencia, seed 42; test archivado sin métricas)')
    script = {'script': 'results/f24_descriptores/f24_descriptores.py',
              'script_sha256_lf': F21.text_sha256(Path(__file__)),
              'reused_code': {'results/f21_techo5d/f21_techo5d.py': F21.text_sha256(REPO / 'results/f21_techo5d/f21_techo5d.py'),
                              'results/f23_pisos/f23_pisos.py': F21.text_sha256(REPO / 'results/f23_pisos/f23_pisos.py')},
              'protocol': 'docs/protocolo_preregistrado.md §4.2, §4.3 y Enmienda 2, punto 1',
              'protocol_config': 'configs/protocolo.json',
              'protocol_config_sha256_lf': inputs['configs/protocolo.json (repo, LF)'], 'decision': 'p_dep > 0.5'}
    base_inputs = {k: inputs[k] for k in ('meta/sample_index.csv (repo)', 'meta/data_manifest.csv (repo)',
                                          'configs/protocolo.json (repo, LF)',
                                          'notebooks/historial/camera_ready_CE.ipynb (repo, LF)')}
    for t in techniques:
        c = chosen[t]
        fam = c['familia']
        p = {s: F21.p_dep_of(c['model'], c['X'][s]) for s in PRED_SPLITS}
        if F21.f1_val_of(c['y']['val'], p['val']) != c['f1_val']:
            raise RuntimeError(f'{t}: p_dep > 0.5 no reproduce el F1-Val de la selección')
        df = pd.concat([F23.frame(index, s, y[s], t, dcfg['arch'], rs, p[s]) for s in PRED_SPLITS],
                       ignore_index=True)
        config = {'features': ('4 descriptores de la imagen nativa de 256 × 256 (modo L): '
                               + ', '.join(FEATURES) + '; definición de configs/protocolo.json → descriptores '
                               '(celda 20 de notebooks/historial/camera_ready_CE.ipynb)'),
                  'familia': fam,
                  'combinacion': F21.jsonable(c['combo']),
                  'pipeline': ('StandardScaler (ajustado en train) + ' if fam in scaled else '')
                  + F21.ESTIMATORS[fam].__name__,
                  'estimator_params': F21.jsonable({**c['fixed'], **c['combo']}),
                  'other_params': f'valores por defecto de scikit-learn {sklearn.__version__}',
                  'descriptor_software': f'scikit-image {skimage.__version__}',
                  'fit': 'solo train', **script}
        selection = {'criterio': ('F1-Macro de validación (ŷ = 1[p_dep > 0.5]) sobre las cinco familias y rejillas '
                                  'de §4.3; empate con tolerancia 1e-12 → familia anterior en family_order y, '
                                  'dentro de ella, combinación anterior en la rejilla'),
                     'familia': fam, 'orden_en_rejilla': c['orden'], 'combinacion': F21.jsonable(c['combo']),
                     'f1_val': c['f1_val'], 'tabla': 'results/f24_descriptores/rejilla_descriptores.csv',
                     'seleccion': 'results/f24_descriptores/seleccion.csv'}
        write(df, config, {**base_inputs, f'IMAGES[{t}]': inputs[f'IMAGES[{t}]']}, selection, log)
    log(f'\nDuración: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--estimar', action='store_true', help='una técnica, 20 imágenes por split, una combinación '
                                                           'por familia; no escribe nada')
    args = ap.parse_args()
    log = F21.Log(OUT_DIR / ('f24_estimacion.log' if args.estimar else 'f24_descriptores.log'))
    try:
        main(args.estimar, log)
    except STOPS as e:
        log(f'\nDETENIDO: {e}')
        raise SystemExit(2)
    finally:
        if not args.estimar:
            log.write()
