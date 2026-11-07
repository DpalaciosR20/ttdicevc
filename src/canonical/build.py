"""
build.py
────────
Archivo de cifras canónicas (plan v2, F0.4) → results/canonical_numbers.json.

Toda cifra del documento de TT-II se toma de ese archivo. Cada entrada lleva:
    status      verificado | extraido | reportado | exploratorio
                  verificado   recalculado aquí desde datos por muestra
                               (predictions.csv, umap_params.pkl)
                  extraido     leído por código de un artefacto del repo, sin
                               recálculo por muestra (p. ej. una tabla de docs/)
                  reportado    transcrito de un documento, sin datos para recalcularlo
                  exploratorio usa un test consultado antes de congelar el protocolo
    value       métricas: siempre el par {"val", "test"}; un split nulo exige
                null_reason. Conteos: entero. Precisión completa (sin redondeo).
    evaluator   cómo se calculó la cifra (obligatorio)
    source      archivo del repo + localizador, o artefacto externo + sha256
    depends_on  cantidades derivadas: ids de los que se calcula (+ derivation)

Orígenes de las entradas:
  1. Colectores en código (este archivo): conteos estructurales, F3.2 (repetición,
     f32.*), F3.2 v0 sustituida (f32v0.*, histórica), selección de F3.3 (f33.*),
     HPO de F3.4 (f34.*, solo val), finalistas de F3.5 (f35.*), controles
     permutados de F3.6 (f36.*, prueba registrada por la Enmienda 4, A16),
     segundos de F5.5 (f55.*) y referencia de longitud de F4.5
     (ref.longitud.*, test bloqueado hasta la Fase 5).
  2. Fragmentos results/<tarea>/canonical_fragment.json, uno por tarea, con las
     cifras transcritas. Una clave repetida entre orígenes es un error, salvo
     sustitución explícita: la entrada que declara supersedes=<id> reemplaza a
     la entrada <id> (con la misma clave o con otra); la sustituida se conserva
     intacta en la sección superseded del archivo.

Uso (desde la raíz del repo):
    CANON_UMAP_PARAMS=<umap_params.pkl> python -m src.canonical.build          # escribe
    CANON_UMAP_PARAMS=<umap_params.pkl> python -m src.canonical.build --check  # compara

Variables de entorno (equivalen a --umap-params y --strict):
    CANON_UMAP_PARAMS  umap_params.pkl (o .npz de umap_io). Se verifica su sha256.
                       Sin él, train/test se extraen de la tabla del hallazgo 7.
    CANON_STRICT       1 → falla si falta CANON_UMAP_PARAMS (así se genera el
                       archivo commiteado).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src' / 'training'))
from umap_io import EXPECTED_SIZES, SPLITS, load_umap_arrays, sha256  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA_PATH = REPO / 'src' / 'canonical' / 'schema.json'
OUT_PATH = REPO / 'results' / 'canonical_numbers.json'
# F3.2, repetición (F1.3/F1.4/F1.5): predicciones F1.5 + resultados del grid → f32.*
F32_DIR = REPO / 'results' / 'f32_primera_pasada'
F32_PRED_DIR = REPO / 'results' / 'predictions' / 'primera_pasada'
F32_FASE = 'primera_pasada'
F32_RUNS = 54
# F3.2, primera pasada v0 (formato anterior a F1.5), sustituida → f32v0.* (histórica)
F32V0_DIR = REPO / 'results' / 'f32_primera_pasada_v0'
# F3.3: mínimo de volteos que cambia el finalista (results/f33_seleccion/fragilidad.py)
F33_VOLTEOS = REPO / 'results' / 'f33_seleccion' / 'fragilidad_volteos.csv'
HALLAZGOS = REPO / 'docs' / 'hallazgos_metodologicos.md'
FRAGMENT_GLOB = 'results/*/canonical_fragment.json'

# sha256 del umap_params.pkl de referencia (mismo valor que tests/test_label_independence.py)
UMAP_SHA256 = '081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224'
SCHEMA_VERSION = '1.0'
# De más a menos fuerte; una derivada hereda el más débil de sus dependencias
STATUS_ORDER = ('verificado', 'extraido', 'reportado', 'exploratorio')
METRIC_SPLITS = ('val', 'test')
SUPPORTS = {'val': 'support.val', 'test': 'support.test'}

F1_EVALUATOR = ("f1_macro_from_confusion (= sklearn.metrics.f1_score(average='macro')) "
                "con y_pred = 1[p_dep > 0.5], equivalente al argmax de train_model.py")
TEST_BLOCKED = ('bloqueado: F3.2 es solo validación (grid_meta.json evaluate_test=false); '
                'test se consulta una vez por finalista en F3.5')
TEST_BLOCKED_F32 = ('bloqueado: F3.2 es solo validación (fase primera_pasada; evaluate_test=false y '
                    'campos de test nulos en cada corrida); test se consulta una vez por finalista en F3.5')
F32V0_NOTE = ('Primera pasada v0 de F3.2 (21 oct 2026), sustituida por la repetición (claves f32.*): '
              'preprocesamiento anterior a F1.4 (volteo en las tres técnicas) y formato anterior a F1.5. '
              'Histórica: no participa en la selección de F3.3 (matriz §5).')
RUN_TEST_FIELDS = ('f1_test', 'acc_test', 'n_test', 'test_preds', 'test_probs', 'test_true')


class CanonicalError(ValueError):
    pass


# ── Utilidades ────────────────────────────────────────────────────────────
def rel(path):
    """Ruta relativa al repo; absoluta si está fuera (fragmentos sintéticos de las pruebas)."""
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.as_posix()


def text_sha256(path):
    """sha256 de un archivo de texto con fines de línea normalizados a LF, para
    que no dependa de core.autocrlf (Windows vs. Kaggle)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def confusion_counts(y_true, y_pred):
    """Matriz de confusión con la clase 1 (depresivo) como positiva."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {'tp': int(((y_true == 1) & (y_pred == 1)).sum()),
            'fp': int(((y_true == 0) & (y_pred == 1)).sum()),
            'fn': int(((y_true == 1) & (y_pred == 0)).sum()),
            'tn': int(((y_true == 0) & (y_pred == 0)).sum())}


def f1_macro_from_confusion(c):
    """F1-Macro binario; F1 de una clase sin predicciones ni soporte = 0 (como sklearn)."""
    def f1(tp, fp, fn):
        d = 2 * tp + fp + fn
        return 2 * tp / d if d else 0.0
    return (f1(c['tp'], c['fp'], c['fn']) + f1(c['tn'], c['fn'], c['fp'])) / 2


def derive(op, values):
    if op == 'sum':
        return sum(values)
    if op == 'mean':
        return float(np.mean(values))
    if op == 'std_ddof1':  # misma convención que run_grid.py (summary_by_config.csv)
        return float(np.std(values, ddof=1))
    if op == 'diff':  # a − b, en el orden de depends_on
        if len(values) != 2:
            raise CanonicalError(f'diff exige exactamente dos dependencias (hay {len(values)})')
        return values[0] - values[1]
    raise CanonicalError(f'derivación desconocida: {op}')


def weakest(statuses):
    return max(statuses, key=STATUS_ORDER.index)


# ── Colector 1: conteos estructurales ─────────────────────────────────────
def _count(value, status, source, evaluator):
    return {'kind': 'count', 'status': status, 'value': int(value),
            'evaluator': evaluator, 'source': source}


def _parse_hallazgo7():
    """Tabla del hallazgo 7: | split | ... | sat/n | sat1/n1 | sat0/n0 |."""
    text = HALLAZGOS.read_text(encoding='utf-8')
    rows = {}
    for m in re.finditer(r'^\| (train|val|test) \| [\d, ]+ \| (\d+)/(\d+) \| (\d+)/(\d+) \| (\d+)/(\d+) \|',
                         text, flags=re.M):
        s, sat, n, _, n1, _, n0 = m.groups()
        rows[s] = {'n': int(n), 'sat': int(sat), 'label1': int(n1), 'label0': int(n0)}
    if set(rows) != set(SPLITS):
        raise CanonicalError(f'{rel(HALLAZGOS)}: no se encontró la tabla del hallazgo 7 '
                             f'(splits leídos: {sorted(rows)})')
    return rows


def _read_predictions_v0():
    with open(F32V0_DIR / 'predictions.csv', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def collect_structural(umap_path, val_labels, inputs):
    """val_labels: (y_true de val de la repetición de F3.2, fuente), igual en las
    54 corridas (lo comprueba collect_f32). Sin pkl, n.val y los soportes de val
    salen de ahí; con pkl salen del pkl y deben coincidir con ellos."""
    out = {}
    y_rep, src_rep = val_labels
    y_rep = np.asarray(y_rep)
    if umap_path:
        digest = sha256(umap_path)
        if digest != UMAP_SHA256:
            raise CanonicalError(f'{umap_path}: sha256 {digest} != referencia {UMAP_SHA256}')
        inputs['umap_params.pkl (externo)'] = digest
        arrays = load_umap_arrays(umap_path)
        for s in SPLITS:
            P, y = arrays[s]['params'].astype(np.float64), arrays[s]['labels']
            src = {'artifact': 'umap_params.pkl', 'sha256': digest, 'locator': f"['{s}']"}
            out[f'n.{s}'] = _count(len(y), 'verificado', src, "len(labels) vía umap_io.load_umap_arrays")
            for k in (0, 1):
                out[f'support.{s}.label{k}'] = _count((y == k).sum(), 'verificado', src,
                                                      f'sum(labels == {k})')
            out[f'saturation.{s}.n_samples'] = _count(
                ((P == 0) | (P == 1)).any(axis=1).sum(), 'verificado', src,
                'muestras con algún componente u exactamente en 0 o 1 (= umap_io.saturation_report)')
    else:
        inputs[rel(HALLAZGOS)] = text_sha256(HALLAZGOS)
        table = _parse_hallazgo7()
        src = {'path': rel(HALLAZGOS), 'locator': 'hallazgo 7, tabla de saturación'}
        for s in SPLITS:
            row = table[s]
            if row['n'] != EXPECTED_SIZES[s]:
                raise CanonicalError(f'hallazgo 7: n({s}) = {row["n"]} != {EXPECTED_SIZES[s]}')
            out[f'n.{s}'] = _count(row['n'], 'extraido', src, 'denominador de «Muestras con alguno»')
            out[f'support.{s}.label1'] = _count(row['label1'], 'extraido', src, 'denominador de «label 1»')
            out[f'support.{s}.label0'] = _count(row['label0'], 'extraido', src, 'denominador de «label 0»')
            out[f'saturation.{s}.n_samples'] = _count(row['sat'], 'extraido', src,
                                                      'numerador de «Muestras con alguno»')
        # val sí tiene datos por muestra en el repo: y_true de la repetición de F3.2
        out['n.val'] = _count(len(y_rep), 'verificado', src_rep, 'filas de la corrida')
        for k in (0, 1):
            out[f'support.val.label{k}'] = _count((y_rep == k).sum(), 'verificado', src_rep,
                                                  f'sum(y_true == {k})')
    # En ambas vías, la repetición de F3.2 debe dar los mismos conteos de val
    rep = {'n.val': len(y_rep), 'support.val.label0': int((y_rep == 0).sum()),
           'support.val.label1': int((y_rep == 1).sum())}
    for nid, v in rep.items():
        if out[nid]['value'] != v:
            raise CanonicalError(f'{nid}: {out[nid]["value"]} != {v} de la repetición de F3.2 ({src_rep["path"]})')

    build_src = {'path': 'src/canonical/build.py', 'locator': 'collect_structural'}
    out['n.total'] = {'kind': 'count', 'evaluator': 'suma de splits', 'source': build_src,
                      'depends_on': [f'n.{s}' for s in SPLITS], 'derivation': 'sum'}
    for k in (0, 1):
        out[f'support.total.label{k}'] = {'kind': 'count', 'evaluator': 'suma de splits',
                                          'source': build_src, 'derivation': 'sum',
                                          'depends_on': [f'support.{s}.label{k}' for s in SPLITS]}
    return out


# ── Colector 2: primera pasada del grid (F3.2, repetición) ────────────────
def _slug(s):
    return re.sub(r'[^a-z0-9_]', '_', s.lower())


def _f32_cfg(prefix, technique, arch, strategy, variant):
    return f'{prefix}.{_slug(technique)}.{_slug(arch)}.{_slug(strategy)}.{_slug(variant)}'


def _read_csv_dict(path, key):
    with open(path, newline='', encoding='utf-8') as f:
        return {key(r): r for r in csv.DictReader(f)}


def _check_f1(label, f1, y_true, y_pred, run, summary_f1):
    from sklearn.metrics import f1_score
    checks = {'sklearn': f1_score(y_true, y_pred, average='macro'), 'best_f1_val del JSON': run['best_f1_val']}
    for what, other in checks.items():
        if not math.isclose(f1, other, rel_tol=0, abs_tol=1e-12):
            raise CanonicalError(f'{label}: F1 {f1!r} != {what} {other!r}')
    if round(f1, 4) != float(summary_f1):
        raise CanonicalError(f'{label}: F1 {f1:.4f} != summary.csv {summary_f1}')


def _f32_derived(configs, by_config_csv, src_path, note=None):
    """Media y DE (ddof=1) por configuración, comprobadas contra summary_by_config.csv."""
    out = {}
    for (cfg, key), ids in sorted(configs.items()):
        src = {'path': rel(src_path), 'locator': ' / '.join(key)}
        for suffix, op, col in (('mean', 'mean', 'f1_val_mean'), ('std', 'std_ddof1', 'f1_val_std')):
            e = {'kind': 'metric', 'metric': 'f1_macro', 'supports_ref': dict(SUPPORTS),
                 'evaluator': f'{op} sobre semillas', 'source': src,
                 'depends_on': sorted(ids), 'derivation': op,
                 '_check_csv': float(by_config_csv[key][col])}
            if note:
                e['note'] = note
            out[f'{cfg}.f1_macro_{suffix}'] = e
    return out


def load_f32_predictions():
    """Predicciones F1.5 de la repetición, validadas con load_predictions."""
    from src.predictions.validate import load_predictions
    return load_predictions(str(F32_PRED_DIR / '*.csv'))


def collect_f32(preds, inputs):
    """F3.2, repetición: un F1-Macro verificado por corrida desde los 54 CSV de
    results/predictions/primera_pasada/ (agrupados por technique, arch, strategy,
    seed), con el JSON de cada corrida tomado de config.run_json del manifiesto,
    y su media y DE por configuración. Devuelve (entradas, y_true de val, fuente)."""
    from src.predictions import schema as pred_schema

    files = sorted(F32_PRED_DIR.glob('*.csv'))
    if len(files) != F32_RUNS:
        raise CanonicalError(f'{rel(F32_PRED_DIR)}: {len(files)} CSV; se esperan {F32_RUNS}')
    if set(preds['split']) != {'val'} or set(preds['fase']) != {F32_FASE}:
        raise CanonicalError(f'{rel(F32_PRED_DIR)}: splits {sorted(set(preds["split"]))}, '
                             f'fases {sorted(set(preds["fase"]))}; se espera solo val de {F32_FASE}')
    for name in ('summary.csv', 'summary_by_config.csv', 'grid_meta.json'):
        inputs[rel(F32_DIR / name)] = text_sha256(F32_DIR / name)
    summary = _read_csv_dict(F32_DIR / 'summary.csv',
                             lambda r: (r['technique'], r['arch'], r['strategy'], r['variant'], int(r['seed'])))
    by_config_csv = _read_csv_dict(F32_DIR / 'summary_by_config.csv',
                                   lambda r: (r['technique'], r['arch'], r['strategy'], r['variant']))
    groups = {k: g for k, g in preds.groupby(['technique', 'arch', 'strategy', 'seed'], sort=True)}
    if len(groups) != F32_RUNS:
        raise CanonicalError(f'{rel(F32_PRED_DIR)}: {len(groups)} corridas; se esperan {F32_RUNS}')

    expected_ids = [f'val_{i:04d}' for i in range(EXPECTED_SIZES['val'])]
    out, configs, y_ref, y_src = {}, defaultdict(list), None, None
    for path in files:
        label = path.name
        mpath = path.with_name(pred_schema.manifest_name(path.name))
        manifest = json.loads(mpath.read_text(encoding='utf-8'))
        if manifest.get('fase') != F32_FASE:
            raise CanonicalError(f'{rel(mpath)}: fase {manifest.get("fase")!r}; se espera {F32_FASE!r}')
        if manifest.get('csv_sha256') != sha256(path):
            raise CanonicalError(f'{rel(mpath)}: csv_sha256 no coincide con {label}')
        run_path = F32_DIR / 'runs' / manifest['config']['run_json']
        run = json.loads(run_path.read_text(encoding='utf-8'))
        if run.get('fase') != F32_FASE or run.get('evaluate_test') is not False:
            raise CanonicalError(f'{rel(run_path)}: fase {run.get("fase")!r}, evaluate_test '
                                 f'{run.get("evaluate_test")!r}; se espera {F32_FASE!r} y false')
        filled = [k for k in RUN_TEST_FIELDS if run.get(k) is not None]
        if filled:
            raise CanonicalError(f'{rel(run_path)}: campos de test no nulos {filled}')
        for p in (path, mpath, run_path):
            inputs[rel(p)] = text_sha256(p)

        tech, arch, strat, seed = run['technique'], run['arch'], run['strategy'], int(run['seed'])
        key = (tech, arch, strat, seed)
        if key not in groups:
            raise CanonicalError(f'{rel(run_path)}: {key} no está entre las predicciones')
        g = groups[key]
        if path.stem != f'{tech}__{arch}__{strat}__seed{seed}':
            raise CanonicalError(f'{label}: config.run_json del manifiesto apunta a {run_path.name}')
        if run['variant'] != manifest['config']['variant']:
            raise CanonicalError(f'{label}: variante {run["variant"]!r} del JSON != {manifest["config"]["variant"]!r}')
        if set(g['split']) != {'val'} or list(g['sample_id']) != expected_ids:
            raise CanonicalError(f'{label}: se esperan solo las {len(expected_ids)} filas de val, en orden')
        y_true = [int(v) for v in g['y_true']]
        y_pred = [int(p > 0.5) for p in g['p_dep']]
        if y_pred != run['val_preds'] or y_true != run['val_true']:
            raise CanonicalError(f'{label}: p_dep/y_true no coinciden con val_preds/val_true de {run_path.name}')
        if y_ref is None:
            y_ref, y_src = y_true, {'path': rel(path), 'locator': 'y_true, split=val'}
        elif y_true != y_ref:
            raise CanonicalError(f'{label}: y_true de val distinto del de {y_src["path"]}')
        cm = confusion_counts(y_true, y_pred)
        f1 = f1_macro_from_confusion(cm)
        _check_f1(label, f1, y_true, y_pred, run, summary[(tech, arch, strat, run['variant'], seed)]['f1_val'])

        cfg = _f32_cfg('f32', tech, arch, strat, run['variant'])
        nid = f'{cfg}.seed{seed}.f1_macro'
        out[nid] = {'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
                    'value': {'val': f1, 'test': None}, 'null_reason': {'test': TEST_BLOCKED_F32},
                    'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS),
                    'evaluator': F1_EVALUATOR, 'source': {'path': rel(path), 'locator': 'split=val'}}
        configs[(cfg, (tech, arch, strat, run['variant']))].append(nid)
    out.update(_f32_derived(configs, by_config_csv, F32_DIR / 'summary_by_config.csv'))
    return out, y_ref, y_src


# ── Colector 3: primera pasada v0 (F3.2, sustituida; histórica) ──────────
def collect_f32v0(predictions, inputs):
    """Las mismas cifras que daba el colector F3.2 antes de la repetición, con el
    prefijo f32v0.* y una nota que las marca como sustituidas."""
    meta_path = F32V0_DIR / 'grid_meta.json'
    if json.loads(meta_path.read_text(encoding='utf-8')).get('evaluate_test') is not False:
        raise CanonicalError(f'{rel(meta_path)}: se esperaba evaluate_test=false')
    for name in ('predictions.csv', 'summary.csv', 'summary_by_config.csv', 'grid_meta.json'):
        inputs[rel(F32V0_DIR / name)] = text_sha256(F32V0_DIR / name)

    by_run = defaultdict(list)
    for r in predictions:
        if r['split'] != 'val':
            raise CanonicalError(f'predictions.csv: fila con split={r["split"]} en F3.2 v0')
        by_run[r['run_id']].append(r)
    summary = _read_csv_dict(F32V0_DIR / 'summary.csv',
                             lambda r: (r['technique'], r['arch'], r['strategy'], r['variant'], int(r['seed'])))
    by_config_csv = _read_csv_dict(F32V0_DIR / 'summary_by_config.csv',
                                   lambda r: (r['technique'], r['arch'], r['strategy'], r['variant']))

    out, configs = {}, defaultdict(list)
    for run_id in sorted(by_run):
        rows = by_run[run_id]
        if [r['sample_id'] for r in rows] != [f'val_{i:04d}' for i in range(EXPECTED_SIZES['val'])]:
            raise CanonicalError(f'{run_id}: sample_id fuera de orden o incompleto')
        run_path = F32V0_DIR / 'runs' / f'{run_id}.json'
        run = json.loads(run_path.read_text(encoding='utf-8'))
        inputs[rel(run_path)] = text_sha256(run_path)

        y_true = [int(r['y_true']) for r in rows]
        y_pred = [int(float(r['p_dep']) > 0.5) for r in rows]
        if y_pred != run['val_preds'] or y_true != run['val_true']:
            raise CanonicalError(f'{run_id}: predictions.csv no coincide con val_preds/val_true del JSON')
        cm = confusion_counts(y_true, y_pred)
        f1 = f1_macro_from_confusion(cm)
        key = (run['technique'], run['arch'], run['strategy'], run['variant'], run['seed'])
        _check_f1(run_id, f1, y_true, y_pred, run, summary[key]['f1_val'])

        cfg = _f32_cfg('f32v0', run['technique'], run['arch'], run['strategy'], run['variant'])
        nid = f'{cfg}.seed{run["seed"]}.f1_macro'
        out[nid] = {'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
                    'value': {'val': f1, 'test': None}, 'null_reason': {'test': TEST_BLOCKED},
                    'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS),
                    'evaluator': F1_EVALUATOR, 'note': F32V0_NOTE,
                    'source': {'path': rel(F32V0_DIR / 'predictions.csv'), 'locator': f'run_id={run_id}'}}
        configs[(cfg, key[:4])].append(nid)
    out.update(_f32_derived(configs, by_config_csv, F32V0_DIR / 'summary_by_config.csv', note=F32V0_NOTE))
    return out


# ── Colector 4: selección del finalista (F3.3) ────────────────────────────
F33_SEL_SRC = 'src/selection/select_finalist.py'
F33_ROLES = {'finalista': ('finalist', 'finalista (select_finalist, matriz §5)'),
             'segundo': ('second', 'segundo mejor (select_second, protocolo §8)'),
             'lider': ('leader', 'líder por media (matriz §5, paso 2)')}


def _f33_metric(evaluator, locator, depends_on, derivation, note):
    return {'kind': 'metric', 'metric': 'f1_macro', 'supports_ref': dict(SUPPORTS), 'evaluator': evaluator,
            'source': {'path': F33_SEL_SRC, 'locator': locator}, 'depends_on': list(depends_on),
            'derivation': derivation, 'note': note}


def _read_volteos(inputs):
    if not F33_VOLTEOS.is_file():
        raise CanonicalError(f'falta {rel(F33_VOLTEOS)} (results/f33_seleccion/fragilidad.py)')
    inputs[rel(F33_VOLTEOS)] = text_sha256(F33_VOLTEOS)
    with open(F33_VOLTEOS, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    by_tech = defaultdict(list)
    for r in rows:
        if not re.fullmatch(r'\d+', r['min_volteos']):
            raise CanonicalError(f'{rel(F33_VOLTEOS)}: {r["tecnica"]} / {r["corrida"]}: '
                                 f'min_volteos {r["min_volteos"]!r} no es un entero (búsqueda sin solución)')
        by_tech[r['tecnica']].append(r)
    return by_tech


def collect_f33(preds, f32_numbers, inputs):
    """F3.3: finalista (select_finalist) y segundo (select_second) de cada técnica,
    sobre las mismas predicciones que collect_f32. La identidad va en la clave
    (f33.<tec>.{finalista,segundo,lider}.<arch>.<estrategia>.*); cada cifra es una
    derivada de las claves f32.* por semilla y se comprueba contra la tabla de
    select_finalist. El mínimo de volteos se extrae de fragilidad_volteos.csv."""
    from src.selection import select_finalist as sf

    matrix = sf.load_matrix()
    table = sf.select_second(sf.config_table(sf.f1_by_run(preds, matrix), matrix))
    inputs[rel(sf.MATRIX_PATH)] = text_sha256(sf.MATRIX_PATH)
    volteos = _read_volteos(inputs)
    n_seeds = len(matrix['seleccion']['seeds'])

    def seed_ids(row):
        prefix = f'f32.{_slug(row.technique)}.{_slug(row.arch)}.{_slug(row.strategy)}.'
        ids = sorted(k for k in f32_numbers if k.startswith(prefix) and '.seed' in k and k.endswith('.f1_macro'))
        if len(ids) != n_seeds:
            raise CanonicalError(f'{prefix}*: {len(ids)} corridas en f32.*; se esperan {n_seeds}')
        return ids

    def vals(ids):
        return [f32_numbers[i]['value']['val'] for i in ids]

    out = {}
    for tech, g in table.groupby('technique', sort=False):
        t = _slug(tech)
        loc = f'select_second(config_table(f1_by_run(primera_pasada))), técnica {tech}'
        base, ids, rows = {}, {}, {}
        for role, (col, what) in F33_ROLES.items():
            sel = g[g[col]]
            if len(sel) != 1:
                raise CanonicalError(f'f33.{t}: {len(sel)} configuraciones con rol {role}; '
                                     f'las claves f33.* representan una por técnica')
            r = rows[role] = next(sel.itertuples())
            ids[role] = seed_ids(r)
            base[role] = f'f33.{t}.{role}.{_slug(r.arch)}.{_slug(r.strategy)}'
            note = f'{what}: {r.arch}/{r.strategy}; F1-val de la primera pasada (repetición), semillas 42, 7, 123'
            for suffix, op, col_v in (('mean', 'mean', 'f1_val_mean'), ('std', 'std_ddof1', 'f1_val_std')):
                v = derive(op, vals(ids[role]))
                if not math.isclose(v, getattr(r, col_v), rel_tol=0, abs_tol=1e-12):
                    raise CanonicalError(f'{base[role]}: {op} {v!r} != select_finalist {getattr(r, col_v)!r}')
                out[f'{base[role]}.f1_macro_{suffix}'] = _f33_metric(f'{op} sobre semillas', loc, ids[role], op, note)

        lead = rows['lider']
        lim_id = f'f33.{t}.limite_empate.f1_macro'
        lim = derive('mean', vals(ids['lider'])) - derive('std_ddof1', vals(ids['lider']))
        if not math.isclose(lim, lead.f1_val_mean - lead.tolerance, rel_tol=0, abs_tol=1e-12):
            raise CanonicalError(f'{lim_id}: {lim!r} != select_finalist {lead.f1_val_mean - lead.tolerance!r}')
        out[lim_id] = _f33_metric('media del líder − DE (ddof=1) del líder (matriz §5, paso 3)', loc,
                                  [f'{base["lider"]}.f1_macro_mean', f'{base["lider"]}.f1_macro_std'], 'diff',
                                  'límite inferior de la zona de empate')
        out[f'f33.{t}.holgura_finalista.f1_macro'] = _f33_metric(
            'media del finalista − límite de la zona de empate', loc,
            [f'{base["finalista"]}.f1_macro_mean', lim_id], 'diff',
            'holgura del finalista respecto del límite de la zona de empate (≥ 0: dentro de la zona)')

        # Volteos (results/f33_seleccion/fragilidad.py): deben referirse al finalista de select_finalist
        fin_name = f'{rows["finalista"].arch}/{rows["finalista"].strategy}'
        vrows = volteos.get(tech, [])
        single = [r for r in vrows if r['busqueda'] == 'una corrida']
        spread = [r for r in vrows if r['busqueda'].startswith('repartida')]
        if not single or len(spread) != 1:
            raise CanonicalError(f'{rel(F33_VOLTEOS)}: {tech}: faltan filas de búsqueda')
        if not any(r['corrida'].startswith(fin_name + ' ') for r in single) or \
                any(r['nuevo_finalista'] == fin_name for r in vrows):
            raise CanonicalError(f'{rel(F33_VOLTEOS)}: {tech}: no corresponde al finalista {fin_name}')
        note = (f'volteos de predicción en val que cambian el finalista {fin_name} (regla §5 completa, '
                f'demás configuraciones fijas); descriptivo, no cambia la selección')
        out[f'f33.{t}.volteos_min.una_corrida'] = {
            'kind': 'count', 'status': 'extraido', 'value': min(int(r['min_volteos']) for r in single),
            'evaluator': ('mínimo de min_volteos sobre las filas «una corrida» (búsqueda exacta hasta 40 '
                          'volteos en una sola corrida del finalista o del líder)'),
            'source': {'path': rel(F33_VOLTEOS), 'locator': f'tecnica={tech}, busqueda=una corrida'}, 'note': note}
        out[f'f33.{t}.volteos_min.repartido'] = {
            'kind': 'count', 'status': 'extraido', 'value': int(spread[0]['min_volteos']),
            'evaluator': ('min_volteos de la fila «repartida» (búsqueda exacta hasta 6 volteos entre las '
                          'corridas del finalista y del líder)'),
            'source': {'path': rel(F33_VOLTEOS), 'locator': f'tecnica={tech}, busqueda=repartida'}, 'note': note}
    return out


# ── Colector 4: HPO de los finalistas (F3.4, solo validación) ─────────────
F34_DIR = REPO / 'results' / 'f34_hpo'
F34_TEST_NULL = 'F3.4 solo validación (matriz §4)'
F34_SEEDS = (42, 7, 123)
F34_EXPECTED_TRIALS = 12
TECHNIQUES = ('gray_scott', 'perlin', 'lorenz')


def _verified_val(label, y_true, probs, src, null_test, note=None):
    """F1-Macro verificado de val desde probabilidades por muestra (ŷ = 1[p > 0.5])."""
    y_pred = [int(p > 0.5) for p in probs]
    cm = confusion_counts(y_true, y_pred)
    e = {'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
         'value': {'val': f1_macro_from_confusion(cm), 'test': None}, 'null_reason': {'test': null_test},
         'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS), 'evaluator': F1_EVALUATOR, 'source': src}
    if note:
        e['note'] = note
    return e


def _derived(ids, op, src, evaluator, note=None):
    e = {'kind': 'metric', 'metric': 'f1_macro', 'supports_ref': dict(SUPPORTS), 'evaluator': evaluator,
         'source': src, 'depends_on': list(ids), 'derivation': op}
    if note:
        e['note'] = note
    return e


def collect_f34(inputs):
    """F3.4: trial 0 y mejor trial de cada finalista (media y DE ddof=1 sobre las
    semillas 42, 7, 123), ganancia, trial adoptado y número de trials. Los F1 se
    recalculan desde val_probs/val_true de results/f34_hpo/<tec>/runs/*.json y se
    contrastan con hpo_result.json y trials.csv. Sin test (matriz §4)."""
    from sklearn.metrics import f1_score
    out = {}
    for tech in TECHNIQUES:
        d = F34_DIR / tech
        res_path, trials_path = d / 'hpo_result.json', d / 'trials.csv'
        for p in (res_path, trials_path):
            inputs[rel(p)] = text_sha256(p)
        res = json.loads(res_path.read_text(encoding='utf-8'))
        dec = res['decision']
        trials_csv = _read_csv_dict(trials_path, lambda r: int(r['trial']))
        runs = {}
        for p in sorted((d / 'runs').glob('trial*__seed*.json')):
            r = json.loads(p.read_text(encoding='utf-8'))
            if r.get('fase') != 'hpo' or r.get('smoke') is not False:
                raise CanonicalError(f'{rel(p)}: fase {r.get("fase")!r}, smoke {r.get("smoke")!r}')
            filled = [k for k in RUN_TEST_FIELDS if r.get(k) is not None]
            if filled:
                raise CanonicalError(f'{rel(p)}: campos de test no nulos {filled}')
            runs[(int(r['trial']), int(r['seed']))] = (p, r)
        trial_ids = sorted({t for t, _ in runs})
        if trial_ids != list(range(F34_EXPECTED_TRIALS)) or res['n_trials'] != F34_EXPECTED_TRIALS \
                or sorted(trials_csv) != trial_ids:
            raise CanonicalError(f'{rel(d)}: trials {trial_ids}; hpo_result n_trials {res["n_trials"]}')
        f1 = {}
        for (t, s), (p, r) in runs.items():
            if s not in F34_SEEDS:
                raise CanonicalError(f'{rel(p)}: semilla {s}')
            y_pred = [int(v > 0.5) for v in r['val_probs']]
            if y_pred != r['val_preds']:
                raise CanonicalError(f'{rel(p)}: val_preds != 1[val_probs > 0.5]')
            f1[(t, s)] = f1_macro_from_confusion(confusion_counts(r['val_true'], y_pred))
            for other in (r['best_f1_val'], f1_score(r['val_true'], y_pred, average='macro')):
                if not math.isclose(f1[(t, s)], other, rel_tol=0, abs_tol=1e-12):
                    raise CanonicalError(f'{rel(p)}: F1 {f1[(t, s)]!r} != {other!r}')
        if len(f1) != F34_EXPECTED_TRIALS * len(F34_SEEDS):
            raise CanonicalError(f'{rel(d)}: {len(f1)} corridas')
        means = {t: float(np.mean([f1[(t, s)] for s in F34_SEEDS])) for t in trial_ids}
        stds = {t: float(np.std([f1[(t, s)] for s in F34_SEEDS], ddof=1)) for t in trial_ids}
        for t in trial_ids:
            row = trials_csv[t]
            if not (math.isclose(means[t], float(row['f1_val_mean']), rel_tol=0, abs_tol=1e-12)
                    and math.isclose(stds[t], float(row['f1_val_std']), rel_tol=0, abs_tol=1e-12)):
                raise CanonicalError(f'{rel(trials_path)}: trial {t} no coincide con runs/')
        best = max(trial_ids, key=lambda t: (means[t], -t))
        gain = means[best] - means[0]
        adopted = best if gain > stds[0] else 0          # matriz §6: se conserva el trial 0 si gana ≤ 1 DE
        checks = {'trial0_mean': means[0], 'trial0_std': stds[0], 'best_mean': means[best], 'gain': gain}
        for k, v in checks.items():
            if not math.isclose(v, dec[k], rel_tol=0, abs_tol=1e-12):
                raise CanonicalError(f'{rel(res_path)}: decision.{k} {dec[k]!r} != {v!r}')
        if (dec['best_trial'], dec['adopted_trial'], dec['trial0']) != (best, adopted, 0):
            raise CanonicalError(f'{rel(res_path)}: decision {dec} != best {best}, adoptado {adopted}')

        base = f'f34.{tech}'
        note = (f'{res["arch"]}/{res["strategy"]} ({res["variant"]}); F1-val por semilla recalculado desde '
                f'val_probs de runs/; test no existe en F3.4')
        ids = {}
        for role, t in (('trial0', 0), ('mejor_trial', best)):
            ids[role] = []
            for s in F34_SEEDS:
                p, r = runs[(t, s)]
                inputs[rel(p)] = text_sha256(p)
                nid = f'{base}.{role}.seed{s}.f1_macro'
                out[nid] = _verified_val(nid, r['val_true'], r['val_probs'],
                                         {'path': rel(p), 'locator': 'val_probs, val_true'}, F34_TEST_NULL,
                                         note=f'trial {t}; ' + note)
                ids[role].append(nid)
        src = {'path': rel(res_path), 'locator': 'decision'}
        for suffix, op in (('mean', 'mean'), ('std', 'std_ddof1')):
            out[f'{base}.trial0.f1_macro_{suffix}'] = _derived(ids['trial0'], op, src, f'{op} sobre semillas',
                                                               'trial 0 (configuración común de F3.2)')
        out[f'{base}.mejor_trial.f1_macro_mean'] = _derived(ids['mejor_trial'], 'mean', src, 'mean sobre semillas',
                                                            f'mejor trial por F1-val medio: trial {best}')
        out[f'{base}.ganancia.f1_macro'] = _derived(
            [f'{base}.mejor_trial.f1_macro_mean', f'{base}.trial0.f1_macro_mean'], 'diff', src,
            'media del mejor trial − media del trial 0',
            'se adopta el mejor trial solo si la ganancia supera 1 DE (ddof=1) del trial 0 (matriz §6)')
        out[f'{base}.trial_adoptado'] = {
            'kind': 'count', 'status': 'verificado', 'value': int(adopted),
            'evaluator': ('regla de la matriz §6 aplicada a las medias y DE recalculadas desde runs/ '
                          '(los 36 entrenamientos); coincide con decision.adopted_trial'),
            'source': src}
        out[f'{base}.trials'] = {
            'kind': 'count', 'status': 'extraido', 'value': len(trial_ids),
            'evaluator': 'trials distintos en runs/ (= n_trials de hpo_result.json = filas de trials.csv)',
            'source': {'path': rel(trials_path), 'locator': 'columna trial'}}
    return out


# ── Colectores 5–7: fases con prueba (F3.5, F3.6, F5.5) ──────────────────
F3X_PHASES = {
    # prefijo: (fase, carpeta de resultados, rol, columna de select_finalist, nota)
    'f35': ('final', 'f35_final', 'finalista', 'finalist',
            'finalista (select_finalist), hiperparámetros de F3.4; consulta única de prueba (protocolo §10)'),
    'f36': ('control_permutacion', 'f36_permutacion', 'permutado', 'finalist',
            ('control permutado del finalista (y_train permutado, default_rng(99)); su prueba se registra como '
             'canónica por la Enmienda 4, A16 (publicada en F3.6 antes de la Fase 5: desviación menor, sin '
             'decisiones dependientes). Su media es el piso visual permutado (§4.4)')),
    'f55': ('sensibilidad', 'f55_sensibilidad', 'segundo', 'second',
            'segundo mejor (select_second, protocolo §8), configuración común; consulta única de prueba'),
}
F3X_SEEDS = (42, 7, 123, 2024, 99)


def collect_f3x(prefix, selection_table, inputs):
    """F1-Macro de val y de prueba por semilla (verificado desde
    results/predictions/<fase>/*.csv) y media y DE (ddof=1) por técnica, para la
    configuración que select_finalist / select_second asigna a cada técnica.
    Cada CSV se contrasta con el JSON de su corrida, summary.csv y
    summary_by_config.csv."""
    from sklearn.metrics import f1_score
    from src.predictions import schema as pred_schema
    from src.predictions.validate import load_predictions

    fase, rdir, role, col, note = F3X_PHASES[prefix]
    pred_dir = REPO / 'results' / 'predictions' / fase
    preds = load_predictions(str(pred_dir / '*.csv'))
    expected = {r.technique: (r.arch, r.strategy) for r in selection_table[selection_table[col]].itertuples()}
    if sorted(expected) != sorted(TECHNIQUES):
        raise CanonicalError(f'{prefix}: la selección no cubre las tres técnicas: {expected}')
    out = {}
    ids_val = {s: [f'val_{i:04d}' for i in range(EXPECTED_SIZES[s])] if s == 'val' else
               [f'test_{i:04d}' for i in range(EXPECTED_SIZES[s])] for s in METRIC_SPLITS}
    for tech in TECHNIQUES:
        arch, strat = expected[tech]
        g_tech = preds[preds['technique'] == tech]
        if set(zip(g_tech['arch'], g_tech['strategy'])) != {(arch, strat)}:
            raise CanonicalError(f'{prefix}.{tech}: configuraciones {sorted(set(zip(g_tech["arch"], g_tech["strategy"])))}'
                                 f' != selección {(arch, strat)}')
        if sorted(set(g_tech['seed'])) != sorted(F3X_SEEDS):
            raise CanonicalError(f'{prefix}.{tech}: semillas {sorted(set(g_tech["seed"]))}')
        res_dir = REPO / 'results' / rdir / tech
        for name in ('summary.csv', 'summary_by_config.csv'):
            inputs[rel(res_dir / name)] = text_sha256(res_dir / name)
        summary = _read_csv_dict(res_dir / 'summary.csv', lambda r: int(r['seed']))
        by_config = list(_read_csv_dict(res_dir / 'summary_by_config.csv', lambda r: r['technique']).values())
        if len(by_config) != 1 or (by_config[0]['arch'], by_config[0]['strategy']) != (arch, strat):
            raise CanonicalError(f'{rel(res_dir / "summary_by_config.csv")}: se espera una fila {arch}/{strat}')
        base = f'{prefix}.{tech}.{role}'
        seed_ids = []
        for seed in F3X_SEEDS:
            path = pred_dir / pred_schema.file_name(tech, arch, strat, seed)
            mpath = path.with_name(pred_schema.manifest_name(path.name))
            manifest = json.loads(mpath.read_text(encoding='utf-8'))
            if manifest.get('csv_sha256') != sha256(path) or manifest.get('fase') != fase:
                raise CanonicalError(f'{rel(mpath)}: csv_sha256 o fase no coinciden')
            run_path = REPO / manifest['config']['run_json'] if '/' in manifest['config']['run_json'] else \
                res_dir / 'runs' / manifest['config']['run_json']
            run = json.loads(run_path.read_text(encoding='utf-8'))
            if (run['fase'], run['evaluate_test'], run['smoke']) != (fase, True, False) \
                    or not run['selection'].get('matches') or run['selection'].get('override'):
                raise CanonicalError(f'{rel(run_path)}: fase/evaluate_test/selección inesperados')
            if (run['permute_labels'] == 99) != (fase == 'control_permutacion'):
                raise CanonicalError(f'{rel(run_path)}: permute_labels {run["permute_labels"]!r}')
            for p in (path, mpath, run_path):
                inputs[rel(p)] = text_sha256(p)
            g = g_tech[g_tech['seed'] == seed]
            value, conf = {}, {}
            for s in METRIC_SPLITS:
                gs = g[g['split'] == s]
                if list(gs['sample_id']) != ids_val[s]:
                    raise CanonicalError(f'{path.name}: filas de {s} distintas del índice')
                y_true = [int(v) for v in gs['y_true']]
                probs = list(gs['p_dep'])
                if probs != list(run[f'{s}_probs']) or y_true != list(run[f'{s}_true']):
                    raise CanonicalError(f'{path.name}: {s} distinto de {run_path.name}')
                y_pred = [int(p > 0.5) for p in probs]
                conf[s] = confusion_counts(y_true, y_pred)
                value[s] = f1_macro_from_confusion(conf[s])
                ref = run['best_f1_val'] if s == 'val' else run['f1_test']
                for other in (ref, f1_score(y_true, y_pred, average='macro')):
                    if not math.isclose(value[s], other, rel_tol=0, abs_tol=1e-12):
                        raise CanonicalError(f'{path.name}: F1 {s} {value[s]!r} != {other!r}')
                if round(value[s], 4) != float(summary[seed][f'f1_{s}']):
                    raise CanonicalError(f'{path.name}: F1 {s} != summary.csv {summary[seed][f"f1_{s}"]}')
            nid = f'{base}.{_slug(arch)}.{_slug(strat)}.seed{seed}.f1_macro'
            out[nid] = {'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado', 'value': value,
                        'confusion': conf, 'supports_ref': dict(SUPPORTS), 'evaluator': F1_EVALUATOR,
                        'source': {'path': rel(path), 'locator': 'split=val, split=test'},
                        'note': f'{arch}/{strat}; ' + note}
            seed_ids.append(nid)
        src = {'path': rel(res_dir / 'summary_by_config.csv'), 'locator': f'{tech} / {arch} / {strat}'}
        for suffix, op, f in (('mean', 'mean', np.mean), ('std', 'std_ddof1', lambda x: np.std(x, ddof=1))):
            for s in METRIC_SPLITS:
                v = float(f([out[i]['value'][s] for i in seed_ids]))
                if round(v, 4) != float(by_config[0][f'f1_{s}_{suffix}']):
                    raise CanonicalError(f'{base}.f1_macro_{suffix}[{s}]: {v:.4f} != summary_by_config.csv')
            out[f'{base}.f1_macro_{suffix}'] = _derived(seed_ids, op, src, f'{op} sobre las 5 semillas',
                                                        f'{arch}/{strat}; ' + note)
    return out


# ── Colector 8: referencia de longitud (F4.5, Enmienda 2 punto 4 y Enmienda 4 A9) ─
LONGITUD_CSV = REPO / 'results' / 'predictions' / 'referencia' / 'text_length__logreg__none__seed42.csv'
LONGITUD_RESULT = REPO / 'results' / 'f45_longitud' / 'resultado.json'
REF_TEST_BLOCKED = ('bloqueado hasta la Fase 5 (docs/protocolo_preregistrado.md §10); '
                    'las predicciones de test están archivadas')


def collect_ref_longitud(inputs):
    """F1-Macro de val verificado desde el CSV de la referencia de longitud; test
    nulo (bloqueado hasta la Fase 5, como las demás referencias). Solo se leen
    las filas de val."""
    from src.predictions import schema as pred_schema
    from src.predictions.validate import validate_predictions
    df = validate_predictions(LONGITUD_CSV)
    mpath = LONGITUD_CSV.with_name(pred_schema.manifest_name(LONGITUD_CSV.name))
    for p in (LONGITUD_CSV, mpath, LONGITUD_RESULT):
        inputs[rel(p)] = text_sha256(p)
    val = df[df['split'] == 'val'].sort_values('sample_id')
    if len(val) != EXPECTED_SIZES['val']:
        raise CanonicalError(f'{rel(LONGITUD_CSV)}: {len(val)} filas de val')
    nid = 'ref.longitud.f1_macro'
    e = _verified_val(nid, [int(v) for v in val['y_true']], list(val['p_dep']),
                      {'path': rel(LONGITUD_CSV), 'locator': 'split=val'}, REF_TEST_BLOCKED,
                      note=('LogReg sobre ln(n_tokens_beto) y truncated_128 (F4.5; Enmienda 2, punto 4; '
                            'Enmienda 4, A9); selección por F1-val en la rejilla de §4.3'))
    res = json.loads(LONGITUD_RESULT.read_text(encoding='utf-8'))
    if not math.isclose(e['value']['val'], res['f1_val'], rel_tol=0, abs_tol=1e-12):
        raise CanonicalError(f'{nid}: {e["value"]["val"]!r} != resultado.json {res["f1_val"]!r}')
    return {nid: e}

# ── Fragmentos, fusión y derivadas ────────────────────────────────────────
def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))


def validate(doc, fragment=False):
    import jsonschema
    schema = load_schema()
    if fragment:
        schema = {**schema, '$ref': '#/$defs/fragment'}
        for k in ('type', 'additionalProperties', 'required', 'properties'):
            schema.pop(k)
    jsonschema.validate(doc, schema, cls=jsonschema.Draft202012Validator)


def load_fragments(paths, inputs):
    out = {}
    for path in paths:
        frag = json.loads(Path(path).read_text(encoding='utf-8'))
        validate(frag, fragment=True)
        inputs[rel(path)] = text_sha256(path)
        out[rel(path)] = frag['numbers']
    return out


def merge(sources):
    """sources: {origen: {id: entrada}} → (numbers, superseded).

    Una clave repetida es un error, salvo que exactamente una de las dos
    entradas declare supersedes con esa misma clave: esa gana, sin importar el
    orden de los orígenes. Una entrada con supersedes=<otro id> retira esa otra
    entrada. La sustituida va intacta a superseded[id] = {by, origin, entry}.
    Error si el objetivo no existe o si dos entradas sustituyen al mismo id."""
    by_id = defaultdict(list)
    for name, numbers in sources.items():
        for nid, entry in numbers.items():
            by_id[nid].append((name, entry))

    merged, origin, superseded = {}, {}, {}
    for nid, items in by_id.items():
        if len(items) == 1:
            origin[nid], merged[nid] = items[0]
            continue
        claim = [(n, e) for n, e in items if e.get('supersedes') == nid]
        if len(items) != 2 or len(claim) != 1:
            names = ' y en '.join(n for n, _ in items)
            extra = '; más de una entrada la sustituye' if len(claim) > 1 else ''
            raise CanonicalError(f'clave duplicada {nid!r}: aparece en {names}{extra}')
        (win_name, win), = claim
        (old_name, old), = [(n, e) for n, e in items if e is not win]
        merged[nid], origin[nid] = win, win_name
        superseded[nid] = {'by': f'{win_name}:{nid}', 'origin': old_name, 'entry': old}

    targets = {}  # id sustituido → id que lo sustituye
    for nid in sorted(merged):
        target = merged[nid].get('supersedes')
        if target is None:
            continue
        if target in targets:
            raise CanonicalError(f'{target!r} sustituida dos veces: por {targets[target]!r} y por {nid!r}')
        targets[target] = nid
        if target == nid:
            if nid not in superseded:
                raise CanonicalError(f'{nid}: supersedes apunta a {target!r}, que no existe en otro origen')
            continue
        if target in superseded:
            raise CanonicalError(f'{target!r} sustituida dos veces')
        if target not in merged:
            raise CanonicalError(f'{nid}: supersedes apunta a {target!r}, que no existe')
        superseded[target] = {'by': f'{origin[nid]}:{nid}', 'origin': origin[target],
                              'entry': merged[target]}
    for target, nid in targets.items():
        if target != nid:
            del merged[target]
    return merged, superseded


def resolve_derived(numbers):
    """Calcula value y status de las derivadas en orden topológico. Si la entrada
    ya trae value/status (fragmentos), deben coincidir con lo calculado."""
    state = {}  # id → 'visiting' | 'done'

    def visit(nid, chain):
        if nid not in numbers:
            raise CanonicalError(f'dependencia inexistente {nid!r} (desde {" → ".join(chain)})')
        if state.get(nid) == 'done':
            return
        if state.get(nid) == 'visiting':
            raise CanonicalError(f'ciclo en depends_on: {" → ".join(chain + [nid])}')
        e = numbers[nid]
        if 'depends_on' in e:
            state[nid] = 'visiting'
            for dep in e['depends_on']:
                visit(dep, chain + [nid])
            _compute(nid, e, [numbers[d] for d in e['depends_on']])
        state[nid] = 'done'

    for nid in sorted(numbers):
        visit(nid, [])


def _compute(nid, e, deps):
    if any(d['kind'] != e['kind'] for d in deps):
        raise CanonicalError(f'{nid}: mezcla de kinds en depends_on')
    if e['derivation'] == 'diff' and len(deps) != 2:
        raise CanonicalError(f'{nid}: diff exige exactamente dos dependencias (hay {len(deps)})')
    if e['kind'] == 'count':
        value = derive(e['derivation'], [d['value'] for d in deps])
        null_reason = None
    else:
        value, null_reason = {}, {}
        for s in METRIC_SPLITS:
            vals = [d['value'][s] for d in deps]
            if e['derivation'] == 'diff' and any(v is None for v in vals):
                # Una diferencia con un término ausente no existe: el split queda nulo
                value[s] = None
                missing = [i for i, v in zip(e['depends_on'], vals) if v is None]
                null_reason[s] = f'diferencia sin término en este split ({", ".join(missing)} es nulo)'
            elif all(v is None for v in vals):
                value[s] = None
                reasons = {d['null_reason'][s] for d in deps}
                null_reason[s] = reasons.pop() if len(reasons) == 1 else 'dependencias nulas en este split'
            elif any(v is None for v in vals):
                raise CanonicalError(f'{nid}: split {s} mezcla dependencias nulas y no nulas')
            else:
                value[s] = derive(e['derivation'], vals)
    status = weakest([d['status'] for d in deps])
    for field, computed in (('value', value), ('status', status)):
        if field in e and e[field] != computed:
            raise CanonicalError(f'{nid}: {field} declarado {e[field]!r} != calculado {computed!r}')
    e['value'], e['status'] = value, status
    if null_reason:
        e['null_reason'] = null_reason
    check = e.pop('_check_csv', None)
    if check is not None and round(value['val'], 4) != check:
        raise CanonicalError(f'{nid}: {value["val"]:.4f} != summary_by_config.csv {check}')


# ── Ensamblado ────────────────────────────────────────────────────────────
def build(umap_path=None, fragment_paths=None, strict=False):
    if strict and not umap_path:
        raise CanonicalError('CANON_STRICT=1 exige CANON_UMAP_PARAMS (umap_params.pkl)')
    if umap_path and not Path(umap_path).is_file():
        raise CanonicalError(f'CANON_UMAP_PARAMS no existe: {umap_path}')
    if fragment_paths is None:
        fragment_paths = sorted(REPO.glob(FRAGMENT_GLOB))
    inputs = {}
    preds = load_f32_predictions()
    f32, y_val, y_src = collect_f32(preds, inputs)
    sources = {'build.py:collect_structural': collect_structural(umap_path, (y_val, y_src), inputs),
               'build.py:collect_f32': f32,
               'build.py:collect_f32v0': collect_f32v0(_read_predictions_v0(), inputs),
               'build.py:collect_f33': collect_f33(preds, f32, inputs),
               'build.py:collect_f34': collect_f34(inputs)}
    from src.selection import select_finalist as sf
    matrix = sf.load_matrix()
    selection = sf.select_second(sf.config_table(sf.f1_by_run(preds, matrix), matrix))
    for prefix in F3X_PHASES:
        sources[f'build.py:collect_{prefix}'] = collect_f3x(prefix, selection, inputs)
    sources['build.py:collect_ref_longitud'] = collect_ref_longitud(inputs)
    sources.update(load_fragments(fragment_paths, inputs))
    numbers, superseded = merge(sources)
    resolve_derived(numbers)
    doc = {'schema_version': SCHEMA_VERSION, 'inputs': inputs, 'numbers': numbers}
    if superseded:
        doc['superseded'] = superseded
    validate(doc)
    return doc


def dumps(doc):
    return json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--umap-params', default=os.environ.get('CANON_UMAP_PARAMS') or None)
    ap.add_argument('--strict', action='store_true', default=os.environ.get('CANON_STRICT', '0') == '1')
    ap.add_argument('--out', default=str(OUT_PATH))
    ap.add_argument('--check', action='store_true', help='no escribe; falla si --out difiere del build')
    args = ap.parse_args()

    text = dumps(build(args.umap_params, strict=args.strict))
    out = Path(args.out)
    if args.check:
        current = out.read_bytes().replace(b'\r\n', b'\n').decode('utf-8') if out.exists() else None
        if current != text:
            sys.exit(f'{out}: difiere del build actual; regenerar con python -m src.canonical.build')
        print(f'{out}: al día ({text.count(chr(10))} líneas)')
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f'escrito {out}')


if __name__ == '__main__':
    main()
