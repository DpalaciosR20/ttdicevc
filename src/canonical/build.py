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
  1. Colectores en código (este archivo): conteos estructurales y F3.2.
  2. Fragmentos results/<tarea>/canonical_fragment.json, uno por tarea, con las
     cifras transcritas. Una clave repetida entre orígenes es un error.

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

SCHEMA_PATH = REPO / 'src' / 'canonical' / 'schema.json'
OUT_PATH = REPO / 'results' / 'canonical_numbers.json'
F32_DIR = REPO / 'results' / 'f32_primera_pasada'
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


def _read_predictions():
    with open(F32_DIR / 'predictions.csv', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def collect_structural(umap_path, predictions, inputs):
    out = {}
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
        # val sí tiene datos por muestra en el repo: y_true de predictions.csv
        first_run = predictions[0]['run_id']
        y = np.array([int(r['y_true']) for r in predictions if r['run_id'] == first_run])
        src = {'path': rel(F32_DIR / 'predictions.csv'), 'locator': f'y_true, run_id={first_run}'}
        out['n.val'] = _count(len(y), 'verificado', src, 'filas de la corrida')
        for k in (0, 1):
            out[f'support.val.label{k}'] = _count((y == k).sum(), 'verificado', src, f'sum(y_true == {k})')

    build_src = {'path': 'src/canonical/build.py', 'locator': 'collect_structural'}
    out['n.total'] = {'kind': 'count', 'evaluator': 'suma de splits', 'source': build_src,
                      'depends_on': [f'n.{s}' for s in SPLITS], 'derivation': 'sum'}
    for k in (0, 1):
        out[f'support.total.label{k}'] = {'kind': 'count', 'evaluator': 'suma de splits',
                                          'source': build_src, 'derivation': 'sum',
                                          'depends_on': [f'support.{s}.label{k}' for s in SPLITS]}
    return out


# ── Colector 2: primera pasada del grid (F3.2) ────────────────────────────
def _slug(s):
    return re.sub(r'[^a-z0-9_]', '_', s.lower())


def collect_f32(predictions, inputs):
    meta_path = F32_DIR / 'grid_meta.json'
    if json.loads(meta_path.read_text(encoding='utf-8')).get('evaluate_test') is not False:
        raise CanonicalError(f'{rel(meta_path)}: se esperaba evaluate_test=false')
    for name in ('predictions.csv', 'summary.csv', 'summary_by_config.csv', 'grid_meta.json'):
        inputs[rel(F32_DIR / name)] = text_sha256(F32_DIR / name)

    from sklearn.metrics import f1_score

    by_run = defaultdict(list)
    for r in predictions:
        if r['split'] != 'val':
            raise CanonicalError(f'predictions.csv: fila con split={r["split"]} en F3.2')
        by_run[r['run_id']].append(r)
    with open(F32_DIR / 'summary.csv', newline='', encoding='utf-8') as f:
        summary = {(r['technique'], r['arch'], r['strategy'], r['variant'], int(r['seed'])): r
                   for r in csv.DictReader(f)}
    with open(F32_DIR / 'summary_by_config.csv', newline='', encoding='utf-8') as f:
        by_config_csv = {(r['technique'], r['arch'], r['strategy'], r['variant']): r
                         for r in csv.DictReader(f)}

    out, configs = {}, defaultdict(list)
    for run_id in sorted(by_run):
        rows = by_run[run_id]
        if [r['sample_id'] for r in rows] != [f'val_{i:04d}' for i in range(EXPECTED_SIZES['val'])]:
            raise CanonicalError(f'{run_id}: sample_id fuera de orden o incompleto')
        run_path = F32_DIR / 'runs' / f'{run_id}.json'
        run = json.loads(run_path.read_text(encoding='utf-8'))
        inputs[rel(run_path)] = text_sha256(run_path)

        y_true = [int(r['y_true']) for r in rows]
        y_pred = [int(float(r['p_dep']) > 0.5) for r in rows]
        if y_pred != run['val_preds'] or y_true != run['val_true']:
            raise CanonicalError(f'{run_id}: predictions.csv no coincide con val_preds/val_true del JSON')
        cm = confusion_counts(y_true, y_pred)
        f1 = f1_macro_from_confusion(cm)
        key = (run['technique'], run['arch'], run['strategy'], run['variant'], run['seed'])
        checks = {'sklearn': f1_score(y_true, y_pred, average='macro'),
                  'best_f1_val del JSON': run['best_f1_val']}
        for what, other in checks.items():
            if not math.isclose(f1, other, rel_tol=0, abs_tol=1e-12):
                raise CanonicalError(f'{run_id}: F1 {f1!r} != {what} {other!r}')
        if round(f1, 4) != float(summary[key]['f1_val']):
            raise CanonicalError(f'{run_id}: F1 {f1:.4f} != summary.csv {summary[key]["f1_val"]}')

        cfg = f'f32.{_slug(run["technique"])}.{_slug(run["arch"])}.{_slug(run["strategy"])}.{_slug(run["variant"])}'
        nid = f'{cfg}.seed{run["seed"]}.f1_macro'
        out[nid] = {'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
                    'value': {'val': f1, 'test': None}, 'null_reason': {'test': TEST_BLOCKED},
                    'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS),
                    'evaluator': F1_EVALUATOR,
                    'source': {'path': rel(F32_DIR / 'predictions.csv'), 'locator': f'run_id={run_id}'}}
        configs[(cfg, key[:4])].append(nid)

    for (cfg, key), ids in sorted(configs.items()):
        src = {'path': rel(F32_DIR / 'summary_by_config.csv'), 'locator': ' / '.join(key)}
        for suffix, op, col in (('mean', 'mean', 'f1_val_mean'), ('std', 'std_ddof1', 'f1_val_std')):
            out[f'{cfg}.f1_macro_{suffix}'] = {
                'kind': 'metric', 'metric': 'f1_macro', 'supports_ref': dict(SUPPORTS),
                'evaluator': f'{op} sobre semillas', 'source': src,
                'depends_on': sorted(ids), 'derivation': op,
                '_check_csv': float(by_config_csv[key][col])}
    return out


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
    """sources: {origen: {id: entrada}} → {id: entrada}. Clave repetida → error."""
    merged, origin = {}, {}
    for name, numbers in sources.items():
        for nid, entry in numbers.items():
            if nid in merged:
                raise CanonicalError(f'clave duplicada {nid!r}: aparece en {origin[nid]} y en {name}')
            merged[nid], origin[nid] = entry, name
    return merged


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
    if e['kind'] == 'count':
        value = derive(e['derivation'], [d['value'] for d in deps])
        null_reason = None
    else:
        value, null_reason = {}, {}
        for s in METRIC_SPLITS:
            vals = [d['value'][s] for d in deps]
            if all(v is None for v in vals):
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
    predictions = _read_predictions()
    sources = {'build.py:collect_structural': collect_structural(umap_path, predictions, inputs),
               'build.py:collect_f32': collect_f32(predictions, inputs)}
    sources.update(load_fragments(fragment_paths, inputs))
    numbers = merge(sources)
    resolve_derived(numbers)
    doc = {'schema_version': SCHEMA_VERSION, 'inputs': inputs, 'numbers': numbers}
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
