"""
run_grid.py
────────────
Grid de clasificacion visual del Modulo 3 (plan v2, matriz de experimentos
docs/matriz_experimentos.md): 3 tecnicas x 3 arquitecturas x 2 estrategias.

Fuentes unicas de configuracion (src/training/config.py):
  - configs/preprocesamiento.json (F1.4): tamano, interpolacion y volteo por
    tecnica. --img-size/--no-hflip solo se aceptan si coinciden con el JSON.
  - configs/matriz_experimentos.json (F1.3): factores, hiperparametros
    comunes (AdamW, lr, weight_decay, lote, epocas, paciencia), semillas por
    fase y ajustes de determinismo.

Fase (docs/especificacion_predicciones.md §3.4), segun el modo:
  sin banderas                       -> primera_pasada (F3.2): solo val, test no se carga
  --evaluate-test                    -> final (F3.5)
  --evaluate-test --permute-labels S -> control_permutacion (F3.6)
  --evaluate-test --sensitivity      -> sensibilidad (F5.5)
Con --evaluate-test el script se niega si docs/protocolo_preregistrado.md no
esta commiteado sin cambios o si la etiqueta protocolo-v1 no esta en el
historial de HEAD. En primera_pasada y sensibilidad los hiperparametros son
los de la matriz y no se pueden cambiar por linea de comandos.

Seleccion (matriz §11.10; F3.3 H10): final y control_permutacion solo aceptan
los finalistas de select_finalist, y sensibilidad los segundos de
select_second, ambos calculados sobre results/predictions/primera_pasada.
Otra configuracion exige --override-selection "<motivo>", que queda en cada
JSON. Con --smoke no se comprueba.

Insumos: TT_DATA_ROOT (docs/datos.md, src/paths.py). Antes de entrenar se
verifica cada carpeta de imagenes contra meta/data_manifest.csv (SHA-256
agregado) y cada nombre de archivo y etiqueta contra meta/sample_index.csv.

Reanudable: cada corrida se guarda en cuanto termina en
<out-dir>/runs/<tecnica>__<arq>__<estrategia>__<variante>__seed<k>.json y se
salta si ya existe. Si una corrida truena, el traceback queda en
<out-dir>/errors/ y el grid sigue; al volver a correr se reintenta. Un
out-dir es de una sola fase.

Salidas (se regeneran tras cada corrida):
  - <predictions-root>/<fase>/<tecnica>__<arq>__<estrategia>__seed<k>.csv
    (+ .manifest.json): formato F1.5, p_dep sin redondear. Default:
    results/predictions/ del repo. El script se niega a sobrescribir un
    archivo con la misma clave y otra variante de preprocesamiento.
  - <out-dir>/summary.csv            una fila por corrida
  - <out-dir>/summary_by_config.csv  media/DE por configuracion; la columna
    seleccion aplica src/selection/select_finalist.py (informativa: la
    seleccion oficial de F3.3 la corre SV).

--smoke: prueba de humo. Permite cambiar epocas, escribe las predicciones en
<out-dir>/predictions/ (fuera de results/predictions) y lo marca en el
manifiesto.

Uso en Kaggle: ver docs/correr_grid_kaggle.md
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

TRAINING = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TRAINING, '..', '..'))
sys.path.insert(0, TRAINING)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import config  # noqa: E402
from umap_io import sha256  # noqa: E402

PROTOCOL_DEFAULT = os.path.join('docs', 'protocolo_preregistrado.md')
TECH_ORDER = ['gray_scott', 'perlin', 'lorenz']   # orden de ejecucion
TASK_BY_PHASE = {'primera_pasada': 'F3.2', 'final': 'F3.5', 'control_permutacion': 'F3.6',
                 'sensibilidad': 'F5.5'}
FREEZE = {'finetune': False, 'frozen': True}

RUN_FIELDS = ['technique', 'arch', 'strategy', 'variant', 'seed', 'lr', 'weight_decay', 'best_epoch',
              'epochs_run', 'n_trainable_params', 'n_total_params', 'f1_val', 'f1_test', 'acc_test', 'minutes',
              'warning']
CONFIG_FIELDS = ['technique', 'arch', 'strategy', 'variant', 'n_seeds', 'seeds', 'n_trainable_params',
                 'n_total_params', 'f1_val_mean', 'f1_val_std', 'f1_test_mean', 'f1_test_std', 'seleccion']


# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────

def phase_for(evaluate_test, permute_seed, sensitivity):
    if permute_seed is not None and sensitivity:
        raise SystemExit('--permute-labels y --sensitivity son fases distintas; no se combinan')
    if not evaluate_test:
        if permute_seed is not None or sensitivity:
            raise SystemExit('control_permutacion y sensibilidad producen predicciones de test (matriz §4): '
                             'requieren --evaluate-test')
        return 'primera_pasada'
    if permute_seed is not None:
        return 'control_permutacion'
    return 'sensibilidad' if sensitivity else 'final'


def run_variant(prep_variant, permute_seed):
    return prep_variant + (f'-perm{permute_seed}' if permute_seed is not None else '')


def run_name(technique, arch, strategy, variant, seed):
    return f'{technique}__{arch}__{strategy}__{variant}__seed{seed}'


def write_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def git(*args):
    try:
        return subprocess.check_output(['git', '-C', REPO, *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def rel_repo(path):
    return os.path.relpath(path, REPO).replace(os.sep, '/')


def file_ref(path):
    return {'path': rel_repo(path), 'sha256': sha256(path)}


def check_protocol_frozen(protocol_rel, tag):
    """Compuerta de test (matriz §4, §11.5): el protocolo debe estar
    commiteado y sin cambios locales, y la etiqueta `tag` debe existir y ser
    ancestro de HEAD. Devuelve commit y fecha del protocolo y la etiqueta."""
    path = os.path.join(REPO, protocol_rel)
    problems = []
    if not os.path.exists(path):
        problems.append(f'no existe {protocol_rel}')
    elif git('ls-files', '--error-unmatch', protocol_rel) is None:
        problems.append(f'{protocol_rel} no esta commiteado en git')
    elif git('status', '--porcelain', '--', protocol_rel):
        problems.append(f'{protocol_rel} tiene cambios sin commitear')
    tag_commit = git('rev-parse', '-q', '--verify', f'refs/tags/{tag}^{{commit}}') if tag else None
    if tag and not tag_commit:
        problems.append(f'no existe la etiqueta {tag} (git fetch --tags)')
    elif tag and git('merge-base', '--is-ancestor', tag_commit, 'HEAD') is None:
        problems.append(f'la etiqueta {tag} no es ancestro de HEAD')
    if problems:
        raise SystemExit('--evaluate-test rechazado: el protocolo preregistrado no esta congelado '
                         f'({"; ".join(problems)}). Ver docs/matriz_experimentos.md §4.')
    commit, date = git('log', '-1', '--format=%h %cI', '--', protocol_rel).split(' ', 1)
    return {'path': protocol_rel, 'commit': commit, 'date': date, 'tag': tag, 'tag_commit': tag_commit}


def check_cli_preprocessing(prep, img_size, no_hflip):
    """--img-size/--no-hflip no pueden contradecir configs/preprocesamiento.json."""
    problems = []
    for tech, cfg in prep.items():
        if img_size is not None and img_size != cfg['size']:
            problems.append(f'{tech}: --img-size {img_size}, el JSON dice {cfg["size"]}')
        if no_hflip and cfg['hflip']:
            problems.append(f'{tech}: --no-hflip, el JSON permite el volteo')
    if problems:
        raise SystemExit('La linea de comandos contradice configs/preprocesamiento.json (F1.4), que es la '
                         'unica fuente del preprocesamiento:\n  ' + '\n  '.join(problems))


def resolve_hparams(defaults, args, fase):
    """Hiperparametros efectivos. En primera_pasada y sensibilidad son los de
    la matriz (§3, §4); solo --smoke permite acortar epocas/paciencia."""
    hp = {k: (dict(v) if isinstance(v, dict) else v) for k, v in defaults.items()}
    overrides = {'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size,
                 'epochs': args.epochs, 'patience': args.patience}
    given = {k: v for k, v in overrides.items() if v is not None}
    if fase in ('primera_pasada', 'sensibilidad'):
        allowed = {'epochs', 'patience'} if args.smoke else set()
        bad = {k: v for k, v in given.items()
               if k not in allowed and v != (defaults[k] if k != 'lr' else None)}
        if bad:
            raise SystemExit(f'En la fase {fase} los hiperparametros son los de configs/matriz_experimentos.json; '
                             f'no se aceptan {sorted(bad)} por linea de comandos (solo --smoke acorta epocas).')
    for k, v in given.items():
        if k == 'lr':
            hp['lr'] = {'finetune': v, 'frozen': v}
        else:
            hp[k] = v
    if args.num_workers is not None:
        hp['num_workers'] = args.num_workers
    return hp


def check_out_dir_mode(out_dir, fase, smoke):
    meta_path = os.path.join(out_dir, 'grid_meta.json')
    meta = {'fase': fase, 'smoke': smoke}
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            prev = json.load(f)
        if {k: prev.get(k) for k in meta} != meta:
            raise SystemExit(f'{out_dir} es de {prev}; usa otro --out-dir para {meta} (no se mezclan fases).')
    else:
        write_json_atomic(meta_path, meta)


def degenerate_warning(f1_val, preds, split):
    """Marca resultados sospechosos sin descartarlos."""
    if len(set(preds)) == 1:
        return f'predice una sola clase en {split} ({preds[0]})'
    if f1_val >= 0.95:
        return f'F1-val sospechosamente alto ({f1_val:.4f}) -- revisar fuga'
    return ''


def import_prev_results(prev_dirs, runs_dir):
    """Una ruta mal escrita no debe importar 0 corridas en silencio y
    re-entrenar todo: si no hay JSONs en <prev>/runs/, se aborta."""
    n = 0
    for prev in prev_dirs or []:
        found = glob.glob(os.path.join(prev, 'runs', '*.json'))
        if not found:
            why = 'no existe' if not os.path.isdir(prev) else "no contiene runs/*.json"
            raise SystemExit(f'--prev-results: {prev} {why}. Debe ser la carpeta de salida de una '
                             f'corrida anterior (la que contiene summary.csv y runs/). Tip: '
                             f'!find /kaggle/input -name summary_by_config.csv')
        for src in found:
            dst = os.path.join(runs_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                n += 1
    if n:
        print(f'Importadas {n} corridas previas de {prev_dirs}')


# ─────────────────────────────────────────────────────────────
# Insumos: indice maestro y manifiesto de datos
# ─────────────────────────────────────────────────────────────

def load_sample_index(path):
    """{split: [(idx, y_true, {tecnica: nombre de imagen})]} ordenado por idx."""
    from src import paths
    out = {s: [] for s in paths.SPLITS}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['split']].append((int(r['idx']), int(r['y_true']),
                                    {t: r[f'img_{t}'] for t in paths.TECHNIQUES}))
    for s, rows in out.items():
        rows.sort()
        if [r[0] for r in rows] != list(range(paths.EXPECTED_SIZES[s])):
            raise SystemExit(f'{path}: el split {s} no tiene idx 0..{paths.EXPECTED_SIZES[s] - 1}')
    return out


def verify_images(technique, manifest_rows):
    """SHA-256 agregado de la carpeta de imagenes (dir_digest de
    build_manifest.py) contra meta/data_manifest.csv. Devuelve la entrada
    de `inputs` del manifiesto de predicciones."""
    from src import paths
    from src.build_manifest import dir_digest, file_row
    logical, rel = f'IMAGES[{technique}]', paths.REL['IMAGES'][technique]
    root = paths.data_root()
    files, _ = paths.list_files(root / rel)
    rows = [file_row(logical, f'{rel}/{p.name}', root) for p in files]
    digest = dir_digest(rows)
    exp = [r for r in manifest_rows if r['logical_name'] == logical and r['kind'] == 'dir']
    if not exp:
        raise SystemExit(f'{logical} no esta en meta/data_manifest.csv')
    if (digest, len(rows)) != (exp[0]['sha256'], int(exp[0]['n_files'])):
        extra = sorted({f'{rel}/{p.name}' for p in files} -
                       {r['rel_path'] for r in manifest_rows if r['logical_name'] == logical})
        raise SystemExit(f'{logical}: la carpeta de TT_DATA_ROOT no coincide con meta/data_manifest.csv '
                         f'({len(rows)} archivos, sha256 {digest[:12]}...; se esperan {exp[0]["n_files"]}, '
                         f'{exp[0]["sha256"][:12]}...). No registrados: {extra[:5]}. '
                         f'Ver python -m src.build_manifest --verify')
    return {'rel_path': rel, 'sha256': digest, 'n_files': len(rows)}


def preflight(techniques, index, evaluate_test):
    """Verifica TODO antes de entrenar (una ruta mal escrita para Lorenz no
    debe descubrirse despues de horas de Gray-Scott): integridad de cada
    carpeta y nombres/etiquetas contra el indice maestro. Sin --evaluate-test
    solo se listan nombres de test, no se abren. Devuelve inputs y etiquetas."""
    from src import paths
    from src.build_manifest import read_manifest
    from data import list_split_files, PREFIXES
    manifest_rows = read_manifest()
    inputs, expected = {}, {s: [r[1] for r in index[s]] for s in paths.SPLITS}
    problems = []
    for tech in techniques:
        try:
            inputs[f'IMAGES[{tech}]'] = verify_images(tech, manifest_rows)
        except SystemExit as e:
            problems.append(str(e))
            continue
        d = str(paths.images_dir(tech))
        for split in paths.SPLITS:
            try:
                files, _ = list_split_files(d, PREFIXES[tech], split, expected[split])
                names = [os.path.basename(p) for p in files]
                if names != [r[2][tech] for r in index[split]]:
                    raise ValueError('los nombres no coinciden con img_' + tech + ' de meta/sample_index.csv')
                print(f'  {tech:<11}{split:<6}{len(files):>5} imagenes OK')
            except (OSError, ValueError) as e:
                problems.append(f'{tech}/{split}: {e}')
    if problems:
        raise SystemExit('Verificacion previa fallo, no se entreno nada:\n  ' + '\n  '.join(problems))
    inputs['SAMPLE_INDEX'] = file_ref(str(paths.SAMPLE_INDEX))
    return inputs, expected


# ─────────────────────────────────────────────────────────────
# Predicciones (F1.5)
# ─────────────────────────────────────────────────────────────

def predictions_frame(record):
    import pandas as pd
    rows = []
    for split in ('val', 'test'):
        if record.get(f'{split}_probs') is None:
            continue
        for idx, (y, p) in enumerate(zip(record[f'{split}_true'], record[f'{split}_probs'])):
            rows.append({'sample_id': f'{split}_{idx:04d}', 'split': split, 'y_true': int(y),
                         'technique': record['technique'], 'arch': record['arch'],
                         'strategy': record['strategy'], 'seed': int(record['seed']), 'p_dep': float(p)})
    return pd.DataFrame(rows, columns=['sample_id', 'split', 'y_true', 'technique', 'arch', 'strategy',
                                       'seed', 'p_dep'])


def manifest_config(record):
    return {
        'variant': record['prep_variant'], 'preprocessing': record['preprocessing'],
        'preprocessing_file': record['config_files']['preprocesamiento'],
        'matrix_file': record['config_files']['matriz'],
        'optimizer': 'AdamW', 'lr': record['lr'], 'weight_decay': record['weight_decay'],
        'batch_size': record['batch_size'], 'max_epochs': record['n_epochs_max'], 'patience': record['patience'],
        'loss': 'CrossEntropyLoss', 'class_weights': record['class_weights'],
        'pretrained_weights': 'IMAGENET1K_V1', 'determinism': record['determinism'],
        'permute_labels': record['permute_labels'], 'permutation_indices': record['permutation_indices'],
        'smoke': record['smoke'], 'run_json': record['run_id'] + '.json',
    }


def check_no_variant_clash(csv_path, record):
    """Misma clave (fase, tecnica, arq, estrategia, semilla) con otra variante
    de preprocesamiento o de permutacion = otro experimento: no se sobrescribe.
    Devuelve True si el archivo existente ya es el de esta corrida."""
    mpath = csv_path[:-len('.csv')] + '.manifest.json'
    if not os.path.exists(mpath):
        return False
    with open(mpath, encoding='utf-8') as f:
        prev_manifest = json.load(f)
    prev = prev_manifest.get('config', {})
    mine = manifest_config(record)
    clash = {k: (prev.get(k), mine[k]) for k in ('variant', 'permute_labels') if prev.get(k) != mine[k]}
    if clash:
        raise SystemExit(f'{csv_path} ya existe con otra configuracion {clash} (existente, nueva). '
                         f'No se sobrescribe: usa otro --predictions-root o borra el archivo a proposito.')
    return (os.path.exists(csv_path) and prev.get('run_json') == mine['run_json']
            and prev_manifest.get('created_at') == record['created_at']
            and prev_manifest.get('csv_sha256') == sha256(csv_path))


def write_run_predictions(record, pred_root):
    from src.predictions import schema
    from src.predictions.write import write_predictions
    df = predictions_frame(record)
    name = schema.file_name(record['technique'], record['arch'], record['strategy'], record['seed'])
    csv_path = os.path.join(pred_root, record['fase'], name)
    if check_no_variant_clash(csv_path, record):
        return csv_path
    return write_predictions(
        df, fase=record['fase'], task_id=TASK_BY_PHASE[record['fase']], config=manifest_config(record),
        inputs=record['inputs'],
        model_selection={'criterion': 'max F1-Macro val (p_dep > 0.5), mejora estricta, restaura la mejor epoca',
                         'patience': record['patience'], 'best_epoch': record['best_epoch'],
                         'epochs_run': record['epochs_run'], 'best_f1_val': record['best_f1_val']},
        extra={'git_commit': record['git_commit'], 'created_at': record['created_at'],
               'environment': record['environment']},
        out_root=pred_root)


# ─────────────────────────────────────────────────────────────
# Salidas agregadas
# ─────────────────────────────────────────────────────────────

def _r(x, nd=4):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _tech_order(t):
    return TECH_ORDER.index(t) if t in TECH_ORDER else 99


def load_runs(out_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, 'runs', '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        r['run_id'] = os.path.splitext(os.path.basename(path))[0]
        if 'fase' not in r:
            raise SystemExit(f'{path} es de una version anterior de run_grid.py (sin fase/manifiesto F1.5); '
                             f'no se mezcla con esta')
        runs.append(r)
    return runs


def selection_marks(runs, matrix):
    """{(tecnica, arq, estrategia): marca} con la regla oficial
    (select_finalist.py) sobre las tecnicas completas de la primera pasada."""
    import pandas as pd
    from src.selection import select_finalist as sf
    rows = [{'technique': r['technique'], 'arch': r['arch'], 'strategy': r['strategy'], 'seed': r['seed'],
             'f1_val': r['best_f1_val']} for r in runs if r['fase'] == 'primera_pasada' and not r['smoke']]
    marks = {}
    for tech in {r['technique'] for r in rows}:
        df = pd.DataFrame([r for r in rows if r['technique'] == tech])
        try:
            table = sf.select_from_runs(df, matrix)
        except sf.UnresolvedTieError:
            marks.update({(tech, a, s): 'empate_no_resuelto' for a, s in zip(df['arch'], df['strategy'])})
            continue
        except sf.SelectionError:
            continue   # tecnica incompleta
        for t in table.itertuples():
            marks[(t.technique, t.arch, t.strategy)] = ('finalista_sugerido' if t.finalist
                                                        else 'empate_1std' if t.tied else '')
    return marks


def aggregate_by_config(runs, matrix):
    groups = {}
    for r in runs:
        groups.setdefault((r['technique'], r['arch'], r['strategy'], r['variant']), []).append(r)
    marks = selection_marks(runs, matrix)
    rows = []
    for (tech, arch, strat, variant), rs in groups.items():
        vals = np.array([r['best_f1_val'] for r in rs])
        tests = [r['f1_test'] for r in rs if r['f1_test'] is not None]
        rows.append({
            'technique': tech, 'arch': arch, 'strategy': strat, 'variant': variant,
            'n_seeds': len(rs), 'seeds': ' '.join(str(s) for s in sorted(r['seed'] for r in rs)),
            'n_trainable_params': rs[0]['n_trainable_params'], 'n_total_params': rs[0]['n_total_params'],
            'f1_val_mean': vals.mean(), 'f1_val_std': vals.std(ddof=1) if len(vals) > 1 else np.nan,
            'f1_test_mean': np.mean(tests) if tests else None,
            'f1_test_std': np.std(tests, ddof=1) if len(tests) > 1 else None,
            'seleccion': marks.get((tech, arch, strat), ''),
        })
    rows.sort(key=lambda r: (_tech_order(r['technique']), -r['f1_val_mean']))
    return rows


def write_outputs(out_dir, pred_root, matrix):
    runs = load_runs(out_dir)
    for r in runs:
        write_run_predictions(r, pred_root)

    run_rows = [{
        'technique': r['technique'], 'arch': r['arch'], 'strategy': r['strategy'], 'variant': r['variant'],
        'seed': r['seed'], 'lr': r['lr'], 'weight_decay': r['weight_decay'], 'best_epoch': r['best_epoch'],
        'epochs_run': r['epochs_run'], 'n_trainable_params': r['n_trainable_params'],
        'n_total_params': r['n_total_params'], 'f1_val': _r(r['best_f1_val']), 'f1_test': _r(r['f1_test']),
        'acc_test': _r(r['acc_test']), 'minutes': _r(r['seconds'] / 60, 1), 'warning': r['warning']}
        for r in runs]
    run_rows.sort(key=lambda r: (_tech_order(r['technique']), r['arch'], r['strategy'], r['variant'], r['seed']))
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(run_rows)

    config_rows = aggregate_by_config(runs, matrix)
    with open(os.path.join(out_dir, 'summary_by_config.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG_FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows([{k: (_r(v) if k.startswith('f1_') else v) for k, v in r.items()} for r in config_rows])
    return run_rows, config_rows


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_configs(items, matrix):
    f = matrix['factores']
    out = []
    for item in items:
        parts = item.split(':')
        if len(parts) != 3 or parts[0] not in f['technique'] or parts[1] not in f['arch'] \
                or parts[2] not in f['strategy']:
            raise SystemExit(f'--configs {item!r}: se espera tecnica:arquitectura:estrategia con valores de '
                             f'{f["technique"]}, {f["arch"]}, {f["strategy"]}')
        out.append(tuple(parts))
    return out


SELECTION_ROLE = {'final': 'finalist', 'control_permutacion': 'finalist', 'sensibilidad': 'second'}


def expected_selection(fase, matrix, pattern=None):
    """Configuraciones que la regla §5 asigna a la fase: {tecnica: (arq, estrategia)}."""
    from src.selection import select_finalist as sf
    pattern = pattern or os.path.join(REPO, 'results', 'predictions', 'primera_pasada', '*.csv')
    table, runs = sf.select_finalists(pattern, matrix=matrix)
    role = SELECTION_ROLE[fase]
    if role == 'second':
        table = sf.select_second_from_runs(runs, matrix)
    chosen = table[table[role]]
    return {r.technique: (r.arch, r.strategy) for r in chosen.itertuples()}, rel_repo(os.path.dirname(pattern))


def check_selection(fase, triples, override, matrix, smoke=False, pattern=None):
    """F3.3 H10: en final/control_permutacion/sensibilidad cada configuracion
    debe ser la que select_finalist/select_second asigna a su tecnica."""
    if fase not in SELECTION_ROLE or smoke:
        return None
    expected, source = expected_selection(fase, matrix, pattern)
    mismatches = [f'{t}:{a}:{s} (esperado {t}:{":".join(expected[t]) if t in expected else "ninguno"})'
                  for t, a, s in triples if expected.get(t) != (a, s)]
    info = {'role': SELECTION_ROLE[fase], 'source': source,
            'expected': {t: list(v) for t, v in expected.items()},
            'matches': not mismatches, 'override': override}
    if mismatches and not override:
        raise SystemExit(f'Fase {fase}: estas configuraciones no son las de {SELECTION_ROLE[fase]} '
                         f'segun la regla §5: {mismatches}. Si SV las eligio, pasa --override-selection "<motivo>".')
    return info


def main():
    matrix = config.load_matrix()
    f = matrix['factores']
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', required=True, help='runs/, errors/ y resumenes (p. ej. results/f32_primera_pasada)')
    ap.add_argument('--data-root', default=None, help='sustituye a la variable TT_DATA_ROOT (docs/datos.md §2)')
    ap.add_argument('--predictions-root', default=None,
                    help='raiz de las predicciones F1.5. Default: results/predictions del repo '
                         '(<out-dir>/predictions con --smoke)')
    ap.add_argument('--prev-results', nargs='*', default=None,
                    help='out-dirs de sesiones anteriores; sus corridas se copian y se saltan')
    ap.add_argument('--techniques', nargs='+', default=None, choices=f['technique'])
    ap.add_argument('--archs', nargs='+', default=None, choices=f['arch'])
    ap.add_argument('--strategies', nargs='+', default=None, choices=f['strategy'])
    ap.add_argument('--configs', nargs='+', default=None, metavar='TEC:ARQ:EST',
                    help='lista explicita de configuraciones (finalistas, segundo mejor); excluye los filtros')
    ap.add_argument('--override-selection', default=None, metavar='MOTIVO',
                    help='permite en final/control/sensibilidad configuraciones distintas de las de la regla §5')
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help='default: las de la fase en la matriz')
    ap.add_argument('--img-size', type=int, default=None, help='solo se acepta si coincide con el JSON de F1.4')
    ap.add_argument('--no-hflip', action='store_true', help='solo se acepta si el JSON de F1.4 no permite volteo')
    ap.add_argument('--permute-labels', type=int, default=None, metavar='SEED',
                    help='control de permutacion (F3.6): permuta las etiquetas de TRAIN con esta semilla')
    ap.add_argument('--sensitivity', action='store_true', help='fase sensibilidad (F5.5, segundo mejor modelo)')
    ap.add_argument('--evaluate-test', action='store_true',
                    help='evaluar test (F3.5/F3.6/F5.5). Requiere protocolo congelado y etiqueta protocolo-v1')
    ap.add_argument('--protocol', default=PROTOCOL_DEFAULT,
                    help=f'ruta (relativa al repo) del protocolo preregistrado. Default: {PROTOCOL_DEFAULT}')
    ap.add_argument('--lr', type=float, default=None, help='solo fases final/control (hiperparametros de F3.4)')
    ap.add_argument('--weight-decay', type=float, default=None, help='idem --lr')
    ap.add_argument('--batch-size', type=int, default=None, help='idem --lr')
    ap.add_argument('--epochs', type=int, default=None, help='idem --lr; en primera_pasada solo con --smoke')
    ap.add_argument('--patience', type=int, default=None, help='idem --epochs')
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--deterministic-warn-only', action='store_true',
                    help='use_deterministic_algorithms(warn_only=True): solo si SV lo aprueba (matriz §11.6); '
                         'queda registrado en cada manifiesto')
    ap.add_argument('--smoke', action='store_true', help='prueba de humo (ver docstring)')
    ap.add_argument('--save-models', action='store_true',
                    help='guardar state_dict de cada corrida en <out-dir>/models/ (ViT ~330MB c/u)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.data_root:
        os.environ['TT_DATA_ROOT'] = args.data_root
    fase = phase_for(args.evaluate_test, args.permute_labels, args.sensitivity)
    if args.configs:
        if args.techniques or args.archs or args.strategies:
            raise SystemExit('--configs excluye --techniques/--archs/--strategies')
        triples = parse_configs(args.configs, matrix)
    else:
        triples = [(t, a, s) for t in (args.techniques or f['technique']) for a in (args.archs or f['arch'])
                   for s in (args.strategies or f['strategy'])]
    triples.sort(key=lambda c: (_tech_order(c[0]), f['arch'].index(c[1]), f['strategy'].index(c[2])))
    techniques = sorted({c[0] for c in triples}, key=_tech_order)
    selection = check_selection(fase, triples, args.override_selection, matrix, smoke=args.smoke)
    if selection:
        print('Seleccion:', selection)

    prep = config.load_preprocessing(techniques)
    check_cli_preprocessing(prep, args.img_size, args.no_hflip)
    hp = resolve_hparams(config.training_defaults(matrix), args, fase)
    seeds = args.seeds or config.phase_seeds(matrix, fase)
    det = config.determinism(matrix)
    det['warn_only'] = args.deterministic_warn_only
    protocol = check_protocol_frozen(args.protocol, config.phase_requires_tag(matrix, fase)) \
        if args.evaluate_test else None
    pred_root = args.predictions_root or (os.path.join(args.out_dir, 'predictions') if args.smoke
                                          else os.path.join(REPO, 'results', 'predictions'))

    print(f'Fase: {fase}{" (SMOKE)" if args.smoke else ""} | '
          + (f'CON TEST (protocolo {protocol["commit"]}, etiqueta {protocol["tag"]})' if protocol
             else 'SOLO VALIDACION (test no se carga)'))
    print('Preprocesamiento:', {t: prep[t]['variant'] for t in techniques})
    print('Hiperparametros:', hp)
    print('Predicciones ->', pred_root)

    runs_dir = os.path.join(args.out_dir, 'runs')
    errors_dir = os.path.join(args.out_dir, 'errors')
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    check_out_dir_mode(args.out_dir, fase, args.smoke)
    import_prev_results(args.prev_results, runs_dir)

    configs = [(t, a, s, run_variant(prep[t]['variant'], args.permute_labels), seed)
               for t, a, s in triples for seed in seeds]
    pending = [c for c in configs if not os.path.exists(os.path.join(runs_dir, run_name(*c) + '.json'))]
    print(f'{len(configs)} corridas, {len(configs) - len(pending)} ya hechas, {len(pending)} pendientes')
    if args.dry_run:
        for c in pending:
            print('  pendiente:', run_name(*c))
        return

    # Determinismo antes de tocar CUDA (CUBLAS_WORKSPACE_CONFIG)
    import torch
    from sklearn.metrics import accuracy_score
    from data import make_loaders, PREFIXES
    from train_model import configure_determinism, train_model
    from src import paths
    from src.predictions.write import environment as pred_environment, git_commit as pred_git_commit
    det_applied = configure_determinism(det)
    if det['warn_only']:
        torch.use_deterministic_algorithms(det['use_deterministic_algorithms'], warn_only=True)
        det_applied['applied']['warn_only'] = torch.is_deterministic_algorithms_warn_only_enabled()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = pred_environment()
    env.update({'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(), 'device': str(device)})
    try:
        import torchvision
        env['torchvision'] = torchvision.__version__
    except ImportError:
        pass
    commit = pred_git_commit()
    print('Entorno:', env, '| commit', commit)
    print('Determinismo:', det_applied['applied'])
    config_files = {'preprocesamiento': file_ref(config.PREPROCESSING_PATH), 'matriz': file_ref(config.MATRIX_PATH)}

    index = load_sample_index(str(paths.SAMPLE_INDEX))
    inputs, expected = preflight(sorted({c[0] for c in pending}, key=_tech_order), index, args.evaluate_test)

    loaders_cache = {}
    n_errors = 0
    for i, (tech, arch, strategy, variant, seed) in enumerate(pending, 1):
        name = run_name(tech, arch, strategy, variant, seed)
        print(f'\n===== [{i}/{len(pending)}] {name} =====', flush=True)
        t0 = time.time()
        try:
            if tech not in loaders_cache:
                loaders_cache.clear()  # una tecnica a la vez en memoria
                # train_model vuelve a sembrar el generador del train_loader con cada seed
                loaders_cache[tech] = make_loaders(
                    str(paths.images_dir(tech)), tech, seed=seed, batch_size=hp['batch_size'],
                    num_workers=hp['num_workers'], expected_labels=expected, prep=prep[tech],
                    permute_train_seed=args.permute_labels, with_test=args.evaluate_test)
            train_loader, val_loader, test_loader, class_weights = loaders_cache[tech]

            created_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
            model, history, res = train_model(
                config.internal_arch(arch), FREEZE[strategy], train_loader, val_loader, test_loader,
                class_weights, device, lr=hp['lr'][strategy], weight_decay=hp['weight_decay'],
                n_epochs=hp['epochs'], patience=hp['patience'], seed=seed)

            def ints(a):
                return None if a is None else [int(x) for x in a]

            def floats(a):   # sin redondear: float64 de ida y vuelta exacta en JSON
                return None if a is None else [float(x) for x in a]

            has_test = res['test_preds'] is not None
            record = {
                'technique': tech, 'arch': arch, 'strategy': strategy, 'variant': variant,
                'prep_variant': prep[tech]['variant'], 'preprocessing': prep[tech], 'seed': seed,
                'fase': fase, 'smoke': args.smoke, 'freeze_backbone': FREEZE[strategy],
                'lr': res['lr'], 'weight_decay': res['weight_decay'],
                'permute_labels': args.permute_labels,
                'permutation_indices': getattr(train_loader.dataset, 'permutation', None),
                'batch_size': hp['batch_size'], 'n_epochs_max': hp['epochs'], 'patience': hp['patience'],
                'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run'],
                'n_trainable_params': res['n_trainable_params'], 'n_total_params': res['n_total_params'],
                'best_f1_val': float(res['best_f1_val']),
                'evaluate_test': args.evaluate_test, 'protocol': protocol, 'selection': selection,
                'f1_test': float(res['f1_test']) if has_test else None,
                'acc_test': float(accuracy_score(res['test_true'], res['test_preds'])) if has_test else None,
                'class_weights': class_weights.tolist(),
                'n_train': len(train_loader.dataset), 'n_val': len(val_loader.dataset),
                'n_test': len(test_loader.dataset) if has_test else None, 'image_prefix': PREFIXES[tech],
                'history': {k: [float(v) for v in vals] for k, vals in history.items()},
                'val_preds': ints(res['val_preds']), 'val_probs': floats(res['val_probs']),
                'val_true': ints(res['val_true']),
                'test_preds': ints(res['test_preds']), 'test_probs': floats(res['test_probs']),
                'test_true': ints(res['test_true']),
                'inputs': {k: v for k, v in inputs.items() if k in (f'IMAGES[{tech}]', 'SAMPLE_INDEX')},
                'config_files': config_files, 'determinism': det_applied,
                'git_commit': commit, 'created_at': created_at, 'environment': env,
                'seconds': time.time() - t0,
            }
            record['warning'] = degenerate_warning(
                record['best_f1_val'], record['test_preds'] if has_test else record['val_preds'],
                'test' if has_test else 'val')
            if args.save_models:
                os.makedirs(os.path.join(args.out_dir, 'models'), exist_ok=True)
                mpath = os.path.join(args.out_dir, 'models', name + '.pt')
                torch.save(model.state_dict(), mpath)
                record['model_file'] = {'path': os.path.basename(mpath), 'sha256': sha256(mpath)}
            write_json_atomic(os.path.join(runs_dir, name + '.json'), record)
            err_path = os.path.join(errors_dir, name + '.txt')
            if os.path.exists(err_path):
                os.remove(err_path)
            if record['warning']:
                print(f'!!! AVISO {name}: {record["warning"]}')
            del model
        except Exception:
            n_errors += 1
            tb = traceback.format_exc()
            print(f'!!! ERROR en {name}:\n{tb}', flush=True)
            with open(os.path.join(errors_dir, name + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write(tb)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        run_rows, _ = write_outputs(args.out_dir, pred_root, matrix)
        print(f'Salidas actualizadas ({len(run_rows)} corridas, {(time.time() - t0) / 60:.1f} min esta)')

    _, config_rows = write_outputs(args.out_dir, pred_root, matrix)
    print('\n===== RESUMEN POR CONFIGURACION (F1-val media +- std sobre semillas) =====')
    for r in config_rows:
        std = '' if np.isnan(r['f1_val_std']) else f'+-{r["f1_val_std"]:.4f}'
        test = '' if r['f1_test_mean'] is None else f'  test={r["f1_test_mean"]:.4f}'
        print(f'{r["technique"]:<11}{r["arch"]:<17}{r["strategy"]:<9}{r["variant"]:<20}'
              f'n={r["n_seeds"]}  val={r["f1_val_mean"]:.4f}{std:<9}{test}  {r["seleccion"]}')
    if n_errors:
        print(f'\n{n_errors} corridas con error -- ver {errors_dir}. Volver a correr el script las reintenta.')
        sys.exit(1)


if __name__ == '__main__':
    main()
