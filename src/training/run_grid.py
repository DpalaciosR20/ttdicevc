"""
run_grid.py
────────────
Grid de clasificacion visual del plan v2 (plan/plan_tt2_v2.md):
3 tecnicas x 3 arquitecturas x 2 estrategias = 18 configuraciones.

Por defecto corre la PRIMERA PASADA (F3.2): 3 semillas (42, 7, 123) e
inferencia SOLO sobre validacion -- el conjunto de prueba no se carga.
Evaluar test (F3.5, finalistas) requiere --evaluate-test, y el script se
niega si docs/protocolo_preregistrado.md no existe commiteado y sin cambios
locales (compuerta de la Fase 1). El commit y la fecha del protocolo quedan
registrados en cada corrida.

Reanudable: cada corrida se guarda en cuanto termina, en un JSON propio
(<out-dir>/runs/<tecnica>__<arq>__<estrategia>__<variante>__seed<k>.json), y
se salta si ese JSON ya existe. La variante codifica el preprocesamiento y el
control de permutacion (p. ej. px224-flip, px256-noflip-perm99). Si una
corrida truena, el traceback queda en <out-dir>/errors/ y el grid sigue; al
volver a correr, los errores se reintentan. Un out-dir es de un solo modo
(solo-val o con test): no se mezclan.

Salidas (se regeneran tras cada corrida):
  - summary.csv            una fila por corrida
  - summary_by_config.csv  media/std por configuracion sobre semillas y, por
                           tecnica, el finalista sugerido (F3.3: mejor F1-val
                           promedio; empate dentro de 1 std -> menos parametros
                           entrenables). La seleccion final la confirma SV.
  - predictions.csv        predicciones por muestra (F1.5):
                           sample_id, split, y_true, technique, arch, strategy,
                           seed, p_dep, run_id   (p_dep = P(label=1), depresivo;
                           sample_id = "<split>_<idx:04d>"; run_id identifica
                           variante de preprocesamiento/permutacion)

Uso en Kaggle: ver docs/correr_grid_kaggle.md
"""

import argparse
import csv
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import IMG_SIZE, PREFIXES, list_split_files, make_loaders  # noqa: E402
from train_model import train_model  # noqa: E402
from umap_io import load_umap_arrays, sha256  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PROTOCOL_DEFAULT = os.path.join('docs', 'protocolo_preregistrado.md')

TECHNIQUES = ['gray_scott', 'perlin', 'lorenz']   # Gray-Scott primero: senal ya validada
ARCHS = ['resnet18', 'efficientnet_b0', 'vit_b_16']
STRATEGIES = {'finetune': False, 'frozen': True}  # nombre -> freeze_backbone
FIRST_PASS_SEEDS = [42, 7, 123]                   # plan v2, F3.2

RUN_FIELDS = ['technique', 'arch', 'strategy', 'variant', 'seed', 'lr', 'best_epoch', 'epochs_run',
              'n_trainable_params', 'f1_val', 'f1_test', 'acc_test', 'minutes', 'warning']
CONFIG_FIELDS = ['technique', 'arch', 'strategy', 'variant', 'n_seeds', 'seeds', 'n_trainable_params',
                 'f1_val_mean', 'f1_val_std', 'f1_test_mean', 'f1_test_std', 'seleccion']
PRED_FIELDS = ['sample_id', 'split', 'y_true', 'technique', 'arch', 'strategy', 'seed', 'p_dep', 'run_id']


# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────

def variant_name(img_size, hflip, permute_seed):
    v = f'px{img_size}-{"flip" if hflip else "noflip"}'
    return v + (f'-perm{permute_seed}' if permute_seed is not None else '')


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


def environment_info(device):
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'cudnn': torch.backends.cudnn.version(), 'device': str(device),
            'git_commit': git('rev-parse', '--short', 'HEAD')}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


def check_protocol_frozen(protocol_rel):
    """Compuerta de la Fase 1: el protocolo debe existir commiteado y sin
    cambios locales antes de consultar test. Devuelve commit y fecha."""
    path = os.path.join(REPO, protocol_rel)
    problems = []
    if not os.path.exists(path):
        problems.append(f'no existe {protocol_rel}')
    elif git('ls-files', '--error-unmatch', protocol_rel) is None:
        problems.append(f'{protocol_rel} no esta commiteado en git')
    elif git('status', '--porcelain', '--', protocol_rel):
        problems.append(f'{protocol_rel} tiene cambios sin commitear')
    if problems:
        raise SystemExit('--evaluate-test rechazado: el protocolo preregistrado no esta congelado '
                         f'({"; ".join(problems)}). Ver plan/plan_tt2_v2.md, seccion 3 y Fase 1.')
    info = git('log', '-1', '--format=%h %cI', '--', protocol_rel)
    commit, date = info.split(' ', 1)
    return {'path': protocol_rel, 'commit': commit, 'date': date}


def load_expected_labels(umap_params_path):
    """Labels de umap_params.pkl (o del .npz exportado), para verificar que el
    idx de cada imagen corresponde a la misma muestra (necesario para pruebas
    pareadas). No requiere umap-learn ni numba (ver umap_io.py, F0.2)."""
    arrays = load_umap_arrays(umap_params_path)
    return {s: [int(l) for l in arrays[s]['labels']] for s in ('train', 'val', 'test')}


def preflight(data_root, techniques, expected, archs, img_size):
    """Verifica TODO antes de entrenar: una ruta mal escrita para Lorenz no
    debe descubrirse despues de horas de Gray-Scott. Solo lee nombres de
    archivo (no consulta test)."""
    problems = []
    if 'vit_b_16' in archs and img_size != 224:
        problems.append(f'vit_b_16 preentrenado de torchvision solo acepta 224 px (pedido: {img_size}); '
                        f'correr ViT por separado con --img-size 224')
    for tech in techniques:
        d = os.path.join(data_root, tech)
        for split in ('train', 'val', 'test'):
            try:
                paths, _ = list_split_files(d, PREFIXES[tech], split,
                                            expected[split] if expected else None)
                print(f'  {tech:<11}{split:<6}{len(paths):>5} imagenes OK')
            except (OSError, ValueError) as e:
                problems.append(f'{tech}/{split}: {e}')
    if problems:
        raise SystemExit('Verificacion previa fallo, no se entreno nada:\n  ' + '\n  '.join(problems))


def check_out_dir_mode(out_dir, evaluate_test):
    meta_path = os.path.join(out_dir, 'grid_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            prev = json.load(f)['evaluate_test']
        if prev != evaluate_test:
            raise SystemExit(f'{out_dir} es de modo evaluate_test={prev}; usa otro --out-dir para '
                             f'evaluate_test={evaluate_test} (no se mezclan corridas con y sin test).')
    else:
        write_json_atomic(meta_path, {'evaluate_test': evaluate_test})


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
# Salidas agregadas
# ─────────────────────────────────────────────────────────────

def _r(x, nd=4):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _tech_order(t):
    return TECHNIQUES.index(t) if t in TECHNIQUES else 99


def load_runs(out_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, 'runs', '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        r['run_id'] = os.path.splitext(os.path.basename(path))[0]
        runs.append(r)
    return runs


def aggregate_by_config(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r['technique'], r['arch'], r['strategy'], r['variant']), []).append(r)
    rows = []
    for (tech, arch, strat, variant), rs in groups.items():
        vals = np.array([r['best_f1_val'] for r in rs])
        tests = [r['f1_test'] for r in rs if r['f1_test'] is not None]
        rows.append({
            'technique': tech, 'arch': arch, 'strategy': strat, 'variant': variant,
            'n_seeds': len(rs), 'seeds': ' '.join(str(s) for s in sorted(r['seed'] for r in rs)),
            'n_trainable_params': rs[0]['n_trainable_params'],
            'f1_val_mean': vals.mean(), 'f1_val_std': vals.std(ddof=1) if len(vals) > 1 else np.nan,
            'f1_test_mean': np.mean(tests) if tests else None,
            'f1_test_std': np.std(tests, ddof=1) if len(tests) > 1 else None,
            'seleccion': '',
        })
    # F3.3: por tecnica, entre configuraciones reales (no controles de permutacion)
    # con el maximo numero de semillas: mejor F1-val promedio; empate dentro de
    # 1 std del mejor -> la de menos parametros entrenables.
    for tech in {r['technique'] for r in rows}:
        cands = [r for r in rows if r['technique'] == tech and '-perm' not in r['variant']]
        if not cands:
            continue
        max_seeds = max(r['n_seeds'] for r in cands)
        cands = [r for r in cands if r['n_seeds'] == max_seeds]
        best = max(cands, key=lambda r: r['f1_val_mean'])
        tol = 0.0 if np.isnan(best['f1_val_std']) else best['f1_val_std']
        tied = [r for r in cands if r['f1_val_mean'] >= best['f1_val_mean'] - tol]
        chosen = min(tied, key=lambda r: (r['n_trainable_params'], -r['f1_val_mean']))
        for r in tied:
            r['seleccion'] = 'empate_1std'
        chosen['seleccion'] = 'finalista_sugerido'
    rows.sort(key=lambda r: (_tech_order(r['technique']), -r['f1_val_mean']))
    return rows


def write_outputs(out_dir):
    runs = load_runs(out_dir)

    run_rows = [{
        'technique': r['technique'], 'arch': r['arch'], 'strategy': r['strategy'], 'variant': r['variant'],
        'seed': r['seed'], 'lr': r['lr'], 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
        'n_trainable_params': r['n_trainable_params'], 'f1_val': _r(r['best_f1_val']),
        'f1_test': _r(r['f1_test']), 'acc_test': _r(r['acc_test']), 'minutes': _r(r['seconds'] / 60, 1),
        'warning': r['warning']} for r in runs]
    run_rows.sort(key=lambda r: (_tech_order(r['technique']), r['arch'], r['strategy'], r['variant'], r['seed']))
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        w.writerows(run_rows)

    config_rows = aggregate_by_config(runs)
    with open(os.path.join(out_dir, 'summary_by_config.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG_FIELDS)
        w.writeheader()
        w.writerows([{k: (_r(v) if k.startswith('f1_') else v) for k, v in r.items()} for r in config_rows])

    with open(os.path.join(out_dir, 'predictions.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        w.writeheader()
        for r in runs:
            for split in ('val', 'test'):
                if r.get(f'{split}_probs') is None:
                    continue
                for idx, (y, p) in enumerate(zip(r[f'{split}_true'], r[f'{split}_probs'])):
                    w.writerow({'sample_id': f'{split}_{idx:04d}', 'split': split, 'y_true': y,
                                'technique': r['technique'], 'arch': r['arch'], 'strategy': r['strategy'],
                                'seed': r['seed'], 'p_dep': p, 'run_id': r['run_id']})
    return run_rows, config_rows


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True,
                    help='carpeta con subcarpetas gray_scott/, perlin/, lorenz/ (class-blind)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--umap-params', default=None,
                    help='umap_params.pkl para verificar que los labels coinciden por idx (recomendado)')
    ap.add_argument('--prev-results', nargs='*', default=None,
                    help='out-dirs de sesiones anteriores; sus corridas se copian y se saltan')
    ap.add_argument('--techniques', nargs='+', default=TECHNIQUES, choices=TECHNIQUES)
    ap.add_argument('--archs', nargs='+', default=ARCHS, choices=ARCHS)
    ap.add_argument('--strategies', nargs='+', default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument('--seeds', nargs='+', type=int, default=FIRST_PASS_SEEDS)
    ap.add_argument('--img-size', type=int, default=IMG_SIZE,
                    help='resolucion de entrada (224 = camera-ready; 256 = sin redimensionar). F1.4')
    ap.add_argument('--no-hflip', action='store_true', help='desactiva RandomHorizontalFlip en train. F1.4')
    ap.add_argument('--permute-labels', type=int, default=None, metavar='SEED',
                    help='control de permutacion (F3.6): permuta las etiquetas de TRAIN con esta semilla')
    ap.add_argument('--evaluate-test', action='store_true',
                    help='evaluar test (solo finalistas, F3.5). Requiere el protocolo congelado')
    ap.add_argument('--protocol', default=PROTOCOL_DEFAULT,
                    help=f'ruta (relativa al repo) del protocolo preregistrado. Default: {PROTOCOL_DEFAULT}')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--save-models', action='store_true',
                    help='guardar state_dict de cada corrida en <out-dir>/models/ (ViT ~330MB c/u)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    hflip = not args.no_hflip
    variant = variant_name(args.img_size, hflip, args.permute_labels)
    protocol = check_protocol_frozen(args.protocol) if args.evaluate_test else None
    mode = f'CON TEST (protocolo {protocol["commit"]}, {protocol["date"]})' if protocol \
        else 'SOLO VALIDACION (test no se carga)'
    print(f'Modo: {mode} | variante: {variant}')

    runs_dir = os.path.join(args.out_dir, 'runs')
    errors_dir = os.path.join(args.out_dir, 'errors')
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    check_out_dir_mode(args.out_dir, args.evaluate_test)
    import_prev_results(args.prev_results, runs_dir)

    configs = [(t, a, s, variant, seed) for t in args.techniques for a in args.archs
               for s in args.strategies for seed in args.seeds]
    pending = [c for c in configs if not os.path.exists(os.path.join(runs_dir, run_name(*c) + '.json'))]
    print(f'{len(configs)} corridas, {len(configs) - len(pending)} ya hechas, {len(pending)} pendientes')
    if args.dry_run:
        for c in pending:
            print('  pendiente:', run_name(*c))
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = environment_info(device)
    print('Entorno:', env)
    expected = load_expected_labels(args.umap_params) if args.umap_params else None
    umap_ref = {'path': args.umap_params, 'sha256': sha256(args.umap_params)} if args.umap_params else None
    if expected is None:
        print('AVISO: sin --umap-params no se verifica que el idx de cada imagen coincida con umap_params.pkl')
    preflight(args.data_root, sorted({c[0] for c in pending}, key=_tech_order), expected,
              {c[1] for c in pending}, args.img_size)

    loaders_cache = {}
    n_errors = 0
    for i, (tech, arch, strategy, _, seed) in enumerate(pending, 1):
        name = run_name(tech, arch, strategy, variant, seed)
        print(f'\n===== [{i}/{len(pending)}] {name} =====', flush=True)
        t0 = time.time()
        try:
            if tech not in loaders_cache:
                loaders_cache.clear()  # una tecnica a la vez en memoria
                # train_model vuelve a sembrar el generador del train_loader con cada seed
                loaders_cache[tech] = make_loaders(
                    os.path.join(args.data_root, tech), tech, seed=seed, batch_size=args.batch_size,
                    num_workers=args.num_workers, expected_labels=expected, img_size=args.img_size,
                    hflip=hflip, permute_train_seed=args.permute_labels, with_test=args.evaluate_test)
            train_loader, val_loader, test_loader, class_weights = loaders_cache[tech]

            model, history, res = train_model(
                arch, STRATEGIES[strategy], train_loader, val_loader, test_loader, class_weights,
                device, n_epochs=args.epochs, patience=args.patience, seed=seed)

            def tolist(a, nd=None):
                if a is None:
                    return None
                return [round(float(x), nd) for x in a] if nd else [int(x) for x in a]

            has_test = res['test_preds'] is not None
            record = {
                'technique': tech, 'arch': arch, 'strategy': strategy, 'variant': variant, 'seed': seed,
                'freeze_backbone': STRATEGIES[strategy], 'lr': res['lr'],
                'img_size': args.img_size, 'hflip': hflip, 'permute_labels': args.permute_labels,
                'batch_size': args.batch_size, 'n_epochs_max': args.epochs, 'patience': args.patience,
                'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run'],
                'n_trainable_params': res['n_trainable_params'],
                'best_f1_val': float(res['best_f1_val']),
                'evaluate_test': args.evaluate_test, 'protocol': protocol,
                'labels_verified_against': umap_ref,
                'f1_test': float(res['f1_test']) if has_test else None,
                'acc_test': float(accuracy_score(res['test_true'], res['test_preds'])) if has_test else None,
                'class_weights': class_weights.tolist(),
                'n_train': len(train_loader.dataset), 'n_val': len(val_loader.dataset),
                'n_test': len(test_loader.dataset) if has_test else None, 'image_prefix': PREFIXES[tech],
                'history': {k: [float(v) for v in vals] for k, vals in history.items()},
                'val_preds': tolist(res['val_preds']), 'val_probs': tolist(res['val_probs'], 6),
                'val_true': tolist(res['val_true']),
                'test_preds': tolist(res['test_preds']), 'test_probs': tolist(res['test_probs'], 6),
                'test_true': tolist(res['test_true']),
                'seconds': time.time() - t0, 'env': env,
            }
            record['warning'] = degenerate_warning(
                record['best_f1_val'], record['test_preds'] if has_test else record['val_preds'],
                'test' if has_test else 'val')
            if args.save_models:
                os.makedirs(os.path.join(args.out_dir, 'models'), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'models', name + '.pt'))
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
            with open(os.path.join(errors_dir, name + '.txt'), 'w', encoding='utf-8') as f:
                f.write(tb)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        run_rows, _ = write_outputs(args.out_dir)
        print(f'Salidas actualizadas ({len(run_rows)} corridas, {(time.time() - t0) / 60:.1f} min esta)')

    _, config_rows = write_outputs(args.out_dir)
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
