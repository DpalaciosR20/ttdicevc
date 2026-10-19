"""
run_grid.py
────────────
Grid de TT2 (Semanas 2-3, CE): 3 tecnicas x 3 arquitecturas x 2 estrategias
= 18 configuraciones, por defecto con una sola semilla (42). El mismo script
sirve despues para el multi-seed de los ganadores (--seeds 42 7 123 2024 99).

Reanudable: cada corrida se guarda en cuanto termina, en un JSON propio
(<out-dir>/runs/<tecnica>__<arq>__<estrategia>__seed<k>.json), y se salta si
ese JSON ya existe. Si una corrida truena, el traceback queda en
<out-dir>/errors/ y el grid sigue con la siguiente -- no se descarta en
silencio. Al volver a correr, los errores se reintentan.

Tras cada corrida se regenera <out-dir>/summary.csv (una fila por corrida).
La columna 'mejor_val' marca, por tecnica, la configuracion con mayor F1-VAL
(nunca test) entre las ya terminadas.

Cada JSON guarda tambien val_preds, test_preds, test_probs y test_true (en el
orden de idx del split) para McNemar contra BETO / baselines 5D-UMAP.

Uso en Kaggle:
    !python src/training/run_grid.py \
        --data-root /kaggle/input/datasets/palaciosroblesdiego/data-files/TT/images_grayscale_classblind \
        --umap-params /kaggle/input/.../umap_params.pkl \
        --out-dir /kaggle/working/grid_seed42

    # Para continuar en una sesion nueva, a partir del output de una version previa:
    #   --prev-results /kaggle/input/<output-de-la-version-anterior>/grid_seed42

    # Solo ver que se correria:
    #   --dry-run
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
from data import PREFIXES, list_split_files, make_loaders  # noqa: E402
from train_model import train_model  # noqa: E402

TECHNIQUES = ['gray_scott', 'perlin', 'lorenz']   # Gray-Scott primero: senal ya validada
ARCHS = ['resnet18', 'efficientnet_b0', 'vit_b_16']
STRATEGIES = {'finetune': False, 'frozen': True}  # nombre -> freeze_backbone

SUMMARY_FIELDS = ['technique', 'arch', 'strategy', 'seed', 'lr', 'best_epoch', 'epochs_run',
                  'f1_val', 'f1_test', 'acc_test', 'minutes', 'warning', 'mejor_val']


def run_name(technique, arch, strategy, seed):
    return f'{technique}__{arch}__{strategy}__seed{seed}'


def write_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def git_commit():
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    try:
        return subprocess.check_output(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def environment_info(device):
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'cudnn': torch.backends.cudnn.version(), 'device': str(device), 'git_commit': git_commit()}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


def load_expected_labels(umap_params_path):
    """Labels de umap_params.pkl, para verificar que el idx de cada imagen
    corresponde a la misma muestra (necesario para parear McNemar)."""
    import joblib
    try:
        up = joblib.load(umap_params_path)
    except ModuleNotFoundError as e:
        raise SystemExit(f'No se pudo cargar {umap_params_path} ({e}). El reductor UMAP requiere '
                         f'umap-learn: !pip install umap-learn')
    return {s: [int(l) for l in up[s]['labels']] for s in ('train', 'val', 'test')}


def preflight(data_root, techniques, expected):
    """Verifica TODAS las carpetas antes de entrenar nada: una ruta mal
    escrita para Lorenz no debe descubrirse despues de horas de Gray-Scott."""
    problems = []
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


def degenerate_warning(f1_val, test_preds):
    """Marca resultados sospechosos (plan Semanas 2-3, punto 5) sin descartarlos."""
    if len(set(test_preds)) == 1:
        return f'predice una sola clase en test ({test_preds[0]})'
    if f1_val >= 0.95:
        return f'F1-val sospechosamente alto ({f1_val:.4f}) -- revisar fuga'
    return ''


def write_summary(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, 'runs', '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        rows.append({'technique': r['technique'], 'arch': r['arch'], 'strategy': r['strategy'],
                     'seed': r['seed'], 'lr': r['lr'], 'best_epoch': r['best_epoch'],
                     'epochs_run': r['epochs_run'], 'f1_val': round(r['best_f1_val'], 4),
                     'f1_test': round(r['f1_test'], 4), 'acc_test': round(r['acc_test'], 4),
                     'minutes': round(r['seconds'] / 60, 1), 'warning': r['warning'], 'mejor_val': ''})
    for tech in {r['technique'] for r in rows}:
        tech_rows = [r for r in rows if r['technique'] == tech]
        max(tech_rows, key=lambda r: r['f1_val'])['mejor_val'] = '*'
    rows.sort(key=lambda r: (TECHNIQUES.index(r['technique']) if r['technique'] in TECHNIQUES else 99,
                             r['arch'], r['strategy'], r['seed']))
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def import_prev_results(prev_dirs, runs_dir):
    n = 0
    for prev in prev_dirs or []:
        for src in glob.glob(os.path.join(prev, 'runs', '*.json')):
            dst = os.path.join(runs_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                n += 1
    if n:
        print(f'Importadas {n} corridas previas de {prev_dirs}')


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
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--save-models', action='store_true',
                    help='guardar state_dict de cada corrida en <out-dir>/models/ (ViT ~330MB c/u)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    runs_dir = os.path.join(args.out_dir, 'runs')
    errors_dir = os.path.join(args.out_dir, 'errors')
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    import_prev_results(args.prev_results, runs_dir)

    configs = [(t, a, s, seed) for t in args.techniques for a in args.archs
               for s in args.strategies for seed in args.seeds]
    pending = [c for c in configs if not os.path.exists(os.path.join(runs_dir, run_name(*c) + '.json'))]
    print(f'{len(configs)} configuraciones, {len(configs) - len(pending)} ya hechas, {len(pending)} pendientes')
    if args.dry_run:
        for c in pending:
            print('  pendiente:', run_name(*c))
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = environment_info(device)
    print('Entorno:', env)
    expected = load_expected_labels(args.umap_params) if args.umap_params else None
    if expected is None:
        print('AVISO: sin --umap-params no se verifica que el idx de cada imagen coincida con umap_params.pkl')

    preflight(args.data_root, sorted({c[0] for c in pending}, key=TECHNIQUES.index), expected)

    loaders_cache = {}
    n_errors = 0
    for i, (tech, arch, strategy, seed) in enumerate(pending, 1):
        name = run_name(tech, arch, strategy, seed)
        print(f'\n===== [{i}/{len(pending)}] {name} =====', flush=True)
        t0 = time.time()
        try:
            if tech not in loaders_cache:
                loaders_cache.clear()  # una tecnica a la vez en memoria
                # train_model vuelve a sembrar el generador del train_loader con cada seed
                loaders_cache[tech] = make_loaders(
                    os.path.join(args.data_root, tech), tech, seed=seed, batch_size=args.batch_size,
                    num_workers=args.num_workers, expected_labels=expected)
            train_loader, val_loader, test_loader, class_weights = loaders_cache[tech]

            model, history, res = train_model(
                arch, STRATEGIES[strategy], train_loader, val_loader, test_loader, class_weights,
                device, n_epochs=args.epochs, patience=args.patience, seed=seed)

            test_preds = res['test_preds'].tolist()
            record = {
                'technique': tech, 'arch': arch, 'strategy': strategy, 'seed': seed,
                'freeze_backbone': STRATEGIES[strategy], 'lr': res['lr'],
                'batch_size': args.batch_size, 'n_epochs_max': args.epochs, 'patience': args.patience,
                'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run'],
                'best_f1_val': float(res['best_f1_val']), 'f1_test': float(res['f1_test']),
                'acc_test': float(accuracy_score(res['test_true'], res['test_preds'])),
                'class_weights': class_weights.tolist(),
                'n_train': len(train_loader.dataset), 'n_val': len(val_loader.dataset),
                'n_test': len(test_loader.dataset), 'image_prefix': PREFIXES[tech],
                'history': {k: [float(v) for v in vals] for k, vals in history.items()},
                'val_preds': res['val_preds'].tolist(), 'test_preds': test_preds,
                'test_probs': [round(float(p), 6) for p in res['test_probs']],
                'test_true': res['test_true'].tolist(),
                'warning': degenerate_warning(res['best_f1_val'], test_preds),
                'seconds': time.time() - t0, 'env': env,
            }
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

        rows = write_summary(args.out_dir)
        print(f'summary.csv actualizado ({len(rows)} corridas, {(time.time() - t0) / 60:.1f} min esta)')

    rows = write_summary(args.out_dir)
    print('\n===== RESUMEN =====')
    print(f'{"tecnica":<11}{"arq":<17}{"estrategia":<10}{"seed":>5}{"F1-val":>8}{"F1-test":>8}  ')
    for r in rows:
        print(f'{r["technique"]:<11}{r["arch"]:<17}{r["strategy"]:<10}{r["seed"]:>5}'
              f'{r["f1_val"]:>8.4f}{r["f1_test"]:>8.4f}  {r["mejor_val"]} {r["warning"]}')
    if n_errors:
        print(f'\n{n_errors} corridas con error -- ver {errors_dir}. Volver a correr el script las reintenta.')
        sys.exit(1)


if __name__ == '__main__':
    main()
