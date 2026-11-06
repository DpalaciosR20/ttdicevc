"""
run_hpo.py
───────────
Ajuste de hiperparametros de un finalista (F3.4, docs/matriz_experimentos.md
§6). Todo sale de configs/matriz_experimentos.json (bloque `hpo`):

  - Optuna, TPE con sampler_seed y parametros por defecto, sin pruner;
  - n_trials por finalista; el trial 0 es la configuracion comun (§3);
  - objetivo: media del F1-Macro de validacion sobre las semillas del bloque;
  - espacio por estrategia (lr logaritmico, weight_decay uniforme, lote
    categorico, con override por arquitectura);
  - regla final: se adopta el mejor trial salvo que su media no supere a la
    del trial 0 en mas de una DE (ddof=1) del trial 0.

Solo validacion: test nunca se construye. Las predicciones de validacion
quedan en los JSON de cada corrida, no en results/predictions/ (matriz §4).

Una configuracion (tecnica:arquitectura:estrategia) por out-dir. Debe ser el
finalista sugerido por select_finalist sobre results/predictions/primera_pasada;
otra configuracion exige --override-selection "<motivo>", que queda registrado.

Reanudable sin perder la reproducibilidad del muestreo: cada entrenamiento se
guarda en <out-dir>/runs/trial<NN>__seed<k>.json. Al volver a correr, el
estudio se reconstruye desde cero con el mismo sampler; el objetivo lee de
disco las corridas que ya existen (TPE con semilla propone lo mismo ante los
mismos valores) y solo entrena lo que falta. Si los parametros propuestos no
coinciden con los guardados, el script se detiene.

Salidas:
  - <out-dir>/hpo_meta.json        configuracion, espacio y seleccion
  - <out-dir>/trials.csv           un renglon por trial (params, F1 por semilla, media, DE)
  - <out-dir>/study.pkl            estudio de Optuna serializado
  - <out-dir>/hpo_result.json      regla final, hiperparametros adoptados y
                                   banderas para run_grid (F3.5/F3.6)

--smoke: prueba de humo; permite --n-trials, --seeds y --epochs.

Uso en Kaggle: ver docs/correr_hpo_kaggle.md
"""

import argparse
import csv
import glob
import json
import os
import pickle
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
import run_grid  # noqa: E402

FASE = 'hpo'
TASK = 'F3.4'
FINAL_RULE = 'best_trial_unless_gain_le_trial0_std'
TRIAL_FIELDS = ['trial', 'lr', 'weight_decay', 'batch_size', 'f1_val_seeds', 'f1_val_mean', 'f1_val_std',
                'origen']


class ReplayMismatch(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────
# Configuracion de la busqueda (matriz §6)
# ─────────────────────────────────────────────────────────────

def hpo_settings(matrix):
    h = matrix['hpo']
    if (h['algorithm'], h['sampler_params'], h['pruner'], h['trial_0'], h['final_rule']) != \
            ('optuna_tpe', 'default', None, 'configuracion_comun', FINAL_RULE):
        raise config.ConfigError(f'bloque hpo no implementado: {h}')
    obj = h['objective']
    if (obj['metric'], obj['split'], obj['aggregate']) != ('f1_macro', 'val', 'mean'):
        raise config.ConfigError(f'objetivo de HPO no implementado: {obj}')
    return {'sampler_seed': int(h['sampler_seed']), 'n_trials': int(h['n_trials']),
            'seeds': [int(s) for s in obj['seeds']], 'space': h['space']}


def search_space(space, arch, strategy):
    s = space[strategy]
    batch = s.get('batch_size_override', {}).get(arch, s['batch_size'])
    return {'lr': dict(s['lr']), 'weight_decay': dict(s['weight_decay']), 'batch_size': [int(b) for b in batch]}


def trial0_params(defaults, strategy, space):
    p = {'lr': defaults['lr'][strategy], 'weight_decay': defaults['weight_decay'],
         'batch_size': defaults['batch_size']}
    for k in ('lr', 'weight_decay'):
        if not space[k]['low'] <= p[k] <= space[k]['high']:
            raise config.ConfigError(f'trial 0: {k}={p[k]} fuera del espacio {space[k]}')
    if p['batch_size'] not in space['batch_size']:
        raise config.ConfigError(f'trial 0: lote {p["batch_size"]} fuera de {space["batch_size"]}')
    return p


def suggest(trial, space):
    return {
        'lr': trial.suggest_float('lr', space['lr']['low'], space['lr']['high'], log=space['lr']['log']),
        'weight_decay': trial.suggest_float('weight_decay', space['weight_decay']['low'],
                                            space['weight_decay']['high'], log=space['weight_decay']['log']),
        'batch_size': trial.suggest_categorical('batch_size', space['batch_size']),
    }


def final_rule(trials):
    """trials: lista de {'trial', 'f1_val_mean', 'f1_val_std', 'params'}.
    Mejor trial = mayor media (el de menor numero ante empate). Se adopta si
    su media supera a la del trial 0 en MAS de una DE del trial 0."""
    t0 = next(t for t in trials if t['trial'] == 0)
    best = max(trials, key=lambda t: (t['f1_val_mean'], -t['trial']))
    gain = best['f1_val_mean'] - t0['f1_val_mean']
    adopt = best['trial'] != 0 and gain > t0['f1_val_std']
    chosen = best if adopt else t0
    return {'rule': FINAL_RULE, 'trial0': t0['trial'], 'trial0_mean': t0['f1_val_mean'],
            'trial0_std': t0['f1_val_std'], 'best_trial': best['trial'], 'best_mean': best['f1_val_mean'],
            'gain': gain, 'adopted_trial': chosen['trial'], 'adopted_best': adopt,
            'params': dict(chosen['params'])}


def trial0_vs_first_pass(technique, arch, strategy, variant, f1_by_seed, f32_dir=None):
    """El trial 0 repite la configuracion de F3.2: su F1-val por semilla
    deberia coincidir con results/f32_primera_pasada/runs. Es un chequeo de
    cordura (otro entorno de Kaggle puede mover algun kernel), no un criterio."""
    f32_dir = f32_dir or os.path.join(REPO, 'results', 'f32_primera_pasada', 'runs')
    out = {}
    for seed, f1 in f1_by_seed.items():
        path = os.path.join(f32_dir, run_grid.run_name(technique, arch, strategy, variant, seed) + '.json')
        if not os.path.exists(path):
            out[str(seed)] = None
            continue
        with open(path, encoding='utf-8') as f:
            ref = json.load(f)['best_f1_val']
        out[str(seed)] = {'f32': ref, 'hpo_trial0': f1, 'identical': ref == f1}
    return out


def run_grid_flags(params):
    return f'--lr {params["lr"]!r} --weight-decay {params["weight_decay"]!r} --batch-size {params["batch_size"]}'


# ─────────────────────────────────────────────────────────────
# Seleccion: la configuracion debe ser el finalista sugerido
# ─────────────────────────────────────────────────────────────

def check_finalist(technique, arch, strategy, override, matrix, pattern=None):
    from src.selection import select_finalist as sf
    pattern = pattern or os.path.join(REPO, 'results', 'predictions', 'primera_pasada', '*.csv')
    table, _ = sf.select_finalists(pattern, matrix=matrix)
    fin = table[(table.technique == technique) & table.finalist]
    suggested = (fin.arch.item(), fin.strategy.item()) if len(fin) == 1 else None
    info = {'source': run_grid.rel_repo(os.path.dirname(pattern)), 'suggested': suggested,
            'is_suggested_finalist': suggested == (arch, strategy), 'override': override}
    if not info['is_suggested_finalist'] and not override:
        raise SystemExit(f'{technique}:{arch}:{strategy} no es el finalista sugerido de {technique} '
                         f'({suggested}). Si SV lo eligio en F3.3, pasa --override-selection "<motivo>".')
    return info


# ─────────────────────────────────────────────────────────────
# Corridas
# ─────────────────────────────────────────────────────────────

def run_path(runs_dir, trial, seed):
    return os.path.join(runs_dir, f'trial{trial:02d}__seed{seed}.json')


def load_cached(path, params):
    with open(path, encoding='utf-8') as f:
        r = json.load(f)
    saved = {k: r[k] for k in params}
    if saved != params:
        raise ReplayMismatch(f'{os.path.basename(path)}: guardado {saved}, el estudio propone {params}. '
                             f'El muestreo no se reprodujo; no se mezclan estudios.')
    return r


def write_trials(out_dir, rows):
    with open(os.path.join(out_dir, 'trials.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({'trial': r['trial'], **r['params'],
                        'f1_val_seeds': ' '.join(f'{s}:{v!r}' for s, v in r['f1_by_seed'].items()),
                        'f1_val_mean': repr(r['f1_val_mean']), 'f1_val_std': repr(r['f1_val_std']),
                        'origen': r['origen']})


def main():
    matrix = config.load_matrix()
    settings = hpo_settings(matrix)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, metavar='TEC:ARQ:EST', help='el finalista de F3.3')
    ap.add_argument('--out-dir', required=True, help='p. ej. results/f34_hpo/gray_scott')
    ap.add_argument('--data-root', default=None, help='sustituye a TT_DATA_ROOT (docs/datos.md §2)')
    ap.add_argument('--override-selection', default=None, metavar='MOTIVO',
                    help='permite una configuracion que no es el finalista sugerido (decision de SV en F3.3)')
    ap.add_argument('--prev-results', nargs='*', default=None,
                    help='out-dirs de sesiones anteriores del MISMO estudio; sus corridas se copian')
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--smoke', action='store_true', help='prueba de humo: permite --n-trials, --seeds y --epochs')
    ap.add_argument('--n-trials', type=int, default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not args.smoke and (args.n_trials is not None or args.seeds or args.epochs is not None):
        raise SystemExit('--n-trials, --seeds y --epochs solo se aceptan con --smoke: el presupuesto, las '
                         'semillas y las epocas de F3.4 son los de configs/matriz_experimentos.json')
    if args.data_root:
        os.environ['TT_DATA_ROOT'] = args.data_root
    ((technique, arch, strategy),) = run_grid.parse_configs([args.config], matrix)
    prep = config.load_preprocessing([technique])[technique]
    defaults = config.training_defaults(matrix)
    space = search_space(settings['space'], arch, strategy)
    params0 = trial0_params(defaults, strategy, space)
    n_trials = args.n_trials or settings['n_trials']
    seeds = args.seeds or settings['seeds']
    epochs = args.epochs or defaults['epochs']
    num_workers = defaults['num_workers'] if args.num_workers is None else args.num_workers
    selection = check_finalist(technique, arch, strategy, args.override_selection, matrix)

    print(f'F3.4 HPO {"(SMOKE) " if args.smoke else ""}| {technique}:{arch}:{strategy} | '
          f'SOLO VALIDACION (test no se carga) | variante {prep["variant"]}')
    print(f'{n_trials} trials x semillas {seeds} = {n_trials * len(seeds)} entrenamientos | '
          f'epocas {epochs}, paciencia {defaults["patience"]}')
    print('Espacio:', space, '| trial 0:', params0)
    print('Seleccion:', selection)

    runs_dir = os.path.join(args.out_dir, 'runs')
    errors_dir = os.path.join(args.out_dir, 'errors')
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    run_grid.check_out_dir_mode(args.out_dir, FASE, args.smoke)
    meta = {'task': TASK, 'fase': FASE, 'smoke': args.smoke, 'technique': technique, 'arch': arch,
            'strategy': strategy, 'variant': prep['variant'], 'preprocessing': prep,
            'sampler': {'name': 'TPESampler', 'seed': settings['sampler_seed'], 'params': 'default'},
            'pruner': None, 'n_trials': n_trials, 'seeds': seeds, 'space': space, 'trial0': params0,
            'fixed': {'optimizer': 'AdamW', 'max_epochs': epochs, 'patience': defaults['patience']},
            'selection': selection}
    meta_path = os.path.join(args.out_dir, 'hpo_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            prev = json.load(f)
        keys = ('technique', 'arch', 'strategy', 'variant', 'sampler', 'n_trials', 'seeds', 'space', 'trial0',
                'fixed')
        diff = [k for k in keys if prev.get(k) != json.loads(json.dumps(meta[k]))]
        if diff:
            raise SystemExit(f'{args.out_dir} es de otro estudio (difiere {diff}); usa otro --out-dir')
    run_grid.write_json_atomic(meta_path, meta)
    if args.prev_results:
        run_grid.import_prev_results(args.prev_results, runs_dir)
    n_done = len(glob.glob(os.path.join(runs_dir, 'trial*__seed*.json')))
    print(f'{n_done} de {n_trials * len(seeds)} entrenamientos ya hechos')
    if args.dry_run:
        return

    import optuna
    import torch
    from data import make_loaders
    from train_model import configure_determinism, train_model
    from src import paths
    from src.predictions.write import environment as pred_environment, git_commit as pred_git_commit

    det = config.determinism(matrix)
    det['warn_only'] = False
    det_applied = configure_determinism(det)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = pred_environment()
    env.update({'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(), 'device': str(device),
                'optuna': optuna.__version__})
    commit = pred_git_commit()
    print('Entorno:', env, '| commit', commit)
    config_files = {'preprocesamiento': run_grid.file_ref(config.PREPROCESSING_PATH),
                    'matriz': run_grid.file_ref(config.MATRIX_PATH)}
    index = run_grid.load_sample_index(str(paths.SAMPLE_INDEX))
    inputs, expected = run_grid.preflight([technique], index, False)

    loaders = {}   # por lote; train_model vuelve a sembrar el generador con cada semilla

    def get_loaders(batch_size):
        if batch_size not in loaders:
            loaders.clear()
            loaders[batch_size] = make_loaders(
                str(paths.images_dir(technique)), technique, seed=seeds[0], batch_size=batch_size,
                num_workers=num_workers, expected_labels=expected, prep=prep, with_test=False)
        return loaders[batch_size]

    rows = []
    n_trained = [0]

    def objective(trial):
        params = suggest(trial, space)   # el trial 0 devuelve los valores encolados
        if trial.number == 0 and params != params0:
            raise ReplayMismatch(f'trial 0 no es la configuracion comun: {params} != {params0}')
        f1s, origen = {}, set()
        for seed in seeds:
            path = run_path(runs_dir, trial.number, seed)
            if os.path.exists(path):
                f1s[seed] = load_cached(path, params)['best_f1_val']
                origen.add('disco')
                continue
            name = os.path.basename(path)[:-5]
            print(f'\n===== trial {trial.number}/{n_trials - 1} semilla {seed} {params} =====', flush=True)
            t0 = time.time()
            try:
                train_loader, val_loader, _, class_weights = get_loaders(params['batch_size'])
                model, history, res = train_model(
                    config.internal_arch(arch), run_grid.FREEZE[strategy], train_loader, val_loader, None,
                    class_weights, device, lr=params['lr'], weight_decay=params['weight_decay'],
                    n_epochs=epochs, patience=defaults['patience'], seed=seed)
            except Exception:
                with open(os.path.join(errors_dir, name + '.txt'), 'w', encoding='utf-8') as fh:
                    fh.write(traceback.format_exc())
                raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            record = {
                'task': TASK, 'fase': FASE, 'smoke': args.smoke, 'trial': trial.number, 'seed': seed,
                'technique': technique, 'arch': arch, 'strategy': strategy, 'variant': prep['variant'],
                'preprocessing': prep, **params, 'n_epochs_max': epochs, 'patience': defaults['patience'],
                'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run'],
                'n_trainable_params': res['n_trainable_params'], 'n_total_params': res['n_total_params'],
                'best_f1_val': float(res['best_f1_val']), 'class_weights': class_weights.tolist(),
                'history': {k: [float(v) for v in vals] for k, vals in history.items()},
                'val_preds': [int(x) for x in res['val_preds']], 'val_probs': [float(x) for x in res['val_probs']],
                'val_true': [int(x) for x in res['val_true']], 'warning': run_grid.degenerate_warning(
                    float(res['best_f1_val']), list(res['val_preds']), 'val'),
                'inputs': inputs, 'config_files': config_files, 'determinism': det_applied,
                'git_commit': commit, 'environment': env,
                'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'seconds': time.time() - t0,
            }
            run_grid.write_json_atomic(path, record)
            err = os.path.join(errors_dir, name + '.txt')
            if os.path.exists(err):
                os.remove(err)
            f1s[seed] = record['best_f1_val']
            origen.add('entrenado')
            n_trained[0] += 1
            del model
        vals = np.array([f1s[s] for s in seeds])
        row = {'trial': trial.number, 'params': params, 'f1_by_seed': f1s, 'f1_val_mean': float(vals.mean()),
               'f1_val_std': float(vals.std(ddof=1)) if len(vals) > 1 else float('nan'),
               'origen': '+'.join(sorted(origen))}
        rows.append(row)
        write_trials(args.out_dir, rows)
        trial.set_user_attr('f1_val_by_seed', {str(k): v for k, v in f1s.items()})
        trial.set_user_attr('f1_val_std', row['f1_val_std'])
        return row['f1_val_mean']

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=settings['sampler_seed']),
                                pruner=optuna.pruners.NopPruner(), study_name=f'f34__{technique}__{arch}__{strategy}')
    study.enqueue_trial(params0)
    study.optimize(objective, n_trials=n_trials, catch=())
    with open(os.path.join(args.out_dir, 'study.pkl'), 'wb') as f:
        pickle.dump(study, f)

    decision = final_rule(rows)
    check0 = trial0_vs_first_pass(technique, arch, strategy, prep['variant'],
                                  next(r for r in rows if r['trial'] == 0)['f1_by_seed']) \
        if epochs == defaults['epochs'] else None
    result = {
        'task': TASK, 'smoke': args.smoke, 'technique': technique, 'arch': arch, 'strategy': strategy,
        'variant': prep['variant'], 'selection': selection, 'decision': decision,
        'trial0_vs_primera_pasada': check0,
        'run_grid_flags': run_grid_flags(decision['params']), 'n_trials': len(rows),
        'n_trained_this_session': n_trained[0], 'git_commit': commit, 'environment': env,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    run_grid.write_json_atomic(os.path.join(args.out_dir, 'hpo_result.json'), result)

    print('\n===== TRIALS (F1-val media +- DE sobre semillas) =====')
    for r in sorted(rows, key=lambda r: -r['f1_val_mean']):
        p = r['params']
        print(f'trial {r["trial"]:>2}  lr={p["lr"]:.3g}  wd={p["weight_decay"]:.3g}  lote={p["batch_size"]:<3} '
              f'{r["f1_val_mean"]:.4f} +- {r["f1_val_std"]:.4f}')
    print(f'\nMejor trial {decision["best_trial"]} ({decision["best_mean"]:.4f}); trial 0 '
          f'{decision["trial0_mean"]:.4f} +- {decision["trial0_std"]:.4f}; ganancia {decision["gain"]:+.4f} -> '
          f'{"se adopta el mejor" if decision["adopted_best"] else "se conserva el trial 0"}')
    if check0:
        print('Trial 0 frente a F3.2 (F1-val por semilla):',
              {s: (c and ('identico' if c['identical'] else f'{c["hpo_trial0"]:.4f} vs {c["f32"]:.4f}'))
               for s, c in check0.items()})
    print('Para F3.5/F3.6 (run_grid):', result['run_grid_flags'])


if __name__ == '__main__':
    main()
