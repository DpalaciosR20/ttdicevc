"""
check_determinism.py
─────────────────────
Prueba de humo de determinismo (matriz de experimentos §3 y §11.6): entrena
dos veces la misma configuracion, con la misma semilla y los ajustes de
determinismo de configs/matriz_experimentos.json, y compara los pesos finales
tensor por tensor y las p_dep de validacion.

Criterio de la matriz: dos corridas de vit_b16 finetune (semilla 42, 2
epocas) con pesos finales identicos. Si no lo son, se registra y se mantiene
la limitacion del hallazgo 8. Las demas configuraciones sirven para detectar
antes del grid si algun kernel no tiene version determinista (con
use_deterministic_algorithms(True) PyTorch lanza RuntimeError en ese caso).

Solo usa train y val. Escribe un JSON con el resultado por configuracion.

Uso (desde la raiz del repo, con TT_DATA_ROOT definida):
    python -m src.training.check_determinism --technique gray_scott \\
        --configs vit_b16:finetune resnet18:finetune efficientnet_b0:finetune \\
        --out results/f32_primera_pasada/determinismo.json
"""

import argparse
import json
import os
import sys
import time
import traceback

TRAINING = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TRAINING, '..', '..'))
sys.path.insert(0, TRAINING)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import config  # noqa: E402


def compare_states(a, b):
    """(identicos, n_tensores, n_distintos, max |diferencia|)."""
    import torch
    diff, max_abs = 0, 0.0
    for k in a:
        if not torch.equal(a[k], b[k]):
            diff += 1
            if a[k].is_floating_point():
                max_abs = max(max_abs, (a[k] - b[k]).abs().max().item())
    return diff == 0, len(a), diff, max_abs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--technique', default='gray_scott')
    ap.add_argument('--configs', nargs='+', default=['vit_b16:finetune'], metavar='ARQ:EST')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--warn-only', action='store_true', help='use_deterministic_algorithms(warn_only=True)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    matrix = config.load_matrix()
    det = config.determinism(matrix)
    hp = config.training_defaults(matrix)
    prep = config.load_preprocessing([args.technique])[args.technique]

    import torch
    from train_model import configure_determinism, train_model
    from data import make_loaders
    from src import paths
    applied = configure_determinism(det)
    if args.warn_only:
        torch.use_deterministic_algorithms(det['use_deterministic_algorithms'], warn_only=True)
        applied['applied']['warn_only'] = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Determinismo:', applied['applied'], '| device', device)

    loaders = make_loaders(str(paths.images_dir(args.technique)), args.technique, seed=args.seed,
                           batch_size=hp['batch_size'], num_workers=hp['num_workers'], prep=prep,
                           with_test=False)
    train_loader, val_loader, _, class_weights = loaders
    results = []
    for item in args.configs:
        arch, strategy = item.split(':')
        entry = {'technique': args.technique, 'arch': arch, 'strategy': strategy, 'seed': args.seed,
                 'epochs': args.epochs}
        try:
            states, probs, t0 = [], [], time.time()
            for rep in range(2):
                # patience > epochs: las dos corridas recorren todas las epocas
                model, _, res = train_model(
                    config.internal_arch(arch), strategy == 'frozen', train_loader, val_loader, None,
                    class_weights, device, lr=hp['lr'][strategy], weight_decay=hp['weight_decay'],
                    n_epochs=args.epochs, patience=args.epochs + 1, seed=args.seed)
                states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                probs.append(res['val_probs'])
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            same, n, n_diff, max_abs = compare_states(*states)
            entry.update({'identical_weights': same, 'n_tensors': n, 'n_tensors_different': n_diff,
                          'max_abs_weight_diff': max_abs,
                          'identical_val_probs': bool((probs[0] == probs[1]).all()),
                          'max_abs_p_dep_diff': float(abs(probs[0] - probs[1]).max()),
                          'minutes': round((time.time() - t0) / 60, 2), 'error': None})
        except Exception as e:
            entry.update({'identical_weights': None, 'error': f'{type(e).__name__}: {e}',
                          'traceback': traceback.format_exc()})
        print(json.dumps({k: v for k, v in entry.items() if k != 'traceback'}, ensure_ascii=False), flush=True)
        results.append(entry)

    from src.predictions.write import environment, git_commit
    env = environment()
    env.update({'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(), 'device': str(device)})
    out = {'git_commit': git_commit(), 'determinism': applied, 'environment': env, 'results': results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('->', args.out)
    bad = [r for r in results if r['identical_weights'] is not True]
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
