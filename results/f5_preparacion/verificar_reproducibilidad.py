"""
verificar_reproducibilidad.py — Bloque A (SV), punto 2. Solo validación.

  2a. F3.5, semillas 42, 7, 123: val_probs (JSON) y p_dep de val (CSV) iguales
      bit a bit a la primera pasada (Gray-Scott, Lorenz) y al trial 10 de F3.4
      (Perlin); también la mejor época. Extra: F5.5 (sensibilidad) frente a la
      primera pasada, que su bitácora afirma.
  2b. F3.4 H5: re-ejecuta los comandos git de results/f34_hpo/git_seleccion.log.

Uso: python results/f5_preparacion/verificar_reproducibilidad.py
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.predictions.validate import validate_predictions, load_index  # noqa: E402
from src import paths  # noqa: E402

index = load_index(paths.SAMPLE_INDEX)
fails = []


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def val_pdep_csv(path):
    df = validate_predictions(path, index=index)
    return df[df.split == 'val'].sort_values('sample_id').p_dep.to_numpy()


def one(pattern):
    hits = sorted(glob.glob(str(REPO / pattern)))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def bits(a):
    return np.asarray(a, dtype=np.float64).tobytes()


print('== 2a. Reproducibilidad en validación (bit a bit)')
FINAL = {'gray_scott': ('efficientnet_b0', 'finetune'), 'lorenz': ('vit_b16', 'frozen'), 'perlin': ('vit_b16', 'frozen')}
SECOND = {'gray_scott': ('resnet18', 'finetune'), 'lorenz': ('efficientnet_b0', 'finetune'), 'perlin': ('efficientnet_b0', 'frozen')}
for label, rdir, pdir, cfgs in (('F3.5', 'f35_final', 'final', FINAL), ('F5.5', 'f55_sensibilidad', 'sensibilidad', SECOND)):
    for tech, (arch, strat) in cfgs.items():
        for seed in (42, 7, 123):
            new = load_json(one(f'results/{rdir}/{tech}/runs/{tech}__{arch}__{strat}__*__seed{seed}.json'))
            new_csv = val_pdep_csv(one(f'results/predictions/{pdir}/{tech}__{arch}__{strat}__seed{seed}.csv'))
            if label == 'F3.5' and tech == 'perlin':
                ref = load_json(REPO / f'results/f34_hpo/perlin/runs/trial10__seed{seed}.json')
                ref_name = f'F3.4 trial 10 seed {seed}'
                ref_probs, ref_epoch = ref['val_probs'], ref['best_epoch']
                ref_csv = None
            else:
                ref_csvp = one(f'results/predictions/primera_pasada/{tech}__{arch}__{strat}__seed{seed}.csv')
                ref_csv = val_pdep_csv(ref_csvp)
                ref_json = one(f'results/f32_primera_pasada/runs/{tech}__{arch}__{strat}__*__seed{seed}.json') \
                    if glob.glob(str(REPO / f'results/f32_primera_pasada/runs/{tech}__{arch}__{strat}__*__seed{seed}.json')) else None
                ref = load_json(ref_json) if ref_json else None
                ref_name = f'primera_pasada seed {seed}'
                ref_probs = ref['val_probs'] if ref else ref_csv
                ref_epoch = ref['best_epoch'] if ref else None
            same_json = bits(new['val_probs']) == bits(ref_probs)
            same_csv = ref_csv is None or bits(new_csv) == bits(ref_csv)
            same_epoch = ref_epoch is None or new['best_epoch'] == ref_epoch
            same_f1 = ref is None or new['best_f1_val'] == ref['best_f1_val']
            ok = same_json and same_csv and same_epoch and same_f1
            print(f'  {"OK   " if ok else "FALLA"} {label} {tech:<10} {arch}/{strat} s{seed:<4} vs {ref_name}: '
                  f'val_probs={same_json} p_dep_csv={same_csv} best_epoch={new["best_epoch"]}/{ref_epoch} '
                  f'best_f1_val={same_f1} (max|Δ|={np.max(np.abs(np.array(new["val_probs"]) - np.array(ref_probs))):.1e})')
            if not ok:
                fails.append(f'{label} {tech} s{seed}')

print('\n== 2b. F3.4 H5: comandos de results/f34_hpo/git_seleccion.log')
CMDS = [
    ['rev-parse', 'seleccion-f33^{commit}'],
    ['log', '-1', '--format=%H %cI', 'seleccion-f33'],
    ['merge-base', '--is-ancestor', 'seleccion-f33', 'b848d2c'],
    ['merge-base', '--is-ancestor', 'seleccion-f33', 'a92e4b6'],
    ['merge-base', '--is-ancestor', 'protocolo-v1', 'b848d2c'],
    ['diff', '--stat', 'b848d2c', 'seleccion-f33', '--', 'src/training', 'src/generation', 'src/selection', 'configs'],
    ['diff', '--stat', 'b848d2c', 'a92e4b6', '--', 'src/training/train_model.py', 'src/training/data.py',
     'src/training/build_model.py', 'src/training/config.py', 'src/training/run_hpo.py', 'src/generation',
     'src/selection', 'configs'],
    ['for-each-ref', 'refs/tags/seleccion-f33', '--format=%(objecttype) %(taggerdate:iso8601)'],
]
for c in CMDS:
    r = subprocess.run(['git', *c], cwd=REPO, capture_output=True, text=True)
    out = r.stdout.strip() or '(sin salida)'
    print(f'$ git {" ".join(c)}   [rc={r.returncode}]\n{out}\n')
print('Nota: el log original usa HEAD = a92e4b6 (rama ce/f3.4-hpo); aquí se usa a92e4b6 explícito.')
rs = [load_json(p)['created_at'] for p in glob.glob(str(REPO / 'results/f34_hpo/*/runs/*.json'))]
commits = sorted({load_json(p)['git_commit'] for p in glob.glob(str(REPO / 'results/f34_hpo/*/runs/*.json'))})
print(f'F3.4 runs: {len(rs)} JSON; created_at [{min(rs)}, {max(rs)}]; git_commit {commits}')
gpu = (REPO / 'results/f34_hpo/entorno/gpu.txt').read_text(encoding='utf-8', errors='replace').splitlines()[:3]
print('entorno/gpu.txt (primeras líneas):', gpu)

print(f'\nFALLAS: {len(fails)}', fails)
sys.exit(1 if fails else 0)
