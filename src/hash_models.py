"""
hash_models.py
──────────────
Escribe `<dir>/models_sha256.csv` con la ruta, los bytes y el SHA-256 de cada
`<dir>/<técnica>/models/*.pt`, con rutas relativas a la raíz del repo. Es la
entrada de `meta/artefactos_manifest.csv` (matriz §9). Falla si no encuentra
el número esperado de modelos.

Uso:
    python -m src.hash_models results/f36_permutacion --expected 15
"""
import argparse
import csv
import glob
import os
import sys

from src.verify_artefacts import sha256

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_hashes(results_dir, expected=None):
    models = sorted(glob.glob(os.path.join(results_dir, '*', 'models', '*.pt')))
    if expected is not None and len(models) != expected:
        raise SystemExit(f'{len(models)} modelos en {results_dir}; se esperaban {expected}')
    out = os.path.join(results_dir, 'models_sha256.csv')
    root = os.path.abspath(os.path.join(results_dir, os.pardir, os.pardir))   # raíz con results/
    with open(out, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['path', 'bytes', 'sha256'])
        for p in models:
            w.writerow([os.path.relpath(p, root).replace(os.sep, '/'), os.path.getsize(p), sha256(p)])
    return out, len(models)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results_dir', help='p. ej. results/f36_permutacion')
    ap.add_argument('--expected', type=int, default=None)
    a = ap.parse_args(argv)
    out, n = write_hashes(a.results_dir, a.expected)
    print(open(out).read(), end='')
    print(f'{n} modelos -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
