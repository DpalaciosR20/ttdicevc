"""
build_manifest.py
─────────────────
Construye meta/data_manifest.csv (docs/datos.md, regla 6): ruta relativa a
TT_DATA_ROOT, tamaño y SHA-256 de cada insumo de datos.md §3.

Columnas: logical_name, rel_path, kind, size_bytes, sha256, n_files
  - kind='file': un archivo. Las carpetas (checkpoint e imágenes) llevan una
    fila por archivo que contienen.
  - kind='dir': una fila por carpeta; sha256 = SHA-256 de las líneas
    "rel_path\\tsha256\\n" de sus archivos, ordenadas por rel_path
    (dir_digest). Es el valor que va en `inputs` del manifiesto de
    predicciones (especificacion_predicciones.md §4).

Fuera del manifiesto: gray_scott_nested_cv/ (datos.md §7),
processed/*_indices.csv (no figuran en §3) y los archivos de sistema de
paths.SYSTEM_FILES (p. ej. el desktop.ini que crea Drive para escritorio).

Uso (desde la raíz del repo):
    python -m src.build_manifest            # escribe meta/data_manifest.csv
    python -m src.build_manifest --verify   # compara TT_DATA_ROOT contra el manifiesto
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from src import paths
from src.training.umap_io import sha256

FIELDS = ['logical_name', 'rel_path', 'kind', 'size_bytes', 'sha256', 'n_files']


def inputs():
    """(logical_name, rel_path, kind) de cada insumo de datos.md §3."""
    items = [(f'CSV[{s}]', paths.REL['CSV'][s], 'file') for s in paths.SPLITS]
    items += [('UMAP_PARAMS', paths.REL['UMAP_PARAMS'], 'file'),
              ('EMB_41', paths.REL['EMB_41'], 'file'),
              ('CKPT_41', paths.REL['CKPT_41'], 'dir')]
    items += [(f'IMAGES[{t}]', paths.REL['IMAGES'][t], 'dir') for t in paths.TECHNIQUES]
    return items


def dir_digest(file_rows):
    lines = ''.join(f"{r['rel_path']}\t{r['sha256']}\n" for r in sorted(file_rows, key=lambda r: r['rel_path']))
    return hashlib.sha256(lines.encode('utf-8')).hexdigest()


def file_row(logical, rel, root):
    p = root / rel
    return {'logical_name': logical, 'rel_path': rel, 'kind': 'file',
            'size_bytes': p.stat().st_size, 'sha256': sha256(p), 'n_files': 1}


def scan(log=print):
    root = paths.data_root()
    rows = []
    for logical, rel, kind in inputs():
        if kind == 'file':
            rows.append(file_row(logical, rel, root))
            log(f'{logical:<20} {rel}  {rows[-1]["sha256"]}')
            continue
        files, ignored = paths.list_files(root / rel)
        if ignored:
            log(f'{logical:<20} ignorados (archivos de sistema): {", ".join(p.name for p in ignored)}')
        children = [file_row(logical, f'{rel}/{p.name}', root) for p in files]
        rows += children
        rows.append({'logical_name': logical, 'rel_path': rel, 'kind': 'dir',
                     'size_bytes': sum(r['size_bytes'] for r in children),
                     'sha256': dir_digest(children), 'n_files': len(children)})
        log(f'{logical:<20} {rel}/  {len(children)} archivos  {rows[-1]["sha256"]}')
    return rows


def read_manifest(path=paths.DATA_MANIFEST):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def verify_manifest(logical_name=None, path=paths.DATA_MANIFEST):
    """Regla 6 de datos.md: recalcula el SHA-256 de los insumos (todos o uno)
    y devuelve la lista de discrepancias (vacía si todo coincide)."""
    expected = [r for r in read_manifest(path) if logical_name in (None, r['logical_name'])]
    if not expected:
        return [f'{logical_name}: no está en el manifiesto']
    root = paths.data_root()
    problems = []
    for logical in dict.fromkeys(r['logical_name'] for r in expected):
        exp = [r for r in expected if r['logical_name'] == logical]
        exp_dir = [r for r in exp if r['kind'] == 'dir']
        exp_files = {r['rel_path']: r for r in exp if r['kind'] == 'file'}
        if exp_dir:
            present = {f"{exp_dir[0]['rel_path']}/{p.name}"
                       for p in paths.list_files(root / exp_dir[0]['rel_path'])[0]}
            for rel in sorted(present - set(exp_files)):
                problems.append(f'{logical}: archivo no registrado {rel}')
        for rel, r in exp_files.items():
            p = root / rel
            if not p.is_file():
                problems.append(f'{logical}: falta {rel}')
            elif sha256(p) != r['sha256']:
                problems.append(f'{logical}: SHA-256 distinto en {rel}')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(paths.DATA_MANIFEST))
    ap.add_argument('--verify', action='store_true', help='verificar en lugar de escribir')
    args = ap.parse_args()

    if args.verify:
        problems = verify_manifest(path=args.out)
        print('\n'.join(problems) if problems else 'manifiesto OK')
        return 1 if problems else 0

    out = Path(args.out)
    paths.assert_outside_data_root(out)
    print(f'TT_DATA_ROOT = {paths.data_root()}')
    rows = scan()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} filas -> {out.relative_to(paths.REPO)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
