"""
verify_artefacts.py
───────────────────
Verifica una copia publicada de los modelos (dataset privado `tt2-artefactos`,
matriz §9) contra `meta/artefactos_manifest.csv`.

Busca bajo --root los archivos con las extensiones del manifiesto (`.pt` de los
modelos, `.npz` de los mapas de F3.7), los empareja por nombre de archivo con el
manifiesto y compara bytes y SHA-256. Falla (código 1) si falta algún modelo
del manifiesto, si alguno no coincide o si hay nombres duplicados. Los archivos
que no están en el manifiesto se reportan como extra, sin fallar.

Uso (Kaggle, con el dataset como input):
    python -m src.verify_artefacts --root /kaggle/input \
        --out results/f35_final/artefactos_verificacion.json
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def find_models(root, exts=('.pt',)):
    found = {}
    dup = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(tuple(exts)):
                p = os.path.join(dirpath, name)
                if name in found:
                    dup.append(name)
                found.setdefault(name, p)
    return found, sorted(set(dup))


def verify(root, manifest, task=None):
    rows = list(csv.DictReader(open(manifest, encoding='utf-8')))
    if task:
        rows = [r for r in rows if r['task'] == task]
    expected = {os.path.basename(r['rel_path']): r for r in rows}
    found, dup = find_models(root, sorted({os.path.splitext(n)[1] for n in expected}) or ['.pt'])
    files = []
    for name, r in sorted(expected.items()):
        p = found.get(name)
        if p is None:
            files.append({'file': name, 'status': 'falta'})
            continue
        size, digest = os.path.getsize(p), sha256(p)
        ok = size == int(r['size_bytes']) and digest == r['sha256']
        files.append({'file': name, 'status': 'ok' if ok else 'distinto', 'found_at': os.path.relpath(p, root),
                      'bytes': size, 'sha256': digest})
    extra = sorted(set(found) - set(expected))
    n_ok = sum(f['status'] == 'ok' for f in files)
    return {
        'manifest': os.path.relpath(manifest, REPO).replace(os.sep, '/'),
        'task': task,
        'root': root,
        'dataset_dirs': sorted({os.path.relpath(os.path.dirname(found[f['file']]), root).split(os.sep)[0]
                                for f in files if f['file'] in found}),
        'n_expected': len(expected), 'n_ok': n_ok,
        'missing': [f['file'] for f in files if f['status'] == 'falta'],
        'mismatch': [f['file'] for f in files if f['status'] == 'distinto'],
        'duplicates': dup, 'extra': extra,
        'passed': n_ok == len(expected) and not dup,
        'checked_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'files': files,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='carpeta donde buscar los archivos (p. ej. /kaggle/input)')
    ap.add_argument('--manifest', default=os.path.join(REPO, 'meta', 'artefactos_manifest.csv'))
    ap.add_argument('--task', default=None, help='verificar solo las filas de esta tarea (p. ej. F3.5)')
    ap.add_argument('--out', default=None, help='JSON con el reporte')
    a = ap.parse_args(argv)
    rep = verify(a.root, a.manifest, a.task)
    for f in rep['files']:
        print(f"{f['status']:9s} {f['file']}")
    for k in ('missing', 'mismatch', 'duplicates', 'extra'):
        if rep[k]:
            print(f'{k}: {rep[k]}')
    print(f"\n{rep['n_ok']}/{rep['n_expected']} modelos coinciden con el manifiesto "
          f"-> {'OK' if rep['passed'] else 'FALLA'}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
    return 0 if rep['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
