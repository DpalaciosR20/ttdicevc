"""
verify_artefacts.py
───────────────────
Verifica una copia publicada de los modelos (dataset privado `tt2-artefactos`,
matriz §9) contra `meta/artefactos_manifest.csv`.

Busca bajo --root los archivos con las extensiones del manifiesto (`.pt` de los
modelos, `.npz` de los mapas de F3.7) y los empareja con el manifiesto por
nombre de archivo. Si un nombre aparece más de una vez (p. ej. un *dataset*
creado desde la salida de un notebook que también guardó una prueba de humo),
se usa la copia cuya ruta termina en `rel_path`. Si ninguna o más de una
termina así, el archivo es ambiguo. Compara bytes y SHA-256. Falla (código 1)
si falta algún archivo, si alguno no coincide o si alguno es ambiguo. Los
archivos que no están en el manifiesto y las copias descartadas se reportan,
sin fallar.

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


def find_all(root, exts=('.pt',)):
    """{nombre: [rutas]} de los archivos con esas extensiones bajo root."""
    found = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(tuple(exts)):
                found.setdefault(name, []).append(os.path.join(dirpath, name))
    for v in found.values():
        v.sort()
    return found


def locate(candidates, rel_path):
    """Elige la copia de un archivo: la única, o la única cuya ruta termina en
    rel_path. Devuelve (ruta o None, descartadas, ambiguo)."""
    if not candidates:
        return None, [], False
    if len(candidates) == 1:
        return candidates[0], [], False
    suffix = '/' + rel_path.replace(os.sep, '/').lstrip('/')
    match = [c for c in candidates if c.replace(os.sep, '/').endswith(suffix)]
    if len(match) == 1:
        return match[0], [c for c in candidates if c != match[0]], False
    return None, candidates, True


def find_models(root, exts=('.pt',)):
    """Compatibilidad: {nombre: primera ruta} y nombres repetidos."""
    allf = find_all(root, exts)
    return {k: v[0] for k, v in allf.items()}, sorted(k for k, v in allf.items() if len(v) > 1)


def verify(root, manifest, task=None):
    rows = list(csv.DictReader(open(manifest, encoding='utf-8')))
    if task:
        rows = [r for r in rows if r['task'] == task]
    expected = {os.path.basename(r['rel_path']): r for r in rows}
    found = find_all(root, sorted({os.path.splitext(n)[1] for n in expected}) or ['.pt'])
    files, discarded, ambiguous, used = [], [], [], {}
    for name, r in sorted(expected.items()):
        p, others, amb = locate(found.get(name, []), r['rel_path'])
        discarded += [os.path.relpath(o, root).replace(os.sep, '/') for o in others if not amb]
        if amb:
            ambiguous.append(name)
            files.append({'file': name, 'status': 'ambiguo',
                          'candidates': [os.path.relpath(c, root).replace(os.sep, '/') for c in others]})
            continue
        if p is None:
            files.append({'file': name, 'status': 'falta'})
            continue
        used[name] = p
        size, digest = os.path.getsize(p), sha256(p)
        ok = size == int(r['size_bytes']) and digest == r['sha256']
        files.append({'file': name, 'status': 'ok' if ok else 'distinto',
                      'found_at': os.path.relpath(p, root).replace(os.sep, '/'), 'bytes': size, 'sha256': digest})
    extra = sorted(set(found) - set(expected))
    n_ok = sum(f['status'] == 'ok' for f in files)
    return {
        'manifest': os.path.relpath(manifest, REPO).replace(os.sep, '/'),
        'task': task,
        'root': root,
        'dataset_dirs': sorted({os.path.relpath(os.path.dirname(p), root).split(os.sep)[0] for p in used.values()}),
        'n_expected': len(expected), 'n_ok': n_ok,
        'missing': [f['file'] for f in files if f['status'] == 'falta'],
        'mismatch': [f['file'] for f in files if f['status'] == 'distinto'],
        'ambiguous': ambiguous,
        'duplicates': sorted(k for k in expected if len(found.get(k, [])) > 1),
        'discarded_copies': sorted(discarded), 'extra': extra,
        'passed': n_ok == len(expected) and not ambiguous,
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
    for k in ('missing', 'mismatch', 'ambiguous', 'discarded_copies', 'extra'):
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
