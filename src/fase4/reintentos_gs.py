"""
reintentos_gs.py
────────────────
F4.2, punto 5 (docs/plan_fase4.md): reintentos de Gray-Scott. Se corre en Kaggle con CPU.

Regenera las 908 imágenes de train de Gray-Scott por la ruta CPU de producción
(gray_scott_lib.generate_and_save_grayscale → run_gray_scott_safe → run_gray_scott)
y registra, por muestra, cuántos reintentos hizo el wrapper y si usó el patrón
de respaldo. La ruta GPU no se usa (decisión 6 de TT-II; F0.3, H2).

Instrumentación sin cambiar la lógica del generador:
  - Durante cada muestra, el atributo de módulo gray_scott_lib.run_gray_scott
    se sustituye por un envoltorio. El envoltorio llama a la función original
    con los mismos argumentos, devuelve su mismo resultado y anota los
    parámetros de la llamada y la varianza de V.
  - run_gray_scott_safe resuelve run_gray_scott como global del módulo en cada
    llamada, así que la ve sin modificarse.
  - Al terminar la muestra se restaura la original. Nada de src/generation se edita.
  - run_gray_scott_safe devuelve, además de V, el número de intento. Ese valor
    se propaga hasta generate_and_save_grayscale, y de ahí se toman
    reintentos (0…max_retries) y usa_respaldo (reintentos == max_retries).
  - Se comprueba que esos valores sean coherentes con las llamadas
    registradas: reintentos + 1 llamadas sin respaldo, max_retries + 1 con
    respaldo, y la última con los parámetros fijos del respaldo.

Verificación «bit a bit» de cada imagen regenerada:
  - sha256 del PNG regenerado frente a meta/data_manifest.csv (sha256_coincide).
  - Píxeles del regenerado frente al PNG almacenado en TT_DATA_ROOT
    (pixeles_coinciden). Es el diagnóstico si un sha256 difiere solo por el
    codificador PNG.
  - sha256 del PNG almacenado frente al manifiesto (almacenada_integra).
  El criterio que manda en F4.2 es PENDIENTE (bit_a_bit, reporte_e2a.md).
  Aquí se registran los tres.

u se toma de umap_params.pkl con el dtype almacenado. Convertirlo alteraría
la aritmética de Gray-Scott (tests/test_label_independence.py, U_y).

Salida en --out-dir, y se niega a sobrescribir:
  reintentos_gs_train.csv       una fila por muestra, en orden de sample_id
  reintentos_gs_manifest.json   commit, entorno, totales, sha256 del CSV, tiempos
Las imágenes regeneradas van a --regen-dir, que debe no existir o estar vacío.
No se versionan.

Uso:
  python -m src.fase4.reintentos_gs --out-dir <dir> --regen-dir <dir> [--limit 16] [--workers N]
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / 'src' / 'training'), str(REPO / 'src' / 'generation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gray_scott_lib as gsl  # noqa: E402

SPLIT = 'train'
TECH = 'gray_scott'
GATE_TAG = 'enmienda-5'
CSV_NAME = 'reintentos_gs_train.csv'
MANIFEST_NAME = 'reintentos_gs_manifest.json'
MAX_RETRIES = inspect.signature(gsl.run_gray_scott_safe).parameters['max_retries'].default
# Parámetros fijos del patrón de respaldo de run_gray_scott_safe. Si el
# generador cambia, la comprobación de coherencia lo detecta.
RESPALDO = {'f': 0.035, 'k': 0.061, 'Du': 0.16, 'Dv': 0.06, 'seed_radius': 10, 'n_iter': 5000}
COLUMNAS = ['sample_id', 'idx', 'y_true', 'u0', 'u1', 'u2', 'u3', 'u4', 'f', 'k', 'Dv', 'seed_radius', 'n_iter',
            'reintentos', 'usa_respaldo', 'n_llamadas', 'var_por_llamada', 'archivo', 'sha256_regenerada',
            'sha256_manifiesto', 'sha256_coincide', 'pixeles_coinciden', 'almacenada_integra', 'segundos']


class Detener(RuntimeError):
    """La corrida se detiene porque falla una comprobación."""


@contextmanager
def registro():
    """Durante el bloque, anota cada llamada a gray_scott_lib.run_gray_scott sin alterar su resultado."""
    original = gsl.run_gray_scott
    llamadas = []

    def envoltorio(*args, **kwargs):
        V = original(*args, **kwargs)
        llamadas.append({**kwargs, 'n_args_posicionales': len(args), 'var': float(V.var())})
        return V

    gsl.run_gray_scott = envoltorio
    try:
        yield llamadas
    finally:
        gsl.run_gray_scott = original


def coherencia(reintentos, llamadas):
    """Comprueba que los reintentos devueltos cuadren con las llamadas registradas. Devuelve usa_respaldo."""
    usa_respaldo = reintentos == MAX_RETRIES
    esperadas = MAX_RETRIES + 1 if usa_respaldo else reintentos + 1
    if len(llamadas) != esperadas:
        raise Detener(f'{len(llamadas)} llamadas registradas; con reintentos = {reintentos} se esperan {esperadas}')
    if usa_respaldo and any(llamadas[-1].get(k) != v for k, v in RESPALDO.items()):
        raise Detener(f'la última llamada no es el patrón de respaldo: {llamadas[-1]}')
    if not usa_respaldo and llamadas[-1]['var'] <= 1e-4:
        raise Detener('sin respaldo, la última llamada debería tener varianza > 1e-4')
    return usa_respaldo


def regenerar(idx, u, label, out_dir):
    """Una muestra por la ruta CPU de producción, con registro. Devuelve (ruta, reintentos, params, usa_respaldo, llamadas)."""
    with registro() as llamadas:
        fpath, reintentos, params = gsl.generate_and_save_grayscale(idx, u, label, SPLIT, str(out_dir))
    return fpath, int(reintentos), params, coherencia(int(reintentos), llamadas), llamadas


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pixeles(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im)


def _trabajo(args):
    """Una muestra en un proceso hijo: regenera, verifica y devuelve la fila del CSV."""
    sid, idx, label, u, out_dir, stored_path, sha_manifest = args
    t0 = time.perf_counter()
    fpath, reintentos, params, respaldo, llamadas = regenerar(idx, u, label, out_dir)
    seg = time.perf_counter() - t0
    sha_regen = sha256_file(fpath)
    return {'sample_id': sid, 'idx': idx, 'y_true': label, **{f'u{j}': repr(float(u[j])) for j in range(5)},
            'f': repr(params['f']), 'k': repr(params['k']), 'Dv': repr(params['Dv']),
            'seed_radius': params['seed_radius'], 'n_iter': params['n_iter'],
            'reintentos': reintentos, 'usa_respaldo': respaldo, 'n_llamadas': len(llamadas),
            'var_por_llamada': ';'.join(f"{c['var']:.6g}" for c in llamadas),
            'archivo': Path(fpath).name, 'sha256_regenerada': sha_regen, 'sha256_manifiesto': sha_manifest,
            'sha256_coincide': sha_regen == sha_manifest,
            'pixeles_coinciden': bool(np.array_equal(pixeles(fpath), pixeles(stored_path))),
            'almacenada_integra': sha256_file(stored_path) == sha_manifest,
            'segundos': round(seg, 3)}


# ── Insumos ───────────────────────────────────────────────────────────────
def load_index():
    with open(REPO / 'meta' / 'sample_index.csv', newline='', encoding='utf-8') as f:
        return sorted((r for r in csv.DictReader(f) if r['split'] == SPLIT), key=lambda r: r['sample_id'])


def load_manifest_shas():
    from src import paths
    pre = paths.REL['IMAGES'][TECH] + '/'
    with open(paths.DATA_MANIFEST, newline='', encoding='utf-8') as f:
        return {r['rel_path'][len(pre):]: r['sha256'] for r in csv.DictReader(f)
                if r['kind'] == 'file' and r['rel_path'].startswith(pre)}


def tareas(index, U, labels, out_dir, limit=None):
    """Argumentos por muestra. Con limit, toma muestras espaciadas de forma uniforme (prueba de humo)."""
    from src import paths
    shas = load_manifest_shas()
    sel = range(len(index)) if limit is None else np.unique(np.linspace(0, len(index) - 1, limit).round().astype(int))
    out = []
    for j in sel:
        r = index[j]
        idx, y = int(r['idx']), int(r['y_true'])
        if int(labels[idx]) != y:
            raise Detener(f"{r['sample_id']}: etiqueta de umap_params.pkl ≠ sample_index")
        name = paths.image_name(TECH, SPLIT, idx, y)
        if name != Path(gsl.expected_path(idx, y, SPLIT, '.')).name:
            raise Detener(f'{name}: nombre distinto de gray_scott_lib.expected_path')
        if name not in shas:
            raise Detener(f'{name} no está en meta/data_manifest.csv')
        out.append((r['sample_id'], idx, y, U[idx], str(out_dir), str(paths.images_dir(TECH) / name), shas[name]))
    return out


def check_dirs(out_dir, regen_dir):
    """Negativa a sobrescribir: ni el CSV ni el manifiesto deben existir, y regen_dir debe no existir o estar vacío."""
    for p in (Path(out_dir) / CSV_NAME, Path(out_dir) / MANIFEST_NAME):
        if p.exists():
            raise Detener(f'{p} ya existe; no se sobrescribe')
    rd = Path(regen_dir)
    if rd.exists() and any(rd.iterdir()):
        raise Detener(f'{rd} no está vacío; no se sobrescriben imágenes regeneradas')


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)


def write_csv(path, filas):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS, lineterminator='\n')
        w.writeheader()
        w.writerows(filas)


def resumen(filas):
    def por_clase(key):
        return {c: int(sum(bool(r[key]) if isinstance(r[key], bool) else int(r[key]) > 0
                           for r in filas if r['y_true'] == c)) for c in (0, 1)}
    return {'n': len(filas), 'n_dep': sum(r['y_true'] == 1 for r in filas), 'n_nodep': sum(r['y_true'] == 0 for r in filas),
            'reintentos_mayor_0_por_clase': por_clase('reintentos'), 'usa_respaldo_por_clase': por_clase('usa_respaldo'),
            'distribucion_reintentos': {str(k): sum(r['reintentos'] == k for r in filas) for k in range(MAX_RETRIES + 1)},
            'n_sha256_coincide': sum(r['sha256_coincide'] for r in filas),
            'n_pixeles_coinciden': sum(r['pixeles_coinciden'] for r in filas),
            'n_almacenada_integra': sum(r['almacenada_integra'] for r in filas),
            'no_coinciden_sha256': [r['sample_id'] for r in filas if not r['sha256_coincide']],
            'no_coinciden_pixeles': [r['sample_id'] for r in filas if not r['pixeles_coinciden']]}


def main(argv=None):
    ap = argparse.ArgumentParser(description='F4.2: reintentos de Gray-Scott (train, CPU)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--regen-dir', required=True)
    ap.add_argument('--limit', type=int, default=None, help='prueba de humo: n muestras espaciadas')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    from src import paths
    from src.build_manifest import verify_manifest
    from umap_io import load_umap_arrays

    check_dirs(a.out_dir, a.regen_dir)
    for d in (a.out_dir, a.regen_dir):
        paths.assert_outside_data_root(d)
    if _git('merge-base', '--is-ancestor', GATE_TAG, 'HEAD').returncode != 0:
        raise Detener(f'la etiqueta {GATE_TAG} no está en la historia de HEAD')
    problems = verify_manifest('UMAP_PARAMS')
    if problems:
        raise Detener(f'umap_params.pkl: {problems}')
    arr = load_umap_arrays(paths.umap_params_path())[SPLIT]
    index = load_index()
    args = tareas(index, arr['params'], arr['labels'], a.regen_dir, a.limit)
    Path(a.regen_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    filas = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, fila in enumerate(ex.map(_trabajo, args, chunksize=1), start=1):
            filas.append(fila)
            if k % 50 == 0 or k == len(args):
                print(f'[{k}/{len(args)}] {time.perf_counter() - t0:.0f} s', flush=True)
    total = time.perf_counter() - t0
    filas.sort(key=lambda r: r['sample_id'])
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(a.out_dir) / CSV_NAME
    write_csv(csv_path, filas)
    seg = [r['segundos'] for r in filas]
    import PIL
    man = {'tarea': 'F4.2 (reintentos de Gray-Scott)', 'git_commit': _git('rev-parse', 'HEAD').stdout.strip(),
           'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'smoke': a.limit is not None, 'limit': a.limit, 'split': SPLIT, 'ruta': 'cpu',
           'workers': a.workers, 'cpu_count': os.cpu_count(), 'max_retries': MAX_RETRIES, 'respaldo': RESPALDO,
           'umap_params_sha256': sha256_file(paths.umap_params_path()), 'u_dtype': str(arr['params'].dtype),
           'entorno': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
                       'plataforma': platform.platform()},
           'resumen': resumen(filas),
           'tiempo': {'segundos_total': round(total, 1), 'segundos_muestra_media': round(float(np.mean(seg)), 3),
                      'segundos_muestra_max': round(float(np.max(seg)), 3),
                      'estimacion_908_segundos': round(float(np.mean(seg)) * 908 / a.workers, 0)},
           'csv': {'archivo': CSV_NAME, 'sha256': sha256_file(csv_path), 'filas': len(filas)}}
    (Path(a.out_dir) / MANIFEST_NAME).write_text(json.dumps(man, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: man[k] for k in ('smoke', 'workers', 'resumen', 'tiempo')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
