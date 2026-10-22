"""
umap_io.py
───────────
Lectura de umap_params.pkl SIN umap-learn ni numba (plan v2, F0.2).

Problema: umap_params.pkl guarda, ademas de los arreglos, el objeto UMAP ya
ajustado. Deserializar ese objeto exige umap-learn/pynndescent/numba en
versiones compatibles con las que lo crearon, y falla con el bug conocido de
numba al reconstruir funciones compiladas.

Solucion: en TT2 el reductor no se usa (no se transforman datos nuevos; el
re-ajuste por fold se descarto, ver docs/hallazgos_metodologicos.md 3-4). Lo
unico que se necesita son 'params' (u, 5 componentes) y 'labels' por split,
que son arreglos de numpy. load_umap_arrays() carga el pkl con joblib pero
sustituye por un "cascaron" vacio cualquier clase de umap/pynndescent/numba
(o de un modulo no instalado), asi que funciona en cualquier entorno con
numpy + joblib.

Uso como script:
    python src/training/umap_io.py <umap_params.pkl>                # resumen + hallazgo 7
    python src/training/umap_io.py <umap_params.pkl> --export u.npz # arreglos planos

El .npz exportado (train_params, train_labels, val_params, ...) se puede usar
en lugar del .pkl en cualquier lado (run_grid.py --umap-params acepta ambos).

ADVERTENCIA: como todo pickle, cargar el .pkl ejecuta codigo del archivo. Usar
solo con el umap_params.pkl del proyecto (verificar su sha256, ver abajo).
"""

import argparse
import contextlib
import hashlib
import sys

import numpy as np

SPLITS = ('train', 'val', 'test')
EXPECTED_SIZES = {'train': 908, 'val': 114, 'test': 114}
# Modulos que nunca se reconstruyen: umap/pynndescent/numba rompen la
# deserializacion, y el MinMaxScaler se guardo con sklearn 1.7.2 (Kaggle trae
# 1.6.1). Ninguno hace falta para leer 'params' y 'labels'.
STUB_PREFIXES = ('umap', 'pynndescent', 'numba', 'llvmlite', 'sklearn')


class _Stub:
    """Cascaron que acepta cualquier construccion/estado de pickle."""
    _stub_name = '?'

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _make_stub('<llamada>')()

    def __repr__(self):
        return f'<stub {self._stub_name}>'


def _make_stub(qualname):
    return type('Stub', (_Stub,), {'_stub_name': qualname})


@contextlib.contextmanager
def _stubbing_unpickler(stubbed):
    from joblib import numpy_pickle
    cls = numpy_pickle.NumpyUnpickler
    original = cls.find_class

    def find_class(self, module, name):
        if module.split('.')[0] in STUB_PREFIXES:
            stubbed.add(f'{module}.{name}')
            return _make_stub(f'{module}.{name}')
        try:
            return original(self, module, name)
        except (ImportError, AttributeError):
            stubbed.add(f'{module}.{name}')
            return _make_stub(f'{module}.{name}')

    cls.find_class = find_class
    try:
        yield
    finally:
        cls.find_class = original


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def load_umap_arrays(path):
    """Devuelve {split: {'params': ndarray (n,5), 'labels': ndarray (n,)}}.

    Acepta el .pkl original (joblib) o el .npz exportado por este modulo.
    Valida tamanos 908/114/114, 5 componentes y etiquetas binarias. No valida
    el rango: en el pkl del proyecto val/test vienen recortados a [0,1]
    (hallazgo 7); out_of_range_report() y saturation_report() lo miden."""
    if str(path).endswith('.npz'):
        z = np.load(path)
        out = {s: {'params': z[f'{s}_params'], 'labels': z[f'{s}_labels']} for s in SPLITS}
    else:
        import joblib
        stubbed = set()
        with _stubbing_unpickler(stubbed):
            pkg = joblib.load(path)
        out = {s: {'params': np.asarray(pkg[s]['params']), 'labels': np.asarray(pkg[s]['labels'])}
               for s in SPLITS}
    for s in SPLITS:
        P, y = out[s]['params'], out[s]['labels']
        if P.shape != (EXPECTED_SIZES[s], 5) or y.shape != (EXPECTED_SIZES[s],):
            raise ValueError(f'{s}: params {P.shape}, labels {y.shape}; se esperan '
                             f'({EXPECTED_SIZES[s]}, 5) y ({EXPECTED_SIZES[s]},)')
        if not set(np.unique(y).tolist()) <= {0, 1}:
            raise ValueError(f'{s}: etiquetas no binarias {np.unique(y)}')
    return out


def export_npz(arrays, out_path):
    np.savez(out_path, **{f'{s}_{k}': arrays[s][k] for s in SPLITS for k in ('params', 'labels')})


def out_of_range_report(arrays, tol=1e-6):
    """Cuantas muestras de cada split tienen algun componente fuera de [0,1]
    (hallazgo 7). tol absorbe el redondeo de float32 (train cae en [0,1]
    salvo por ~1e-7). Devuelve lineas de texto."""
    lines = []
    for s in SPLITS:
        P = arrays[s]['params']
        below, above = P < -tol, P > 1 + tol
        rows = (below | above).any(axis=1)
        lines.append(f'{s:<6} min={P.min():+.4f} max={P.max():.4f}  muestras con algun u fuera de [0,1]: '
                     f'{rows.sum()}/{len(P)}')
        for c in range(P.shape[1]):
            nb, na = below[:, c].sum(), above[:, c].sum()
            if nb or na:
                lines.append(f'         u{c}: {nb} < 0 (min {P[:, c].min():+.4f}), '
                             f'{na} > 1 (max {P[:, c].max():.4f})')
    return lines


def saturation_report(arrays):
    """Muestras de val/test con algun componente exactamente en 0 o 1. En
    train hay exactamente un 0 y un 1 por componente (min/max del ajuste);
    en val/test, valores apilados en 0/1 indican recorte (hallazgo 7)."""
    lines = []
    for s in SPLITS:
        P, y = arrays[s]['params'].astype(np.float64), arrays[s]['labels']
        at = (P == 0) | (P == 1)
        rows = at.any(axis=1)
        lines.append(f'{s:<6} valores exactamente en 0/1 por componente: {at.sum(axis=0).tolist()}  '
                     f'muestras con alguno: {rows.sum()}/{len(P)} '
                     f'(label1 {int(rows[y == 1].sum())}/{int((y == 1).sum())}, '
                     f'label0 {int(rows[y == 0].sum())}/{int((y == 0).sum())})')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='umap_params.pkl (o .npz exportado)')
    ap.add_argument('--export', default=None, help='escribir los arreglos a este .npz')
    args = ap.parse_args()

    print(f'archivo: {args.path}\nsha256:  {sha256(args.path)}')
    arrays = load_umap_arrays(args.path)
    for s in SPLITS:
        y = arrays[s]['labels']
        print(f'{s:<6} n={len(y):>4}  label1={int(y.sum()):>4}  label0={int((y == 0).sum()):>4}  '
              f'dtype params={arrays[s]["params"].dtype}')
    print('\nRango de u (hallazgo 7):')
    print('\n'.join(out_of_range_report(arrays)))
    print('\nSaturacion en 0/1 (hallazgo 7):')
    print('\n'.join(saturation_report(arrays)))
    if args.export:
        export_npz(arrays, args.export)
        # El sha256 del .npz no es estable (el zip guarda la hora); el canonico es el del .pkl
        print(f'\nexportado: {args.export}')


if __name__ == '__main__':
    sys.exit(main())
