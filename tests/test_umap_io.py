"""
tests/test_umap_io.py
═════════════════════
F0.2 — Lectura de umap_params.pkl sin umap-learn / numba / sklearn.

Pruebas:
  1. Sintetica (requiere umap-learn): un pkl con un UMAP REAL ajustado y la
     misma estructura que el del proyecto se lee con load_umap_arrays() en un
     proceso donde umap/numba/pynndescent/sklearn estan bloqueados, y los
     arreglos salen identicos; el .npz exportado se relee identico.
  2. Con el pkl real (variable de entorno UMAP_PKL; se omite si no existe):
     - sha256 = el de referencia de docs/entorno_reproducibilidad.md
     - se lee con umap/numba/pynndescent/sklearn bloqueados
     - carga auditada: solo pide clases de numpy/joblib/scipy/sklearn (+ umap, sustituidas)
     - tamanos 908/114/114 y etiquetas de val = y_true de results/f32_primera_pasada

Ejecucion:
    UMAP_PKL=/ruta/umap_params.pkl python -m pytest tests/test_umap_io.py -v -s
"""
import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'src' / 'training'
sys.path.insert(0, str(TRAINING))
import umap_io  # noqa: E402

REF_SHA256 = '081f5baba6eacf26de2993304747671365f1807620e55ed2be01cf3b75c7f224'
REAL_PKL = os.environ.get('UMAP_PKL')
BLOCKED = ('umap', 'numba', 'pynndescent', 'llvmlite', 'sklearn')

needs_real = pytest.mark.skipif(not REAL_PKL or not Path(REAL_PKL).exists(),
                                reason='UMAP_PKL no definido o inexistente')


def _run_blocked(code, *args):
    """Ejecuta `code` en un proceso nuevo con los modulos BLOCKED bloqueados."""
    prelude = textwrap.dedent(f"""
        import sys, importlib.abc
        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split('.')[0] in {BLOCKED!r}:
                    raise ImportError('BLOQUEADO: ' + name)
        sys.meta_path.insert(0, _Block())
        sys.path.insert(0, {str(TRAINING)!r})
    """)
    out = subprocess.run([sys.executable, '-I', '-c', prelude + textwrap.dedent(code), *map(str, args)],
                         capture_output=True, text=True)
    print(out.stdout)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_sintetico_umap_real_bloqueado(tmp_path):
    umap = pytest.importorskip('umap')
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    rng = np.random.default_rng(0)
    X = {s: rng.normal(size=(n, 32)) for s, n in umap_io.EXPECTED_SIZES.items()}
    red = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', n_components=5, random_state=42)
    ctr = red.fit_transform(X['train'])
    sc = MinMaxScaler().fit(ctr)
    pkg = {'reducer': red, 'scaler': sc, 'n_components': 5, 'metric': 'cosine', 'random_state': 42}
    pkg['train'] = {'params': sc.transform(ctr), 'labels': (rng.uniform(size=908) < .39).astype(int)}
    for s in ('val', 'test'):
        pkg[s] = {'params': np.clip(sc.transform(red.transform(X[s])), 0, 1),
                  'labels': (rng.uniform(size=114) < .39).astype(int)}
    pkl, ref = tmp_path / 'sint.pkl', tmp_path / 'ref.npz'
    joblib.dump(pkg, pkl)
    umap_io.export_npz({s: pkg[s] for s in umap_io.SPLITS}, ref)

    out = _run_blocked("""
        import numpy as np, umap_io
        try:
            import umap; print('umap importable'); sys.exit(1)
        except ImportError: pass
        a = umap_io.load_umap_arrays(sys.argv[1]); ref = umap_io.load_umap_arrays(sys.argv[2])
        assert all(np.array_equal(a[s][k], ref[s][k]) for s in a for k in a[s])
        umap_io.export_npz(a, sys.argv[3]); b = umap_io.load_umap_arrays(sys.argv[3])
        assert all(np.array_equal(a[s][k], b[s][k]) for s in a for k in a[s])
        print('SINTETICO: arreglos identicos con umap/numba/sklearn bloqueados; npz identico')
    """, pkl, ref, tmp_path / 'exp.npz')
    assert 'SINTETICO' in out


@needs_real
def test_real_sha256():
    sha = umap_io.sha256(REAL_PKL)
    print(f'sha256 {sha}  tamano {Path(REAL_PKL).stat().st_size} bytes')
    assert sha == REF_SHA256


@needs_real
def test_real_lectura_bloqueada():
    out = _run_blocked("""
        import umap_io
        a = umap_io.load_umap_arrays(sys.argv[1])
        for s in umap_io.SPLITS:
            y = a[s]['labels']
            print(f"REAL {s}: params {a[s]['params'].shape} {a[s]['params'].dtype} "
                  f"label1={int(y.sum())} label0={int((y == 0).sum())}")
        print('\\n'.join(umap_io.out_of_range_report(a)))
        print('\\n'.join(umap_io.saturation_report(a)))
    """, REAL_PKL)
    assert 'REAL test' in out


@needs_real
def test_real_carga_auditada():
    out = _run_blocked("""
        import collections, umap_io
        from joblib import numpy_pickle
        seen = collections.Counter()
        orig = numpy_pickle.NumpyUnpickler.find_class
        def audit(self, module, name):
            seen[f'{module}.{name}'] += 1
            return orig(self, module, name)
        numpy_pickle.NumpyUnpickler.find_class = audit
        stubbed = set()
        with umap_io._stubbing_unpickler(stubbed):
            import joblib; joblib.load(sys.argv[1])
        roots = sorted({k.split('.')[0] for k in seen})
        print('AUDITORIA modulos resueltos normalmente:', roots)
        print('AUDITORIA clases sustituidas:', sorted({k.split('.')[0] for k in stubbed}), len(stubbed))
        print('AUDITORIA no numpy/joblib:', sorted(k for k in seen if k.split('.')[0] not in ('numpy', 'joblib')))
        assert set(roots) <= {'numpy', 'joblib', 'scipy', 'builtins', 'copyreg', 'collections'}, roots
    """, REAL_PKL)
    assert 'AUDITORIA' in out


@needs_real
def test_real_etiquetas_val_coinciden_con_f32():
    a = umap_io.load_umap_arrays(REAL_PKL)
    pred = ROOT / 'results' / 'f32_primera_pasada' / 'predictions.csv'
    with open(pred, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f)
                if r['run_id'] == 'gray_scott__resnet18__finetune__px224-flip__seed42']
    y_f32 = np.array([int(r['y_true']) for r in rows])
    print(f'val: {len(y_f32)} filas de F3.2; coinciden con umap_params: {np.array_equal(y_f32, a["val"]["labels"])}')
    assert np.array_equal(y_f32, a['val']['labels'])
