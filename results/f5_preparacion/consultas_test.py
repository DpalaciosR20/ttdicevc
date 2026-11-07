"""
consultas_test.py — Bloque A (SV), punto 1h: inventario de predicciones de prueba
de modelos visuales en todo results/.

Recorre results/ (recursivo) y registra:
  · CSV con columnas split y p_dep: filas split == test con p_dep no nulo y
    technique ∈ {gray_scott, perlin, lorenz}, arch ∈ {resnet18, efficientnet_b0, vit_b16};
  · JSON con test_probs / test_preds no nulos (cualquier nivel de anidamiento).
Además lista, aparte, CSV con una columna f1_test no nula (métricas, no predicciones).

No calcula ninguna métrica. Para distinguir copias de consultas distintas,
calcula una huella SHA-256 del vector de p_dep de prueba ordenado por sample_id
(CSV) o de test_probs (JSON); si dos registros de la misma clave tienen la misma
huella, son copias de la misma inferencia.

Uso: python results/f5_preparacion/consultas_test.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / 'results'
OUT_DIR = RESULTS / 'f5_preparacion'
TECHS = {'gray_scott', 'perlin', 'lorenz'}
ARCHS = {'resnet18', 'efficientnet_b0', 'vit_b16'}


def h(vec):
    return hashlib.sha256(np.asarray(vec, dtype=np.float64).tobytes()).hexdigest()[:16]


def walk_json(obj, path=()):
    """Devuelve los dicts que tienen test_probs o test_preds no nulos."""
    if isinstance(obj, dict):
        if obj.get('test_probs') is not None or obj.get('test_preds') is not None:
            yield path, obj
        for k, v in obj.items():
            yield from walk_json(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_json(v, path + (i,))


rows, metric_files, skipped = [], [], []
for f in sorted(RESULTS.rglob('*')):
    if not f.is_file() or OUT_DIR in f.parents:
        continue
    rel = f.relative_to(REPO).as_posix()
    if f.suffix == '.csv':
        try:
            head = pd.read_csv(f, nrows=0).columns
        except Exception as e:  # noqa: BLE001
            skipped.append((rel, f'no legible: {e}'))
            continue
        if {'split', 'p_dep', 'technique', 'arch'} <= set(head):
            df = pd.read_csv(f, dtype={'p_dep': str})
            df['p_dep'] = [float(x) if isinstance(x, str) else np.nan for x in df.p_dep]  # float() exacto (F2.2: read_csv difiere 1 ulp)
            t = df[(df.split == 'test') & df.p_dep.notna() & df.technique.isin(TECHS) & df.arch.isin(ARCHS)]
            for (tech, arch, strat, seed), g in t.groupby(['technique', 'arch', 'strategy', 'seed']):
                g = g.sort_values('sample_id')
                rows.append({'file': rel, 'kind': 'csv', 'fase_dir': f.parent.name, 'technique': tech, 'arch': arch,
                             'strategy': strat, 'seed': int(seed), 'variant': '', 'fase': f.parent.name,
                             'n_test': len(g), 'hash_test': h(g.p_dep.to_numpy())})
        elif 'f1_test' in head:
            df = pd.read_csv(f)
            n = int(df['f1_test'].notna().sum())
            if n:
                metric_files.append((rel, n))
    elif f.suffix == '.json':
        try:
            obj = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:  # noqa: BLE001
            skipped.append((rel, f'no legible: {e}'))
            continue
        for path, d in walk_json(obj):
            tech, arch = d.get('technique'), d.get('arch')
            if tech is not None and tech not in TECHS:
                continue
            if arch is not None and arch not in ARCHS:
                continue
            probs = d.get('test_probs')
            rows.append({'file': rel, 'kind': 'json' + (f'@{"/".join(map(str, path))}' if path else ''),
                         'fase_dir': f.parent.parent.name if f.parent.name == 'runs' else f.parent.name,
                         'technique': tech, 'arch': arch, 'strategy': d.get('strategy'), 'seed': d.get('seed'),
                         'variant': d.get('variant', ''), 'fase': d.get('fase'),
                         'n_test': len(probs) if probs is not None else len(d.get('test_preds') or []),
                         'hash_test': h(probs) if probs is not None else 'solo_preds',
                         'smoke': d.get('smoke')})

R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / 'consultas_test.csv', index=False)
print(f'Registros con predicciones de prueba de modelos visuales: {len(R)} '
      f'(CSV: {(R.kind == "csv").sum()}, JSON: {R.kind.str.startswith("json").sum()})')
print('\nPor directorio:')
print(R.assign(tipo=R.kind.str.split('@').str[0]).groupby(['fase_dir', 'tipo']).size().to_string())

# Clave lógica de consulta: (técnica, arch, estrategia, fase, semilla)
R['fase_key'] = R['fase'].fillna(R['fase_dir'])
key = ['technique', 'arch', 'strategy', 'fase_key', 'seed']
g = R.groupby(key, dropna=False).agg(n_registros=('file', 'size'), n_huellas=('hash_test', 'nunique'),
                                     archivos=('file', lambda s: ' | '.join(sorted(s))))
print(f'\nClaves (técnica, arch, estrategia, fase, semilla): {len(g)}')
print(g['n_registros'].value_counts().rename('claves con n registros').to_string())
bad = g[g.n_huellas != 1]
print(f'\nClaves con más de una inferencia distinta (huellas distintas): {len(bad)}')
if len(bad):
    print(bad.to_string())
print('\nClaves por fase:')
print(R.drop_duplicates(key).groupby('fase_key', dropna=False).size().to_string())
other = R[~R.fase_key.isin(['final', 'control_permutacion', 'sensibilidad'])]
print(f'\nRegistros fuera de final / control_permutacion / sensibilidad: {len(other)}')
if len(other):
    print(other[['file', 'kind', 'fase_key', 'technique', 'arch', 'strategy', 'seed', 'smoke']].to_string(index=False))
# Una fila por clave para comprobar «exactamente una por clave» en las 45
ok45 = g.reset_index()
ok45 = ok45[ok45.fase_key.isin(['final', 'control_permutacion', 'sensibilidad'])]
print(f'\nClaves en las tres fases: {len(ok45)} (se esperan 45); todas con una sola huella: {bool((ok45.n_huellas == 1).all())}')
print('\nCSV con columna f1_test no nula (métricas de prueba, no predicciones):')
for rel, n in metric_files:
    print(f'  {rel}: {n} filas')
if skipped:
    print('\nArchivos no legibles:')
    for rel, why in skipped:
        print(f'  {rel}: {why}')
sys.exit(0)
