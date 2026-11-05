"""
f24_fragment.py
───────────────
Escribe results/f24_descriptores/canonical_fragment.json a partir de las salidas
de F2.4 (solo filas de val; test bloqueado hasta la Fase 5, protocolo §10).

  ref.descriptores.{perlin,gray_scott,lorenz}.f1_macro
      verificado; clasificador seleccionado por F1-Val entre las cinco familias
      de §4.3 sobre los 4 descriptores de bajo nivel
  m2.tabla11.recalculada.n_sig_holm
      verificado (count); comparaciones significativas tras Holm (k = 12) en
      tabla11_recalculada.csv (imágenes versionadas de tt2-insumos, solo train)
  m2.tabla11.original.n_sig_holm
      reportado (count); salida guardada de la celda 20 del notebook de CE
      (imágenes de la sesión de Kaggle, no versionadas)

El esquema solo admite F1-Macro y conteos: t, p y d quedan en
tabla11_recalculada.csv. Las dos entradas de la Tabla 11 tienen claves
distintas y no llevan supersedes (cada una se refiere a imágenes distintas).

Uso (desde la raíz del repo):
    python results/f24_descriptores/f24_fragment.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src import paths  # noqa: E402
from src.canonical.build import (SCHEMA_VERSION, confusion_counts, f1_macro_from_confusion,  # noqa: E402
                                 validate)
from src.predictions.validate import validate_predictions  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
REL_DIR = 'results/f24_descriptores'
PRED = 'results/predictions/referencia'
NOTEBOOK = 'notebooks/historial/camera_ready_CE.ipynb'
SUPPORTS = {'val': 'support.val', 'test': 'support.test'}
TOL = 1e-12
TEST_BLOCKED = ('bloqueado hasta la Fase 5 (docs/protocolo_preregistrado.md §10); '
                'las predicciones de test están archivadas')
DECISION = "y_pred = 1[p_dep > 0.5]; f1_macro_from_confusion (= sklearn f1_score(average='macro'))"


def val_metric(rel):
    df = validate_predictions(paths.REPO / rel)
    sel = df[df['split'] == 'val']  # test no se toca (§10)
    cm = confusion_counts(sel['y_true'], (sel['p_dep'] > 0.5).astype(int))
    return f1_macro_from_confusion(cm), cm


def original_n_sig():
    nb = json.loads((REPO / NOTEBOOK).read_text(encoding='utf-8'))
    text = ''.join(''.join(o.get('text', '')) for o in nb['cells'][20].get('outputs', []))
    lines = [ln.split() for ln in text.strip().splitlines()]
    col = lines[0].index('sig_holm')
    vals = [ln[col] for ln in lines[1:]]
    if len(vals) != 12 or set(vals) - {'True', 'False'}:
        raise SystemExit('salida de la celda 20 inesperada')
    return sum(v == 'True' for v in vals)


def main():
    numbers = {}
    grid = pd.read_csv(OUT_DIR / 'rejilla_descriptores.csv')
    sel = pd.read_csv(OUT_DIR / 'seleccion.csv')
    for t in paths.TECHNIQUES:
        rel = f'{PRED}/{t}__descriptors__none__seed42.csv'
        value, cm = val_metric(rel)
        m = json.loads((paths.REPO / rel.replace('.csv', '.manifest.json')).read_text(encoding='utf-8'))
        ms = m['model_selection']
        row = grid[(grid['tecnica'] == t) & (grid['familia'] == ms['familia'])
                   & (grid['orden'] == ms['orden_en_rejilla'])]
        if len(row) != 1:
            raise SystemExit(f'{t}: la combinación seleccionada no está en rejilla_descriptores.csv')
        for name, expected in (('rejilla', float(row['f1_val'].iloc[0])), ('manifiesto', ms['f1_val'])):
            if abs(value - expected) > TOL:
                raise SystemExit(f'{t}: F1-Val de las predicciones {value!r} != {name} {expected!r}')
        if value < grid.loc[grid['tecnica'] == t, 'f1_val'].max() - TOL:
            raise SystemExit(f'{t}: la combinación seleccionada no es la de mayor F1-Val')
        chosen = sel[(sel['tecnica'] == t) & sel['seleccionada']]
        if len(chosen) != 1 or chosen['familia'].iloc[0] != ms['familia']:
            raise SystemExit(f'{t}: seleccion.csv no coincide con el manifiesto')
        params = json.dumps(m['config']['estimator_params'], ensure_ascii=False)
        numbers[f'ref.descriptores.{t}.f1_macro'] = {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
            'value': {'val': value, 'test': None}, 'null_reason': {'test': TEST_BLOCKED},
            'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS),
            'evaluator': (f'clasificador sobre 4 descriptores de bajo nivel ({t}; entropía de Shannon, contraste, '
                          'densidad de bordes y energía LBP de la imagen nativa, configs/protocolo.json → '
                          f'descriptores): {m["config"]["pipeline"]} ({params}), mejor de las cinco familias y '
                          'rejillas de §4.3 por F1-Val (empate 1e-12 → familia anterior, luego combinación '
                          'anterior); ' + DECISION),
            'source': {'path': rel, 'locator': 'split val'},
            'note': f'F2.4; rejilla en {REL_DIR}/rejilla_descriptores.csv, selección en {REL_DIR}/seleccion.csv.'}

    t11 = pd.read_csv(OUT_DIR / 'tabla11_recalculada.csv')
    if len(t11) != 12:
        raise SystemExit('tabla11_recalculada.csv no tiene 12 filas')
    n_orig = original_n_sig()
    if int(t11['sig_holm_orig_A'].sum()) != n_orig:
        raise SystemExit('sig_holm_orig_A no coincide con la salida de la celda 20')
    numbers['m2.tabla11.recalculada.n_sig_holm'] = {
        'kind': 'count', 'status': 'verificado', 'value': int(t11['sig_holm'].sum()),
        'evaluator': ('comparaciones con p de Holm < 0.05 (statsmodels multipletests, k = 12) de la t de Welch '
                      'depresivo − no depresivo (scipy ttest_ind, equal_var=False) por técnica y descriptor; solo '
                      'train (908); imágenes de tt2-insumos verificadas con meta/data_manifest.csv'),
        'source': {'path': f'{REL_DIR}/tabla11_recalculada.csv', 'locator': 'suma de la columna sig_holm'},
        'note': ('F2.4. Tabla 11 del Módulo 2 recalculada con el código de la celda 20 de '
                 f'{NOTEBOOK}; t, p y d de Cohen en el CSV.')}
    numbers['m2.tabla11.original.n_sig_holm'] = {
        'kind': 'count', 'status': 'reportado', 'value': n_orig,
        'evaluator': 'mismo cálculo que la Tabla 11 recalculada, ejecutado en Kaggle el 29 jul 2026',
        'source': {'path': NOTEBOOK, 'locator': 'celda 20, salida guardada, columna sig_holm'},
        'note': ('imágenes de sesión de Kaggle, no versionadas (/kaggle/working/images_grayscale_classblind). '
                 'No se sustituye por m2.tabla11.recalculada.n_sig_holm: se refieren a imágenes distintas.')}

    frag = {'schema_version': SCHEMA_VERSION, 'numbers': numbers}
    validate(frag, fragment=True)
    out = OUT_DIR / 'canonical_fragment.json'
    out.write_text(json.dumps(frag, sort_keys=True, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')
    print(f'escrito {out.relative_to(paths.REPO).as_posix()}')
    for k, v in numbers.items():
        print(f'  {k}  ' + (f'val {v["value"]["val"]:.6f}' if v['kind'] == 'metric' else f'{v["value"]}'))


if __name__ == '__main__':
    main()
