"""
f23_fragment.py
───────────────
Escribe results/f23_pisos/canonical_fragment.json a partir de los archivos de
predicciones de F2.3 (solo filas de val; test bloqueado hasta la Fase 5,
protocolo §10).

  ref.piso.majority.f1_macro
      verificado; p_dep = 0.0. Su F1-Test es la cifra de diseño
      f12.piso_mayoritaria.f1_macro (aquí test queda nulo)
  ref.piso.stratified.f1_macro
      verificado; archivo de completitud (Enmienda 2, punto 3); en los
      contrastes interviene la constante 0.5
  ref.piso.brightness.{perlin,gray_scott,lorenz}.f1_macro
      verificado; LogReg sobre la intensidad media, mejor combinación por F1-Val

Las entradas WITCOM floor.brightness.* (otro evaluador) se conservan: estas
claves son nuevas y no llevan supersedes.

Uso (desde la raíz del repo):
    python results/f23_pisos/f23_fragment.py
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
PRED = 'results/predictions/referencia'
SUPPORTS = {'val': 'support.val', 'test': 'support.test'}
TOL = 1e-12
TEST_BLOCKED = ('bloqueado hasta la Fase 5 (docs/protocolo_preregistrado.md §10); '
                'las predicciones de test están archivadas')
DECISION = "y_pred = 1[p_dep > 0.5]; f1_macro_from_confusion (= sklearn f1_score(average='macro'))"


def csv_of(technique, arch):
    return f'{PRED}/{technique}__{arch}__none__seed42.csv'


def val_metric(rel):
    df = validate_predictions(paths.REPO / rel)
    sel = df[df['split'] == 'val']  # test no se toca (§10)
    cm = confusion_counts(sel['y_true'], (sel['p_dep'] > 0.5).astype(int))
    return f1_macro_from_confusion(cm), cm


def manifest_of(rel):
    return json.loads((paths.REPO / rel.replace('.csv', '.manifest.json')).read_text(encoding='utf-8'))


def entry(rel, value, cm, evaluator, note):
    return {'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
            'value': {'val': value, 'test': None}, 'null_reason': {'test': TEST_BLOCKED},
            'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS),
            'evaluator': evaluator + '; ' + DECISION,
            'source': {'path': rel, 'locator': 'split val'}, 'note': note}


def check(name, value, expected):
    if value != expected and abs(value - expected) > TOL:
        raise SystemExit(f'{name}: F1-Val de las predicciones {value!r} != {expected!r}')


def main():
    numbers = {}

    rel = csv_of('none', 'majority')
    value, cm = val_metric(rel)
    check('majority', value, manifest_of(rel)['model_selection']['f1_val'])
    numbers['ref.piso.majority.f1_macro'] = entry(
        rel, value, cm, 'piso de clase mayoritaria: p_dep = 0.0 en todas las muestras (protocolo §4.4; Enmienda 2, '
                        'punto 3)',
        'F2.3. Su F1-Test es la cifra de diseño f12.piso_mayoritaria.f1_macro; aquí test queda nulo.')

    rel = csv_of('none', 'stratified')
    value, cm = val_metric(rel)
    m = manifest_of(rel)
    check('stratified', value, m['model_selection']['f1_val'])
    numbers['ref.piso.stratified.f1_macro'] = entry(
        rel, value, cm, f'piso aleatorio estratificado: {m["config"]["rule"]}',
        'F2.3. Archivo por completitud del formato (Enmienda 2, punto 3): en los contrastes interviene la '
        'constante 0.5 (§5.3), no este archivo.')

    grid = pd.read_csv(OUT_DIR / 'rejilla_brillo.csv')
    for t in paths.TECHNIQUES:
        rel = csv_of(t, 'brightness')
        value, cm = val_metric(rel)
        m = manifest_of(rel)
        row = grid[(grid['tecnica'] == t) & (grid['orden'] == m['model_selection']['orden_en_rejilla'])]
        if len(row) != 1:
            raise SystemExit(f'{t}: la combinación seleccionada no está en rejilla_brillo.csv')
        check(t, value, float(row['f1_val'].iloc[0]))
        best = grid.loc[grid['tecnica'] == t, 'f1_val'].max()
        if value < best - TOL:
            raise SystemExit(f'{t}: la combinación seleccionada no es la de mayor F1-Val')
        params = json.dumps(m['config']['estimator_params'], ensure_ascii=False)
        numbers[f'ref.piso.brightness.{t}.f1_macro'] = entry(
            rel, value, cm,
            (f'piso de brillo ({t}): intensidad media / 255 de la imagen nativa de 256 × 256; '
             f'{m["config"]["pipeline"]} ({params}), mejor combinación de la rejilla completa de LogReg de §4.3 '
             f'por F1-Val (empate 1e-12 → anterior en la rejilla)'),
            ('F2.3; rejilla en results/f23_pisos/rejilla_brillo.csv. No sustituye a la cifra WITCOM '
             f'floor.brightness.{t}.f1_macro.'))

    frag = {'schema_version': SCHEMA_VERSION, 'numbers': numbers}
    validate(frag, fragment=True)
    out = OUT_DIR / 'canonical_fragment.json'
    out.write_text(json.dumps(frag, sort_keys=True, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')
    print(f'escrito {out.relative_to(paths.REPO).as_posix()}')
    for k, v in numbers.items():
        print(f'  {k}  val {v["value"]["val"]:.6f}')


if __name__ == '__main__':
    main()
