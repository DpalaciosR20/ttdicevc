"""
f21_fragment.py
───────────────
Escribe results/f21_techo5d/canonical_fragment.json a partir de los archivos
de predicciones de F2.1 (solo filas de val; test bloqueado hasta la Fase 5,
protocolo §10).

  ref.techo5d.{logreg,svm_rbf,rf,gbm,mlp}.f1_macro
      verificado; mejor combinación de cada familia de §4.3 por F1-Val
  ref.techo5d.f1_macro
      verificado; techo 5D = familia seleccionada por F1-Val (§4.3). Su
      source.path es el CSV de esa familia y evaluator/note la nombran

Las entradas WITCOM ref.umap5d.* (reportadas, otro evaluador) se conservan:
estas claves son nuevas y no las sustituyen.

Uso (desde la raíz del repo):
    python results/f21_techo5d/f21_fragment.py
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
NAMES = {'logreg': 'LogReg (L2, lbfgs)', 'svm_rbf': 'SVM-RBF', 'rf': 'Random Forest',
         'gbm': 'HistGradientBoosting', 'mlp': 'MLP'}


def csv_of(fam):
    return f'{PRED}/umap5d__{fam}__none__seed42.csv'


def val_metric(fam):
    df = validate_predictions(paths.REPO / csv_of(fam))
    sel = df[df['split'] == 'val']  # test no se toca (§10)
    cm = confusion_counts(sel['y_true'], (sel['p_dep'] > 0.5).astype(int))
    return f1_macro_from_confusion(cm), cm


def main():
    cfg = json.loads((REPO / 'configs' / 'protocolo.json').read_text(encoding='utf-8'))['techo_5d']
    order = cfg['family_order']
    sel = pd.read_csv(OUT_DIR / 'seleccion.csv')
    if list(sel['familia']) != order:
        raise SystemExit(f'seleccion.csv: familias {list(sel["familia"])} != family_order {order}')

    numbers, f1 = {}, {}
    for _, r in sel.iterrows():
        fam = r['familia']
        manifest = json.loads((paths.REPO / csv_of(fam).replace('.csv', '.manifest.json'))
                              .read_text(encoding='utf-8'))
        value, cm = val_metric(fam)
        if value != r['f1_val'] and abs(value - r['f1_val']) > TOL:
            raise SystemExit(f'{fam}: F1-Val de las predicciones {value!r} != seleccion.csv {r["f1_val"]!r}')
        f1[fam] = (value, cm, manifest)
        params = json.dumps(manifest['config']['estimator_params'], ensure_ascii=False)
        numbers[f'ref.techo5d.{fam}.f1_macro'] = {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
            'value': {'val': value, 'test': None}, 'null_reason': {'test': TEST_BLOCKED},
            'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS),
            'evaluator': (f'{NAMES[fam]} sobre los 5 componentes u de umap_params.pkl '
                          f'({manifest["config"]["pipeline"]}; {params}), mejor combinación de la rejilla '
                          f'de §4.3 por F1-Val (empate 1e-12 → anterior en la rejilla); ' + DECISION),
            'source': {'path': csv_of(fam), 'locator': 'split val'},
            'note': f'Familia de F2.1; rejilla completa en results/f21_techo5d/rejilla_completa.csv.',
        }

    best = max(v for v, _, _ in f1.values())
    techo = next(f for f in order if f1[f][0] >= best - TOL)
    flagged = sel.loc[sel['techo'], 'familia'].tolist()
    if flagged != [techo]:
        raise SystemExit(f'seleccion.csv marca techo={flagged}; recalculado: {techo}')
    value, cm, manifest = f1[techo]
    numbers['ref.techo5d.f1_macro'] = {
        'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
        'value': {'val': value, 'test': None}, 'null_reason': {'test': TEST_BLOCKED},
        'confusion': {'val': cm}, 'supports_ref': dict(SUPPORTS),
        'evaluator': (f'techo 5D = familia {techo} ({NAMES[techo]}, '
                      f'{json.dumps(manifest["config"]["estimator_params"], ensure_ascii=False)}), '
                      f'la de mayor F1-Val entre {", ".join(order)} (§4.3; empate 1e-12 → familia anterior); '
                      + DECISION),
        'source': {'path': csv_of(techo), 'locator': 'split val'},
        'note': (f'Familia seleccionada como techo 5D: {techo}. Mismas predicciones que '
                 f'ref.techo5d.{techo}.f1_macro. No sustituye a las cifras WITCOM ref.umap5d.*.'),
    }

    frag = {'schema_version': SCHEMA_VERSION, 'numbers': numbers}
    validate(frag, fragment=True)
    out = OUT_DIR / 'canonical_fragment.json'
    out.write_text(json.dumps(frag, sort_keys=True, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')
    print(f'escrito {out.relative_to(paths.REPO).as_posix()}')
    for f in order:
        print(f'  ref.techo5d.{f}.f1_macro  val {f1[f][0]:.6f}')
    print(f'  ref.techo5d.f1_macro  val {value:.6f}  (familia {techo})')


if __name__ == '__main__':
    main()
