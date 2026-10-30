"""
f22_fragment.py
───────────────
Escribe results/f22_referencias_textuales/canonical_fragment.json a partir de
los archivos de predicciones de F2.2 (docs/especificacion_f22.md §5).

  ref.beto_exp41.f1_macro         verificado, trainer_head; sustituye a la
                                  entrada reportada de F0.4 (supersedes)
  ref.emb768_logreg.f1_macro      verificado, logreg_on_embeddings; solo val
                                  (test bloqueado hasta la Fase 5, §4)
  ref.delta_ni.f1_macro           δ = BETO − TF-IDF lematizado (Exp 1.2)
  witcom.delta_beto_gs_resnet18.f1_macro
                                  Δ exploratorio de WITCOM = BETO − media de
                                  Gray-Scott + ResNet-18 (5 semillas)

Uso (desde la raíz del repo):
    python results/f22_referencias_textuales/f22_fragment.py
"""

import json

import pandas as pd

from common import OUT_DIR, paths
from src.canonical.build import (SCHEMA_VERSION, confusion_counts, f1_macro_from_confusion,
                                 resolve_derived, validate)
from src.predictions.validate import validate_predictions

PRED = 'results/predictions/referencia'
BETO_CSV = f'{PRED}/text__beto__finetune__seed42.csv'
LOGREG_CSV = f'{PRED}/emb768__logreg__none__seed42.csv'
F04_FRAGMENT = 'results/f04_cifras_canonicas/canonical_fragment.json'
SUPPORTS = {'val': 'support.val', 'test': 'support.test'}
DECISION = 'y_pred = 1[p_dep > 0.5]; f1_macro_from_confusion (= sklearn f1_score(average=\'macro\'))'


def per_split(path, splits):
    df = validate_predictions(paths.REPO / path)
    value, confusion = {'val': None, 'test': None}, {}
    for s in splits:
        sel = df[df['split'] == s]
        cm = confusion_counts(sel['y_true'], (sel['p_dep'] > 0.5).astype(int))
        value[s], confusion[s] = f1_macro_from_confusion(cm), cm
    return value, confusion


def main():
    beto_v, beto_cm = per_split(BETO_CSV, ('val', 'test'))
    lr_v, lr_cm = per_split(LOGREG_CSV, ('val',))  # test no se toca (§4)
    selection = pd.read_csv(OUT_DIR / 'seleccion_C.csv')
    C = float(selection.loc[selection['selected'], 'C'].iloc[0])

    numbers = {
        'ref.beto_exp41.f1_macro': {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
            'value': beto_v, 'confusion': beto_cm, 'supports_ref': dict(SUPPORTS),
            'evaluator': ('trainer_head: cabeza de clasificación de checkpoint-342 (BertForSequenceClassification), '
                          'CPU fp32, max_length=128 con padding fijo, p_dep = softmax(logits)[:, 1]; ' + DECISION),
            'source': {'path': BETO_CSV, 'locator': 'split val y test'},
            'supersedes': 'ref.beto_exp41.f1_macro',
            'note': ('Sustituye la cifra reportada de TT-I (0.7586, inalcanzable con soportes 70/44). '
                     'Recalculada en F2.2 desde el checkpoint; ver docs/especificacion_f22.md §6-§7.'),
        },
        'ref.emb768_logreg.f1_macro': {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'verificado',
            'value': lr_v, 'confusion': lr_cm, 'supports_ref': dict(SUPPORTS),
            'null_reason': {'test': 'bloqueado hasta la Fase 5 (docs/especificacion_f22.md §4); '
                                    'las predicciones de test están archivadas'},
            'evaluator': (f'logreg_on_embeddings: StandardScaler (train) + LogisticRegression L2 lbfgs, '
                          f'sin pesos de clase, C = {C:.6g} elegido por F1-Val sobre EMB_41; ' + DECISION),
            'source': {'path': LOGREG_CSV, 'locator': 'split val'},
            'note': 'Mismo tipo de evaluador que Exp 2.x y 3.x; separa la cabeza de BETO de la representación.',
        },
        'ref.delta_ni.f1_macro': {
            'kind': 'metric', 'metric': 'f1_macro', 'supports_ref': dict(SUPPORTS),
            'evaluator': 'δ = F1_BETO − F1_Exp1.2 (TF-IDF lematizado), margen de no inferioridad (plan v2 §3.2)',
            'source': {'path': 'docs/especificacion_f22.md', 'locator': '§5.2'},
            'depends_on': ['ref.beto_exp41.f1_macro', 'ref.tfidf_lemma.f1_macro'], 'derivation': 'diff',
        },
        'witcom.delta_beto_gs_resnet18.f1_macro': {
            'kind': 'metric', 'metric': 'f1_macro', 'supports_ref': dict(SUPPORTS),
            'evaluator': 'Δ exploratorio de WITCOM = F1_BETO − media de Gray-Scott + ResNet-18 (5 semillas)',
            'source': {'path': 'docs/especificacion_f22.md', 'locator': '§5.2'},
            'depends_on': ['ref.beto_exp41.f1_macro', 'witcom.gs_resnet18.f1_macro_mean'], 'derivation': 'diff',
        },
    }
    # Las derivadas de un fragmento declaran value/status/null_reason; build.py verifica
    # que coincidan con lo que calcula. Se calculan aquí con el mismo resolve_derived
    deps = json.loads((paths.REPO / F04_FRAGMENT).read_text(encoding='utf-8'))['numbers']
    pool = {k: json.loads(json.dumps(v)) for k, v in numbers.items()}
    pool.update({k: json.loads(json.dumps(deps[k])) for k in ('ref.tfidf_lemma.f1_macro',
                                                              'witcom.gs_resnet18.f1_macro_mean')})
    resolve_derived(pool)
    for k, e in numbers.items():
        if 'depends_on' in e:
            e.update({f: pool[k][f] for f in ('value', 'status', 'null_reason') if f in pool[k]})

    frag = {'schema_version': SCHEMA_VERSION, 'numbers': numbers}
    validate(frag, fragment=True)
    out = OUT_DIR / 'canonical_fragment.json'
    out.write_text(json.dumps(frag, sort_keys=True, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')
    print(f'escrito {out.relative_to(paths.REPO).as_posix()}')
    print(f'  BETO F1 val {beto_v["val"]:.6f} test {beto_v["test"]:.6f}; LogReg-768D F1 val {lr_v["val"]:.6f}')
    for k in ('ref.delta_ni.f1_macro', 'witcom.delta_beto_gs_resnet18.f1_macro'):
        print(f'  {k}: {numbers[k]["value"]} ({numbers[k]["status"]})')


if __name__ == '__main__':
    main()
