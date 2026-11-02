"""
f12_fragment.py
───────────────
Escribe results/f12_protocolo/canonical_fragment.json con las cifras de diseño
del protocolo preregistrado (docs/protocolo_preregistrado.md §4.4, §5.1, §8 y
§13). Ninguna cifra se escribe a mano: todas se calculan aquí.

  f12.umbral_f1_visual
        0.5 + r (F1_BETO − 0.5), con r = contraste.r de configs/protocolo.json
        y F1_BETO = ref.beto_exp41.f1_macro (test). Umbral de F1 visual medio
        para θ = 0 con F1_BETO observado (§5.1, §13).
  f12.piso_mayoritaria.f1_macro
        F1-Macro de prueba del piso que predice siempre la clase 0, a partir de
        los soportes de test de meta/sample_index.csv (§4.4).
  f12.variabilidad_beto
        Diferencia de F1-Val (best_metric) entre las dos corridas nominalmente
        idénticas de Exp 4.1: checkpoint del 10 de marzo − checkpoint del 11 de
        marzo, leída de los logs de F2.2 (§8).

El umbral no puede declararse como derivada formal (depends_on): el esquema
solo admite sum, mean, std_ddof1 y diff. tests/test_protocolo.py lo recalcula.
Son cifras de diseño: tests/test_canonical_numbers.py las exime de la prueba de
alcanzabilidad por su prefijo f12.

Uso (desde la raíz del repo):
    python results/f12_protocolo/f12_fragment.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.canonical.build import SCHEMA_VERSION, f1_macro_from_confusion, validate  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_JSON = REPO / 'configs' / 'protocolo.json'
SAMPLE_INDEX = REPO / 'meta' / 'sample_index.csv'
LOG_10MARZO = REPO / 'results' / 'f22_referencias_textuales' / 'diagnostico_v3' / 'f22_beto_ckpt_10marzo.log'
LOG_11MARZO = REPO / 'results' / 'f22_referencias_textuales' / 'f22_beto.log'
SUPPORTS = {'val': 'support.val', 'test': 'support.test'}
BEST_METRIC_RE = re.compile(r"^V1: OK .*'best_metric': ([0-9.]+)\}\s*$")


def best_metric(path):
    """(número de línea, best_metric) de la única línea «V1: OK» del log."""
    hits = [(i, float(m.group(1))) for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
            if (m := BEST_METRIC_RE.match(line))]
    if len(hits) != 1:
        raise ValueError(f'{path}: se esperaba una línea «V1: OK … best_metric», hay {len(hits)}')
    return hits[0]


def rel(path):
    return path.relative_to(REPO).as_posix()


def main():
    numbers = json.loads((REPO / 'results' / 'canonical_numbers.json').read_text(encoding='utf-8'))['numbers']
    protocol = json.loads(PROTOCOL_JSON.read_text(encoding='utf-8'))
    r, azar = protocol['contraste']['r'], protocol['contraste']['f1_azar']
    f1_beto = numbers['ref.beto_exp41.f1_macro']['value']['test']

    index = pd.read_csv(SAMPLE_INDEX, usecols=['split', 'y_true'])
    y_test = index.loc[index['split'] == 'test', 'y_true']
    majority = protocol['pisos']['majority']['predict']
    conf = {'tp': int(((y_test == 1) & (majority == 1)).sum()), 'fp': int(((y_test == 0) & (majority == 1)).sum()),
            'fn': int(((y_test == 1) & (majority == 0)).sum()), 'tn': int(((y_test == 0) & (majority == 0)).sum())}

    line10, bm10 = best_metric(LOG_10MARZO)
    line11, bm11 = best_metric(LOG_11MARZO)
    if bm11 != numbers['ref.beto_exp41.f1_macro']['value']['val']:
        raise ValueError(f'best_metric del 11 de marzo {bm11} != ref.beto_exp41.f1_macro (val)')

    frag = {
        'f12.umbral_f1_visual': {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'extraido',
            'value': {'val': None, 'test': azar + r * (f1_beto - azar)},
            'null_reason': {'val': 'cifra de diseño definida sobre F1_BETO de prueba (protocolo §5.1); sin término en val'},
            'evaluator': f'F1_azar + r (F1_BETO − F1_azar), con r = contraste.r ({r}) y F1_azar = contraste.f1_azar '
                         f'({azar}) de configs/protocolo.json y F1_BETO = ref.beto_exp41.f1_macro (test)',
            'source': {'path': 'configs/protocolo.json', 'locator': 'contraste.r, contraste.f1_azar'},
            'supports_ref': SUPPORTS,
            'note': 'Cifra de diseño: F1 visual medio para θ = 0 con F1_BETO observado; no es un F1 observado.',
        },
        'f12.piso_mayoritaria.f1_macro': {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'extraido',
            'value': {'val': None, 'test': f1_macro_from_confusion(conf)},
            'null_reason': {'val': 'el protocolo (§4.4) solo fija el piso sobre prueba'},
            'confusion': {'test': conf},
            'evaluator': f'f1_macro_from_confusion con ŷ = {majority} para todas las muestras de test '
                         '(clase mayoritaria), soportes de meta/sample_index.csv',
            'source': {'path': 'meta/sample_index.csv', 'locator': "split == 'test', y_true"},
            'supports_ref': SUPPORTS,
            'note': 'Piso determinista; no depende de ningún modelo.',
        },
        'f12.variabilidad_beto': {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'extraido',
            'value': {'val': bm10 - bm11, 'test': None},
            'null_reason': {'test': 'solo se conoce el F1-Val de la corrida del 10 de marzo (F2.2, H1)'},
            'evaluator': f'best_metric del checkpoint del 10 de marzo ({rel(LOG_10MARZO)}:{line10}) − '
                         f'best_metric del checkpoint del 11 de marzo ({rel(LOG_11MARZO)}:{line11})',
            'source': {'path': rel(LOG_10MARZO), 'locator': f'línea {line10}, best_metric'},
            'supports_ref': SUPPORTS,
            'note': 'Diferencia de F1-Val entre dos corridas nominalmente idénticas de Exp 4.1; '
                    'desplazamiento ± de F1_BETO en el análisis de sensibilidad (protocolo §8).',
        },
    }

    doc = {'schema_version': SCHEMA_VERSION, 'numbers': frag}
    validate(doc, fragment=True)
    out = OUT_DIR / 'canonical_fragment.json'
    out.write_text(json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')
    print(f'{out.relative_to(REPO)}: {len(frag)} cifras')
    for k, v in frag.items():
        print(f'  {k}: {v["value"]}')


if __name__ == '__main__':
    main()
