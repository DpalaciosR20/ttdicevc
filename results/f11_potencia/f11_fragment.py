"""
f11_fragment.py
───────────────
Escribe results/f11_potencia/canonical_fragment.json a partir de las salidas
de potencia.py (docs/especificacion_f11.md §2 y §5).

  f11.delta_margen.r0_50 / r0_75 / r0_80
        δ = (1 − r)(F1_BETO − 0.5), con F1_BETO = ref.beto_exp41 (test)
  f11.f1_min_detectable.{a,b}.r0_50 / r0_75 / r0_80
        0.5 + ρ_min (F1_BETO − 0.5) en el escenario central (discordancia p50,
        DE entre semillas 0.010); nulo si la malla no alcanza P(V1) ≥ 0.80

Son cifras de diseño, no F1 observados: tests/test_canonical_numbers.py las
exime de la prueba de alcanzabilidad por su prefijo f11.

Uso (desde la raíz del repo):
    python results/f11_potencia/f11_fragment.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.canonical.build import SCHEMA_VERSION, validate  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
R_VALUES = (0.50, 0.75, 0.80)
CENTRAL = ('p50', 0.010)
SUPPORTS = {'val': 'support.val', 'test': 'support.test'}
NULL_VAL = 'cifra de diseño definida sobre F1_BETO de prueba (especificacion_f11 §2); sin término en val'


def rkey(r):
    return f'r{r:.2f}'.replace('.', '_')


def main():
    numbers = json.loads((REPO / 'results' / 'canonical_numbers.json').read_text(encoding='utf-8'))['numbers']
    f1_beto = numbers['ref.beto_exp41.f1_macro']['value']['test']
    rmin = pd.read_csv(OUT_DIR / 'retencion_minima.csv')
    central = rmin[(rmin['disc_etiqueta'] == CENTRAL[0]) & np.isclose(rmin['de_objetivo'], CENTRAL[1])]

    frag = {}
    for r in R_VALUES:
        frag[f'f11.delta_margen.{rkey(r)}'] = {
            'kind': 'metric', 'metric': 'f1_macro', 'status': 'extraido',
            'value': {'val': None, 'test': (1 - r) * (f1_beto - 0.5)},
            'null_reason': {'val': NULL_VAL},
            'evaluator': f'δ = (1 − r)(F1_BETO − 0.5) con r = {r:.2f} y F1_BETO = ref.beto_exp41.f1_macro (test); '
                         'margen de la formulación A (margen fijo)',
            'source': {'path': 'docs/especificacion_f11.md', 'locator': '§2'},
            'supports_ref': SUPPORTS,
            'note': 'Cifra de diseño (margen candidato para F1.2), no un F1 observado.',
        }
        for form in ('A', 'B'):
            row = central[(central['formulacion'] == form) & np.isclose(central['r'], r)].iloc[0]
            value = None if pd.isna(row['f1_min']) else float(row['f1_min'])
            entry = {
                'kind': 'metric', 'metric': 'f1_macro', 'status': 'extraido',
                'value': {'val': None, 'test': value},
                'null_reason': {'val': NULL_VAL},
                'evaluator': f'0.5 + ρ_min (F1_BETO − 0.5), ρ_min = menor retención con P(V1) ≥ 0.80 por '
                             f'interpolación lineal en la malla de S; formulación {form}, r = {r:.2f}, escenario '
                             f'central (discordancia p50, DE 0.010); 1,000 simulaciones, B = 2,000, Holm sobre 3 técnicas',
                'source': {'path': 'results/f11_potencia/retencion_minima.csv',
                           'locator': f'formulacion={form}, r={r:.2f}, disc_etiqueta=p50, de_objetivo=0.010'},
                'supports_ref': SUPPORTS,
                'note': 'Cifra de diseño (estimación por simulación), no un F1 observado.',
            }
            if value is None:
                entry['null_reason']['test'] = 'P(V1) ≥ 0.80 no se alcanza en la malla (ρ ≤ 1.10)'
            frag[f'f11.f1_min_detectable.{form.lower()}.{rkey(r)}'] = entry

    doc = {'schema_version': SCHEMA_VERSION, 'numbers': frag}
    validate(doc, fragment=True)
    out = OUT_DIR / 'canonical_fragment.json'
    out.write_text(json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')
    print(f'{out.relative_to(REPO)}: {len(frag)} cifras')


if __name__ == '__main__':
    main()
