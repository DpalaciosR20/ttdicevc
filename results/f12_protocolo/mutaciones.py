"""
mutaciones.py
─────────────
Pruebas por mutación de F1.2: introduce, una a la vez, un defecto en
src/protocol/verdict.py, src/selection/select_finalist.py (select_second),
configs/protocolo.json, configs/matriz_experimentos.json o el protocolo, corre
las pruebas que deberían detectarlo y restaura el archivo original.

Una mutación está «detectada» si pytest termina con código distinto de 0.
Al final verifica que todos los archivos quedaron idénticos al original
(SHA-256 de los bytes) y escribe results/f12_protocolo/mutaciones.log.

Uso (desde la raíz del repo, con el entorno del proyecto):
    .venv\\Scripts\\python.exe results/f12_protocolo/mutaciones.py
"""

import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = Path(__file__).resolve().parent / 'mutaciones.log'

VERDICT = 'src/protocol/verdict.py'
SELECT = 'src/selection/select_finalist.py'
PROTO_JSON = 'configs/protocolo.json'
MATRIX_JSON = 'configs/matriz_experimentos.json'
PROTO_DOC = 'docs/protocolo_preregistrado.md'

T_VERDICT = ['tests/test_verdict.py']
T_SELECT = ['tests/test_select_finalist.py']
T_PROTO = ['tests/test_protocolo.py']

# (id, descripción, archivo, original, mutación, pruebas)
MUTATIONS = [
    ('V-1', '§5.5: V2 exige límite superior de θ > 0 (desigualdad invertida)', VERDICT,
     "    if theta_upper < 0:\n        return 'V2'", "    if theta_upper > 0:\n        return 'V2'", T_VERDICT),
    ('V-2', '§5.5: V3 deja de tener precedencia (se evalúa V1 antes que el piso)', VERDICT,
     "    if not beats_floor:\n        return 'V3'\n    if ni_reject:\n        return 'V1'",
     "    if ni_reject:\n        return 'V1'\n    if not beats_floor:\n        return 'V3'", T_VERDICT),
    ('V-3', '§5.4: sin regla de empates (empatadas reciben el α_h del mayor rango)', VERDICT,
     "out[tech] = {'rank': first, 'alpha_h': alpha / (m - first + 1),",
     "out[tech] = {'rank': last, 'alpha_h': alpha / (m - last + 1),", T_VERDICT),
    ('V-4', '§5.4: Holm sin parada escalonada', VERDICT,
     'if not stopped and p <= alpha / (m - k + 1):', 'if p <= alpha / (m - k + 1):', T_VERDICT),
    ('V-5', '§5.4: Holm con niveles de Bonferroni (α/m en todos los rangos)', VERDICT,
     'if not stopped and p <= alpha / (m - k + 1):', 'if not stopped and p <= alpha / m:', T_VERDICT),
    ('V-6', '§5.2: p con < en vez de ≤', VERDICT,
     'p = float(np.mean(reps <= 0))', 'p = float(np.mean(reps < 0))', T_VERDICT),
    ('V-7', '§5.4: intervalo al nivel 1 − α_h en vez de 1 − 2α_h', VERDICT,
     '[alpha_h, 1 - alpha_h]', '[alpha_h / 2, 1 - alpha_h / 2]', T_VERDICT),
    ('V-8', '§5.5: G2 se declara con V4 en vez de V2', VERDICT,
     "    if 'V2' in vs:\n        return 'G2'", "    if 'V4' in vs:\n        return 'G2'", T_VERDICT),
    ('V-9', '§5.4: intervalo con α_h de la familia de superioridad', VERDICT,
     "ci = percentile_interval(arrays[('θ', t)], h_ni[t]['alpha_h'])",
     "ci = percentile_interval(arrays[('θ', t)], h_sup[t]['alpha_h'])", T_VERDICT),
    ('S-1', 'select_second no excluye al finalista', SELECT,
     'rest = table[[k not in fin for k in keys]].reset_index(drop=True)',
     'rest = table.reset_index(drop=True)', T_SELECT),
    ('S-2', 'select_second toma al líder de las restantes, sin la regla de complejidad', SELECT,
     "sec = set(second.loc[second['finalist'], CONFIG_KEY]",
     "sec = set(second.loc[second['leader'], CONFIG_KEY]", T_SELECT),
    ('P-1', 'configs/protocolo.json: r = 0.75 en el contraste principal', PROTO_JSON,
     '"formulacion": "B",\n    "r": 0.5,', '"formulacion": "B",\n    "r": 0.75,', T_PROTO),
    ('P-2', 'configs/protocolo.json: B = 2000', PROTO_JSON,
     '"B": 10000,\n      "rng"', '"B": 2000,\n      "rng"', T_PROTO),
    ('P-3', 'configs/protocolo.json: regla de empates distinta', PROTO_JSON,
     '"ties": "min_alpha_h"', '"ties": "fixed_order"', T_PROTO + T_VERDICT),
    ('P-4', 'configs/protocolo.json: rejilla de C de LogReg con 12 valores', PROTO_JSON,
     '"C": {"log10_low": -4, "log10_high": 2, "n": 13}', '"C": {"log10_low": -4, "log10_high": 2, "n": 12}', T_PROTO),
    ('P-5', 'configs/matriz_experimentos.json: sensibilidad con 3 semillas', MATRIX_JSON,
     '"hyperparams": "configuracion_comun",\n      "seeds": [42, 7, 123, 2024, 99],',
     '"hyperparams": "configuracion_comun",\n      "seeds": [42, 7, 123],', T_PROTO),
    ('D-1', 'Protocolo §13: umbral 0.6020 en vez de 0.6019', PROTO_DOC,
     'con F1_BETO observado | 0.6019 |', 'con F1_BETO observado | 0.6020 |', T_PROTO),
    ('D-2', 'Protocolo §5.4: se elimina la frase de la regla de empates', PROTO_DOC,
     ' Si varias técnicas tienen el mismo valor p, todas reciben el menor α_h de los rangos que ocupan.', '', T_PROTO),
    ('D-3', 'Protocolo §8: variabilidad de BETO ±0.0129', PROTO_DOC, '±0.0128', '±0.0129', T_PROTO),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pytest(tests):
    cmd = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rf', *tests]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = p.stdout.strip().splitlines()
    failed = [re.sub(r'^FAILED ', '', l).split(' - ')[0] for l in out if l.startswith('FAILED ')]
    summary = out[-1] if out else ''
    return p.returncode, summary, failed


def main():
    files = {f: REPO / f for f in {m[2] for m in MUTATIONS}}
    before = {f: digest(p) for f, p in files.items()}
    lines = [f'# Pruebas por mutación de F1.2 — {datetime.now(timezone.utc).isoformat(timespec="seconds")}',
             f'# python {sys.version.split()[0]}', '']

    rc, summary, _ = run_pytest(T_VERDICT + T_SELECT + T_PROTO)
    lines.append(f'Línea base (sin mutación): código {rc} | {summary}')
    if rc != 0:
        raise SystemExit('la línea base no pasa; no se ejecutan mutaciones')
    lines.append('')

    detected = 0
    for mid, desc, rel, old, new, tests in MUTATIONS:
        path = files[rel]
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        eol = '\r\n' if '\r\n' in text else '\n'
        old_e, new_e = old.replace('\n', eol), new.replace('\n', eol)
        count = text.count(old_e)
        if count != 1:
            raise SystemExit(f'{mid}: el fragmento original aparece {count} veces en {rel}')
        try:
            path.write_bytes(text.replace(old_e, new_e).encode('utf-8'))
            rc, summary, failed = run_pytest(tests)
        finally:
            path.write_bytes(raw)
        ok = rc != 0
        detected += ok
        lines.append(f'[{mid}] {"DETECTADA" if ok else "NO DETECTADA"} — {desc}')
        lines.append(f'      archivo: {rel} | pruebas: {" ".join(tests)}')
        lines.append(f'      pytest: código {rc} | {summary}')
        for f in failed:
            lines.append(f'      falla: {f}')
        lines.append('')

    after = {f: digest(p) for f, p in files.items()}
    restored = after == before
    rc, summary, _ = run_pytest(T_VERDICT + T_SELECT + T_PROTO)
    lines.append(f'Mutaciones detectadas: {detected}/{len(MUTATIONS)}')
    lines.append(f'Archivos restaurados (SHA-256 idéntico al original): {"sí" if restored else "NO"}')
    lines.append(f'Línea base tras restaurar: código {rc} | {summary}')
    LOG.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(lines))
    return 0 if restored and rc == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
