"""
analisis_error_texto.py
───────────────────────
Análisis de error por texto (diapositiva «Análisis de error II», Anexo A18).

Es un análisis descriptivo sobre predicciones de prueba que ya existen: no
entrena, no corre inferencia y no modifica results/predictions/ ni
results/f5_contraste/.

Por técnica (Gray-Scott, Perlin, Lorenz):
  a) Promedia p_dep de las 5 semillas del finalista (fase final, F3.5);
     pred_vis = 1[media > 0.5].
  b) pred_beto = 1[p_dep_BETO > 0.5] (Exp 4.1, F2.2).
  c) Grupo por texto: Conservado (BETO ✓, visual ✓), Perdido en la imagen
     (BETO ✓, visual ✗), Recuperado (BETO ✗, visual ✓), Error heredado (✗, ✗).
  d) Por grupo: n, mediana de n_tokens_beto, % truncados, % depresivo,
     FN/FP y textos con términos de pandemia.

Insumos (solo lectura):
  results/predictions/final/<finalista>__seed{42,7,123,2024,99}.csv
  results/predictions/referencia/text__beto__finetune__seed42.csv
  meta/sample_index.csv            (n_tokens_beto, truncated_128 de F4.5)
  TT_DATA_ROOT/processed/test.csv  (texto original, solo para pandemia)
  results/canonical_numbers.json   (verificaciones)

Uso (desde la raíz del repo):
    python results/analisis_error_texto/analisis_error_texto.py

Termina con código 1 y no escribe nada si falla una verificación obligatoria.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from src import paths  # noqa: E402
from src.predictions import load_predictions  # noqa: E402
from src.training.umap_io import sha256  # noqa: E402

OUT = paths.RESULTS_DIR / 'analisis_error_texto'
THRESHOLD = 0.5
SEEDS = [42, 7, 123, 2024, 99]
FINALISTS = {  # technique: (arch, strategy), F3.5
    'gray_scott': ('efficientnet_b0', 'finetune'),
    'perlin': ('vit_b16', 'frozen'),
    'lorenz': ('vit_b16', 'frozen'),
}
TECH_LABEL = {'gray_scott': 'Gray-Scott', 'perlin': 'Perlin', 'lorenz': 'Lorenz'}
TECH_SHORT = {'gray_scott': 'GS', 'perlin': 'Perlin', 'lorenz': 'Lorenz'}
BETO_FILE = paths.RESULTS_DIR / 'predictions' / 'referencia' / 'text__beto__finetune__seed42.csv'
GROUPS = ['Conservado', 'Perdido en la imagen', 'Recuperado', 'Error heredado']

# Cifras de control dadas en la tarea
EXPECTED_VIS_ERRORS = {'gray_scott': 30, 'perlin': 34, 'lorenz': 28}  # Tabla 84
EXPECTED_COMMON_ERRORS = {'gray_scott': 27, 'perlin': 25, 'lorenz': 23}  # comunes a los 4 modelos
EXPECTED_BETO_CM = [[54, 16], [16, 28]]
EXPECTED_BETO_ERRORS = 32
CORPUS_TRUNC_PCT = 59.1  # build_sample_index.check_truncation

# ── Términos de pandemia del Módulo 1 ──────────────────────────────────────
# Copia literal de PANDEMIC_TERMS en el repo del Módulo 1
# (github.com/Santiago-Soria/proyecto-transformacion-texto-imagen),
# src/features/pandemic_stopwords.py, commit 8076797048e20d22b5f9a67051c642f04ca3e4f8,
# SHA-256 del archivo af23f8094329f37b5e64ae59204d287cd883a772315a79145256fd4af41ff460.
# Lo usa notebooks/4_analisis_pandemia_bias.ipynb (count_pandemic_terms).
PANDEMIC_SOURCE = {
    'repo': 'proyecto-transformacion-texto-imagen',
    'path': 'src/features/pandemic_stopwords.py',
    'commit': '8076797048e20d22b5f9a67051c642f04ca3e4f8',
    'sha256': 'af23f8094329f37b5e64ae59204d287cd883a772315a79145256fd4af41ff460',
}
PANDEMIC_TERMS = [
    "pandemia", "cuarentena", "covid", "virus", "confinamiento",
    "encierro", "contingencia",
    "coronavirus", "sars", "covd", "covid19", "covid-19",
    "aislamiento", "distanciamiento", "sanitario", "sanitaria",
    "mascarilla", "cubrebocas", "vacuna", "vacunacion",
    "epidemia", "brote", "contagio", "contagios", "infectado",
    "positivo", "negativo",
    "hospital", "icu", "uci", "intubado", "ventilador",
    "nueva normalidad", "fase", "semaforo",
    "sana distancia", "quedateencasa", "quédate en casa",
    "ssa", "imss", "issste", "secretaria de salud",
]
# El mismo patrón que _PANDEMIC_PATTERN del Módulo 1 (límites de palabra, IGNORECASE),
# aplicado sobre el texto en minúsculas
PANDEMIC_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in PANDEMIC_TERMS) + r')\b', flags=re.IGNORECASE)


class Stop(RuntimeError):
    pass


def has_pandemic(text):
    return PANDEMIC_PATTERN.search(text.lower()) is not None


def f1_macro(y, p):
    return f1_score(y, p, average='macro', labels=[0, 1])


def canonical():
    with open(paths.RESULTS_DIR / 'canonical_numbers.json', encoding='utf-8') as f:
        return json.load(f)['numbers']


# ── Carga ──────────────────────────────────────────────────────────────────

def load_visual(technique):
    """p_dep de prueba de las 5 semillas del finalista → DataFrame ancho por sample_id."""
    arch, strategy = FINALISTS[technique]
    pattern = paths.RESULTS_DIR / 'predictions' / 'final' / f'{technique}__{arch}__{strategy}__seed*.csv'
    files = sorted(p for p in pattern.parent.glob(pattern.name))
    if len(files) != len(SEEDS):
        raise Stop(f'{technique}: {len(files)} archivos finales; se esperan {len(SEEDS)}')
    df = load_predictions(str(pattern))
    if set(df['fase']) != {'final'}:
        raise Stop(f'{technique}: fases {set(df["fase"])}')
    df = df[df['split'] == 'test']
    if sorted(df['seed'].unique()) != sorted(SEEDS):
        raise Stop(f'{technique}: semillas {sorted(df["seed"].unique())}')
    for seed, g in df.groupby('seed'):
        if len(g) != paths.EXPECTED_SIZES['test'] or g['sample_id'].duplicated().any():
            raise Stop(f'{technique} seed {seed}: {len(g)} filas de test')
    if not (df['arch'].eq(arch).all() and df['strategy'].eq(strategy).all()):
        raise Stop(f'{technique}: arch/strategy inesperados')
    wide = df.pivot(index='sample_id', columns='seed', values='p_dep')
    if wide.isna().any().any():
        raise Stop(f'{technique}: los sample_id no coinciden entre semillas')
    y = df.groupby('sample_id')['y_true'].agg(['min', 'max'])
    if (y['min'] != y['max']).any():
        raise Stop(f'{technique}: y_true difiere entre semillas')
    out = pd.DataFrame({'p_vis_media': wide[SEEDS].mean(axis=1), 'y_true_vis': y['min']})
    return out, files


def load_beto():
    df = load_predictions(str(BETO_FILE))
    df = df[df['split'] == 'test']
    if len(df) != paths.EXPECTED_SIZES['test'] or df['sample_id'].duplicated().any():
        raise Stop(f'BETO: {len(df)} filas de test')
    return df.set_index('sample_id')[['p_dep', 'y_true']].rename(columns={'p_dep': 'p_beto', 'y_true': 'y_true_beto'})


def load_index_and_text():
    """n_tokens_beto y truncated_128 de meta/sample_index.csv (los mismos que lee F4.5)
    y bandera de pandemia a partir del texto original de processed/test.csv."""
    idx = pd.read_csv(paths.SAMPLE_INDEX)
    idx = idx[idx['split'] == 'test'].copy()
    if len(idx) != paths.EXPECTED_SIZES['test']:
        raise Stop(f'sample_index: {len(idx)} filas de test')
    # Misma conversión que results/f45_longitud/f45_longitud.py::features
    tr = idx['truncated_128'].map(lambda v: str(v).strip().lower()).map({'false': 0, 'true': 1})
    if tr.isna().any():
        raise Stop('truncated_128 fuera de {False, True}')
    idx['truncated_128'] = tr.astype(int)
    idx['n_tokens_beto'] = idx['n_tokens_beto'].astype(int)

    text = pd.read_csv(paths.csv_path('test'), usecols=['id', 'text']).reset_index(drop=True)
    if len(text) != len(idx):
        raise Stop(f'processed/test.csv: {len(text)} filas')
    idx = idx.sort_values('idx').reset_index(drop=True)
    if not (idx['idx'].to_numpy() == np.arange(len(text))).all():
        raise Stop('idx de prueba no es 0..113')
    if not (idx['id'].to_numpy() == text['id'].to_numpy()).all():
        raise Stop('id de processed/test.csv no coincide con sample_index')
    h = text['text'].map(lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest())
    if not (h.to_numpy() == idx['text_sha256'].to_numpy()).all():
        raise Stop('el texto de processed/test.csv no coincide con text_sha256 del índice')
    idx['pandemia'] = text['text'].map(has_pandemic).astype(int).to_numpy()
    return idx.set_index('sample_id')[['y_true', 'n_tokens_beto', 'truncated_128', 'pandemia']]


# ── Cálculo ────────────────────────────────────────────────────────────────

def assign_group(ok_beto, ok_vis):
    if ok_beto and ok_vis:
        return 'Conservado'
    if ok_beto:
        return 'Perdido en la imagen'
    if ok_vis:
        return 'Recuperado'
    return 'Error heredado'


def summarize(g, err_col):
    """Campos d) para un subconjunto. err_col: columna de acierto cuyos errores dan FN/FP."""
    n = len(g)
    wrong = g[~g[err_col]] if n else g
    return {
        'n': n,
        'mediana_n_tokens_beto': float(g['n_tokens_beto'].median()) if n else np.nan,
        'pct_truncados': 100 * g['truncated_128'].mean() if n else np.nan,
        'pct_depresivo': 100 * g['y_true'].mean() if n else np.nan,
        'errores_fuente': {'ok_vis': 'visual', 'ok_beto': 'BETO'}[err_col],
        'n_errores': len(wrong),
        'fn': int((wrong['y_true'] == 1).sum()),
        'fp': int((wrong['y_true'] == 0).sum()),
        'n_pandemia': int(g['pandemia'].sum()),
        'pct_pandemia': 100 * g['pandemia'].mean() if n else np.nan,
    }


ERR_SOURCE = {'Conservado': 'ok_vis', 'Perdido en la imagen': 'ok_vis',
              'Recuperado': 'ok_beto', 'Error heredado': 'ok_vis'}


def representatives(g, k=2):
    """k sample_id más cercanos a la mediana de n_tokens_beto del grupo (desempate por sample_id)."""
    if g.empty:
        return []
    d = (g['n_tokens_beto'] - g['n_tokens_beto'].median()).abs()
    return (g.assign(_d=d).reset_index().sort_values(['_d', 'sample_id'])['sample_id'].head(k).tolist())


def main():
    can = canonical()
    beto = load_beto()
    meta = load_index_and_text()
    if not (beto['y_true_beto'] == meta['y_true'].reindex(beto.index)).all():
        raise Stop('y_true de BETO difiere del índice')

    checks = []  # (nombre, obligatoria, ok, detalle)
    inputs = [BETO_FILE, paths.SAMPLE_INDEX, paths.csv_path('test'), paths.RESULTS_DIR / 'canonical_numbers.json']

    # BETO
    yb = meta['y_true'].reindex(beto.index)
    pb = (beto['p_beto'] > THRESHOLD).astype(int)
    cm_b = confusion_matrix(yb, pb, labels=[0, 1]).tolist()
    f1_b = f1_macro(yb, pb)
    f1_b_can = can['ref.beto_exp41.f1_macro']['value']['test']
    ok = cm_b == EXPECTED_BETO_CM and round(f1_b, 4) == 0.7039 and abs(f1_b - f1_b_can) < 1e-12
    checks.append(('Matriz y F1-Macro de BETO', True, ok,
                   f'matriz {cm_b} (esperada {EXPECTED_BETO_CM}); F1-Macro {f1_b:.4f} '
                   f'(ref.beto_exp41.f1_macro = {f1_b_can:.4f})'))

    rows, frames, matrices, reps, vis_f1 = [], [], {}, {}, {}
    for t in FINALISTS:
        vis, files = load_visual(t)
        inputs += files
        df = meta.join(vis, how='inner').join(beto, how='inner')
        if len(df) != paths.EXPECTED_SIZES['test'] or not (df['y_true'] == df['y_true_vis']).all():
            raise Stop(f'{t}: unión con {len(df)} filas o y_true inconsistente')
        df['pred_vis'] = (df['p_vis_media'] > THRESHOLD).astype(int)
        df['pred_beto'] = (df['p_beto'] > THRESHOLD).astype(int)
        df['ok_vis'] = df['pred_vis'] == df['y_true']
        df['ok_beto'] = df['pred_beto'] == df['y_true']
        df['grupo'] = [assign_group(b, v) for b, v in zip(df['ok_beto'], df['ok_vis'])]
        df['technique'] = t

        counts = df['grupo'].value_counts().reindex(GROUPS, fill_value=0)
        matrices[t] = counts
        vis_err = int((~df['ok_vis']).sum())
        n_her = int(counts['Error heredado'])

        checks.append((f'{TECH_LABEL[t]}: Perdido + Heredado = errores visuales (Tabla 84)', True,
                       counts['Perdido en la imagen'] + n_her == EXPECTED_VIS_ERRORS[t] == vis_err,
                       f"{counts['Perdido en la imagen']} + {n_her} = {counts['Perdido en la imagen'] + n_her} "
                       f'(esperado {EXPECTED_VIS_ERRORS[t]})'))
        checks.append((f'{TECH_LABEL[t]}: Recuperado + Heredado = errores de BETO', True,
                       counts['Recuperado'] + n_her == EXPECTED_BETO_ERRORS,
                       f"{counts['Recuperado']} + {n_her} = {counts['Recuperado'] + n_her} "
                       f'(esperado {EXPECTED_BETO_ERRORS})'))
        f1_v = f1_macro(df['y_true'], df['pred_vis'])
        f1_can = can[f'f35.{t}.finalista.f1_macro_mean']['value']['test']
        vis_f1[t] = (f1_v, f1_can)
        checks.append((f'{TECH_LABEL[t]}: F1-Macro de la media de 5 semillas frente a f35', False, True,
                       f'media de p_dep → F1 {f1_v:.4f}; f35.{t}.finalista.f1_macro_mean '
                       f'(media de los F1 por semilla) = {f1_can:.4f}'))
        checks.append((f'{TECH_LABEL[t]}: Heredado ≥ errores comunes a los 4 modelos', True,
                       n_her >= EXPECTED_COMMON_ERRORS[t],
                       f'{n_her} ≥ {EXPECTED_COMMON_ERRORS[t]}'))
        checks.append((f'{TECH_LABEL[t]}: los cuatro grupos suman 114', True,
                       int(counts.sum()) == paths.EXPECTED_SIZES['test'], f'{int(counts.sum())}'))

        for gname in GROUPS:
            g = df[df['grupo'] == gname]
            rows.append({'technique': t, 'grupo': gname, **summarize(g, ERR_SOURCE[gname])})
        rows.append({'technique': t, 'grupo': 'Global (prueba)', **summarize(df, 'ok_vis')})
        if t == 'gray_scott':
            reps = {gname: representatives(df[df['grupo'] == gname])
                    for gname in ['Error heredado', 'Perdido en la imagen', 'Recuperado']}
        frames.append(df.reset_index())

    trunc_global = 100 * meta['truncated_128'].mean()
    checks.append(('% truncados en el global de prueba (informativo)', False, True,
                   f'{trunc_global:.1f} % en prueba frente a {CORPUS_TRUNC_PCT} % del corpus completo'))

    failed = [c for c in checks if c[1] and not c[2]]
    for name, mand, ok, det in checks:
        print(f"[{'OK' if ok else 'FALLA'}{'' if mand else ', informativa'}] {name}: {det}")
    if failed:
        raise Stop(f'{len(failed)} verificaciones obligatorias fallaron; no se escriben entregables')

    write_outputs(frames, rows, matrices, reps, checks, vis_f1, inputs)


# ── Salida ─────────────────────────────────────────────────────────────────

def fmt(x, nd=1):
    return '—' if pd.isna(x) else f'{x:.{nd}f}'


def write_outputs(frames, rows, matrices, reps, checks, vis_f1, inputs):
    OUT.mkdir(parents=True, exist_ok=True)
    for p in ['grupos_por_texto.csv', 'resumen_grupos.csv', 'resumen.md', 'insumos.json']:
        paths.assert_outside_data_root(OUT / p)

    per_text = pd.concat(frames, ignore_index=True)
    cols = ['sample_id', 'technique', 'y_true', 'p_vis_media', 'p_beto', 'grupo',
            'n_tokens_beto', 'truncated_128', 'pandemia']
    per_text[cols].to_csv(OUT / 'grupos_por_texto.csv', index=False, float_format='%.17g', lineterminator='\n')

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'resumen_grupos.csv', index=False, float_format='%.4f', lineterminator='\n')

    rel = lambda p: Path(p).resolve().relative_to(paths.REPO).as_posix() if paths.REPO in Path(p).resolve().parents \
        else 'TT_DATA_ROOT/' + Path(p).resolve().relative_to(paths.data_root().resolve()).as_posix()
    manifest = {rel(p): sha256(p) for p in inputs}
    manifest['pandemic_stopwords.py (Módulo 1)'] = PANDEMIC_SOURCE
    with open(OUT / 'insumos.json', 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write('\n')

    T = list(FINALISTS)
    cell = lambda g: ' / '.join(str(int(matrices[t][g])) for t in T)
    L = []
    L.append('# Análisis de error por texto (prueba, n = 114)\n')
    L.append('Análisis descriptivo sobre predicciones ya existentes. Visual = media de p_dep de las 5 semillas '
             'del finalista (fase `final`, F3.5): Gray-Scott efficientnet_b0 finetune, Perlin vit_b16 frozen y '
             'Lorenz vit_b16 frozen. BETO = Exp 4.1 (F2.2). Regla de decisión: p_dep > 0.5.\n')
    L.append('## Matriz 2×2 por técnica (GS / Perlin / Lorenz)\n')
    L.append('| | Visual acierta | Visual falla |')
    L.append('|---|---|---|')
    L.append(f"| **BETO acierta** | Conservado: {cell('Conservado')} | Perdido en la imagen: {cell('Perdido en la imagen')} |")
    L.append(f"| **BETO falla** | Recuperado: {cell('Recuperado')} | Error heredado: {cell('Error heredado')} |")
    L.append('')
    L.append('## Caracterización por grupo\n')
    L.append('FN/FP: entre los textos mal clasificados del grupo, cuántos son depresivos (FN) y cuántos no '
             'depresivos (FP). En Perdido, Heredado y Global se cuentan los errores del visual; en Recuperado, '
             'los de BETO. Pandemia: textos con al menos un término de la lista del Módulo 1 '
             '(regex con límites de palabra sobre el texto en minúsculas).\n')
    L.append('| Técnica | Grupo | n | Mediana de tokens BETO | % truncados (>128) | % depresivo | FN / FP | Pandemia: n (%) |')
    L.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        fnfp = '—' if r['grupo'] == 'Conservado' else f"{r['fn']} / {r['fp']}"
        L.append(f"| {TECH_LABEL[r['technique']]} | {r['grupo']} | {r['n']} | {fmt(r['mediana_n_tokens_beto'], 1)} | "
                 f"{fmt(r['pct_truncados'])} | {fmt(r['pct_depresivo'])} | {fnfp} | "
                 f"{r['n_pandemia']} ({fmt(r['pct_pandemia'])}) |")
    L.append('')
    L.append('## Verificaciones\n')
    for name, mand, ok, det in checks:
        L.append(f"- **{'OK' if ok else 'FALLA'}**{'' if mand else ' (informativa)'}: {name}. {det}.")
    L.append('')
    L.append('## Textos representativos de Gray-Scott\n')
    L.append('Los 2 sample_id más cercanos a la mediana de tokens del grupo (desempate por sample_id). '
             'No se copian los textos.\n')
    for gname, ids in reps.items():
        L.append(f"- {gname}: {', '.join(f'`{i}`' for i in ids) if ids else '(grupo vacío)'}")
    L.append('')
    (OUT / 'resumen.md').write_text('\n'.join(L), encoding='utf-8', newline='\n')
    print(f'escrito en {OUT.relative_to(paths.REPO).as_posix()}/')


if __name__ == '__main__':
    try:
        main()
    except Stop as e:
        print(f'DETENIDO: {e}', file=sys.stderr)
        sys.exit(1)
