"""
build_sample_index.py
─────────────────────
Construye meta/sample_index.csv, el índice maestro de muestras
(docs/especificacion_predicciones.md §2), y ejecuta sus cinco verificaciones
de construcción. El log se escribe en results/f15_formato/build_sample_index.log.

Correspondencia (docs/datos.md §8): la fila i de cada split es la misma
muestra en processed/{split}.csv, EMB_41, umap_params.pkl y la imagen
{split}_{i:04d}; sample_id = {split}_{i:04d}.

Datos personales (datos.md §4): personal_key solo se usa en memoria para la
verificación 4 y el log registra el resultado agregado. text solo produce su
SHA-256 y su número de tokens.

El índice solo se escribe si pasan las verificaciones 1-4. La 5 (proporción de
truncamiento) se reporta y no detiene la escritura: si no reproduce el 59.1 %
registrado, la definición NO se ajusta.

Uso (desde la raíz del repo):
    python -m src.build_sample_index
"""

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import pandas as pd

from src import paths
from src.training.umap_io import load_umap_arrays, sha256

COLUMNS = ['sample_id', 'split', 'idx', 'id', 'y_true', 'text_sha256', 'n_tokens_beto', 'truncated_128',
           'img_perlin', 'img_gray_scott', 'img_lorenz']
MAX_LENGTH = 128
SUPPORTS = {'val': {0: 70, 1: 44}, 'test': {0: 70, 1: 44}}
# Proporciones registradas de truncated_128 (%), global y por clase
TRUNC_REF = {'global': 59.1, 0: 62.6, 1: 53.5}


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg=''):
        print(msg)
        self.lines.append(msg)

    def write(self, path):
        paths.assert_outside_data_root(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text('\n'.join(self.lines) + '\n', encoding='utf-8')


def load_sources(log):
    """CSV por split (solo id, personal_key, text, manual_classification),
    etiquetas de EMB_41 y de umap_params.pkl."""
    csvs = {}
    for s in paths.SPLITS:
        df = pd.read_csv(paths.csv_path(s), usecols=['id', 'personal_key', 'text', 'manual_classification'])
        csvs[s] = df.reset_index(drop=True)
    emb = joblib.load(paths.emb_41_path())
    emb_labels = {s: emb[f'y_{s}'] for s in paths.SPLITS}
    umap_labels = {s: a['labels'] for s, a in load_umap_arrays(paths.umap_params_path()).items()}
    log(f"EMB_41 model_name={emb.get('model_name')!r} checkpoint={emb.get('checkpoint')!r}")
    return csvs, emb_labels, umap_labels


def load_tokenizer(log):
    import transformers
    from transformers import AutoTokenizer
    # Contar sin truncar dispara un aviso por cada texto > 512 tokens; aquí es lo esperado
    transformers.logging.set_verbosity_error()
    tok =AutoTokenizer.from_pretrained(str(paths.ckpt_41_path()))
    log(f'tokenizador: {type(tok).__name__} desde CKPT_41 (transformers {transformers.__version__}), '
        f'model_max_length={tok.model_max_length}')
    return tok


def build_index(csvs, tok):
    rows = []
    for s in paths.SPLITS:
        df = csvs[s]
        for i, r in enumerate(df.itertuples(index=False)):
            y = int(r.manual_classification)
            # Sin truncar: [CLS] + tokens + [SEP]
            n_tok = len(tok(r.text, truncation=False, add_special_tokens=True)['input_ids'])
            rows.append({
                'sample_id': paths.sample_id(s, i), 'split': s, 'idx': i, 'id': int(r.id), 'y_true': y,
                'text_sha256': hashlib.sha256(r.text.encode('utf-8')).hexdigest(),
                'n_tokens_beto': n_tok, 'truncated_128': n_tok > MAX_LENGTH,
                **{f'img_{t}': paths.image_name(t, s, i, y) for t in paths.TECHNIQUES},
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def check_counts(index):
    """1. 1,136 filas; 908/114/114; soportes 70/44 en val y test."""
    problems = []
    n = index['split'].value_counts().to_dict()
    if len(index) != sum(paths.EXPECTED_SIZES.values()):
        problems.append(f'{len(index)} filas')
    for s, k in paths.EXPECTED_SIZES.items():
        if n.get(s) != k:
            problems.append(f'{s}: {n.get(s)} filas, se esperan {k}')
    for s, sup in SUPPORTS.items():
        got = index.loc[index['split'] == s, 'y_true'].value_counts().to_dict()
        if got != sup:
            problems.append(f'{s}: soportes {got}, se esperan {sup}')
    detail = f"{len(index)} filas; " + ' / '.join(f'{s}={n.get(s)}' for s in paths.SPLITS) + '; ' + \
        '; '.join(f"{s} 0/1 = {(index.loc[index['split'] == s, 'y_true'] == 0).sum()}/"
                  f"{(index.loc[index['split'] == s, 'y_true'] == 1).sum()}" for s in SUPPORTS)
    return problems, detail


def check_labels(index, emb_labels, umap_labels):
    """2. Etiqueta igual en CSV, EMB_41, umap_params.pkl y sufijo label{y} de las 3 imágenes."""
    problems = []
    for s in paths.SPLITS:
        y = index.loc[index['split'] == s, 'y_true'].to_numpy()
        for name, other in (('EMB_41', emb_labels[s]), ('UMAP_PARAMS', umap_labels[s])):
            if len(other) != len(y):
                problems.append(f'{s}: {name} tiene {len(other)} etiquetas, CSV {len(y)}')
            elif (other != y).any():
                problems.append(f'{s}: {name} difiere del CSV en {(other != y).sum()} filas')
    folders = {t: {p.name for p in paths.list_files(paths.images_dir(t))[0]} for t in paths.TECHNIQUES}
    for t in paths.TECHNIQUES:
        missing = index.loc[~index[f'img_{t}'].isin(folders[t]), f'img_{t}']
        if len(missing):
            # El nombre esperado lleva la etiqueta del CSV: si falta, o no existe la
            # imagen o su sufijo label{y} no coincide
            problems.append(f'{t}: {len(missing)} imágenes con el nombre esperado no existen '
                            f'(p. ej. {missing.iloc[0]})')
    return problems, f'{len(index)} filas × (CSV, EMB_41, UMAP_PARAMS, 3 sufijos label{{y}})', folders


def check_inventory(index, folders):
    """3. Cada carpeta de técnica contiene exactamente 1,136 archivos, todos referenciados."""
    problems, parts = [], []
    for t in paths.TECHNIQUES:
        extra = sorted(folders[t] - set(index[f'img_{t}']))
        parts.append(f'{t}={len(folders[t])}')
        if len(folders[t]) != len(index):
            problems.append(f'{t}: {len(folders[t])} archivos, se esperan {len(index)}')
        if extra:
            problems.append(f'{t}: {len(extra)} archivos no referenciados: {", ".join(extra)}')
    return problems, ', '.join(parts)


def check_participants(csvs):
    """4. personal_key única por fila y ningún participante en dos splits (solo agregados)."""
    keys = pd.concat([csvs[s][['personal_key']].assign(split=s) for s in paths.SPLITS], ignore_index=True)
    n_null = int(keys['personal_key'].isna().sum())
    n_unique = keys['personal_key'].nunique()
    shared = int((keys.groupby('personal_key')['split'].nunique() > 1).sum())
    problems = []
    if n_null:
        problems.append(f'{n_null} filas sin personal_key')
    if n_unique != len(keys):
        problems.append(f'{len(keys) - n_unique} filas con personal_key repetida')
    if shared:
        problems.append(f'{shared} participantes en más de un split')
    return problems, f'{n_unique} claves únicas en {len(keys)} filas; participantes en más de un split: {shared}'


def check_truncation(index):
    """5. Proporción de truncated_128 = 59.1 % (No Depresivo 62.6 %, Depresivo 53.5 %)."""
    got = {'global': 100 * index['truncated_128'].mean()}
    for y in (0, 1):
        got[y] = 100 * index.loc[index['y_true'] == y, 'truncated_128'].mean()
    label = {'global': 'global', 0: 'No Depresivo', 1: 'Depresivo'}
    problems = [f'{label[k]}: {got[k]:.1f} % (registrado {ref} %)' for k, ref in TRUNC_REF.items()
                if round(got[k], 1) != ref]
    detail = '; '.join(f'{label[k]} {got[k]:.2f} % (registrado {TRUNC_REF[k]} %)' for k in TRUNC_REF)
    detail += f"; n_tokens_beto mediana={index['n_tokens_beto'].median():.0f}, máx={index['n_tokens_beto'].max()}"
    return problems, detail


def report(log, num, title, problems, detail):
    log(f"[{'OK ' if not problems else 'FALLA'}] {num}. {title}")
    log(f'        {detail}')
    for p in problems:
        log(f'        - {p}')
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(paths.SAMPLE_INDEX))
    ap.add_argument('--log', default=str(paths.RESULTS_DIR / 'f15_formato' / 'build_sample_index.log'))
    args = ap.parse_args()
    for p in (args.out, args.log):
        paths.assert_outside_data_root(p)

    log = Log()
    log(f'TT_DATA_ROOT = {paths.data_root()}')
    for name, p in [('CSV[train]', paths.csv_path('train')), ('CSV[val]', paths.csv_path('val')),
                    ('CSV[test]', paths.csv_path('test')), ('UMAP_PARAMS', paths.umap_params_path()),
                    ('EMB_41', paths.emb_41_path())]:
        log(f'{name:<12} sha256={sha256(p)}')
    csvs, emb_labels, umap_labels = load_sources(log)
    tok = load_tokenizer(log)
    index = build_index(csvs, tok)
    log('')

    ok = []
    problems, detail = check_counts(index)
    ok.append(report(log, 1, 'Conteos y soportes', problems, detail))
    problems, detail, folders = check_labels(index, emb_labels, umap_labels)
    ok.append(report(log, 2, 'Etiquetas coinciden en todas las fuentes', problems, detail))
    problems, detail = check_inventory(index, folders)
    ok.append(report(log, 3, 'Inventario de imágenes', problems, detail))
    problems, detail = check_participants(csvs)
    ok.append(report(log, 4, 'Participantes únicos y sin cruce entre splits', problems, detail))
    problems, detail = check_truncation(index)
    trunc_ok = report(log, 5, 'Proporción de truncated_128', problems, detail)
    log('')

    if all(ok):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(out, index=False, lineterminator='\n')
        log(f'índice escrito: {out.relative_to(paths.REPO)} ({len(index)} filas), sha256={sha256(out)}')
        if not trunc_ok:
            log('AVISO: la verificación 5 no reproduce la proporción registrada; la definición no se ajustó.')
    else:
        log('índice NO escrito: fallan verificaciones de construcción 1-4')
    log.write(args.log)
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
