"""
validate.py
───────────
Validador de archivos de predicciones por muestra
(docs/especificacion_predicciones.md §5). Rechaza un archivo si:

  1. Las columnas, su orden o sus tipos no coinciden con §3.1, hay valores
     faltantes o las filas no siguen el orden de §3.1.
  2. Algún valor está fuera del vocabulario de §3.2 o p_dep no está en [0, 1].
  3. El conjunto de sample_id de un split entregado no es exactamente el del
     índice maestro, o y_true difiere del índice.
  4. Hay claves duplicadas.
  5. El nombre del archivo no coincide con su contenido (technique, arch,
     strategy, seed).
  6. Falta el manifiesto, le falta un campo obligatorio, csv_sha256 no
     coincide o la fase del manifiesto no es la de la carpeta.
  7. Hay filas de test en una fase que no las permite (§3.4). En final y
     control_permutacion, test exige la etiqueta protocolo-v1 en git.

Uso (desde la raíz del repo):
    python -m src.predictions.validate results/predictions/referencia/*.csv
    python -m src.predictions.validate "results/predictions/**/*.csv"

Código de salida 0 si todos los archivos son válidos y 1 si alguno falla.
"""

import argparse
import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import paths
from src.predictions import schema
from src.training.umap_io import sha256

INT_RE = re.compile(r'^-?\d+$')


class PredictionFormatError(ValueError):
    """violations: lista de (regla, mensaje)."""

    def __init__(self, path, violations):
        self.path = str(path)
        self.violations = violations
        super().__init__(f'{path}: ' + '; '.join(f'[{r}] {m}' for r, m in violations))

    @property
    def rules(self):
        return {r for r, _ in self.violations}


def protocol_tag_exists(tag=schema.PROTOCOL_TAG):
    try:
        out = subprocess.check_output(['git', '-C', str(paths.REPO), 'tag', '-l', tag],
                                      stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return False
    return out.strip() == tag


def load_index(index=None):
    """Índice maestro como DataFrame con sample_id, split, y_true."""
    if isinstance(index, pd.DataFrame):
        return index
    return pd.read_csv(index or paths.SAMPLE_INDEX, usecols=['sample_id', 'split', 'y_true'],
                       dtype={'sample_id': str, 'split': str, 'y_true': 'int64'})


def _parse(raw):
    """Regla 1 sobre el CSV leído como texto. Devuelve (DataFrame tipado o None, violaciones)."""
    v = []
    if list(raw.columns) != schema.COLUMNS:
        v.append((1, f'columnas {list(raw.columns)}; se esperan {schema.COLUMNS}'))
        return None, v
    missing = (raw.apply(lambda c: c.str.strip()) == '') | raw.apply(lambda c: c.str.lower().isin(['nan', 'na', 'null']))
    for c in schema.COLUMNS:
        if missing[c].any():
            v.append((1, f'{c}: {int(missing[c].sum())} valores faltantes'))
    if v:
        return None, v
    df = raw.copy()
    for c in schema.INT_COLUMNS:
        bad = ~raw[c].str.match(INT_RE)
        if bad.any():
            v.append((1, f'{c}: valores no enteros (p. ej. {raw.loc[bad, c].iloc[0]!r})'))
        else:
            df[c] = raw[c].astype('int64')
    for c in schema.FLOAT_COLUMNS:
        vals = pd.to_numeric(raw[c], errors='coerce')
        bad = vals.isna() | ~vals.map(lambda x: math.isfinite(x) if pd.notna(x) else False)
        if bad.any():
            v.append((1, f'{c}: valores no numéricos (p. ej. {raw.loc[bad, c].iloc[0]!r})'))
        else:
            df[c] = vals.astype('float64')
    if v:
        return None, v
    order = df['split'].map({s: i for i, s in enumerate(schema.SPLIT_ORDER)})
    if order.notna().all():
        sorted_df = df.assign(_o=order).sort_values(['_o', 'sample_id'], kind='stable')
        if not sorted_df.index.equals(df.index):
            v.append((1, 'filas fuera de orden: se ordenan por split (train, val, test) y luego por sample_id'))
    return df, v


def _check_vocab(df):
    v = []
    for c, allowed in schema.VOCAB.items():
        bad = sorted(set(df[c]) - allowed, key=str)
        if bad:
            v.append((2, f'{c}: valores fuera del vocabulario {bad}'))
    out = (df['p_dep'] < 0) | (df['p_dep'] > 1)
    if out.any():
        v.append((2, f'p_dep: {int(out.sum())} valores fuera de [0, 1] (p. ej. {df.loc[out, "p_dep"].iloc[0]})'))
    return v


def _check_index(df, index):
    v = []
    for s in schema.SPLIT_ORDER:
        got = set(df.loc[df['split'] == s, 'sample_id'])
        if not got:
            continue
        exp = set(index.loc[index['split'] == s, 'sample_id'])
        if got != exp:
            v.append((3, f'{s}: {len(exp - got)} sample_id faltantes y {len(got - exp)} ajenos al índice '
                         f'(p. ej. {sorted(exp - got)[:1] + sorted(got - exp)[:1]})'))
    merged = df.merge(index.rename(columns={'y_true': 'y_index', 'split': 'split_index'}),
                      on='sample_id', how='inner')
    wrong_split = merged['split'] != merged['split_index']
    if wrong_split.any():
        v.append((3, f'{int(wrong_split.sum())} filas con split distinto al del índice'))
    wrong_y = merged['y_true'] != merged['y_index']
    if wrong_y.any():
        v.append((3, f'y_true difiere del índice en {int(wrong_y.sum())} filas '
                     f'(p. ej. {merged.loc[wrong_y, "sample_id"].iloc[0]})'))
    return v


def _check_duplicates(df):
    dup = df.duplicated(schema.FILE_KEY, keep=False)
    if dup.any():
        return [(4, f'{int(dup.sum())} filas con clave duplicada (p. ej. {df.loc[dup, "sample_id"].iloc[0]})')]
    return []


def _check_filename(path, df):
    m = schema.FILENAME_RE.match(path.name)
    if not m:
        return [(5, f'el nombre {path.name!r} no sigue {{technique}}__{{arch}}__{{strategy}}__seed{{seed}}.csv')]
    if df is None:
        return []
    v = []
    for c in ('technique', 'arch', 'strategy', 'seed'):
        values = set(df[c].astype(str))
        if values != {m.group(c)}:
            v.append((5, f'{c}: el nombre dice {m.group(c)!r} y el contenido {sorted(values)}'))
    return v


def _check_manifest(path, fase):
    mpath = path.with_name(schema.manifest_name(path.name))
    if not mpath.is_file():
        return [(6, f'falta el manifiesto {mpath.name}')], None
    try:
        manifest = json.loads(mpath.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        return [(6, f'{mpath.name} no es JSON válido: {e}')], None
    if not isinstance(manifest, dict):
        return [(6, f'{mpath.name} no es un objeto JSON')], None
    v = []
    missing = [f for f in schema.MANIFEST_FIELDS if f not in manifest]
    if missing:
        v.append((6, f'al manifiesto le faltan campos {missing}'))
    if 'csv_sha256' in manifest and manifest['csv_sha256'] != sha256(path):
        v.append((6, 'csv_sha256 del manifiesto no coincide con el CSV'))
    if fase not in schema.FASES:
        v.append((6, f'la carpeta {fase!r} no es una fase de §3.4 {sorted(schema.FASES)}'))
    elif 'fase' in manifest and manifest['fase'] != fase:
        v.append((6, f'el manifiesto dice fase {manifest["fase"]!r} y la carpeta {fase!r}'))
    return v, manifest


def _check_phase_splits(df, fase, tag_exists):
    if fase not in schema.FASES:
        return []
    v = []
    present = set(df['split'])
    for s in sorted(present - schema.FASES[fase]):
        v.append((7, f'la fase {fase!r} no permite filas de {s}'))
    if 'test' in present and fase in schema.FASES_TEST_REQUIRES_TAG and not tag_exists():
        v.append((7, f'la fase {fase!r} solo admite test si existe la etiqueta {schema.PROTOCOL_TAG!r}'))
    return v


def validate_predictions(path, index=None, tag_exists=protocol_tag_exists):
    """Valida un archivo de predicciones; lanza PredictionFormatError con todas
    las violaciones encontradas. La fase es la carpeta que contiene el archivo.
    `index` (DataFrame o ruta) y `tag_exists` se inyectan en las pruebas."""
    path = Path(path)
    if not path.is_file():
        raise PredictionFormatError(path, [(6, 'el archivo no existe')])
    fase = path.parent.name
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df, v = _parse(raw)
    if df is not None:
        v += _check_vocab(df)
        v += _check_index(df, load_index(index))
        v += _check_duplicates(df)
        v += _check_phase_splits(df, fase, tag_exists)
    v += _check_filename(path, df)
    v += _check_manifest(path, fase)[0]
    if v:
        raise PredictionFormatError(path, sorted(v, key=lambda x: x[0]))
    return df


def _expand(patterns):
    files = []
    for p in patterns:
        hits = sorted(glob.glob(str(p), recursive=True))
        files += hits if hits else [p]
    return [Path(f) for f in dict.fromkeys(files) if not str(f).endswith(schema.MANIFEST_SUFFIX)]


def load_predictions(pattern, index=None, tag_exists=protocol_tag_exists):
    """Valida cada archivo que coincide con `pattern` (glob, admite **),
    añade la columna fase y concatena. Falla si algún archivo es inválido o si
    la clave (fase, sample_id, technique, arch, strategy, seed) se repite
    entre archivos."""
    files = _expand([pattern])
    files = [f for f in files if f.is_file()]
    if not files:
        raise FileNotFoundError(f'ningún archivo de predicciones coincide con {pattern}')
    index = load_index(index)
    frames = []
    for f in files:
        df = validate_predictions(f, index=index, tag_exists=tag_exists)
        frames.append(df.assign(fase=f.parent.name))
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(schema.KEY, keep=False)
    if dup.any():
        raise PredictionFormatError(pattern, [(4, f'{int(dup.sum())} filas con clave duplicada entre archivos')])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='archivos o patrones glob de predicciones (.csv)')
    ap.add_argument('--index', default=str(paths.SAMPLE_INDEX), help='índice maestro (meta/sample_index.csv)')
    args = ap.parse_args()

    index = load_index(args.index)
    files = _expand(args.files)
    n_bad = 0
    for f in files:
        try:
            validate_predictions(f, index=index)
            print(f'OK     {f}')
        except PredictionFormatError as e:
            n_bad += 1
            print(f'FALLA  {f}')
            for rule, msg in e.violations:
                print(f'         regla {rule}: {msg}')
    print(f'\n{len(files) - n_bad}/{len(files)} archivos válidos')
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
