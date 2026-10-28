"""
schema.py
─────────
Formato de los archivos de predicciones por muestra
(docs/especificacion_predicciones.md §3-§4). Solo constantes: la validación
está en validate.py.
"""

import re

# §3.1: exactamente estas ocho columnas, en este orden (regla 9)
COLUMNS = ['sample_id', 'split', 'y_true', 'technique', 'arch', 'strategy', 'seed', 'p_dep']
STR_COLUMNS = ['sample_id', 'split', 'technique', 'arch', 'strategy']
INT_COLUMNS = ['y_true', 'seed']
FLOAT_COLUMNS = ['p_dep']
DTYPES = {**{c: 'object' for c in STR_COLUMNS}, **{c: 'int64' for c in INT_COLUMNS},
          **{c: 'float64' for c in FLOAT_COLUMNS}}

# Orden de las filas: por split en este orden y, dentro de cada uno, por sample_id
SPLIT_ORDER = ['train', 'val', 'test']

# §3.2: vocabulario controlado
VOCAB = {
    'split': set(SPLIT_ORDER),
    'y_true': {0, 1},
    'technique': {'perlin', 'gray_scott', 'lorenz', 'text', 'emb768', 'umap5d', 'none'},
    'arch': {'resnet18', 'efficientnet_b0', 'vit_b16', 'beto', 'logreg', 'svm_rbf', 'gbm', 'rf', 'mlp',
             'majority', 'stratified', 'brightness', 'descriptors'},
    'strategy': {'frozen', 'finetune', 'none'},
}

# §3.3: ŷ = 1 si y solo si p_dep > 0.5
DECISION_THRESHOLD = 0.5

# §3.4: splits permitidos por fase. train es opcional en todas
FASES = {
    'referencia': {'train', 'val', 'test'},
    'primera_pasada': {'train', 'val'},
    'final': {'train', 'val', 'test'},
    'control_permutacion': {'train', 'val', 'test'},
}
# Fases en las que test solo se permite si existe la etiqueta del protocolo congelado
FASES_TEST_REQUIRES_TAG = {'final', 'control_permutacion'}
PROTOCOL_TAG = 'protocolo-v1'

# Clave única; 'fase' la añade load_predictions desde la carpeta y el manifiesto
KEY = ['fase', 'sample_id', 'technique', 'arch', 'strategy', 'seed']
FILE_KEY = [c for c in KEY if c != 'fase']

# §4: results/predictions/{fase}/{technique}__{arch}__{strategy}__seed{seed}.csv (+ .manifest.json)
FILENAME_RE = re.compile(r'^(?P<technique>[^_]+(?:_[^_]+)*?)__(?P<arch>[^_]+(?:_[^_]+)*?)__'
                         r'(?P<strategy>[^_]+(?:_[^_]+)*?)__seed(?P<seed>-?\d+)\.csv$')
MANIFEST_SUFFIX = '.manifest.json'
MANIFEST_FIELDS = ['task_id', 'fase', 'git_commit', 'created_at', 'config', 'inputs', 'model_selection',
                   'environment', 'csv_sha256', 'n_rows']


def file_name(technique, arch, strategy, seed):
    return f'{technique}__{arch}__{strategy}__seed{seed}.csv'


def manifest_name(csv_name):
    return csv_name[:-len('.csv')] + MANIFEST_SUFFIX
