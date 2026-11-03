"""
config.py
──────────
Lectura de las dos fuentes unicas de configuracion del Modulo 3:

  - configs/preprocesamiento.json (F1.4): tamano, interpolacion, volteo y
    variante POR TECNICA. Una tecnica que falte en el JSON hace fallar la
    corrida; los valores no soportados por data.py tambien.
  - configs/matriz_experimentos.json (F1.3): factores, configuracion comun de
    entrenamiento, semillas por fase y determinismo.

Nombres: las salidas usan el vocabulario de F1.5 (vit_b16); torchvision y
build_model.py usan vit_b_16. internal_arch() traduce.

Sin torch: se importa tambien desde pruebas que corren sin GPU.
"""

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PREPROCESSING_PATH = os.path.join(REPO, 'configs', 'preprocesamiento.json')
MATRIX_PATH = os.path.join(REPO, 'configs', 'matriz_experimentos.json')

ARCH_INTERNAL = {'vit_b16': 'vit_b_16'}   # nombre de salida (F1.5) -> nombre de build_model.py

# Lo que data.get_transforms sabe aplicar. Otro valor en el JSON = error, no
# una transformacion distinta en silencio.
SUPPORTED_PREPROCESSING = {
    'resize_mode': {'resize'},
    'interpolation': {'bilinear'},
    'vit': {'same'},
}
PREPROCESSING_KEYS = ('size', 'resize_mode', 'interpolation', 'antialias', 'hflip', 'vit', 'variant')


class ConfigError(ValueError):
    pass


def internal_arch(arch):
    return ARCH_INTERNAL.get(arch, arch)


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_preprocessing(techniques, path=PREPROCESSING_PATH):
    """{tecnica: cfg} para las tecnicas pedidas, validado."""
    data = _load(path)
    out = {}
    for tech in techniques:
        if tech not in data:
            raise ConfigError(f'{os.path.basename(path)} no define la tecnica {tech!r}')
        cfg = data[tech]
        missing = [k for k in PREPROCESSING_KEYS if k not in cfg]
        if missing:
            raise ConfigError(f'{tech}: faltan {missing} en {os.path.basename(path)}')
        for key, allowed in SUPPORTED_PREPROCESSING.items():
            if cfg[key] not in allowed:
                raise ConfigError(f'{tech}: {key}={cfg[key]!r} no esta implementado en data.py ({sorted(allowed)})')
        expected_variant = f'px{cfg["size"]}-{"flip" if cfg["hflip"] else "noflip"}'
        if cfg['variant'] != expected_variant:
            raise ConfigError(f'{tech}: variant={cfg["variant"]!r} no corresponde a size/hflip ({expected_variant})')
        out[tech] = {k: cfg[k] for k in PREPROCESSING_KEYS}
    return out


def load_matrix(path=MATRIX_PATH):
    return _load(path)


def training_defaults(matrix):
    """Hiperparametros de la configuracion comun (§3 de la matriz)."""
    c = matrix['configuracion_comun']
    if c['optimizer']['name'] != 'AdamW' or c['optimizer'].get('scheduler') is not None:
        raise ConfigError(f'optimizador {c["optimizer"]} no implementado (train_model usa AdamW sin scheduler)')
    es = c['early_stopping']
    if (es['metric'], es['split'], es['improvement'], es['restore_best']) != ('f1_macro', 'val', 'strict', True):
        raise ConfigError(f'early stopping {es} no implementado')
    if c['amp'] or c['precision'] != 'fp32':
        raise ConfigError('solo fp32 sin AMP esta implementado')
    return {
        'lr': {'finetune': float(c['lr']['finetune']), 'frozen': float(c['lr']['frozen'])},
        'weight_decay': float(c['weight_decay']),
        'batch_size': int(c['batch_size']),
        'epochs': int(c['max_epochs']),
        'patience': int(es['patience']),
        'num_workers': int(c['num_workers']),
    }


def determinism(matrix):
    """Ajustes de determinismo vigentes desde la repeticion de F3.2 (§3, §11.6)."""
    d = matrix['configuracion_comun']['determinism']
    extra = d['desde_repeticion_f32']
    return {
        'cudnn_deterministic': bool(d['cudnn_deterministic']),
        'cudnn_benchmark': bool(d['cudnn_benchmark']),
        'use_deterministic_algorithms': bool(extra['use_deterministic_algorithms']),
        'CUBLAS_WORKSPACE_CONFIG': extra['CUBLAS_WORKSPACE_CONFIG'],
        'flash_sdp': bool(extra['flash_sdp']),
        'mem_efficient_sdp': bool(extra['mem_efficient_sdp']),
    }


def phase_seeds(matrix, fase):
    return list(matrix['fases'][fase]['seeds'])


def phase_requires_tag(matrix, fase):
    return matrix['fases'][fase].get('requires', {}).get('git_tag')
