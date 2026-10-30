"""
tests/test_preprocesamiento_config.py
═════════════════════════════════════
F1.4 — configs/preprocesamiento.json debe coincidir con la tabla de la
seccion 7 de docs/decisiones_preprocesamiento.md, que es donde se registra la
decision. Si alguien cambia uno sin el otro, esta prueba falla.

Ejecucion (desde la raiz del repo):
    python -m pytest tests/test_preprocesamiento_config.py -v
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / 'docs' / 'decisiones_preprocesamiento.md'
CONFIG = REPO / 'configs' / 'preprocesamiento.json'
TECHNIQUES = ('perlin', 'gray_scott', 'lorenz')

# Variantes deterministas de la seccion 3 -> (size, resize_mode)
VARIANTS = {'V0': (256, 'native'), 'V1': (224, 'resize'), 'V2': (224, 'center_crop')}
VIT_MODES = {'same', 'native_interp_pos_emb', 'resize_lower_bound'}
KEYS = {'size', 'resize_mode', 'interpolation', 'antialias', 'hflip', 'vit', 'variant'}


def _section7_rows():
    text = DOC.read_text(encoding='utf-8')
    sec = text[text.index('## 7. Decisión'):]
    rows = {}
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) != 5 or not cells[0].startswith('`'):
            continue
        tech = cells[0].strip('`')
        size_code = re.search(r'`(V[012])`', cells[1]).group(1)
        flip = {'Sí': True, 'No': False}[cells[2]]
        vit_code = re.search(r'`(V[012])`', cells[3]).group(1)
        label = cells[4].strip('`')
        rows[tech] = {'size_code': size_code, 'hflip': flip, 'vit_code': vit_code,
                      'vit_text': cells[3], 'variant': label}
    return rows


@pytest.fixture(scope='module')
def table():
    rows = _section7_rows()
    assert set(rows) == set(TECHNIQUES), f'la tabla de la seccion 7 debe tener una fila por tecnica: {rows}'
    return rows


@pytest.fixture(scope='module')
def config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def test_config_has_all_techniques(config):
    assert set(k for k in config if not k.startswith('_')) == set(TECHNIQUES)
    for t in TECHNIQUES:
        assert set(config[t]) == KEYS, f'{t}: claves {set(config[t])}'


@pytest.mark.parametrize('tech', TECHNIQUES)
def test_config_matches_table(tech, table, config):
    row, cfg = table[tech], config[tech]
    size, mode = VARIANTS[row['size_code']]
    assert (cfg['size'], cfg['resize_mode']) == (size, mode)
    assert cfg['hflip'] is row['hflip']
    assert cfg['variant'] == row['variant']
    # Parametros exactos de data.py (seccion 6.1)
    assert cfg['interpolation'] == 'bilinear' and cfg['antialias'] is True

    # Regla 5.2: ViT usa la misma variante si se admitio V1 o V2
    assert cfg['vit'] in VIT_MODES
    if cfg['vit'] == 'same':
        assert row['vit_code'] == row['size_code']
    elif cfg['vit'] == 'native_interp_pos_emb':
        assert row['size_code'] == row['vit_code'] == 'V0'
    else:
        assert row['size_code'] == 'V0' and row['vit_code'] == 'V1'


@pytest.mark.parametrize('tech', TECHNIQUES)
def test_variant_label_format(tech, config):
    """Misma convencion que run_grid.variant_name (px<size>-<flip|noflip>),
    con crop<size> para el recorte central."""
    cfg = config[tech]
    prefix = 'crop' if cfg['resize_mode'] == 'center_crop' else 'px'
    assert cfg['variant'] == f'{prefix}{cfg["size"]}-{"flip" if cfg["hflip"] else "noflip"}'
