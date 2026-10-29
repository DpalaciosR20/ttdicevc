"""
style.py
────────
Estilo comun de las figuras del proyecto: tamanos de letra, colores por
tecnica y formato de exportacion (PNG a 300 dpi).

Uso:
    import sys; sys.path.insert(0, '<repo>/src')
    from viz import style
    style.apply()
    fig, ax = plt.subplots()
    ...
    style.save(fig, 'results/<tarea>/figures/<nombre>')   # agrega .png

Colores: paleta categorica de referencia validada (skill dataviz:
validate_palette.js, modo claro, superficie #fcfcfb): pasa banda de
luminosidad, croma, separacion CVD y piso de vision normal; el verde de
Lorenz queda bajo 3:1 de contraste, por lo que toda serie lleva tambien su
etiqueta de texto (nunca solo color). Divergente: azul <-> rojo con punto
medio gris.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DPI = 300
FORMAT = 'png'

# Orden fijo por tecnica (no ciclar): mismo color en todas las figuras
TECH_COLOR = {'perlin': '#2a78d6', 'gray_scott': '#eb6834', 'lorenz': '#1baf7a'}
TECH_LABEL = {'perlin': 'Perlin', 'gray_scott': 'Gray-Scott', 'lorenz': 'Lorenz'}

INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
SURFACE = '#fcfcfb'
THRESHOLD_COLOR = '#d03b3b'   # linea de umbral (rojo de estado "critico")

# Divergente azul <-> gris <-> rojo para correlaciones
DIVERGING = LinearSegmentedColormap.from_list(
    'tt_divergente', ['#104281', '#3987e5', '#f0efec', '#e66767', '#a32a2a'])

FONT_SIZES = {'base': 9, 'title': 10, 'suptitle': 12, 'label': 9, 'tick': 8, 'legend': 8, 'annot': 7}


def apply():
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': DPI,
        'savefig.format': FORMAT,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': FONT_SIZES['base'],
        'axes.titlesize': FONT_SIZES['title'],
        'figure.titlesize': FONT_SIZES['suptitle'],
        'axes.labelsize': FONT_SIZES['label'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.frameon': False,
        'axes.edgecolor': MUTED,
        'axes.labelcolor': INK,
        'axes.titlecolor': INK,
        'xtick.color': INK_SECONDARY,
        'ytick.color': INK_SECONDARY,
        'axes.grid': False,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.6,
        'lines.markersize': 5,
        'image.interpolation': 'nearest',
    })


def save(fig, path):
    """Guarda en PNG a 300 dpi; crea la carpeta. Devuelve la ruta final."""
    path = Path(path).with_suffix('.' + FORMAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, format=FORMAT)
    return path


def gray(ax, img_uint8, title=None):
    """Imagen uint8 en escala de grises con rango fijo 0-255."""
    ax.imshow(img_uint8, cmap='gray', vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
