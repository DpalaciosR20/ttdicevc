"""
pixeles_comun.py
────────────────
FX.1 — Análisis exploratorio de píxeles (docs/plan_exploratorio_pixeles.md).
Funciones comunes a pixeles_local.py (B y D, CPU) y pixeles_gpu.py (C1 y C2).
Sin torch ni E/S de datos reales.

Secciones del plan:
  A   mapas de pendiente: grad (Sobel tras gaussiana σ) y log (|LoG σ|); canny solo figuras.
  B   traslape atribución–pendiente por parche (val; Gray-Scott y Perlin).
  C1  suavizado dirigido a la pendiente alta frente a la misma máscara desplazada (val).
  C2  mapa de oclusión 8 × 8 y su cordura por datos (val).
  D   diferencias de pendiente entre clases por píxel (train; sin modelo).

Todos los valores fijados por el plan están en FIJOS y deben coincidir con
results/fx_pixeles/resoluciones.json. Los puntos que el plan no fija y que
cambian resultados están en PENDIENTES: el archivo los trae en null hasta que
SV los resuelva, y check_resoluciones se detiene mientras alguno falte.
"""
import itertools
import math
import os
import subprocess

import numpy as np

from src.fase4 import atribucion_comun as ac
from src.fase4.atribucion_comun import Detener

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAN = 'docs/plan_exploratorio_pixeles.md'
GATE_TAG = 'plan-pixeles'
RESOLUCIONES_PATH = 'results/fx_pixeles/resoluciones.json'
TECNICAS = ('gray_scott', 'perlin', 'lorenz')
SEEDS = ac.SEEDS
N_VAL, N_TRAIN = 114, 908
PX = 256

# ── Valores fijados por el plan (secciones 2–7) ───────────────────────────
FIJOS = {
    'splits': {'B': 'val', 'C1': 'val', 'C2': 'val', 'D': 'train'},
    'tecnicas': {'B': ['gray_scott', 'perlin'], 'C1': ['gray_scott', 'perlin', 'lorenz'],
                 'C2': ['gray_scott', 'perlin', 'lorenz'], 'D': ['gray_scott', 'perlin', 'lorenz']},
    'tol_pdep': 1e-4,
    'umbral_decision': 0.5,
    'sigma_pendiente': 1,
    'sigma_pendiente_sensibilidad': 2,
    'pendiente_borde_gaussiana': 'reflect',
    'resolucion_pendiente': {'B': 224, 'C1': 256, 'C2': 256, 'D': 256},
    'rejilla_b': {'gray_scott': 7, 'perlin': 14},
    'parche_b_px': {'gray_scott': 32, 'perlin': 16},
    'q': [0.1, 0.2],
    'k_b': {'gray_scott': [5, 10], 'perlin': [20, 40]},
    'k_c1': [6554, 13108],
    'transformaciones_nula_b2': {'gray_scott': 391, 'perlin': 1567},
    'sigma_c1_desenfoque': 4,
    'sigma_c2_desenfoque': 8,
    'desenfoque_borde': 'reflect',
    'uint8': 'recortar_redondear',
    'k_sorteos': 20,
    'k_sorteos_reduccion_autorizada': 10,
    'desplazamiento_rango': [-128, 127],
    'desplazamiento_min_l1': 32,
    'orden_rng_c1': ['tecnica', 'mapa', 'q', 'imagen', 'sorteo'],
    'rejilla_c2': 8,
    'parche_c2_px': 32,
    'c2_criterio_f43': {'criterio_semillas': 'mayoria', 'sobre_maximo': 'como_dentro',
                        'similitud_constantes': 'excluir'},
    'umbral_t': 2.58,
    'umbral_t_sensibilidad': 3.29,
    'conectividad': 8,
    'b_permutaciones_d': 1000,
    'p_cluster': 'k_mas_1_sobre_B_mas_1',
    'semilla_rng': 2026,
    'b_bootstrap': 10000,
    'nivel_ic': 0.95,
    'cuantiles_ic': 'linear',
    'alpha': 0.05,
    'familias': {'H-B': 4, 'H-C': 12, 'H-D': 6},
    'limite_gpu_horas': 3,
}

# ── Puntos que el plan no fija (results/fx_pixeles/reporte_x1.md, PENDIENTES) ──
PENDIENTES = {
    'sobel_borde': ('reflect', 'nearest'),
    'c1_orden_azar': ('tras_los_desplazados_de_cada_imagen', 'tras_todos_los_desplazados_de_la_tecnica'),
    'c1_muestreo_desplazamiento': ('rechazo_dx_luego_dy', 'uniforme_sobre_pares_validos'),
    'c1_muestreo_azar': ('choice_sin_reemplazo', 'prefijo_de_permutacion'),
    'b2_mapas': ('grad_y_log', 'solo_grad'),
    'wilcoxon_metodo': ('normal_sin_correccion', 'normal_con_correccion'),
    'd_sensibilidad': ('dos_variantes_sigma2_y_t329', 'una_variante_sigma2_con_t329'),
}


def check_resoluciones(res):
    """Exige exactamente FIJOS (con sus valores) y PENDIENTES (resueltos con un valor previsto)."""
    faltan = [k for k in list(FIJOS) + list(PENDIENTES) if k not in res]
    if faltan:
        raise Detener(f'faltan claves en resoluciones: {faltan}')
    extra = sorted(k for k in res if k not in FIJOS and k not in PENDIENTES and not str(k).startswith('_'))
    if extra:
        raise Detener(f'claves no previstas en resoluciones: {extra}')
    distintos = {k: res[k] for k in FIJOS if res[k] != FIJOS[k]}
    if distintos:
        raise Detener(f'valores distintos de los del plan: {distintos}')
    sin = [k for k in PENDIENTES if res[k] is None]
    if sin:
        raise Detener('PENDIENTE sin resolver en resoluciones: ' + ', '.join(sin))
    malos = {k: res[k] for k in PENDIENTES if res[k] not in PENDIENTES[k]}
    if malos:
        raise Detener(f'valores no previstos para los PENDIENTES: {malos}')
    return {k: res[k] for k in list(FIJOS) + list(PENDIENTES)}


# ── Regla 2.1: ningún insumo de prueba ────────────────────────────────────
def exigir_split(rows, split):
    """Aborta si alguna fila es de prueba o de otro split que el pedido."""
    if split not in ('val', 'train'):
        raise Detener(f'FX.1 solo usa val o train; se pidió {split!r}')
    for r in rows:
        if r['split'] == 'test':
            raise Detener(f"regla 2.1: insumo de prueba ({r.get('sample_id', '?')}); FX.1 no lee el split de prueba")
        if r['split'] != split:
            raise Detener(f"se esperaba split {split!r} y llegó {r['split']!r} ({r.get('sample_id', '?')})")
    return rows


def exigir_arreglo_split(splits, split):
    """Para arreglos con una columna split (p. ej. .npz de F3.7): máscara del split pedido,
    y aborto si se pidiera 'test'."""
    if split == 'test':
        raise Detener('regla 2.1: FX.1 no lee el split de prueba')
    return np.asarray(splits).astype(str) == split


# ── A. Mapas de pendiente ─────────────────────────────────────────────────
def a_unidad(img_u8):
    img = np.asarray(img_u8)
    if img.dtype != np.uint8:
        raise Detener(f'se espera uint8; llegó {img.dtype}')
    return img.astype(np.float64) / 255.0


def mapa_grad(img01, sigma=1, sobel_borde='reflect', borde='reflect'):
    """Magnitud de Sobel sobre la imagen suavizada con gaussiana σ."""
    from scipy.ndimage import gaussian_filter, sobel
    g = gaussian_filter(np.asarray(img01, np.float64), sigma=sigma, mode=borde)
    return np.hypot(sobel(g, axis=1, mode=sobel_borde), sobel(g, axis=0, mode=sobel_borde))


def mapa_log(img01, sigma=1, borde='reflect'):
    """|Laplaciano de la gaussiana σ| (cambio de pendiente)."""
    from scipy.ndimage import gaussian_laplace
    return np.abs(gaussian_laplace(np.asarray(img01, np.float64), sigma=sigma, mode=borde))


def mapa_canny(img01, sigma=1):
    """Solo para figuras (sección 3): sin estadística."""
    from skimage.feature import canny
    return canny(np.asarray(img01, np.float64), sigma=sigma)


def mapas_pendiente(img_u8, sigma, sobel_borde):
    x = a_unidad(img_u8)
    return {'grad': mapa_grad(x, sigma, sobel_borde), 'log': mapa_log(x, sigma)}


# ── B. Traslape por parche ────────────────────────────────────────────────
def media_por_parche(maps, g):
    """(N, H, W) → (N, g, g): media en cada parche (H múltiplo de g)."""
    return ac.pool(np.asarray(maps, np.float64), g)


def suma_por_parche(maps, g):
    """(N, H, W) → (N, g, g): suma en cada parche (masa de atribución, como F4.4)."""
    from src.fase4.convergencia import masa_por_parche
    return masa_por_parche(np.asarray(maps, np.float64), g)


def k_superior(q, n):
    return int(math.ceil(q * n))


def top_k(v, k):
    """Índices de los k valores mayores; empates por orden estable sobre el valor negado."""
    return np.argsort(-np.asarray(v, np.float64).ravel(), kind='stable')[:k]


def bottom_k(v, k):
    """Índices de los k valores menores; empates por orden estable."""
    return np.argsort(np.asarray(v, np.float64).ravel(), kind='stable')[:k]


def dice_topk(a, b, k):
    """|top_k(a) ∩ top_k(b)| / k."""
    return len(np.intersect1d(top_k(a, k), top_k(b, k), assume_unique=True)) / k


def transformaciones_rejilla(g):
    """Permutaciones de las g² posiciones: 8 transformaciones diédricas × g² desplazamientos
    circulares, sin la identidad (reflejo no, rotación 0, desplazamiento (0, 0)).
    Cada fila p cumple: valor transformado en la posición j = valor original en p[j]."""
    base = np.arange(g * g).reshape(g, g)
    out = []
    for refl in (False, True):
        for rot in range(4):
            t0 = np.rot90(np.fliplr(base) if refl else base, rot)
            for dy in range(g):
                for dx in range(g):
                    if not refl and rot == 0 and dy == 0 and dx == 0:
                        continue
                    out.append(np.roll(t0, (dy, dx), axis=(0, 1)).ravel())
    out = np.array(out)
    if (out == np.arange(g * g)).all(1).any():
        raise Detener(f'rejilla {g}: una transformación distinta de la identidad es la identidad')
    return out


def dice_nulo(a, b, k, perms):
    """Media del Dice entre top_k(a) y top_k(b transformado) sobre las transformaciones."""
    a_top = set(top_k(a, k).tolist())
    bt = np.asarray(b, np.float64).ravel()[perms]                 # (T, g²)
    idx = np.argsort(-bt, axis=1, kind='stable')[:, :k]
    return float(np.mean([len(a_top.intersection(r.tolist())) / k for r in idx]))


def spearman_por_muestra(a, b):
    """Spearman (rangos promedio) por muestra; NaN si alguno de los dos vectores es constante."""
    from src.fase4.convergencia import spearman_por_muestra as sp
    return sp(a, b)


# ── Estadística común ─────────────────────────────────────────────────────
def indices_bootstrap(n, seed=2026, b=10000):
    from src.fase4.convergencia import indices_bootstrap as ib
    return ib(n, np.random.default_rng(seed), b)


def ic(valores, idx, nivel=0.95, metodo='linear'):
    """Media e IC percentil (NaN excluidos de cada réplica), como F4.4."""
    from src.fase4.convergencia import ic_bootstrap
    return ic_bootstrap(valores, idx, nivel=nivel, metodo=metodo)


def mann_whitney(x1, x0):
    """Mann–Whitney bilateral asintótico con corrección de continuidad (sin NaN).
    Biserial de rangos r = 2·U₁/(n₁·n₀) − 1, con U₁ el de x1 (r > 0: x1 tiende a ser mayor)."""
    from scipy.stats import mannwhitneyu
    x1 = np.asarray(x1, np.float64)
    x0 = np.asarray(x0, np.float64)
    x1, x0 = x1[~np.isnan(x1)], x0[~np.isnan(x0)]
    r = mannwhitneyu(x1, x0, alternative='two-sided', method='asymptotic', use_continuity=True)
    return {'u1': float(r.statistic), 'p': float(r.pvalue), 'n1': len(x1), 'n0': len(x0),
            'biserial': float(2 * r.statistic / (len(x1) * len(x0)) - 1)}


def wilcoxon(s, metodo):
    """Wilcoxon de rangos con signo, bilateral, zero_method='wilcox'. metodo (PENDIENTE):
    aproximación normal sin o con corrección de continuidad."""
    from scipy.stats import wilcoxon as w
    s = np.asarray(s, np.float64)
    s = s[~np.isnan(s)]
    corr = {'normal_sin_correccion': False, 'normal_con_correccion': True}[metodo]
    r = w(s, zero_method='wilcox', alternative='two-sided', correction=corr, method='approx')
    return {'w': float(r.statistic), 'p': float(r.pvalue), 'n': int(len(s)), 'n_ceros': int((s == 0).sum())}


def holm(pvalues, m_esperado, alpha=0.05):
    if len(pvalues) != m_esperado:
        raise Detener(f'familia con {len(pvalues)} pruebas; el plan fija m = {m_esperado}')
    return ac.holm(pvalues, alpha)


# ── C1. Máscaras y suavizado dirigido ─────────────────────────────────────
def mascara(indices, shape=(PX, PX)):
    m = np.zeros(int(np.prod(shape)), dtype=bool)
    m[indices] = True
    return m.reshape(shape)


def desplazar(m, dy, dx):
    return np.roll(m, (dy, dx), axis=(0, 1))


def a_uint8(y):
    from src.fase4.atribucion_gpu import a_uint8 as au
    return au(y, 'recortar_redondear')


def desenfocar(img_u8, sigma, borde='reflect'):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(img_u8, np.float64), sigma=sigma, mode=borde)


def suavizar(img_u8, m, desenfocada):
    """I' = I·(1 − M) + B(I)·M, recortada y redondeada a uint8."""
    x = np.asarray(img_u8, np.float64)
    mm = np.asarray(m, np.float64)
    return a_uint8(x * (1 - mm) + desenfocada * mm)


def _desplazamiento(rng, metodo, lo=-128, hi=127, minimo=32, validos=None):
    if metodo == 'rechazo_dx_luego_dy':
        while True:
            dx = int(rng.integers(lo, hi + 1))
            dy = int(rng.integers(lo, hi + 1))
            if abs(dx) + abs(dy) >= minimo:
                return dy, dx
    if metodo == 'uniforme_sobre_pares_validos':
        dy, dx = validos[int(rng.integers(0, len(validos)))]
        return int(dy), int(dx)
    raise Detener(f'c1_muestreo_desplazamiento desconocido: {metodo!r}')


def pares_validos(lo=-128, hi=127, minimo=32):
    v = np.arange(lo, hi + 1)
    dy, dx = np.meshgrid(v, v, indexing='ij')
    ok = np.abs(dy) + np.abs(dx) >= minimo
    return np.stack([dy[ok], dx[ok]], 1)


def _azar(rng, k, metodo, n=PX * PX):
    if metodo == 'choice_sin_reemplazo':
        return rng.choice(n, size=k, replace=False)
    if metodo == 'prefijo_de_permutacion':
        return rng.permutation(n)[:k]
    raise Detener(f'c1_muestreo_azar desconocido: {metodo!r}')


def sorteos_c1(rng, mapas, qs, n_img, k_sorteos, k_por_q, res):
    """Itera en el orden del plan (mapa → q → imagen → sorteo) y produce
    (mapa, q, i, [(dy, dx)] × K, [índices de azar] × K).
    res['c1_orden_azar']: los K de azar van tras los K desplazados de cada imagen,
    o tras todos los desplazados de la técnica (PENDIENTE)."""
    validos = pares_validos() if res['c1_muestreo_desplazamiento'] == 'uniforme_sobre_pares_validos' else None
    claves = [(m, q, i) for m in mapas for q in qs for i in range(n_img)]

    def desp():
        return [_desplazamiento(rng, res['c1_muestreo_desplazamiento'], validos=validos) for _ in range(k_sorteos)]

    if res['c1_orden_azar'] == 'tras_los_desplazados_de_cada_imagen':
        for m, q, i in claves:
            d = desp()
            yield m, q, i, d, [_azar(rng, k_por_q[q], res['c1_muestreo_azar']) for _ in range(k_sorteos)]
    elif res['c1_orden_azar'] == 'tras_todos_los_desplazados_de_la_tecnica':
        todos = {c: desp() for c in claves}
        for m, q, i in claves:
            yield m, q, i, todos[(m, q, i)], [_azar(rng, k_por_q[q], res['c1_muestreo_azar']) for _ in range(k_sorteos)]
    else:
        raise Detener(f"c1_orden_azar desconocido: {res['c1_orden_azar']!r}")


def imagenes_c1(img_u8, pend, q, k, desplazamientos, azares, desenfocada):
    """Lista de imágenes modificadas en el orden: alta, baja, K desplazadas, K de azar."""
    alta = mascara(top_k(pend, k))
    baja = mascara(bottom_k(pend, k))
    out = [suavizar(img_u8, alta, desenfocada), suavizar(img_u8, baja, desenfocada)]
    out += [suavizar(img_u8, desplazar(alta, dy, dx), desenfocada) for dy, dx in desplazamientos]
    out += [suavizar(img_u8, mascara(ix), desenfocada) for ix in azares]
    return out


def estadistico_s(p_orig, p_alta, p_desp):
    """S_i = |p_alta − p| − media_k |p_desp,k − p|. p_desp: (N, K)."""
    p_orig = np.asarray(p_orig, np.float64)
    return np.abs(np.asarray(p_alta) - p_orig) - np.abs(np.asarray(p_desp) - p_orig[:, None]).mean(1)


# ── C2. Oclusión ──────────────────────────────────────────────────────────
def parches_c2(g=8, px=32):
    return [(r, c) for r in range(g) for c in range(g)]


def ocluir(img_u8, desenfocada, r, c, px=32):
    y = np.asarray(img_u8, np.float64).copy()
    y[r * px:(r + 1) * px, c * px:(c + 1) * px] = desenfocada[r * px:(r + 1) * px, c * px:(c + 1) * px]
    return a_uint8(y)


def imagenes_c2(img_u8, desenfocada, g=8, px=32):
    return [ocluir(img_u8, desenfocada, r, c, px) for r, c in parches_c2(g, px)]


def mapa_oclusion(p_ocluidas, p_orig, g=8):
    """O = |p(ocluida) − p(original)| por parche: (N, g²) → (N, g, g)."""
    return np.abs(np.asarray(p_ocluidas, np.float64) - np.asarray(p_orig, np.float64)[:, None]).reshape(-1, g, g)


def cordura_c2(o_real, o_perm, criterio):
    """o_real, o_perm: {semilla: (N, 8, 8)}. Spearman por muestra real–permutado de la misma
    semilla, frente al rango de los 10 pares de semillas reales (criterio relativo de F4.3)."""
    fn = lambda a, b: ac.spearman_filas(a, b, g=a.shape[-1])  # noqa: E731
    por, info = {}, {}
    for s in o_real:
        rho = fn(o_real[s], o_perm[s])
        m, n, nn = ac.resumen_similitud(rho, criterio['similitud_constantes'])
        por[s], info[str(s)] = m, {'mediana': m, 'n_usado': n, 'n_nan': nn,
                                   'n_real_constante': int(ac.es_constante(o_real[s]).sum()),
                                   'n_perm_constante': int(ac.es_constante(o_perm[s]).sum())}
    ref = ac.referencia_mapas(o_real, fn, criterio['similitud_constantes'])
    return {'por_semilla_detalle': info, 'referencia_pares': ref, **ac.criterio_similitud(por, ref, criterio)}


# ── D. Prueba por clústeres (Maris y Oostenveld, 2007) ────────────────────
def welch_t(x1, x0):
    """t de Welch por columna (x1 − x0); 0 donde la varianza conjunta es 0."""
    x1 = np.asarray(x1, np.float64)
    x0 = np.asarray(x0, np.float64)
    m1, m0 = x1.mean(0), x0.mean(0)
    v = x1.var(0, ddof=1) / len(x1) + x0.var(0, ddof=1) / len(x0)
    return np.where(v > 0, (m1 - m0) / np.sqrt(np.where(v > 0, v, 1)), 0.0)


class TPermutaciones:
    """t de Welch para muchas permutaciones de y sobre la misma X (n, P): centra X y calcula
    X² una sola vez."""

    def __init__(self, X):
        X = np.asarray(X, np.float64)
        self.n = X.shape[0]
        self.Xc = X - X.mean(0)                  # reduce la cancelación numérica
        self.X2 = self.Xc * self.Xc
        self.s_tot, self.q_tot = self.Xc.sum(0), self.X2.sum(0)

    def __call__(self, Y):
        """Y: (B, n) etiquetas 0/1 con la misma suma por fila → (B, P)."""
        Y = np.asarray(Y, np.float64)
        n1 = Y.sum(1, keepdims=True)
        if not np.all(n1 == n1[0]):
            raise Detener('las permutaciones deben conservar el número de casos por clase')
        n0 = self.n - n1
        s1, q1 = Y @ self.Xc, Y @ self.X2
        m1, m0 = s1 / n1, (self.s_tot - s1) / n0
        v1 = (q1 - n1 * m1 * m1) / (n1 - 1)
        v0 = ((self.q_tot - q1) - n0 * m0 * m0) / (n0 - 1)
        v = np.clip(v1, 0, None) / n1 + np.clip(v0, 0, None) / n0
        return np.where(v > 0, (m1 - m0) / np.sqrt(np.where(v > 0, v, 1)), 0.0)


def welch_t_permutaciones(X, Y):
    """Atajo de TPermutaciones para un solo bloque."""
    return TPermutaciones(X)(Y)


ESTRUCTURA_8 = np.ones((3, 3), dtype=int)


def _etiquetas(t, umbral, signo):
    from scipy.ndimage import label
    return label(signo * t > umbral, structure=ESTRUCTURA_8)


def clusteres(t, umbral, conectividad=8):
    """Clústeres de |t| > umbral separados por signo, con conectividad 8. Masa = Σ|t|.
    Ordenados por masa descendente."""
    if conectividad != 8:
        raise Detener('el plan fija conectividad 8')
    t = np.asarray(t, np.float64)
    h, w = t.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for signo in (1, -1):
        lab, n = _etiquetas(t, umbral, signo)
        if not n:
            continue
        l = lab.ravel()
        masa = np.bincount(l, weights=np.abs(t).ravel(), minlength=n + 1)[1:]
        area = np.bincount(l, minlength=n + 1)[1:]
        cy = np.bincount(l, weights=yy.ravel(), minlength=n + 1)[1:] / area
        cx = np.bincount(l, weights=xx.ravel(), minlength=n + 1)[1:] / area
        out += [{'signo': '+' if signo > 0 else '-', 'masa': float(masa[j]), 'area': int(area[j]),
                 'centroide_yx': [float(cy[j]), float(cx[j])]} for j in range(n)]
    return sorted(out, key=lambda c: -c['masa'])


def masa_maxima(t, umbral, conectividad=8):
    """Máxima masa de clúster (ambos signos); 0 si no hay ninguno."""
    if conectividad != 8:
        raise Detener('el plan fija conectividad 8')
    t = np.asarray(t, np.float64)
    best = 0.0
    for signo in (1, -1):
        lab, n = _etiquetas(t, umbral, signo)
        if n:
            best = max(best, float(np.bincount(lab.ravel(), weights=np.abs(t).ravel())[1:].max()))
    return best


def prueba_clusteres(X, y, umbral, b, seed, bloque=50):
    """X: (N, H, W). Mapa t observado, clústeres con su p = (k + 1)/(B + 1) frente a la máxima
    masa por permutación (default_rng(seed) nuevo; una permutación de etiquetas por réplica)."""
    X = np.asarray(X, np.float64)
    if X.ndim != 3:
        raise Detener('prueba_clusteres espera (N, H, W)')
    shape = X.shape[1:]
    X = X.reshape(len(X), -1)
    y = np.asarray(y).astype(int)
    t_obs = welch_t(X[y == 1], X[y == 0]).reshape(shape)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(y) for _ in range(b)])
    nula = np.empty(b)
    tp = TPermutaciones(X)
    for i in range(0, b, bloque):
        tt = tp(perms[i:i + bloque])
        for j, t in enumerate(tt):
            nula[i + j] = masa_maxima(t.reshape(shape), umbral)
    cl = clusteres(t_obs, umbral)
    for c in cl:
        c['p'] = float((np.sum(nula >= c['masa']) + 1) / (b + 1))
    return {'t': t_obs, 'clusteres': cl, 'nula_masa_maxima': nula,
            'p_mayor': cl[0]['p'] if cl else 1.0}


# ── Compuerta y metadatos ─────────────────────────────────────────────────
def _git(*args):
    return subprocess.run(['git', '-C', REPO, *args], capture_output=True, text=True)


def check_gate(resoluciones_path, out_dir, smoke=False):
    """plan-pixeles ancestro de HEAD, plan sin cambios frente a la etiqueta, resoluciones
    versionado y sin cambios, árbol limpio y sin salidas previas en out_dir."""
    if _git('merge-base', '--is-ancestor', GATE_TAG, 'HEAD').returncode != 0:
        raise Detener(f'la etiqueta {GATE_TAG} no está en la historia de HEAD')
    if _git('diff', '--quiet', GATE_TAG, 'HEAD', '--', PLAN).returncode != 0:
        raise Detener(f'{PLAN} cambió frente a {GATE_TAG}')
    rel = os.path.relpath(os.path.abspath(resoluciones_path), REPO).replace(os.sep, '/')
    if _git('ls-files', '--error-unmatch', rel).returncode != 0:
        raise Detener(f'{rel} no está versionado')
    st = _git('status', '--porcelain').stdout.strip()
    if st:
        raise Detener('árbol de trabajo con cambios:\n' + st)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise Detener(f'{out_dir} ya tiene salidas: FX.1 nunca sobrescribe')
    dentro = os.path.abspath(out_dir).startswith(os.path.join(REPO, 'results', 'fx_pixeles'))
    if smoke and dentro:
        raise Detener('la prueba acotada (--limit) no escribe en results/fx_pixeles/')
    return _git('rev-parse', 'HEAD').stdout.strip()


def entorno():
    import importlib.metadata as md
    import platform
    libs = {}
    for lib in ('numpy', 'scipy', 'scikit-image', 'torch', 'torchvision', 'pandas', 'scikit-learn'):
        try:
            libs[lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'platform': platform.platform(), 'libraries': libs}


def huella_lista(hashes):
    import hashlib
    return hashlib.sha256(''.join(f'{k}\t{v}\n' for k, v in sorted(hashes.items())).encode()).hexdigest()


limpiar = ac.limpiar
pares_semillas = ac.pares_semillas
