"""
atribucion_comun.py
───────────────────
F4.3 — Atribución espacial y espectral (docs/plan_fase4.md, F4.3). Funciones
comunes a atribucion_local.py (CPU) y atribucion_gpu.py (Kaggle). Sin torch ni E/S.

Parámetros del plan (F4.3-1…9) fijados aquí:
  Similitud      Spearman por muestra a 56 × 56 (principal) y SSIM (secundaria).
  Criterio       «no cambia» si la similitud (mediana por muestra) cae dentro del
                 rango de las similitudes entre semillas reales del mismo
                 finalista (10 pares); «cambia» si queda por debajo del mínimo.
  Cosenos        «alto» dentro del rango de la referencia (10 pares de cabezas
                 reales); «bajo» por debajo de su mínimo.
  Gray-Scott     perfil radial desde el centro, 28 anillos de 4 px (224 px).
  Lorenz         terciles de u4 con cortes en train, en coordenadas de imagen.
  Prueba         permutación de etiquetas (5,000, default_rng(2026)) sobre la
                 norma L2 de la diferencia de mapas medios por clase
                 (map_raw · map_raw_scale), en val; Holm m = 3.
  Constantes     si una clase tiene < 10 mapas no constantes, la sensibilidad
                 sin constantes no se reporta para esa clase (sí el conteo).

PENDIENTES que el plan no fija (results/f43_atribucion/reporte_e3a.md). No se
resuelven aquí: las funciones los reciben como parámetros y los dos scripts
exigen el archivo --resoluciones con todos ellos (RESOLUCIONES).
"""
import itertools
import math

import numpy as np

TECHNIQUES = ('gray_scott', 'perlin', 'lorenz')
SEEDS = (42, 7, 123, 2024, 99)
FINALIST = {'gray_scott': ('efficientnet_b0', 'finetune'), 'perlin': ('vit_b16', 'frozen'),
            'lorenz': ('vit_b16', 'frozen')}
STEM = {'gray_scott': 'gray_scott__efficientnet_b0__finetune__px224-flip',
        'perlin': 'perlin__vit_b16__frozen__px224-noflip',
        'lorenz': 'lorenz__vit_b16__frozen__px224-noflip'}
VIT = ('perlin', 'lorenz')
SIZE = 224
GRID = 56
N_ANILLOS = 28
ANCHO_ANILLO = 4
PERM_B = 5000
PERM_SEED = 2026
CASCADA_SEED = 2026
SIGMAS = (1, 2, 4, 8)
ALPHA = 0.05
MIN_NO_CONSTANTES = 10
N_PARES = 10

RESOLUCIONES = {
    # Agregación por clase
    'clase_predicha': ('yhat_media_5_semillas', 'yhat_por_semilla'),
    'escala_entre_semillas': ('sin_reescalar', 'dividir_media_semilla'),
    'umbral_constantes_unidad': ('agrupado', 'por_semilla'),
    # Prueba de permutación
    'l2_resolucion': ('224', '56'),
    'perm_p': ('k_mas_1_sobre_B_mas_1', 'k_sobre_B'),
    'perm_rng': ('por_tecnica', 'secuencial'),
    # Similitud y criterio relativo
    'similitud_constantes': ('excluir', 'cero'),
    'sobre_maximo': ('como_dentro', 'indeterminado'),
    'criterio_semillas': ('mediana_de_semillas', 'mayoria', 'unanimidad'),
    'ssim_resolucion': ('56', '224'),
    # Gray-Scott: correlación de mapas frente a la de p_dep
    'gs_corr_pdep': ('f36_prueba_media_real', 'misma_semilla_val', 'misma_semilla_prueba'),
    'gs_corr_mapas': ('spearman56_mediana', 'pearson_media'),
    # Lectura de los ViT: qué coseno decide «alto / bajo»
    'coseno_lectura': ('w1_menos_w0', 'w1', 'ambos'),
    # Cascada (GPU)
    'vit_ln_final': ('con_cabeza', 'paso_propio', 'sin_reinicializar'),
    'vit_embeddings': ('solo_conv_proj', 'conv_proj_y_tokens'),
    'mha_in_proj': ('reset_mha_completo', 'solo_reset_parameters'),
    'semilla_cascada': ('una_vez_por_modelo', 'en_cada_paso'),
    # Ablación (GPU)
    'ablacion_borde': ('reflect', 'nearest'),
    'ablacion_rango': ('recortar_redondear', 'reescalar_min_max'),
}


class Detener(RuntimeError):
    """El análisis se detiene porque falla una comprobación o una compuerta."""


def check_resoluciones(res, spec=RESOLUCIONES):
    faltan = [k for k in spec if res.get(k) is None]
    if faltan:
        raise Detener('PENDIENTE sin resolver en --resoluciones: ' + ', '.join(faltan))
    malos = {k: res[k] for k in spec if res[k] not in spec[k]}
    if malos:
        raise Detener(f'valores no previstos en --resoluciones: {malos}')
    extra = sorted(k for k in res if k not in spec and not str(k).startswith('_'))
    if extra:
        raise Detener(f'claves no previstas en --resoluciones: {extra}')
    return {k: res[k] for k in spec}


def limpiar(o):
    """Para JSON: numpy → Python, NaN/inf → None, claves a str."""
    if isinstance(o, dict):
        return {str(k): limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpiar(v) for v in o]
    if isinstance(o, np.ndarray):
        return limpiar(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def meta_resoluciones(res):
    return {str(k)[1:]: v for k, v in res.items() if str(k).startswith('_')}


def run_name(tech, seed, perm=False):
    return f"{STEM[tech]}{'-perm99' if perm else ''}__seed{seed}"


# ── Mapas ─────────────────────────────────────────────────────────────────
def mapa_crudo(z, key='map'):
    """map_raw · map_raw_scale (sin normalizar), float32 (N, H, W)."""
    return z[f'{key}_raw'].astype(np.float32) * np.asarray(z[f'{key}_raw_scale'], dtype=np.float32)[:, None, None]


def pool(maps, g=GRID):
    """(N, H, W) -> (N, g, g) por promedio en bloques (H múltiplo de g)."""
    maps = np.asarray(maps)
    n, h, w = maps.shape
    if h % g or w % g:
        raise Detener(f'rejilla {g} no divide {h} × {w}')
    k = h // g
    return maps.reshape(n, g, k, g, k).mean((2, 4))


def es_constante(maps):
    """Por muestra: el mapa no varía (máximo = mínimo)."""
    flat = np.asarray(maps).reshape(len(maps), -1)
    return flat.max(1) == flat.min(1)


def minmax(maps):
    """Normaliza cada mapa a [0, 1]; un mapa constante queda en ceros (igual que F3.7)."""
    flat = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1)
    lo, hi = flat.min(1, keepdims=True), flat.max(1, keepdims=True)
    rng = hi - lo
    out = np.where(rng > 0, (flat - lo) / np.where(rng > 0, rng, 1), 0.0)
    return out.reshape(np.shape(maps))


# ── Similitud ─────────────────────────────────────────────────────────────
def _pearson_filas(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)


def spearman_filas(a, b, g=GRID):
    """Spearman por muestra entre mapas (N, H, W) promediados a g × g. Rangos
    promedio en empates. NaN si alguno de los dos mapas es constante en la rejilla."""
    from scipy.stats import rankdata
    pa = pool(a, g).reshape(len(a), -1).astype(np.float64)
    pb = pool(b, g).reshape(len(b), -1).astype(np.float64)
    return _pearson_filas(rankdata(pa, axis=1), rankdata(pb, axis=1))


def pearson_filas(a, b, g=GRID):
    """Pearson por muestra a g × g (la medida de corr_con_real de F3.7)."""
    pa = pool(a, g).reshape(len(a), -1).astype(np.float64)
    pb = pool(b, g).reshape(len(b), -1).astype(np.float64)
    return _pearson_filas(pa, pb)


def ssim_filas(a, b, g=None, data_range=1.0, sigma=1.5, truncate=3.5, k1=0.01, k2=0.03):
    """SSIM por muestra (Wang et al., 2004) con ventana gaussiana, como
    skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False, data_range=1). Mapas normalizados a [0, 1];
    g: rejilla (None = resolución original)."""
    from scipy.ndimage import gaussian_filter
    if g is not None:
        a, b = pool(a, g), pool(b, g)
    pad = (2 * int(truncate * sigma + 0.5) + 1 - 1) // 2
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    out = np.empty(len(a))
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)

        def f(t):
            return gaussian_filter(t, sigma=sigma, truncate=truncate, mode='reflect')
        ux, uy = f(x), f(y)
        vx, vy, vxy = f(x * x) - ux * ux, f(y * y) - uy * uy, f(x * y) - ux * uy
        s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
        out[i] = s[pad:-pad, pad:-pad].mean()
    return out


def resumen_similitud(s, constantes):
    """Mediana por muestra de una similitud. constantes: 'excluir' (los NaN de
    Spearman, mapas constantes, se descartan) o 'cero' (cuentan como 0).
    Devuelve (mediana, n usado, n NaN)."""
    s = np.asarray(s, dtype=np.float64)
    nan = np.isnan(s)
    if constantes == 'cero':
        s = np.where(nan, 0.0, s)
    elif constantes == 'excluir':
        s = s[~nan]
    else:
        raise Detener(f'similitud_constantes desconocido: {constantes!r}')
    return (float(np.median(s)) if len(s) else math.nan), int(len(s)), int(nan.sum())


# ── Criterio relativo ─────────────────────────────────────────────────────
def pares_semillas(seeds=SEEDS):
    p = list(itertools.combinations(seeds, 2))
    if len(p) != N_PARES and tuple(seeds) == SEEDS:
        raise Detener('se esperan 10 pares de semillas')
    return p


def clasificar(valor, referencia, sobre_maximo):
    """'debajo' (< mínimo), 'dentro' ([mínimo, máximo]) o, por encima del máximo,
    'dentro' (sobre_maximo = 'como_dentro') o 'indeterminado'. NaN → 'indeterminado'."""
    ref = np.asarray([r for r in referencia if r is not None and np.isfinite(r)], dtype=np.float64)
    if not len(ref) or valor is None or not np.isfinite(valor):
        return 'indeterminado'
    lo, hi = ref.min(), ref.max()
    if valor < lo:
        return 'debajo'
    if valor <= hi:
        return 'dentro'
    if sobre_maximo == 'como_dentro':
        return 'dentro'
    if sobre_maximo == 'indeterminado':
        return 'indeterminado'
    raise Detener(f'sobre_maximo desconocido: {sobre_maximo!r}')


def combinar_semillas(valores, referencia, criterio, sobre_maximo):
    """Clasificación por semilla y global ('debajo' | 'dentro' | 'indeterminado').
    criterio: 'mediana_de_semillas' (la mediana de los valores finitos frente al
    rango), 'mayoria' (la categoría de más de la mitad de las semillas) o
    'unanimidad' (todas iguales)."""
    por = [clasificar(v, referencia, sobre_maximo) for v in valores]
    if criterio == 'mediana_de_semillas':
        fin = [v for v in valores if v is not None and np.isfinite(v)]
        glob = clasificar(float(np.median(fin)), referencia, sobre_maximo) if fin else 'indeterminado'
    elif criterio == 'mayoria':
        cats = [c for c in ('debajo', 'dentro') if sum(p == c for p in por) * 2 > len(por)]
        glob = cats[0] if cats else 'indeterminado'
    elif criterio == 'unanimidad':
        glob = por[0] if len(set(por)) == 1 else 'indeterminado'
    else:
        raise Detener(f'criterio_semillas desconocido: {criterio!r}')
    rango = [float(np.nanmin(referencia)), float(np.nanmax(referencia))] if len(referencia) else None
    return {'por_semilla': por, 'global': glob, 'rango_referencia': rango}


MAPA_CAMBIA = {'debajo': 'cambia', 'dentro': 'no_cambia', 'indeterminado': 'indeterminado'}
COSENO_NIVEL = {'debajo': 'bajo', 'dentro': 'alto', 'indeterminado': 'indeterminado'}


def criterio_similitud(por_semilla, referencia_pares, res):
    """por_semilla: {semilla: mediana por muestra}; referencia_pares: {par: mediana}.
    Devuelve la clasificación de «cambia / no cambia»."""
    seeds = list(por_semilla)
    c = combinar_semillas([por_semilla[s] for s in seeds], list(referencia_pares.values()),
                          res['criterio_semillas'], res['sobre_maximo'])
    return {'semillas': seeds, 'valores': [por_semilla[s] for s in seeds],
            'por_semilla': [MAPA_CAMBIA[p] for p in c['por_semilla']], 'global': MAPA_CAMBIA[c['global']],
            'rango_referencia': c['rango_referencia']}


def referencia_mapas(mapas_por_semilla, fn, constantes):
    """{'a-b': mediana por muestra de fn(mapas_a, mapas_b)} para los 10 pares."""
    out = {}
    for a, b in pares_semillas(tuple(mapas_por_semilla)):
        out[f'{a}-{b}'] = resumen_similitud(fn(mapas_por_semilla[a], mapas_por_semilla[b]), constantes)[0]
    return out


# ── Agregación por clase ──────────────────────────────────────────────────
def medias_por_clase(maps, clase):
    """{0: media, 1: media} y conteos; una clase sin muestras da None."""
    clase = np.asarray(clase).astype(int)
    out, n = {}, {}
    for c in (0, 1):
        m = clase == c
        n[c] = int(m.sum())
        out[c] = np.asarray(maps)[m].mean(0) if m.any() else None
    return out, n


def escalar_semillas(maps_por_semilla, escala, ref_por_semilla=None):
    """'sin_reescalar' o 'dividir_media_semilla': cada semilla se divide entre la
    media de sus mapas crudos de referencia (val), para que ninguna semilla pese
    más por la magnitud de su mapa."""
    if escala == 'sin_reescalar':
        return dict(maps_por_semilla), {s: 1.0 for s in maps_por_semilla}
    if escala != 'dividir_media_semilla':
        raise Detener(f'escala_entre_semillas desconocido: {escala!r}')
    ref = ref_por_semilla or maps_por_semilla
    f = {s: float(np.mean(ref[s])) for s in maps_por_semilla}
    if any(not v > 0 for v in f.values()):
        raise Detener(f'media de mapa no positiva en alguna semilla: {f}')
    return {s: maps_por_semilla[s] / f[s] for s in maps_por_semilla}, f


def media_entre_semillas(maps_por_semilla):
    """Mapa por muestra promediado sobre las semillas: (N, H, W)."""
    return np.mean(np.stack([maps_por_semilla[s] for s in maps_por_semilla]), axis=0)


def yhat_media(p_por_semilla, umbral=0.5):
    """§7: ŷ = 1 si la media de p_dep de las semillas > 0.5."""
    return (np.mean(np.stack([np.asarray(p_por_semilla[s]) for s in p_por_semilla]), 0) > umbral).astype(int)


def clase_predicha(p_por_semilla, modo):
    """{semilla: ŷ}: la de la media de 5 semillas (igual en todas) o la de cada semilla."""
    if modo == 'yhat_media_5_semillas':
        y = yhat_media(p_por_semilla)
        return {s: y for s in p_por_semilla}
    if modo == 'yhat_por_semilla':
        return {s: (np.asarray(p) > 0.5).astype(int) for s, p in p_por_semilla.items()}
    raise Detener(f'clase_predicha desconocido: {modo!r}')


def medias_agrupadas(maps_por_semilla, clase_por_semilla):
    """Media por clase de todos los mapas (semilla, muestra) de esa clase."""
    M = np.concatenate([maps_por_semilla[s] for s in maps_por_semilla])
    C = np.concatenate([np.asarray(clase_por_semilla[s]) for s in maps_por_semilla])
    return medias_por_clase(M, C)


def fraccion_constantes(const_por_semilla, clase_por_semilla):
    """Fracción de mapas constantes por clase: por semilla y agrupada."""
    out = {'por_semilla': {}, 'agrupada': {}}
    for s in const_por_semilla:
        k, c = np.asarray(const_por_semilla[s]), np.asarray(clase_por_semilla[s]).astype(int)
        out['por_semilla'][str(s)] = {str(y): {'n': int((c == y).sum()), 'constantes': int(k[c == y].sum()),
                                               'fraccion': float(k[c == y].mean()) if (c == y).any() else None}
                                      for y in (0, 1)}
    K = np.concatenate([np.asarray(const_por_semilla[s]) for s in const_por_semilla])
    C = np.concatenate([np.asarray(clase_por_semilla[s]).astype(int) for s in const_por_semilla])
    out['agrupada'] = {str(y): {'n': int((C == y).sum()), 'constantes': int(K[C == y].sum()),
                                'fraccion': float(K[C == y].mean()) if (C == y).any() else None} for y in (0, 1)}
    return out


def sensibilidad_sin_constantes(norm_por_semilla, const_por_semilla, clase_por_semilla, unidad,
                                minimo=MIN_NO_CONSTANTES):
    """Media por clase de los mapas normalizados sin los constantes. Si una clase
    tiene < minimo mapas no constantes en la unidad, su media es None (se
    reporta el conteo). unidad: 'agrupado' (las 5 semillas juntas) o
    'por_semilla' (cada semilla; la global es la media de las semillas que
    alcanzan el mínimo)."""
    def una(maps, const, clase):
        res, cnt = {}, {}
        for y in (0, 1):
            m = (np.asarray(clase).astype(int) == y) & ~np.asarray(const)
            cnt[y] = int(m.sum())
            res[y] = np.asarray(maps)[m].mean(0) if cnt[y] >= minimo else None
        return res, cnt

    seeds = list(norm_por_semilla)
    if unidad == 'agrupado':
        M = np.concatenate([norm_por_semilla[s] for s in seeds])
        K = np.concatenate([const_por_semilla[s] for s in seeds])
        C = np.concatenate([clase_por_semilla[s] for s in seeds])
        medias, cnt = una(M, K, C)
        return {'unidad': unidad, 'medias': medias, 'n_no_constantes': cnt}
    if unidad != 'por_semilla':
        raise Detener(f'umbral_constantes_unidad desconocido: {unidad!r}')
    per = {s: una(norm_por_semilla[s], const_por_semilla[s], clase_por_semilla[s]) for s in seeds}
    medias, usadas = {}, {}
    for y in (0, 1):
        ok = [per[s][0][y] for s in seeds if per[s][0][y] is not None]
        usadas[y] = int(len(ok))
        medias[y] = np.mean(ok, 0) if ok else None
    return {'unidad': unidad, 'medias': medias, 'semillas_usadas': usadas,
            'n_no_constantes': {str(s): per[s][1] for s in seeds}}


# ── Perfil radial (Gray-Scott) y terciles (Lorenz) ───────────────────────
def anillos(size=SIZE, n=N_ANILLOS, ancho=ANCHO_ANILLO):
    """Índice de anillo por píxel desde el centro geométrico ((size − 1)/2);
    -1 fuera del último anillo (radio ≥ n·ancho)."""
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - c, xx - c)
    idx = np.floor(r / ancho).astype(int)
    idx[r >= n * ancho] = -1
    return idx


def perfil_radial(m, idx=None):
    """Media del mapa (H, W) en cada anillo."""
    idx = anillos(m.shape[0]) if idx is None else idx
    return [float(m[idx == i].mean()) for i in range(idx.max() + 1)]


def cortes_terciles(u4_train):
    """Cortes de terciles de u4 en train (np.quantile, interpolación lineal)."""
    return [float(q) for q in np.quantile(np.asarray(u4_train, dtype=np.float64), [1 / 3, 2 / 3])]


def tercil(u4, cortes):
    """0: u4 ≤ c1; 1: c1 < u4 ≤ c2; 2: u4 > c2."""
    return np.searchsorted(np.asarray(cortes), np.asarray(u4, dtype=np.float64), side='left')


# ── Prueba de permutación ─────────────────────────────────────────────────
def estadistico_l2(maps, y):
    flat = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1)
    y = np.asarray(y).astype(int)
    d = flat[y == 1].mean(0) - flat[y == 0].mean(0)
    return float(np.sqrt((d * d).sum()))


def prueba_permutacion(maps, y, rng, B=PERM_B, regla='k_mas_1_sobre_B_mas_1', bloque=250):
    """Permutación de etiquetas: T = ‖media(y=1) − media(y=0)‖₂. Cada réplica es
    rng.permutation(y). k = réplicas con T ≥ T_obs (tolerancia relativa 1e-12)."""
    flat = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1)
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        raise Detener('la prueba de permutación necesita las dos clases')
    tot = flat.sum(0)
    t_obs = estadistico_l2(flat, y)
    perms = np.stack([rng.permutation(y) for _ in range(B)]).astype(np.float64)
    T = np.empty(B)
    for i in range(0, B, bloque):
        S1 = perms[i:i + bloque] @ flat
        D = S1 / n1 - (tot - S1) / n0
        T[i:i + bloque] = np.sqrt((D * D).sum(1))
    k = int((T >= t_obs * (1 - 1e-12)).sum())
    if regla == 'k_mas_1_sobre_B_mas_1':
        p = (k + 1) / (B + 1)
    elif regla == 'k_sobre_B':
        p = k / B
    else:
        raise Detener(f'perm_p desconocido: {regla!r}')
    return {'t_obs': t_obs, 'k': k, 'B': B, 'p': p, 'perm_media': float(T.mean()), 'perm_de': float(T.std(ddof=1)),
            'n1': n1, 'n0': n0}


def rngs_permutacion(modo, techniques=TECHNIQUES, seed=PERM_SEED):
    """{técnica: Generator}: uno nuevo por técnica, o uno compartido en el orden de techniques."""
    if modo == 'por_tecnica':
        return {t: np.random.default_rng(seed) for t in techniques}
    if modo == 'secuencial':
        g = np.random.default_rng(seed)
        return {t: g for t in techniques}
    raise Detener(f'perm_rng desconocido: {modo!r}')


def holm(pvalues, alpha=ALPHA):
    """Holm de la Fase 5 (src.protocol.verdict.holm) más el p ajustado."""
    from src.protocol.verdict import holm as vholm
    out = vholm(pvalues, alpha)
    m = len(pvalues)
    orden = sorted(pvalues, key=pvalues.get)
    run = 0.0
    for i, k in enumerate(orden):
        run = max(run, min(1.0, (m - i) * pvalues[k]))
        out[k] = {**out[k], 'p': pvalues[k], 'p_holm': run}
    return out


# ── Cosenos de cabezas ────────────────────────────────────────────────────
def coseno(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else math.nan


def vectores_cabeza(W):
    """W (2, d) de la capa lineal final: w1 y w1 − w0 (sin sesgo)."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != 2:
        raise Detener(f'cabeza con forma {W.shape}; se espera (2, d)')
    return {'w1': W[1], 'w1_menos_w0': W[1] - W[0]}


def cosenos_cabezas(reales, permutadas, res):
    """reales, permutadas: {semilla: W}. Coseno real–permutado de la misma semilla
    y referencia de los 10 pares de cabezas reales, para w1 y w1 − w0."""
    out = {}
    for v in ('w1', 'w1_menos_w0'):
        rp = {s: coseno(vectores_cabeza(reales[s])[v], vectores_cabeza(permutadas[s])[v]) for s in reales}
        ref = {f'{a}-{b}': coseno(vectores_cabeza(reales[a])[v], vectores_cabeza(reales[b])[v])
               for a, b in pares_semillas(tuple(reales))}
        c = combinar_semillas(list(rp.values()), list(ref.values()), res['criterio_semillas'], res['sobre_maximo'])
        out[v] = {'real_vs_permutada': {str(s): x for s, x in rp.items()}, 'referencia': ref,
                  'rango_referencia': c['rango_referencia'],
                  'por_semilla': [COSENO_NIVEL[p] for p in c['por_semilla']], 'nivel': COSENO_NIVEL[c['global']]}
    return out


# ── Lectura de los ViT (plan, «Lectura de los ViT») ───────────────────────
def nivel_coseno(cos, modo):
    """Nivel que decide la lectura: el de w1 − w0, el de w1, o el común si ambos coinciden."""
    if modo in ('w1_menos_w0', 'w1'):
        return cos[modo]['nivel']
    if modo == 'ambos':
        a, b = cos['w1']['nivel'], cos['w1_menos_w0']['nivel']
        return a if a == b else 'indeterminado'
    raise Detener(f'coseno_lectura desconocido: {modo!r}')


def lectura_vit(cordura_datos, coseno_nivel, cabeza):
    """cordura_datos: 'cambia' | 'no_cambia' | 'indeterminado' (mapa real frente
    al permutado); coseno_nivel: 'alto' | 'bajo' | 'indeterminado'; cabeza:
    'cambia' | 'no_cambia' | 'indeterminado' | None (paso «cabeza» de la cascada)."""
    if cordura_datos == 'cambia':
        return 'cordura_por_datos_superada'
    if cordura_datos != 'no_cambia':
        return 'indeterminado'
    if coseno_nivel == 'bajo':
        return 'atencion_del_backbone'
    if coseno_nivel != 'alto':
        return 'indeterminado'
    if cabeza is None:
        return 'pendiente_de_la_cascada'
    if cabeza == 'cambia':
        return 'dependiente_de_la_cabeza_sin_validacion_por_datos'
    if cabeza == 'no_cambia':
        return 'atencion_del_backbone'
    return 'indeterminado'


# ── Gray-Scott: correlación de mapas frente a la de p_dep ─────────────────
def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else math.nan


def seguimiento_gs(corr_mapas, corr_pdep):
    """{semilla: valor} de las dos correlaciones; su asociación entre semillas (n = 5)."""
    from scipy.stats import spearmanr
    seeds = list(corr_mapas)
    a, b = [corr_mapas[s] for s in seeds], [corr_pdep[s] for s in seeds]
    return {'semillas': seeds, 'corr_mapas': a, 'corr_pdep': b, 'pearson_entre_semillas': pearson(a, b),
            'spearman_entre_semillas': float(spearmanr(a, b).statistic), 'n': len(seeds)}
