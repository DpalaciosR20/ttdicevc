"""
convergencia.py
───────────────
F4.4 — Convergencia con la Tabla 11 (docs/plan_fase4.md, F4.4). Explicativo
(protocolo §9); sin pruebas de hipótesis, solo IC.

Para cada uno de los 8 descriptores significativos de la Tabla 11 recalculada
(results/f24_descriptores/tabla11_recalculada.csv, sig_holm = True):
  1. El descriptor se calcula por parche sobre las imágenes de val, con las
     funciones de F2.4 (results/f24_descriptores/f24_descriptores.py, copia
     literal de la celda 20; no se reescriben), aplicadas al recorte. Rejilla de
     7 × 7 en Gray-Scott (EfficientNet-B0) y de 14 × 14 en Perlin y Lorenz (ViT).
  2. La masa del mapa de F3.7 por parche: map_raw · map_raw_scale de val, cada
     semilla dividida entre su media en val (dividir_media_semilla, como F4.3) y
     media de las 5 semillas por muestra; suma por parche.
  3. Spearman por muestra entre descriptor y masa (rangos promedio); media sobre
     las muestras de val con IC percentil al 95 % por bootstrap sobre muestras
     (B = 10,000, default_rng(2026)).
Además, sin cálculos nuevos:
  - tabla cualitativa ablación (f43.*) ↔ descriptores (|d| de la Tabla 11);
  - contraste visual − descriptores de la Fase 5 (A11, f5.frente_a_descriptores.*), citado;
  - advertencia de Lorenz desde f43.lorenz.lectura_vit.*.

Solo val (G2): las imágenes pasan por filas_val / nombres_val de atribucion_gpu,
y de los .npz de F3.7 solo se usa el split val. No hay inferencia de modelos.

Las decisiones que el plan no fija están en RESOLUCIONES (PENDIENTES de
results/f44_convergencia/reporte_e4a.md); el script no corre sin --resoluciones.

Uso:
  python -m src.fase4.convergencia --root <tt2-artefactos> --resoluciones <json> [--out results/f44_convergencia] [--limit N]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / 'src' / 'training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.fase4 import atribucion_comun as ac  # noqa: E402
from src.fase4.atribucion_comun import Detener  # noqa: E402

TABLA11 = REPO / 'results' / 'f24_descriptores' / 'tabla11_recalculada.csv'
F24_SCRIPT = 'results/f24_descriptores/f24_descriptores.py'
CANON = REPO / 'results' / 'canonical_numbers.json'
OUT = REPO / 'results' / 'f44_convergencia'
TECNICAS = ('gray_scott', 'perlin', 'lorenz')
REJILLA = {'gray_scott': 7, 'perlin': 14, 'lorenz': 14}
ESPERADOS = {'gray_scott': ('shannon_entropy', 'contrast', 'edge_density', 'lbp_energy'),
             'perlin': ('contrast', 'edge_density'),
             'lorenz': ('shannon_entropy', 'lbp_energy')}
B = 10000
SEMILLA_BOOT = 2026
NIVEL = 0.95
NATIVA, MAPA = 256, 224

RESOLUCIONES = {
    # P1: sobre qué imagen y con qué fronteras se recorta el parche del descriptor
    'geometria_parche': ('nativa_256_proporcional', 'redimensionada_224'),
    # P2: Spearman indefinido (masa o descriptor constante en todos los parches)
    'spearman_indefinido': ('excluir_y_contar', 'cero'),
    # P3: qué mapa de F3.7 da la masa
    'mapa': ('map', 'map_y_backbone'),
    # P4: réplicas bootstrap: una matriz de índices por técnica (compartida entre
    # descriptores) o un generador nuevo por (técnica, descriptor)
    'bootstrap_rng': ('por_tecnica_compartido', 'por_tecnica_descriptor'),
    # P5: cuantiles del IC percentil
    'ic_cuantiles': ('linear', 'inverted_cdf'),
    # P6: naturaleza espectral de cada descriptor en la tabla cualitativa
    'naturaleza_descriptores': ('propuesta', 'sin_clasificar'),
}

# P6, opción 'propuesta': lectura de la definición de cada descriptor (configs/protocolo.json → descriptores).
# Clasificación orientativa; el contraste (DE de niveles de gris) no se asigna a ninguna banda (resolución de SV).
NATURALEZA = {
    'shannon_entropy': 'histograma de niveles de gris (sin estructura espacial)',
    'contrast': 'dispersión global de intensidades (sin banda específica)',
    'edge_density': 'bordes de Sobel (frecuencias altas)',
    'lbp_energy': 'textura local, vecindad de radio 3 px (frecuencias altas y medias)',
}


# ── Resoluciones ──────────────────────────────────────────────────────────
def check_resoluciones(res):
    res = {k: v for k, v in res.items() if not k.startswith('_')}
    falta = [k for k in RESOLUCIONES if k not in res]
    sobra = [k for k in res if k not in RESOLUCIONES]
    if falta or sobra:
        raise Detener(f'resoluciones: faltan {falta}, sobran {sobra}')
    for k, v in res.items():
        if v not in RESOLUCIONES[k]:
            raise Detener(f'resoluciones: {k} = {v!r} no es una de {RESOLUCIONES[k]}')
    return res


# ── Descriptores de F2.4 (sin reescribirlos) ──────────────────────────────
_F24 = None


def f24():
    """Módulo de F2.4 cargado desde su archivo versionado."""
    global _F24
    if _F24 is None:
        spec = importlib.util.spec_from_file_location('f24_descriptores', REPO / F24_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _F24 = mod
    return _F24


def funcion_descriptor(nombre):
    m = f24()
    if nombre not in m.FEATURES:
        raise Detener(f'{nombre} no es un descriptor de F2.4')
    return getattr(m, nombre)


def descriptores_significativos(path=TABLA11):
    """{técnica: (descriptor, ...)} y {(técnica, descriptor): fila} de la Tabla 11 (sig_holm = True)."""
    with open(path, newline='', encoding='utf-8') as f:
        filas = list(csv.DictReader(f))
    sig = {t: tuple(r['descriptor'] for r in filas if r['technique'] == t and r['sig_holm'] == 'True') for t in TECNICAS}
    if sig != ESPERADOS:
        raise Detener(f'Tabla 11: descriptores significativos {sig} != {ESPERADOS} (docs/plan_fase4.md, F4.4)')
    return sig, {(r['technique'], r['descriptor']): r for r in filas}


# ── Geometría de parches ──────────────────────────────────────────────────
def limites(n, g):
    """Fronteras de g parches sobre n píxeles: round(i·n/g), i = 0..g (exactas si g divide n)."""
    return np.round(np.arange(g + 1) * n / g).astype(int)


def recortes(img, g):
    """Recortes g × g de la imagen, por filas."""
    h, w = img.shape
    fy, fx = limites(h, g), limites(w, g)
    return [[img[fy[i]:fy[i + 1], fx[j]:fx[j + 1]] for j in range(g)] for i in range(g)]


def descriptor_por_parche(img, g, fn):
    """(g, g): fn de F2.4 aplicada a cada recorte (uint8)."""
    if img.dtype != np.uint8:
        raise Detener(f'los descriptores de F2.4 esperan uint8; se recibió {img.dtype}')
    return np.array([[fn(c) for c in fila] for fila in recortes(img, g)], dtype=np.float64)


def redimensionar_224(img, prep):
    """La imagen como la ve el modelo antes de normalizar (Resize de F1.4), en uint8."""
    from PIL import Image
    from torchvision import transforms
    from data import INTERPOLATION
    tf = transforms.Resize((prep['size'], prep['size']), interpolation=INTERPOLATION[prep['interpolation']],
                           antialias=prep['antialias'])
    out = np.array(tf(Image.fromarray(img, 'L')))
    if out.shape != (MAPA, MAPA) or out.dtype != np.uint8:
        raise Detener(f'redimensionado: {out.shape} {out.dtype}')
    return out


def imagen_para_parche(img, res, prep):
    if res['geometria_parche'] == 'nativa_256_proporcional':
        return img
    return redimensionar_224(img, prep)


def masa_por_parche(mapas, g):
    """(N, 224, 224) -> (N, g, g): suma del mapa en cada parche (224 es múltiplo de 7 y 14)."""
    k = mapas.shape[1] // g
    return ac.pool(mapas, g) * (k * k)


# ── Estadística ───────────────────────────────────────────────────────────
def spearman_por_muestra(a, b):
    """Spearman por muestra entre (N, g, g) y (N, g, g); NaN si alguno es constante."""
    from scipy.stats import rankdata
    ra = rankdata(np.asarray(a, np.float64).reshape(len(a), -1), axis=1)
    rb = rankdata(np.asarray(b, np.float64).reshape(len(b), -1), axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)


def tratar_indefinidos(rho, modo):
    """P2: 'excluir_y_contar' deja los NaN (no entran en las medias); 'cero' los cambia por 0."""
    rho = np.asarray(rho, np.float64)
    nan = np.isnan(rho)
    if modo == 'excluir_y_contar':
        return rho, int(nan.sum())
    if modo == 'cero':
        return np.where(nan, 0.0, rho), int(nan.sum())
    raise Detener(f'spearman_indefinido desconocido: {modo!r}')


def indices_bootstrap(n, rng, b=B):
    return rng.integers(0, n, size=(b, n))


def ic_bootstrap(rho, idx, nivel=NIVEL, metodo='linear'):
    """Media de rho e IC percentil bilateral con las réplicas idx (B, N) sobre las N muestras.
    Los NaN (Spearman indefinido, ya tratados según P2) no entran en la media de cada réplica."""
    rho = np.asarray(rho, np.float64)
    if idx.shape[1] != len(rho):
        raise Detener(f'réplicas de tamaño {idx.shape[1]} para {len(rho)} valores')
    validos = ~np.isnan(rho)
    if not validos.any():
        raise Detener('ningún Spearman definido')
    x, w = np.where(validos, rho, 0.0)[idx], validos[idx]
    with np.errstate(invalid='ignore', divide='ignore'):
        medias = x.sum(1) / w.sum(1)
    medias = medias[np.isfinite(medias)]
    a = (1 - nivel) / 2
    lo, hi = np.quantile(medias, [a, 1 - a], method=metodo)
    return {'media': float(rho[validos].mean()), 'ic_inf': float(lo), 'ic_sup': float(hi),
            'n': int(validos.sum()), 'b': int(idx.shape[0]), 'b_validas': int(len(medias)), 'nivel': nivel,
            'contiene_0': bool(lo <= 0 <= hi), 'sobre_0': bool(lo > 0), 'bajo_0': bool(hi < 0)}


# ── Contexto sin cálculos nuevos ──────────────────────────────────────────
def _canon_val(numbers, nid, split='val'):
    v = numbers[nid]['value']
    return v[split] if isinstance(v, dict) else v


def identidad(numbers, prefijo):
    ks = [k[len(prefijo) + 1:] for k in numbers if k.startswith(prefijo + '.')]
    if len(ks) != 1:
        raise Detener(f'{prefijo}: se esperaba una clave de identidad, hay {ks}')
    return ks[0]


def tabla_cualitativa(numbers, sig, filas11, res):
    """Por técnica: ΔF1 de la ablación (f43.*) y sus descriptores significativos con |d|."""
    out = {}
    for t in TECNICAS:
        abl = {banda: {f's{s}': _canon_val(numbers, f'f43.{t}.ablacion.{banda}.s{s}.delta_f1') for s in ac.SIGMAS}
               for banda in ('paso_bajo', 'paso_alto')}
        desc = []
        for d in sig[t]:
            r = filas11[(t, d)]
            desc.append({'descriptor': d, 'cohen_d': float(r['cohen_d']), 'abs_d': abs(float(r['cohen_d'])),
                         'p_holm': float(r['p_holm']),
                         'naturaleza': NATURALEZA[d] if res['naturaleza_descriptores'] == 'propuesta' else None})
        out[t] = {'f1_base': _canon_val(numbers, f'f43.{t}.ablacion.original.f1_macro_media'),
                  'delta_f1': abl, 'descriptores': sorted(desc, key=lambda x: -x['abs_d']),
                  'fuente': f'f43.{t}.ablacion.*; {TABLA11.relative_to(REPO).as_posix()}'}
    return out


def contexto_a11(numbers):
    """Contraste visual − descriptores de la Fase 5 (A11): solo se cita."""
    out = {}
    for t in TECNICAS:
        b = f'f5.frente_a_descriptores.{t}'
        out[t] = {k: _canon_val(numbers, f'{b}{suf}', 'test') for k, suf in
                  (('delta', ''), ('ic_inf', '_ic_inf'), ('ic_sup', '_ic_sup'))}
        out[t]['clave'] = b
    return out


def advertencias(numbers):
    """Lectura de F4.3 por técnica ViT (f43.<t>.lectura_vit.*)."""
    return {t: identidad(numbers, f'f43.{t}.lectura_vit') for t in ac.VIT}


# ── Análisis (puro) ───────────────────────────────────────────────────────
def analizar(desc, masas, sig, res, b=B):
    """desc: {t: {d: (N, g, g)}}; masas: {t: {mapa: (N, g, g)}}. Devuelve resultados y filas por muestra.
    P4 'por_tecnica_compartido': una matriz de índices (B, N) por técnica, de default_rng(2026), común a sus
    descriptores y mapas; 'por_tecnica_descriptor': default_rng(2026) nuevo por (técnica, mapa, descriptor)."""
    out, filas = {}, []
    for t in desc:
        n = len(next(iter(desc[t].values())))
        idx_t = indices_bootstrap(n, np.random.default_rng(SEMILLA_BOOT), b)
        out[t] = {}
        for k, masa in masas[t].items():
            out[t][k] = {}
            for d in sig[t]:
                rho = spearman_por_muestra(desc[t][d], masa)
                for i, x in enumerate(rho):
                    filas.append({'tecnica': t, 'mapa': k, 'descriptor': d, 'muestra': i,
                                  'spearman': '' if np.isnan(x) else float(x)})
                v, n_indef = tratar_indefinidos(rho, res['spearman_indefinido'])
                idx = idx_t if res['bootstrap_rng'] == 'por_tecnica_compartido' else                     indices_bootstrap(n, np.random.default_rng(SEMILLA_BOOT), b)
                out[t][k][d] = {**ic_bootstrap(v, idx, metodo=res['ic_cuantiles']), 'n_indefinidos': n_indef}
    return out, filas


# ── Carga (I/O) ───────────────────────────────────────────────────────────
def cargar_imagenes_val(tech, limit=None):
    """uint8 (N, 256, 256) de val, verificadas contra data_manifest (G2: solo val)."""
    from src import paths
    from src.fase4 import atribucion_gpu as ag
    from src.fase4 import caracteristicas as car
    rows = ag.filas_val(car.load_index())
    if limit:
        rows = rows[:limit]
    expected, _ = car.image_rows(car.load_data_manifest(), tech)
    datos, hashes = ag.leer_imagenes_val(str(paths.images_dir(tech)), paths.REL['IMAGES'][tech], rows, tech, expected)
    huella = hashlib.sha256(''.join(f'{k}\t{v}\n' for k, v in sorted(hashes.items())).encode()).hexdigest()
    return np.stack([ag.decodificar(x) for x in datos]), [r['sample_id'] for r in rows], huella


def preprocesamiento(tech):
    from src.fase4 import caracteristicas as car
    runs = {s: json.load(open(car.run_json_path(tech, s), encoding='utf-8')) for s in ac.SEEDS}
    prep = runs[42]['preprocessing']
    if any(r['preprocessing'] != prep for r in runs.values()):
        raise Detener(f'{tech}: preprocesamiento distinto entre semillas')
    return prep


def cargar_masas_val(ub, tech, res, index, limit=None):
    """{mapa: (N, 224, 224)} de val: crudo, dividido por la media de la semilla y promediado entre semillas."""
    from src.fase4 import atribucion_local as al
    kinds = ['map'] + (['backbone_map'] if res['mapa'] == 'map_y_backbone' and tech in ac.VIT else [])
    crudo = {k: {} for k in kinds}
    ids = None
    for s in ac.SEEDS:
        name = ac.run_name(tech, s, False)
        rec = json.loads((al.RECORDS / f'{name}.json').read_text(encoding='utf-8'))
        p, row = ub.ruta(al.logical_maps(tech, s, False))
        if row['sha256'] != rec['npz']['sha256']:
            raise Detener(f'{name}: sha256 del manifiesto distinto del registro de F3.7')
        d = al.cargar_npz(p, rec, index)['val']          # solo val
        if ids is not None and d['sample_id'] != ids:
            raise Detener(f'{name}: sample_id de val distinto entre semillas')
        ids = d['sample_id']
        crudo['map'][s] = d['map_crudo']
        if 'backbone_map' in crudo:
            crudo['backbone_map'][s] = d['backbone_crudo']
    out = {}
    for k, por_semilla in crudo.items():
        esc, _ = ac.escalar_semillas(por_semilla, 'dividir_media_semilla')
        m = ac.media_entre_semillas(esc)
        out[k] = m[:limit] if limit else m
    return out, (ids[:limit] if limit else ids)


def main(argv=None):
    ap = argparse.ArgumentParser(description='F4.4: convergencia descriptor–mapa por parche (val)')
    ap.add_argument('--root', required=True, help='copia de Drive MiUnidad/tt2-artefactos/ o /kaggle/input')
    ap.add_argument('--resoluciones', required=True)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--limit', type=int, default=None, help='prueba acotada: primeras N muestras de val (no definitiva)')
    a = ap.parse_args(argv)
    raw = json.loads(Path(a.resoluciones).read_text(encoding='utf-8'))
    res = check_resoluciones(raw)
    out = Path(a.out)
    salidas = [out / 'resultados.json', out / 'spearman_por_muestra.csv', out / 'parches.npz']
    for p in salidas:
        if p.exists():
            raise Detener(f'{p} ya existe; no se sobrescribe')
    from src.fase4 import atribucion_local as al
    from src.fase4.regimen import check_gate
    commit = check_gate() if not a.limit else 'prueba_acotada'
    t0, tiempos = time.time(), {}
    sig, filas11 = descriptores_significativos()
    numbers = json.loads(CANON.read_text(encoding='utf-8'))['numbers']
    index = al.load_index()
    ub = al.Ubicador(a.root, al.load_artefactos())
    desc, masas, huellas, npz = {}, {}, {}, {}
    for t in TECNICAS:
        ti = time.time()
        imgs, ids_img, huellas[t] = cargar_imagenes_val(t, a.limit)
        prep = preprocesamiento(t)
        g = REJILLA[t]
        desc[t] = {d: np.stack([descriptor_por_parche(imagen_para_parche(im, res, prep), g, funcion_descriptor(d))
                                for im in imgs]) for d in sig[t]}
        tiempos[f'{t}_descriptores_s'] = round(time.time() - ti, 1)
        tm = time.time()
        mapas, ids_map = cargar_masas_val(ub, t, res, index, a.limit)
        if ids_map != ids_img:
            raise Detener(f'{t}: orden de muestras distinto entre imágenes y mapas')
        masas[t] = {k: masa_por_parche(m, g) for k, m in mapas.items()}
        tiempos[f'{t}_mapas_s'] = round(time.time() - tm, 1)
        for d, x in desc[t].items():
            npz[f'{t}|descriptor|{d}'] = x.astype(np.float32)
        for k, x in masas[t].items():
            npz[f'{t}|masa|{k}'] = x.astype(np.float32)
        print(f'{t}: {len(imgs)} imágenes, rejilla {g} × {g}; {tiempos}', flush=True)
    tb = time.time()
    r, filas = analizar(desc, masas, sig, res)
    tiempos['bootstrap_s'] = round(time.time() - tb, 1)
    doc = {'meta': {'tarea': 'F4.4', 'git_commit': commit, 'limit': a.limit,
                    'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'resoluciones': {'archivo': Path(a.resoluciones).as_posix(), 'sha256': al.sha256_file(a.resoluciones),
                                     **ac.meta_resoluciones(raw)},
                    'insumos': {'tabla11': al.sha256_file(TABLA11), 'canonical_numbers': al.sha256_file(CANON),
                                'f24_descriptores.py': al.sha256_file(REPO / F24_SCRIPT)},
                    'imagenes_val_sha256_lista': huellas, 'artefactos_verificados': ub.verificados,
                    'rejilla': REJILLA, 'b': B, 'semilla_bootstrap': SEMILLA_BOOT, 'segundos': tiempos,
                    'segundos_total': round(time.time() - t0, 1)},
           'resultados': {'spearman': r, 'tabla_cualitativa': tabla_cualitativa(numbers, sig, filas11, res),
                          'contexto_a11': contexto_a11(numbers), 'lectura_f43': advertencias(numbers),
                          'advertencia_lorenz': 'F4.3: mapas de Lorenz dependientes de la cabeza, sin validación '
                                                'por datos (f43.lorenz.lectura_vit.*); la convergencia de Lorenz se '
                                                'lee con esa advertencia'}}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'parches.npz', **npz)
    with open(out / 'spearman_por_muestra.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(filas)
    (out / 'resultados.json').write_text(json.dumps(ac.limpiar(doc), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(ac.limpiar({'segundos': doc['meta']['segundos_total'], 'spearman': r}), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Detener as e:
        sys.exit(f'DETENIDO: {e}')
