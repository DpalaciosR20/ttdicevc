"""
tests/test_label_independence.py
════════════════════════════════
F0.3 — Prueba de independencia de etiqueta de los generadores class-blind.
Plan de Trabajo TT2 (v2) · Rol SV

Hipótesis bajo prueba: cada imagen es X = G(u), función determinista de los
cinco componentes UMAP, y no de la etiqueta, el índice, el split ni la ruta.

Capa A (estática)   Ninguna función de renderizado recibe ni referencia la
                    etiqueta, el índice o el split; ninguna semilla aleatoria
                    depende de algo distinto de los propios valores u.
Capa B (dinámica)   Con el mismo u, la imagen es idéntica bit a bit al variar
                    UN solo factor: etiqueta, split/tag, ruta de salida o
                    posición en el lote; en Gray-Scott GPU, también la
                    composición del lote.
Capa C (artefactos) Los PNG almacenados están completos, cumplen el contrato
                    (1 canal, 256×256), su etiqueta coincide con
                    umap_params.pkl y con el PKL de Exp 4.1, y una muestra
                    estratificada se reproduce bit a bit desde u.

Ejecución (Colab):
    !pip install -q noise pytest
    %cd /content/drive/MyDrive/Repo
    !python -m pytest tests/test_label_independence.py -v -rs -s

Variables de entorno opcionales:
    PROJECT_ROOT     raíz del repositorio (por omisión, el padre de tests/)
    F03_N_PER_CELL   muestras por celda split × clase (por omisión 1 → 6 por técnica)
    F03_SEED         semilla del muestreo estratificado (por omisión 2026)
    F03_GS_PATH      ruta de producción de Gray-Scott para la capa C: gpu | cpu
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
import sys
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image

# ── Rutas (relativas a PROJECT_ROOT) ──────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
GEN_DIR = PROJECT_ROOT / "src" / "generators"
UMAP_PKL = PROJECT_ROOT / "data" / "shared" / "umap_params.pkl"
PKL_41 = PROJECT_ROOT / "models" / "best_model" / "Exp_4.1_BETO_HPO_embeddings.pkl"
IMG_ROOT = PROJECT_ROOT / "data" / "images"
sys.path.insert(0, str(GEN_DIR))

# ── Configuración ─────────────────────────────────────────────────────────
N_PER_CELL = int(os.environ.get("F03_N_PER_CELL", "1"))
SAMPLE_SEED = int(os.environ.get("F03_SEED", "2026"))
GS_PATH = os.environ.get("F03_GS_PATH", "gpu").lower()
SPLITS = ("train", "val", "test")
TAG_TO_SPLIT = {"train": "train", "val": "val", "validation": "val", "test": "test"}
IMG_SIZE = 256
N_TOTAL = 1136

FORBIDDEN = {"label", "labels", "y_true", "target", "clase",
             "idx", "index", "split", "split_name", "tag"}
RNG_CALLS = {"default_rng", "seed", "manual_seed", "RandomState", "Generator", "PCG64"}

RENDERERS = {
    "perlin_lib": ["map_umap_to_perlin", "seed_from_umap", "generate_noise_grid",
                   "generate_flow_field", "trace_particles",
                   "draw_trajectories_grayscale", "generate_perlin_image_classblind"],
    "lorenz_lib": ["map_umap_to_lorenz", "lorenz_system", "solve_attractor",
                   "compute_velocity", "normalize_array", "render_attractor_enriched"],
    "gray_scott_lib": ["umap_to_rd_params", "gray_scott_step", "run_gray_scott",
                       "run_gray_scott_safe", "_fixed_noise", "_simulate_batch_gpu"],
}

# técnica almacenada → (subdirectorio, prefijo de archivo, motor de regeneración)
STORED = {
    "perlin": ("perlin", "perlin", "perlin"),
    "lorenz": ("attractor", "lorenz", "lorenz"),
    "gray_scott": ("reaction_diffusion", "rd", f"gs_{GS_PATH}"),
}
FNAME_RE = re.compile(
    r"^(?P<prefix>perlin|lorenz|rd)_(?P<tag>.+)_(?P<idx>\d{4})_label(?P<label>\d+)\.png$")

FACTORS = ("label", "split", "route", "index")
ENGINES = ("perlin", "lorenz", "gs_cpu", "gs_gpu")


# ── Utilidades ────────────────────────────────────────────────────────────
def _import(modname):
    if modname == "perlin_lib":
        pytest.importorskip("noise", reason="perlin_lib requiere el paquete `noise`")
    if not (GEN_DIR / f"{modname}.py").exists():
        pytest.fail(f"No existe {GEN_DIR / (modname + '.py')} (entregable de F0.1)")
    return importlib.import_module(modname)


def _read(paths):
    return [np.array(Image.open(p)) for p in paths]


def _diff(a, b):
    if a.shape != b.shape:
        return f"forma {a.shape} vs {b.shape}"
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return f"{(d > 0).mean():.4%} px distintos · max|Δ| = {d.max()} · MAE = {d.mean():.4f}"


def _engine(name):
    """run(U, y, tag, out_dir) -> list[path], usando la función de lote de producción."""
    if name == "perlin":
        m = _import("perlin_lib")
        return lambda U, y, tag, d: m.generate_batch(U, y, tag, str(d))
    if name == "lorenz":
        m = _import("lorenz_lib")
        return lambda U, y, tag, d: m.generate_batch(U, y, tag, str(d))
    if name == "gs_cpu":
        m = _import("gray_scott_lib")
        return lambda U, y, tag, d: m.generate_batch_cpu(U, y, tag, str(d))
    if name == "gs_gpu":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Sin GPU CUDA: se omite la ruta GPU de Gray-Scott")
        m = _import("gray_scott_lib")
        dev = torch.device("cuda")
        return lambda U, y, tag, d: m.generate_batch_gpu(U, y, tag, str(d), device=dev)
    raise ValueError(name)


def _inventario(tech):
    """Indexa los PNG almacenados por (split, idx) a partir del nombre de archivo."""
    subdir, prefix, _ = STORED[tech]
    root = IMG_ROOT / subdir
    if not root.exists():
        pytest.fail(f"No existe {root}")
    inv, problemas = {}, []
    for p in sorted(root.rglob("*.png")):
        m = FNAME_RE.match(p.name)
        if not m or m["prefix"] != prefix:
            problemas.append(f"nombre no reconocido: {p.relative_to(PROJECT_ROOT)}")
            continue
        split = TAG_TO_SPLIT.get(m["tag"])
        if split is None:
            problemas.append(f"tag desconocido «{m['tag']}»: {p.name}")
            continue
        key = (split, int(m["idx"]))
        if key in inv:
            problemas.append(f"duplicado {key}: {p.name}")
            continue
        inv[key] = (p, int(m["label"]))
    return inv, problemas


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def umap_pkg():
    if not UMAP_PKL.exists():
        pytest.fail(f"No existe {UMAP_PKL}")
    try:
        pkg = joblib.load(UMAP_PKL)
    except Exception as e:  # bug conocido de Numba al deserializar UMAP
        pytest.fail(f"joblib.load(umap_params.pkl) falló ({type(e).__name__}: {e}). "
                    "Depende de F0.2.")
    for s in SPLITS:
        assert s in pkg, f"Falta el split '{s}'; llaves: {list(pkg.keys())}"
        assert {"params", "labels"} <= set(pkg[s].keys()), \
            f"pkg['{s}'] sin 'params'/'labels': {list(pkg[s].keys())}"
        P, y = np.asarray(pkg[s]["params"]), np.asarray(pkg[s]["labels"])
        assert P.ndim == 2 and P.shape[1] == 5, f"{s}: params con forma {P.shape}"
        assert len(P) == len(y), f"{s}: {len(P)} params vs {len(y)} etiquetas"
        assert set(np.unique(y).tolist()) <= {0, 1}, f"{s}: etiquetas no binarias"
        assert P.min() >= 0.0 and P.max() <= 1.0, f"{s}: u fuera de [0, 1]"
    assert sum(len(pkg[s]["labels"]) for s in SPLITS) == N_TOTAL
    return pkg


@pytest.fixture(scope="module")
def sample(umap_pkg):
    """Muestra estratificada split × clase, fija por semilla."""
    rng = np.random.default_rng(SAMPLE_SEED)
    out = []
    for s in SPLITS:
        y = np.asarray(umap_pkg[s]["labels"]).astype(int)
        for c in (0, 1):
            pool = np.flatnonzero(y == c)
            pick = rng.choice(pool, size=min(N_PER_CELL, len(pool)), replace=False)
            out += [(s, int(i)) for i in np.sort(pick)]
    return out


@pytest.fixture(scope="module")
def U_y(umap_pkg, sample):
    # Se conserva el dtype almacenado: castear alteraría la aritmética de Gray-Scott.
    U = np.stack([np.asarray(umap_pkg[s]["params"])[i] for s, i in sample])
    y = np.array([int(np.asarray(umap_pkg[s]["labels"])[i]) for s, i in sample])
    return U, y


@pytest.fixture(scope="module")
def renders(U_y, tmp_path_factory):
    """Genera, una sola vez por motor, la referencia A y una variante por factor."""
    U, y = U_y
    k = len(y)
    perm = np.roll(np.arange(k), 1)   # permutación sin puntos fijos
    inv = np.argsort(perm)            # inv[i] = posición de la muestra i en U[perm]
    cache = {}

    def get(engine):
        if engine not in cache:
            run = _engine(engine)
            base = tmp_path_factory.mktemp(engine)
            pa = run(U, y, "train", base / "A")
            out = {"A": _read(pa), "A_paths": pa}
            # Cada variante cambia UN solo factor respecto de A.
            out["label"] = _read(run(U, 1 - y, "train", base / "A"))
            out["split"] = _read(run(U, y, "zz_split", base / "A"))
            out["route"] = _read(run(U, y, "train", base / "D"))
            # Directorio nuevo obligatorio: generate_batch* salta archivos existentes por nombre.
            p = _read(run(U[perm], y[perm], "train", base / "E"))
            out["index"] = [p[inv[i]] for i in range(k)]
            cache[engine] = out
        return cache[engine]

    return get


# ── Capa A: estática ──────────────────────────────────────────────────────
@pytest.mark.parametrize("modname", list(RENDERERS))
def test_A_estatica_renderers(modname):
    m = _import(modname)
    problemas = []
    for fname in RENDERERS[modname]:
        fn = getattr(m, fname, None)
        if fn is None:
            problemas.append(f"{fname}: no existe en {modname}")
            continue
        firma = set(inspect.signature(fn).parameters)
        if firma & FORBIDDEN:
            problemas.append(f"{fname}: parámetros prohibidos {sorted(firma & FORBIDDEN)}")
        arbol = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        nombres = {n.id for n in ast.walk(arbol) if isinstance(n, ast.Name)}
        if nombres & FORBIDDEN:
            problemas.append(f"{fname}: referencia a {sorted(nombres & FORBIDDEN)}")
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            f = nodo.func
            callee = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if callee in RNG_CALLS:
                args = [*nodo.args, *(kw.value for kw in nodo.keywords)]
                deps = {n.id for a in args for n in ast.walk(a) if isinstance(n, ast.Name)}
                if deps - {"seed"}:
                    problemas.append(
                        f"{fname}: semilla aleatoria depende de {sorted(deps - {'seed'})}")
    assert not problemas, f"{modname}:\n" + "\n".join(problemas)


# ── Capa B: dinámica ──────────────────────────────────────────────────────
@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("engine", ENGINES)
def test_B_invarianza(renders, sample, engine, factor):
    out = renders(engine)
    fallas = [f"{s}[{i}]: {_diff(a, b)}"
              for (s, i), a, b in zip(sample, out["A"], out[factor])
              if not np.array_equal(a, b)]
    assert not fallas, (f"{engine}: la imagen cambia al variar «{factor}» con el mismo u:\n"
                        + "\n".join(fallas))


@pytest.mark.parametrize("engine", ENGINES)
def test_B_contrato_salida(renders, engine):
    out = renders(engine)
    for p in out["A_paths"]:
        with Image.open(p) as im:
            assert im.mode == "L" and im.size == (IMG_SIZE, IMG_SIZE), \
                f"{p}: {im.mode} {im.size} (contrato: L {IMG_SIZE}×{IMG_SIZE})"
    assert all(a.std() > 0 for a in out["A"]), f"{engine}: imagen constante (degenerada)"


def test_B_perlin_funcion_pura(renders, U_y):
    """La imagen guardada por el lote coincide con la función pura G(u)."""
    m = _import("perlin_lib")
    U, _ = U_y
    for j, (u, a) in enumerate(zip(U, renders("perlin")["A"])):
        g = m.generate_perlin_image_classblind(u)
        assert np.array_equal(g, a), f"muestra {j}: {_diff(g, a)}"


def test_B_gray_scott_gpu_composicion_lote(renders, U_y, tmp_path_factory):
    """La imagen de una muestra no depende de con qué otras muestras comparte lote."""
    run = _engine("gs_gpu")
    U, y = U_y
    ref = renders("gs_gpu")["A"]
    base = tmp_path_factory.mktemp("gs_gpu_solo")
    fallas = []
    for j in range(len(y)):
        solo = _read(run(U[j:j + 1], y[j:j + 1], "solo", base / f"s{j}"))[0]
        if not np.array_equal(solo, ref[j]):
            fallas.append(f"muestra {j}: {_diff(solo, ref[j])}")
    assert not fallas, "Gray-Scott GPU depende de la composición del lote:\n" + "\n".join(fallas)


def test_diagnostico_gray_scott_cpu_vs_gpu(renders, sample):
    """Informativo (no es criterio de independencia): documenta si las rutas CPU y GPU coinciden."""
    cpu, gpu = renders("gs_cpu")["A"], renders("gs_gpu")["A"]
    for (s, i), a, b in zip(sample, cpu, gpu):
        print(f"[GS CPU vs GPU] {s}[{i}]: {'idénticas' if np.array_equal(a, b) else _diff(a, b)}")


# ── Capa C: artefactos almacenados ───────────────────────────────────────
@pytest.mark.parametrize("tech", list(STORED))
def test_C_inventario_y_alineacion(tech, umap_pkg):
    inv, problemas = _inventario(tech)
    for s in SPLITS:
        y = np.asarray(umap_pkg[s]["labels"]).astype(int)
        idxs = sorted(i for (sp, i) in inv if sp == s)
        if idxs != list(range(len(y))):
            problemas.append(f"{s}: {len(idxs)} índices presentes; se esperan 0…{len(y) - 1}")
        for i in idxs:
            if i < len(y) and inv[(s, i)][1] != y[i]:
                problemas.append(f"{s}[{i}]: etiqueta en el nombre {inv[(s, i)][1]} "
                                 f"≠ etiqueta en umap_params.pkl {y[i]}")
    for (s, i), (p, _) in inv.items():
        with Image.open(p) as im:
            if im.mode != "L" or im.size != (IMG_SIZE, IMG_SIZE):
                problemas.append(f"{p.name}: {im.mode} {im.size} (contrato: L 256×256)")
    assert not problemas, (f"{tech}: {len(problemas)} problema(s):\n"
                           + "\n".join(problemas[:50]))


@pytest.mark.parametrize("tech", list(STORED))
def test_C_reproduccion_desde_u(tech, sample, renders):
    inv, _ = _inventario(tech)
    motor = STORED[tech][2]
    regen = renders(motor)["A"]
    fallas = []
    for (s, i), a in zip(sample, regen):
        if (s, i) not in inv:
            fallas.append(f"{s}[{i}]: no hay archivo almacenado")
            continue
        b = np.array(Image.open(inv[(s, i)][0]))
        if not np.array_equal(a, b):
            fallas.append(f"{s}[{i}]: {_diff(a, b)}")
    assert not fallas, (
        f"{tech}: las imágenes almacenadas no se reproducen bit a bit desde u con "
        f"src/generators/ (motor {motor}):\n" + "\n".join(fallas)
        + "\nResolver con F0.1/F0.2 antes de calcular checksums.")


def test_C_alineacion_con_pkl_41(umap_pkg):
    """El orden de filas de umap_params.pkl coincide con el del PKL de Exp 4.1."""
    if not PKL_41.exists():
        pytest.skip(f"No existe {PKL_41}")
    pkg_41 = joblib.load(PKL_41)
    assert "X_val" in pkg_41.keys()
    for s, (xk, yk) in {"train": ("X_train", "y_train"),
                        "val": ("X_val", "y_val"),
                        "test": ("X_test", "y_test")}.items():
        assert len(pkg_41[xk]) == len(umap_pkg[s]["params"]), f"{s}: tamaños distintos"
        assert np.array_equal(np.asarray(pkg_41[yk]).astype(int),
                              np.asarray(umap_pkg[s]["labels"]).astype(int)), \
            f"{s}: el orden de etiquetas difiere entre el PKL de Exp 4.1 y umap_params.pkl"