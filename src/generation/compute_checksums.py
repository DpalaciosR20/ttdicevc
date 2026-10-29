"""
src/generators/compute_checksums.py
═══════════════════════════════════
F0.1 (CE) — SHA-256 de las 3,408 imágenes class-blind.

Ejecutar SOLO después de que tests/test_label_independence.py pase la capa C
(test_C_reproduccion_desde_u). Antes de eso, el checksum congelaría un estado
no verificado.

Salidas:
    data/images/checksums.csv       technique, split, idx, label, relpath, sha256
    data/images/checksums_env.json  versiones del entorno que produjo/verificó las imágenes

Uso (Colab):
    %cd /content/drive/MyDrive/Repo
    !python src/generators/compute_checksums.py
"""
import csv
import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
IMG_ROOT = PROJECT_ROOT / "data" / "images"
TECHS = {"perlin": "perlin", "reaction_diffusion": "rd", "attractor": "lorenz"}
TAG_TO_SPLIT = {"train": "train", "val": "val", "validation": "val", "test": "test"}
FNAME_RE = re.compile(r"^(?P<prefix>perlin|lorenz|rd)_(?P<tag>.+)_(?P<idx>\d{4})_label(?P<label>\d+)\.png$")
N_PER_TECH = 1136


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rows = []
    for tech, prefix in TECHS.items():
        files = sorted((IMG_ROOT / tech).rglob("*.png"))
        n = 0
        for p in files:
            m = FNAME_RE.match(p.name)
            if not m or m["prefix"] != prefix or m["tag"] not in TAG_TO_SPLIT:
                raise ValueError(f"Nombre no reconocido: {p}")
            rows.append({"technique": tech, "split": TAG_TO_SPLIT[m["tag"]],
                         "idx": int(m["idx"]), "label": int(m["label"]),
                         "relpath": p.relative_to(PROJECT_ROOT).as_posix(),
                         "sha256": sha256(p)})
            n += 1
        assert n == N_PER_TECH, f"{tech}: {n} imágenes; se esperan {N_PER_TECH}"
        print(f"{tech}: {n} imágenes")

    rows.sort(key=lambda r: (r["technique"], r["split"], r["idx"]))
    out = IMG_ROOT / "checksums.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    env = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "matplotlib", "Pillow", "noise", "torch", "umap-learn", "numba"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        env["gpu"] = None
    (IMG_ROOT / "checksums_env.json").write_text(json.dumps(env, indent=2))
    print(f"{len(rows)} checksums → {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()