"""
verificar_imagenes_vs_manifiesto.py
───────────────────────────────────
Verificación retrospectiva de F0.1 (ejecutada el 2026-11-05, no al cierre de la tarea).

Solo lectura. Recalcula el SHA-256 de cada PNG de
TT_DATA_ROOT/images_grayscale_classblind/{perlin,gray_scott,lorenz} y lo compara,
por técnica, con las filas IMAGES[<técnica>] de meta/data_manifest.csv (F1.5).
Reporta coincidencias, discrepancias de hash, faltantes, sobrantes y el
dir_digest recalculado frente al registrado.

No ejecuta src/generation/compute_checksums.py (escribiría en data/images/).

Uso (desde la raíz del repo):
    python results/f01_checksums/verificacion_retrospectiva/verificar_imagenes_vs_manifiesto.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src import paths                                   # noqa: E402
from src.build_manifest import dir_digest, read_manifest  # noqa: E402
from src.training.umap_io import sha256                  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    root = paths.data_root()
    manifest = read_manifest(paths.DATA_MANIFEST)
    lines = [f"TT_DATA_ROOT = {root}",
             f"Manifiesto  = {paths.DATA_MANIFEST.relative_to(REPO).as_posix()}", ""]
    summary = []
    detail = []
    for tech in paths.TECHNIQUES:
        logical = f"IMAGES[{tech}]"
        rel_dir = paths.REL['IMAGES'][tech]
        exp_files = {r['rel_path']: r['sha256'] for r in manifest
                     if r['logical_name'] == logical and r['kind'] == 'file'}
        exp_dir = [r for r in manifest if r['logical_name'] == logical and r['kind'] == 'dir']
        files, ignored = paths.list_files(root / rel_dir)
        present = {f"{rel_dir}/{p.name}": p for p in files}

        actual_rows = []
        status = Counter()
        for rel in sorted(set(exp_files) | set(present)):
            if rel not in present:
                st, h = 'faltante', ''
            else:
                h = sha256(present[rel])
                actual_rows.append({'rel_path': rel, 'sha256': h})
                if rel not in exp_files:
                    st = 'sobrante'
                elif h == exp_files[rel]:
                    st = 'coincide'
                else:
                    st = 'difiere'
            status[st] += 1
            if st != 'coincide':
                detail.append(f"  {tech}: {st} {rel}")

        dig = dir_digest(actual_rows)
        dig_exp = exp_dir[0]['sha256'] if exp_dir else ''
        row = {'technique': tech,
               'n_manifiesto': len(exp_files),
               'n_insumos': len(present),
               'coinciden': status['coincide'],
               'difieren': status['difiere'],
               'faltantes': status['faltante'],
               'sobrantes': status['sobrante'],
               'dir_digest_manifiesto': dig_exp,
               'dir_digest_recalculado': dig,
               'dir_digest_coincide': dig == dig_exp}
        summary.append(row)
        lines.append(f"{logical:<20} manifiesto={row['n_manifiesto']} insumos={row['n_insumos']} "
                     f"coinciden={row['coinciden']} difieren={row['difieren']} "
                     f"faltantes={row['faltantes']} sobrantes={row['sobrantes']}")
        lines.append(f"{'':<20} dir_digest manifiesto  {dig_exp}")
        lines.append(f"{'':<20} dir_digest recalculado {dig}  "
                     f"({'coincide' if row['dir_digest_coincide'] else 'NO coincide'})")
        if ignored:
            lines.append(f"{'':<20} ignorados (archivos de sistema): {', '.join(p.name for p in ignored)}")

    total = sum(r['coinciden'] for r in summary)
    n = sum(r['n_manifiesto'] for r in summary)
    lines += ["", f"TOTAL coinciden {total} / {n}"]
    if detail:
        lines += ["", "Detalle de no coincidencias:"] + detail

    text = "\n".join(lines) + "\n"
    print(text, end="")
    (OUT_DIR / "verificacion_imagenes.log").write_text(text, encoding="utf-8")
    with open(OUT_DIR / "verificacion_por_tecnica.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
