"""F4.3, recepción de la parte local corrida en Kaggle CPU: «Verificación local» de
docs/correr_f43_local_kaggle.md.

Uso: python results/f43_atribucion/recepcion/verificacion_local_cpu.py
Escribe verificacion_local_cpu.json junto a este archivo y sale con 1 si algo falla.
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
LOC = RAIZ / "results/f43_atribucion/local"
COMMIT = "95813042a618ae3097bec3a80aef8683004475d4"
SALIDAS = ["resultados.json", "medias_clase.npz", "similitudes_por_muestra.csv"]
PREFIJOS = ("ttdicevc/results/f37_mapas/maps/", "ttdicevc/results/f35_final/", "ttdicevc/results/f36_permutacion/")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def blob_sha(rev, path):
    b = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=RAIZ, capture_output=True, check=True).stdout
    return hashlib.sha256(b).hexdigest()


def main():
    chk = {}
    doc = json.loads((LOC / "resultados.json").read_text(encoding="utf-8"))
    meta, r = doc["meta"], doc["resultados"]
    chk["commit_txt"] = (LOC / "commit.txt").read_text().strip() == COMMIT
    chk["meta_commit"] = meta["git_commit"] == COMMIT
    chk["resoluciones_sha256"] = meta["resoluciones"]["sha256"] == blob_sha(COMMIT, "results/f43_atribucion/resoluciones.json")
    chk["cascada_sha256"] = (meta["cascada"]["sha256"] == blob_sha(COMMIT, "results/f43_atribucion/gpu/cascada.json")
                             == sha(RAIZ / "results/f43_atribucion/gpu/cascada.json"))
    chk["datos_manifiesto_ok"] = "manifiesto OK" in (LOC / "datos_verificacion.log").read_text()

    manif = {r_["logical_name"]: r_ for r_ in csv.DictReader(open(RAIZ / "meta/artefactos_manifest.csv", encoding="utf-8"))}
    av = meta["artefactos_verificados"]
    n_npz = sum(1 for k in av if k.startswith("MAPS["))
    n_pt = sum(1 for k in av if k.startswith("MODEL["))
    chk["artefactos_50"] = len(av) == 50 and n_npz == 30 and n_pt == 20
    chk["artefactos_sha_manifiesto"] = all(manif[k]["sha256"] == v["sha256"] and int(manif[k]["size_bytes"]) == v["bytes"]
                                           for k, v in av.items())
    chk["artefactos_rutas"] = all(any(p in v["archivo"] for p in PREFIJOS) and "f37_humo" not in v["archivo"]
                                  for v in av.values())

    for nombre, n in (("modelos_f35_verificacion.json", 15), ("modelos_f36_verificacion.json", 15), ("mapas_verificacion.json", 30)):
        v = json.loads((LOC / nombre).read_text(encoding="utf-8"))
        chk[f"{nombre}_{n}_{n}"] = v["n_expected"] == v["n_ok"] == n and not v["missing"] and not v["mismatch"] and not v["ambiguous"]

    sums = dict(reversed(l.split()) for l in (LOC / "f43_local_sha256sum.txt").read_text().splitlines())
    chk["sha256sum"] = sorted(sums) == sorted(SALIDAS) and all(sha(LOC / f) == h for f, h in sums.items())
    chk["meta_sha_salidas"] = (meta["medias_clase_sha256"] == sha(LOC / "medias_clase.npz")
                               and meta["similitudes_sha256"] == sha(LOC / "similitudes_por_muestra.csv"))

    chk["holm_m3"] = r["holm"]["m"] == 3 and sorted(r["holm"]["familia"]) == ["gray_scott", "lorenz", "perlin"]
    chk["lectura_vit"] = (sorted(r["lectura_vit"]) == ["lorenz", "perlin"]
                          and all(v["cabeza_cascada"] is not None for v in r["lectura_vit"].values()))
    chk["permutacion_B"] = all(r["tecnicas"][t]["permutacion_val"].get("B", 5000) == 5000 for t in r["tecnicas"])

    out = {"tarea": "F4.3", "parte": "recepcion_local_kaggle_cpu", "commit_entregado": COMMIT,
           "comprobaciones": chk, "segundos": meta["segundos"], "todo_ok": all(chk.values())}
    (Path(__file__).parent / "verificacion_local_cpu.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for k, v in chk.items():
        print(f"{'OK ' if v else 'FALLA'} {k}")
    print("segundos:", meta["segundos"], "todo_ok:", out["todo_ok"])
    return 0 if out["todo_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
