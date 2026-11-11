"""F4.3, recepción de la parte GPU: «Verificación local» de docs/correr_f43_kaggle.md.

Uso: python results/f43_atribucion/recepcion/verificacion_local.py
Escribe verificacion_local.json junto a este archivo y sale con 1 si algo falla.
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
GPU = RAIZ / "results/f43_atribucion/gpu"
RES = RAIZ / "results/f43_atribucion/resoluciones.json"
COMMIT = "660985c36d1f118b166428c247c903670f946ca4"
SALIDAS = ["cascada.json", "cascada_similitudes.npz", "cascada_medias.npz",
           "ablacion.json", "ablacion.csv", "atribucion_gpu_meta.json"]
SEMILLAS = ["42", "7", "123", "2024", "99"]
TOL_PDEP, TOL_MAPA = 1e-4, 1e-2
ORDEN = {
    "gray_scott": ["cabeza"] + [f"features.{i}" for i in range(8, -1, -1)],
    "perlin": ["cabeza", "encoder.ln"] + [f"encoder.layers.{i}" for i in range(11, -1, -1)] + ["embeddings"],
}
ORDEN["lorenz"] = ORDEN["perlin"]
CONDICIONES = ["original"] + [f"paso_{b}_s{s}" for b in ("bajo", "alto") for s in (1, 2, 4, 8)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    chk = {}
    meta = json.loads((GPU / "atribucion_gpu_meta.json").read_text(encoding="utf-8"))
    chk["commit_txt"] = (GPU / "commit.txt").read_text().strip() == COMMIT
    chk["meta_commit"] = meta["git_commit"] == COMMIT
    chk["meta_smoke_false"] = meta["smoke"] is False and meta["limit"] is None
    chk["meta_split_val"] = meta["split"] == "val"
    blob = subprocess.run(["git", "show", f"{COMMIT}:results/f43_atribucion/resoluciones.json"],
                          cwd=RAIZ, capture_output=True, check=True).stdout
    chk["resoluciones_sha256"] = (meta["resoluciones"]["sha256"] == hashlib.sha256(blob).hexdigest()
                                  == sha(RES))
    chk["datos_manifiesto_ok"] = "manifiesto OK" in (GPU / "datos_verificacion.log").read_text()

    # sha256 de los modelos frente al manifiesto de artefactos
    manif = {Path(r["rel_path"]).name: r["sha256"] for r in csv.DictReader(open(RAIZ / "meta/artefactos_manifest.csv", encoding="utf-8"))}
    modelos = {}
    for t, info in meta["tecnicas"].items():
        for s in SEMILLAS:
            m = info["modelos"][s]
            vs = {k: v["max_abs_diff_normalizado"] for k, v in m["vs_npz_f37"].items()}
            modelos[f"{t}|{s}"] = {
                "sha256_manifiesto": manif.get(m["archivo"]) == m["sha256"],
                "p_dep_max_abs_diff": m["p_dep_max_abs_diff"],
                "vs_npz_f37_max": max(vs.values()),
                "mapas_comparados": sorted(vs),
                "ablacion_p_dep_original_max_abs_diff": m["ablacion_p_dep_original_max_abs_diff"],
                "ok": (manif.get(m["archivo"]) == m["sha256"] and m["p_dep_max_abs_diff"] <= TOL_PDEP
                       and max(vs.values()) <= TOL_MAPA
                       and m["ablacion_p_dep_original_max_abs_diff"] <= TOL_PDEP
                       and (set(vs) == {"map", "backbone_map"} if t != "gray_scott" else set(vs) == {"map"})),
            }
    chk["modelos_15_ok"] = len(modelos) == 15 and all(v["ok"] for v in modelos.values())

    mv = json.loads((GPU / "modelos_verificacion.json").read_text(encoding="utf-8"))
    pv = json.loads((GPU / "mapas_verificacion.json").read_text(encoding="utf-8"))
    chk["modelos_verificacion_15_15"] = mv["n_expected"] == mv["n_ok"] == 15 and not mv["missing"] and not mv["mismatch"]
    chk["mapas_verificacion_30_30"] = pv["n_expected"] == pv["n_ok"] == 30 and not pv["missing"] and not pv["mismatch"]

    sums = {}
    for linea in (GPU / "f43_sha256sum.txt").read_text().splitlines():
        h, f = linea.split()
        sums[f] = h
    chk["sha256sum"] = sorted(sums) == sorted(SALIDAS) and all(sha(GPU / f) == h for f, h in sums.items())

    cas = json.loads((GPU / "cascada.json").read_text(encoding="utf-8"))
    chk["cascada_meta"] = cas["meta"]["git_commit"] == COMMIT and cas["meta"]["smoke"] is False
    orden_ok, pares_ok = {}, True
    for t, esperado in ORDEN.items():
        ct = cas["tecnicas"][t]
        orden_ok[t] = ct["orden"] == esperado and list(ct["pasos"]) == esperado
        for paso in ct["pasos"].values():
            for mapa, medidas in paso.items():
                if mapa == "modulos":
                    continue
                for med in medidas.values():
                    pares_ok &= len(med["referencia_pares"]) == 10
    chk["cascada_orden"] = all(orden_ok.values())
    chk["cascada_10_pares"] = bool(pares_ok)

    abl = json.loads((GPU / "ablacion.json").read_text(encoding="utf-8"))
    chk["ablacion_9_condiciones"] = all(list(abl["tecnicas"][t]) == CONDICIONES for t in ORDEN)
    chk["ablacion_original_delta_0"] = all(abl["tecnicas"][t]["original"]["delta_f1_media"] == 0 for t in ORDEN)
    with open(GPU / "ablacion.csv", encoding="utf-8") as f:
        chk["ablacion_csv_135_filas"] = sum(1 for _ in csv.DictReader(f)) == 15 * 9

    out = {"tarea": "F4.3", "parte": "recepcion_gpu", "commit_entregado": COMMIT,
           "tolerancias": {"p_dep": TOL_PDEP, "mapa_normalizado": TOL_MAPA},
           "comprobaciones": chk, "cascada_orden_por_tecnica": orden_ok, "modelos": modelos,
           "segundos_gpu": meta["segundos_total"], "todo_ok": all(chk.values())}
    (Path(__file__).parent / "verificacion_local.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for k, v in chk.items():
        print(f"{'OK ' if v else 'FALLA'} {k}")
    print("todo_ok:", out["todo_ok"])
    return 0 if out["todo_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
