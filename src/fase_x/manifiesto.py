"""
manifiesto.py
─────────────
FX.1: escribe <dir>/manifiesto.csv con la ruta relativa, los bytes y el SHA-256 de
cada archivo de una carpeta de salida (results/fx_pixeles/local o gpu), sin incluirse
a sí mismo. Se corre una vez terminada la corrida.

Uso: python -m src.fase_x.manifiesto results/fx_pixeles/gpu
"""
import csv
import os
import sys

from src.verify_artefacts import sha256


def escribir(carpeta):
    out = os.path.join(carpeta, 'manifiesto.csv')
    filas = []
    for raiz, _, archivos in os.walk(carpeta):
        for a in sorted(archivos):
            p = os.path.join(raiz, a)
            if os.path.abspath(p) == os.path.abspath(out):
                continue
            filas.append([os.path.relpath(p, carpeta).replace(os.sep, '/'), os.path.getsize(p), sha256(p)])
    filas.sort()
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['archivo', 'bytes', 'sha256'])
        w.writerows(filas)
    return out, len(filas)


if __name__ == '__main__':
    ruta, n = escribir(sys.argv[1])
    print(f'{n} archivos -> {ruta}')
