"""
contar_parametros.py
────────────────────
F1.3 — Conteo de parámetros entrenables y totales de las 6 combinaciones
arquitectura × estrategia, con el mismo reemplazo de cabeza y el mismo
congelado que src/training/build_model.py (líneas 36-55), pero con
weights=None: los conteos no dependen de los pesos y así no se descarga nada.

No importa build_model.py (que fija weights=IMAGENET1K_V1); replica su lógica.
Compara el resultado con el bloque «parametros» de configs/matriz_experimentos.json
y termina con código 1 si difieren.

Uso (desde la raíz del repo, CPU):
    python results/f13_matriz/contar_parametros.py | tee results/f13_matriz/contar_parametros.log
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import models

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs' / 'matriz_experimentos.json'

# nombre F1.5 -> (constructor, atributo de la cabeza)
ARCHS = {
    'resnet18': (models.resnet18, 'fc'),
    'efficientnet_b0': (models.efficientnet_b0, 'classifier[-1]'),
    'vit_b16': (models.vit_b_16, 'heads.head'),
}


def build(arch, frozen):
    ctor, head = ARCHS[arch]
    m = ctor(weights=None)
    if frozen:
        for p in m.parameters():
            p.requires_grad = False
    if head == 'fc':
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif head == 'classifier[-1]':
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    else:
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    return m


def main():
    print(f'torch {torch.__version__}, torchvision {torchvision.__version__}, python {sys.version.split()[0]}')
    expected = json.loads(CONFIG.read_text(encoding='utf-8'))['parametros']
    bad = 0
    print(f'{"arch":<16}{"strategy":<10}{"trainable":>12}{"total":>12}{"MiB fp32":>10}  JSON')
    for arch in ARCHS:
        for strategy in ('frozen', 'finetune'):
            m = build(arch, strategy == 'frozen')
            tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in m.parameters())
            ok = expected[arch][strategy] == {'trainable': tr, 'total': tot}
            bad += not ok
            print(f'{arch:<16}{strategy:<10}{tr:>12}{tot:>12}{tot * 4 / 2**20:>10.1f}  {"OK" if ok else "DIFIERE"}')
    print('coinciden con configs/matriz_experimentos.json' if not bad else f'{bad} combinaciones difieren')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
