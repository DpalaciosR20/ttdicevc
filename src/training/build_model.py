"""
build_model.py
───────────────
Construccion de las 3 arquitecturas de clasificacion usadas en el Modulo 3
de TT2 (ResNet-18, EfficientNet-B0, ViT-B-16), con soporte para las 2
estrategias de transfer learning (frozen / fine-tuning completo).

Atributos de capa final verificados por ejecucion real (no supuestos):
  - resnet18:         .fc              (in_features=512)
  - efficientnet_b0:  .classifier[-1]  (in_features=1280)
  - vit_b_16:          .heads.head     (in_features=768)

Nota de tiempo: ViT-B-16 tiene ~85.8M parametros (7.7x ResNet-18, 21x
EfficientNet-B0) -- considerablemente mas lento de entrenar, sobre todo en
fine-tuning completo. Tenerlo en cuenta al planear el grid de experimentos.
"""

import torch.nn as nn
from torchvision import models

def build_model(arch_name, freeze_backbone=False, num_classes=2):
    """
    Construye un modelo de las 3 arquitecturas soportadas, con la capa final
    ya reemplazada para 'num_classes'. Si freeze_backbone=True, congela todo
    menos la capa final (transfer learning tipo 'frozen').

    arch_name: 'resnet18' | 'efficientnet_b0' | 'vit_b_16'
    """
    if arch_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif arch_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise ValueError(f"Arquitectura desconocida: '{arch_name}'. Usa 'resnet18', 'efficientnet_b0' o 'vit_b_16'.")

    return model


def get_trainable_params(model):
    """Devuelve solo los parametros con requires_grad=True (para el optimizador)."""
    return [p for p in model.parameters() if p.requires_grad]
