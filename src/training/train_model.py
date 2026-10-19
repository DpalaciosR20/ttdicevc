"""
train_model.py
───────────────
Entrenamiento generico (cualquiera de las 3 arquitecturas de build_model.py)
con la misma logica ya validada para ResNet-18+Gray-Scott en el camera-ready
(F1-Macro=0.7142+-0.0095, 5 semillas): Adam, CrossEntropyLoss con pesos de
clase, early stopping por F1-val, historial de loss/F1 registrado por epoca.

Uso tipico:
    from data import make_loaders
    train_loader, val_loader, test_loader, class_weights = make_loaders(
        image_dir, 'gray_scott', seed=42)
    model, history, results = train_model(
        arch_name='resnet18',       # o 'efficientnet_b0', 'vit_b_16'
        freeze_backbone=False,      # o True
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        class_weights=class_weights, device=device, seed=42
    )
    results['f1_test'], results['test_preds']  # preds -> McNemar

lr por defecto (si lr=None), tomado de la busqueda del camera-ready
(docs/resumen_modulo3_camera_ready.md, seccion 4):
  - fine-tuning completo: 3e-5
  - frozen (solo capa final): 1e-3  (con 3e-5 la cabeza lineal casi no se
    mueve en 15 epocas y el frozen saldria artificialmente peor)
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

try:
    from .build_model import build_model, get_head, get_trainable_params
except ImportError:
    from build_model import build_model, get_head, get_trainable_params

DEFAULT_LR = {False: 3e-5, True: 1e-3}  # clave: freeze_backbone


def set_seed(seed):
    """Fija todas las fuentes de aleatoriedad conocidas. Con cudnn
    determinista, la misma GPU da el mismo resultado; entre GPUs/entornos
    distintos puede seguir variando (ver docs/hallazgos_metodologicos.md, 6)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_train_mode(model, arch_name, freeze_backbone):
    """En frozen, todo el modelo queda en eval() (BatchNorm no actualiza sus
    running stats, dropout apagado) y solo la cabeza en train(). Es un linear
    probe real sobre features fijas."""
    if freeze_backbone:
        model.eval()
        get_head(model, arch_name).train()
    else:
        model.train()


def _predict(model, loader, device):
    model.eval()
    preds, probs, true = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            true.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(true)


def train_model(arch_name, freeze_backbone, train_loader, val_loader, test_loader,
                 class_weights, device, lr=None, n_epochs=15, patience=4, seed=None):
    """
    Entrena un modelo (arch_name, freeze_backbone) con la misma logica que ya
    se uso para ResNet-18 en el camera-ready: Adam, CrossEntropyLoss con pesos
    de clase, early stopping por F1-val, batch ya definido en los loaders.

    La seleccion del mejor epoch usa SOLO F1-val; test se evalua una vez al final.

    Devuelve: (model con los mejores pesos, history, results)
      history = {'train_loss': [...], 'f1_val': [...]}
      results = {'arch', 'freeze_backbone', 'lr', 'seed', 'best_epoch' (1-based),
                 'epochs_run', 'best_f1_val', 'f1_test',
                 'val_preds', 'test_preds', 'test_probs', 'test_true'}
    """
    if lr is None:
        lr = DEFAULT_LR[freeze_backbone]
    if seed is not None:
        set_seed(seed)
        # Mismo orden de barajado para la misma semilla aunque el loader se reuse
        gen = getattr(train_loader, 'generator', None)
        if gen is not None:
            gen.manual_seed(seed)

    model = build_model(arch_name, freeze_backbone=freeze_backbone)
    model = model.to(device)

    optimizer = torch.optim.Adam(get_trainable_params(model), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {'train_loss': [], 'f1_val': []}
    best_f1_val, best_epoch, best_state, best_val_preds, no_improve = -1.0, 0, None, None, 0
    tag = f'[{arch_name}, freeze={freeze_backbone}, lr={lr:g}]'

    for epoch in range(n_epochs):
        _set_train_mode(model, arch_name, freeze_backbone)
        running_loss = 0.0
        n_samples = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

        val_preds, _, val_true = _predict(model, val_loader, device)
        f1_val = f1_score(val_true, val_preds, average='macro')

        history['train_loss'].append(running_loss / n_samples)
        history['f1_val'].append(f1_val)

        print(f'{tag} Epoch {epoch+1}/{n_epochs} '
              f'- loss={running_loss/n_samples:.4f} - F1-val={f1_val:.4f}')

        if f1_val > best_f1_val:
            best_f1_val, best_epoch, best_val_preds = f1_val, epoch + 1, val_preds
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping en epoch {epoch+1} (mejor F1-val={best_f1_val:.4f})')
                break

    model.load_state_dict(best_state)

    test_preds, test_probs, test_true = _predict(model, test_loader, device)
    f1_test = f1_score(test_true, test_preds, average='macro')

    print(f'*** {tag} F1-val(best, epoch {best_epoch})={best_f1_val:.4f}  F1-test={f1_test:.4f} ***\n')

    results = {
        'arch': arch_name, 'freeze_backbone': freeze_backbone, 'lr': lr, 'seed': seed,
        'best_epoch': best_epoch, 'epochs_run': len(history['f1_val']),
        'best_f1_val': best_f1_val, 'f1_test': f1_test,
        'val_preds': best_val_preds, 'test_preds': test_preds,
        'test_probs': test_probs, 'test_true': test_true,
    }
    return model, history, results
