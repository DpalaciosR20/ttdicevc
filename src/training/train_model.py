"""
train_model.py
───────────────
Entrenamiento generico (cualquiera de las 3 arquitecturas de build_model.py)
con la misma logica ya validada para ResNet-18+Gray-Scott en el camera-ready
(F1-Macro=0.7142+-0.0095, 5 semillas): Adam, CrossEntropyLoss con pesos de
clase, early stopping por F1-val, historial de loss/F1 registrado por epoca.

Uso tipico:
    model, history, f1_test = train_model(
        arch_name='resnet18',       # o 'efficientnet_b0', 'vit_b_16'
        freeze_backbone=False,      # o True
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        class_weights=class_weights, device=device, seed=42
    )
"""

import copy
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from build_model import build_model, get_trainable_params


def train_model(arch_name, freeze_backbone, train_loader, val_loader, test_loader,
                 class_weights, device, lr=3e-5, n_epochs=15, patience=4, seed=None):
    """
    Entrena un modelo (arch_name, freeze_backbone) con la misma logica que ya
    se uso para ResNet-18 en el camera-ready: Adam, CrossEntropyLoss con pesos
    de clase, early stopping por F1-val, batch ya definido en los loaders.

    Devuelve: (model entrenado con los mejores pesos, history dict, f1_test)
    history = {'train_loss': [...], 'f1_val': [...]}
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = build_model(arch_name, freeze_backbone=freeze_backbone)
    model = model.to(device)

    optimizer = torch.optim.Adam(get_trainable_params(model), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {'train_loss': [], 'f1_val': []}
    best_f1_val, best_state, no_improve = 0.0, None, 0

    for epoch in range(n_epochs):
        model.train()
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

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.numpy())
        f1_val = f1_score(val_true, val_preds, average='macro')

        history['train_loss'].append(running_loss / n_samples)
        history['f1_val'].append(f1_val)

        print(f'[{arch_name}, freeze={freeze_backbone}] Epoch {epoch+1}/{n_epochs} '
              f'- loss={running_loss/n_samples:.4f} - F1-val={f1_val:.4f}')

        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping en epoch {epoch+1} (mejor F1-val={best_f1_val:.4f})')
                break

    model.load_state_dict(best_state)

    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.numpy())
    f1_test = f1_score(test_true, test_preds, average='macro')

    print(f'*** [{arch_name}, freeze={freeze_backbone}] F1-val(best)={best_f1_val:.4f}  F1-test={f1_test:.4f} ***\n')

    return model, history, f1_test
