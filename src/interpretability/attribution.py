"""
attribution.py
──────────────
Mapas de atribución de F3.7 (matriz §8, Enmienda 3). Solo torch; sin E/S.

  - grad_cam(model, x, layer):        Grad-CAM (Selvaraju et al., 2017) sobre la
                                      salida de `layer` (ResNet-18: layer4;
                                      EfficientNet-B0: features[-1]).
  - vit_attributions(model, x):       para ViT de torchvision, en una sola pasada:
      * 'gradient_rollout': rollout ponderado por gradiente (Chefer, Gur y Wolf,
        2021): por bloque Ā = media_h (∇A ⊙ A)⁺; R ← R + Ā·R desde R = I.
      * 'attention_rollout': atención del backbone (Abnar y Zuidema, 2020):
        Â = norm_filas(media_h A + I); R = Â_L ··· Â_1.
    En ambos, el mapa es la fila del token de clase sobre los tokens de parche
    (se excluye el token de clase), reescalado a la resolución de entrada.

Objetivo: el logit de la clase `target` (1 = depresivo) en todas las muestras.
El modelo debe estar en eval(): las muestras de un lote son independientes, así
que el gradiente de la suma de logits da el gradiente de cada muestra.

Modelos congelados: las activaciones de la capa objetivo (Grad-CAM) o la entrada
del encoder (ViT) no requieren gradiente porque los parámetros del backbone
están congelados; los hooks las convierten en hojas con requires_grad=True. La
cabeza entrenada transmite el gradiente (matriz §8).

ViT: EncoderBlock llama a la atención con need_weights=False (ruta fusionada) y
los pesos devueltos con need_weights=True no están en el camino del gradiente.
Por eso, mientras dura el cálculo, la atención de cada bloque se reemplaza por
una implementación explícita equivalente (in_proj → softmax(QKᵀ/√d) → ·V →
out_proj) que conserva A en el grafo. `check_vit_equivalence` comprueba que la
salida del modelo no cambia.
"""
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────

def upsample(maps, size):
    """(B, h, w) -> (B, size, size), bilineal sin align_corners."""
    return F.interpolate(maps[:, None], size=(size, size), mode='bilinear', align_corners=False)[:, 0]


def minmax(maps):
    """Normaliza cada mapa a [0, 1]; un mapa constante queda en ceros."""
    flat = maps.reshape(maps.shape[0], -1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    rng = hi - lo
    out = torch.where(rng > 0, (flat - lo) / torch.where(rng > 0, rng, torch.ones_like(rng)), torch.zeros_like(flat))
    return out.reshape(maps.shape)


def _leaf_requiring_grad(t):
    return t if t.requires_grad else t.detach().requires_grad_(True)


# ─────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────

def resolve_layer(model, spec):
    """'layer4' -> model.layer4; 'features[-1]' -> model.features[-1]."""
    name, _, idx = spec.partition('[')
    mod = getattr(model, name)
    return mod[int(idx.rstrip(']'))] if idx else mod


def grad_cam(model, x, layer, target=1):
    """Devuelve (cam (B, H, W) a la resolución de x, sin normalizar; logits)."""
    store = {}

    def hook(_m, _inp, out):
        out = _leaf_requiring_grad(out)
        store['a'] = out
        return out

    h = layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            a = store['a']
            g, = torch.autograd.grad(logits[:, target].sum(), a)
    finally:
        h.remove()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1)).detach()
    return upsample(cam, x.shape[-1]), logits.detach()


# ─────────────────────────────────────────────────────────────
# ViT
# ─────────────────────────────────────────────────────────────

class ExplicitAttention(nn.Module):
    """Sustituto de nn.MultiheadAttention (batch_first, autoatención) que guarda
    las probabilidades de atención A (B, h, N, N) con su lugar en el grafo."""

    def __init__(self, mha, store):
        super().__init__()
        self.mha, self.store = mha, store

    def forward(self, query, key, value, need_weights=False, **_):
        m = self.mha
        assert m.batch_first and query is key is value and m._qkv_same_embed_dim
        B, N, D = query.shape
        h = m.num_heads
        d = D // h
        q, k, v = F.linear(query, m.in_proj_weight, m.in_proj_bias).split(D, dim=-1)
        q, k, v = (t.reshape(B, N, h, d).transpose(1, 2) for t in (q, k, v))
        attn = torch.softmax((q @ k.transpose(-2, -1)) * d ** -0.5, dim=-1)
        self.store.append(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return m.out_proj(out), None


@contextmanager
def explicit_attention(model):
    """Reemplaza la atención de cada bloque del encoder; restaura al salir.
    Hace que la entrada del encoder requiera gradiente (backbone congelado)."""
    store = []
    blocks = list(model.encoder.layers)
    originals = [b.self_attention for b in blocks]
    pre = model.encoder.register_forward_pre_hook(lambda _m, args: (_leaf_requiring_grad(args[0]),))
    for b, mha in zip(blocks, originals):
        b.self_attention = ExplicitAttention(mha, store)
    try:
        yield store
    finally:
        pre.remove()
        for b, mha in zip(blocks, originals):
            b.self_attention = mha


def gradient_rollout(attns, grads):
    """Chefer et al. (2021): R ← R + Ā·R, Ā = media_h (∇A ⊙ A)⁺, R0 = I."""
    B, _, N, _ = attns[0].shape
    R = torch.eye(N, dtype=attns[0].dtype, device=attns[0].device).expand(B, N, N).clone()
    for a, g in zip(attns, grads):
        abar = (g * a).clamp(min=0).mean(1)
        R = R + abar @ R
    return R


def attention_rollout(attns):
    """Abnar y Zuidema (2020): Â = norm_filas(media_h A + I); R = Â_L ··· Â_1."""
    B, _, N, _ = attns[0].shape
    eye = torch.eye(N, dtype=attns[0].dtype, device=attns[0].device)
    R = eye.expand(B, N, N).clone()
    for a in attns:
        ahat = a.mean(1) + eye
        ahat = ahat / ahat.sum(-1, keepdim=True)
        R = ahat @ R
    return R


def cls_patch_map(R, size):
    """Fila del token de clase sobre los parches -> (B, size, size)."""
    row = R[:, 0, 1:]
    g = int(round(row.shape[1] ** 0.5))
    assert g * g == row.shape[1], 'número de parches no cuadrado'
    return upsample(row.reshape(-1, g, g), size)


def vit_attributions(model, x, target=1):
    """{'gradient_rollout', 'attention_rollout'}: (B, H, W) sin normalizar; y logits."""
    with explicit_attention(model) as store, torch.enable_grad():
        logits = model(x)
        grads = torch.autograd.grad(logits[:, target].sum(), store)
    attns = [a.detach() for a in store]
    grads = [g.detach() for g in grads]
    size = x.shape[-1]
    return ({'gradient_rollout': cls_patch_map(gradient_rollout(attns, grads), size),
             'attention_rollout': cls_patch_map(attention_rollout(attns), size)},
            logits.detach())


@torch.no_grad()
def check_vit_equivalence(model, x):
    """Diferencia máxima absoluta de logits entre el modelo original y con la
    atención explícita (debe ser del orden del error de redondeo)."""
    ref = model(x)
    with explicit_attention(model):
        alt = model(x)
    return float((ref - alt).abs().max())
