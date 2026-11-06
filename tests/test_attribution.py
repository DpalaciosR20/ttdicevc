"""F3.7: métodos de atribución con modelos pequeños en CPU (sin pesos descargados)."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, resnet18
from torchvision.models.vision_transformer import VisionTransformer

from src.interpretability import attribution as at

torch.manual_seed(0)


def tiny_vit(frozen=False):
    torch.manual_seed(1)
    m = VisionTransformer(image_size=32, patch_size=8, num_layers=3, num_heads=2, hidden_dim=16, mlp_dim=32,
                          num_classes=2).eval()
    if frozen:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.heads.parameters():
            p.requires_grad = True
    return m


def x_batch(n=3, size=32):
    torch.manual_seed(2)
    return torch.randn(n, 3, size, size)


# ─── ViT ─────────────────────────────────────────────────────

def test_atencion_explicita_no_cambia_la_salida():
    m, x = tiny_vit(), x_batch()
    assert at.check_vit_equivalence(m, x) < 1e-5
    # se restaura la atención original
    assert all(isinstance(b.self_attention, nn.MultiheadAttention) for b in m.encoder.layers)


def test_atencion_explicita_coincide_con_need_weights():
    m, x = tiny_vit(), x_batch()
    store = []
    blk = m.encoder.layers[0]
    with at.explicit_attention(m) as store, torch.no_grad():
        m(x)
    # entrada real del bloque 0, recalculada
    with torch.no_grad():
        z = m._process_input(x)
        z = torch.cat([m.class_token.expand(len(x), -1, -1), z], 1) + m.encoder.pos_embedding
        y = blk.ln_1(z)
        _, w = blk.self_attention(y, y, y, need_weights=True, average_attn_weights=False)
    assert torch.allclose(store[0], w, atol=1e-6)


@pytest.mark.parametrize('frozen', [False, True])
def test_vit_mapas_forma_y_rollout(frozen):
    m, x = tiny_vit(frozen), x_batch()
    maps, logits = at.vit_attributions(m, x)
    for k in ('gradient_rollout', 'attention_rollout'):
        assert maps[k].shape == (3, 32, 32) and torch.isfinite(maps[k]).all()
    assert torch.allclose(logits, m(x).detach(), atol=1e-5)
    # rollout puro: filas estocásticas
    with at.explicit_attention(m) as store, torch.no_grad():
        m(x)
    R = at.attention_rollout(store)
    assert torch.allclose(R.sum(-1), torch.ones(3, R.shape[1]), atol=1e-5)


def test_rollout_ponderado_formula_explicita():
    torch.manual_seed(3)
    attns = [torch.softmax(torch.randn(2, 2, 5, 5), -1) for _ in range(3)]
    grads = [torch.randn(2, 2, 5, 5) for _ in range(3)]
    R = torch.eye(5).repeat(2, 1, 1)
    for a, g in zip(attns, grads):
        abar = torch.stack([((g[:, h] * a[:, h]).clamp(min=0)) for h in range(2)], 1).mean(1)
        R = R + torch.bmm(abar, R)
    assert torch.allclose(at.gradient_rollout(attns, grads), R, atol=1e-6)


def test_cordura_cabeza_aleatoria_vit_congelado():
    """Enmienda 3: al aleatorizar la cabeza, el mapa ponderado por gradiente
    cambia y la atención del backbone no."""
    m, x = tiny_vit(frozen=True), x_batch()
    before, _ = at.vit_attributions(m, x)
    torch.manual_seed(99)
    m.heads.head.reset_parameters()
    after, _ = at.vit_attributions(m, x)
    assert torch.equal(before['attention_rollout'], after['attention_rollout'])
    assert not torch.allclose(before['gradient_rollout'], after['gradient_rollout'], atol=1e-6)


def test_backbone_congelado_no_acumula_gradientes():
    m, x = tiny_vit(frozen=True), x_batch()
    at.vit_attributions(m, x)
    assert all(p.grad is None for p in m.parameters())


# ─── Grad-CAM ────────────────────────────────────────────────

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 5, 3, padding=1, stride=2))
        self.fc = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3)))


def test_grad_cam_formula_explicita():
    torch.manual_seed(4)
    m, x = TinyCNN().eval(), x_batch(2, 16)
    cam, logits = at.grad_cam(m, x, at.resolve_layer(m, 'features[-1]'))
    a = m.features(x)
    g = torch.autograd.grad(m.fc(a.mean((2, 3)))[:, 1].sum(), a)[0]
    ref = F.interpolate(F.relu((g.mean((2, 3), keepdim=True) * a).sum(1))[:, None], size=(16, 16),
                        mode='bilinear', align_corners=False)[:, 0]
    assert torch.allclose(cam, ref.detach(), atol=1e-6) and cam.shape == (2, 16, 16)


@pytest.mark.parametrize('frozen', [False, True])
def test_grad_cam_efficientnet_y_resnet(frozen):
    x = x_batch(2, 64)
    for build, spec in ((efficientnet_b0, 'features[-1]'), (resnet18, 'layer4')):
        torch.manual_seed(5)
        m = build(weights=None, num_classes=2).eval()
        if frozen:
            for p in m.parameters():
                p.requires_grad = False
        cam, logits = at.grad_cam(m, x, at.resolve_layer(m, spec))
        assert cam.shape == (2, 64, 64) and torch.isfinite(cam).all() and (cam >= 0).all()
        assert torch.allclose(logits, m(x), atol=1e-5)
        h = m.fc if build is resnet18 else m.classifier[-1]
        # cambia con la cabeza (el gradiente llega por ella)
        h.reset_parameters()
        cam2, _ = at.grad_cam(m, x, at.resolve_layer(m, spec))
        assert not torch.allclose(at.minmax(cam), at.minmax(cam2), atol=1e-3)


def test_minmax():
    m = torch.tensor([[[1., 3.], [2., 5.]], [[2., 2.], [2., 2.]]])
    out = at.minmax(m)
    assert out[0].min() == 0 and out[0].max() == 1 and torch.equal(out[1], torch.zeros(2, 2))
