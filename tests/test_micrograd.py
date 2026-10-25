"""micrograd gradients must match PyTorch autograd on the same expressions."""

import torch

from zero_to_hero.micrograd.engine import Value
from zero_to_hero.micrograd.nn import MLP


def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    assert ymg.data == y.data.item()
    assert xmg.grad == x.grad.item()


def test_more_ops():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f
    g = g.tanh() * 3 + (g * 1e-3).exp() + f.log()
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double().requires_grad_()
    b = torch.Tensor([2.0]).double().requires_grad_()
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g = g.tanh() * 3 + (g * 1e-3).exp() + f.log()
    g.backward()
    tol = 1e-6
    assert abs(gmg.data - g.data.item()) < tol
    assert abs(amg.grad - a.grad.item()) < tol
    assert abs(bmg.grad - b.grad.item()) < tol


def test_mlp_gradients_match_torch():
    model = MLP(3, [4, 4, 1])
    x = [2.0, 3.0, -1.0]
    out = model(x)
    (out * out).backward()
    # rebuild the same network in torch from the micrograd weights
    xt = torch.tensor(x, dtype=torch.float64)
    params = []
    for li, layer in enumerate(model.layers):
        W = torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True)
        params.append((W, b))
        xt = W @ xt + b
        if li != len(model.layers) - 1:
            xt = xt.relu()
    (xt * xt).sum().backward()
    for layer, (W, b) in zip(model.layers, params):
        for i, n in enumerate(layer.neurons):
            assert all(abs(w.grad - g) < 1e-9 for w, g in zip(n.w, W.grad[i].tolist()))
            assert abs(n.b.grad - b.grad[i].item()) < 1e-9
