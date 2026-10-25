"""GPT checks: the self-attention 'mathematical trick', and per-head vs batched attention."""

import pytest
import torch
from torch.nn import functional as F

from zero_to_hero.gpt.model import GPT, GPTConfig, heads_to_batched_state_dict


def test_weighted_aggregation_trick():
    """Lecture 7: averaging over the past with a loop == matmul with tril == masked softmax."""
    torch.manual_seed(1337)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xbow[b, t] = torch.mean(x[b, :t + 1], 0)
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    xbow2 = wei @ x
    tril = torch.tril(torch.ones(T, T))
    wei3 = F.softmax(torch.zeros((T, T)).masked_fill(tril == 0, float("-inf")), dim=-1)
    xbow3 = wei3 @ x
    assert torch.allclose(xbow, xbow2, atol=1e-6) and torch.allclose(xbow, xbow3, atol=1e-6)


@pytest.mark.parametrize("attn_scale", ["head_size", "n_embd"])
def test_heads_and_batched_attention_are_equivalent(attn_scale):
    torch.manual_seed(0)
    cfg = dict(vocab_size=65, block_size=32, n_embd=48, n_head=4, n_layer=2, dropout=0.0, attn_scale=attn_scale)
    heads = GPT(GPTConfig(attn="heads", **cfg)).double().eval()
    batched = GPT(GPTConfig(attn="batched", **cfg)).double().eval()
    sd = {k: v for k, v in heads.state_dict().items() if not k.endswith(".tril")}
    batched.load_state_dict(heads_to_batched_state_dict(sd, cfg["n_head"]))
    idx = torch.randint(0, 65, (3, 32))
    targets = torch.randint(0, 65, (3, 32))
    l1, loss1 = heads(idx, targets)
    l2, loss2 = batched(idx, targets)
    assert torch.allclose(l1, l2, atol=1e-10)
    assert abs(loss1.item() - loss2.item()) < 1e-10


def test_causality():
    """Changing a future token must not change the logits at earlier positions."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=65, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)).eval()
    idx = torch.randint(0, 65, (1, 16))
    idx2 = idx.clone()
    idx2[0, 10] = (idx[0, 10] + 1) % 65
    l1, _ = model(idx)
    l2, _ = model(idx2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-6)
    assert not torch.allclose(l1[0, 10:], l2[0, 10:], atol=1e-6)


def test_initial_loss_near_uniform():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=65, block_size=16, n_embd=32, n_head=4, n_layer=2)).eval()
    idx = torch.randint(0, 65, (8, 16))
    _, loss = model(idx, idx)
    assert abs(loss.item() - torch.log(torch.tensor(65.0)).item()) < 0.1
