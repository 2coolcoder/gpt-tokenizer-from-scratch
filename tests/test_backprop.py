"""Manual (lecture 5) gradients must equal PyTorch autograd for every tensor."""

from zero_to_hero.makemore.backprop import gradient_check
from zero_to_hero.makemore.data import splits


def test_manual_and_fused_backward_match_autograd():
    (Xtr, Ytr), _, _ = splits(block_size=3)
    manual, fused = gradient_check(Xtr[:5000], Ytr[:5000])
    assert len(manual) >= 25  # every intermediate tensor is checked
    assert max(manual.values()) < 1e-12, manual
    assert max(fused.values()) < 1e-12, fused
