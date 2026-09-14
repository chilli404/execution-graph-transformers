"""Test _gradnorm_cw_inline: loss-ratio proxy for gradient-normalized cw.

The inline version uses loss magnitudes as a proxy for gradient norms,
avoiding all backward passes. This is memory-safe at any scale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fogen.model import GPT, ModelConfig
from fogen.training.train import (
    _compute_consistency,
    _gradnorm_cache,
    _gradnorm_cw_inline,
)


def _make_tiny_model():
    cfg = ModelConfig(
        vocab_size=256, n_layer=4, d_model=64, n_head=4, ctx_len=32,
        execution_mode="sequential",
    )
    return GPT(cfg), cfg


def _make_batch(cfg, batch=4):
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.ctx_len))
    return x, y


def _train_diverged(model, cfg, steps=50):
    """Train with polymorphic loss so consistency is non-trivial."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = _make_batch(cfg)
        seq_logits = model(x, mode="sequential")
        par_logits = model(x, mode="parallel")
        seq_loss = F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
        par_loss = F.cross_entropy(par_logits.view(-1, par_logits.size(-1)), y.reshape(-1))
        consistency = _compute_consistency(seq_logits, par_logits, "centered_mse")
        loss = 0.5 * seq_loss + 0.5 * par_loss + consistency
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_inline_produces_valid_cw():
    """Step 0 should produce a valid, non-degenerate cw from loss ratio."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    rho = 0.2
    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})

    seq_logits = model(x, mode="sequential")
    par_logits = model(x, mode="parallel")
    lm_loss = 0.5 * F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1)) + \
              0.5 * F.cross_entropy(par_logits.view(-1, par_logits.size(-1)), y.reshape(-1))
    consistency = _compute_consistency(seq_logits, par_logits, "centered_mse")

    cw = _gradnorm_cw_inline(model, lm_loss, consistency, rho, step=0)

    print(f"  lm={float(lm_loss):.4f} con={float(consistency):.6f} cw={cw:.4f}")
    assert 1e-4 <= cw <= 10.0, f"cw={cw} out of bounds"
    assert "base_cw" in _gradnorm_cache
    assert "base_con_loss" in _gradnorm_cache


def test_inline_no_backward_needed():
    """The inline method should work with no_grad / detached tensors."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    rho = 0.2
    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})

    # Step 0 with regular tensors
    with torch.no_grad():
        seq_logits = model(x, mode="sequential")
        par_logits = model(x, mode="parallel")
    lm_loss = 0.5 * F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1)) + \
              0.5 * F.cross_entropy(par_logits.view(-1, par_logits.size(-1)), y.reshape(-1))
    consistency = _compute_consistency(seq_logits, par_logits, "centered_mse")

    # Should work fine — only uses .detach() on losses
    cw = _gradnorm_cw_inline(model, lm_loss, consistency, rho, step=0)
    assert 1e-4 <= cw <= 10.0
    print(f"  cw={cw:.4f} (no grad context)")


def test_inline_proxy_scales_with_consistency():
    """After step 0, cw should scale inversely with sqrt(con_loss)."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    rho = 0.2
    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})

    seq_logits = model(x, mode="sequential")
    par_logits = model(x, mode="parallel")
    lm_loss = 0.5 * F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1)) + \
              0.5 * F.cross_entropy(par_logits.view(-1, par_logits.size(-1)), y.reshape(-1))
    consistency = _compute_consistency(seq_logits, par_logits, "centered_mse")

    cw0 = _gradnorm_cw_inline(model, lm_loss, consistency, rho, step=0)
    base_con = _gradnorm_cache["base_con_loss"]

    # Step 100: if consistency drops by 4x, cw should increase by ~2x (sqrt)
    fake_con = torch.tensor(base_con / 4.0)
    cw100 = _gradnorm_cw_inline(model, None, fake_con, rho, step=100)
    expected_scale = (base_con / (base_con / 4.0)) ** 0.5  # = 2.0
    expected_cw = min(cw0 * expected_scale, 10.0)

    print(f"  cw0={cw0:.4f} cw100={cw100:.4f} expected={expected_cw:.4f}")
    assert abs(cw100 - expected_cw) < 0.01, f"proxy scaling wrong: {cw100} vs {expected_cw}"


def test_inline_skips_between_every():
    """Between gradnorm_every intervals, should return cached cw."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    rho = 0.2
    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100, "every": 100})

    seq_logits = model(x, mode="sequential")
    par_logits = model(x, mode="parallel")
    lm_loss = 0.5 * F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1)) + \
              0.5 * F.cross_entropy(par_logits.view(-1, par_logits.size(-1)), y.reshape(-1))
    consistency = _compute_consistency(seq_logits, par_logits, "centered_mse")

    cw0 = _gradnorm_cw_inline(model, lm_loss, consistency, rho, step=0)
    cw50 = _gradnorm_cw_inline(model, lm_loss, consistency, rho, step=50)
    assert cw50 == cw0, f"step 50 should return cached cw0={cw0}, got {cw50}"
    print(f"  cw0={cw0:.6f} cw50={cw50:.6f} (same)")


def test_inline_cw_proportional_to_rho():
    """cw should be proportional to rho when not clamped."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()

    # Use balanced fake losses to avoid clamping
    fake_lm = torch.tensor(3.0)
    fake_con = torch.tensor(3.0)  # Equal losses → cw ≈ rho

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw_02 = _gradnorm_cw_inline(model, fake_lm, fake_con, rho=0.2, step=0)

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw_04 = _gradnorm_cw_inline(model, fake_lm, fake_con, rho=0.4, step=0)

    ratio = cw_04 / cw_02
    print(f"  cw(0.2)={cw_02:.4f} cw(0.4)={cw_04:.4f} ratio={ratio:.3f}")
    assert 1.9 < ratio < 2.1, f"cw should double when rho doubles, got ratio={ratio:.3f}"


if __name__ == "__main__":
    test_inline_produces_valid_cw()
    print("PASS: produces valid cw")

    test_inline_no_backward_needed()
    print("PASS: no backward needed")

    test_inline_proxy_scales_with_consistency()
    print("PASS: proxy scales correctly")

    test_inline_skips_between_every()
    print("PASS: skips between every intervals")

    test_inline_cw_proportional_to_rho()
    print("PASS: cw proportional to rho")

    print("\nAll tests passed.")
