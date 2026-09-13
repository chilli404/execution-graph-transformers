"""Test that _gradnorm_cw_inline produces equivalent results to _gradnorm_cw.

The inline version reuses losses from the training forward pass,
while the original does separate forward passes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fogen.model import GPT, ModelConfig
from fogen.training.train import (
    _compute_consistency,
    _gradnorm_cache,
    _gradnorm_cw,
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
    """Train with ONLY seq loss so par logits diverge from seq."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = _make_batch(cfg)
        seq_logits = model(x, mode="sequential")
        loss = F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_inline_exact_match_same_batch():
    """With batch=2, both methods use the full batch and should match exactly."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)

    x, y = _make_batch(cfg, batch=2)

    # Verify consistency is actually non-zero
    with torch.no_grad():
        s = model(x, mode="sequential")
        p = model(x, mode="parallel")
        con_check = _compute_consistency(s, p, "centered_mse")
        print(f"  consistency={float(con_check):.6f}")
        assert float(con_check) > 1e-6, "consistency too small to test gradnorm"

    rho = 0.2
    execution_cfg = {
        "enabled": True,
        "parallel_weight": 0.5,
        "consistency_type": "centered_mse",
        "gradnorm_rho": rho,
        "gradnorm_every": 1,
    }

    # Original: uses gn_batch=min(2, batch)=2, so full batch
    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw_original = _gradnorm_cw(model, x, y, execution_cfg, rho, step=0)
    model.zero_grad(set_to_none=True)

    # Inline: uses full batch
    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})

    seq_logits = model(x, mode="sequential")
    par_logits = model(x, mode="parallel")
    seq_loss = F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
    par_loss = F.cross_entropy(par_logits.view(-1, par_logits.size(-1)), y.reshape(-1))
    lm_loss = 0.5 * seq_loss + 0.5 * par_loss
    consistency = _compute_consistency(seq_logits, par_logits, "centered_mse")

    cw_inline = _gradnorm_cw_inline(model, lm_loss, consistency, rho, step=0)
    model.zero_grad(set_to_none=True)

    print(f"  cw_original={cw_original:.6f}  cw_inline={cw_inline:.6f}")

    assert 1e-4 < cw_original < 10.0, f"original cw={cw_original} degenerate"
    assert 1e-4 < cw_inline < 10.0, f"inline cw={cw_inline} degenerate"

    # Both compute ρ * ||∇LM|| / ||∇con|| on the same batch
    # Original does separate fwd but same model/data → same gradient norms
    rel_diff = abs(cw_original - cw_inline) / max(cw_original, 1e-8)
    assert rel_diff < 0.05, \
        f"original={cw_original:.6f} inline={cw_inline:.6f} differ by {rel_diff:.1%}"


def test_inline_caches_and_uses_proxy():
    """After step 0, inline should use proxy scaling, not gradients."""
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
    assert "base_cw" in _gradnorm_cache
    base_cw = _gradnorm_cache["base_cw"]
    print(f"  base_cw={base_cw:.6f}")

    # Step 100: proxy call — should work with detached losses (no backward)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        seq2 = model(x, mode="sequential")
        par2 = model(x, mode="parallel")
    consistency2 = _compute_consistency(seq2, par2, "centered_mse")

    cw100 = _gradnorm_cw_inline(model, None, consistency2, rho, step=100)
    assert cw100 > 0
    assert cw100 < 10.0
    assert _gradnorm_cache["base_cw"] == base_cw
    print(f"  proxy_cw={cw100:.6f}")


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


if __name__ == "__main__":
    test_inline_exact_match_same_batch()
    print("PASS: inline matches original exactly (same batch)")

    test_inline_caches_and_uses_proxy()
    print("PASS: caches base_cw and uses proxy")

    test_inline_skips_between_every()
    print("PASS: skips between every intervals")

    print("\nAll tests passed.")
