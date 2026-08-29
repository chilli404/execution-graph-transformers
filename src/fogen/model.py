"""Nanochat-style autoregressive transformer, matched to the v1 paper.

v1 Table 1: 4L / 256d / 2H, vocab 8192, 11.5M total params, "Muon + AdamW
for embeddings, scalars, and gains". The Dec-2025 nanochat baseline
(rotary, qk-norm, relu^2 MLP, untied head, embedding norm, logit softcap,
zero-init projections) gives only 7.3M at this shape; the missing 4.2M is
exactly two vocab x d tables, i.e. modded-nanogpt-style value embeddings
on alternating layers (last layer always included) with learnable mixing
scalars (v = l1*v + l2*ve). Exact v1 source unavailable; fidelity is
validated against val_bpb ~ 1.149-1.152 and the behavioral gate
(see DECISIONS.md).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 2
    ctx_len: int = 2048
    execution_mode: str = "sequential"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


def _rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh) with Dh even
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Value embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # ve mixing scalars (modded-nanogpt convention): v = l[0]*v + l[1]*ve
        self.ve_lambdas = (nn.Parameter(torch.tensor([0.5, 0.5]))
                           if has_ve(layer_idx, cfg.n_layer) else None)

    def forward_projected_cached(self, q, k, v, ve, cos, sin, past=None):
        B, T, C = q.shape
        H, Dh = self.cfg.n_head, self.cfg.head_dim
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        if self.ve_lambdas is not None:
            ve = ve.view(B, T, H, Dh).transpose(1, 2)
            v = self.ve_lambdas[0] * v + self.ve_lambdas[1] * ve
        q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        q, k = _rmsnorm(q), _rmsnorm(k)
        if past is not None:
            k = torch.cat([past[0], k], dim=-2)
            v = torch.cat([past[1], v], dim=-2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past is None and T > 1))
        output = self.wo(y.transpose(1, 2).reshape(B, T, C))
        return output, (k, v)

    def forward_projected(self, q, k, v, ve, cos, sin):
        output, _ = self.forward_projected_cached(q, k, v, ve, cos, sin)
        return output

    def forward(self, x, ve, cos, sin):
        return self.forward_projected(
            self.wq(x), self.wk(x), self.wv(x), ve, cos, sin)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.down = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self._attention_stream = None
        self._mlp_stream = None
        self._fused_input_weight = None

    def forward(self, x, ve, cos, sin, mode="sequential"):
        normalized = _rmsnorm(x)
        if mode == "parallel_fused" and not self.training:
            if self._fused_input_weight is None:
                self._fused_input_weight = torch.cat([
                    self.attn.wq.weight,
                    self.attn.wk.weight,
                    self.attn.wv.weight,
                    self.mlp.up.weight,
                ]).detach()
            with torch.autocast(device_type=normalized.device.type, enabled=False):
                projected = F.linear(
                    normalized.float(), self._fused_input_weight.float()
                ).to(normalized.dtype)
            size = normalized.size(-1)
            q, k, v, mlp_hidden = projected.split([size, size, size, 4 * size], dim=-1)
            attention = self.attn.forward_projected(q, k, v, ve, cos, sin)
            mlp = self.mlp.down(F.relu(mlp_hidden).square())
            return x + attention + mlp
        if mode == "parallel_cuda" and x.is_cuda and not self.training:
            if self._attention_stream is None:
                self._attention_stream = torch.cuda.Stream(device=x.device)
                self._mlp_stream = torch.cuda.Stream(device=x.device)
            current = torch.cuda.current_stream(x.device)
            self._attention_stream.wait_stream(current)
            self._mlp_stream.wait_stream(current)
            with torch.cuda.stream(self._attention_stream):
                attention = self.attn(normalized, ve, cos, sin)
            with torch.cuda.stream(self._mlp_stream):
                mlp = self.mlp(normalized)
            current.wait_stream(self._attention_stream)
            current.wait_stream(self._mlp_stream)
            attention.record_stream(current)
            mlp.record_stream(current)
            return x + attention + mlp
        attention = self.attn(normalized, ve, cos, sin)
        if mode in ("parallel", "parallel_cuda"):
            return x + attention + self.mlp(normalized)
        if mode != "sequential":
            raise ValueError(f"Unknown execution mode: {mode}")
        x = x + attention
        return x + self.mlp(_rmsnorm(x))

    def forward_cached(self, x, ve, cos, sin, mode, past=None):
        normalized = _rmsnorm(x)
        if mode == "parallel_fused" and not self.training:
            if self._fused_input_weight is None:
                self._fused_input_weight = torch.cat([
                    self.attn.wq.weight,
                    self.attn.wk.weight,
                    self.attn.wv.weight,
                    self.mlp.up.weight,
                ]).detach()
            with torch.autocast(device_type=normalized.device.type, enabled=False):
                projected = F.linear(
                    normalized.float(), self._fused_input_weight.float()
                ).to(normalized.dtype)
            size = normalized.size(-1)
            q, k, v, mlp_hidden = projected.split([size, size, size, 4 * size], dim=-1)
            attention, cache = self.attn.forward_projected_cached(
                q, k, v, ve, cos, sin, past)
            mlp = self.mlp.down(F.relu(mlp_hidden).square())
            return x + attention + mlp, cache
        attention, cache = self.attn.forward_projected_cached(
            self.attn.wq(normalized), self.attn.wk(normalized),
            self.attn.wv(normalized), ve, cos, sin, past)
        if mode in ("parallel", "parallel_cuda"):
            return x + attention + self.mlp(normalized), cache
        if mode != "sequential":
            raise ValueError(f"Unknown cached execution mode: {mode}")
        attended = x + attention
        return attended + self.mlp(_rmsnorm(attended)), cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(cfg.vocab_size, cfg.d_model)
             for i in range(cfg.n_layer) if has_ve(i, cfg.n_layer)})
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        half = cfg.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(half) / half))
        t = torch.arange(cfg.ctx_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", freqs.sin()[None, None], persistent=False)

        self.init_weights()

    def init_weights(self):
        # nanochat scheme: zero-init residual projections and lm_head so the
        # model starts as (approximately) the identity over token embeddings
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_out, fan_in = m.weight.shape
                std = (1.0 / math.sqrt(fan_in)) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.lm_head.weight)
        for b in self.blocks:
            nn.init.zeros_(b.attn.wo.weight)
            nn.init.zeros_(b.mlp.down.weight)
            if b.attn.ve_lambdas is not None:
                with torch.no_grad():
                    b.attn.ve_lambdas.copy_(torch.tensor([0.5, 0.5]))

    def layer_defects(self, idx: torch.Tensor) -> torch.Tensor:
        length = idx.size(1)
        cos, sin = self.rope_cos[:, :, :length], self.rope_sin[:, :, :length]
        x = _rmsnorm(self.wte(idx))
        defects = []
        for layer, block in enumerate(self.blocks):
            ve = self.value_embeds[str(layer)](idx) if str(layer) in self.value_embeds else None
            normalized = _rmsnorm(x)
            attention = block.attn(normalized, ve, cos, sin)
            parallel = x + attention + block.mlp(normalized)
            attended = x + attention
            sequential = attended + block.mlp(_rmsnorm(attended))
            numerator = torch.mean((sequential - parallel).float() ** 2).sqrt()
            denominator = torch.mean((sequential - x).float() ** 2).sqrt().clamp_min(1e-12)
            defects.append(numerator / denominator)
            x = sequential
        return torch.stack(defects)

    def forward(self, idx: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        T = idx.size(1)
        cos, sin = self.rope_cos[:, :, :T], self.rope_sin[:, :, :T]
        x = _rmsnorm(self.wte(idx))
        execution_mode = mode or self.cfg.execution_mode
        if isinstance(execution_mode, str):
            layer_modes = [execution_mode] * len(self.blocks)
        else:
            layer_modes = list(execution_mode)
            if len(layer_modes) != len(self.blocks):
                raise ValueError("Execution mask length must equal n_layer")
        for i, b in enumerate(self.blocks):
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = b(x, ve, cos, sin, mode=layer_modes[i])
        logits = self.lm_head(_rmsnorm(x)).float()
        return 15.0 * torch.tanh(logits / 15.0)  # nanochat logit softcap

    def forward_cached(self, idx: torch.Tensor, cache=None,
                       mode: str | None = None):
        start = 0 if cache is None else cache[0][0].size(-2)
        length = idx.size(1)
        if start + length > self.cfg.ctx_len:
            raise ValueError("KV cache exceeds configured context length")
        cos = self.rope_cos[:, :, start:start + length]
        sin = self.rope_sin[:, :, start:start + length]
        x = _rmsnorm(self.wte(idx))
        execution_mode = mode or self.cfg.execution_mode
        layer_modes = ([execution_mode] * len(self.blocks)
                       if isinstance(execution_mode, str) else list(execution_mode))
        if len(layer_modes) != len(self.blocks):
            raise ValueError("Execution mask length must equal n_layer")
        past_layers = [None] * len(self.blocks) if cache is None else cache
        new_cache = []
        for layer, block in enumerate(self.blocks):
            ve = self.value_embeds[str(layer)](idx) if str(layer) in self.value_embeds else None
            x, layer_cache = block.forward_cached(
                x, ve, cos, sin, layer_modes[layer], past_layers[layer])
            new_cache.append(layer_cache)
        logits = self.lm_head(_rmsnorm(x)).float()
        return 15.0 * torch.tanh(logits / 15.0), new_cache

    def loss(self, idx: torch.Tensor, targets: torch.Tensor,
             mode: str | None = None) -> torch.Tensor:
        logits = self(idx, mode=mode)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def matrix_params(self):
        """2-D block weights for Muon (not embeddings/head/ve)."""
        return [p for n, p in self.named_parameters()
                if p.ndim == 2 and n.startswith("blocks")]

    def embed_params(self, exclude_head: bool = False):
        return [p for n, p in self.named_parameters()
                if ("wte" in n or "value_embeds" in n
                    or (not exclude_head and "lm_head" in n))]

    def head_params(self):
        return [p for n, p in self.named_parameters() if "lm_head" in n]

    def scalar_params(self):
        return [p for n, p in self.named_parameters() if "ve_lambdas" in n]
