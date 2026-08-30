"""vLLM-native implementation of the Fogen execution-graph Transformer.

Implements sequential, parallel, and fused-parallel execution modes
with PagedAttention, RoPE, QK-norm, ReLU² MLP, value embeddings,
and logit softcap.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors


def _has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


class FogenRMSNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.d_model,))


class FogenMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.up = ColumnParallelLinear(
            d_model, 4 * d_model, bias=False)
        self.down = RowParallelLinear(
            4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up(x)
        return self.down(F.relu(x).square())[0]


class FogenAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        layer_idx: int,
        n_layer: int,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv_proj = MergedColumnParallelLinear(
            d_model,
            [d_model, d_model, d_model],
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            d_model, d_model, bias=False)

        self.has_ve = _has_ve(layer_idx, n_layer)
        if self.has_ve:
            self.ve_lambda_0 = nn.Parameter(torch.tensor(0.5))
            self.ve_lambda_1 = nn.Parameter(torch.tensor(0.5))

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=8192,
            base=10000,
            is_neox_style=True,
        )

        self.attn = Attention(
            num_heads=n_head,
            head_size=self.head_dim,
            scale=1.0 / math.sqrt(self.head_dim),
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self.d_model, self.d_model, self.d_model], dim=-1)

        if self.has_ve and ve is not None:
            v = self.ve_lambda_0 * v + self.ve_lambda_1 * ve

        # QK-norm before RoPE
        q = F.rms_norm(
            q.view(-1, self.n_head, self.head_dim),
            (self.head_dim,),
        ).reshape(-1, self.d_model)
        k = F.rms_norm(
            k.view(-1, self.n_head, self.head_dim),
            (self.head_dim,),
        ).reshape(-1, self.d_model)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class FogenBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        layer_idx: int,
        n_layer: int,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm = FogenRMSNorm(d_model)
        self.attn = FogenAttention(
            d_model, n_head, layer_idx, n_layer,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = FogenMLP(d_model)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mode: str = "sequential",
    ) -> torch.Tensor:
        normalized = self.norm(x)

        if mode == "parallel_fused":
            # Fused QKV + MLP-up in one matmul is not implemented
            # in vLLM parallel linear layers; fall back to plain parallel
            mode = "parallel"

        attention = self.attn(normalized, ve, positions, kv_cache, attn_metadata)

        if mode == "parallel":
            return x + attention + self.mlp(normalized)

        # Sequential: FFN reads post-attention state
        x = x + attention
        return x + self.mlp(self.norm(x))


class FogenForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.d_model = config.d_model
        self.n_head = config.n_head

        # Parse execution mode (string or per-layer list)
        exec_mode = getattr(config, "execution_mode", "sequential")
        if isinstance(exec_mode, str):
            self.layer_modes = [exec_mode] * self.n_layer
        else:
            self.layer_modes = list(exec_mode)

        self.wte = VocabParallelEmbedding(
            config.vocab_size, config.d_model)
        self.embed_norm = FogenRMSNorm(config.d_model)

        self.blocks = nn.ModuleList([
            FogenBlock(
                config.d_model,
                config.n_head,
                i,
                config.n_layer,
                cache_config=cache_config,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.n_layer)
        ])

        self.value_embeds = nn.ModuleDict({
            str(i): VocabParallelEmbedding(config.vocab_size, config.d_model)
            for i in range(config.n_layer)
            if _has_ve(i, config.n_layer)
        })

        self.final_norm = FogenRMSNorm(config.d_model)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.d_model, bias=False)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        x = self.embed_norm(self.wte(input_ids))

        for i, block in enumerate(self.blocks):
            ve = (
                self.value_embeds[str(i)](input_ids)
                if str(i) in self.value_embeds
                else None
            )
            x = block(
                x, ve, positions, kv_caches[i],
                attn_metadata, mode=self.layer_modes[i],
            )

        x = self.final_norm(x)
        return x

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        # Logit softcap
        if logits is not None:
            logits = 15.0 * torch.tanh(logits / 15.0)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata):
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())

        # Weight name mapping from HF checkpoint to vLLM model
        mapping = {
            "model.wte.weight": "wte.weight",
            "model.lm_head.weight": "lm_head.weight",
        }
        for i in range(self.n_layer):
            p = f"model.blocks.{i}"
            t = f"blocks.{i}"
            mapping[f"{p}.attn.wq.weight"] = f"{t}.attn.qkv_proj.weight"
            mapping[f"{p}.attn.wk.weight"] = f"{t}.attn.qkv_proj.weight"
            mapping[f"{p}.attn.wv.weight"] = f"{t}.attn.qkv_proj.weight"
            mapping[f"{p}.attn.wo.weight"] = f"{t}.attn.o_proj.weight"
            mapping[f"{p}.mlp.up.weight"] = f"{t}.mlp.up.weight"
            mapping[f"{p}.mlp.down.weight"] = f"{t}.mlp.down.weight"
            if _has_ve(i, self.n_layer):
                mapping[f"{p}.attn.ve_lambdas"] = None  # handled separately

        # Collect QKV weights per layer for merged loading
        qkv_buffers = {}

        for name, loaded_weight in weights:
            # Handle ve_lambdas
            if "ve_lambdas" in name:
                # e.g. model.blocks.1.attn.ve_lambdas -> blocks.1.attn.ve_lambda_0/1
                layer_str = name.split(".")[2]
                target_0 = f"blocks.{layer_str}.attn.ve_lambda_0"
                target_1 = f"blocks.{layer_str}.attn.ve_lambda_1"
                if target_0 in params:
                    params[target_0].data.copy_(loaded_weight[0])
                if target_1 in params:
                    params[target_1].data.copy_(loaded_weight[1])
                continue

            # Handle value embeddings
            if "value_embeds" in name:
                # model.value_embeds.1.weight -> value_embeds.1.weight
                target = name.replace("model.", "")
                if target in params:
                    default_weight_loader(params[target], loaded_weight)
                continue

            # Handle QKV merging
            if ".attn.wq.weight" in name:
                layer = name.split(".")[2]
                qkv_buffers.setdefault(layer, {})[0] = loaded_weight
                continue
            if ".attn.wk.weight" in name:
                layer = name.split(".")[2]
                qkv_buffers.setdefault(layer, {})[1] = loaded_weight
                continue
            if ".attn.wv.weight" in name:
                layer = name.split(".")[2]
                qkv_buffers.setdefault(layer, {})[2] = loaded_weight
                continue

            # Direct mapping
            target = name.replace("model.", "")
            if target in params:
                default_weight_loader(params[target], loaded_weight)

        # Load merged QKV weights
        for layer, qkv in qkv_buffers.items():
            target = f"blocks.{layer}.attn.qkv_proj.weight"
            if target in params:
                merged = torch.cat([qkv[0], qkv[1], qkv[2]], dim=0)
                default_weight_loader(params[target], merged)
