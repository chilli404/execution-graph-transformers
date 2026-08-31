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
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
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
        max_position: int = 2048,
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
            max_position=max_position,
            is_neox_style=True,
        )

        self.attn = Attention(
            num_heads=n_head,
            head_size=self.head_dim,
            scale=1.0 / math.sqrt(self.head_dim),
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward_projected(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ve: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_ve and ve is not None:
            v = self.ve_lambda_0 * v + self.ve_lambda_1 * ve

        # RoPE first, then QK-norm (matches reference model)
        q, k = self.rotary_emb(positions, q, k)

        q = F.rms_norm(
            q.view(-1, self.n_head, self.head_dim),
            (self.head_dim,),
        ).reshape(-1, self.d_model)
        k = F.rms_norm(
            k.view(-1, self.n_head, self.head_dim),
            (self.head_dim,),
        ).reshape(-1, self.d_model)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self.d_model, self.d_model, self.d_model], dim=-1)
        return self.forward_projected(q, k, v, ve, positions)


class FogenBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        layer_idx: int,
        n_layer: int,
        max_position: int = 2048,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm = FogenRMSNorm(d_model)
        self.attn = FogenAttention(
            d_model, n_head, layer_idx, n_layer,
            max_position=max_position,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = FogenMLP(d_model)
        self.d_model = d_model
        # Populated once in FogenForCausalLM.load_weights (after qkv_proj and
        # mlp.up are loaded) so parallel_fused is a plain tensor op with no
        # first-call caching branch — that mutable-state pattern defeats
        # Dynamo tracing under torch.compile.
        self.register_buffer(
            "fused_qkv_up_weight",
            torch.zeros(3 * d_model + 4 * d_model, d_model),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        positions: torch.Tensor,
        mode: str = "sequential",
    ) -> torch.Tensor:
        normalized = self.norm(x)

        if mode == "parallel_fused":
            projected = F.linear(normalized, self.fused_qkv_up_weight)
            q, k, v, mlp_hidden = projected.split(
                [self.d_model, self.d_model, self.d_model, 4 * self.d_model],
                dim=-1,
            )
            attention = self.attn.forward_projected(q, k, v, ve, positions)
            mlp = self.mlp.down(F.relu(mlp_hidden).square())[0]
            return x + attention + mlp

        attention = self.attn(normalized, ve, positions)

        if mode == "parallel":
            return x + attention + self.mlp(normalized)

        if mode != "sequential":
            raise ValueError(f"Unknown execution mode: {mode}")

        # Sequential: FFN reads post-attention state
        x = x + attention
        return x + self.mlp(self.norm(x))


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
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
                max_position=config.ctx_len,
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
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=15.0)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_norm(
            inputs_embeds if inputs_embeds is not None
            else self.wte(input_ids))

        for i, block in enumerate(self.blocks):
            ve = (
                self.value_embeds[str(i)](input_ids)
                if str(i) in self.value_embeds
                else None
            )
            x = block(x, ve, positions, mode=self.layer_modes[i])

        x = self.final_norm(x)
        return x

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # LogitsProcessor applies lm_head, gathers for the sampling
        # positions, and applies the softcap (scale=soft_cap*tanh(x/soft_cap))
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())

        # Collect QKV weights per layer for merged loading
        qkv_buffers: dict[str, dict[int, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            # Handle ve_lambdas (scalars, not sharded)
            if "ve_lambdas" in name:
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
                target = name.replace("model.", "")
                if target in params:
                    param = params[target]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                continue

            # Collect QKV for merged loading
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

            # Map HF names to vLLM names
            target = name.replace("model.", "")
            # wo -> o_proj
            target = target.replace(".attn.wo.", ".attn.o_proj.")
            if target in params:
                param = params[target]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

        # Load merged QKV weights using MergedColumnParallelLinear's
        # weight_loader which handles TP sharding per shard
        for layer, qkv in qkv_buffers.items():
            target = f"blocks.{layer}.attn.qkv_proj.weight"
            if target in params:
                param = params[target]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                for shard_id, shard_weight in sorted(qkv.items()):
                    weight_loader(param, shard_weight, shard_id)

        # Populate each block's fused weight from loaded (possibly sharded) weights.
        # Skip if sizes don't match (tensor parallelism changes shard dimensions).
        for block in self.blocks:
            fused = torch.cat([block.attn.qkv_proj.weight, block.mlp.up.weight], dim=0)
            if block.fused_qkv_up_weight.shape == fused.shape:
                block.fused_qkv_up_weight.copy_(fused)
            else:
                block.fused_qkv_up_weight = fused
