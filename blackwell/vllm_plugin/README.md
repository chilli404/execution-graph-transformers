# Native vLLM Plugin Work Area

The Hugging Face export path is implemented and logit-exact. This directory is reserved for the out-of-tree vLLM integration.

Do not register `src/fogen/hf_model.py` directly as a native vLLM model and assume correctness. A real plugin must implement:

1. `vllm.general_plugins` entry point;
2. `ModelRegistry.register_model("FogenForCausalLM", ...)`;
3. flattened token and position inputs;
4. PagedAttention with RoPE and QK normalization;
5. alternating value embeddings and value-mixing scalars;
6. logit softcap;
7. vLLM weight loading;
8. sequential, parallel, and fused-parallel execution modes.

Suggested package:

```text
fogen_vllm/
  pyproject.toml
  fogen_vllm/
    __init__.py
    model.py
    attention.py
    fused_block.py
    weight_loader.py
  tests/
    test_logits.py
    test_cached_decode.py
```

Correctness gates before benchmarking:

- prefill logit comparison;
- first decode-step comparison;
- multi-step cached decode comparison;
- greedy token equality;
- ordinary parallel versus fused-parallel equivalence.

Pin vLLM and its supported Torch/CUDA combination in an isolated environment. Do not install into a shared runtime.
