"""vLLM out-of-tree model plugin for execution-graph Transformers."""


def register():
    from vllm import ModelRegistry
    from fogen_vllm.model import FogenForCausalLM

    ModelRegistry.register_model("FogenForCausalLM", FogenForCausalLM)
