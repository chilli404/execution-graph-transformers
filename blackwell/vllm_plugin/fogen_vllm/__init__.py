"""vLLM out-of-tree model plugin for execution-graph Transformers."""


def register():
    from vllm import ModelRegistry
    from fogen_vllm.model import FogenForCausalLM

    ModelRegistry.register_model("FogenForCausalLM", FogenForCausalLM)

    # Also registers "fogen" with transformers' AutoConfig/AutoModelForCausalLM,
    # which vLLM's own ModelConfig validation needs to resolve config.json
    # before ModelRegistry is ever consulted.
    import fogen.hf_model  # noqa: F401
