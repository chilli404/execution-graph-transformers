import torch
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from fogen.model import GPT, ModelConfig


class FogenConfig(PretrainedConfig):
    model_type = "fogen"

    def __init__(
        self,
        vocab_size=8192,
        n_layer=4,
        d_model=256,
        n_head=2,
        ctx_len=2048,
        execution_mode="sequential",
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.execution_mode = execution_mode
        self.architectures = ["FogenForCausalLM"]

    def model_config(self):
        return ModelConfig(
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            d_model=self.d_model,
            n_head=self.n_head,
            ctx_len=self.ctx_len,
            execution_mode=self.execution_mode,
        )


class FogenForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FogenConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(config.model_config())

    def get_input_embeddings(self):
        return self.model.wte

    def set_input_embeddings(self, value):
        self.model.wte = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def forward(
        self,
        input_ids,
        labels=None,
        past_key_values=None,
        use_cache=False,
        execution_mode=None,
        **kwargs,
    ):
        mode = execution_mode or self.config.execution_mode
        if use_cache or past_key_values is not None:
            logits, cache = self.model.forward_cached(
                input_ids, cache=past_key_values, mode=mode)
        else:
            logits = self.model(input_ids, mode=mode)
            cache = None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=cache,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
