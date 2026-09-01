import argparse
from pathlib import Path

import yaml
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from fogen.hf_model import FogenConfig, FogenForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training_config = yaml.safe_load(open(args.config))
    model_config = FogenConfig(**training_config["model"])
    model = FogenForCausalLM(model_config)
    state = {key: value.float() for key, value in load_file(args.ckpt).items()}
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer_dir) / "tokenizer.json"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(output)
    print(output)


if __name__ == "__main__":
    main()
