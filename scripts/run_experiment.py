from mcq_eval.config.schema import load_config
from mcq_eval.backends.api_backend import APIBackend
from mcq_eval.backends.dummy_backend import DummyBackend
from mcq_eval.backends.hf_backend import HuggingFaceBackend
import argparse

parser = argparse.ArgumentParser(description="Run an experiment with the specified configuration.")

parser.add_argument("--config",
                    type=str,
                    required=True,
                    help="Path to the configuration file.")

args = parser.parse_args()

config = load_config(args.config)   

def build_backend(config):
    backend_type = config.model.backend 
    if backend_type == "api":
        return APIBackend()
    elif backend_type == "huggingface":
        return HuggingFaceBackend(config.model.model_name_or_path, config.model.device)
    elif backend_type == "dummy":
        return DummyBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")