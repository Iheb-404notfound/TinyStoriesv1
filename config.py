# config.py - Exact TinyStories configurations
from transformers import GPTNeoConfig

def get_tinystories_config(model_size: str = "28M"):
    base = {
        "vocab_size": 50257,
        "attention_types": [[["global", "local"], 4]],  # 8 layers: global/local alternating
        "window_size": 256,
        "activation_function": "gelu_new",
        "resid_dropout": 0.0,
        "embed_dropout": 0.0,
        "attention_dropout": 0.0,
        "max_position_embeddings": 2048,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }

    configs = {
        "10M": {"hidden_size": 320, "num_layers": 8,  "num_heads": 16},
        "28M": {"hidden_size": 512, "num_layers": 8,  "num_heads": 16},
        "33M": {"hidden_size": 512, "num_layers": 10, "num_heads": 16},
    }

    if model_size == "33M":
        base["attention_types"] = [[["global", "local"], 5]]  # 10 layers

    return GPTNeoConfig(**(base | configs[model_size]))
