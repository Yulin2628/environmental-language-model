"""Load the original checkpoints without enabling arbitrary pickle execution."""
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import EnvironmentalLanguageModel


ROOT = Path(__file__).resolve().parents[1]


def load_model(checkpoint_path, device="cpu"):
    with torch.serialization.safe_globals([ModelConfig]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = checkpoint.get("config", {})
    config = ModelConfig(**(asdict(saved) if isinstance(saved, ModelConfig) else saved))
    model = EnvironmentalLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    metadata = {key: value for key, value in checkpoint.items()
                if key not in {"model_state_dict", "optimizer_state_dict", "scaler_state_dict", "config"}}
    return model, config, metadata


def load_tokenizer(path=None):
    return Tokenizer.from_file(str(path or ROOT / "tokenizer.json"))
