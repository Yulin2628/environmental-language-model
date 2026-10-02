from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 12000
    context_length: int = 512
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


if __name__ == "__main__":
    config = ModelConfig()

    print(config)
    print("Each attention head dimension:", config.head_dim)