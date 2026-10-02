import torch
from torch import nn

from config import ModelConfig


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.context_length = config.context_length

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.d_model,
        )

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]

        if sequence_length > self.context_length:
            raise ValueError(
                "Input sequence is longer than context_length."
            )

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(
            token_ids
        )

        position_vectors = self.position_embedding(
            position_ids
        )

        combined_vectors = (
            token_vectors
            + position_vectors
        )

        return combined_vectors


class QKVProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.query_layer = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=False,
        )

        self.key_layer = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=False,
        )

        self.value_layer = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=False,
        )

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        return query, key, value


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.qkv_projection = QKVProjection(
            config
        )

        self.output_projection = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=False,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        query, key, value = (
            self.qkv_projection(x)
        )

        query = query.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        attention_scores = (
            attention_scores
            / (self.head_dim ** 0.5)
        )

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attention_scores = (
            attention_scores.masked_fill(
                causal_mask,
                float("-inf"),
            )
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        head_outputs = torch.matmul(
            attention_weights,
            value,
        )

        merged_output = (
            head_outputs
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )

        attention_output = (
            self.output_projection(
                merged_output
            )
        )

        return attention_output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_up = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ff,
        )

        self.activation = nn.GELU()

        self.linear_down = nn.Linear(
            in_features=config.d_ff,
            out_features=config.d_model,
        )

        self.dropout = nn.Dropout(
            p=config.dropout
        )

    def forward(self, x):
        x = self.linear_up(x)
        x = self.activation(x)
        x = self.linear_down(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm_before_attention = (
            nn.LayerNorm(config.d_model)
        )

        self.attention = MultiHeadAttention(
            config
        )

        self.norm_before_ffn = (
            nn.LayerNorm(config.d_model)
        )

        self.feed_forward = FeedForward(
            config
        )

    def forward(self, x):
        attention_input = (
            self.norm_before_attention(x)
        )

        attention_change = self.attention(
            attention_input
        )

        x = x + attention_change

        ffn_input = self.norm_before_ffn(x)

        ffn_change = self.feed_forward(
            ffn_input
        )

        x = x + ffn_change

        return x


class EnvironmentalLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = InputEmbedding(
            config
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.output_head = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False,
        )

        self.apply(
            self._initialize_weights
        )

        self.output_head.weight = (
            self.embedding
            .token_embedding
            .weight
        )

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(
                module.weight
            )

            nn.init.zeros_(
                module.bias
            )

    def forward(self, token_ids):
        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        return logits


if __name__ == "__main__":
    config = ModelConfig()

    model = EnvironmentalLanguageModel(
        config
    )

    test_token_ids = torch.tensor(
        [
            [10, 25, 31, 48],
            [17, 36, 92, 55],
        ],
        dtype=torch.long,
    )

    test_logits = model(
        test_token_ids
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Input shape:")
    print(test_token_ids.shape)

    print("\nOutput shape:")
    print(test_logits.shape)

    print("\nParameter count:")
    print(parameter_count)

    print("\nParameters in millions:")
    print(parameter_count / 1_000_000)