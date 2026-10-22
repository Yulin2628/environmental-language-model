"""Checkpoint-compatible attention and request-local KV caching."""
import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

from runtime import load_model, load_tokenizer


class Decoder(nn.Module):
    def __init__(self, model, backend="sdpa"):
        super().__init__()
        if backend not in {"manual", "sdpa"}:
            raise ValueError("backend must be manual or sdpa")
        self.model = model
        self.backend = backend
        self.context_length = model.embedding.context_length

    def forward(self, input_ids, cache=None, use_cache=False, last_only=False):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
        if use_cache and self.model.training:
            raise ValueError("KV caching requires model.eval()")
        batch, length = input_ids.shape
        past = 0
        if cache is not None:
            if not use_cache or len(cache) != len(self.model.blocks):
                raise ValueError("cache requires use_cache and one key/value pair per layer")
            past = cache[0][0].shape[2]
            for block, (key, value) in zip(self.model.blocks, cache):
                attention = block.attention
                expected = (batch, attention.n_heads, past, attention.head_dim)
                if tuple(key.shape) != expected or tuple(value.shape) != expected:
                    raise ValueError("cache batch, head, sequence or channel shape mismatch")
                if key.device != input_ids.device or value.device != input_ids.device:
                    raise ValueError("cache and input must be on the same device")
        if past + length > self.context_length:
            raise ValueError("cached sequence exceeds the learned position window")
        positions = torch.arange(past, past + length, device=input_ids.device)
        hidden = (self.model.embedding.token_embedding(input_ids)
                  + self.model.embedding.position_embedding(positions))
        updated = []
        for index, block in enumerate(self.model.blocks):
            attention = block.attention
            q, k, v = attention.qkv_projection(block.norm_before_attention(hidden))
            def heads(tensor):
                return tensor.view(batch, length, attention.n_heads, attention.head_dim).transpose(1, 2)
            q, k, v = map(heads, (q, k, v))
            if past:
                old_k, old_v = cache[index]
                if old_k.dtype != k.dtype or old_v.dtype != v.dtype:
                    raise ValueError("cache precision must match the current attention precision")
                k = torch.cat((old_k, k), dim=2)
                v = torch.cat((old_v, v), dim=2)
            if use_cache:
                updated.append((k, v))
            mask = None
            # A cached query is offset in the key sequence. A square causal
            # mask would incorrectly hide its earlier keys.
            if self.backend == "manual" or (past and length > 1):
                mask = torch.arange(past + length, device=q.device)[None, :] <= positions[:, None]
            if self.backend == "manual":
                scores = (q @ k.transpose(-2, -1)) / attention.head_dim ** 0.5
                result = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v
            else:
                result = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=(past == 0))
            result = result.transpose(1, 2).contiguous().view(batch, length, attention.d_model)
            hidden = hidden + attention.output_projection(result)
            hidden = hidden + block.feed_forward(block.norm_before_ffn(hidden))
        if last_only:
            hidden = hidden[:, -1:, :]
        logits = self.model.output_head(self.model.final_norm(hidden))
        return logits, tuple(updated) if use_cache else None


@torch.inference_mode()
def greedy_generate(decoder, input_ids, max_new_tokens=64, use_cache=True, eos_id=None):
    if decoder.model.training:
        raise ValueError("generation requires model.eval()")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    tokens = input_ids.clone()
    cache = None
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    for _ in range(max_new_tokens):
        if use_cache and cache is not None and cache[0][0].shape[2] < decoder.context_length:
            current = tokens[:, -1:]
        else:
            # Learned absolute positions shift when the window slides. Rebuild
            # every layer instead of reusing keys with stale position vectors.
            cache = None
            current = tokens[:, -decoder.context_length:]
        logits, cache = decoder(current, cache=cache, use_cache=use_cache, last_only=True)
        next_token = logits[:, -1].argmax(-1)
        if eos_id is not None:
            next_token = torch.where(finished, eos_id, next_token)
            finished |= next_token == eos_id
        tokens = torch.cat((tokens, next_token[:, None]), dim=1)
        if eos_id is not None and bool(finished.all()):
            break
    return tokens


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--backend", choices=["manual", "sdpa"], default="sdpa")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, _, _ = load_model(args.checkpoint, args.device)
    tokenizer = load_tokenizer()
    prompt = f"Question: {args.question}\nAnswer:"
    ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(prompt).ids
    inputs = torch.tensor([ids], device=args.device)
    output = greedy_generate(Decoder(model, args.backend), inputs, args.max_new_tokens,
                             not args.no_cache, tokenizer.token_to_id("<eos>"))
    print(json.dumps({"answer": tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True),
                      "generated_tokens": output.shape[1] - len(ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
