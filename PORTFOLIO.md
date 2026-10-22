# Portfolio positioning

| Project | Main evidence of ability |
| --- | --- |
| Environmental Language Model | Transformer implementation, attention/cache correctness, profiling, latency/memory tradeoffs and data leakage audit |
| CMU AG News | Controlled model comparisons, prompt development, LoRA training and statistical error analysis |
| AI Buyer Assistant | Catalogue constraints, retrieval/reranking, structured tool decisions and application reliability |

## Suggested CV wording

- Implemented a 25.3M-parameter decoder-only Transformer in PyTorch with six layers, eight attention heads, tied embeddings and a 512-token context window.
- Added request-local KV caching, PyTorch SDPA and last-token vocabulary projection, reducing median FP32 latency from 643 to 184 ms (3.51x) for a 384-token prefix and 64 generated tokens on an RTX 3050 Laptop GPU.
- Validated cached and uncached execution with causal-mask, gradient, request-isolation and context-rollover tests; built interleaved latency, throughput and GPU-memory benchmarks with nine repeats per configuration.

The speed figure describes one specified workload. Use the supporting table when discussing other batch sizes or shorter prompts. Do not substitute the unverified 200-example claim, call token matching QA accuracy, claim to have implemented FlashAttention kernels, or claim that faster inference improved factual knowledge.

## Further development

The next systems step would be preallocated KV buffers and a small continuous-batching scheduler, measured against the current dynamic cache. Long-context architectures such as RoPE would require a new training study; extending the learned positional table alone is not evidence that the model can use longer contexts. Better answers would require substantially better data and independent evaluation. These remain future work rather than completed features.
