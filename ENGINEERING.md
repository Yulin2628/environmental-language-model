# Decoder execution

The existing `EnvironmentalLanguageModel` remains the training/reference implementation. `Decoder` uses its weights directly and changes the execution path without adding learned parameters. Old checkpoints load strictly through a restricted weights-only loader, including the original dataclass configuration where present.

## Prefill and decoding

Prefill processes the prompt once. Each of the six layers returns projected keys and values shaped `[batch, heads, sequence, head_dim]`. A decode step projects only the new query/key/value token and attends to the preceding keys. The cache belongs to the request and is passed explicitly; interleaved requests do not share hidden state.

Manual attention remains available as a readable reference. The SDPA path delegates to PyTorch's supported attention backend. A CPU-side operator trace records `aten::_scaled_dot_product_efficient_attention` in both observed precisions. This is backend dispatch evidence, not a custom kernel or a CUDA-kernel timing trace.

Full prefill uses a square causal mask. A one-token cached decode attends to every available key and must not use an upper-left triangular mask. A multi-token update builds a boolean mask using absolute query positions. The behavior follows [PyTorch's documented SDPA mask conventions](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).

Generation projects only the final hidden position into the vocabulary. The model's original forward still returns logits for every position for training and teacher-forced evaluation.

## Position-window semantics

The model uses learned absolute positions, not RoPE. Within the window, the next token receives the position immediately after the cached prefix. Once the window exceeds 512 tokens, the reference implementation discards the oldest tokens and renumbers the remaining positions. Dropping the oldest cached key alone would retain obsolete position information at every layer. The generator therefore rebuilds the cache from the latest 512 input tokens whenever the window slides. Correctness is preserved, while the cache speed benefit disappears in that regime.

The cache is dynamic and uses concatenation, so it still incurs allocation and copy work. Static buffers, paged attention and continuous batching are future work. Equal-length batches need no padding mask; variable-length padded batches would need another mask and are not supported by this API.

## What is checked

CPU tests compare the new forward to the original implementation, compare loss and every parameter gradient with dropout disabled, exercise multi-token cache offsets, mutate future tokens, interleave requests and cross the position-window boundary. Data-split tests explicitly compare raw token spans, including next-token targets.

The saved trained-checkpoint study evaluates both original checkpoints at four batch/length combinations in FP32 and FP16. Tolerances and actual errors are recorded separately from top-1 agreement. Fixed-length greedy benchmark sequences are compared directly with the original execution at the same precision. These checks do not prove equality for every possible input or a production serving stack.

## Reading the measurements

The ablations are the original implementation, a manual decoder with last-token projection, SDPA without cache, manual attention with cache, and SDPA with cache. The original/manual-decoder comparison includes minor wrapper and mask-construction differences, so it is an estimate of the head-projection change rather than a perfectly isolated causal effect. Backend and cache comparisons within the decoder keep the other execution choices fixed.

Timings synchronize CUDA around prefill and decode phases, retain raw repetitions and randomize mode order. Request latency includes Python dispatch and device allocation. Peak allocated memory includes model weights, active tensors and the KV cache. It excludes allocator reservation and CUDA driver overhead. The combined optimization trades persistent cache storage for less repeated computation; memory can increase.

No source weights are changed by inference. QA validation and generated text are audited separately. Faster tokens are not evidence of more accurate environmental answers.
