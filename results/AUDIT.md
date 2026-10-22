# Checkpoint and metric audit

The screenshot claim of 200 held-out examples, a 23% answer-loss reduction and a 7% to 28% target-token match improvement could not be traced to an evaluation file. The published repository and local source contain 13 training pairs and 3 validation pairs, with one duplicated training question. These are the only QA validation examples used below. They were also evaluated during the original fine-tuning process, so this is a validation-set reproduction, not an independent test benchmark.

| Metric on the same 3 validation pairs | Base | Fine-tuned |
| --- | ---: | ---: |
| Token-weighted answer NLL, including EOS | 6.468261 | 5.403364 |
| Mean of example losses, as in the original trainer | 6.442120 | 5.408775 |
| Teacher-forced top-1 token match, including EOS | 7.25% | 20.29% |
| Teacher-forced top-1 token match, excluding EOS | 7.58% | 16.67% |

The token-weighted NLL reduction is 16.46%; the example-mean reduction is 16.04%. Each run scores 69 target positions, including EOS. Teacher forcing supplies the correct preceding answer tokens; it does not measure whether a free-running answer is factually correct. The three questions cannot support a reliable generalization claim.

## Architecture and provenance

25,309,184 unique parameters, six pre-LayerNorm blocks, eight attention heads, hidden size 512, feed-forward size 2,048, learned positions up to 512 and a 12,000-token ByteLevel BPE tokenizer. The input and output token weights are tied. Both checkpoint files load strictly against the repository architecture. audit.json records checkpoint and tokenizer SHA-256 hashes, checkpoint metadata and every scored example.

The base checkpoint records one epoch with training loss 5.939303 and validation loss 7.063682. The fine-tuned checkpoint records ten epochs, training loss 3.470508 and validation loss 5.408775. The latter validation loss is reproduced by the per-example mean above. Saving checkpoints is implemented; activation/gradient checkpointing is not claimed.

## Data boundary

The local processed corpus has 222,473 tokens including report separators: 143,909 IPCC tokens, 78,482 UNEP tokens and 79 tokens in sample.txt. The old pretraining code uses 868 overlapping windows, with 781 train and 87 validation windows. At the boundary, 256 raw input positions and 256 next-token targets are shared. Within-split neighboring windows overlap too, so window counts are not independent document counts. The tokenizer was trained on the available corpus; no unseen-document evaluation is established.

Future pretraining now purges two boundary windows, leaving 781 train and 85 validation windows for this corpus. The fix removes shared raw token positions between the windows, including their next-token targets. It does not create document-level independence, retrain the tokenizer, or repair the historical checkpoint's training history. The original checkpoints were not retrained in this upgrade. A stronger language-quality study needs a larger corpus, document-level partitions fixed before tokenizer training, and an independent question set.

## Greedy outputs

Greedy decoding uses the training-style BOS and prompt, at most 64 new tokens and EOS stopping. The legacy chat omitted BOS and used sampling, so these are deterministic diagnostic outputs rather than an exact replay of a sampled chat.

### How do fossil fuels contribute to climate change?

Reference: Burning coal, oil and natural gas releases carbon dioxide and other greenhouse gases, which increase heat retention in the atmosphere.

Base:   United States of  of  of  United States of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the

Fine-tuned:  Climate change to the atmosphere.

### What is the difference between mitigation and adaptation?

Reference: Mitigation addresses the causes of climate change by reducing emissions, while adaptation reduces vulnerability to its effects.

Base:   of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the  of the 

Fine-tuned:  Climate change to be dioxide and dioxide in the atmosphere.

### Why are oceans important in the climate system?

Reference: Oceans absorb and redistribute large amounts of heat and carbon dioxide, strongly influencing global climate and weather patterns.

Base:   United States of the  United States of the  of the  United States of the  United  United States of the  United  of the  of the  of the  United States of the  of the  of the  of the  of the  of the  of the  of the  of

Fine-tuned:  Climate change in the atmosphere.

These examples show that the fine-tuned checkpoint still gives incomplete or incorrect answers. The inference upgrade changes execution and measurement, not the model's knowledge.

## Reproduce

```bash
python audit_model.py --checkpoint-dir checkpoints --corpus-dir data/processed
```

The raw PDFs, report text and checkpoints remain local. Source files and historical loss history are preserved. No 200-question dataset or score is inferred from the screenshot.
