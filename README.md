# Environmental Language Model

This project implements a small decoder-only Transformer language model for environmental text. The model was built in PyTorch and trained from scratch using text extracted from IPCC and UNEP reports.

The purpose of the project was to understand the complete language-model pipeline, including text preparation, tokenisation, dataset construction, Transformer implementation, pre-training, supervised fine-tuning and autoregressive generation.

## Model architecture

The model contains approximately 25.3 million parameters and uses:

- 6 pre-LayerNorm Transformer blocks
- 8 causal self-attention heads
- 512-dimensional token representations
- A context length of 512 tokens
- A 2,048-dimensional feed-forward layer
- Learned token and positional embeddings
- Tied input and output embedding weights
- A 12,000-token ByteLevel BPE vocabulary

The main Transformer components, including multi-head attention, causal masking, residual connections and feed-forward layers, were implemented directly in PyTorch.

## Project pipeline

The project follows this pipeline:

1. Extract text from environmental reports.
2. Train a ByteLevel BPE tokenizer.
3. Convert the processed corpus into next-token prediction examples.
4. Pre-train a decoder-only Transformer.
5. Prepare question-answer data for supervised fine-tuning.
6. Fine-tune the base model on answer tokens.
7. Generate responses using temperature and top-k sampling.

## Project structure

```text
environmental_slm_project/
├── data/
│   └── finetune/
│       ├── train.jsonl
│       └── validation.jsonl
├── src/
│   ├── config.py
│   ├── prepare_corpus.py
│   ├── train_tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── finetune_dataset.py
│   ├── finetune.py
│   └── chat.py
├── results/
├── tokenizer.json
├── requirements.txt
└── README.md
```

## File descriptions

- `config.py`: stores the model hyperparameters.
- `prepare_corpus.py`: extracts and cleans text from PDF reports.
- `train_tokenizer.py`: trains and saves the ByteLevel BPE tokenizer.
- `dataset.py`: creates next-token prediction examples.
- `model.py`: implements the decoder-only Transformer.
- `train.py`: pre-trains the base language model.
- `generate.py`: generates text using the base model.
- `finetune_dataset.py`: prepares question-answer fine-tuning data and masks prompt tokens from the loss.
- `finetune.py`: loads the base checkpoint and performs supervised fine-tuning.
- `chat.py`: loads the fine-tuned checkpoint and provides an interactive question-answer interface.

## Installation

Python 3.11 was used for this project.

Install the required libraries with:

```bash
python -m pip install -r requirements.txt
```

A CUDA-compatible GPU is recommended for training, although the code can also run on the CPU.

## Data preparation

The original reports and extracted corpus are not included in this repository.

To reproduce the data-processing stage, place the relevant PDF reports inside:

```text
data/raw/
```

Then run:

```bash
python src/prepare_corpus.py
```

The extracted text will be saved in:

```text
data/processed/
```

## Training the tokenizer

Train the ByteLevel BPE tokenizer with:

```bash
python src/train_tokenizer.py
```

This creates:

```text
tokenizer.json
```

## Pre-training

Start base-model training with:

```bash
python src/train.py
```

The training pipeline uses:

- Cross-entropy loss
- AdamW optimisation
- Mixed-precision training on CUDA
- Gradient clipping
- Training and validation splits
- Model checkpointing

Model checkpoints are saved locally in the `checkpoints` directory and are excluded from the repository because of their size.

## Supervised fine-tuning

The fine-tuning data uses JSON Lines format:

```json
{"question": "What is climate change?", "answer": "Climate change is a long-term change in climate patterns."}
```

During fine-tuning, the question is provided as context, but only the answer tokens contribute to the loss. Prompt and padding positions are assigned a target value of `-100`, which is ignored by the cross-entropy loss.

Run fine-tuning with:

```bash
python src/finetune.py
```

## Interactive generation

After fine-tuning, start the question-answer interface with:

```bash
python src/chat.py
```

The model generates one token at a time using temperature scaling and top-k sampling.

Enter:

```text
quit
```

to stop the program.

## Current limitations

This repository demonstrates an end-to-end language-model pipeline rather than a production-ready question-answering system.

The model was trained from scratch on a limited corpus and fine-tuned using a small demonstration dataset. Its generated responses may therefore be repetitive, grammatically inconsistent or factually unreliable.

Potential improvements include:

- Expanding and cleaning the training corpus
- Creating a larger question-answer dataset
- Training for more steps
- Saving the checkpoint with the lowest validation loss
- Adding systematic held-out evaluation
- Comparing the model with a pre-trained baseline

## What I learned

This project helped me understand how tokenisation, causal attention, next-token prediction, optimisation and autoregressive generation work together in a complete language-model system. It also demonstrated the difference between making a training pipeline function correctly and developing a model that generalises reliably.