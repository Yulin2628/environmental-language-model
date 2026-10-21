# Environmental Language Model

A small decoder-only Transformer built in PyTorch and trained from scratch on text from IPCC and UNEP reports. Includes corpus preparation, BPE tokenisation, pre-training and question-answer fine-tuning.

## Model

About 25.3M parameters: 6 pre-LayerNorm blocks, 8 attention heads, a hidden size of 512 and a feed-forward size of 2,048. The context window is 512 tokens. Token and position embeddings are learned; input and output token weights are tied. The ByteLevel BPE vocabulary has 12,000 tokens.

Model settings are in `src/config.py`; the Transformer is in `src/model.py`.

## Setup

Developed with Python 3.11. CUDA is recommended for training; CPU is also supported.

```bash
python -m pip install -r requirements.txt
```

Run the commands below from the repository root.

## Training

Place PDF reports in `data/raw/`, then extract the text into `data/processed/`:

```bash
python src/prepare_corpus.py
```

A trained tokenizer is included as `tokenizer.json`. To retrain it on your corpus:

```bash
python src/train_tokenizer.py
```

Pre-train the model:

```bash
python src/train.py
```

Training uses AdamW, gradient clipping and mixed precision on CUDA. Checkpoints go to `checkpoints/`; loss history goes to `results/base_training_history.json`. Training settings are at the top of `src/train.py`.

For question-answer fine-tuning, use `data/finetune/train.jsonl` and `validation.jsonl`, with one pair per line:

```json
{"question": "What is climate change?", "answer": "Climate change is a long-term change in climate patterns."}
```

```bash
python src/finetune.py
```

Only answer tokens contribute to the fine-tuning loss. The script loads `checkpoints/base_model_epoch_1.pt` and saves `checkpoints/finetuned_model.pt`. Adjust the paths in `src/finetune.py` if needed.

## Generation

Generate text from the base model:

```bash
python src/generate.py
```

Start the fine-tuned question-answer interface:

```bash
python src/chat.py
```

Generation uses temperature and top-k sampling. Type `quit` to exit the chat.

## Data and limitations

The source PDFs, extracted corpus and model checkpoints are not included. The repository contains the tokenizer, a small question-answer dataset and training loss history.

This is a learning project with limited training data. Responses can be repetitive or incorrect; there is no systematic held-out evaluation yet.
