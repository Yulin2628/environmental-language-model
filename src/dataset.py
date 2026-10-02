from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from config import ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.json"


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        text_files,
        tokenizer_path,
        context_length,
        stride,
    ):
        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.context_length = context_length
        self.stride = stride
        self.token_ids = []

        eos_id = self.tokenizer.token_to_id("<eos>")

        for text_file in text_files:
            text = text_file.read_text(
                encoding="utf-8"
            )

            encoding = self.tokenizer.encode(text)

            self.token_ids.extend(encoding.ids)
            self.token_ids.append(eos_id)

        if len(self.token_ids) <= self.context_length:
            raise ValueError(
                "The corpus is shorter than the context length."
            )

    def __len__(self):
        usable_tokens = (
            len(self.token_ids)
            - self.context_length
            - 1
        )

        return usable_tokens // self.stride + 1

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.context_length + 1

        token_chunk = self.token_ids[start:end]

        input_ids = torch.tensor(
            token_chunk[:-1],
            dtype=torch.long,
        )

        target_ids = torch.tensor(
            token_chunk[1:],
            dtype=torch.long,
        )

        return input_ids, target_ids


if __name__ == "__main__":
    config = ModelConfig()

    text_files = sorted(
        DATA_DIR.glob("*.txt")
    )

    dataset = LanguageModelDataset(
        text_files=text_files,
        tokenizer_path=TOKENIZER_PATH,
        context_length=config.context_length,
        stride=256,
    )

    input_ids, target_ids = dataset[0]

    print(
        "Number of training examples:",
        len(dataset),
    )

    print(
        "Input shape:",
        input_ids.shape,
    )

    print(
        "Target shape:",
        target_ids.shape,
    )

    print("\nFirst 10 input IDs:")
    print(input_ids[:10])

    print("\nFirst 10 target IDs:")
    print(target_ids[:10])

    print("\nDecoded input:")
    print(
        dataset.tokenizer.decode(
            input_ids[:20].tolist()
        )
    )

    print("\nDecoded target:")
    print(
        dataset.tokenizer.decode(
            target_ids[:20].tolist()
        )
    )

    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
    )

    batch_input_ids, batch_target_ids = next(
        iter(data_loader)
    )

    print("\nBatch input shape:")
    print(batch_input_ids.shape)

    print("\nBatch target shape:")
    print(batch_target_ids.shape)