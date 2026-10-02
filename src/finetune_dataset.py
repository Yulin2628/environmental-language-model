import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.json"

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "finetune"
    / "train.jsonl"
)


class FineTuningDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer_path,
        max_length,
    ):
        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.max_length = max_length

        self.pad_id = self.tokenizer.token_to_id(
            "<pad>"
        )

        self.bos_id = self.tokenizer.token_to_id(
            "<bos>"
        )

        self.eos_id = self.tokenizer.token_to_id(
            "<eos>"
        )

        if None in (
            self.pad_id,
            self.bos_id,
            self.eos_id,
        ):
            raise ValueError(
                "The tokenizer is missing a special token."
            )

        self.examples = []

        with Path(data_path).open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                    question = record["question"].strip()
                    answer = record["answer"].strip()

                except (
                    json.JSONDecodeError,
                    KeyError,
                    AttributeError,
                ) as error:
                    raise ValueError(
                        f"Invalid data on line {line_number}"
                    ) from error

                prompt = (
                    f"Question: {question}\n"
                    f"Answer:"
                )

                prompt_ids = self.tokenizer.encode(
                    prompt
                ).ids

                answer_ids = self.tokenizer.encode(
                    " " + answer
                ).ids

                full_ids = (
                    [self.bos_id]
                    + prompt_ids
                    + answer_ids
                    + [self.eos_id]
                )

                if len(full_ids) > self.max_length:
                    full_ids = full_ids[
                        :self.max_length
                    ]

                    full_ids[-1] = self.eos_id

                input_ids = full_ids[:-1]
                target_ids = full_ids[1:]

                target_ids[
                    :len(prompt_ids)
                ] = (
                    [-100] * len(prompt_ids)
                )

                padding_length = (
                    self.max_length
                    - 1
                    - len(input_ids)
                )

                input_ids.extend(
                    [self.pad_id] * padding_length
                )

                target_ids.extend(
                    [-100] * padding_length
                )

                self.examples.append(
                    (
                        torch.tensor(
                            input_ids,
                            dtype=torch.long,
                        ),
                        torch.tensor(
                            target_ids,
                            dtype=torch.long,
                        ),
                    )
                )

        if not self.examples:
            raise ValueError(
                "The fine-tuning dataset is empty."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


if __name__ == "__main__":
    dataset = FineTuningDataset(
        data_path=TRAIN_DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        max_length=256,
    )

    input_ids, target_ids = dataset[0]

    answer_token_ids = target_ids[
        target_ids != -100
    ].tolist()

    print("Number of examples:")
    print(len(dataset))

    print("\nInput shape:")
    print(input_ids.shape)

    print("\nTarget shape:")
    print(target_ids.shape)

    print("\nNumber of answer tokens:")
    print(len(answer_token_ids))

    print("\nAnswer used for training:")
    print(
        dataset.tokenizer.decode(
            answer_token_ids,
            skip_special_tokens=True,
        )
    )