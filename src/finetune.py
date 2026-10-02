from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import ModelConfig
from finetune_dataset import FineTuningDataset
from model import EnvironmentalLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.json"

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "finetune"
    / "train.jsonl"
)

VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "finetune"
    / "validation.jsonl"
)

BASE_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "base_model_epoch_1.pt"
)

FINETUNED_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "finetuned_model.pt"
)


BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
MAX_LENGTH = 256


def calculate_loss(
    model,
    input_ids,
    target_ids,
    loss_function,
):
    logits = model(input_ids)

    logits_flat = logits.reshape(
        -1,
        logits.shape[-1],
    )

    targets_flat = target_ids.reshape(-1)

    loss = loss_function(
        logits_flat,
        targets_flat,
    )

    return loss


def evaluate(
    model,
    validation_loader,
    loss_function,
    device,
):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for input_ids, target_ids in validation_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            loss = calculate_loss(
                model=model,
                input_ids=input_ids,
                target_ids=target_ids,
                loss_function=loss_function,
            )

            total_loss += loss.item()

    average_loss = (
        total_loss / len(validation_loader)
    )

    return average_loss


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = ModelConfig()

    training_dataset = FineTuningDataset(
        data_path=TRAIN_DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        max_length=MAX_LENGTH,
    )

    validation_dataset = FineTuningDataset(
        data_path=VALIDATION_DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        max_length=MAX_LENGTH,
    )

    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = EnvironmentalLanguageModel(config)

    checkpoint = torch.load(
        BASE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    print("Device:", device)
    print(
        "Base checkpoint:",
        BASE_CHECKPOINT_PATH,
    )
    print(
        "Training examples:",
        len(training_dataset),
    )
    print(
        "Validation examples:",
        len(validation_dataset),
    )

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_training_loss = 0.0

        for input_ids, target_ids in training_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            loss = calculate_loss(
                model=model,
                input_ids=input_ids,
                target_ids=target_ids,
                loss_function=loss_function,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_training_loss += loss.item()

        average_training_loss = (
            total_training_loss
            / len(training_loader)
        )

        average_validation_loss = evaluate(
            model=model,
            validation_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"training loss: "
            f"{average_training_loss:.4f} | "
            f"validation loss: "
            f"{average_validation_loss:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "number_of_epochs": NUM_EPOCHS,
            "training_loss": average_training_loss,
            "validation_loss": average_validation_loss,
        },
        FINETUNED_CHECKPOINT_PATH,
    )

    print("\nFine-tuning finished.")
    print(
        "Fine-tuned checkpoint saved to:",
        FINETUNED_CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()