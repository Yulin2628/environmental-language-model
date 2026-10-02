import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from config import ModelConfig
from dataset import LanguageModelDataset
from model import EnvironmentalLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4
TRAIN_FRACTION = 0.9


def calculate_loss(
    model,
    input_ids,
    target_ids,
    loss_function,
    config,
):
    logits = model(input_ids)

    logits_flat = logits.reshape(
        -1,
        config.vocab_size,
    )

    targets_flat = target_ids.reshape(
        -1
    )

    loss = loss_function(
        logits_flat,
        targets_flat,
    )

    return loss


def evaluate(
    model,
    validation_loader,
    loss_function,
    config,
    device,
    use_mixed_precision,
):
    model.eval()

    total_loss = 0.0
    number_of_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in validation_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_mixed_precision,
            ):
                loss = calculate_loss(
                    model=model,
                    input_ids=input_ids,
                    target_ids=target_ids,
                    loss_function=loss_function,
                    config=config,
                )

            total_loss += loss.item()
            number_of_batches += 1

    average_loss = (
        total_loss / number_of_batches
    )

    return average_loss


def main():
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = ModelConfig()

    text_files = sorted(
        DATA_DIR.glob("*.txt")
    )

    if not text_files:
        raise FileNotFoundError(
            "No processed text files were found."
        )

    complete_dataset = LanguageModelDataset(
        text_files=text_files,
        tokenizer_path=TOKENIZER_PATH,
        context_length=config.context_length,
        stride=256,
    )

    number_of_examples = len(
        complete_dataset
    )

    number_of_training_examples = int(
        number_of_examples * TRAIN_FRACTION
    )

    training_indices = range(
        0,
        number_of_training_examples,
    )

    validation_indices = range(
        number_of_training_examples,
        number_of_examples,
    )

    training_dataset = Subset(
        complete_dataset,
        training_indices,
    )

    validation_dataset = Subset(
        complete_dataset,
        validation_indices,
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_mixed_precision = (
        device.type == "cuda"
    )

    model = EnvironmentalLanguageModel(
        config
    )

    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_mixed_precision,
    )

    training_history = {
        "training_loss": [],
        "validation_loss": [],
    }

    print("Device:", device)
    print("Mixed precision:", use_mixed_precision)
    print("Total examples:", number_of_examples)
    print(
        "Training examples:",
        len(training_dataset),
    )
    print(
        "Validation examples:",
        len(validation_dataset),
    )
    print("Number of epochs:", NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_training_loss = 0.0

        for step, (
            input_ids,
            target_ids,
        ) in enumerate(training_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_mixed_precision,
            ):
                loss = calculate_loss(
                    model=model,
                    input_ids=input_ids,
                    target_ids=target_ids,
                    loss_function=loss_function,
                    config=config,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            total_training_loss += loss.item()

            if (step + 1) % 50 == 0:
                current_average_loss = (
                    total_training_loss
                    / (step + 1)
                )

                print(
                    f"Epoch {epoch + 1}, "
                    f"step {step + 1}/"
                    f"{len(training_loader)}, "
                    f"average loss: "
                    f"{current_average_loss:.4f}"
                )

        average_training_loss = (
            total_training_loss
            / len(training_loader)
        )

        average_validation_loss = evaluate(
            model=model,
            validation_loader=validation_loader,
            loss_function=loss_function,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )

        training_history[
            "training_loss"
        ].append(average_training_loss)

        training_history[
            "validation_loss"
        ].append(average_validation_loss)

        print(
            f"\nEpoch {epoch + 1} finished"
        )

        print(
            "Training loss:",
            round(average_training_loss, 4),
        )

        print(
            "Validation loss:",
            round(average_validation_loss, 4),
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"base_model_epoch_{epoch + 1}.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scaler_state_dict": (
                    scaler.state_dict()
                ),
                "config": asdict(config),
                "training_loss": (
                    average_training_loss
                ),
                "validation_loss": (
                    average_validation_loss
                ),
            },
            checkpoint_path,
        )

        print(
            "Checkpoint saved to:",
            checkpoint_path,
        )

    history_path = (
        RESULTS_DIR
        / "base_training_history.json"
    )

    history_path.write_text(
        json.dumps(
            training_history,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nTraining finished."
    )

    print(
        "Training history saved to:",
        history_path,
    )


if __name__ == "__main__":
    main()