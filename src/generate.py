from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import EnvironmentalLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.json"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "base_model_epoch_1.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50


def generate_text(
    model,
    tokenizer,
    prompt,
    config,
    device,
):
    encoding = tokenizer.encode(prompt)

    prompt_token_ids = encoding.ids

    if not prompt_token_ids:
        raise ValueError(
            "The prompt produced no tokens."
        )

    token_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.token_to_id(
        "<eos>"
    )

    for _ in range(MAX_NEW_TOKENS):
        context_token_ids = token_ids[
            :,
            -config.context_length:,
        ]

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(
                    context_token_ids
                )

        next_token_logits = logits[
            :,
            -1,
            :,
        ]

        next_token_logits = (
            next_token_logits
            / TEMPERATURE
        )

        number_of_candidates = min(
            TOP_K,
            next_token_logits.shape[-1],
        )

        top_values, top_indices = torch.topk(
            next_token_logits,
            k=number_of_candidates,
            dim=-1,
        )

        probabilities = torch.softmax(
            top_values,
            dim=-1,
        )

        sampled_position = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        next_token_id = top_indices.gather(
            dim=-1,
            index=sampled_position,
        )

        token_ids = torch.cat(
            [token_ids, next_token_id],
            dim=1,
        )

        if (
            eos_id is not None
            and next_token_id.item() == eos_id
        ):
            break

    generated_token_ids = (
        token_ids[0]
        .detach()
        .cpu()
        .tolist()
    )

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

    return generated_text


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = ModelConfig()

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    model = EnvironmentalLanguageModel(
        config
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("Device:", device)
    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"],
    )

    prompt = input(
        "\nEnter an English prompt: "
    ).strip()

    if not prompt:
        prompt = "Climate change"

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
        device=device,
    )

    print("\nGenerated text:\n")
    print(generated_text)

    output_path = (
        RESULTS_DIR
        / "generated_sample.txt"
    )

    output_path.write_text(
        generated_text,
        encoding="utf-8",
    )

    print(
        "\nGenerated text saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()