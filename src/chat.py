from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import EnvironmentalLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "finetuned_model.pt"
)

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_K = 30


def generate_answer(
    model,
    tokenizer,
    question,
    config,
    device,
):
    prompt = (
        f"Question: {question}\n"
        f"Answer:"
    )

    prompt_ids = tokenizer.encode(
        prompt
    ).ids

    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.token_to_id(
        "<eos>"
    )

    for _ in range(MAX_NEW_TOKENS):
        context_ids = token_ids[
            :,
            -config.context_length:,
        ]

        with torch.inference_mode():
            logits = model(context_ids)

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

    answer_ids = token_ids[
        0,
        len(prompt_ids):,
    ].detach().cpu().tolist()

    answer = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    )

    return answer


def main():
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
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("Device:", device)
    print("Fine-tuned model loaded.")
    print("Type 'quit' to stop.\n")

    while True:
        question = input(
            "Your question: "
        ).strip()

        if question.lower() == "quit":
            print("Conversation ended.")
            break

        if not question:
            print(
                "Please enter a question.\n"
            )
            continue

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            config=config,
            device=device,
        )

        print("\nModel answer:")
        print(answer)
        print()


if __name__ == "__main__":
    main()