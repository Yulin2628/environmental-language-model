from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from config import ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.json"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_tokenizer():
    text_files = list(DATA_DIR.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            f"No text files were found in {DATA_DIR}"
        )

    config = ModelConfig()

    tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train(
        files=[str(path) for path in text_files],
        trainer=trainer,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    return tokenizer


if __name__ == "__main__":
    tokenizer = train_tokenizer()

    test_text = "Climate change affects global temperatures."
    encoding = tokenizer.encode(test_text)

    print("Original text:", test_text)
    print("Tokens:", encoding.tokens)
    print("Token IDs:", encoding.ids)
    print("Decoded text:", tokenizer.decode(encoding.ids))
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Saved to:", TOKENIZER_PATH)