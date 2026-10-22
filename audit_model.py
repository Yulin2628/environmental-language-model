"""Re-evaluate available checkpoints and preserve the exact evidence trail."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from torch.nn import functional as F
from finetune_dataset import FineTuningDataset
from inference import Decoder, greedy_generate
from runtime import ROOT, load_model, load_tokenizer


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.inference_mode()
def evaluate(model, dataset, data, tokenizer, device):
    cases = []
    for (inputs, targets), example in zip(dataset, data):
        logits = model(inputs[None].to(device))[0].float()
        targets = targets.to(device)
        active = targets != -100
        labels = targets[active]
        losses = F.cross_entropy(logits[active], labels, reduction="none")
        correct = logits[active].argmax(-1) == labels
        without_eos = labels != tokenizer.token_to_id("<eos>")
        prompt = f'Question: {example["question"]}\nAnswer:'
        prefix = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(prompt).ids
        generated = greedy_generate(Decoder(model), torch.tensor([prefix], device=device),
                                    64, True, tokenizer.token_to_id("<eos>"))[0, len(prefix):].tolist()
        cases.append({**example, "scored_tokens_including_eos": int(active.sum()),
                      "nll_sum": losses.sum().item(), "mean_nll": losses.mean().item(),
                      "correct_tokens_including_eos": int(correct.sum()),
                      "answer_tokens_excluding_eos": int(without_eos.sum()),
                      "correct_answer_tokens_excluding_eos": int(correct[without_eos].sum()),
                      "first_answer_token_correct": bool(correct[0]),
                      "generated_ids": generated,
                      "greedy_answer": tokenizer.decode(generated, skip_special_tokens=True)})
    total = sum(row["scored_tokens_including_eos"] for row in cases)
    nll = sum(row["nll_sum"] for row in cases) / total
    answer_total = sum(row["answer_tokens_excluding_eos"] for row in cases)
    return {"examples": len(cases), "tokens_including_eos": total,
            "token_weighted_nll": nll, "perplexity": math.exp(nll),
            "example_mean_nll": sum(row["mean_nll"] for row in cases) / len(cases),
            "teacher_forced_token_match_including_eos": sum(row["correct_tokens_including_eos"] for row in cases) / total,
            "teacher_forced_token_match_excluding_eos": sum(row["correct_answer_tokens_excluding_eos"] for row in cases) / answer_total,
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    paths = {name: ROOT / f"data/finetune/{name}.jsonl" for name in ("train", "validation")}
    data = {name: records(path) for name, path in paths.items()}
    normal = lambda text: " ".join(text.lower().split())
    train_questions = {normal(row["question"]) for row in data["train"]}
    val_questions = {normal(row["question"]) for row in data["validation"]}
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "precision": "float32, TF32 disabled", "tokenizer_sha256": sha(ROOT / "tokenizer.json"),
              "tokenizer_vocabulary": tokenizer.get_vocab_size(),
              "data": {name: {"examples": len(data[name]), "sha256": sha(path),
                       "normalized_lf_sha256": hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()}
                       for name, path in paths.items()},
              "question_overlap": sorted(train_questions & val_questions),
              "duplicate_training_questions": len(data["train"]) - len(train_questions),
              "claimed_200_example_evaluation": "not found in the checked repository or local source",
              "models": {}}
    validation = FineTuningDataset(paths["validation"], ROOT / "tokenizer.json", 256)
    for name, filename in [("base", "base_model_epoch_1.pt"), ("finetuned", "finetuned_model.pt")]:
        path = args.checkpoint_dir / filename
        model, config, metadata = load_model(path, device)
        item = {"checkpoint_sha256": sha(path), "checkpoint_bytes": path.stat().st_size,
                "metadata": metadata, "config": asdict(config),
                "unique_parameters": sum(p.numel() for p in model.parameters()),
                "input_output_weights_tied": model.output_head.weight is model.embedding.token_embedding.weight,
                "validation": evaluate(model, validation, data["validation"], tokenizer, device)}
        report["models"][name] = item
        print(name, {k: v for k, v in item["validation"].items() if k != "cases"}, flush=True)
        del model
    if args.corpus_dir:
        corpus = []
        all_ids = []
        for path in sorted(args.corpus_dir.glob("*.txt")):
            ids = tokenizer.encode(path.read_text(encoding="utf-8")).ids
            corpus.append({"file": path.name, "bytes": path.stat().st_size,
                           "tokens": len(ids), "sha256": sha(path)})
            all_ids.extend(ids + [tokenizer.token_to_id("<eos>")])
        windows = (len(all_ids) - 513) // 256 + 1
        split = int(windows * .9)
        report["pretraining_corpus"] = {"files": corpus, "total_tokens": len(all_ids),
                "windows": windows, "train_windows": split, "validation_windows": windows - split,
                "shared_input_token_positions_at_split": 256,
                "shared_next_token_targets_at_split": 256,
                "note": "512-token windows at stride 256 overlap across the original 90/10 split; this is not document holdout."}
    base = report["models"]["base"]["validation"]
    tuned = report["models"]["finetuned"]["validation"]
    report["comparison"] = {"token_nll_relative_reduction": 1 - tuned["token_weighted_nll"] / base["token_weighted_nll"],
                            "example_mean_nll_relative_reduction": 1 - tuned["example_mean_nll"] / base["example_mean_nll"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
