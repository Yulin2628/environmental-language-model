"""Check trained-checkpoint logits before interpreting speed measurements."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from inference import Decoder
from runtime import ROOT, load_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in ("base_model_epoch_1", "finetuned_model"):
        model, config, _ = load_model(args.checkpoint_dir / (name + ".pt"), device)
        for precision in ("float32", "float16") if device == "cuda" else ("float32",):
            model.to(dtype=getattr(torch, precision))
            for batch, length in ((1, 32), (1, 128), (1, 384), (4, 128)):
                ids = torch.randint(4, config.vocab_size, (batch, length), device=device)
                reference = model(ids)
                for backend in ("manual", "sdpa"):
                    decoder = Decoder(model, backend)
                    full, _ = decoder(ids)
                    prefill, cache = decoder(ids[:, :-3], use_cache=True)
                    suffix, _ = decoder(ids[:, -3:], cache, use_cache=True)
                    cached = torch.cat((prefill, suffix), dim=1)
                    row = {"checkpoint": name, "precision": precision, "batch": batch,
                           "sequence": length, "backend": backend,
                           "full_max_abs_logit_error": (full-reference).abs().max().item(),
                           "cached_max_abs_logit_error": (cached-reference).abs().max().item(),
                           "cached_top1_agreement": (cached.argmax(-1) == reference.argmax(-1)).float().mean().item()}
                    tolerance = .05 if precision == "float16" else .0003
                    row["absolute_tolerance"] = tolerance
                    row["passed"] = max(row["full_max_abs_logit_error"], row["cached_max_abs_logit_error"]) < tolerance
                    rows.append(row)
        del model
    result = {"seed": 19, "torch": torch.__version__, "device": device,
              "all_passed": all(row["passed"] for row in rows), "checks": rows}
    (ROOT / "results/optimization_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Trained-checkpoint logit checks:", len(rows), "passed:", result["all_passed"], flush=True)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
