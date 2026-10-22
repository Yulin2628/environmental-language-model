"""Record actual SDPA operator dispatch; profiling timings are not benchmarks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from inference import Decoder, greedy_generate
from runtime import ROOT, load_model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, _ = load_model(args.checkpoint, device)
    inputs = torch.randint(4, config.vocab_size, (1, 128), device=device)
    rows = []
    for precision in ("float32", "float16") if device == "cuda" else ("float32",):
        model.to(dtype=getattr(torch, precision))
        greedy_generate(Decoder(model), inputs, 8)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            greedy_generate(Decoder(model), inputs, 8)
        rows.append({"precision": precision,
                     "attention_dispatch": [{"operator": event.key, "calls": event.count}
                                            for event in profile.key_averages()
                                            if "scaled_dot_product" in event.key or "flash_attention" in event.key]})
    (ROOT / "results/attention_dispatch.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
