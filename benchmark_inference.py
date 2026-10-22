"""Interleaved inference ablations with synchronized raw timing samples."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from inference import Decoder
from runtime import ROOT, load_model


MODES = {"original": (None, False), "manual_last": ("manual", False),
         "sdpa_last": ("sdpa", False), "manual_cache": ("manual", True),
         "sdpa_cache": ("sdpa", True)}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def request(model, mode, prefix, new_tokens):
    backend, cached = MODES[mode]
    decoder = Decoder(model, backend) if backend else None
    device = prefix.device
    cache = None
    tokens = prefix.clone()
    def forward(current, cache):
        if decoder is None:
            return model(current), None
        return decoder(current, cache=cache, use_cache=cached, last_only=True)
    sync(device)
    start = time.perf_counter()
    logits, cache = forward(tokens, cache)
    next_id = logits[:, -1].argmax(-1)
    tokens = torch.cat((tokens, next_id[:, None]), dim=1)
    del logits
    sync(device)
    first = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits, cache = forward(tokens[:, -1:] if cached else tokens, cache)
        next_id = logits[:, -1].argmax(-1)
        tokens = torch.cat((tokens, next_id[:, None]), dim=1)
        del logits
    sync(device)
    end = time.perf_counter()
    cache_bytes = sum(key.numel() * key.element_size() + value.numel() * value.element_size()
                      for key, value in cache) if cache else 0
    return tokens, {"time_to_first_token_ms": (first-start)*1000,
                    "decode_ms": (end-first)*1000, "request_ms": (end-start)*1000,
                    "decode_tokens_per_second": prefix.shape[0]*(new_tokens-1)/(end-first),
                    "generated_tokens_per_second": prefix.shape[0]*new_tokens/(end-start),
                    "cache_bytes": cache_bytes}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmark.json")
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1 or args.new_tokens < 2 or args.new_tokens > 128:
        parser.error("Use >=3 repeats, >=1 warmup and 2-128 generated tokens")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This latency/memory report requires CUDA; CPU correctness tests do not.")
    model, config, _ = load_model(args.checkpoint, device)
    if args.precision == "float16":
        model.half()
    report = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "python": platform.python_version(), "torch": torch.__version__,
              "precision": args.precision, "tf32": False, "cpu_threads": 4,
              "warmups_per_mode": args.warmup, "repeats": args.repeats,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "method": "batch requests, synthetic token IDs, fixed decode length, EOS ignored; synchronize CUDA at phase boundaries; random interleaving within repeat",
              "scenarios": []}
    for batch, length in [(1, 32), (1, 128), (1, 384), (4, 128)]:
        if length + args.new_tokens - 1 > config.context_length:
            continue
        prefix = torch.randint(4, config.vocab_size, (batch, length), device=device)
        scenario = {"batch": batch, "prefix_tokens": length, "new_tokens": args.new_tokens,
                    "prefix_ids": prefix.cpu().tolist(), "modes": {mode: {"samples": []} for mode in MODES}}
        references = None
        for mode in MODES:
            for _ in range(args.warmup):
                actual, _ = request(model, mode, prefix, args.new_tokens)
            if mode == "original":
                references = actual
            scenario["modes"][mode]["greedy_tokens_equal_original"] = bool(torch.equal(actual, references))
            scenario["modes"][mode]["generated_ids"] = actual[:, length:].cpu().tolist()
        rng = random.Random(42)
        for repeat in range(args.repeats):
            order = list(MODES)
            rng.shuffle(order)
            for mode in order:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    allocated_before = torch.cuda.memory_allocated(device)
                actual, sample = request(model, mode, prefix, args.new_tokens)
                sample["repeat"] = repeat
                sample["order"] = order.index(mode)
                sample["greedy_tokens_equal_original"] = bool(torch.equal(actual, references))
                if device.type == "cuda":
                    sample["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    sample["peak_incremental_bytes"] = sample["peak_allocated_bytes"] - allocated_before
                scenario["modes"][mode]["samples"].append(sample)
        for mode, item in scenario["modes"].items():
            metrics = [key for key in item["samples"][0] if key not in {"repeat", "order", "greedy_tokens_equal_original"}]
            item["median"] = {key: statistics.median(row[key] for row in item["samples"]) for key in metrics}
            times = sorted(row["request_ms"] for row in item["samples"])
            item["request_ms_range"] = [times[0], times[-1]]
            item["request_ms_iqr"] = [times[len(times)//4], times[3*len(times)//4]]
        report["scenarios"].append(scenario)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(batch, length, {name: round(item["median"]["request_ms"], 2) for name, item in scenario["modes"].items()}, flush=True)


if __name__ == "__main__":
    main()
