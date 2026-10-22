"""Build tables and figures from the saved measurements."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NAMES = {"original": "Original", "manual_last": "Last-token head", "sdpa_last": "SDPA",
         "manual_cache": "KV cache", "sdpa_cache": "SDPA + KV cache"}
COLORS = ["#6b7280", "#9bb6c8", "#427a9e", "#dfa06a", "#ad5532"]


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    audit = read("audit.json")
    fp32 = read("benchmark_fp32.json")
    fp16 = read("benchmark_fp16.json")
    checks = read("optimization_checks.json")
    rows = []
    for precision, benchmark in (("fp32", fp32), ("fp16", fp16)):
        for scenario in benchmark["scenarios"]:
            original = scenario["modes"]["original"]["median"]
            for name, mode in scenario["modes"].items():
                median = mode["median"]
                rows.append({"precision": precision, "batch": scenario["batch"],
                             "prefix_tokens": scenario["prefix_tokens"], "new_tokens": scenario["new_tokens"],
                             "mode": name, **median, "speedup_vs_original": original["request_ms"] / median["request_ms"],
                             "same_greedy_ids": mode["greedy_tokens_equal_original"]})
    with (RESULTS / "performance.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    scenarios = fp32["scenarios"]
    labels = [f'B={s["batch"]}, prefix={s["prefix_tokens"]}' for s in scenarios]
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(11.5, 5.5), layout="constrained")
    for i, (name, label) in enumerate(NAMES.items()):
        medians = [s["modes"][name]["median"]["request_ms"] for s in scenarios]
        lows = [s["modes"][name]["request_ms_iqr"][0] for s in scenarios]
        highs = [s["modes"][name]["request_ms_iqr"][1] for s in scenarios]
        ax.bar(x + (i-2)*.16, medians, .15, label=label, color=COLORS[i],
               yerr=[np.array(medians)-lows, np.array(highs)-medians], capsize=2)
    ax.set(xticks=x, xticklabels=labels, ylabel="Request latency (ms)",
           title="FP32 inference ablation: median and interquartile range")
    ax.legend(ncols=3, frameon=False, loc="upper left")
    fig.text(.5, -.015, "RTX 3050 Laptop GPU · 64 generated tokens · 9 interleaved repeats per mode · lower is better",
             ha="center", fontsize=10)
    fig.savefig(RESULTS / "latency_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for i, name in enumerate(("original", "sdpa_cache")):
        peak = [s["modes"][name]["median"]["peak_allocated_bytes"] / 2**20 for s in scenarios]
        rate = [s["modes"][name]["median"]["generated_tokens_per_second"] for s in scenarios]
        color = COLORS[0 if i == 0 else -1]
        axes[0].bar(x + (i-.5)*.34, peak, .32, label=NAMES[name], color=color)
        axes[1].bar(x + (i-.5)*.34, rate, .32, label=NAMES[name], color=color)
    for ax in axes:
        ax.set_xticks(x, [f'{s["batch"]} / {s["prefix_tokens"]}' for s in scenarios])
        ax.set_xlabel("Batch / prefix tokens")
        ax.legend(frameon=False)
    axes[0].set(ylabel="Peak tensor allocation (MiB)", title="Memory includes model weights and KV cache")
    axes[1].set(ylabel="Generated tokens per second", title="Aggregate request throughput")
    fig.savefig(RESULTS / "memory_throughput.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Inference measurements", "",
             f'Device: {fp32["device"]}. PyTorch {fp32["torch"]}; {fp32["cpu_threads"]} CPU threads.', "",
             "Each request generates exactly 64 tokens from fixed synthetic token IDs. EOS stopping is disabled so all modes do equal work. CUDA is synchronized at phase boundaries. Two warmup requests precede nine measured requests per mode; mode order is randomized within each repeat. The allocator remains warm. Model loading, tokenization and profiling are outside timing. All modes use the same fine-tuned checkpoint. These are local batch-request measurements, not concurrent-server latency or quality scores.", "",
             "## FP32 ablations", "",
             "| Batch | Prefix | Mode | Median request (ms) | TTFT (ms) | Decode tokens/s | Peak allocated (MiB) | Speedup | Same greedy IDs |",
             "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        if row["precision"] == "fp32":
            lines.append(f'| {row["batch"]} | {row["prefix_tokens"]} | {row["mode"]} | {row["request_ms"]:.2f} | {row["time_to_first_token_ms"]:.2f} | {row["decode_tokens_per_second"]:.1f} | {row["peak_allocated_bytes"]/2**20:.2f} | {row["speedup_vs_original"]:.2f}x | {row["same_greedy_ids"]} |')
    lines += ["", "![Latency ablation](latency_ablation.png)", "", "![Memory and throughput](memory_throughput.png)", "",
              "Memory uses torch.cuda.max_memory_allocated, including parameters and transient tensors; it excludes reserved allocator blocks, CUDA context and other applications. Cache payload bytes are also recorded separately in the raw files. SDPA dispatch can differ by dtype and GPU; attention_dispatch.json records observed operators. This integrates PyTorch's kernels and does not implement a custom CUDA kernel.", "",
              "Caching is not universally faster: small requests can be dominated by Python dispatch and allocation overhead. Compare sdpa_last with sdpa_cache to isolate the added cache under the same attention backend. Compare original with manual_last to estimate the last-position projection change; minor decoder-wrapper and mask-construction differences also contribute. Compare manual_last with sdpa_last to isolate the attention backend. The combined speedup must not be attributed entirely to KV caching.", "",
              "## Numerical checks", "",
              f'{len(checks["checks"])} trained-checkpoint checks compare full and cached logits against the original forward pass, using both saved checkpoints, two precisions and four batch/length cases. All passed: {checks["all_passed"]}. The maximum permitted absolute error is 0.0003 for FP32 and 0.05 for FP16; actual errors and token-level top-1 agreement are preserved in optimization_checks.json. Finite-precision ties can change decisions, so tolerance checks and generated-ID equality are reported separately. CPU regression tests also cover causal isolation, all-parameter gradients, request isolation and context rollover.', "",
              "## FP16 diagnostic run", "",
              "| Batch | Prefix | Original request (ms) | Optimized request (ms) | Within-FP16 speedup | Same greedy IDs |",
              "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for s in fp16["scenarios"]:
        old = s["modes"]["original"]["median"]["request_ms"]
        new = s["modes"]["sdpa_cache"]["median"]["request_ms"]
        lines.append(f'| {s["batch"]} | {s["prefix_tokens"]} | {old:.2f} | {new:.2f} | {old/new:.2f}x | {s["modes"]["sdpa_cache"]["greedy_tokens_equal_original"]} |')
    lines += ["", "FP32 and FP16 were measured in separate laptop sessions. Their absolute timing difference is not a controlled estimate of the effect of precision. Neither run supports a universal half-precision speedup. Each mode comparison within a run is interleaved, but CPU/GPU clocks, thermal state, background load and scheduling can still vary. Nine repetitions of one checkpoint and one device do not establish performance across deployments.", "",
              "## Reproduce", "", "```bash", "python benchmark_inference.py --checkpoint checkpoints/finetuned_model.pt --output results/benchmark_fp32.json",
              "python benchmark_inference.py --checkpoint checkpoints/finetuned_model.pt --precision float16 --output results/benchmark_fp16.json",
              "python check_optimization.py --checkpoint-dir checkpoints", "python profile_attention.py --checkpoint checkpoints/finetuned_model.pt",
              "python -m pip install -r requirements-analysis.txt", "python make_results.py", "python verify_results.py", "```", "",
              "Sources: [PyTorch SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html). The original parameterization and weights are unchanged."]
    (RESULTS / "PERFORMANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    base = audit["models"]["base"]["validation"]
    tuned = audit["models"]["finetuned"]["validation"]
    audit_lines = ["# Checkpoint and metric audit", "",
          "The available checkpoint evaluation data contains 13 training pairs and 3 validation pairs, with one duplicated training question. These are the only QA validation examples used below. They were also evaluated during the original fine-tuning process, so this is a validation-set reproduction, not an independent test benchmark.", "",
          "| Metric on the same 3 validation pairs | Base | Fine-tuned |", "| --- | ---: | ---: |",
          f'| Token-weighted answer NLL, including EOS | {base["token_weighted_nll"]:.6f} | {tuned["token_weighted_nll"]:.6f} |',
          f'| Mean of example losses, as in the original trainer | {base["example_mean_nll"]:.6f} | {tuned["example_mean_nll"]:.6f} |',
          f'| Teacher-forced top-1 token match, including EOS | {base["teacher_forced_token_match_including_eos"]:.2%} | {tuned["teacher_forced_token_match_including_eos"]:.2%} |',
          f'| Teacher-forced top-1 token match, excluding EOS | {base["teacher_forced_token_match_excluding_eos"]:.2%} | {tuned["teacher_forced_token_match_excluding_eos"]:.2%} |', "",
          f'The token-weighted NLL reduction is {audit["comparison"]["token_nll_relative_reduction"]:.2%}; the example-mean reduction is {audit["comparison"]["example_mean_nll_relative_reduction"]:.2%}. Each run scores {base["tokens_including_eos"]} target positions, including EOS. Teacher forcing supplies the correct preceding answer tokens; it does not measure whether a free-running answer is factually correct. The three questions cannot support a reliable generalization claim.', "",
          "## Architecture and provenance", "",
          "25,309,184 unique parameters, six pre-LayerNorm blocks, eight attention heads, hidden size 512, feed-forward size 2,048, learned positions up to 512 and a 12,000-token ByteLevel BPE tokenizer. The input and output token weights are tied. Both checkpoint files load strictly against the repository architecture. audit.json records checkpoint and tokenizer SHA-256 hashes, checkpoint metadata and every scored example.", "",
          "The base checkpoint records one epoch with training loss 5.939303 and validation loss 7.063682. The fine-tuned checkpoint records ten epochs, training loss 3.470508 and validation loss 5.408775. The latter validation loss is reproduced by the per-example mean above. Saving checkpoints is implemented; activation/gradient checkpointing is not claimed.", "",
          "## Data boundary", "",
          "The local processed corpus has 222,473 tokens including report separators: 143,909 IPCC tokens, 78,482 UNEP tokens and 79 tokens in sample.txt. The old pretraining code uses 868 overlapping windows, with 781 train and 87 validation windows. At the boundary, 256 raw input positions and 256 next-token targets are shared. Within-split neighboring windows overlap too, so window counts are not independent document counts. The tokenizer was trained on the available corpus; no unseen-document evaluation is established.", "",
          "Future pretraining now purges two boundary windows, leaving 781 train and 85 validation windows for this corpus. The fix removes shared raw token positions between the windows, including their next-token targets. It does not create document-level independence, retrain the tokenizer, or repair the historical checkpoint's training history. The original checkpoints were not retrained in this upgrade. A stronger language-quality study needs a larger corpus, document-level partitions fixed before tokenizer training, and an independent question set.", "",
          "## Greedy outputs", "",
          "Greedy decoding uses the training-style BOS and prompt, at most 64 new tokens and EOS stopping. The legacy chat omitted BOS and used sampling, so these are deterministic diagnostic outputs rather than an exact replay of a sampled chat."]
    for before, after in zip(base["cases"], tuned["cases"]):
        audit_lines += ["", f'### {before["question"]}', "", f'Reference: {before["answer"]}', "", f'Base: {before["greedy_answer"]}', "", f'Fine-tuned: {after["greedy_answer"]}']
    audit_lines += ["", "These examples show that the fine-tuned checkpoint still gives incomplete or incorrect answers. The inference upgrade changes execution and measurement, not the model's knowledge.", "", "## Reproduce", "", "```bash", "python audit_model.py --checkpoint-dir checkpoints --corpus-dir data/processed", "```", "", "The raw PDFs, report text and checkpoints remain local. Source files and historical loss history are preserved. No independent question-answer test set is established by these files."]
    (RESULTS / "AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    long = next(row for row in rows if row["precision"] == "fp32" and row["batch"] == 1 and row["prefix_tokens"] == 384 and row["mode"] == "sdpa_cache")
    baseline = next(row for row in rows if row["precision"] == "fp32" and row["batch"] == 1 and row["prefix_tokens"] == 384 and row["mode"] == "original")
    cv = ["# Portfolio positioning", "", "| Project | Main evidence of ability |", "| --- | --- |",
          "| Environmental Language Model | Transformer implementation, attention/cache correctness, profiling, latency/memory tradeoffs and data leakage audit |",
          "| CMU AG News | Controlled model comparisons, prompt development, LoRA training and statistical error analysis |",
          "| AI Buyer Assistant | Catalogue constraints, retrieval/reranking, structured tool decisions and application reliability |", "",
          "## Suggested CV wording", "",
          "- Implemented a 25.3M-parameter decoder-only Transformer in PyTorch with six layers, eight attention heads, tied embeddings and a 512-token context window.",
          f'- Added request-local KV caching, PyTorch SDPA and last-token vocabulary projection, reducing median FP32 latency from {baseline["request_ms"]:.0f} to {long["request_ms"]:.0f} ms ({long["speedup_vs_original"]:.2f}x) for a 384-token prefix and 64 generated tokens on an RTX 3050 Laptop GPU.',
          "- Validated cached and uncached execution with causal-mask, gradient, request-isolation and context-rollover tests; built interleaved latency, throughput and GPU-memory benchmarks with nine repeats per configuration.", "",
          "The speed figure describes one specified workload. Use the supporting table when discussing other batch sizes or shorter prompts. Teacher-forced token matching does not measure QA accuracy. SDPA integrates PyTorch kernels; faster execution does not improve factual knowledge.", "",
          "## Further development", "",
          "The next systems step would be preallocated KV buffers and a small continuous-batching scheduler, measured against the current dynamic cache. Long-context architectures such as RoPE would require a new training study; extending the learned positional table alone is not evidence that the model can use longer contexts. Better answers would require substantially better data and independent evaluation. These remain future work rather than completed features."]
    (ROOT / "PORTFOLIO.md").write_text("\n".join(cv) + "\n", encoding="utf-8")
    print("Created audit, performance tables, two figures and portfolio notes.")


if __name__ == "__main__":
    main()
